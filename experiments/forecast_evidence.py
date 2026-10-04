"""Post-training fit, consumption and fixed-critic action-outcome diagnostics.

No updates, new scenarios, rewards or deployment truth inputs. Archives retain
every root/result and separate outcome-blind primary from action-change probes.
"""

import argparse
import configparser
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState
from experiments.forecast_control_diagnostic import (digest, examples, filter_scores,
                                                     state_from_tokens)
from experiments.latent_information import snapshot
from experiments.occlusion import source_hash
from shixu.features import stack_histories, window
from shixu.model import build_model, load_weights
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment, orca_teacher, run_episode


ROOT = Path("outputs/forecast_control_b")
OUT = ROOT / "evidence_diagnostic"
SEEDS = (419, 443, 467, 491)
KINDS = ("cv", "current", "gru", "kda")
RULES = {
    "sample": "Every fourth frame, identical lawful history/label masks across arms.",
    "train": "All128 archived successful IL episodes; not the missing online RL replay.",
    "test": "Archived Parent419 commands:32 independent5-circle cases80000-80031; "
            "other five geometry/population cells use first four cases.",
    "primary_roots": "One per original24 episodes: first sampled root at t>=2s with "
                     "current lawful surface clearance<=0.8m; otherwise closest sampled root at t>=2s.",
    "secondary_roots": "First sampled t>=2s root per original24 episodes where true-one-step and "
                       "CV choose different actions in any frozen consumer; action-selected, not outcome-selected.",
    "danger": "Native CV clearance of the CV-selected action is below the original0.2m discomfort margin.",
    "additional_danger": "Current lawful surface clearance<=0.8m, fixed before new results.",
    "consumers": "KDA-trained and fresh-CV-trained critic for each of four paired seeds; never updated.",
    "continuation": "One smoothed native root command, then common legal ORCA; "
                    "unbraked ORCA is a separately reported sensitivity control. Original reward/gamma.",
}


def weights(seed, kind, phase="model"):
    base = ROOT.parent / "forecast_control_b_fresh_cv" if kind == "cv" else ROOT
    return base / str(seed) / kind / (phase + ".pt")


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    model = build_model(cfg).to(device).eval()
    load_weights(model, path, device)
    return model, cfg


def write(path, result):
    if path.exists():
        raise RuntimeError("Refusing to overwrite diagnostic: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))


def replay_record(record, cfg, encoder, roots):
    env = environment(cfg, encoder, record["geometry"], record["people"])
    env.reset(options={"test_case": record["case"]})
    if env.robot.visible:
        raise ValueError("Common human futures require robot-invisible native dynamics")
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    frames, snapshots, truths, rewards, path = [], [], [], [], 0.
    for tick, command in enumerate(record["actions"]):
        state = observer.observe(env)
        token = encoder.encode(state)
        frames.append(token)
        if tick in roots:
            snapshots.append((tick, snapshot(env, observer)))
        old = np.asarray(env.robot.get_position())
        _, reward, done, truncated, info = env.step(ActionXY(*command))
        rewards.append(reward)
        path += float(np.linalg.norm(np.asarray(env.robot.get_position()) - old))
        truth = np.zeros((state.track_count, 2), np.float32)
        for human, (key, _, _) in observer.tracks.items():
            truth[key] = human.get_position()
        truths.append(truth)
        if (done or truncated) != (tick == len(record["actions"]) - 1):
            raise ValueError("Replay termination differs")
    if (info["event"] != record["terminal"] or abs(path - record["path"]) > 1e-6
            or abs(env.global_time - record["navigation_time"]) > 1e-8):
        raise ValueError("Replay archive parity failed")
    return frames, dict(snapshots), truths, rewards


def choose_primary(rows):
    grouped = {}
    for index, row in enumerate(rows):
        if row["tick"] >= 8:
            grouped.setdefault(row["episode"], []).append((index, row))
    result = []
    for episode in sorted(grouped):
        eligible = grouped[episode]
        close = [item for item in eligible if item[1]["current_clearance"] <= .8]
        result.append((close[0] if close else min(eligible, key=lambda item: item[1]["current_clearance"]))[0])
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "protocol.json", {"rules": RULES, "source_sha256": source_hash(), "seeds": SEEDS})
    data = torch.load(ROOT / "shared_diagnostic.pt", map_location="cpu", weights_only=False)
    demonstrations = torch.load(ROOT / "demonstrations.pt", map_location="cpu", weights_only=False)
    _, cfg = load(weights(419, "kda"), "cpu")
    encoder = ValuePolicy(build_model(cfg), cfg)
    archived = json.loads((ROOT.parent / "forecast_control_a/419/parent_gru/result.json").read_text())
    lookup = {(r["geometry"], r["people"], r["case"]): r for r in archived["episodes"]}
    shared_rows = examples(data)
    for row in shared_rows:
        frame = row["history"][-1]
        active = frame[1:, 12] > 0
        clear = np.linalg.norm(frame[1:, :2], axis=-1) - frame[1:, 6] - frame[0, 4]
        row["current_clearance"] = float(clear[active].min()) if active.any() else 100.
    primary = choose_primary(shared_rows)
    records = []
    for episode_index, episode in enumerate(data["episodes"]):
        record = lookup[episode["geometry"], episode["people"], episode["case"]]
        root_ticks = {row["tick"] for row in shared_rows if row["episode"] == episode_index}
        frames, snapshots, truths, actual_rewards = replay_record(record, cfg, encoder, root_ticks)
        for actual, expected in zip(frames, episode["tokens"]):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        records.append({**episode, "actions": record["actions"], "snapshots": snapshots,
                        "truths": truths, "rewards": actual_rewards})
    for case in range(80004, 80032):
        record = lookup["circle", 5, case]
        frames, _, _, _ = replay_record(record, cfg, encoder, set())
        records.append({**record, "tokens": frames,
                        "roots": [{"tick": t} for t in range(0, len(frames), 4)]})
    test = dict(data, episodes=records)
    # Non-shared test roots need only physical labels, not candidate scoring.
    for ep in test["episodes"][24:]:
        for root in ep["roots"]:
            root.update(queries=np.zeros((0, 1, 13), np.float32), rewards=np.zeros(0), clearance=np.zeros(0))
    train = dict(data, episodes=[dict(ep, geometry="circle", people=5,
                                   roots=[dict(tick=t, queries=np.zeros((0, 1, 13), np.float32),
                                               rewards=np.zeros(0), clearance=np.zeros(0))
                                          for t in range(0, len(ep["tokens"]), 4)])
                                for ep in demonstrations["episodes"]])
    for row in shared_rows:
        ep = records[row["episode"]]
        row["truth"] = ep["truths"][row["tick"]]
    torch.save(dict(train=train, test=test, shared_rows=shared_rows, primary=primary,
                    source_sha256=source_hash(), demonstration_sha256=digest(ROOT / "demonstrations.pt"),
                    cohort_sha256=digest(ROOT / "shared_diagnostic.pt")), OUT / "cohort.pt")
    print("PREPARED", len(examples(train)), "IL roots", len(examples(test)), "test roots",
          len(primary), "primary decision roots", flush=True)


def stats(errors, squared, valid):
    count = int(valid.sum())
    result = {"targets": count, "ade_sum": float(errors[valid].sum()),
              "velocity_squared_sum": float(squared[valid].sum())}
    for step in (1, 4, 8):
        mask = valid[:, step - 1]
        result[str(step * .25)] = {"targets": int(mask.sum()),
                                   "ade_sum": float(errors[:, step - 1][mask].sum()),
                                   "velocity_squared_sum": float(squared[:, step - 1][mask].sum())}
    return result


@torch.inference_mode()
def accuracy(device, teacher_only=False):
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    if teacher_only:
        data = {"test": torch.load(OUT / "teacher_test.pt", map_location="cpu", weights_only=False)}
    for seed in SEEDS:
        for kind in KINDS:
            for phase in ("il", "model"):
                prefix = "teacher_accuracy" if teacher_only else "accuracy"
                destination = OUT / f"{prefix}_{seed}_{kind}_{phase}.json"
                if destination.exists():
                    continue
                model, _ = load(weights(seed, kind, phase), device)
                records = []
                for split in data.keys() if teacher_only else ("train", "test"):
                    episodes = data[split]["episodes"]
                    rows = examples(data[split])
                    for start in range(0, len(rows), 32):
                        batch = rows[start:start + 32]
                        history = torch.as_tensor(stack_histories([row["history"] for row in batch]), device=device)
                        prediction = model.forecast_positions(history)[:, :, 1:].cpu().numpy()
                        humans, world = model._physical(history)
                        cv = (world[:, -1, :, None] + humans[:, -1, :, None, 3:5]
                              * model.times[None, None, 1:, None]).cpu().numpy()
                        for index, row in enumerate(batch):
                            count = len(row["future"])
                            delta = prediction[index, :count] - row["future"]
                            cv_delta = cv[index, :count] - row["future"]
                            errors = np.linalg.norm(delta, axis=-1)
                            squared = (delta / (np.arange(1, 10)[None, :, None] * .25)) ** 2
                            ep = episodes[row["episode"]]
                            item = dict(split=split, episode=row["episode"], tick=row["tick"],
                                        geometry=ep["geometry"], people=ep["people"], case=ep["case"],
                                        error=stats(errors, squared.sum(-1), row["valid"]),
                                        cv_error=stats(np.linalg.norm(cv_delta, axis=-1),
                                                       ((cv_delta / (np.arange(1, 10)[None, :, None] * .25)) ** 2).sum(-1),
                                                       row["valid"]))
                            records.append(item)
                write(destination, dict(seed=seed, kind=kind, phase=phase, roots=records,
                                        checkpoint_sha256=digest(weights(seed, kind, phase))))
                print("ACCURACY", seed, kind, phase, len(records), flush=True)


def teacher_control():
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    model, cfg = load(weights(419, "kda"), "cpu")
    policy, teacher = ValuePolicy(model, cfg), orca_teacher(cfg)
    episodes = []
    for case in range(80000, 80032):
        env = environment(cfg, policy, "circle", 5)
        episode = run_episode(env, policy, case, teacher=teacher)
        episode.update(geometry="circle", people=5,
                       roots=[dict(tick=t, queries=np.zeros((0, 1, 13), np.float32),
                                   rewards=np.zeros(0), clearance=np.zeros(0))
                              for t in range(0, len(episode["tokens"]), 4)])
        episodes.append(episode)
    path = OUT / "teacher_test.pt"
    if path.exists():
        raise RuntimeError("Do not overwrite teacher control")
    torch.save(dict(data["train"], episodes=episodes,
                    scope="Independent32 native5-circle cases, same ORCA teacher as IL. "
                          "All terminals retained; unlike IL no success-only case selection."), path)
    print("TEACHER_CONTROL", len(episodes), [ep["terminal"] for ep in episodes], flush=True)


@torch.inference_mode()
def history_usage(device):
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    teacher = torch.load(OUT / "teacher_test.pt", map_location="cpu", weights_only=False)
    cohorts = {"ORCA_independent": examples(teacher),
               "Parent_5circle_independent": [r for r in examples(data["test"])
                           if data["test"]["episodes"][r["episode"]]["geometry"] == "circle"
                           and data["test"]["episodes"][r["episode"]]["people"] == 5]}
    records = []
    for seed in SEEDS:
        for kind in ("current", "gru", "kda"):
            model, _ = load(weights(seed, kind), device)
            for cohort, rows in cohorts.items():
                for keep in (24, 3, 1):
                    metrics = []
                    for start in range(0, len(rows), 32):
                        batch = rows[start:start + 32]
                        history = torch.as_tensor(stack_histories([r["history"] for r in batch]), device=device)
                        history[:, :-keep] = 0
                        predicted = model.forecast_positions(history)[:, :, 1:].cpu().numpy()
                        for i, row in enumerate(batch):
                            delta = predicted[i, :len(row["future"])] - row["future"]
                            squared = ((delta / (np.arange(1, 10)[None, :, None] * .25)) ** 2).sum(-1)
                            metrics.append(dict(error=stats(np.linalg.norm(delta, axis=-1), squared, row["valid"])))
                    records.append(dict(seed=seed, kind=kind, cohort=cohort, retained_frames=keep,
                                        **error_summary(metrics)))
            print("HISTORY_USAGE", seed, kind, flush=True)
    write(OUT / "history_usage.json", dict(records=records,
          scope="Frozen-weight removal of older lawful observations; current frame/neighbours/labels unchanged. "
                "Sensitivity/use diagnostic, not a trained Current/last-k baseline or capacity-controlled comparison."))


def execute(action, previous, cfg):
    alpha = cfg.getfloat("eval_protocol", "action_smoothing")
    return np.asarray(action) if previous is None else alpha * np.asarray(previous) + (1 - alpha) * np.asarray(action)


@torch.inference_mode()
def decisions(device):
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    rows, records = data["shared_rows"], []
    for seed in SEEDS:
        predictors = {kind: load(weights(seed, kind), device)[0] for kind in KINDS}
        for consumer_kind in ("cv", "kda"):
            consumer, cfg = load(weights(seed, consumer_kind), device)
            native_actions = np.asarray(ValuePolicy(consumer, cfg, device).action_space)
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                history = torch.as_tensor(stack_histories([r["history"] for r in batch]), device=device)
                people = history.shape[2] - 1
                queries = torch.as_tensor(np.stack([np.pad(row["queries"],
                                            ((0, 0), (0, people + 1 - row["queries"].shape[1]), (0, 0)))
                                             for row in batch]), device=device)
                forecasts = {kind: model.forecast_positions(history) for kind, model in predictors.items()}
                truth = forecasts["cv"].clone()
                for i, row in enumerate(batch):
                    truth[i, :len(row["truth"]), 1] = torch.as_tensor(row["truth"], device=device)
                forecasts["truth"] = truth
                rewards = torch.as_tensor(np.stack([r["rewards"] for r in batch]), device=device)
                clearance = torch.as_tensor(np.stack([r["clearance"] for r in batch]), device=device)
                alternatives = {}
                for kind, predicted in forecasts.items():
                    corrected = consumer.corrected_queries(history, queries, predicted)
                    values = consumer.critic.score_candidates(history[:, 1:], corrected)
                    raw = rewards + cfg.getfloat("train", "gamma") * values
                    alternatives[kind] = (raw.cpu().numpy(), filter_scores(raw, clearance, cfg).cpu().numpy())
                for i, row in enumerate(batch):
                    root_index = start + i
                    ep = data["test"]["episodes"][row["episode"]]
                    previous = ep["actions"][row["tick"] - 1] if row["tick"] else None
                    item = dict(seed=seed, consumer=consumer_kind, root=root_index,
                                episode=row["episode"], tick=row["tick"], current_clearance=row["current_clearance"],
                                variants={})
                    for kind, (raw, filtered) in alternatives.items():
                        raw_index, index = int(raw[i].argmax()), int(filtered[i].argmax())
                        command = execute(native_actions[index], previous, cfg)
                        count = len(row["truth"])
                        active = row["history"][-1, 1:, 12] > 0
                        delta = forecasts[kind][i, :count, 1].cpu().numpy() - row["truth"]
                        error = np.linalg.norm(delta, axis=-1)
                        item["variants"][kind] = dict(raw_action=raw_index, action=index, command=command.tolist(),
                                                     selected_native_clearance=float(row["clearance"][index]),
                                                     truth_error_mean=float(error[active].mean()) if active.any() else 0.,
                                                     actor_truth_errors=error.tolist(),
                                                     scores=filtered[i].tolist(), raw_scores=raw[i].tolist())
                    records.append(item)
            print("DECISIONS", seed, consumer_kind, len(rows), flush=True)
    write(OUT / "decisions.json", dict(records=records, rules=RULES,
          checkpoints={f"{seed}/{kind}": digest(weights(seed, kind)) for seed in SEEDS for kind in KINDS}))
    secondary = []
    for episode in range(24):
        eligible = [r for r in records if r["episode"] == episode and r["tick"] >= 8
                    and r["variants"]["cv"]["action"] != r["variants"]["truth"]["action"]]
        if eligible:
            secondary.append(min(eligible, key=lambda r: r["tick"])["root"])
    write(OUT / "selection.json", dict(primary=data["primary"], secondary=sorted(set(secondary)),
                                      rules=RULES, selection_before_outcome=True))


def restored(cfg, episode, saved):
    env = environment(cfg, orca_teacher(cfg), episode["geometry"], episode["people"])
    env.reset(options={"test_case": episode["case"]})
    for agent, row in zip([env.robot] + env.humans, [saved["robot"]] + saved["humans"]):
        agent.set(*[row[i] for i in (0, 1, 5, 6, 2, 3, 8)], radius=row[4], v_pref=row[7])
    for human, previous in zip(env.humans, saved["preferred"]):
        human.policy.sim = None
        human.policy._last_pref_vel = None if previous is None else np.asarray(previous).copy()
    env.global_time = saved["time"]
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    observer.tracks = {env.humans[i]: (key, ObservableState(*measurement), stamp)
                       for i, key, measurement, stamp in saved["tracks"]}
    return env, observer


def consequence(task):
    root, episode, tick, command, continuation, config = task
    cfg = configparser.ConfigParser()
    cfg.read_dict(config)
    env, observer = restored(cfg, episode, episode["snapshots"][tick])
    teacher = orca_teacher(cfg)
    teacher.ttc_brake = continuation == "orca"
    for frame in episode["tokens"][:tick]:
        teacher.predict(state_from_tokens(frame))
    discount, total, minimum, steps = 1., 0., float("inf"), 0
    action = ActionXY(*command)
    initial = np.linalg.norm(np.asarray(env.robot.get_position()) - env.robot.get_goal_position())
    while True:
        _, reward, done, truncated, info = env.step(action)
        total += discount * reward
        discount *= cfg.getfloat("train", "gamma")
        steps += 1
        minimum = min(minimum, info["dmin"])
        if done or truncated:
            break
        action = teacher.predict(observer.observe(env))
    distance = np.linalg.norm(np.asarray(env.robot.get_position()) - env.robot.get_goal_position())
    return dict(root=root, command=command, continuation=continuation, terminal=info["event"],
                discounted_return=float(total), minimum_clearance=float(minimum), seconds=steps * env.time_step,
                progress=float(initial - distance))


def outcomes(workers):
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    selection = json.loads((OUT / "selection.json").read_text())
    decisions_data = json.loads((OUT / "decisions.json").read_text())
    selected = sorted(set(selection["primary"] + selection["secondary"]))
    _, cfg = load(weights(419, "kda"), "cpu")
    config = {s: dict(cfg[s]) for s in cfg.sections()}
    checks = []
    for root in selected:
        row = data["shared_rows"][root]
        ep = data["test"]["episodes"][row["episode"]]
        tick = row["tick"]
        env, observer = restored(cfg, ep, ep["snapshots"][tick])
        _, reward, _, _, _ = env.step(ActionXY(*ep["actions"][tick]))
        mapping = {key: human for human, (key, _, _) in observer.tracks.items()}
        actual = np.asarray([mapping[k].get_position() for k in range(len(row["truth"]))])
        error = float(np.max(np.abs(actual - row["truth"])))
        if error > 2e-6 or abs(reward - ep["rewards"][tick]) > 1e-8:
            raise ValueError("Root restoration failed: " + str(root))
        checks.append(dict(root=root, maximum_human_position_error=error,
                           reward_error=float(abs(reward - ep["rewards"][tick]))))
    write(OUT / "restore_checks.json", dict(checks=checks))
    tasks, seen = [], set()
    for record in decisions_data["records"]:
        root = record["root"]
        if root not in selected:
            continue
        row = data["shared_rows"][root]
        ep = data["test"]["episodes"][row["episode"]]
        for kind in ("cv", "current", "kda", "truth"):
            command = record["variants"][kind]["command"]
            for continuation in ("orca", "unbraked_orca"):
                key = (root, tuple(command), continuation)
                if key not in seen:
                    tasks.append((root, ep, row["tick"], command, continuation, config))
                    seen.add(key)
    results = []
    with ProcessPoolExecutor(workers) as pool:
        for i, result in enumerate(pool.map(consequence, tasks, chunksize=1)):
            results.append(result)
            if (i + 1) % 40 == 0:
                print("OUTCOME", i + 1, "/", len(tasks), flush=True)
    write(OUT / "outcomes.json", dict(records=results, selection=selection, checks=checks,
          scope="Same frozen critic and one native smoothed root command; common lawful continuation. "
                "Q^pi for selected actions, not Q*, not complete deployed-policy success rates."))
    print("OUTCOME_DONE", len(results), "unique branches", flush=True)


def error_summary(rows):
    result = {}
    for horizon in ("all", "0.25", "1.0", "2.0"):
        stats_rows = [row["error"] if horizon == "all" else row["error"][horizon] for row in rows]
        count = sum(row["targets"] for row in stats_rows)
        result[horizon] = dict(targets=count,
                               ade_m=sum(row["ade_sum"] for row in stats_rows) / max(count, 1),
                               velocity_mse=sum(row["velocity_squared_sum"] for row in stats_rows) / max(2 * count, 1))
    return result


def summarize():
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    teacher = torch.load(OUT / "teacher_test.pt", map_location="cpu", weights_only=False)
    fit = []
    for seed in SEEDS:
        for kind in KINDS:
            for phase in ("il", "model"):
                rows = json.loads((OUT / f"accuracy_{seed}_{kind}_{phase}.json").read_text())["roots"]
                teacher_rows = json.loads((OUT / f"teacher_accuracy_{seed}_{kind}_{phase}.json").read_text())["roots"]
                groups = {
                    "IL_seen": [r for r in rows if r["split"] == "train"],
                    "ORCA_independent": teacher_rows,
                    "ORCA_independent_success_only": [r for r in teacher_rows
                                if teacher["episodes"][r["episode"]]["terminal"] == "reach_goal"],
                    "Parent_5circle_independent": [r for r in rows if r["split"] == "test"
                                                   and r["geometry"] == "circle" and r["people"] == 5],
                    "Parent_other_cells": [r for r in rows if r["split"] == "test"
                                           and not (r["geometry"] == "circle" and r["people"] == 5)],
                }
                for group, subset in groups.items():
                    fit.append(dict(seed=seed, kind=kind, phase=phase, cohort=group,
                                    episodes=len({r["episode"] for r in subset}), roots=len(subset),
                                    **error_summary(subset)))
    decisions_data = json.loads((OUT / "decisions.json").read_text())["records"]
    consumption = []
    for seed in SEEDS:
        for consumer in ("cv", "kda"):
            subset = [r for r in decisions_data if r["seed"] == seed and r["consumer"] == consumer]
            for contrast in ("kda_cv", "kda_current", "truth_cv"):
                target, reference = contrast.split("_")
                for changed in (False, True):
                    rows = [r for r in subset if (r["variants"][target]["action"]
                                                 != r["variants"][reference]["action"]) == changed]
                    count = len(rows)
                    errors, near_errors, hidden_errors = [], [], []
                    for r in rows:
                        root = data["shared_rows"][r["root"]]
                        actors = root["history"][-1, 1:]
                        active = actors[:, 12] > 0
                        near, hidden = active & (actors[:, 2] < 2), active & (actors[:, 10] == 0)
                        error = np.asarray(r["variants"][target]["actor_truth_errors"])
                        errors.extend(error[active].tolist())
                        near_errors.extend(error[near].tolist())
                        hidden_errors.extend(error[hidden].tolist())
                    consumption.append(dict(seed=seed, consumer=consumer, contrast=contrast, changed=changed,
                                            states=count,
                                            close_states=sum(r["current_clearance"] <= .8 for r in rows),
                                            cv_selected_unsafe=sum(r["variants"]["cv"]["selected_native_clearance"] < .2 for r in rows),
                                            prediction_error_m=float(np.mean(errors)) if errors else None,
                                            near_prediction_error_m=float(np.mean(near_errors)) if near_errors else None,
                                            hidden_prediction_error_m=float(np.mean(hidden_errors)) if hidden_errors else None,
                                            hidden_targets=len(hidden_errors)))
    selection = json.loads((OUT / "selection.json").read_text())
    outcomes_data = json.loads((OUT / "outcomes.json").read_text())["records"]
    lookup = {(r["root"], tuple(r["command"]), r["continuation"]): r for r in outcomes_data}
    realized = []
    for group in ("primary", "secondary"):
        selected = set(selection[group])
        for seed in SEEDS:
            for consumer in ("cv", "kda"):
                rows = [r for r in decisions_data if r["seed"] == seed and r["consumer"] == consumer and r["root"] in selected]
                for continuation in ("orca", "unbraked_orca"):
                    for kind in ("cv", "current", "kda", "truth"):
                        actual = [lookup[r["root"], tuple(r["variants"][kind]["command"]), continuation] for r in rows]
                        base = [lookup[r["root"], tuple(r["variants"]["cv"]["command"]), continuation] for r in rows]
                        returns = np.asarray([r["discounted_return"] for r in actual])
                        baseline = np.asarray([r["discounted_return"] for r in base])
                        realized.append(dict(group=group, seed=seed, consumer=consumer, continuation=continuation,
                                             variant=kind, roots=len(rows),
                                             action_changes=sum(r["variants"][kind]["action"] != r["variants"]["cv"]["action"] for r in rows),
                                             successes=sum(r["terminal"] == "reach_goal" for r in actual),
                                             collisions=sum(r["terminal"] == "collision" for r in actual),
                                             timeouts=sum(r["terminal"] == "timeout" for r in actual),
                                             mean_return=float(returns.mean()), delta_return=float((returns - baseline).mean()),
                                             positive_return_roots=int(((returns - baseline) > 1e-8).sum()),
                                             negative_return_roots=int(((returns - baseline) < -1e-8).sum()),
                                             mean_seconds=float(np.mean([r["seconds"] for r in actual])),
                                             mean_progress=float(np.mean([r["progress"] for r in actual])),
                                             mean_clearance=float(np.mean([r["minimum_clearance"] for r in actual]))))
    write(OUT / "summary.json", dict(fit=fit, consumption=consumption, realized=realized, rules=RULES,
                                     source_sha256=source_hash(),
                                     scope="Seed/checkpoint conditions reuse states: not independent episode samples. "
                                           "ORCA/Parent visitation and IL success-selection remain explicit. "
                                           "True future changes position only, not reward/filter/velocity/masks; "
                                           "untrained substituted queries may be distribution shifted."))
    for phase in ("il", "model"):
        for cohort in ("IL_seen", "ORCA_independent", "Parent_5circle_independent", "Parent_other_cells"):
            for kind in KINDS:
                rows = [r for r in fit if r["phase"] == phase and r["cohort"] == cohort and r["kind"] == kind]
                print("FIT", phase, cohort, kind, "episodes", rows[0]["episodes"], "ADE.25/2.0/loss",
                      np.mean([[r["0.25"]["ade_m"], r["2.0"]["ade_m"], r["all"]["velocity_mse"]] for r in rows], 0).tolist())
    for group in ("primary", "secondary"):
        for consumer in ("cv", "kda"):
            for continuation in ("orca", "unbraked_orca"):
                for kind in ("cv", "current", "kda", "truth"):
                    rows = [r for r in realized if r["group"] == group and r["consumer"] == consumer
                            and r["continuation"] == continuation and r["variant"] == kind]
                    print("REALIZED", group, consumer, continuation, kind, "S/C/T,return,delta",
                          np.mean([[r["successes"], r["collisions"], r["timeouts"], r["mean_return"], r["delta_return"]] for r in rows], 0).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "accuracy", "teacher-control", "teacher-accuracy", "history-usage", "decisions", "outcomes", "summarize"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode != "prepare":
        frozen = json.loads((OUT / "protocol.json").read_text())["source_sha256"]
        if source_hash() != frozen:
            raise RuntimeError("Scientific source differs from the diagnostic freeze")
    if args.mode == "prepare":
        prepare()
    elif args.mode == "accuracy":
        accuracy(args.device)
    elif args.mode == "teacher-control":
        teacher_control()
    elif args.mode == "teacher-accuracy":
        accuracy(args.device, True)
    elif args.mode == "history-usage":
        history_usage(args.device)
    elif args.mode == "decisions":
        decisions(args.device)
    elif args.mode == "summarize":
        summarize()
    else:
        outcomes(args.workers)


if __name__ == "__main__":
    main()
