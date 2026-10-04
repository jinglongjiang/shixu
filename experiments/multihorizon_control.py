"""Frozen two-second forecast / common CV-tail action repair experiment.

Uses the native reward/dynamics and unchanged CV value policy for continuation.
Hypothetical worlds contain only currently active known tracks. Real outcomes
restore all native humans. No training, privileged goals or new reward terms.
"""

import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.forecast_control_diagnostic import digest, filter_scores
from experiments.forecast_evidence import load, replay_record, restored, weights
from experiments.occlusion import source_hash
from shixu.features import stack_histories, window
from shixu.observations import OccludedTracks, ObservedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/multihorizon_control")
SEEDS = (419, 443, 467, 491)
RULES = {
    "parent": "Fresh-CV419 final trained navigation policy, unchanged for every predictor/seed.",
    "failures": "First two cases per geometry/population/terminal from its existing192-case archive; "
                "terminal collision or timeout; case-sorted, no method-result selection.",
    "failure_anchors": "10s and4s before archived termination, clamp to tick0; deduplicate coincident anchors.",
    "success_guard": "First archived successful case per geometry/population;4s before termination.",
    "history": "Same lawful24-frame window, active known actor set and root tracker for all forecasts.",
    "forecast": "One frozen forecast at root,0.25..2s. Truth exposes only those actors' positions, not goals/new actors.",
    "shadow": "Native kernel, initial legal actor positions/velocities/radii; replay forecast positions for8 steps. "
              "Then CV using (p2s-p1.75s)/0.25, until the original task deadline. "
              "Future observations use original body occlusion/retention on this approximate known-actor world.",
    "root": "All80 native root actions, original smoothing once; root lasts0.25s only.",
    "feedback": "Same frozen native CV policy every0.25s thereafter, unchanged masks/risk/smoothing/history.",
    "score": "Full native discounted task return, gamma0.99; no new progress/reward or learned value tail. "
             "Original root CV safety/risk selector retained. Ties within1e-8 use original Parent root ranking.",
    "realized": "Selected first command once, then identical frozen policy in restored real native world to termination.",
    "scope": "Repair of selected failures and separate limited success guard, not overall SR or full retrained method.",
    "numeric": "Shadow scoring batches identical critic operations; native scalar fallback when top-two "
               "gap<=1e-4. Validate root score/action parity at1e-4. Actual outcomes use strict native "
               "single-world predict(), not batched controls. Replay each selected shadow using native "
               "predict(); if its trace differs, rescore all80 natively. TF32 defaults/checkpoints are unchanged.",
    "stop": "Finish frozen five-family comparison. No retraining/retuning; truth negative closes this scorer only.",
}


def save_json(path, value):
    if path.exists():
        raise RuntimeError("Refusing to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def legal_root(saved, token):
    active = np.flatnonzero(token[1:, 12] > 0)
    tracks = {key: (measurement, stamp) for _, key, measurement, stamp in saved["tracks"]}
    humans, ages = [], []
    for key in active:
        measurement, stamp = tracks[key]
        age = saved["time"] - stamp
        humans.append(ObservableState(measurement[0] + age * measurement[2],
                                      measurement[1] + age * measurement[3], *measurement[2:]))
        ages.append(age)
    return ObservedTracks(FullState(*saved["robot"]), humans, tuple(active),
                          tuple(bool(token[1 + key, 10]) for key in active), tuple(ages), len(token) - 1)


def select_episodes(episodes):
    selected = []
    for people in (5, 10, 20):
        for geometry in ("circle", "square"):
            rows = sorted((r for r in episodes if r["people"] == people
                           and r["geometry"] == geometry), key=lambda r: r["case"])
            for terminal in ("collision", "timeout"):
                selected.extend(("failure", row) for row in [r for r in rows if r["terminal"] == terminal][:2])
            selected.append(("success_guard", next(r for r in rows if r["terminal"] == "reach_goal")))
    return selected


def precise_snapshots(record, cfg, encoder, ticks):
    env = environment(cfg, encoder, record["geometry"], record["people"])
    env.reset(options={"test_case": record["case"]})
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    snapshots = {}
    for tick, command in enumerate(record["actions"]):
        observer.observe(env)
        if tick in ticks:
            agents = [env.robot] + env.humans
            fields = ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta")
            states = [np.asarray([getattr(a, key) for key in fields], np.float64) for a in agents]
            indices = {human: i for i, human in enumerate(env.humans)}
            snapshots[tick] = dict(robot=states[0], humans=states[1:], time=env.global_time,
                preferred=[None if h.policy._last_pref_vel is None else h.policy._last_pref_vel.copy()
                           for h in env.humans],
                tracks=[(indices[h], key, np.asarray([m.px, m.py, m.vx, m.vy, m.radius], np.float64), stamp)
                        for h, (key, m, stamp) in observer.tracks.items()])
        env.step(ActionXY(*command))
    return snapshots


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    archive_path = Path("outputs/forecast_control_b_fresh_cv/419/cv/result.json")
    archive = json.loads(archive_path.read_text())
    parent_path = weights(419, "cv")
    model, cfg = load(parent_path, "cpu")
    encoder = ValuePolicy(model, cfg)
    selected = select_episodes(archive["episodes"])
    selection = []
    for group, record in selected:
        anchors = (10., 4.) if group == "failure" else (4.,)
        ticks = sorted({max(0, len(record["actions"]) - round(s / .25)) for s in anchors})
        selection.extend(dict(group=group, people=record["people"], geometry=record["geometry"],
                              case=record["case"], terminal=record["terminal"], tick=tick) for tick in ticks)
    save_json(OUT / "protocol.json", dict(rules=RULES, selection=selection,
              archive_sha256=digest(archive_path), source_sha256=source_hash(), seeds=SEEDS))
    inputs = OUT / "inputs"
    inputs.mkdir()
    shutil.copy2(parent_path, inputs / "parent.pt")
    hashes = {"parent.pt": digest(parent_path)}
    for seed in SEEDS:
        for kind in ("current", "gru", "kda"):
            path = weights(seed, kind)
            name = f"{seed}_{kind}.pt"
            shutil.copy2(path, inputs / name)
            hashes[name] = digest(path)
    roots = []
    for group, record in selected:
        ticks = {r["tick"] for r in selection if r["group"] == group and r["people"] == record["people"]
                 and r["geometry"] == record["geometry"] and r["case"] == record["case"]}
        frames, snapshots, truths, rewards = replay_record(record, cfg, encoder, ticks)
        snapshots = precise_snapshots(record, cfg, encoder, ticks)
        for tick in sorted(ticks):
            if tick + 8 > len(truths):
                raise ValueError("Prespecified root lacks two seconds of archived human truth")
            state = legal_root(snapshots[tick], frames[tick])
            count = state.track_count
            future = np.stack([truths[tick + h][:count] for h in range(8)], axis=1)
            roots.append(dict(group=group, episode={k: record[k] for k in ("people", "geometry", "case", "terminal")},
                              tick=tick, state=state, snapshot=snapshots[tick],
                              history=window(frames[:tick + 1], 24, "zero"),
                              previous=record["actions"][tick - 1] if tick else None, truth=future,
                              archived_action=record["actions"][tick], archived_rewards=rewards[tick:],
                              archived_tail_actions=record["actions"][tick:],
                              archived_remaining_seconds=(len(record["actions"]) - tick) * .25))
    torch.save(dict(roots=roots, source_sha256=source_hash(), config={s: dict(cfg[s]) for s in cfg.sections()}),
               inputs / "cohort.pt")
    hashes["cohort.pt"] = digest(inputs / "cohort.pt")
    save_json(inputs / "hashes.json", hashes)
    print("PREPARED", len(roots), "roots", {g: sum(r["group"] == g for r in roots)
                                              for g in ("failure", "success_guard")}, flush=True)


class Trajectory:
    def __init__(self, points):
        self.points, self.step = np.asarray(points), 0
        self.tail_velocity = (self.points[8] - self.points[7]) / .25

    def predict(self, state):
        if self.step < 8:
            velocity = (self.points[self.step + 1] - state.self_state.position) / .25
        else:
            velocity = self.tail_velocity
        self.step += 1
        return ActionXY(*velocity)


def fork(root, model, cfg, device, points=None):
    env, observer = restored(cfg, root["episode"], root["snapshot"])
    policy = ValuePolicy(model, cfg, device)
    policy.history.extend(root["history"][:-1])
    policy.last_action = None if root["previous"] is None else ActionXY(*root["previous"])
    env.robot.set_policy(policy)
    if points is not None:
        humans, observer.tracks = [], {}
        saved_tracks = {key: (measurement, stamp) for _, key, measurement, stamp in root["snapshot"]["tracks"]}
        for key, state in zip(root["state"].track_ids, root["state"].human_states):
            human = Human(cfg, "humans")
            human.time_step = .25
            human.set(state.px, state.py, state.px, state.py, state.vx, state.vy, 0., radius=state.radius)
            human.set_policy(Trajectory(points[key]))
            humans.append(human)
            measurement, stamp = saved_tracks[key]
            observer.tracks[human] = (key, ObservableState(*measurement), stamp)
        env.humans = humans
        env.human_times = [0.] * len(humans)
    return dict(env=env, observer=observer, policy=policy, shadow=points is not None,
                slots=root["state"].track_count, total=0., discount=1., minimum=1e6, steps=0,
                commands=[], done=False, event=None)


@torch.inference_mode()
def feedback(worlds, cfg, validate=False):
    states, prefixes, queries, rewards, clearances = [], [], [], [], []
    for world in worlds:
        state = world["observer"].observe(world["env"])
        if world["shadow"]:
            state = state._replace(track_count=world["slots"])
        policy = world["policy"]
        current = policy.encode(state)
        states.append((state, current))
        prefixes.append(window(list(policy.history) + [current], 24, "zero"))
        query, reward, clearance = policy.aligned_candidates(state)
        queries.append(query)
        rewards.append(reward)
        clearances.append(clearance)
    histories = stack_histories(prefixes)
    queries = np.stack([np.pad(q, ((0, 0), (0, histories.shape[2] - q.shape[1]), (0, 0))) for q in queries])
    policy = worlds[0]["policy"]
    value = policy.model.score_candidates(torch.as_tensor(histories, device=policy.device),
                                         torch.as_tensor(queries, device=policy.device))
    raw = torch.as_tensor(np.stack(rewards), device=policy.device) + policy.gamma * value
    scores = filter_scores(raw, torch.as_tensor(np.stack(clearances), device=policy.device), cfg).cpu().numpy()
    commands, checks = [], []
    for i, world in enumerate(worlds):
        policy = world["policy"]
        finite = np.sort(scores[i])
        gap = float(finite[-1] - finite[-2])
        if validate or gap <= 1e-4:
            native = policy.score(states[i][0])
            checks.append(dict(max_error=float(np.max(np.abs(native - scores[i]))),
                               batched_action=int(scores[i].argmax()), scalar_action=int(native.argmax()), gap=gap))
            if validate and (checks[-1]["max_error"] > 1e-4 or
                    (checks[-1]["batched_action"] != checks[-1]["scalar_action"] and gap > 1e-4)):
                raise RuntimeError("Batched critic scores violate native tolerance")
            scores[i] = native
        action = policy.action_space[int(scores[i].argmax())]
        previous, alpha = policy.last_action, cfg.getfloat("eval_protocol", "action_smoothing")
        if previous is not None:
            action = ActionXY(alpha * previous.vx + (1 - alpha) * action.vx,
                              alpha * previous.vy + (1 - alpha) * action.vy)
        policy.history.append(states[i][1])
        policy.last_action = action
        commands.append(action)
    return commands, checks


def advance(world, command, cfg):
    _, reward, done, truncated, info = world["env"].step(command)
    world["total"] += world["discount"] * reward
    world["discount"] *= cfg.getfloat("train", "gamma")
    world["steps"] += 1
    world["minimum"] = min(world["minimum"], info["dmin"])
    world["commands"].append(list(command))
    world["done"], world["event"] = done or truncated, info["event"]


def run_branches(root, model, cfg, device, commands, points=None, strict=False):
    worlds = [fork(root, model, cfg, device, points) for _ in commands]
    for world, command in zip(worlds, commands):
        world["policy"].history.append(world["policy"].encode(root["state"]))
        world["policy"].last_action = ActionXY(*command)
        advance(world, ActionXY(*command), cfg)
    while any(not w["done"] for w in worlds):
        live = [w for w in worlds if not w["done"]]
        if points is None or strict:
            actions = []
            for world in live:
                state = world["observer"].observe(world["env"])
                if world["shadow"]:
                    state = state._replace(track_count=world["slots"])
                actions.append(world["policy"].predict(state))
        else:
            actions, _ = feedback(live, cfg)
        for world, action in zip(live, actions):
            advance(world, action, cfg)
    return [dict(terminal=w["event"], return_=w["total"], seconds=w["steps"] * .25,
                minimum_clearance=w["minimum"], commands=w["commands"]) for w in worlds]


def archive_parity(root, native, cfg):
    expected = np.asarray(root["archived_tail_actions"])
    actual = np.asarray(native["commands"])
    command_error = float(np.max(np.abs(expected - actual))) if expected.shape == actual.shape else None
    gamma = cfg.getfloat("train", "gamma")
    expected_return = sum(gamma ** i * reward for i, reward in enumerate(root["archived_rewards"]))
    result = dict(same_terminal=native["terminal"] == root["episode"]["terminal"],
                  same_length=expected.shape == actual.shape, command_error=command_error,
                  return_error=float(abs(expected_return - native["return_"])),
                  seconds_error=float(abs(root["archived_remaining_seconds"] - native["seconds"])))
    result["passed"] = (result["same_terminal"] and result["same_length"]
                        and command_error <= 1e-6 and result["return_error"] <= 1e-6
                        and result["seconds_error"] <= 1e-8)
    return result


def root_commands(root, model, cfg, device):
    policy = ValuePolicy(model, cfg, device)
    policy.history.extend(root["history"][:-1])
    policy.last_action = None if root["previous"] is None else ActionXY(*root["previous"])
    baseline_scores = policy.score(root["state"])
    _, _, clearance = policy.aligned_candidates(root["state"])
    alpha = cfg.getfloat("eval_protocol", "action_smoothing")
    commands = np.asarray(policy.action_space)
    if root["previous"] is not None:
        commands = alpha * np.asarray(root["previous"])[None] + (1 - alpha) * commands
    return commands, baseline_scores, clearance


def select(raw, baseline, clearance, cfg):
    score = filter_scores(torch.as_tensor(raw)[None], torch.as_tensor(clearance)[None], cfg)[0].numpy()
    tied = np.flatnonzero(score >= score.max() - 1e-8)
    index = int(tied[int(np.argmax(baseline[tied]))])
    return dict(action=index, raw_returns=raw.tolist(), selected_scores=score.tolist(),
                raw_range=float(np.ptp(raw)), raw_best_ties=int(np.sum(raw >= raw.max() - 1e-8)),
                filtered_best_ties=len(tied))


@torch.inference_mode()
def execute(inputs, output, device, root_start=0, root_limit=None):
    input_hashes = json.loads((inputs / "hashes.json").read_text())
    for name, expected in input_hashes.items():
        if digest(inputs / name) != expected:
            raise RuntimeError("Frozen input hash changed: " + name)
    data = torch.load(inputs / "cohort.pt", map_location="cpu", weights_only=False)
    if source_hash() != data["source_sha256"]:
        raise RuntimeError("Frozen scientific core differs")
    model, cfg = load(inputs / "parent.pt", device)
    if model.kind != "cv":
        raise ValueError("Continuation must be the frozen native CV policy")
    predictors = {f"{seed}_{kind}": load(inputs / f"{seed}_{kind}.pt", device)[0]
                  for seed in SEEDS for kind in ("current", "gru", "kda")}
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(inputs=input_hashes, experiment_sha256=digest(Path(__file__)), rules=RULES,
                    source_sha256=data["source_sha256"], torch=torch.__version__, device=device,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Resumed experiment differs from saved manifest")
    else:
        save_json(manifest_path, manifest)
    completed = 0
    for index, root in enumerate(data["roots"]):
        if index < root_start or (root_limit is not None and index >= root_start + root_limit):
            continue
        path = output / f"root_{index:02d}.json"
        if path.exists():
            continue
        started = time.perf_counter()
        history = torch.as_tensor(root["history"][None], device=device)
        cv = model.forecast_positions(history)[0, :, :9].cpu().numpy().astype(np.float64)
        forecasts = dict(cv=cv, truth=cv.copy())
        forecasts["truth"][:, 1:9] = root["truth"]
        for name, predictor in predictors.items():
            forecasts[name] = predictor.forecast_positions(history)[0, :, :9].cpu().numpy().astype(np.float64)
        for points in forecasts.values():
            for key, state in zip(root["state"].track_ids, root["state"].human_states):
                points[key, 0] = state.position
            if not np.isfinite(points).all():
                raise RuntimeError("Nonfinite frozen forecast")
        commands, baseline, clearance = root_commands(root, model, cfg, device)
        # Native baseline trajectory is checked before inspecting scorer results.
        native_command = commands[int(baseline.argmax())]
        native = run_branches(root, model, cfg, device, [native_command])[0]
        archive_command_error = float(np.max(np.abs(native_command - root["archived_action"])))
        baseline_same_terminal = native["terminal"] == root["episode"]["terminal"]
        replay_parity = archive_parity(root, native, cfg)
        if not replay_parity["passed"]:
            save_json(output / f"parity_failure_{index:02d}.json", dict(root=index, parity=replay_parity,
                      baseline=native, archive_command_error=archive_command_error))
            raise RuntimeError("Frozen baseline does not reproduce archived continuation")
        parity_worlds = [fork(root, model, cfg, device, points=cv) for _ in range(4)]
        for world in parity_worlds:
            world["policy"].history.clear()
            world["policy"].history.extend(root["history"][:-1])
        _, parity = feedback(parity_worlds, cfg, validate=True)
        choices = {}
        print("ROOT_START", index, root["group"], root["episode"], root["tick"], flush=True)
        for name, points in forecasts.items():
            scoring_start = time.perf_counter()
            simulated = run_branches(root, model, cfg, device, commands, points)
            raw = np.asarray([r["return_"] for r in simulated])
            choices[name] = select(raw, baseline, clearance, cfg)
            chosen = choices[name]["action"]
            verified = run_branches(root, model, cfg, device, [commands[chosen]], points, strict=True)[0]
            proposed = simulated[chosen]
            same_trace = (len(verified["commands"]) == len(proposed["commands"])
                          and np.allclose(verified["commands"], proposed["commands"], atol=1e-6, rtol=0))
            native_fallback = not (same_trace and verified["terminal"] == proposed["terminal"]
                                   and abs(verified["return_"] - proposed["return_"]) <= 1e-6)
            if native_fallback:
                simulated = run_branches(root, model, cfg, device, commands, points, strict=True)
                raw = np.asarray([r["return_"] for r in simulated])
                choices[name] = select(raw, baseline, clearance, cfg)
            choices[name]["native_shadow_fallback"] = native_fallback
            choices[name]["shadow_native_validation"] = dict(initial_trace_agreed=same_trace,
                return_error=float(abs(verified["return_"] - proposed["return_"])),
                terminal_agreed=verified["terminal"] == proposed["terminal"])
            choices[name]["predicted_terminal"] = simulated[choices[name]["action"]]["terminal"]
            choices[name]["predicted_seconds"] = simulated[choices[name]["action"]]["seconds"]
            choices[name]["scoring_seconds"] = time.perf_counter() - scoring_start
            choices[name]["simulated_steps"] = sum(len(r["commands"]) for r in simulated)
            print("SCORED", index, name, choices[name]["action"],
                  round(choices[name]["scoring_seconds"], 2), flush=True)
        distinct = sorted({choice["action"] for choice in choices.values()})
        actual = run_branches(root, model, cfg, device, [commands[i] for i in distinct])
        realized = dict(zip(distinct, actual))
        for choice in choices.values():
            choice["realized"] = realized[choice["action"]]
            choice["rescued"] = native["terminal"] != "reach_goal" and choice["realized"]["terminal"] == "reach_goal"
            choice["damaged_success"] = native["terminal"] == "reach_goal" and choice["realized"]["terminal"] != "reach_goal"
            choice["delta_return"] = choice["realized"]["return_"] - native["return_"]
        save_json(path, dict(root=index, group=root["group"], episode=root["episode"], tick=root["tick"],
                  baseline=native, archive_command_error=archive_command_error,
                  baseline_same_terminal=baseline_same_terminal, archive_parity=replay_parity,
                  batch_validation=parity, forecasts={name: points.tolist() for name, points in forecasts.items()},
                  choices=choices,
                  elapsed_seconds=time.perf_counter() - started, torch=torch.__version__, device=device,
                  hardware=torch.cuda.get_device_name(0) if device == "cuda" else platform.node(),
                  scope=RULES["scope"]))
        print("ROOT_DONE", index, root["group"], root["episode"], root["tick"],
              "baseline", native["terminal"], "cv/truth", choices["cv"]["realized"]["terminal"],
              choices["truth"]["realized"]["terminal"], "seconds", round(time.perf_counter() - started, 2), flush=True)
        completed += 1
    print("COMPLETED", completed, "new roots", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--inputs", type=Path, default=OUT / "inputs")
    parser.add_argument("--output", type=Path, default=OUT / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-start", type=int, default=0)
    parser.add_argument("--root-limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare()
    else:
        execute(args.inputs, args.output, args.device, args.root_start, args.root_limit)


if __name__ == "__main__":
    main()
