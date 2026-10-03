"""Auditable occlusion IL/RL loop: shared demonstrations and reloaded weights."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.temporal_order import summarize
from shixu.model import build_model, load_weights
from shixu.policy import ValuePolicy
from shixu.observations import OccludedTracks
from shixu.runner import environment, orca_teacher, run_episode
from shixu.training import train


PROTOCOL = Path(__file__).with_name("occlusion_protocol.json")


def configuration(protocol, arm):
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
    cfg.add_section("observation")
    for section, key, value in (
        ("model", "architecture", "occlusion"), ("model", "representation", "tracks"),
        ("model", "backbone", arm), ("model", "width", protocol["width"]),
        ("model", "read_clock", protocol.get("read_clock", "candidate")),
        ("model", "layers", protocol["layers"]), ("buffer", "seq_len", protocol["history"]),
        ("observation", "retention_seconds", protocol["retention_seconds"]),
        ("train", "il_epochs", protocol["il_epochs"]),
        ("train", "batch_size", protocol["batch_size"]),
        ("train", "il_batch_size", protocol["batch_size"]),
        ("train", "updates_per_ep", protocol["rl_updates_per_episode"])
    ):
        cfg.set(section, key, str(value))
    if "interaction_order" in protocol:
        cfg.set("model", "interaction_order", protocol["interaction_order"])
    return cfg


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("shixu", "vendor/crowd_sim"):
        for file in sorted((root / folder).rglob("*.py")):
            digest.update(str(file.relative_to(root)).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def collect(path, protocol):
    cfg = configuration(protocol, "current")
    policy = ValuePolicy(build_model(cfg), cfg)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    teacher, episodes, attempts = orca_teacher(cfg), [], []
    for case in range(protocol["il_case_start"], protocol["il_case_start"] + cfg.getint("imitation_learning", "max_il_prefill")):
        episode = run_episode(env, policy, case, teacher=teacher, phase="train")
        attempts.append({"case": case, "terminal": episode["terminal"]})
        if episode["terminal"] == "reach_goal":
            episodes.append(episode)
        if len(episodes) == protocol["il_episodes"]:
            break
    if len(episodes) != protocol["il_episodes"]:
        raise RuntimeError("Insufficient successful legal-observation ORCA demonstrations")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": protocol, "episodes": episodes, "attempts": attempts}, path)
    print("COLLECTED", len(episodes), "successful /", len(attempts), "attempts", flush=True)


def exposure(episode):
    rows = episode["observations"]
    reentries, previous, seen = 0, set(), set()
    for row in rows:
        visible = {key["track_id"] for key, measured in zip(row["humans"], row["observed"]) if measured}
        reentries += len((visible - previous) & seen)
        seen.update(visible)
        previous = visible
    return {"person_frames": sum(row["population"] for row in rows),
            "hidden_person_frames": sum(row["population"] - row["visible"] for row in rows),
            "retained_hidden_person_frames": sum(row["retained_hidden"] for row in rows),
            "reentries": reentries}


def retention_diagnostic(data_path, protocol, destination):
    """Replay natural demonstrations; privileged geometry is diagnostic only."""
    from crowd_sim.envs.utils.action import ActionXY
    from shixu.features import encode_tracks
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    cfg = configuration(protocol, "current")
    policy = ValuePolicy(build_model(cfg), cfg)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    errors, near_errors, ages, velocity_errors = [], [], [], []
    compared, different_safe_sets = 0, {str(h): 0 for h in (.25, 1., 2.)}
    for episode in data["episodes"]:
        env.reset(options={"test_case": episode["case"]})
        observer = OccludedTracks(protocol["retention_seconds"])
        for frame, command in zip(episode["tokens"], episode["actions"]):
            state = observer.observe(env)
            np.testing.assert_allclose(encode_tracks(state), frame, atol=2e-5, rtol=1e-5)
            by_key = {row[0]: obj for obj, row in observer.tracks.items()}
            true_humans = []
            has_hidden = False
            for key, estimate, measured, age in zip(state.track_ids, state.human_states, state.observed, state.ages):
                truth = by_key[key].get_observable_state()
                true_humans.append(truth)
                if not measured:
                    has_hidden = True
                    error = float(np.hypot(estimate.px - truth.px, estimate.py - truth.py))
                    errors.append(error)
                    ages.append(age)
                    velocity_errors.append(float(np.hypot(estimate.vx - truth.vx, estimate.vy - truth.vy)))
                    if np.hypot(truth.px - state.self_state.px, truth.py - state.self_state.py) < 2:
                        near_errors.append(error)
            if has_hidden:
                compared += 1
                for horizon in (.25, 1., 2.):
                    policy.time_step = horizon
                    _, _, nominal = policy.aligned_candidates(state)
                    _, _, actual = policy.aligned_candidates(state._replace(human_states=true_humans))
                    different_safe_sets[str(horizon)] += int(np.any((nominal >= .2) != (actual >= .2)))
                policy.time_step = cfg.getfloat("env", "time_step")
            env.step(ActionXY(*command))
    def distribution(values):
        return {"count": len(values), "mean": float(np.mean(values)) if values else None,
                "quantiles_50_90_99": np.quantile(values, [.5, .9, .99]).tolist() if values else None,
                "above_005": float(np.mean(np.array(values) > .05)) if values else None}
    result = {"episodes": len(data["episodes"]), "position_error_m": distribution(errors),
              "near_2m_position_error_m": distribution(near_errors),
              "velocity_error_mps": distribution(velocity_errors), "ages_seconds": distribution(ages),
              "states_with_retained_hidden": compared, "endpoint_safe_support_changed_by_horizon": different_safe_sets,
              "scope": "Offline hidden-current-state truth substitution with linear endpoint projection. Not full-future truth, swept-path safety, a deployed oracle or proof of closed-loop gain"}
    Path(destination).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def reentry_probe(data_path, destination):
    """Grouped linear probe; all predictors are frozen at the last visible frame."""
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    rows = []
    for group, episode in enumerate(data["episodes"]):
        history, previous = {}, {}
        for observation in episode["observations"]:
            now = observation["time"]
            robot = np.asarray(observation["robot"])
            humans = {h["track_id"]: np.asarray(h["state"]) for h in observation["humans"]}
            visible = dict(zip(humans, observation["observed"]))
            for key, human in humans.items():
                past = [p for p in history.get(key, []) if now - p[0] <= 5.75]
                if visible[key] and previous.get(key) is False and len(past) >= 2:
                    stamp, last, neighbours, distance = past[-1]
                    gap = now - stamp
                    if gap > .25 + 1e-6:
                        velocities = np.array([p[1][2:4] for p in past[-4:]])
                        relative_velocity = np.mean([h[2:4] - last[2:4] for h in neighbours], axis=0) if neighbours else np.zeros(2)
                        features = np.r_[last[2:4], velocities.mean(0), velocities[-1] - velocities[0],
                                         gap, distance, relative_velocity, len(neighbours)]
                        rows.append((group, features, human[2:4], last[2:4], distance < 2, gap))
                if visible[key]:
                    neighbours = [h.copy() for j, h in humans.items() if j != key and visible[j]
                                  and np.linalg.norm(h[:2] - human[:2]) < 2]
                    distance = float(np.linalg.norm(human[:2] - robot[:2]))
                    history.setdefault(key, []).append((now, human.copy(), neighbours, distance))
                previous[key] = visible[key]
    if not rows:
        raise RuntimeError("No eligible natural re-entry observations")
    groups = np.asarray([r[0] for r in rows])
    features = np.asarray([r[1] for r in rows])
    targets, cv = [np.asarray([r[index] for r in rows]) for index in (2, 3)]
    near = np.asarray([r[4] for r in rows])
    def metrics(predicted):
        error = np.linalg.norm(predicted - targets, axis=1)
        return {"mean_mps": float(error.mean()), "median_mps": float(np.median(error)),
                "p90_mps": float(np.quantile(error, .9)),
                "near_2m_mean_mps": float(error[near].mean()) if near.any() else None}
    errors = {"CV": metrics(cv)}
    for name, columns in (("own_history", slice(0, 8)), ("own_history_plus_neighbour_summary", slice(None))):
        prediction = np.zeros_like(targets)
        for fold in range(5):
            test = groups % 5 == fold
            train = ~test
            inputs = features[:, columns]
            mean, scale = inputs[train].mean(0), np.maximum(inputs[train].std(0), 1e-4)
            inputs = np.c_[(inputs - mean) / scale, np.ones(len(inputs))]
            penalty = np.eye(inputs.shape[1])
            penalty[-1, -1] = 0
            weights = np.linalg.solve(inputs[train].T @ inputs[train] + 5 * penalty,
                                      inputs[train].T @ (targets[train] - cv[train]))
            prediction[test] = cv[test] + inputs[test] @ weights
        errors[name] = metrics(prediction)
    result = {"scope": "Five-fold episode-disjoint linear probe on successful ORCA visits, fixed ridge penalty 5. "
                       "Inputs use last-visible measurements only; gap is elapsed prediction time. Future re-entry "
                       "velocity is an offline target. Not navigation training or action-value headroom proof.",
              "rows": len(rows), "near_2m_rows": int(near.sum()), "prediction_errors": errors,
              "gap_median_seconds": float(np.median([r[5] for r in rows]))}
    Path(destination).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def collision_diagnostic(root, protocol, destination):
    """Replay archived controls; privileged identities classify failures only."""
    from crowd_sim.envs.utils.action import ActionXY
    result = {}
    for arm in protocol["arms"]:
        origin = Path(protocol.get("reference_roots", {}).get(arm, root))
        records = []
        for seed in protocol["seeds"]:
            archived = json.loads((origin / str(seed) / arm / "result.json").read_text())
            cfg = configuration(archived["protocol"], arm)
            policy = ValuePolicy(build_model(cfg), cfg)
            for row in archived["episodes"]:
                if row["terminal"] != "collision":
                    continue
                env = environment(cfg, policy, row["geometry"], row["people"])
                env.reset(options={"test_case": row["case"]})
                observer = OccludedTracks(protocol["retention_seconds"])
                measured_run, previous_command = {}, None
                for command in row["actions"]:
                    command = np.asarray(command)
                    state = observer.observe(env)
                    measured_keys = {key for key, measured in zip(state.track_ids, state.observed) if measured}
                    for key in range(state.track_count):
                        measured_run[key] = measured_run.get(key, 0) + 1 if key in measured_keys else 0
                    action = ActionXY(*command)
                    collision_human = None
                    for human in env.humans:
                        relative = np.array([human.px - env.robot.px, human.py - env.robot.py])
                        displacement = env.time_step * (np.array([human.vx, human.vy]) - command)
                        fraction = float(np.clip(-relative @ displacement / max(displacement @ displacement, 1e-12), 0, 1))
                        clearance = np.linalg.norm(relative + fraction * displacement) - human.radius - env.robot.radius
                        if clearance < 0:
                            collision_human = human
                            break
                    _, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        if info["event"] != "collision" or collision_human is None:
                            raise RuntimeError("Archived collision could not be reproduced")
                        stored = observer.tracks.get(collision_human)
                        if stored is None:
                            category = "never_observed"
                        elif stored[0] not in state.track_ids:
                            category = "known_but_expired"
                        elif state.observed[state.track_ids.index(stored[0])]:
                            category = "visible"
                        else:
                            category = "retained_hidden"
                        estimates = np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in state.human_states]).reshape(-1, 5)
                        endpoint = np.array([state.self_state.px, state.self_state.py]) + env.time_step * command
                        clearance = float(np.min(np.linalg.norm(estimates[:, :2] + env.time_step * estimates[:, 2:4] - endpoint, axis=1)
                                                 - estimates[:, 4] - state.self_state.radius)) if len(estimates) else None
                        _, _, support_clearance = policy.aligned_candidates(state)
                        alpha = cfg.getfloat("eval_protocol", "action_smoothing") if previous_command is not None else 0
                        selected = (command - alpha * previous_command) / (1 - alpha) if alpha else command
                        index = int(np.argmin(np.linalg.norm(np.asarray(policy.action_space) - selected, axis=1)))
                        records.append({"seed": seed, "case": row["case"], "people": row["people"],
                                        "geometry": row["geometry"], "category": category,
                                        "visible_streak_steps": measured_run.get(stored[0], 0) if stored else 0,
                                        "safe_actions_before_smoothing": int(np.sum(support_clearance >= .2)),
                                        "selected_before_smoothing_endpoint_clearance": float(support_clearance[index]),
                                        "executed_action_retained_endpoint_clearance": clearance})
                        break
                    previous_command = command
        result[arm] = {"counts": {name: sum(r["category"] == name for r in records)
                                  for name in ("visible", "retained_hidden", "known_but_expired", "never_observed")},
                       "records": records}
    result["scope"] = ("Archived executed-action collision replay only. Simulator truth classifies the collider, "
                       "never enters the policy. Different visited trajectories preclude a causal comparison. "
                       "Executed controls include inherited smoothing; endpoint safety is not swept-path safety.")
    Path(destination).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["counts"] for k, v in result.items() if isinstance(v, dict)}, indent=2), flush=True)


def recall_probe(weights, visits, device, destination):
    """Frozen legal states and weights; memory removal is diagnostic, not retraining."""
    from crowd_sim.envs.utils.action import ActionXY
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    policy = ValuePolicy(build_model(cfg), cfg, device)
    load_weights(policy.model, weights, device)
    archived = json.loads(Path(visits).read_text())
    chosen, cells = [], {}
    for row in archived["episodes"]:
        key = (row["people"], row["geometry"])
        if cells.get(key, 0) < 2:
            chosen.append(row)
            cells[key] = cells.get(key, 0) + 1
    records = []
    original = policy.model.read_history
    for row in chosen:
        env = environment(cfg, policy, row["geometry"], row["people"])
        env.reset(options={"test_case": row["case"]})
        observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
        policy.reset()
        for tick, command in enumerate(row["actions"]):
            state = observer.observe(env)
            if tick % 4 == 0:
                scores = {}
                for scope in ("all", "visible", "none"):
                    def read(memory, queries):
                        states, seen, latest = memory
                        allowed = torch.zeros_like(seen)
                        if scope == "all":
                            allowed = torch.ones_like(seen)
                        elif scope == "visible":
                            for key, measured in zip(state.track_ids, state.observed):
                                allowed[:, key] = measured
                        return original((states, seen & allowed, latest), queries)
                    policy.model.read_history = read
                    try:
                        scores[scope] = policy.score(state)
                    finally:
                        policy.model.read_history = original
                records.append({"case": row["case"], "geometry": row["geometry"], "people": row["people"],
                                "tick": tick, "hidden": observer.counts["retained_hidden"],
                                "selected": {name: int(np.argmax(values)) for name, values in scores.items()},
                                "hidden_recall_changes_action": int(np.argmax(scores["all"]) != np.argmax(scores["visible"])),
                                "all_recall_changes_action": int(np.argmax(scores["all"]) != np.argmax(scores["none"]))})
            policy.history.append(policy.encode(state))
            _, _, done, truncated, _ = env.step(ActionXY(*command))
            if done or truncated:
                break
    hidden = [row for row in records if row["hidden"]]
    result = {"checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
              "states": len(records), "states_with_retained_hidden": len(hidden),
              "hidden_recall_action_changes": sum(r["hidden_recall_changes_action"] for r in hidden),
              "all_recall_action_changes": sum(r["all_recall_changes_action"] for r in records), "records": records,
              "scope": "Twelve prespecified archived current-value visits (first two cases per geometry/population), "
                       "uniform every-fourth-step sampling. Identical states and weights for all ablations. "
                       "Root candidate changes are not proof of better actions or closed-loop improvement."}
    Path(destination).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2), flush=True)


def address_probe(weights, data_path, device, destination):
    """Compare legal actor addresses and matrix states, without fitting a probe."""
    from shixu.features import stack_histories, window
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    model = build_model(cfg).to(device).eval()
    load_weights(model, weights, device)
    if model.substrate != "kda":
        raise ValueError("Address probe requires a KDA matrix memory")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    windows = [window(ep["tokens"][:tick + 1], 24, "zero") for ep in data["episodes"]
               for tick in range(0, len(ep["tokens"]), 8)]
    values = {name: [] for name in ("own_features", "contextual_features", "keys", "first_matrix", "last_matrix")}
    with torch.inference_mode():
        for start in range(0, len(windows), 64):
            tokens = torch.as_tensor(stack_histories(windows[start:start + 64]), device=device)
            prefix = tokens[:, :-1]
            own, _, valid = model._features(prefix)
            contextual = model._attend(own, valid)
            encoded = contextual if model.interaction_order == "write" else own
            cell = model.temporal_encoder.cells[0]
            keys = cell.k(cell.input_norm(encoded[:, -1]))
            states, seen, _ = model.encode_history(prefix)
            batch, _, people, _ = own.shape
            mask = valid[:, -1] & seen
            pairs = (mask[:, :, None] & mask[:, None, :]) & torch.ones(people, people, device=device, dtype=torch.bool).triu(1)
            representations = {"own_features": own[:, -1], "contextual_features": contextual[:, -1], "keys": keys,
                               "first_matrix": states[0].reshape(batch, people, -1),
                               "last_matrix": states[-1].reshape(batch, people, -1)}
            for name, representation in representations.items():
                normed = torch.nn.functional.normalize(representation, dim=-1)
                cosine = normed @ normed.transpose(-1, -2)
                values[name].extend(cosine[pairs].cpu().tolist())
    result = {"checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
              "phase": checkpoint.get("phase", "final"), "interaction_order": model.interaction_order,
              "states": len(windows),
              "pairwise_cosine": {name: {"pairs": len(rows), "mean": float(np.mean(rows)),
                                        "median": float(np.median(rows)), "above_099": float(np.mean(np.asarray(rows) > .99))}
                                  for name, rows in values.items()},
              "scope": "Uniform every-eighth-frame successful-demonstration probe. Only simultaneously measured, "
                       "previously seen actors are compared. High cosine is not proof of information equivalence "
                       "or poor navigation. Compare matched training phases; this does not select checkpoints."}
    Path(destination).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def evaluate_weights(path, protocol, device, cases, seed, read_clock=None):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    training_clock = cfg.get("model", "read_clock", fallback="candidate")
    if read_clock is not None:
        cfg.set("model", "read_clock", read_clock)
    policy = ValuePolicy(build_model(cfg), cfg, device)
    load_weights(policy.model, path, device)
    np.random.seed(seed)
    records, timings = [], []
    score = policy.score

    def timed_score(state):
        start = time.perf_counter()
        value = score(state)
        timings.append(1000 * (time.perf_counter() - start))
        return value

    policy.score = timed_score
    for people in protocol["people"]:
        for geometry in protocol["geometries"]:
            env = environment(cfg, policy, geometry, people)
            for case in cases:
                episode = run_episode(env, policy, case)
                row = {key: episode[key] for key in ("case", "terminal", "navigation_time", "path", "minimum_clearance")}
                row.update(people=people, geometry=geometry, return_=float(sum(episode["rewards"])),
                           exposure=exposure(episode), actions=episode["actions"])
                records.append(row)
            print("EVALUATED", path.parent.name, people, geometry, summarize(records[-len(cases):]), flush=True)
    return {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training_read_clock": training_clock,
            "evaluation_read_clock": cfg.get("model", "read_clock", fallback="candidate"),
            "intervention_scope": "Frozen-weight read-clock ablation, not a retrained algorithm" if read_clock else None,
            "checkpoint_phase": checkpoint.get("phase", "final"), "episodes": records,
            "summary": summarize(records),
            "score_median_ms": float(np.median(timings)), "score_p95_ms": float(np.quantile(timings, .95)),
            "timing_scope": "Complete 80-action scoring during evaluation, includes transfers; concurrent training may contend",
            "cells": [{"people": people, **summarize([r for r in records if r["people"] == people])}
                      for people in protocol["people"]]}


def run(root, protocol, arm, seed, data_path, device, il_weights=None):
    output = root / str(seed) / arm
    if (output / "result.json").exists():
        raise RuntimeError("A completed result already exists; do not overwrite evidence")
    demonstrations = torch.load(data_path, map_location="cpu", weights_only=False)
    if demonstrations["protocol"] != protocol:
        raise ValueError("Demonstration collection protocol mismatch")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = configuration(protocol, arm)
    policy = ValuePolicy(build_model(cfg), cfg, device)
    prior_il, prior_il_seconds, discarded_rl_seconds = [], 0.0, 0.0
    if il_weights:
        checkpoint = torch.load(il_weights, map_location="cpu", weights_only=False)
        if (checkpoint.get("phase") != "il" or checkpoint["seed"] != seed
                or checkpoint["config"] != {s: dict(cfg[s]) for s in cfg.sections()}):
            raise ValueError("The resumed checkpoint is not the matched final-IL model")
        old_log = [json.loads(line) for line in Path(il_weights).with_name("learning.jsonl").read_text().splitlines()]
        prior_il = [row for row in old_log if row["phase"] == "il"]
        if [row["epoch"] for row in prior_il] != list(range(1, protocol["il_epochs"] + 1)):
            raise ValueError("Final-IL reuse requires the complete original IL log")
        prior_il_seconds = prior_il[-1]["elapsed_seconds"]
        discarded_rl_seconds = max(0, old_log[-1]["elapsed_seconds"] - prior_il_seconds)
        load_weights(policy.model, il_weights, device)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(policy.device)
    with (output / "learning.jsonl").open("w") as log:
        for row in prior_il:
            log.write(json.dumps(dict(row, reused_final_il=True)) + "\n")
        if il_weights:
            torch.save(checkpoint, output / "il.pt")
        def report(row):
            if not np.isfinite(row["loss"]):
                raise RuntimeError("Nonfinite training loss")
            row["elapsed_seconds"] = prior_il_seconds + time.perf_counter() - started
            log.write(json.dumps(row) + "\n")
            log.flush()
            if row["phase"] == "il" and row["epoch"] == protocol["il_epochs"]:
                torch.save({"model": policy.model.state_dict(), "seed": seed, "phase": "il",
                            "config": {s: dict(cfg[s]) for s in cfg.sections()}}, output / "il.pt")
            if row["phase"] == "il" and row["epoch"] % 10 == 0 or row["phase"] == "rl" and row["episode"] % 100 == 0:
                print(arm, seed, row, flush=True)
        train(env, policy, cfg, output / "model.pt", seed, protocol["il_episodes"], protocol["rl_episodes"],
              demonstrations=demonstrations["episodes"], rl_case_start=protocol["rl_case_start"], report=report,
              pretrained_il=bool(il_weights))
    training_seconds = prior_il_seconds + time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(policy.device) / 1024 ** 2 if policy.device.type == "cuda" else None
    parameters = sum(p.numel() for p in policy.model.parameters())
    del policy, env
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    start = time.perf_counter()
    result = evaluate_weights(output / "model.pt", protocol, device, protocol["development_cases"], seed)
    result.update(protocol=protocol, source_sha256=source_hash(), seed=seed, arm=arm,
                  demonstration_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                  parameters=parameters, training_seconds=training_seconds,
                  reused_il_sha256=hashlib.sha256(Path(il_weights).read_bytes()).hexdigest() if il_weights else None,
                  reused_il_seconds=prior_il_seconds, discarded_partial_rl_seconds=discarded_rl_seconds,
                  total_consumed_training_seconds=training_seconds + discarded_rl_seconds,
                  evaluation_seconds=time.perf_counter() - start, peak_allocated_mib=peak_mib,
                  torch=torch.__version__, host=platform.node(), device=str(device),
                  hardware=torch.cuda.get_device_name() if str(device).startswith("cuda") else "CPU")
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", arm, seed, result["summary"], flush=True)


def queue(args, protocol):
    pending = [(seed, arm) for seed in (args.seeds or protocol["seeds"])
               for arm in (args.arms or protocol["arms"])]
    active, failed = [], []
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < args.workers:
            seed, arm = pending.pop(0)
            if (root / str(seed) / arm / "result.json").exists():
                print("ALREADY_COMPLETE", seed, arm, flush=True)
                continue
            log = (root / (str(seed) + "_" + arm + ".log")).open("w")
            command = [sys.executable, "-m", "experiments.occlusion", "run", "--protocol", args.protocol,
                       "--root", args.root, "--data", args.data, "--device", args.device,
                       "--seed", str(seed), "--arm", arm]
            if args.il_root:
                command.extend(("--il-weights", str(Path(args.il_root) / str(seed) / arm / "il.pt")))
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active.append((child, log, seed, arm))
            print("STARTED", seed, arm, child.pid, flush=True)
        remaining = []
        for child, log, seed, arm in active:
            status = child.poll()
            if status is None:
                remaining.append((child, log, seed, arm))
            else:
                log.close()
                print("EXIT", seed, arm, status, flush=True)
                if status:
                    failed.append((seed, arm, status))
        active = remaining
        time.sleep(2)
    if failed:
        raise RuntimeError("Failed arms: " + repr(failed))


def comparison(root, protocol):
    models = {}
    for arm in protocol["arms"]:
        reference = protocol.get("reference_roots", {}).get(arm)
        origin = Path(reference) if reference else root
        rows = [json.loads((origin / str(seed) / arm / "result.json").read_text()) for seed in protocol["seeds"]]
        if reference:
            if arm != "current":
                raise ValueError("Only the unchanged current-track arm may be reused")
            shared = ("width", "layers", "history", "retention_seconds", "sensor", "association", "il_episodes",
                      "il_epochs", "il_case_start", "rl_episodes", "rl_case_start", "batch_size",
                      "rl_updates_per_episode", "training_people", "training_geometry", "people", "geometries",
                      "development_cases", "seeds")
            if any(any(r["protocol"][key] != protocol[key] for key in shared) for r in rows):
                raise ValueError("Reused baseline has different data, budget or evaluation")
        elif any(r["protocol"] != protocol for r in rows):
            raise ValueError("Result protocol mismatch")
        models[arm] = {"summary": summarize([e for r in rows for e in r["episodes"]]),
                       "primary_by_seed": [summarize([e for e in r["episodes"] if e["people"] > 5]) for r in rows],
                       "cells": [{"people": n, **summarize([e for r in rows for e in r["episodes"] if e["people"] == n])}
                                 for n in protocol["people"]],
                       "parameters": rows[0]["parameters"], "reference_root": reference,
                       "training_seconds": [r["training_seconds"] for r in rows],
                       "inference_median_ms": [r["score_median_ms"] for r in rows],
                       "exposure": {key: sum(e["exposure"][key] for r in rows for e in r["episodes"])
                                    for key in rows[0]["episodes"][0]["exposure"]}}
    contrasts = []
    for control in ("gru", "current"):
        a, b = models["kda"]["primary_by_seed"], models[control]["primary_by_seed"]
        changes = {key: [100 * (x[key] - y[key]) for x, y in zip(a, b)] for key in ("sr", "cr", "tr")}
        time_change = np.mean([x["success_time"] / y["success_time"] - 1 for x, y in zip(a, b)
                               if x["success_time"] and y["success_time"]])
        positive = (np.mean(changes["sr"]) >= protocol["meaningful_gain_pp"]
                    and sum(d > 0 for d in changes["sr"]) >= protocol["required_same_direction_seeds"]
                    and np.mean(changes["cr"]) <= protocol["maximum_collision_increase_pp"]
                    and np.mean(changes["tr"]) <= protocol["maximum_timeout_increase_pp"]
                    and np.isfinite(time_change) and time_change <= protocol["maximum_success_time_increase_fraction"])
        contrasts.append({"control": control, "seed_changes_pp": changes,
                          "mean_changes_pp": {k: float(np.mean(v)) for k, v in changes.items()},
                          "success_time_change_fraction": float(time_change),
                          "development_positive": bool(positive)})
    result = {"protocol": protocol, "models": models, "contrasts": contrasts,
              "status": "DEVELOPMENT_POSITIVE_REQUIRES_FRESH_CONFIRMATION" if all(c["development_positive"] for c in contrasts)
                        else "VERSION_REQUIRES_DIAGNOSIS_NOT_DIRECTION_REJECTED"}
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "run", "evaluate", "queue", "summarize", "diagnose", "history-probe", "collisions", "recall-probe", "address-probe"))
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--data")
    parser.add_argument("--root")
    parser.add_argument("--arm", choices=("current", "gru", "kda"))
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights")
    parser.add_argument("--il-weights")
    parser.add_argument("--il-root")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--read-clock", choices=("candidate", "observation"))
    parser.add_argument("--arms", nargs="+", choices=("current", "gru", "kda"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(Path(args.protocol).read_text())
    if args.mode == "collect":
        collect(Path(args.data), protocol)
    elif args.mode == "run":
        run(Path(args.root), protocol, args.arm, args.seed, args.data, args.device, args.il_weights)
    elif args.mode == "queue":
        queue(args, protocol)
    elif args.mode == "summarize":
        comparison(Path(args.root), protocol)
    elif args.mode == "diagnose":
        retention_diagnostic(args.data, protocol, args.root)
    elif args.mode == "history-probe":
        reentry_probe(args.data, args.root)
    elif args.mode == "collisions":
        collision_diagnostic(Path(args.root), protocol, Path(args.root) / "collision_diagnostic.json")
    elif args.mode == "recall-probe":
        recall_probe(args.weights, args.data, args.device, args.root)
    elif args.mode == "address-probe":
        address_probe(args.weights, args.data, args.device, args.root)
    else:
        rows = evaluate_weights(args.weights, protocol, args.device,
                                protocol["confirmation_cases"] if args.confirmation else protocol["development_cases"],
                                args.seed, args.read_clock)
        destination = Path(args.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
