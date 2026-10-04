"""Controlled goal ownership and static-anchor/history information audit."""

import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.latent_information import (build_samples, check_vendor, configuration_for_audit,
                                           errors, fit_probe, legal_actors, predict, snapshot, summarize_errors)
from experiments.latent_decisions import restore_check, root_inputs, rollout, summarize, teacher_prefix
from experiments.occlusion import source_hash
from shixu.observations import OccludedTracks
from shixu.policy import actions
from shixu.runner import environment, orca_teacher


PROTOCOL = Path(__file__).with_name("circle_coupling_protocol.json")


def assign_goals(env, regime, case):
    original = np.asarray([human.get_goal_position() for human in env.humans])
    assignment = np.arange(len(env.humans))
    if regime == "circle_permuted":
        assignment = np.roll(assignment, 1 + case % (len(env.humans) - 1))
        for human, index in zip(env.humans, assignment):
            human.gx, human.gy = original[index]
    return original, assignment


def collect_case(task):
    regime, split, case = task
    cfg = configuration_for_audit()
    teacher = orca_teacher(cfg)
    geometry = "square" if regime == "square" else "circle"
    env = environment(cfg, teacher, geometry, 5)
    env.reset(options={"test_case": case})
    positions = np.asarray([human.get_position() for human in env.humans])
    original, assignment = assign_goals(env, regime, case)
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    frames, commands, rewards = [], [], []
    while True:
        state = observer.observe(env)
        saved = snapshot(env, observer)
        frames.append({"legal": legal_actors(state), "robot": saved["robot"], "snapshot": saved,
                       "visible": observer.counts["visible"], "retained_hidden": observer.counts["retained_hidden"]})
        action = teacher.predict(state)
        _, reward, done, truncated, info = env.step(action)
        commands.append([action.vx, action.vy])
        rewards.append(reward)
        if done or truncated:
            break
    return {"regime": regime, "split": split, "geometry": geometry, "case": case,
            "initial_positions": positions, "original_goals": original, "assignment": assignment,
            "frames": frames, "actions": commands, "rewards": rewards, "terminal": info["event"]}


def controlled_design(rows, mode):
    current = np.stack([row["context"] for row in rows])
    birth = np.stack([row.get("birth", row["statistics"][:2]) for row in rows])
    past = np.stack([row["history"][:-1].copy() for row in rows])
    if mode == "birth_bag24":
        # Canonical content ordering retains the observation multiset, not arrival order.
        past = np.stack([np.asarray(sorted(sequence.tolist())) for sequence in past])
    elif mode == "birth_statistics":
        past[:] = 0
        past.reshape(len(rows), -1)[:, :16] = np.stack([row["statistics"][2:] for row in rows])
    elif mode in ("current", "birth"):
        past[:] = 0
    elif mode == "birth_history3":
        past[:, :21] = 0
    elif mode not in ("history24", "birth_history24"):
        raise ValueError(mode)
    if mode in ("current", "history24"):
        birth[:] = 0
    return np.concatenate((current, birth, past.reshape(len(rows), -1)), axis=1)


def all_rows(data):
    rows = []
    for regime in data["protocol"]["regimes"]:
        episodes = [ep for ep in data["episodes"] if ep["regime"] == regime]
        first_seen = {}
        for episode in episodes:
            for tick, frame in enumerate(episode["frames"]):
                for key, actor in enumerate(frame["legal"]):
                    if actor[6] and actor[7]:
                        first_seen.setdefault((episode["case"], key), tick)
        for row in build_samples(episodes, data["protocol"]):
            row["regime"] = regime
            row["first_sighting_tick"] = first_seen[(row["case"], row["key"])]
            row["born_at_reset"] = row["first_sighting_tick"] == 0
            rows.append(row)
    return rows


def bootstrap_difference(left, right, cases, repetitions, confidence):
    keys = sorted(set(cases))
    values = np.asarray([np.mean(left[np.asarray(cases) == key] - right[np.asarray(cases) == key]) for key in keys])
    if not len(values):
        return {"cases": 0, "difference": None, "interval": None}
    sampled = np.random.default_rng(724).integers(len(values), size=(repetitions, len(values)))
    means = values[sampled].mean(1)
    tail = (1 - confidence) / 2
    return {"cases": len(values), "difference": float(values.mean()),
            "interval": np.quantile(means, [tail, 1 - tail]).tolist(), "confidence": confidence}


def paired_metrics(left, right, rows, protocol):
    result = {}
    for metric in left:
        selected = np.asarray([metric != "goal_angle_degrees" or row["goal_distance"] >= .25 for row in rows])
        cases = [row["case"] for row, keep in zip(rows, selected) if keep]
        result[metric] = bootstrap_difference(left[metric][selected], right[metric][selected], cases,
                                              protocol["bootstrap_repetitions"], 1 - .05 / 3)
    return result


def fit_all(data, output):
    protocol, rows = data["protocol"], all_rows(data)
    results, models = {}, {}
    for regime in protocol["regimes"]:
        train = [row for row in rows if row["regime"] == regime and row["split"] == "train"]
        validation = [row for row in rows if row["regime"] == regime and row["split"] == "validation"]
        test = [row for row in rows if row["regime"] == regime and row["split"] == "test"]
        indices = np.random.default_rng(723).choice(len(train), min(protocol["landmarks"], len(train)), replace=False)
        models[regime], metrics = {}, {}
        for mode in protocol["modes"]:
            started = time.perf_counter()
            model = fit_probe(train, validation, mode, protocol, indices, design_fn=controlled_design)
            model["fit_seconds"] = time.perf_counter() - started
            models[regime][mode] = model
            metrics[mode] = errors(predict(model, test, design_fn=controlled_design), test)
            print("FITTED", regime, mode, round(model["fit_seconds"], 2), flush=True)
        cv = np.zeros((len(test), 8))
        current_velocity = np.stack([row["context"][2:4] for row in test])
        cv[:, :2] = -np.stack([row["birth"] for row in test]) - np.stack([row["context"][:2] for row in test])
        cv[:, 2:4], cv[:, 4:6], cv[:, 6:8] = current_velocity, current_velocity, 2 * current_velocity
        metrics["circle_birth_prior"] = errors(cv, test)
        born = np.asarray([row["born_at_reset"] for row in test])
        rule_error = metrics["circle_birth_prior"]["goal_position_error_m"]
        results[regime] = {"train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test),
                           "test_cases": len({row["case"] for row in test}),
                           "errors": {mode: summarize_errors(metric, test) for mode, metric in metrics.items()},
                           "contrasts": {left + "_minus_" + right: paired_metrics(metrics[left], metrics[right], test, protocol)
                                         for left, right in protocol["contrasts"]},
                           "born_at_reset": {"samples": int(born.sum()),
                                             "fraction": float(born.mean()),
                                             "mean_rule_goal_error": float(rule_error[born].mean()),
                                             "maximum_rule_goal_error": float(rule_error[born].max()),
                                             "goal_exact_within_1e6": int(np.sum(rule_error[born] < 1e-6))},
                           "model_meta": [{name: model[name] for name in ("mode", "alpha", "scale", "coefficients", "validation_error", "fit_seconds")}
                                          for model in models[regime].values()]}
        torch.save({"models": models, "protocol": protocol}, output / "models.pt")
        (output / "probe_results.json").write_text(json.dumps({"regimes": results, "protocol": protocol,
                                                              "source_sha256": data["source_sha256"]}, indent=2))
    return results


def intervention_checks(data):
    native = {ep["case"]: ep for ep in data["episodes"] if ep["regime"] == "circle"}
    matched = 0
    for changed in (ep for ep in data["episodes"] if ep["regime"] == "circle_permuted"):
        original = native[changed["case"]]
        np.testing.assert_array_equal(original["initial_positions"], changed["initial_positions"])
        np.testing.assert_array_equal(original["original_goals"], changed["original_goals"])
        assert np.all(changed["assignment"] != np.arange(5))
        np.testing.assert_array_equal(changed["frames"][0]["snapshot"]["humans"],
                                      [np.asarray([*position, 0, 0, .3, *goal, 1, 0], dtype=np.float32)
                                       for position, goal in zip(changed["initial_positions"],
                                                                 changed["original_goals"][changed["assignment"]])])
        matched += 1
    return {"matched_start_and_goal_set_pairs": matched, "goal_ownership_has_no_fixed_points": True}


def cross_all(data, output):
    models = torch.load(output / "models.pt", map_location="cpu", weights_only=False)["models"]
    rows = all_rows(data)
    matrices, contrasts = {}, {}
    for target in data["protocol"]["regimes"]:
        selected = [row for row in rows if row["regime"] == target and row["split"] == "test"]
        by_source = {}
        for source in data["protocol"]["regimes"]:
            by_source[source] = {mode: errors(predict(models[source][mode], selected, design_fn=controlled_design), selected)
                                 for mode in ("current", "birth", "history24", "birth_history24")}
            matrices[source + "->" + target] = {mode: summarize_errors(metrics, selected)
                                                for mode, metrics in by_source[source].items()}
        if target != "circle":
            contrasts[target] = paired_metrics(by_source["circle"]["history24"], by_source[target]["history24"],
                                                selected, data["protocol"])
    (output / "cross_distribution_results.json").write_text(json.dumps({"errors": matrices, "contrasts": contrasts,
         "scope": "Post-hoc same-held-out-state evaluation of already frozen fits. No refitting or test selection. "
                  "Differences concern training distribution, not proof of the cause of any old navigation model failure."}, indent=2))


def decision_roots(data):
    rule = data["protocol"]["decision_selection"]
    chosen = []
    for regime in data["protocol"]["regimes"]:
        episodes = sorted((ep for ep in data["episodes"] if ep["regime"] == regime and ep["split"] == rule["split"]),
                          key=lambda ep: ep["case"])
        added = 0
        for episode in episodes:
            for tick, frame in enumerate(episode["frames"][:-1]):
                robot, humans = frame["robot"], frame["legal"]
                if frame["snapshot"]["time"] < rule["minimum_seconds"] or frame["visible"] != 5:
                    continue
                if np.linalg.norm(robot[:2] - robot[5:7]) <= rule["robot_goal_distance_min_m"]:
                    continue
                clearance = np.min(np.linalg.norm(humans[:, :2] - robot[:2], axis=1) - humans[:, 4] - robot[4])
                if clearance <= rule["maximum_surface_clearance_m"]:
                    chosen.append((episode, tick))
                    added += 1
                    break
            if added >= rule["per_regime"]:
                break
    return chosen


def prediction_world(episode, tick, mode, models):
    saved = copy.deepcopy(episode["frames"][tick]["snapshot"])
    if mode in ("truth", "cv"):
        return saved
    rows = root_inputs(episode, tick)
    if mode == "circle_birth_prior":
        estimates = np.zeros((len(rows), 8))
        for index, row in enumerate(rows):
            estimates[index, :2] = -row["statistics"][:2] - row["context"][:2]
            estimates[index, 2:4] = row["context"][2:4]
    else:
        estimates = predict(models[mode], rows, design_fn=controlled_design)
    indices = {key: index for index, key, _, _ in saved["tracks"]}
    for row, estimate in zip(rows, estimates):
        index = indices[row["key"]]
        human = np.asarray(saved["humans"][index], dtype=float)
        human[5:7] = row["context"][:2] + estimate[:2]
        saved["humans"][index] = human
        saved["preferred"][index] = estimate[2:4] / max(1., float(np.linalg.norm(estimate[2:4])))
    return saved


def decision_task(task):
    episode, tick, mode, continuation, models = task
    cfg = configuration_for_audit()
    saved = prediction_world(episode, tick, mode, models)
    previous = teacher_prefix(cfg, episode, tick, continuation)
    started = time.perf_counter()
    outcomes = [rollout(cfg, episode["geometry"], episode["case"], saved, command, continuation,
                        previous, cv=mode == "cv") for command in actions(cfg)]
    print("SOLVED", episode["regime"], episode["case"], mode, continuation, flush=True)
    return {"regime": episode["regime"], "geometry": episode["geometry"], "case": episode["case"], "tick": tick,
            "mode": mode, "continuation": continuation, "outcomes": outcomes, "seconds": time.perf_counter() - started}


def run_decisions(data, output, workers):
    models = torch.load(output / "models.pt", map_location="cpu", weights_only=False)["models"]
    roots = decision_roots(data)
    selection = [{"regime": ep["regime"], "case": ep["case"], "tick": tick} for ep, tick in roots]
    (output / "decision_selection.json").write_text(json.dumps(selection, indent=2))
    checks = [dict(restore_check(ep, tick), regime=ep["regime"]) for ep, tick in roots]
    tasks = [(ep, tick, mode, continuation, models[ep["regime"]]) for ep, tick in roots
             for continuation in data["protocol"]["continuations"] for mode in data["protocol"]["decision_modes"]]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(decision_task, tasks):
            results.append(result)
    torch.save(results, output / "decision_branches.pt")
    reports = {}
    for regime in data["protocol"]["regimes"]:
        report = summarize([result for result in results if result["regime"] == regime], configuration_for_audit())
        contrasts = {}
        for continuation in data["protocol"]["continuations"]:
            selected = [row for row in report["chosen"] if row["continuation"] == continuation]
            for left, right in data["protocol"]["contrasts"]:
                if not any(row["mode"] == left for row in selected) or not any(row["mode"] == right for row in selected):
                    continue
                lrows = sorted((row for row in selected if row["mode"] == left), key=lambda row: row["case"])
                rrows = sorted((row for row in selected if row["mode"] == right), key=lambda row: row["case"])
                cases = [row["case"] for row in lrows]
                contrasts[continuation + ":" + left + "_minus_" + right] = {
                    "regret": bootstrap_difference(np.asarray([row["regret"] for row in lrows]),
                                                     np.asarray([row["regret"] for row in rrows]), cases,
                                                     data["protocol"]["bootstrap_repetitions"], 1 - .05 / 3),
                    "seconds": bootstrap_difference(np.asarray([row["realized"]["seconds"] for row in lrows]),
                                                      np.asarray([row["realized"]["seconds"] for row in rrows]), cases,
                                                      data["protocol"]["bootstrap_repetitions"], 1 - .05 / 3)}
        report["contrasts"] = contrasts
        reports[regime] = report
    final = {"regimes": reports, "selection": selection, "restore_checks": checks, "protocol": data["protocol"],
             "source_sha256": data["source_sha256"], "branches": sum(len(row["outcomes"]) for row in results)}
    (output / "decision_results.json").write_text(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "probe", "cross", "decisions"))
    parser.add_argument("--output", type=Path, default=Path("outputs/circle_coupling"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check_vendor()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "collect":
        protocol = json.loads(PROTOCOL.read_text())
        tasks = [(regime, split["name"], case) for regime in protocol["regimes"] for split in protocol["splits"]
                 for case in range(split["start"], split["start"] + split["count"])]
        episodes = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for episode in executor.map(collect_case, tasks):
                episodes.append(episode)
                print("COLLECTED", len(episodes), len(tasks), episode["regime"], episode["case"], flush=True)
        data = {"episodes": episodes, "protocol": protocol, "source_sha256": source_hash(),
                "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
        data["intervention_checks"] = intervention_checks(data)
        torch.save(data, args.output / "episodes.pt")
    else:
        data = torch.load(args.output / "episodes.pt", map_location="cpu", weights_only=False)
        if source_hash() != data["source_sha256"]:
            raise RuntimeError("Frozen navigation sources changed")
        if args.command == "probe":
            fit_all(data, args.output)
        elif args.command == "cross":
            cross_all(data, args.output)
        else:
            run_decisions(data, args.output, args.workers)


if __name__ == "__main__":
    main()
