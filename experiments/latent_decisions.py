"""One native root action, then a common legal continuation: Q^pi, not Q*."""

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.latent_information import (check_vendor, configuration_for_audit, context,
                                           history_statistics, own_history, predict)
from experiments.occlusion import source_hash
from shixu.observations import OccludedTracks, ObservedTracks
from shixu.policy import actions
from shixu.runner import environment, orca_teacher


class ConstantHuman:
    def __init__(self, velocity):
        self.velocity = ActionXY(*velocity)

    def predict(self, state):
        return self.velocity


def legal_state(frame):
    active = np.flatnonzero(frame["legal"][:, 7] > 0)
    return ObservedTracks(FullState(*frame["robot"]),
                          [ObservableState(*frame["legal"][key, :5]) for key in active],
                          tuple(active), tuple(bool(frame["legal"][key, 6]) for key in active),
                          tuple(frame["legal"][key, 5] for key in active), len(frame["legal"]))


def root_inputs(episode, tick):
    frame, births = episode["frames"][tick], {}
    for past in episode["frames"][:tick + 1]:
        for key, actor in enumerate(past["legal"]):
            if actor[6] and actor[7] and key not in births:
                births[key] = actor[:2].copy()
    inputs = []
    for key, actor in enumerate(frame["legal"]):
        if actor[7]:
            history = own_history(episode["frames"], tick, key)
            inputs.append({"key": key, "context": context(frame, key), "history": history,
                           "statistics": history_statistics(history, births[key])})
    return inputs


def restored_world(cfg, geometry, case, saved):
    env = environment(cfg, orca_teacher(cfg), geometry, 5)
    env.reset(options={"test_case": case})
    for agent, row in zip([env.robot] + env.humans, [saved["robot"]] + saved["humans"]):
        agent.set(*[row[index] for index in (0, 1, 5, 6, 2, 3, 8)], radius=row[4], v_pref=row[7])
    for human, previous in zip(env.humans, saved["preferred"]):
        human.policy.sim = None
        human.policy._last_pref_vel = None if previous is None else np.asarray(previous).copy()
    env.global_time = saved["time"]
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    observer.tracks = {env.humans[index]: (key, ObservableState(*measurement), stamp)
                       for index, key, measurement, stamp in saved["tracks"]}
    return env, observer


def teacher_prefix(cfg, episode, tick, continuation):
    teacher = orca_teacher(cfg)
    teacher.ttc_brake = continuation != "unbraked_orca"
    for frame in episode["frames"][:tick]:
        teacher.predict(legal_state(frame))
    return None if teacher._last_pref_vel is None else teacher._last_pref_vel.copy()


def estimated_snapshot(episode, tick, mode, models):
    original = episode["frames"][tick]["snapshot"]
    saved = copy.deepcopy(original)
    if mode in ("truth", "cv"):
        return saved
    rows = root_inputs(episode, tick)
    component = mode if mode in ("current_goal_oracle", "current_preferred_oracle") else None
    if mode == "circle_birth_prior":
        estimates = np.zeros((len(rows), 8))
        for index, row in enumerate(rows):
            estimates[index, :2] = -row["statistics"][:2] - row["context"][:2]
            estimates[index, 2:4] = row["context"][2:4]
    else:
        estimates = predict(models["current" if component else mode], rows)
    indices = {key: index for index, key, _, _ in saved["tracks"]}
    for row, estimate in zip(rows, estimates):
        index = indices[row["key"]]
        human = np.asarray(saved["humans"][index], dtype=float)
        human[5:7] = row["context"][:2] + estimate[:2]
        saved["humans"][index] = human
        # All human preferred speeds are bounded by 1 in the frozen generator.
        preferred = estimate[2:4] / max(1., float(np.linalg.norm(estimate[2:4])))
        saved["preferred"][index] = preferred
        if component == "current_goal_oracle":
            saved["humans"][index][5:7] = original["humans"][index][5:7]
        elif component == "current_preferred_oracle":
            saved["preferred"][index] = copy.deepcopy(original["preferred"][index])
    return saved


def rollout(cfg, geometry, case, saved, command, continuation, previous, cv=False):
    env, observer = restored_world(cfg, geometry, case, saved)
    if cv:
        for human in env.humans:
            human.policy = ConstantHuman((human.vx, human.vy))
    teacher = orca_teacher(cfg)
    teacher.ttc_brake = continuation != "unbraked_orca"
    teacher._last_pref_vel = None if previous is None else previous.copy()
    total, minimum, discount, count = 0., float("inf"), 1., 0
    initial_distance = np.linalg.norm(saved["robot"][:2] - saved["robot"][5:7])
    action = command
    while True:
        _, reward, done, truncated, info = env.step(action)
        total += discount * reward
        discount *= .99
        count += 1
        minimum = min(minimum, info["dmin"])
        if done or truncated:
            break
        action = teacher.predict(observer.observe(env))
    distance = float(np.linalg.norm(np.asarray(env.robot.get_position()) - env.robot.get_goal_position()))
    return {"return": float(total), "terminal": info["event"], "minimum_clearance": float(minimum),
            "seconds": count * env.time_step, "progress_m": float(initial_distance - distance)}


def decision_task(task):
    episode, tick, mode, continuation, models = task
    cfg = configuration_for_audit()
    saved = estimated_snapshot(episode, tick, mode, models)
    previous = teacher_prefix(cfg, episode, tick, continuation)
    start = time.perf_counter()
    outcomes = [rollout(cfg, episode["geometry"], episode["case"], saved, command, continuation,
                        previous, cv=mode == "cv") for command in actions(cfg)]
    result = {"case": episode["case"], "geometry": episode["geometry"], "tick": tick,
              "mode": mode, "continuation": continuation, "outcomes": outcomes,
              "seconds": time.perf_counter() - start}
    print("SOLVED", result["geometry"], result["case"], mode, continuation,
          round(result["seconds"], 2), flush=True)
    return result


def select_roots(data):
    roots, excluded = [], []
    for geometry, cases in data["protocol"]["decision_cases"].items():
        for case in cases:
            episode = next(ep for ep in data["episodes"] if ep["geometry"] == geometry and ep["case"] == case)
            ticks = [tick for tick, frame in enumerate(episode["frames"])
                     if frame["snapshot"]["time"] >= 6 and frame["visible"] == 5
                     and np.linalg.norm(frame["robot"][:2] - frame["robot"][5:7]) > 1]
            if ticks:
                roots.append((episode, ticks[0]))
            else:
                excluded.append({"geometry": geometry, "case": case, "reason": "No prespecified eligible root"})
    return roots, excluded


INTERACTION_SELECTION = {"minimum_seconds": 2., "maximum_surface_clearance_m": .8,
                         "roots_per_geometry": 4,
                         "rule": "First qualifying visible root in each case; first four qualifying held-out "
                                 "cases per geometry, sorted by case ID. No outcome/method filtering."}


def select_interaction_roots(data):
    roots, excluded = [], []
    for geometry, split in (("circle", "test"), ("square", "ood")):
        episodes = sorted((ep for ep in data["episodes"] if ep["split"] == split), key=lambda ep: ep["case"])
        added = 0
        for episode in episodes:
            for tick, frame in enumerate(episode["frames"]):
                robot, humans = frame["robot"], frame["legal"]
                if frame["snapshot"]["time"] < INTERACTION_SELECTION["minimum_seconds"] or frame["visible"] != 5:
                    continue
                if np.linalg.norm(robot[:2] - robot[5:7]) <= 1:
                    continue
                clearance = np.min(np.linalg.norm(humans[:, :2] - robot[:2], axis=1) - humans[:, 4] - robot[4])
                if clearance <= INTERACTION_SELECTION["maximum_surface_clearance_m"]:
                    roots.append((episode, tick))
                    added += 1
                    break
            if added >= INTERACTION_SELECTION["roots_per_geometry"]:
                break
        if added < INTERACTION_SELECTION["roots_per_geometry"]:
            excluded.append({"geometry": geometry, "eligible_cases": added, "reason": "Insufficient qualifying cases"})
    return roots, excluded


def restore_check(episode, tick):
    cfg = configuration_for_audit()
    env, _ = restored_world(cfg, episode["geometry"], episode["case"], episode["frames"][tick]["snapshot"])
    result = env.step(ActionXY(*episode["actions"][tick]))
    future = episode["frames"][tick + 1]["snapshot"]
    positions = [env.robot.get_full_state().to_array()] + [human.get_full_state().to_array() for human in env.humans]
    target = [future["robot"]] + future["humans"]
    error = float(np.max(np.abs(np.asarray(positions) - np.asarray(target))))
    if error > 5e-6 or abs(result[1] - episode["rewards"][tick]) > 5e-6:
        raise RuntimeError(f"Restore parity failed for {episode['case']}: {error}")
    return {"case": episode["case"], "tick": tick, "maximum_state_error": error,
            "reward_error": float(abs(result[1] - episode["rewards"][tick]))}


def summarize(results, cfg):
    groups = {}
    for row in results:
        groups.setdefault((row["geometry"], row["case"], row["continuation"]), {})[row["mode"]] = row
    chosen, velocities = [], np.asarray(actions(cfg))
    for (geometry, case, continuation), modes in groups.items():
        reference = np.asarray([item["return"] for item in modes["truth"]["outcomes"]])
        for mode, row in modes.items():
            predicted = np.asarray([item["return"] for item in row["outcomes"]])
            index = int(np.argmax(predicted))
            realized = modes["truth"]["outcomes"][index]
            chosen.append({"geometry": geometry, "case": case, "continuation": continuation, "mode": mode,
                           "action_index": index, "action": velocities[index].tolist(),
                           "predicted_return": float(predicted[index]), "realized": realized,
                           "regret": float(reference.max() - reference[index]),
                           "reference_rank": int(1 + np.sum(reference > reference[index] + 1e-8)),
                           "reference_optimum_ties": int(np.sum(np.abs(reference - reference.max()) <= 1e-8)),
                           "reference_successful_actions": sum(item["terminal"] == "reach_goal"
                                                               for item in modes["truth"]["outcomes"]),
                           "rollout_compute_seconds": row["seconds"]})
    aggregate = []
    for continuation in sorted({row["continuation"] for row in chosen}):
        for geometry in sorted({row["geometry"] for row in chosen}):
            for mode in sorted({row["mode"] for row in chosen}):
                rows = [row for row in chosen if row["continuation"] == continuation
                        and row["geometry"] == geometry and row["mode"] == mode]
                aggregate.append({"continuation": continuation, "geometry": geometry, "mode": mode,
                                  "roots": len(rows), "successes": sum(row["realized"]["terminal"] == "reach_goal" for row in rows),
                                  "collisions": sum(row["realized"]["terminal"] == "collision" for row in rows),
                                  "timeouts": sum(row["realized"]["terminal"] == "timeout" for row in rows),
                                  "mean_regret": float(np.mean([row["regret"] for row in rows])),
                                  "mean_return": float(np.mean([row["realized"]["return"] for row in rows])),
                                  "mean_seconds": float(np.mean([row["realized"]["seconds"] for row in rows])),
                                  "mean_minimum_clearance": float(np.mean([row["realized"]["minimum_clearance"] for row in rows]))})
    return {"chosen": chosen, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/latent_information"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cohort", choices=("primary", "interaction"), default="primary")
    parser.add_argument("--components", action="store_true", help="Post-hoc offline single-latent oracle controls")
    parser.add_argument("--birth-prior", action="store_true", help="Post-hoc legal circle-generator baseline; also test OOD")
    args = parser.parse_args()
    if args.components and args.birth_prior:
        parser.error("Run component oracles and the legal birth prior separately")
    check_vendor()
    torch.set_num_threads(1)
    data = torch.load(args.output / "episodes.pt", map_location="cpu", weights_only=False)
    models = torch.load(args.output / "probes.pt", map_location="cpu", weights_only=False)["models"]
    if source_hash() != data["source_sha256"]:
        raise RuntimeError("Frozen parent changed since collection")
    roots, excluded = select_roots(data) if args.cohort == "primary" else select_interaction_roots(data)
    prefix = "" if args.cohort == "primary" else "interaction_"
    selection = {"cohort": args.cohort, "roots": [{"case": ep["case"], "geometry": ep["geometry"], "tick": tick}
                                                 for ep, tick in roots], "excluded": excluded}
    if args.cohort == "interaction":
        selection["rule"] = INTERACTION_SELECTION
    if args.birth_prior:
        selection["post_hoc_component_controls"] = ["circle_birth_prior"]
        output_prefix = prefix + "birth_"
    elif args.components:
        selection["post_hoc_component_controls"] = ["current_goal_oracle", "current_preferred_oracle"]
        output_prefix = prefix + "components_"
    else:
        output_prefix = prefix
    (args.output / (output_prefix + "decision_selection.json")).write_text(json.dumps(selection, indent=2))
    checks = [restore_check(episode, tick) for episode, tick in roots]
    print("RESTORE_PARITY", checks, flush=True)
    modes = selection.get("post_hoc_component_controls", data["protocol"]["decision_modes"])
    tasks = [(episode, tick, mode, continuation, models) for episode, tick in roots
             for continuation in data["protocol"]["continuations"] for mode in modes]
    existing = prefix + ("components_decision_branches.pt" if args.birth_prior else "decision_branches.pt")
    results = (torch.load(args.output / existing, map_location="cpu", weights_only=False)
               if args.components or args.birth_prior else [])
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(decision_task, tasks):
            results.append(result)
            torch.save(results, args.output / (output_prefix + "decision_branches.pt"))
    report = summarize(results, configuration_for_audit())
    report.update(protocol=data["protocol"], source_sha256=source_hash(), restore_checks=checks,
                  selection=selection, excluded=excluded, root_count=len(roots),
                  branches=sum(len(row["outcomes"]) for row in results),
                  scope="Small visible-root diagnostic, one native root step then common legal ORCA continuation. "
                        "Estimated hidden goals/preferred velocities are frozen world hypotheses, not a learned navigation "
                        "policy or a posterior planner. Native returns to terminal define Q^pi, not Q*. Predicted rewards "
                        "choose the action; truth-world rewards only evaluate it. Alternate continuation tests sensitivity.")
    (args.output / (output_prefix + "decision_results.json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
