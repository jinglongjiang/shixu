"""Natural contextual evidence during occlusion; no model training or scene changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.occlusion import configuration
from shixu.features import encode_tracks
from shixu.model import build_model
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


def audit(data_path):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    cfg = configuration(data["protocol"], "current")
    policy = ValuePolicy(build_model(cfg), cfg)
    env = environment(cfg, policy, data["protocol"]["training_geometry"], data["protocol"]["training_people"])
    counts = {key: 0 for key in ("frames", "population_frames", "hidden_frames", "retained_hidden_frames",
                                "hidden_with_measured_neighbour", "hidden_with_changed_neighbour",
                                "changed_neighbour_near_robot_2m", "known_expired_frames",
                                "known_expired_inside_prefix", "known_expired_inside_prefix_near_2m")}
    errors = {key: [] for key in ("all_retained", "measured_neighbour", "changed_neighbour", "expired_inside_prefix")}
    prefix_seconds = (data["protocol"]["history"] - 2) * cfg.getfloat("env", "time_step")
    records = []
    for ep in data["episodes"]:
        env.reset(options={"test_case": ep["case"]})
        observer, velocities = OccludedTracks(data["protocol"]["retention_seconds"]), {}
        for tick, (frame, command) in enumerate(zip(ep["tokens"], ep["actions"])):
            state = observer.observe(env)
            np.testing.assert_allclose(encode_tracks(state), frame, rtol=1e-5, atol=2e-5)
            truths = {row[0]: human for human, row in observer.tracks.items()}
            fresh = {}
            for key, human, measured in zip(state.track_ids, state.human_states, state.observed):
                if measured:
                    velocity = np.array([human.vx, human.vy])
                    fresh[key] = float(np.linalg.norm(velocity - velocities[key])) if key in velocities else 0.
                    velocities[key] = velocity
            counts["frames"] += 1
            counts["population_frames"] += len(env.humans)
            counts["hidden_frames"] += len(env.humans) - sum(state.observed)
            for human, (_, measurement, stamp) in observer.tracks.items():
                age = env.global_time - stamp
                if age > observer.retention_seconds:
                    counts["known_expired_frames"] += 1
                    if age <= prefix_seconds:
                        counts["known_expired_inside_prefix"] += 1
                        counts["known_expired_inside_prefix_near_2m"] += int(
                            np.hypot(human.px - env.robot.px, human.py - env.robot.py) < 2)
                        errors["expired_inside_prefix"].append(float(np.hypot(
                            measurement.px + age * measurement.vx - human.px,
                            measurement.py + age * measurement.vy - human.py)))
            for key, human, measured, age in zip(state.track_ids, state.human_states, state.observed, state.ages):
                if measured:
                    continue
                counts["retained_hidden_frames"] += 1
                neighbours = [j for j, other, arrived in zip(state.track_ids, state.human_states, state.observed)
                              if arrived and np.hypot(human.px - other.px, human.py - other.py) < 2.]
                changed = [j for j in neighbours if fresh[j] > .1]
                truth = truths[key]
                error = float(np.hypot(human.px - truth.px, human.py - truth.py))
                errors["all_retained"].append(error)
                if neighbours:
                    counts["hidden_with_measured_neighbour"] += 1
                    errors["measured_neighbour"].append(error)
                if changed:
                    counts["hidden_with_changed_neighbour"] += 1
                    errors["changed_neighbour"].append(error)
                    near = np.hypot(human.px - env.robot.px, human.py - env.robot.py) < 2.
                    counts["changed_neighbour_near_robot_2m"] += int(near)
                    records.append({"case": ep["case"], "tick": tick, "target": key,
                                    "age": age, "changed_neighbours": changed, "near_robot": bool(near),
                                    "hidden_position_error_m": error})
            env.step(ActionXY(*command))
    return {"counts": counts,
            "error_distributions": {name: {"count": len(rows), "mean_m": float(np.mean(rows)) if rows else None,
                                           "p90_m": float(np.quantile(rows, .9)) if rows else None,
                                           "above_005_fraction": float(np.mean(np.asarray(rows) > .05)) if rows else None}
                                    for name, rows in errors.items()},
            "thresholds": {"neighbour_distance_m": 2., "arrived_velocity_change_mps": .1,
                           "root_prefix_span_seconds": prefix_seconds},
            "records": records,
            "scope": "Every frame of the 128 saved successful legal-observation ORCA demonstrations; no event "
                     "selection for the denominator. Neighbour events use only arrived observations and legal CV "
                     "track locations. Hidden ground truth quantifies diagnostic error only. A context event is "
                     "not proof of useful hidden-state prediction or improved action; other actors' memories may "
                     "already preserve its information."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
