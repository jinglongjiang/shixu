"""Replay archived timeout controls; never change a trained controller."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.occlusion import configuration, evaluate_weights
from shixu.model import build_model
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


def action_availability(policy, state):
    queries, _, clearances = policy.aligned_candidates(state)
    robots = queries[:, 0, :9]
    current = state.self_state
    distance = np.hypot(current.px - current.gx, current.py - current.gy)
    progress = distance - np.linalg.norm(robots[:, :2] - robots[:, 5:7], axis=1)
    safe = clearances >= policy.config.getfloat("eval_protocol", "safety_margin")
    near_hidden = sum(not measured and np.hypot(h.px - current.px, h.py - current.py) < 2
                      for h, measured in zip(state.human_states, state.observed))
    return {"distance": float(distance), "safe_actions": int(safe.sum()),
            "safe_progress_actions": int((safe & (progress >= .02)).sum()),
            "hidden": int(sum(not flag for flag in state.observed)), "near_hidden": int(near_hidden)}


def probe(root, arms, seeds):
    records, manifests = [], []
    for arm in arms:
        for seed in seeds:
            source = root / str(seed) / arm / "result.json"
            archive = json.loads(source.read_text())
            manifests.append({"arm": arm, "seed": seed,
                              "result_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            cfg = configuration(archive["protocol"], arm)
            policy = ValuePolicy(build_model(cfg), cfg)
            for row in archive["episodes"]:
                if row["terminal"] != "timeout":
                    continue
                env = environment(cfg, policy, row["geometry"], row["people"])
                env.reset(options={"test_case": row["case"]})
                observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
                trace, path, previous = [], 0., np.asarray(env.robot.get_position())
                for command in row["actions"]:
                    state = observer.observe(env)
                    trace.append(dict(action_availability(policy, state),
                                      time=float(env.global_time), path=path,
                                      speed=float(np.linalg.norm(command))))
                    _, _, done, truncated, info = env.step(ActionXY(*command))
                    position = np.asarray(env.robot.get_position())
                    path += float(np.linalg.norm(position - previous))
                    previous = position
                    if done or truncated:
                        break
                if info["event"] != row["terminal"] or len(trace) != len(row["actions"]):
                    raise RuntimeError("Archived timeout terminal/trajectory could not be reproduced")
                if not np.isclose(path, row["path"], rtol=1e-6, atol=1e-6):
                    raise RuntimeError("Archived executed path changed during replay")
                end_distance = float(np.linalg.norm(position - [env.robot.gx, env.robot.gy]))
                tail = [tick for tick in trace if tick["time"] >= env.global_time - 10]
                records.append({"arm": arm, "seed": seed, "case": row["case"],
                                "people": row["people"], "geometry": row["geometry"],
                                "distance_to_goal": end_distance,
                                "last_10s_progress": tail[0]["distance"] - end_distance,
                                "last_10s_path": path - tail[0]["path"],
                                "last_10s_low_speed_fraction": float(np.mean([t["speed"] < .1 for t in tail])),
                                "last_10s_safe_progress_fraction": float(np.mean([t["safe_progress_actions"] > 0 for t in tail])),
                                "last_10s_no_safe_action_fraction": float(np.mean([not t["safe_actions"] for t in tail])),
                                "last_10s_near_hidden_fraction": float(np.mean([t["near_hidden"] > 0 for t in tail])),
                                "trace": trace})
    summaries = {}
    for arm in arms:
        rows = [row for row in records if row["arm"] == arm]
        summaries[arm] = {"timeouts": len(rows)}
        for key in ("distance_to_goal", "last_10s_progress", "last_10s_path",
                    "last_10s_low_speed_fraction", "last_10s_safe_progress_fraction",
                    "last_10s_no_safe_action_fraction", "last_10s_near_hidden_fraction"):
            values = np.asarray([row[key] for row in rows])
            summaries[arm][key] = {"mean": float(values.mean()), "median": float(np.median(values))} if len(values) else None
    return {"summaries": summaries, "manifests": manifests, "records": records,
            "scope": "All archived timeouts in the pre-fixed arm/seed cohort, not a method evaluation. "
                     "Replay executed controls without loading or changing policy weights. Safe means the inherited "
                     "one-step CV endpoint clearance threshold; progress means at least .02 m in that same step. "
                     "These are not guaranteed safe future trajectories. Different methods visit different states; "
                     "timeout-only summaries cannot establish causal effects of occlusion or memory."}


def behavior_probe(root, arms, seeds):
    """Fixed training-final epsilon; no tuning or formal-evaluator replacement."""
    original = ValuePolicy.predict
    records = []
    for arm in arms:
        for seed in seeds:
            directory = root / str(seed) / arm
            archived = json.loads((directory / "result.json").read_text())
            protocol = archived["protocol"]
            cfg = configuration(protocol, arm)
            fixed_epsilon = cfg.getfloat("sarl", "epsilon_end")

            def predict(policy, state, epsilon=0.):
                return original(policy, state, fixed_epsilon)

            with patch.object(ValuePolicy, "predict", predict):
                result = evaluate_weights(directory / "model.pt", protocol, "cpu",
                                          protocol["development_cases"][:2], seed)
            result.update(arm=arm, seed=seed, epsilon=fixed_epsilon)
            records.append(result)
    return {"records": records,
            "scope": "Frozen final weights, first two pre-fixed development cases per cell. Only evaluation "
                     "epsilon changes to the already fixed training-final value; reward, smoothing, inputs and "
                     "action support remain unchanged. Random choices can bypass the inherited greedy safety "
                     "mask, so both collision and timeout must be reported. This checks exploration-assisted "
                     "escape, not equivalence with all training semantics or a deployable new method. "
                     "Formal greedy results and the active training protocol are unchanged."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[419, 443, 467, 491])
    parser.add_argument("--output", required=True)
    parser.add_argument("--behavior-shadow", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = (behavior_probe(Path(args.root), args.arms, args.seeds) if args.behavior_shadow
              else probe(Path(args.root), args.arms, args.seeds))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    summary = (result["summaries"] if "summaries" in result else
               [{"arm": row["arm"], "seed": row["seed"], "summary": row["summary"]}
                for row in result["records"]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
