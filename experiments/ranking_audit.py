"""Read-only failure ranking audit; the future reference is not an optimal Q."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.query_contract_probe import transitions
from shixu.features import encode_tracks, stack_histories, window
from shixu.model import build_model, load_weights
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    policy = ValuePolicy(build_model(cfg), cfg)
    load_weights(policy.model, path, "cpu")
    policy.model.eval()
    return policy, hashlib.sha256(path.read_bytes()).hexdigest()


def future_reference(state, velocities, humans, dt):
    """Exact segment distances on archived, robot-independent human motion."""
    robot = state.self_state
    ticks = np.arange(len(humans)) * dt
    positions = np.array([robot.px, robot.py])[None, None] + velocities[:, None] * ticks[None, :, None]
    relative = positions[:, :, None] - humans[None, :, :, :2]
    start, delta = relative[:, :-1], np.diff(relative, axis=1)
    fractions = np.clip(-(start * delta).sum(-1) / np.maximum((delta ** 2).sum(-1), 1e-12), 0, 1)
    distance = np.linalg.norm(start + fractions[..., None] * delta, axis=-1)
    clearance = (distance - robot.radius - humans[None, :-1, :, 2]).min(axis=(1, 2))
    old_distance = np.hypot(robot.px - robot.gx, robot.py - robot.gy)
    progress = old_distance - np.linalg.norm(positions[:, -1] - [robot.gx, robot.gy], axis=-1)
    return clearance, progress


def best_rank(scores, eligible):
    if not np.any(eligible):
        return None
    order = np.argsort(-scores, kind="stable")
    return int(np.flatnonzero(eligible[order])[0] + 1)


@torch.inference_mode()
def stages(policy, state, prefix, previous, truth):
    queries, rewards, clearances = policy.aligned_candidates(state)
    values = policy.model.score_candidates(torch.as_tensor(prefix[None]),
                                           torch.as_tensor(queries[None])).squeeze(0)
    scores = torch.as_tensor(rewards) + policy.gamma * values
    raw = scores.clone()
    raw[0] -= 1e-3
    margin = policy.config.getfloat("eval_protocol", "safety_margin")
    risk = policy.config.getfloat("eval_protocol", "risk_lambda")
    distance = torch.as_tensor(clearances)
    if margin > 0 and bool((distance >= margin).any()):
        scores = scores.masked_fill(distance < margin, -1e9)
    if risk > 0:
        threshold = margin if margin > 0 else policy.config.getfloat("reward", "discomfort_dist")
        scores -= risk * torch.clamp(threshold - distance, min=0)
    scores[0] -= 1e-3
    raw, filtered = raw.numpy(), scores.numpy()
    policy.history.clear()
    policy.history.extend(prefix[:-1])
    np.testing.assert_allclose(filtered, policy.score(state), rtol=1e-6, atol=1e-6)
    index = int(np.argmax(filtered))
    velocities = np.asarray(policy.action_space)
    selected = velocities[index]
    alpha = policy.config.getfloat("eval_protocol", "action_smoothing")
    executed = selected if previous is None else alpha * np.asarray(previous) + (1 - alpha) * selected
    future_clearance, future_progress = future_reference(state, velocities, truth, policy.time_step)
    smooth_clearance, smooth_progress = future_reference(state, executed[None], truth, policy.time_step)
    robot = state.self_state
    progress = np.hypot(robot.px - robot.gx, robot.py - robot.gy) - np.linalg.norm(
        queries[:, 0, :2] - queries[:, 0, 5:7], axis=-1)
    eligibility = {"cv_safe_progress": (clearances >= margin) & (progress >= .02),
                   "true_2s_margin_progress": (future_clearance >= margin) & (future_progress >= .16),
                   "true_2s_collisionfree_progress": (future_clearance >= 0) & (future_progress >= .16)}
    smooth_position = np.asarray([robot.px, robot.py]) + policy.time_step * executed
    smooth_cv_clearance = min((np.linalg.norm(smooth_position - [h.px + policy.time_step * h.vx,
                                                                h.py + policy.time_step * h.vy])
                               - robot.radius - h.radius for h in state.human_states), default=float("inf"))
    smooth_cv_progress = np.hypot(robot.px - robot.gx, robot.py - robot.gy) - np.linalg.norm(
        smooth_position - [robot.gx, robot.gy])
    original = policy.model.use_history
    try:
        policy.model.use_history = False
        zero_values = policy.model.score_candidates(torch.as_tensor(prefix[None]),
                                                    torch.as_tensor(queries[None])).squeeze(0).numpy()
    finally:
        policy.model.use_history = original
    zero_raw = rewards + policy.gamma * zero_values
    zero_raw[0] -= 1e-3
    result = {"raw_top": int(np.argmax(raw)), "filtered_top": index,
              "filter_changed_top": bool(index != np.argmax(raw)),
              "executed": executed.tolist(), "raw_scores": raw.tolist(),
              "filtered_scores": filtered.tolist(), "values": values.tolist(),
              "zero_history_top_changed": bool(np.argmax(zero_raw) != np.argmax(raw)),
              "zero_history_mean_abs_value_change": float(np.abs(zero_values - values.numpy()).mean()),
              "selected_true_2s_clearance": float(future_clearance[index]),
              "smoothed_true_2s_clearance": float(smooth_clearance[0]),
              "selected_true_2s_progress": float(future_progress[index]),
              "smoothed_true_2s_progress": float(smooth_progress[0])}
    result["smoothed_cv_safe_progress"] = bool(smooth_cv_clearance >= margin and smooth_cv_progress >= .02)
    for name, mask in eligibility.items():
        result[name] = {"available": bool(mask.any()), "count": int(mask.sum()),
                        "raw_best_rank": best_rank(raw, mask), "filtered_best_rank": best_rank(filtered, mask),
                        "raw_top_eligible": bool(mask[np.argmax(raw)]), "filtered_top_eligible": bool(mask[index]),
                        "value_best_rank": best_rank(values.numpy(), mask),
                        "value_top_eligible": bool(mask[int(values.argmax())])}
    return result


def fixed_failures(root, seeds):
    selected = []
    for seed in seeds:
        for arm in ("motion_gru", "motion_kda"):
            archive = json.loads((root / str(seed) / arm / "result.json").read_text())
            for terminal in ("timeout", "collision"):
                candidates = [row for row in archive["episodes"] if row["people"] in (10, 20) and row["terminal"] == terminal]
                candidates.sort(key=lambda row: (row["people"], row["geometry"], row["case"]))
                if candidates:
                    selected.append(dict(candidates[0], seed=seed, source_arm=arm))
    return selected


def anchors(root, selected):
    records = []
    for row in selected:
        policy, _ = load_policy(root / str(row["seed"]) / row["source_arm"] / "model.pt")
        cfg = policy.config
        if cfg.getboolean("robot", "visible"):
            raise ValueError("Archived human futures are valid only for robot-independent dynamics")
        env = environment(cfg, policy, row["geometry"], row["people"])
        env.reset(options={"test_case": row["case"]})
        observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
        frames, states, truths, path = [], [], [], 0.
        for command in row["actions"]:
            state = observer.observe(env)
            states.append(state)
            frames.append(encode_tracks(state))
            truths.append(np.asarray([[h.px, h.py, h.radius] for h in env.humans]))
            old = np.asarray(env.robot.get_position())
            _, _, done, truncated, info = env.step(ActionXY(*command))
            path += float(np.linalg.norm(np.asarray(env.robot.get_position()) - old))
            if done or truncated:
                break
        truths.append(np.asarray([[h.px, h.py, h.radius] for h in env.humans]))
        if info["event"] != row["terminal"] or len(states) != len(row["actions"]):
            raise RuntimeError("Archived failure did not reproduce")
        np.testing.assert_allclose(path, row["path"], rtol=1e-6, atol=1e-6)
        dt, length = policy.time_step, policy.length
        for seconds_before in (10, 8, 6, 4, 2):
            tick = len(states) - round(seconds_before / dt)
            if tick < 0:
                continue
            future = np.asarray(truths[tick:tick + round(2 / dt) + 1])
            if len(future) != round(2 / dt) + 1:
                raise RuntimeError("Incomplete fixed two-second reference")
            records.append({"seed": row["seed"], "source_arm": row["source_arm"], "case": row["case"],
                            "terminal": row["terminal"], "people": row["people"], "geometry": row["geometry"],
                            "seconds_before": seconds_before, "time": tick * dt, "state": states[tick],
                            "prefix": window(frames[:tick + 1], length, "zero")[1:], "truth": future,
                            "previous": row["actions"][tick - 1] if tick else None,
                            "archived_action": row["actions"][tick]})
    return records


def aggregate(records):
    groups = {}
    for phase in ("il50", "rl1000", "rl3000"):
        for arm in ("motion_gru", "motion_kda"):
            for terminal in ("timeout", "collision"):
                rows = [row["models"][phase][arm] for row in records if row["terminal"] == terminal]
                summary = {"states": len(rows), "filter_changed_top": sum(row["filter_changed_top"] for row in rows),
                           "zero_history_top_changed": sum(row["zero_history_top_changed"] for row in rows),
                           "smoothed_cv_safe_progress": sum(row["smoothed_cv_safe_progress"] for row in rows)}
                for name in ("cv_safe_progress", "true_2s_margin_progress", "true_2s_collisionfree_progress"):
                    items = [row[name] for row in rows if row[name]["available"]]
                    summary[name] = {"available_states": len(items),
                                     "raw_top_eligible": sum(row["raw_top_eligible"] for row in items),
                                     "filtered_top_eligible": sum(row["filtered_top_eligible"] for row in items),
                                     "value_top_eligible": sum(row["value_top_eligible"] for row in items),
                                     "value_best_rank_median": float(np.median([row["value_best_rank"] for row in items])) if items else None,
                                     "raw_best_rank_median": float(np.median([row["raw_best_rank"] for row in items])) if items else None,
                                     "filtered_best_rank_median": float(np.median([row["filtered_best_rank"] for row in items])) if items else None}
                groups[f"{phase}/{arm}/{terminal}"] = summary
    return groups


@torch.inference_mode()
def audit(root, old_root, expert_path, seeds):
    selected = fixed_failures(root, seeds)
    frozen = anchors(root, selected)
    data = torch.load(expert_path, map_location="cpu", weights_only=False)
    _, (samples, _, _), targets, skipped = transitions(data, 24, .25)
    manifests, reference_errors = {}, []
    policies = {}
    for seed in seeds:
        policies[seed] = {}
        for phase, source, filename in (("il50", root, "il.pt"), ("rl1000", old_root, "model.pt"), ("rl3000", root, "model.pt")):
            policies[seed][phase] = {}
            for arm in ("motion_gru", "motion_kda"):
                path = source / str(seed) / arm / filename
                policy, sha = load_policy(path)
                policies[seed][phase][arm] = policy
                manifests[str(path)] = sha
                outputs = []
                for start in range(0, len(samples), 32):
                    values = policy.model(torch.as_tensor(stack_histories(samples[start:start + 32])))
                    outputs.extend(values.tolist())
                difference = np.asarray(outputs) - targets
                reference_errors.append({"seed": seed, "phase": phase, "arm": arm, "states": len(targets),
                                         "mse_against_expert_returns": float(np.mean(difference ** 2)),
                                         "bias_against_expert_returns": float(difference.mean())})
    records = []
    for anchor in frozen:
        row = {key: value for key, value in anchor.items() if key not in ("state", "prefix", "truth", "previous")}
        row["models"] = {}
        for phase, controls in policies[anchor["seed"]].items():
            row["models"][phase] = {arm: stages(policy, anchor["state"], anchor["prefix"], anchor["previous"], anchor["truth"])
                                     for arm, policy in controls.items()}
        np.testing.assert_allclose(row["models"]["rl3000"][row["source_arm"]]["executed"],
                                   row["archived_action"], rtol=1e-6, atol=1e-6)
        records.append(row)
    return {"selected_failures": selected, "records": records, "summary": aggregate(records),
            "checkpoint_sha256": manifests, "expert_reference_errors": reference_errors,
            "expert_data_sha256": hashlib.sha256(expert_path.read_bytes()).hexdigest(),
            "expert_support_change_skips": skipped,
            "scope": "First primary-density collision and timeout per source arm/seed, archive-key order fixed before "
                     "model scoring; five fixed anchors 10/8/6/4/2 seconds before each failure. Both models and all phases "
                     "use identical legal states, prefixes, 80 candidates and independently replayed true human futures. "
                     "Two-second references hold candidate velocity constant against robot-independent piecewise-linear "
                     "human trajectories; they are not optimal navigation Q or a completed closed-loop intervention. "
                     "Zero-history changes are out-of-distribution sensitivity tests. RL expert-return discrepancies "
                     "are not true on-policy value MSE, and failure-conditioned retrospective anchors are not a learning curve."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--expert-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[419, 443, 467, 491])
    args = parser.parse_args()
    from crowd_sim.envs import crowd_sim
    expected = Path(__file__).resolve().parents[1] / "vendor/crowd_sim/envs/crowd_sim.py"
    if Path(crowd_sim.__file__).resolve() != expected:
        raise RuntimeError("Run with PYTHONPATH=vendor:. to use the frozen simulator, not a global legacy install")
    torch.set_num_threads(1)
    result = audit(args.root, args.old_root, args.expert_data, args.seeds)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": result["summary"], "expert_reference_errors": result["expert_reference_errors"]}, indent=2))


if __name__ == "__main__":
    main()
