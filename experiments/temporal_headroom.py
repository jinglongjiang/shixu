"""Natural legal-history forecast/action diagnostic; no parameter updates."""

import argparse
import configparser
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
from shixu.features import encode_state, window
from shixu.model import ValueModel, load_weights
from shixu.policy import ValuePolicy
from shixu.runner import environment, orca_teacher, run_episode


def state(record):
    return JointState(FullState(*record["robot"]),
                      [ObservableState(*human["state"]) for human in record["humans"]])


def human_path(episode, tick, horizon, mode, dt):
    current = np.array([h["state"] for h in episode["observations"][tick]["humans"]])
    if mode == "oracle":
        return np.array([[h["state"] for h in observation["humans"]]
                         for observation in episode["observations"][tick:tick + horizon + 1]])
    result = np.repeat(current[None], horizon + 1, axis=0)
    velocity = current[:, 2:4].copy()
    acceleration = np.zeros_like(velocity)
    if mode == "history" and tick >= 3:
        previous = np.array([h["state"] for h in episode["observations"][tick - 3]["humans"]])
        acceleration = np.clip((velocity - previous[:, 2:4]) / (3 * dt), -1, 1)
    for step in range(1, horizon + 1):
        result[step, :, :2] = result[step - 1, :, :2] + dt * velocity
        velocity = velocity + dt * acceleration
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity *= np.minimum(1, 1.5 / np.maximum(speed, 1e-8))
        result[step, :, 2:4] = velocity
    return result


@torch.inference_mode()
def score_paths(policy, current, past, path):
    scores, sequences, terminals, clearance, progress = [], [], [], [], []
    for action in policy.action_space:
        robot = current.self_state
        sequence, total, done, minimum = list(past) + [encode_state(current)], 0., False, float("inf")
        previous = current
        for step in range(1, len(path)):
            next_robot = FullState(robot.px + action.vx * policy.time_step, robot.py + action.vy * policy.time_step,
                                   action.vx, action.vy, robot.radius, robot.gx, robot.gy, robot.v_pref, robot.theta)
            future = JointState(next_robot, [ObservableState(*human) for human in path[step]])
            # The parent environment checks swept relative motion using current human velocity.
            for human in previous.human_states:
                relative = np.array([human.px - robot.px, human.py - robot.py])
                delta = policy.time_step * np.array([human.vx - action.vx, human.vy - action.vy])
                fraction = np.clip(-relative @ delta / max(delta @ delta, 1e-12), 0, 1)
                minimum = min(minimum, float(np.linalg.norm(relative + fraction * delta) - human.radius - robot.radius))
            reward, _ = policy.immediate_reward(previous, future, action)
            if minimum < 0:
                reward = policy.config.getfloat("reward", "collision_penalty")
                done = True
            elif np.hypot(next_robot.px - robot.gx, next_robot.py - robot.gy) < robot.radius:
                reward = policy.config.getfloat("reward", "success_reward")
                done = True
            total += policy.gamma ** (step - 1) * reward
            sequence.append(encode_state(future))
            robot, previous = next_robot, future
            if done:
                break
        scores.append(total)
        sequences.append(window(sequence, policy.length))
        terminals.append(done)
        clearance.append(minimum)
        progress.append(np.hypot(current.self_state.px - robot.gx, current.self_state.py - robot.gy)
                        - np.hypot(robot.px - robot.gx, robot.py - robot.gy))
    values = policy.model(torch.as_tensor(np.asarray(sequences), device=policy.device)).cpu().numpy()
    scores = np.asarray(scores) + policy.gamma ** (len(path) - 1) * values * ~np.asarray(terminals)
    return scores, np.asarray(clearance), np.asarray(progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(Path(__file__).with_name("temporal_protocol.json").read_text())
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
    model = ValueModel("mamba").cuda().eval()
    load_weights(model, args.weights, "cuda")
    policy = ValuePolicy(model, cfg, "cuda")
    rows, episodes = [], []
    horizon, dt = protocol["headroom_horizon"], policy.time_step
    for geometry in protocol["geometries"]:
        for people in protocol["people"]:
            env = environment(cfg, policy, geometry, people)
            teacher = orca_teacher(cfg)
            for case in protocol["headroom_cases"]:
                episode = run_episode(env, policy, case, teacher=teacher)
                episodes.append({"geometry": geometry, "people": people, "case": case,
                                 "terminal": episode["terminal"], "frames": len(episode["tokens"])})
                for tick in range(12, len(episode["tokens"]) - horizon, protocol["headroom_stride"]):
                    current = state(episode["observations"][tick])
                    past = episode["tokens"][max(0, tick - 24):tick]
                    paths = {mode: human_path(episode, tick, horizon, mode, dt)
                             for mode in ("cv", "history", "oracle")}
                    evaluated = {mode: score_paths(policy, current, past, path) for mode, path in paths.items()}
                    indices = {mode: int(np.argmax(result[0])) for mode, result in evaluated.items()}
                    truth = evaluated["oracle"]
                    row = {"geometry": geometry, "people": people, "case": case, "tick": tick,
                           "actions": indices, "oracle_regret": {mode: float(truth[0].max() - truth[0][index])
                                                                  for mode, index in indices.items()},
                           "clearance": {mode: float(truth[1][index]) for mode, index in indices.items()},
                           "progress": {mode: float(truth[2][index]) for mode, index in indices.items()},
                           "history_ade": float(np.linalg.norm(paths["history"][1:, :, :2] - paths["oracle"][1:, :, :2], axis=-1).mean()),
                           "cv_ade": float(np.linalg.norm(paths["cv"][1:, :, :2] - paths["oracle"][1:, :, :2], axis=-1).mean())}
                    rows.append(row)
    improved = [row for row in rows if row["oracle_regret"]["cv"] - row["oracle_regret"]["history"]
                >= protocol["headroom_value_gap"] and row["clearance"]["history"] >= 0]
    summary = {"native_episodes": len(episodes), "uniform_states": len(rows),
               "oracle_changes_action": sum(row["actions"]["oracle"] != row["actions"]["cv"] for row in rows),
               "legal_history_changes_action": sum(row["actions"]["history"] != row["actions"]["cv"] for row in rows),
               "safe_history_value_improvements": len(improved),
               "gate": "HISTORY_ACTION_SIGNAL" if len(improved) >= protocol["headroom_min_states"] else "HEADROOM_NOT_ESTABLISHED",
               "scope": "Offline 3-second held-action shadow using common parent-style reward/value bootstrap and swept safety checks; not exact simulator reward replay or closed-loop policy superiority"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": protocol, "summary": summary, "episodes": episodes, "states": rows}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
