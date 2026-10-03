"""Natural-event history interventions; offline headroom, never a trained method."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from shixu.model import OrderedValueModel, load_weights
from shixu.observations import TrackKeys
from shixu.policy import ValuePolicy
from shixu.runner import environment, run_episode
from experiments.temporal_order import configuration, PROTOCOL


def observed_event(record, tick):
    if tick < 12:
        return None
    robot = np.array(record["observations"][tick]["robot"][:2])
    for key in range(5):
        velocities = np.array([observation["humans"][key]["state"][2:4]
                               for observation in record["observations"][tick - 8:tick + 1]])
        old, recent = velocities[:5].mean(0), velocities[-3:].mean(0)
        old_speed, recent_speed = np.linalg.norm(old), np.linalg.norm(recent)
        angle = np.arccos(np.clip(old @ recent / max(old_speed * recent_speed, 1e-8), -1, 1))
        turn = min(old_speed, recent_speed) >= .15 and angle >= np.pi / 6
        slowdown = old_speed >= .4 and recent_speed <= .6 * old_speed
        position = np.array(record["observations"][tick]["humans"][key]["state"][:2])
        if (turn or slowdown) and np.linalg.norm(position - robot) <= 2:
            return key
    return None


def branch(policy, cfg, geometry, record, tick, key, mode, horizon):
    env = environment(cfg, policy, geometry, 5)
    policy.reset()
    env.reset(options={"test_case": record["case"]})
    for action in record["actions"][:tick]:
        _, _, done, truncated, _ = env.step(ActionXY(*action))
        if done or truncated:
            raise RuntimeError("Native continuation could not reconstruct the root")
    reconstructed = env.robot.get_full_state().to_array()
    np.testing.assert_allclose(reconstructed, record["observations"][tick]["robot"], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose([human.get_observable_state().to_array() for human in env.humans],
                               [human["state"] for human in record["observations"][tick]["humans"]],
                               atol=2e-6, rtol=2e-6)
    past = [frame.copy() for frame in record["tokens"][max(0, tick - policy.length):tick]]
    if mode == "all_short":
        past = past[-3:]
    elif mode == "actor_short":
        for frame in past[:-3]:
            frame[3 + key] = 0
    policy.history = deque(past, maxlen=policy.length)
    policy.last_action = ActionXY(*record["actions"][tick - 1])
    tracks = TrackKeys()
    root = tracks.observe(env)
    root_index = int(np.argmax(policy.score(root)))
    start_distance = np.hypot(env.robot.px - env.robot.gx, env.robot.py - env.robot.gy)
    minimum, reward_sum, path, terminal = float("inf"), 0., 0., "running"
    first_action = None
    for step in range(horizon):
        state = tracks.observe(env)
        action = policy.predict(state)
        if first_action is None:
            first_action = list(action)
        previous = np.array(env.robot.get_position())
        _, reward, done, truncated, info = env.step(action)
        reward_sum += policy.gamma ** step * reward
        path += float(np.linalg.norm(np.array(env.robot.get_position()) - previous))
        minimum = min(minimum, info["dmin"])
        if done or truncated:
            terminal = info["event"]
            break
    return {"root_index": root_index, "first_action": first_action, "minimum_clearance": minimum,
            "progress": float(start_distance - np.hypot(env.robot.px - env.robot.gx, env.robot.py - env.robot.gy)),
            "discounted_reward": reward_sum, "path": path, "steps": step + 1, "terminal": terminal}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol, root = json.loads(PROTOCOL.read_text()), Path(args.root)
    cfg = configuration(protocol, "actor")
    policy = ValuePolicy(OrderedValueModel("actor", protocol["width"], protocol["layers"]), cfg, "cuda")
    load_weights(policy.model, root / str(args.seed) / "actor/model.pt", "cuda")
    rows, episodes = [], []
    for geometry in protocol["geometries"]:
        for case in protocol["headroom_cases"]:
            record = run_episode(environment(cfg, policy, geometry, 5), policy, case)
            event = None
            for tick in range(12, len(record["tokens"])):
                key = observed_event(record, tick)
                if key is not None:
                    event = tick, key
                    break
            episodes.append({"geometry": geometry, "case": case, "frames": len(record["tokens"]),
                             "terminal": record["terminal"], "eligible_event": event})
            if event is None:
                continue
            tick, key = event
            branches = {mode: branch(policy, cfg, geometry, record, tick, key, mode, 12)
                        for mode in ("full", "all_short", "actor_short")}
            rows.append({"geometry": geometry, "case": case, "tick": tick, "actor_key": key, "branches": branches})
    summary = {"native_episodes": len(episodes), "uniform_person_frames": 5 * sum(row["frames"] for row in episodes),
               "eligible_first_events": len(rows),
               "actor_reset_changes_root": sum(row["branches"]["actor_short"]["root_index"] != row["branches"]["full"]["root_index"] for row in rows),
               "safe_actor_progress_improvements_005m": sum(
                   row["branches"]["actor_short"]["progress"] - row["branches"]["full"]["progress"] >= .05
                   and row["branches"]["actor_short"]["minimum_clearance"] >= 0 for row in rows),
               "scope": "Exploratory 3-second native continuations; masked-prefix intervention is out of distribution and removes all old evidence for that actor, not motion alone"}
    (root / "revision_shadow.json").write_text(json.dumps({"seed": args.seed, "summary": summary,
                                                          "episodes": episodes, "events": rows}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
