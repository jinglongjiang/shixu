"""Bounded SICNav response-memory kill test; no planner solve or training."""

import configparser
import copy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path("/home/abc/temp/safe-interactive-crowdnav")
ARCHIVE = Path("/home/abc/workspace/bayes_occ_mpc/src/safe-interactive-crowdnav/results_sicnav-np")
OUT = ROOT / "outputs/interaction_response_audit"
CONFIGS = ("hallway_bottleneck_N_3", "circle_crossing_N_5")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    if path.exists():
        raise RuntimeError("Refusing to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def physical_state(env):
    robot = env.robot
    velocity = robot.vx*np.cos(robot.theta) + robot.vy*np.sin(robot.theta)
    return np.asarray([robot.px, robot.py, robot.theta, velocity] +
                      [v for human in env.humans for v in (human.px, human.py, human.vx, human.vy)])


def logged_state(value, count):
    return np.r_[value[:4, 0], np.asarray([value[6+6*i:10+6*i, 0] for i in range(count)]).ravel()]


def main():
    sys.path.insert(0, str(PARENT))
    from sicnav.policy.policy_factory import policy_factory
    from crowd_sim_plus.envs.crowd_sim_plus import CrowdSimPlus
    from crowd_sim_plus.envs.utils.robot_plus import Robot
    from crowd_sim_plus.envs.policy.orca_plus import ORCAPlus
    from crowd_sim_plus.envs.utils.action import ActionRot

    sources = ["crowd_sim_plus/envs/crowd_sim_plus.py", "crowd_sim_plus/envs/policy/orca.py",
               "crowd_sim_plus/envs/policy/orca_plus.py", "crowd_sim_plus/envs/policy/social_force.py",
               "crowd_sim_plus/envs/utils/human_plus.py", "crowd_sim_plus/envs/utils/agent_plus.py",
               "sicnav/policy/campc.py", "sicnav/configs/env.config", "sicnav/configs/policy.config"]
    files = [next((ARCHIVE / name).glob("*tc_0.pkl")) for name in CONFIGS]
    protocol = dict(parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PARENT,
                    text=True).strip(), parent_dirty=subprocess.check_output(["git", "status", "--porcelain"],
                    cwd=PARENT, text=True), source_sha256={p: digest(PARENT/p) for p in sources},
                    archive_sha256={str(p): digest(p) for p in files}, script_sha256=digest(Path(__file__)),
                    selection="Existing SICNav-np case0 hallway and circle only. Roots0, midpoint, last "
                              "logged step, fixed before response queries. No new episodes or root search.",
                    contract="Replay logged first controls with original ActionRot omega*dt conversion; "
                             "verify current p/v/heading/speed within1e-5 of logged initial MPC columns. "
                             "Compare history-bearing ORCAPlus with fresh same-parameter ORCAPlus, "
                             "at identical full human state, neighbours, robot state and walls. "
                             "Goals/v_pref held fixed. No new modes, planner solve, controller or training.",
                    sensitivity="Only a function-level positive control changes current robot velocity "
                                "to +/-v_pref along its heading. Not a navigation intervention or benefit.",
                    limits="Two old successful runs; missing raw observation/reward/executed-command archives. "
                           "Logged controls are inferred from saved optimizer outputs and parity-checked. "
                           "No natural same-observation/different-history failure pair or navigation gain claim.")
    save(OUT/"protocol.json", protocol)
    episodes, rows = [], []
    for path in files:
        with path.open("rb") as stream:
            stored = pickle.load(stream)
        geometry, people = path.parent.name.rsplit("_N_", 1)
        config = configparser.RawConfigParser()
        config.read(PARENT/"sicnav/configs/env.config")
        config.set("sim", "test_sim", geometry)
        config.set("sim", "human_num", people)
        env = CrowdSimPlus()
        env.configure(config)
        robot = Robot(config, "robot")
        policy = policy_factory["campc"]()
        policy_config = configparser.RawConfigParser()
        policy_config.read(PARENT/"sicnav/configs/policy.config")
        policy.configure(policy_config)
        robot.set_policy(policy)
        env.set_robot(robot)
        env.reset("test", stored["test_case"])
        assert robot.visible and all(isinstance(h.policy, ORCAPlus) for h in env.humans)
        data = stored["campc/campc_data"]
        length = len(data["all_x_val"])
        roots = (0, length//2, length-1)
        maximum_error = 0.
        for tick, (value, control) in enumerate(zip(data["all_x_val"], data["all_u_val"])):
            error = float(np.max(abs(physical_state(env)-logged_state(value, len(env.humans)))))
            maximum_error = max(maximum_error, error)
            np.testing.assert_allclose(physical_state(env), logged_state(value, len(env.humans)), atol=1e-5, rtol=0)
            if tick in roots:
                for index, human in enumerate(env.humans):
                    neighbours = [h.get_observable_state() for h in env.humans if h is not human]
                    neighbours += [robot.get_observable_state()]
                    existing = human.policy
                    warm = np.asarray(human.act(neighbours, env.static_obstacles))
                    assert existing.sim is None
                    cold = ORCAPlus()
                    cold.configure(config, "humans")
                    human.policy = cold
                    try:
                        fresh = np.asarray(human.act(neighbours, env.static_obstacles))
                        sensitive = []
                        for sign in (-1, 1):
                            changed = copy.deepcopy(neighbours)
                            robot_state = changed[-1]
                            robot_state.vx = sign*robot.v_pref*np.cos(robot.theta)
                            robot_state.vy = sign*robot.v_pref*np.sin(robot.theta)
                            robot_state.velocity = (robot_state.vx, robot_state.vy)
                            sensitive.append(float(np.max(abs(np.asarray(human.act(changed, env.static_obstacles))-fresh))))
                    finally:
                        human.policy = existing
                    np.testing.assert_array_equal(warm, fresh)
                    rows.append(dict(geometry=geometry, people=int(people), case=stored["test_case"],
                                     tick=tick, actor=index, warm=warm.tolist(), cold=fresh.tolist(),
                                     history_reset_difference=float(np.max(abs(warm-fresh))),
                                     current_robot_velocity_difference=max(sensitive)))
            env.step(ActionRot(float(control[0, 0]), float(control[1, 0])*env.time_step))
        episodes.append(dict(archive=str(path), geometry=geometry, people=int(people), frames=length,
                             roots=list(roots), maximum_physical_error=maximum_error,
                             archived_success=stored["test_case_success"], archived_collisions=stored["num_collisions"]))
    assert all(digest(PARENT/p) == h for p, h in protocol["source_sha256"].items())
    assert all(digest(Path(p)) == h for p, h in protocol["archive_sha256"].items())
    result = dict(protocol_sha256=digest(OUT/"protocol.json"), episodes=episodes, rows=rows,
                  response_queries=len(rows), changed_by_history_reset=sum(r["history_reset_difference"]>0 for r in rows),
                  changed_by_current_robot_velocity=sum(r["current_robot_velocity_difference"]>0 for r in rows),
                  navigation_action_value_evidence=False,
                  verdict="NO_PERSISTENT_RESPONSE_STATE_IN_TESTED_NATIVE_ORCA")
    save(OUT/"results.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
