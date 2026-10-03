"""A single episode runner shared by IL, RL and evaluation."""

import copy
import numpy as np

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.robot import Robot
from .observations import TrackKeys


def environment(config, policy, geometry="circle", people=5):
    cfg = copy.deepcopy(config)
    cfg.set("sim", "human_num", str(people))
    cfg.set("sim", "test_sim", geometry + "_crossing")
    env = CrowdSim()
    env.configure(cfg)
    env.phase = "test"
    robot = Robot(cfg, "robot")
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)
    return env


def orca_teacher(config):
    teacher = ORCA()
    teacher.configure(config)
    teacher.multiagent_training = False
    teacher.safety_space = config.getfloat("imitation_learning", "teacher_safety_space")
    teacher.neighbor_dist = config.getfloat("imitation_learning", "teacher_neighbor_dist")
    teacher.max_neighbors = config.getint("imitation_learning", "teacher_max_neighbors")
    teacher.time_horizon = config.getfloat("imitation_learning", "teacher_time_horizon")
    teacher.time_horizon_obst = config.getfloat("imitation_learning", "teacher_time_horizon_obst")
    return teacher


def run_episode(env, policy, case, epsilon=0.0, teacher=None, phase="test"):
    policy.reset()
    policy.set_phase(phase)
    env.phase = "test"
    env.reset(options={"test_case": int(case)})
    tracks = TrackKeys()
    if teacher is not None:
        teacher.sim, teacher._last_pref_vel = None, None
    record = {"case": int(case), "tokens": [], "observations": [], "actions": [], "rewards": [], "clearances": []}
    done, truncated, path = False, False, 0.0
    while not done and not truncated:
        state = tracks.observe(env)
        record["tokens"].append(policy.encode(state))
        # Simulator identity is a stable association key, never a learned numeric feature.
        record["observations"].append({"time": env.global_time, "robot": state.self_state.to_array().tolist(),
                                        "humans": [{"track_id": key, "state": h.to_array().tolist()}
                                                   for key, h in zip(state.track_ids, state.human_states)]})
        if teacher is None:
            action = policy.predict(state, epsilon)
        else:
            action = teacher.predict(state)
        previous = np.array(env.robot.get_position())
        _, reward, done, truncated, info = env.step(action)
        path += float(np.linalg.norm(np.array(env.robot.get_position()) - previous))
        record["actions"].append([action.vx, action.vy])
        record["rewards"].append(reward)
        record["clearances"].append(info["dmin"])
    record.update(terminal=info["event"], navigation_time=env.global_time, path=path,
                  minimum_clearance=min(record["clearances"]))
    return record
