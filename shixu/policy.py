"""Native candidate generation and one-step Bellman-style value lookahead."""

from collections import deque
import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
from .features import encode_state, window


def actions(config):
    speeds, headings = config.getint("policy", "n_speeds"), config.getint("policy", "n_headings")
    maximum = config.getfloat("policy", "v_max")
    grid = []
    if config.getboolean("policy", "include_stop"):
        grid.append(ActionXY(0, 0))
    for heading in range(headings):
        angle = 2 * np.pi * heading / headings
        for level in range(speeds):
            if config.get("policy", "sampling") == "exponential":
                speed = (np.exp((level + 1) / speeds) - 1) / (np.e - 1) * maximum
            else:
                minimum = config.getfloat("policy", "v_min")
                speed = maximum if speeds == 1 else minimum + (maximum - minimum) * level / (speeds - 1)
            grid.append(ActionXY(float(speed * np.cos(angle)), float(speed * np.sin(angle))))
    return grid


def successor(state, action, dt):
    robot = state.self_state
    next_robot = FullState(robot.px + dt * action.vx, robot.py + dt * action.vy,
                           action.vx, action.vy, robot.radius, robot.gx, robot.gy, robot.v_pref, robot.theta)
    humans = [ObservableState(h.px + dt * h.vx, h.py + dt * h.vy, h.vx, h.vy, h.radius) for h in state.human_states]
    return JointState(next_robot, humans)


class ValuePolicy:
    multiagent_training = False
    kinematics = "holonomic"

    def __init__(self, model, config, device="cpu"):
        self.model, self.config = model.to(device), config
        self.device = torch.device(device)
        self.action_space = actions(config)
        self.time_step = config.getfloat("env", "time_step")
        self.gamma = config.getfloat("train", "gamma")
        self.length = config.getint("buffer", "seq_len")
        self.history = deque(maxlen=self.length)
        self.last_action = None
        self.phase = "test"

    def set_phase(self, phase):
        self.phase = phase

    def reset(self):
        self.history.clear()
        self.last_action = None

    def immediate_reward(self, current, future, action):
        robot = future.self_state
        clearance = min(np.hypot(robot.px - h.px, robot.py - h.py) - robot.radius - h.radius
                        for h in future.human_states)
        if clearance < 0:
            return self.config.getfloat("reward", "collision_penalty"), clearance
        if np.hypot(robot.px - robot.gx, robot.py - robot.gy) < self.config.getfloat("robot", "success_radius"):
            return self.config.getfloat("reward", "success_reward"), clearance
        old = current.self_state
        progress = np.hypot(old.px - old.gx, old.py - old.gy) - np.hypot(robot.px - robot.gx, robot.py - robot.gy)
        cfg = self.config
        reward = cfg.getfloat("reward", "progress_reward") * progress + cfg.getfloat("reward", "time_penalty")
        if np.hypot(action.vx, action.vy) < 0.05:
            reward += cfg.getfloat("reward", "stand_penalty")
        margin = cfg.getfloat("reward", "discomfort_dist")
        if clearance < margin:
            reward -= cfg.getfloat("reward", "discomfort_penalty_factor") * (margin - clearance) * self.time_step
        return float(reward), float(clearance)

    @torch.inference_mode()
    def score(self, state):
        self.model.eval()
        current = encode_state(state)
        sequences, rewards, clearances = [], [], []
        for action in self.action_space:
            future = successor(state, action, self.time_step)
            sequences.append(window(list(self.history) + [current, encode_state(future)], self.length))
            reward, clearance = self.immediate_reward(state, future, action)
            rewards.append(reward)
            clearances.append(clearance)
        values = self.model(torch.as_tensor(np.asarray(sequences), device=self.device))
        scores = torch.as_tensor(rewards, device=self.device) + self.gamma * values
        cfg = self.config
        if self.phase != "train":
            distance = torch.as_tensor(clearances, device=self.device)
            margin = cfg.getfloat("eval_protocol", "safety_margin")
            risk = cfg.getfloat("eval_protocol", "risk_lambda")
            if margin > 0 and bool((distance >= margin).any()):
                scores = scores.masked_fill(distance < margin, -1e9)
            if risk > 0:
                threshold = margin if margin > 0 else cfg.getfloat("reward", "discomfort_dist")
                scores -= risk * torch.clamp(threshold - distance, min=0)
        scores[0] -= 1e-3
        return scores.cpu().numpy()

    def predict(self, state, epsilon=0.0):
        robot = state.self_state
        if np.hypot(robot.px - robot.gx, robot.py - robot.gy) < self.config.getfloat("robot", "success_radius"):
            return ActionXY(0, 0)
        scores = self.score(state)
        index = np.random.randint(len(scores)) if np.random.random() < epsilon else int(np.argmax(scores))
        selected = self.action_space[index]
        self.history.append(encode_state(state))
        alpha = self.config.getfloat("eval_protocol", "action_smoothing") if self.phase != "train" else 0
        if self.last_action is not None and alpha > 0:
            selected = ActionXY(alpha * self.last_action.vx + (1 - alpha) * selected.vx,
                                alpha * self.last_action.vy + (1 - alpha) * selected.vy)
        self.last_action = selected
        return selected
