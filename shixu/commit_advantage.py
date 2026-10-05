"""Frozen one-shot advantage estimates initiate the existing safe option."""

from collections import deque

import numpy as np
import torch
from torch import nn

from crowd_sim.envs.utils.action import ActionXY
from .commit_features import extract
from .policy import ValuePolicy


class AdvantageEstimator:
    def __init__(self, fitted):
        self.fitted = fitted
        self.network = None
        if fitted["kind"] == "mlp":
            with torch.random.fork_rng(devices=[]):
                self.network = nn.Sequential(nn.Linear(fitted["dim"], 16), nn.ReLU(), nn.Linear(16, 1))
            self.network.load_state_dict(fitted["parameters"])
            self.network.eval()

    @property
    def parameters(self):
        return (len(self.fitted["coefficients"]) if self.network is None
                else sum(p.numel() for p in self.network.parameters()))

    @torch.inference_mode()
    def __call__(self, features):
        p = self.fitted
        x = (np.asarray(features[:p["dim"]])-p["mean"])/p["std"]
        if self.network is None:
            return float(np.append(x, 1.)@p["coefficients"])
        return float(self.network(torch.as_tensor(x[None], dtype=torch.float32)).item()*p["scale"])


class CommitAdvantagePolicy(ValuePolicy):
    """Change only initiation; hold a grid for eight ticks with V0.1 release."""

    def __init__(self, model, config, gate, device="cpu"):
        super().__init__(model, config, device)
        if self.time_step != .25 or config.getfloat("eval_protocol", "safety_margin") != .2:
            raise ValueError("Expected the frozen control clock and safety margin")
        self.gate = gate
        self.reset()

    def reset(self):
        super().reset()
        self.distances, self.grids, self.commands = (deque(maxlen=8) for _ in range(3))
        self.active_grid, self.remaining = None, 0
        self.armed = True
        self.tick = self.starts = self.held_steps = 0
        self.last_decision = None
        self.last_features = None

    @torch.inference_mode()
    def predict(self, state, epsilon=0.0):
        if self.phase == "train" or epsilon != 0:
            raise ValueError("Frozen Parent and supervised initiation only")
        robot = state.self_state
        distance = float(np.hypot(robot.px-robot.gx, robot.py-robot.gy))
        if distance < self.config.getfloat("robot", "success_radius"):
            return ActionXY(0, 0)
        commands = self.candidate_actions()
        _, _, clearance = self.aligned_candidates(state, commands)
        safe = clearance >= .2
        released, started, proposal, estimate = None, False, None, None
        self.last_features = None
        if self.active_grid is not None and (not safe.any() or not safe[self.active_grid]):
            released = "all-unsafe" if not safe.any() else "safety-blocked"
            self.active_grid, self.remaining = None, 0
        if self.active_grid is None:
            info = extract(self, state, self.distances, self.grids, self.commands)
            proposal = info["index"]
            self.last_features = info["features"]
            estimate = self.gate(self.last_features)
            eligible = len(self.distances) == 8 and distance > 1.
            trigger = eligible and estimate > 0.
            # As in V0.1, initiation must clear before it can recur.
            if not trigger:
                self.armed = True
            if trigger and self.armed and released is None and safe[proposal]:
                self.active_grid, self.remaining = proposal, 8
                self.armed = False
                self.starts += 1
                started = True
        else:
            trigger = None
        np.random.random()
        held = self.active_grid is not None
        index = self.active_grid if held else proposal
        selected = commands[index]
        self.history.append(self.encode(state))
        self.last_action = selected
        progress = self.distances[0]-distance if len(self.distances) == 8 else None
        self.distances.append(distance)
        self.grids.append(index)
        self.commands.append(selected)
        if held:
            self.held_steps += 1
            self.remaining -= 1
            if self.remaining == 0:
                self.active_grid = None
                released = "budget-completed"
        self.last_decision = dict(tick=self.tick, time=self.tick*.25, distance=distance,
            past_two_second_progress=progress, trigger=trigger, advantage=estimate,
            proposal=proposal, selected_grid=int(index), held=held, started=started,
            release=released, predicted_cv_clearance=float(clearance[index]),
            blocked=bool(safe.any() and not safe[index]), no_margin_safe_candidate=not bool(safe.any()))
        self.tick += 1
        return selected
