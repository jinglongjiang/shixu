"""Evaluation-only, causal two-second commitment of a native grid action."""

from collections import deque

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from .policy import ValuePolicy


def low_progress(distances):
    return len(distances) == 9 and distances[-1] > 1. and distances[0] - distances[-1] <= .2 + 1e-12


def native_blocked(clearances, margin):
    clearances = np.asarray(clearances)
    return (clearances < margin) & (margin > 0) & bool((clearances >= margin).any())


class ActionCommitmentPolicy(ValuePolicy):
    """No extra learned parameters; the parent score and execution stay intact."""

    def __init__(self, model, config, device="cpu"):
        super().__init__(model, config, device)
        if self.time_step != .25:
            raise ValueError("V0 requires the original0.25s control clock")
        self.reset()

    def reset(self):
        super().reset()
        self.distances = deque(maxlen=9)
        self.active_grid = None
        self.remaining = 0
        self.armed = True
        self.tick = 0
        self.starts = 0
        self.held_steps = 0
        self.last_decision = None

    @torch.inference_mode()
    def predict(self, state, epsilon=0.0):
        if self.phase == "train" or epsilon != 0:
            raise ValueError("V0 is a frozen, evaluation-only prototype")
        robot = state.self_state
        distance = float(np.hypot(robot.px-robot.gx, robot.py-robot.gy))
        if distance < self.config.getfloat("robot", "success_radius"):
            return ActionXY(0, 0)
        self.distances.append(distance)
        trigger = low_progress(self.distances)
        progress = self.distances[0]-distance if len(self.distances) == 9 else None
        if self.active_grid is None and not trigger:
            self.armed = True

        commands = self.candidate_actions()
        _, _, clearances = self.aligned_candidates(state, commands)
        blocked = native_blocked(clearances, self.config.getfloat("eval_protocol", "safety_margin"))
        released, started = None, False
        if self.active_grid is not None and blocked[self.active_grid]:
            self.active_grid, self.remaining = None, 0
            released = "safety-blocked"

        proposal = None
        if self.active_grid is None:
            proposal = int(np.argmax(self.score(state, commands)))
            if trigger and self.armed and released is None and not blocked[proposal]:
                self.active_grid, self.remaining = proposal, 8
                self.armed = False
                self.starts += 1
                started = True
        # Preserve the parent's RNG draw even when a held action skips the critic.
        # Otherwise simulator randomness could differ for reasons other than control.
        np.random.random()
        held = self.active_grid is not None
        index = self.active_grid if held else proposal
        selected = commands[index]
        self.history.append(self.encode(state))
        self.last_action = selected
        if held:
            self.held_steps += 1
            self.remaining -= 1
            if self.remaining == 0:
                self.active_grid = None
                released = "budget-completed"
        self.last_decision = dict(tick=self.tick, time=self.tick*.25, distance=distance,
            past_two_second_progress=progress, trigger=bool(trigger), proposal=proposal,
            selected_grid=int(index), held=held, started=started, release=released,
            predicted_cv_clearance=float(clearances[index]), blocked=bool(blocked[index]),
            no_margin_safe_candidate=not bool((clearances >= self.config.getfloat("eval_protocol", "safety_margin")).any()))
        self.tick += 1
        return selected
