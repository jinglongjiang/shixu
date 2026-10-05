"""Frozen V0 with one additional same-tick all-unsafe release rule."""

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from .commitment import ActionCommitmentPolicy, low_progress, native_blocked


class AllUnsafeReleasePolicy(ActionCommitmentPolicy):
    """Keep V0 state/trigger intact; never hold without a margin-safe candidate."""

    @torch.inference_mode()
    def predict(self, state, epsilon=0.0):
        if self.phase == "train" or epsilon != 0:
            raise ValueError("V0.1 is a frozen, evaluation-only prototype")
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
        margin = self.config.getfloat("eval_protocol", "safety_margin")
        blocked = native_blocked(clearances, margin)
        no_safe = not bool((clearances >= margin).any())
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
        # Apply the same release to an existing or just-started commitment.
        if self.active_grid is not None and no_safe:
            self.active_grid, self.remaining = None, 0
            released = "all-unsafe"
            if proposal is None:
                proposal = int(np.argmax(self.score(state, commands)))
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
            no_margin_safe_candidate=no_safe)
        self.tick += 1
        return selected
