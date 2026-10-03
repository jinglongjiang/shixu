"""Observed track association, separate from numeric network features."""

from typing import NamedTuple


class TrackedState(NamedTuple):
    self_state: object
    human_states: list
    track_ids: tuple


class TrackKeys:
    def __init__(self):
        self.keys = {}

    def observe(self, env):
        for human in env.humans:
            if human not in self.keys:
                self.keys[human] = len(self.keys)
        return TrackedState(env.robot.get_full_state(),
                            [human.get_observable_state() for human in env.humans],
                            tuple(self.keys[human] for human in env.humans))
