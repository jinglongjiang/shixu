"""Observed track association, separate from numeric network features."""

from typing import NamedTuple

import numpy as np

from crowd_sim.envs.utils.state import ObservableState


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


class ObservedTracks(NamedTuple):
    self_state: object
    human_states: list
    track_ids: tuple
    observed: tuple
    ages: tuple
    track_count: int


class OccludedTracks:
    """Centre-ray body occlusion; unseen actors never enter the track store.

    Association is ideal once an actor has been seen. Hidden updates use only
    its last measurement and elapsed time, not its simulator state.
    """

    def __init__(self, retention_seconds=2.0):
        self.retention_seconds = retention_seconds
        self.tracks = {}
        self.counts = {}

    @staticmethod
    def visibility(robot, humans):
        if not humans:
            return np.zeros(0, dtype=bool)
        offsets = np.array([[h.px - robot.px, h.py - robot.py] for h in humans])
        lengths = np.linalg.norm(offsets, axis=1)
        directions = offsets / np.maximum(lengths[:, None], 1e-9)
        along = directions @ offsets.T
        transverse2 = (offsets ** 2).sum(1)[None] - along ** 2
        radii = np.array([h.radius for h in humans])
        blockers = (along > 0) & (along < lengths[:, None]) & (transverse2 < radii[None] ** 2)
        np.fill_diagonal(blockers, False)
        return ~blockers.any(1)

    def observe(self, env):
        robot = env.robot.get_full_state()
        visible = self.visibility(robot, env.humans)
        now = env.global_time
        for human, allowed in zip(env.humans, visible):
            if allowed:
                key = self.tracks[human][0] if human in self.tracks else len(self.tracks)
                self.tracks[human] = (key, human.get_observable_state(), now)
        states, keys, flags, ages = [], [], [], []
        visible_objects = {h for h, allowed in zip(env.humans, visible) if allowed}
        for human, (key, measurement, stamp) in self.tracks.items():
            age = now - stamp
            if age <= self.retention_seconds:
                states.append(ObservableState(measurement.px + age * measurement.vx,
                                              measurement.py + age * measurement.vy,
                                              measurement.vx, measurement.vy, measurement.radius))
                keys.append(key)
                flags.append(human in visible_objects)
                ages.append(age)
        self.counts = {"visible": int(visible.sum()), "population": len(env.humans),
                       "retained_hidden": sum(not flag for flag in flags)}
        return ObservedTracks(robot, states, tuple(keys), tuple(flags), tuple(ages), len(self.tracks))
