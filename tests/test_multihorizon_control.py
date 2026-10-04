"""Frozen multihorizon scoring must preserve the native execution contract."""

import configparser
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState
from experiments.latent_information import snapshot
from experiments.multihorizon_control import (Trajectory, archive_parity, feedback,
                                              fork, legal_root, select, select_episodes)
from shixu.features import window
from shixu.forecast import ForecastSuccessorValueModel
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


class MultihorizonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.cfg = configparser.ConfigParser()
        self.cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        self.cfg.set("model", "representation", "tracks")
        self.cfg.add_section("observation")
        self.cfg.set("observation", "retention_seconds", "2")

    def test_selection_uses_case_order_not_prediction_or_outcome_gain(self):
        episodes = []
        for people in (5, 10, 20):
            for geometry in ("circle", "square"):
                for case, terminal in ((4, "timeout"), (2, "collision"), (3, "timeout"),
                                       (0, "reach_goal"), (1, "timeout")):
                    episodes.append(dict(people=people, geometry=geometry, case=case, terminal=terminal))
        selection = select_episodes(episodes)
        for start in range(0, len(selection), 4):
            self.assertEqual([r["case"] for _, r in selection[start:start + 4]], [2, 1, 3, 0])
            self.assertEqual([g for g, _ in selection[start:start + 4]],
                             ["failure", "failure", "failure", "success_guard"])

    def test_trajectory_replays_eight_points_then_only_cv_tail(self):
        points = np.stack((np.arange(9) ** 2 / 100, np.arange(9) / 10), axis=1)
        policy = Trajectory(points)
        position = points[0].copy()
        for tick in range(11):
            state = SimpleNamespace(self_state=SimpleNamespace(position=tuple(position)))
            command = policy.predict(state)
            position += .25 * np.asarray(command)
            expected = points[tick + 1] if tick < 8 else points[8] + (tick - 7) * (points[8] - points[7])
            np.testing.assert_allclose(position, expected, atol=1e-12)

    def test_retained_root_uses_legal_measurement_not_hidden_simulator_position(self):
        saved = dict(robot=FullState(0, 0, 0, 0, .3, 0, 4, 1, 0).to_array(), time=1.,
                     tracks=[(1, 2, np.asarray([2., 1., .2, -.1, .3]), .5)])
        token = np.zeros((4, 13), np.float32)
        token[3, 12] = 1
        root = legal_root(saved, token)
        self.assertEqual(root.track_ids, (2,))
        self.assertEqual(root.track_count, 3)
        self.assertEqual(root.observed, (False,))
        np.testing.assert_allclose(root.human_states[0].position, [2.1, .95])

    def test_reward_ties_use_parent_not_new_progress_metric(self):
        choice = select(np.zeros(4), np.array([5., 1., 2., 4.]), np.ones(4), self.cfg)
        self.assertEqual(choice["action"], 3)
        self.assertEqual(choice["raw_best_ties"], 4)
        self.assertEqual(choice["filtered_best_ties"], 3)

    def test_archive_parity_checks_entire_continuation_not_only_terminal(self):
        root = dict(archived_tail_actions=[[1., 0.], [0., 1.]], archived_rewards=[0., 1.],
                    archived_remaining_seconds=.5, episode=dict(terminal="reach_goal"))
        native = dict(commands=[[1., 0.], [0., 1.]], return_=.99, seconds=.5, terminal="reach_goal")
        self.assertTrue(archive_parity(root, native, self.cfg)["passed"])
        native["commands"][1] = [1., 0.]
        self.assertFalse(archive_parity(root, native, self.cfg)["passed"])

    def test_batched_feedback_matches_native_predict_and_writes_once(self):
        model = ForecastSuccessorValueModel("cv", 16, 1).eval()
        policy = ValuePolicy(model, self.cfg)
        env = environment(self.cfg, policy, "circle", 5)
        env.reset(options={"test_case": 80000})
        observer = OccludedTracks(2)
        state = observer.observe(env)
        root = dict(episode=dict(people=5, geometry="circle", case=80000),
                    snapshot=snapshot(env, observer), state=state,
                    history=window([policy.encode(state)], 24, "zero"), previous=[.1, .2])
        worlds = [fork(root, model, self.cfg, "cpu") for _ in range(2)]
        actions, checks = feedback(worlds, self.cfg, validate=True)
        reference = fork(root, model, self.cfg, "cpu")
        native = reference["policy"].predict(reference["observer"].observe(reference["env"]))
        for world, action in zip(worlds, actions):
            np.testing.assert_allclose(action, native, atol=1e-12)
            self.assertEqual(len(world["policy"].history), 24)
            np.testing.assert_allclose(world["policy"].history[-1], policy.encode(state), atol=2e-7, rtol=0)
        self.assertTrue(all(c["batched_action"] == c["scalar_action"] for c in checks))


if __name__ == "__main__":
    unittest.main()
