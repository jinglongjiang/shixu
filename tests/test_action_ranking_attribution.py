"""Protect reward/rank/execution semantics in the attribution-only probe."""

import configparser
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.action_ranking_attribution import discrepancy, ranks, stages, truth_same_support
from shixu.features import encode_tracks
from shixu.observations import ObservedTracks


class ActionRankingAttributionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configparser.ConfigParser()
        self.cfg.read_dict({"train": {"gamma": ".99"}, "eval_protocol": {"safety_margin": ".2", "risk_lambda": ".8"},
                            "reward": {"discomfort_dist": ".2"}})

    def test_ranks_are_stable_one_based(self):
        np.testing.assert_array_equal(ranks([1, 3, 3, -1]), [3, 1, 2, 4])

    def test_filter_and_reward_are_not_blended_into_value(self):
        row = stages(np.zeros(3), torch.tensor([.9, .8, .7]), np.array([.1, .3, .4]), self.cfg)
        self.assertEqual(row["raw_top"], 0)
        self.assertEqual(row["final_top"], 1)
        self.assertEqual(row["blocked"], [True, False, False])
        self.assertAlmostEqual(row["risk_penalty"][0], .08)
        self.assertAlmostEqual(row["raw"][1], float(.99 * torch.tensor(.8)))

    def test_terminal_diagnostic_does_not_bootstrap(self):
        row = stages(np.array([1., -.5]), torch.tensor([3., 3.]), np.ones(2), self.cfg, [True, False])
        self.assertEqual(row["raw"][0], 1.)
        self.assertAlmostEqual(row["raw"][1], -.5 + float(.99 * torch.tensor(3.)))

    def test_all_unsafe_preserves_native_fallback(self):
        row = stages(np.zeros(2), torch.tensor([.9, .8]), np.array([.1, .1]), self.cfg)
        self.assertEqual(row["blocked"], [False, False])

    def test_accounting_is_telescoping_not_pure_value_error(self):
        row = discrepancy(.7, .6, .55, .5, .2)
        self.assertAlmostEqual(row["grid_to_executed"], .1)
        self.assertAlmostEqual(row["legal_value_residual"], .3)
        self.assertAlmostEqual(sum(v for k, v in row.items() if k != "total"), row["total"])

    def test_truth_preserves_association_age_mask_and_history(self):
        robot = FullState(0, 0, 0, 0, .3, 0, 4, 1, 0)
        state = ObservedTracks(robot, [ObservableState(1, 0, 0, 0, .3)], (1,), (True,), (.5,), 3)
        root = dict(state=state, snapshot=dict(tracks=[(0, 1, np.zeros(5), 0)]))
        human = SimpleNamespace(px=2., py=1., vx=.2, vy=.1, radius=.3)
        env = SimpleNamespace(time_step=.25, robot=SimpleNamespace(get_full_state=lambda: robot), humans=[human])
        truth = truth_same_support(root, env)
        self.assertEqual(truth.track_ids, (1,))
        self.assertEqual(truth.track_count, 3)
        self.assertEqual(truth.observed, (False,))
        self.assertEqual(truth.ages, (.75,))
        self.assertEqual(state.observed, (True,))
        np.testing.assert_array_equal(encode_tracks(truth)[:, 12], [0, 0, 1, 0])
        self.assertEqual(state.human_states[0].px, 1)


if __name__ == "__main__":
    unittest.main()
