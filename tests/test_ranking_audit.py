"""Ranking diagnostics preserve the evaluator and use continuous clearance."""

import configparser
import unittest

import numpy as np

from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.ranking_audit import best_rank, future_reference, stages
from shixu.features import encode_tracks, window
from shixu.motion import MotionValueModel
from shixu.observations import ObservedTracks
from shixu.policy import ValuePolicy


class RankingAuditTests(unittest.TestCase):
    def state(self):
        return ObservedTracks(FullState(0, 0, 0, 0, .1, 0, 4, 1, 0),
                              [ObservableState(1, 0, 0, 0, .1)], (0,), (True,), (0.,), 1)

    def test_continuous_collision_between_clear_endpoints(self):
        truth = np.asarray([[[1., 0., .1]], [[1., 0., .1]]])
        clearance, progress = future_reference(self.state(), np.asarray([[2., 0.], [0., 1.]]), truth, 1.)
        self.assertAlmostEqual(clearance[0], -.2)
        self.assertGreater(clearance[1], 0)
        self.assertAlmostEqual(progress[1], 1.)

    def test_rank_is_one_based_and_unavailable_is_not_success(self):
        self.assertEqual(best_rank(np.asarray([.1, .3, .2]), np.asarray([True, False, True])), 2)
        self.assertIsNone(best_rank(np.asarray([.1, .3, .2]), np.zeros(3, dtype=bool)))

    def test_stage_matches_native_score_without_double_current_frame(self):
        cfg = configparser.ConfigParser()
        cfg.read("shixu/default.ini")
        cfg.set("model", "representation", "tracks")
        cfg.set("buffer", "seq_len", "4")
        for kind in ("gru", "kda"):
            policy = ValuePolicy(MotionValueModel(kind, width=16, layers=1, clock="observation"), cfg)
            state = self.state()
            prefix = window([encode_tracks(state)], 4, "zero")[1:]
            truth = np.asarray([[[1., 0., .1]]] * 9)
            result = stages(policy, state, prefix, [0., .1], truth)
            np.testing.assert_allclose(result["filtered_scores"], policy.score(state), atol=1e-6, rtol=1e-6)
            self.assertTrue(policy.model.use_history)


if __name__ == "__main__":
    unittest.main()
