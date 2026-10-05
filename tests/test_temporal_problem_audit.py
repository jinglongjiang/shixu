"""Offline truth admission must stay separate from deployable history reads."""

import copy
import unittest

import numpy as np

from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.temporal_problem_audit import append_actors, choose_roots, expired_measurements, query_states
from shixu.features import window, encode_tracks
from shixu.observations import ObservedTracks


class TemporalProblemAuditTests(unittest.TestCase):
    def setUp(self):
        robot = FullState(0, 0, 0, 0, .3, 0, 4, 1, 0)
        self.state = ObservedTracks(robot, [ObservableState(1, 0, 0, 0, .3)], (0,), (True,), (0.,), 2)
        history = np.zeros((24, 3, 13), np.float32)
        history[11, 2, [10, 12]] = 1.
        self.root = dict(state=self.state, history=history,
            snapshot=dict(time=6., tracks=[(0, 0, np.array([1., 0, 0, 0, .3]), 6.),
                                         (1, 1, np.array([0., 1, .1, 0, .3]), 3.)],
                          humans=[np.array([1., 0, 0, 0, .3]), np.array([9., 9, 2, 2, .3]),
                                  np.array([3., 2, 0, 0, .3])]))

    def test_simple_is_only_old_measurement_plus_elapsed_time(self):
        simple = query_states(self.root)["legal_long_cv"]
        np.testing.assert_allclose(simple.human_states[1].position, [.3, 1.])
        self.assertEqual(simple.track_ids, (0, 1))
        self.assertEqual(simple.observed, (True, False))
        self.assertEqual(simple.ages, (0., 3.))
        self.assertEqual(simple.track_count, 2)

    def test_changing_truth_cannot_change_legal_baseline(self):
        first = query_states(self.root)["legal_long_cv"]
        self.root["snapshot"]["humans"][1][:4] = 100.
        second = query_states(self.root)["legal_long_cv"]
        np.testing.assert_array_equal(first.human_states[1].to_array(), second.human_states[1].to_array())

    def test_reveal_has_no_fabricated_history_and_cannot_commit(self):
        before = copy.deepcopy(self.root)
        queries = query_states(self.root)
        oracle = queries["oracle_unseen"]
        self.assertEqual(oracle.track_ids, (0, 2))
        self.assertEqual(oracle.track_count, 3)
        self.assertEqual(oracle.observed, (True, True))
        padded = window([*self.root["history"][:-1], encode_tracks(oracle)], 24, "zero")
        self.assertFalse(padded[:-1, 3].any())
        np.testing.assert_array_equal(self.root["history"], before["history"])
        self.assertEqual(self.root["state"].track_ids, before["state"].track_ids)
        self.assertEqual(len(self.root["snapshot"]["tracks"]), 2)

    def test_no_measurement_or_too_old_means_no_legal_readmission(self):
        self.root["history"][11, 2, 10] = 0
        self.assertEqual(expired_measurements(self.root["snapshot"], self.root["history"]), [])
        self.root["history"][11, 2, 10] = 1
        self.root["snapshot"]["time"] = 9.
        self.assertEqual(expired_measurements(self.root["snapshot"], self.root["history"]), [])

    def test_active_actor_cannot_be_replaced(self):
        with self.assertRaises(ValueError):
            append_actors(self.state, [(0, np.ones(5), True, 0.)])

    def test_selection_does_not_use_scores_or_returns(self):
        rows = [dict(stratum="expired", group=g, episode=dict(people=n, geometry="circle", case=c))
                for g in ("failure", "success") for n in (5, 10) for c in (2, 1)]
        chosen = choose_roots(rows, 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual([r["episode"]["case"] for r in chosen], [1] * 4)
        self.assertEqual({(r["group"], r["episode"]["people"]) for r in chosen},
                         {("failure", 5), ("failure", 10), ("success", 5), ("success", 10)})


if __name__ == "__main__":
    unittest.main()
