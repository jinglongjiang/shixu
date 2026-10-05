"""The confirmation queue and history accounting cannot select by outcome."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.pas_history_confirmation import ARMS, CASES, information


class PaSHistoryConfirmationTests(unittest.TestCase):
    def test_queue_is_balanced_unique_and_independent_of_discovery_cases(self):
        self.assertEqual(len(CASES), 32)
        self.assertEqual(len(set(CASES)), 32)
        for people in (10, 20):
            for geometry in ("circle_crossing", "square_crossing"):
                self.assertEqual([c for n, g, c in CASES if n == people and g == geometry],
                                 list(range(93000, 93008)))
        self.assertFalse(set(c for _, _, c in CASES) & set(range(92000, 92004)))

    def test_every_eligible_case_has_the_same_five_arms(self):
        self.assertEqual(ARMS, ("full4", "projected_parent", "short_cv", "long_cv", "long_hold"))

    def test_extra_occupancy_separates_true_and_false_history_without_mutation(self):
        short = np.zeros((2, 2))
        long = np.asarray([[1., 1.], [0., 1.]])
        root = dict(time=4., observation={"grid": np.asarray([[[.5, .5], [.5, 0.]]]),
                    "label_grid": np.asarray([[[1., 0.], [0., 0.]]])},
                    tracker=SimpleNamespace(tracks=[{"time": 0.}, {"time": 1.},
                                                    {"time": 3.25}, {"time": 4.}]))
        before = copy.deepcopy(root)
        result = information(short, long, root)
        self.assertEqual(result, dict(added_cells=2, added_true_occupied=1,
                                     added_true_free=1, older_track_ages=[4., 3.]))
        np.testing.assert_array_equal(root["observation"]["grid"], before["observation"]["grid"])
        self.assertEqual(root["tracker"].tracks, before["tracker"].tracks)
        np.testing.assert_array_equal(short, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
