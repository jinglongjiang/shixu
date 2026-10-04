"""Offline hidden-state labels cannot become legal-history inputs."""

import copy
import unittest

import numpy as np

from experiments.latent_information import context, design, history_statistics, own_history


class LatentInformationTests(unittest.TestCase):
    def frame(self, tick):
        legal = np.asarray([[tick, 1, .2, .3, .1, 0, 1, 1], [2, tick, .4, .5, .1, 0, 1, 1]], dtype=float)
        return {"legal": legal, "robot": np.arange(9, dtype=float),
                "snapshot": {"humans": [[0, 0, 0, 0, .1, 99, 99, 1, 0]], "preferred": [[99, 99]]}}

    def test_context_ignores_hidden_labels(self):
        before = self.frame(3)
        after = copy.deepcopy(before)
        after["snapshot"]["humans"][0][5:7] = [-999, -999]
        after["snapshot"]["preferred"] = [[-999, -999]]
        np.testing.assert_array_equal(context(before, 0), context(after, 0))
        self.assertEqual(context(before, 0).shape, (49,))

    def test_history_is_causal_and_keeps_actor_association(self):
        frames = [self.frame(tick) for tick in range(7)]
        before = own_history(frames, 3, 1)
        frames[4]["legal"][:] = 999
        np.testing.assert_array_equal(before, own_history(frames, 3, 1))
        np.testing.assert_array_equal(before[-1], frames[3]["legal"][1])
        np.testing.assert_array_equal(before[:20], np.zeros((20, 8)))

    def test_windows_keep_same_shape_and_only_requested_past(self):
        frames = [self.frame(tick) for tick in range(24)]
        row = {"context": context(frames[-1], 0), "history": own_history(frames, 23, 0),
               "statistics": history_statistics(own_history(frames, 23, 0), np.asarray([0, 1])),
               "target": np.ones(8)}
        current = design([row], "current")
        short = design([row], "history3")
        long = design([row], "history24")
        self.assertEqual(current.shape, (1, 233))
        self.assertEqual(short.shape, long.shape)
        np.testing.assert_array_equal(current[:, 49:], np.zeros((1, 184)))
        np.testing.assert_array_equal(short[:, 49:49 + 21 * 8], np.zeros((1, 168)))
        np.testing.assert_array_equal(short[:, -16:], row["history"][-3:-1].reshape(1, -1))
        changed = copy.deepcopy(row)
        changed["target"][:] = -999
        np.testing.assert_array_equal(long, design([changed], "history24"))

    def test_statistics_only_use_valid_measurements(self):
        history = np.zeros((24, 8))
        history[-2] = [0, 1, .2, .3, .1, 0, 1, 1]
        history[-1] = [5, 6, 99, 99, .1, 1, 0, 1]
        result = history_statistics(history, np.asarray([0, 1]))
        self.assertEqual(result.shape, (18,))
        np.testing.assert_allclose(result[6:8], [.2, .3])
        np.testing.assert_array_equal(result[10:12], [0, 0])


if __name__ == "__main__":
    unittest.main()
