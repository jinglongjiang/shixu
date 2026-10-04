"""Offline hidden-state labels cannot become legal-history inputs."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.latent_information import context, design, history_statistics, own_history
from experiments.latent_decisions import estimated_snapshot, root_inputs, select_interaction_roots


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

    def test_decision_inputs_ignore_future_and_hidden_goal(self):
        frames = [self.frame(tick) for tick in range(4)]
        episode = {"frames": frames}
        before = root_inputs(episode, 2)
        episode["frames"][3]["legal"][:] = -999
        episode["frames"][2]["snapshot"]["humans"][0][5:7] = [-999, -999]
        after = root_inputs(episode, 2)
        for left, right in zip(before, after):
            for name in ("context", "history", "statistics"):
                np.testing.assert_array_equal(left[name], right[name])

    def test_truth_snapshot_not_mutated_and_cv_does_not_estimate_goal(self):
        episode = {"frames": [self.frame(0)]}
        for mode in ("truth", "cv"):
            result = estimated_snapshot(episode, 0, mode, {})
            result["humans"][0][5] = -999
            self.assertEqual(episode["frames"][0]["snapshot"]["humans"][0][5], 99)

    def test_component_oracles_replace_only_the_named_hidden_quantity(self):
        frame = self.frame(0)
        frame["snapshot"].update(humans=[[0, 1, .2, .3, .1, 9, 10, 1, 0], [2, 0, .4, .5, .1, 8, 11, 1, 0]],
                                 preferred=[[.2, .3], [.4, .5]], tracks=[(0, 0, [], 0), (1, 1, [], 0)])
        episode = {"frames": [frame]}
        with patch("experiments.latent_decisions.predict", return_value=np.ones((2, 8))):
            estimated = estimated_snapshot(episode, 0, "current", {"current": {}})
            goals = estimated_snapshot(episode, 0, "current_goal_oracle", {"current": {}})
            preferred = estimated_snapshot(episode, 0, "current_preferred_oracle", {"current": {}})
        np.testing.assert_array_equal(goals["preferred"], estimated["preferred"])
        np.testing.assert_array_equal(preferred["humans"], estimated["humans"])
        for index in range(2):
            np.testing.assert_array_equal(goals["humans"][index][5:7], frame["snapshot"]["humans"][index][5:7])
            np.testing.assert_array_equal(preferred["preferred"][index], frame["snapshot"]["preferred"][index])

    def test_circle_birth_prior_uses_first_legal_sighting_not_hidden_spawn(self):
        frame = self.frame(0)
        frame["snapshot"].update(humans=[[0, 1, .2, .3, .1, 99, 99, 1, 0], [2, 0, .4, .5, .1, 99, 99, 1, 0]],
                                 preferred=[[99, 99], [99, 99]], tracks=[(0, 0, [], 0), (1, 1, [], 0)])
        result = estimated_snapshot({"frames": [frame]}, 0, "circle_birth_prior", {})
        np.testing.assert_array_equal(result["humans"][0][5:7], [0, -1])
        np.testing.assert_array_equal(result["humans"][1][5:7], [-2, 0])
        np.testing.assert_array_equal(result["preferred"], [[.2, .3], [.4, .5]])

    def test_secondary_selection_does_not_use_outcome_or_goal_labels(self):
        frame = self.frame(0)
        frame.update(visible=5)
        frame["robot"] = np.asarray([0, 0, 0, 0, .3, 0, 4, 1, 0.])
        frame["legal"] = np.asarray([[1., 0, 0, 0, .3, 0, 1, 1]] * 5)
        frame["snapshot"]["time"] = 2.
        episode = {"frames": [frame], "case": 60080, "split": "test", "geometry": "circle", "terminal": "collision"}
        first, _ = select_interaction_roots({"episodes": [episode]})
        episode["terminal"] = "reach_goal"
        episode["frames"][0]["snapshot"]["humans"][0][5:7] = [-999, -999]
        second, _ = select_interaction_roots({"episodes": [episode]})
        self.assertEqual(first[0][1], second[0][1])
        self.assertEqual(len(second), 1)


if __name__ == "__main__":
    unittest.main()
