"""Goal interventions and history-order controls change only their named factor."""

import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.circle_coupling import PROTOCOL, assign_goals, bootstrap_difference, controlled_design


class CouplingTests(unittest.TestCase):
    def row(self):
        return {"context": np.arange(49, dtype=float), "birth": np.asarray([3., 4.]),
                "statistics": np.arange(18, dtype=float), "history": np.arange(24 * 8, dtype=float).reshape(24, 8),
                "target": np.ones(8), "regime": "circle"}

    def test_goal_ownership_preserves_goal_set_and_other_state(self):
        humans = []
        for index in range(5):
            h = SimpleNamespace(px=index, py=1, vx=0, vy=0, gx=-index, gy=-1)
            h.get_goal_position = lambda human=h: (human.gx, human.gy)
            humans.append(h)
        env = SimpleNamespace(humans=humans)
        before = [(h.px, h.py, h.vx, h.vy) for h in humans]
        original, assignment = assign_goals(env, "circle_permuted", 72000)
        self.assertTrue(np.all(assignment != np.arange(5)))
        np.testing.assert_array_equal([h.get_goal_position() for h in humans], original[assignment])
        np.testing.assert_array_equal(sorted(h.get_goal_position() for h in humans), sorted(map(tuple, original)))
        self.assertEqual(before, [(h.px, h.py, h.vx, h.vy) for h in humans])

    def test_current_does_not_read_birth_or_history(self):
        row = self.row()
        before = controlled_design([row], "current")
        row["birth"][:] = -999
        row["history"][:] = -999
        np.testing.assert_array_equal(before, controlled_design([row], "current"))
        self.assertEqual(before.shape, (1, 235))

    def test_birth_is_only_two_extra_inputs(self):
        row = self.row()
        feature = controlled_design([row], "birth")
        np.testing.assert_array_equal(feature[0, :49], row["context"])
        np.testing.assert_array_equal(feature[0, 49:51], row["birth"])
        np.testing.assert_array_equal(feature[0, 51:], np.zeros(184))

    def test_bag_invariant_to_order_but_sequence_is_not(self):
        row, changed = self.row(), self.row()
        changed["history"][:-1] = changed["history"][:-1][::-1]
        np.testing.assert_array_equal(controlled_design([row], "birth_bag24"),
                                      controlled_design([changed], "birth_bag24"))
        self.assertFalse(np.array_equal(controlled_design([row], "birth_history24"),
                                        controlled_design([changed], "birth_history24")))

    def test_hidden_labels_and_regime_are_not_features(self):
        row, changed = self.row(), self.row()
        changed["target"][:] = -999
        changed["regime"] = "square"
        for mode in json.loads(PROTOCOL.read_text())["modes"]:
            np.testing.assert_array_equal(controlled_design([row], mode), controlled_design([changed], mode))

    def test_short_window_keeps_only_two_past_observations(self):
        row = self.row()
        feature = controlled_design([row], "birth_history3")
        np.testing.assert_array_equal(feature[0, 51:51 + 168], np.zeros(168))
        np.testing.assert_array_equal(feature[0, -16:], row["history"][-3:-1].ravel())

    def test_every_arm_same_coefficient_layout(self):
        row = self.row()
        for mode in json.loads(PROTOCOL.read_text())["modes"]:
            self.assertEqual(controlled_design([copy.deepcopy(row)], mode).shape, (1, 235))

    def test_bootstrap_is_case_paired_not_frame_paired(self):
        result = bootstrap_difference(np.asarray([1., 1., 2.]), np.zeros(3), [1, 1, 2], 100, .95)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["difference"], 1.5)

    def test_fixed_fit_validation_test_are_disjoint(self):
        protocol = json.loads(PROTOCOL.read_text())
        sets = [set(range(split["start"], split["start"] + split["count"])) for split in protocol["splits"]]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


if __name__ == "__main__":
    unittest.main()
