import unittest

import numpy as np

from crowd_sim.envs.utils.state import FullState, ObservableState
from shixu.features import encode_aligned
from shixu.observations import TrackedState
from shixu.policy import successor
from crowd_sim.envs.utils.action import ActionXY


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.robot = FullState(0, -4, 0, 0, .3, 0, 4, 1, 0)
        self.people = [ObservableState(1, 0, -.2, .1, .3), ObservableState(-1, 0, .1, -.2, .3)]

    def test_list_permutation_does_not_change_slots(self):
        original = encode_aligned(TrackedState(self.robot, self.people, (0, 1)))
        reordered = encode_aligned(TrackedState(self.robot, self.people[::-1], (1, 0)))
        np.testing.assert_array_equal(original, reordered)

    def test_missing_track_is_not_reassigned(self):
        frame = encode_aligned(TrackedState(self.robot, self.people[1:], (1,)))
        self.assertEqual(frame[3, 12], 0)
        self.assertEqual(frame[4, 12], 1)

    def test_reentry_uses_same_slot(self):
        first = encode_aligned(TrackedState(self.robot, self.people, (0, 1)))
        encode_aligned(TrackedState(self.robot, self.people[1:], (1,)))
        last = encode_aligned(TrackedState(self.robot, self.people, (0, 1)))
        np.testing.assert_array_equal(first, last)

    def test_missing_association_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_aligned(TrackedState(self.robot, self.people, (0, 0)))

    def test_counterfactual_successor_preserves_keys(self):
        state = TrackedState(self.robot, self.people, (0, 1))
        future = successor(state, ActionXY(.1, .2), .25)
        self.assertEqual(future.track_ids, state.track_ids)
        self.assertFalse(hasattr(future.human_states[0], "gx"))


if __name__ == "__main__":
    unittest.main()
