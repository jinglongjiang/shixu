"""State correction must not invent measurements or change actor lifecycle."""

import copy
import unittest

import numpy as np

from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.decision_state import history_correction, matched_cases, truth_correction
from shixu.observations import ObservedTracks


class DecisionStateTests(unittest.TestCase):
    def setUp(self):
        self.robot = FullState(0, 0, 0, 0, .3, 0, 4, 1, 0)
        self.state = ObservedTracks(self.robot, [ObservableState(1, 0, .2, 0, .3)],
                                   (0,), (False,), (.5,), 1)
        self.history = np.zeros((24, 2, 13), np.float32)
        self.history[-5, 1, [3, 10, 12]] = [0., 1., 1.]
        self.history[-3, 1, [3, 10, 12]] = [.2, 1., 1.]
        self.history[-1, 1, [3, 10, 12]] = [99., 0., 1.]

    def test_correction_uses_only_measured_velocity_and_elapsed_time(self):
        out = history_correction(self.state, self.history)
        np.testing.assert_allclose(out.human_states[0].position, [1.05, 0], atol=1e-7)
        np.testing.assert_allclose(out.human_states[0].velocity, [.4, 0], atol=1e-7)
        self.assertEqual(out.observed, (False,))
        self.assertEqual(out.ages, (.5,))

    def test_visible_actor_is_never_rewritten(self):
        visible = self.state._replace(observed=(True,), ages=(0.,))
        out = history_correction(visible, self.history)
        self.assertIs(out.human_states[0], visible.human_states[0])
        truth = truth_correction(visible, {0: np.array([99., 99., 99., 99., .3])})
        self.assertIs(truth.human_states[0], visible.human_states[0])

    def test_one_measurement_cannot_invent_an_acceleration(self):
        self.history[-5, 1] = 0
        out = history_correction(self.state, self.history)
        self.assertIs(out.human_states[0], self.state.human_states[0])

    def test_each_actor_uses_its_own_association_slot(self):
        history = np.pad(self.history, ((0, 0), (0, 1), (0, 0)))
        state = self.state._replace(human_states=[self.state.human_states[0], ObservableState(0, 2, 0, 0, .3)],
                                   track_ids=(0, 1), observed=(False, False), ages=(.5, .5), track_count=2)
        expected = history_correction(state, history)
        history[:, 2, 3:5] = 123
        history[:, 2, 10:13] = 1
        actual = history_correction(state, history)
        np.testing.assert_array_equal(actual.human_states[0].to_array(), expected.human_states[0].to_array())

    def test_queries_do_not_mutate_evidence_or_resurrect_expired_actor(self):
        before = self.history.copy()
        old_human = copy.deepcopy(self.state.human_states[0].to_array())
        first = history_correction(self.state, self.history)
        second = history_correction(self.state, self.history)
        np.testing.assert_array_equal(first.human_states[0].to_array(), second.human_states[0].to_array())
        np.testing.assert_array_equal(self.history, before)
        np.testing.assert_array_equal(self.state.human_states[0].to_array(), old_human)
        empty = self.state._replace(human_states=[], track_ids=(), ages=(), observed=())
        self.assertEqual(history_correction(empty, self.history).human_states, [])
        self.assertEqual(truth_correction(empty, {0: np.zeros(5)}).human_states, [])

    def test_success_matches_are_unique_and_in_same_cell(self):
        rows = [dict(people=5, geometry="square", case=c, terminal=t)
                for c, t in ((1, "timeout"), (3, "collision"), (0, "reach_goal"), (2, "reach_goal"))]
        selected = matched_cases(rows)
        self.assertEqual([r["case"] for r in selected], [1, 0, 3, 2])
        self.assertEqual([r["group"] for r in selected], ["failure", "success_guard", "failure", "success_guard"])


if __name__ == "__main__":
    unittest.main()
