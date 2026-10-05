"""Learned initiation must not change the Parent or existing release contract."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.commit_advantage_probe import fit, predict
from shixu.commit_advantage import AdvantageEstimator, CommitAdvantagePolicy
from shixu.policy import ValuePolicy
from tests import test_online_commitment as fixtures


class CommitAdvantageTests(unittest.TestCase):
    def fixture(self, gate):
        helper = fixtures.OnlineCommitmentTests()
        old = helper.policy()
        return CommitAdvantagePolicy(old.model, old.config, gate), helper

    def test_negative_gate_keeps_entire_native_control_contract(self):
        policy, helper = self.fixture(lambda x: -1.)
        parent = ValuePolicy(policy.model, policy.config)
        for _ in range(12):
            state = helper.state()
            rng = np.random.get_state()
            expected = parent.predict(state)
            expected_rng = np.random.get_state()
            np.random.set_state(rng)
            np.testing.assert_array_equal(policy.predict(state), expected)
            for a, b in zip(expected_rng, np.random.get_state()):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(policy.history, parent.history)
        self.assertEqual(policy.starts, 0)

    def fake_info(self, policy, index=24):
        return dict(index=index, features=np.zeros(146))

    def test_hold_exactly_eight_ticks_without_immediate_restart(self):
        policy, helper = self.fixture(lambda x: 1.)
        with patch("shixu.commit_advantage.extract", side_effect=lambda *a: self.fake_info(policy)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, np.ones(80))):
            for _ in range(16):
                policy.predict(helper.state())
            self.assertEqual(policy.held_steps, 8)
            self.assertEqual(policy.last_decision["release"], "budget-completed")
            for _ in range(3):
                policy.predict(helper.state())
                self.assertFalse(policy.last_decision["held"])
        self.assertEqual(policy.starts, 1)

    def test_all_unsafe_releases_and_uses_current_parent_on_same_tick(self):
        policy, helper = self.fixture(lambda x: 1.)
        with patch("shixu.commit_advantage.extract", side_effect=lambda *a: self.fake_info(policy)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, np.ones(80))):
            for _ in range(9):
                policy.predict(helper.state())
        with patch("shixu.commit_advantage.extract", side_effect=lambda *a: self.fake_info(policy, 34)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, np.full(80, .1))):
            policy.predict(helper.state())
        self.assertEqual(policy.last_decision["release"], "all-unsafe")
        self.assertEqual(policy.last_decision["selected_grid"], 34)
        self.assertFalse(policy.last_decision["held"])

    def test_estimator_matches_frozen_offline_prediction(self):
        torch.set_num_threads(1)
        rng = np.random.RandomState(8)
        rows = [dict(seed=419, case=i, features=rng.normal(size=146).tolist(), y=float(rng.normal())) for i in range(12)]
        for kind in ("linear", "mlp"):
            fitted = fit(rows[:8], rows[8:10], kind, 5)
            model = AdvantageEstimator(fitted)
            np.testing.assert_allclose([model(r["features"]) for r in rows[10:]], predict(fitted, rows[10:]), rtol=1e-6, atol=1e-7)

    def test_reset_clears_causal_state_not_the_gate(self):
        policy, helper = self.fixture(lambda x: -1.)
        policy.predict(helper.state())
        policy.reset()
        self.assertEqual(len(policy.history), 0)
        self.assertEqual(len(policy.distances), 0)
        self.assertEqual(len(policy.grids), 0)
        self.assertEqual(len(policy.commands), 0)
        self.assertTrue(policy.armed)
        self.assertIsNone(policy.last_action)


if __name__ == "__main__":
    unittest.main()
