"""Causal trigger, native contract parity and release/rearm behavior."""

import configparser
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, ObservableState
from shixu.commitment import ActionCommitmentPolicy, low_progress, native_blocked
from shixu.forecast import ForecastSuccessorValueModel
from shixu.observations import ObservedTracks
from shixu.policy import ValuePolicy
from experiments.online_action_commitment_v0 import CELLS, SEEDS, compare, trigger_ticks


class OnlineCommitmentTests(unittest.TestCase):
    def policy(self):
        torch.set_num_threads(1)
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1]/"shixu/default.ini")
        cfg.set("model", "representation", "tracks")
        return ActionCommitmentPolicy(ForecastSuccessorValueModel("cv", width=16, layers=1), cfg)

    def state(self, distance=5.):
        return ObservedTracks(FullState(0, 5.-distance, 0, 0, .3, 0, 5, 1, 0),
                              [ObservableState(2, 0, 0, 0, .3)], (0,), (True,), (0.,), 1)

    def scores(self, index):
        result = np.zeros(80)
        result[index] = 1.
        return result

    def test_exact_causal_threshold_and_needs_full_window(self):
        self.assertFalse(low_progress([5.]*8))
        self.assertTrue(low_progress([5.]*9))
        self.assertTrue(low_progress([5.]*8+[4.8]))
        self.assertFalse(low_progress([5.]*8+[4.79]))
        self.assertFalse(low_progress([1.]*9))

    def test_native_filter_fallback_is_preserved(self):
        np.testing.assert_array_equal(native_blocked([.1, .3], .2), [True, False])
        np.testing.assert_array_equal(native_blocked([.1, -.1], .2), [False, False])
        np.testing.assert_array_equal(native_blocked([-.1, .3], 0.), [False, False])

    def test_inactive_parity_including_rng_draw_and_history(self):
        custom = self.policy()
        parent = ValuePolicy(custom.model, custom.config)
        for tick in range(20):
            state = self.state(8.-.25*tick)
            rng = np.random.get_state()
            expected = parent.predict(state)
            expected_rng = np.random.get_state()
            np.random.set_state(rng)
            actual = custom.predict(state)
            np.testing.assert_array_equal(actual, expected)
            current_rng = np.random.get_state()
            for a, b in zip(expected_rng, current_rng):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(np.asarray(custom.history), np.asarray(parent.history))
            self.assertFalse(custom.last_decision["held"])

    def test_eight_held_steps_smooth_once_and_no_automatic_recommit(self):
        policy = self.policy()
        clearance = np.ones(80)
        with patch.object(policy, "score", return_value=self.scores(24)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, clearance)):
            for _ in range(8):
                policy.predict(self.state())
            previous = np.asarray(policy.last_action)
            held = []
            for _ in range(8):
                action = policy.predict(self.state())
                np.testing.assert_array_equal(action, .3*previous+.7*np.asarray(policy.action_space[24]))
                previous = np.asarray(action)
                held.append(policy.last_decision["held"])
            self.assertEqual(held, [True]*8)
            self.assertEqual(policy.last_decision["release"], "budget-completed")
            for _ in range(12):
                policy.predict(self.state())
                self.assertFalse(policy.last_decision["held"])
        self.assertEqual(policy.starts, 1)
        self.assertEqual(policy.held_steps, 8)
        self.assertEqual(len(policy.history), min(policy.length, 28))

    def test_safety_release_replans_same_step_and_does_not_restart(self):
        policy = self.policy()
        with patch.object(policy, "score", return_value=self.scores(24)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, np.ones(80))):
            for _ in range(9):
                policy.predict(self.state())
        clearance = np.ones(80)
        clearance[24] = .1
        with patch.object(policy, "score", return_value=self.scores(34)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, clearance)):
            action = policy.predict(self.state())
        self.assertEqual(policy.last_decision["release"], "safety-blocked")
        self.assertFalse(policy.last_decision["held"])
        self.assertEqual(policy.last_decision["selected_grid"], 34)
        self.assertEqual(policy.starts, 1)
        self.assertEqual(policy.held_steps, 1)
        self.assertEqual(len(policy.history), 10)

    def test_rearm_requires_condition_to_clear_then_recur(self):
        policy = self.policy()
        with patch.object(policy, "score", return_value=self.scores(24)), \
             patch.object(policy, "aligned_candidates", return_value=(None, None, np.ones(80))):
            for _ in range(16):
                policy.predict(self.state())
            policy.predict(self.state(4.))
            self.assertTrue(policy.armed)
            for _ in range(8):
                policy.predict(self.state(4.))
        self.assertEqual(policy.starts, 2)

    def test_reset_and_eval_only_contract(self):
        policy = self.policy()
        policy.predict(self.state())
        policy.reset()
        self.assertEqual(len(policy.history), 0)
        self.assertEqual(len(policy.distances), 0)
        self.assertTrue(policy.armed)
        self.assertIsNone(policy.last_action)
        self.assertIsNone(policy.active_grid)
        with self.assertRaises(ValueError):
            policy.predict(self.state(), epsilon=.1)
        policy.set_phase("train")
        with self.assertRaises(ValueError):
            policy.predict(self.state())

    def test_preflight_prefix_does_not_depend_on_future(self):
        trace = dict(goal_distance=np.full(20, 5.), blocked=np.zeros((20, 80), bool),
                     grid_action=np.full(20, 24))
        before = trigger_ticks(trace)
        trace["goal_distance"][9:] = 0.
        after = trigger_ticks(trace)
        self.assertIn(8, before)
        self.assertEqual(after, [8])
        trace["blocked"][8, 24] = True
        self.assertEqual(trigger_ticks(trace), [])

    def fixture_records(self):
        result = []
        for seed in SEEDS:
            for n, g in CELLS:
                for case in range(86000, 86004):
                    for arm in ("parent", "commitment"):
                        success = g == "circle" or arm == "commitment"
                        result.append(dict(seed=seed, people=n, geometry=g, case=case, arm=arm,
                            terminal="reach_goal" if success else "timeout",
                            behavior="success" if success else "low-progress-timeout",
                            initial_world_sha256=f"{n}-{g}-{case}", minimum_clearance=.3,
                            navigation_time=8. if success else 50.25, discounted_return=.8 if success else -.1,
                            starts=0 if arm == "parent" else 1, commitment_seconds=0. if arm == "parent" else 2.,
                            inference_seconds=.05))
        return result

    def test_frozen_gates_and_pairing_do_not_treat_seeds_as_distinct_cases(self):
        records = self.fixture_records()
        result = compare(records)
        self.assertEqual(result["verdict"], "A_FIXED_COMMITMENT_EFFECTIVE")
        self.assertEqual(result["square_qualified_rescue_cases"], list(range(86000, 86004)))
        self.assertTrue(all(result["gates"].values()))
        records[0]["initial_world_sha256"] = "different"
        with self.assertRaises(ValueError):
            compare(records)

    def test_rescues_cannot_hide_protection_damage(self):
        records = self.fixture_records()
        for row in records:
            if row["arm"] == "commitment" and row["geometry"] == "circle":
                row.update(terminal="collision", behavior="collision", minimum_clearance=-.1)
        result = compare(records)
        self.assertEqual(result["verdict"], "B_RESCUE_WITH_UNACCEPTABLE_DAMAGE")
        self.assertFalse(result["gates"]["circle_collision_guard"])


if __name__ == "__main__":
    unittest.main()
