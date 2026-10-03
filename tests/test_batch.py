"""The batched consumer must preserve the scalar successor implementation."""

import configparser
from pathlib import Path
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, ObservableState
from shixu.features import encode_aligned, window
from shixu.model import MemoryValueModel, OrderedValueModel
from shixu.observations import TrackedState
from shixu.policy import ValuePolicy, successor


class BatchParityTests(unittest.TestCase):
    def test_read_gate_probe_restores_model_and_history(self):
        from experiments.temporal_memory import gate_probe
        torch.set_num_threads(1)
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        cfg.set("model", "representation", "aligned")
        robot = FullState(.2, -1.4, .1, .2, .3, 0, 4, 1, 0)
        humans = [ObservableState(1 + i, 1, -.2, .3, .3) for i in range(5)]
        state = TrackedState(robot, humans, tuple(range(5)))
        policy = ValuePolicy(MemoryValueModel("kda", "evidence", 32, 1), cfg)
        policy.history.extend([encode_aligned(state)] * 3)
        weights = {k: v.clone() for k, v in policy.model.state_dict().items()}
        reference = policy.score(state)
        native, stats = gate_probe(policy, state)
        np.testing.assert_array_equal(native, reference)
        self.assertGreater(stats["gate_mean"], 0)
        for mode in ("half_gate", "constant_gate", "ungated", "zero_evidence", "validity_only", "shuffled_motion"):
            scores, _ = gate_probe(policy, state, mode, torch.full((32,), .5, dtype=torch.float64), torch.zeros(5, 3))
            self.assertTrue(np.isfinite(scores).all())
            np.testing.assert_array_equal(policy.score(state), reference)
        self.assertEqual(len(policy.history), 3)
        self.assertFalse(policy.model.evidence_gate._forward_hooks)
        self.assertFalse(policy.model.evidence_gate._forward_pre_hooks)
        for key, value in policy.model.state_dict().items():
            torch.testing.assert_close(value, weights[key])

    def test_memory_shared_prefix_preserves_candidate_consumer(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        cfg.set("model", "representation", "aligned")
        robot = FullState(0.2, -1.4, 0.1, 0.2, 0.3, 0, 4, 1, 0)
        humans = [ObservableState(1 + i, 1, -.2, .3, .3) for i in range(5)]
        state = TrackedState(robot, humans, tuple(range(5)))
        for kind, readout in (("gru", "evidence"), ("kda", "full"), ("kda", "read"),
                              ("kda", "gate"), ("kda", "evidence"),
                              ("gdn2", "read"), ("gdn2", "revision")):
            policy = ValuePolicy(MemoryValueModel(kind, readout, 32, 1), cfg)
            policy.set_phase("train")
            policy.history.extend([encode_aligned(state)] * 3)
            sequences = [window(list(policy.history) + [encode_aligned(state), encode_aligned(successor(state, a, .25))], 24)
                         for a in policy.action_space]
            with torch.inference_mode():
                values = policy.model(torch.as_tensor(np.array(sequences))).numpy()
            rewards = [policy.immediate_reward(state, successor(state, a, .25), a)[0] for a in policy.action_space]
            expected = np.array(rewards) + (policy.gamma * torch.as_tensor(values)).numpy()
            expected[0] -= 1e-3
            actual = policy.score(state)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
            self.assertEqual(np.argmax(actual), np.argmax(expected))
            self.assertEqual(len(policy.history), 3)

    def test_candidates_and_scores_match_scalar_reference(self):
        torch.set_num_threads(1)
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        cfg.set("model", "representation", "aligned")
        rng = np.random.default_rng(91)
        for order in ("scene", "actor"):
            policy = ValuePolicy(OrderedValueModel(order, width=32, layers=1), cfg)
            policy.model.eval()
            for count in (5, 10, 20):
                robot = FullState(0.2, -1.4, 0.1, 0.2, 0.3, 0, 4, 1, 0)
                humans = [ObservableState(*rng.normal(size=4), 0.3) for _ in range(count)]
                state = TrackedState(robot, humans, tuple(range(count)))
                tokens, rewards, clearances = policy.aligned_candidates(state)
                scalar_tokens, scalar_rewards, scalar_clearances = [], [], []
                for action in policy.action_space:
                    future = successor(state, action, policy.time_step)
                    scalar_tokens.append(encode_aligned(future))
                    reward, clearance = policy.immediate_reward(state, future, action)
                    scalar_rewards.append(reward)
                    scalar_clearances.append(clearance)
                np.testing.assert_array_equal(tokens, scalar_tokens)
                np.testing.assert_allclose(rewards, scalar_rewards, atol=1e-15, rtol=1e-15)
                np.testing.assert_allclose(clearances, scalar_clearances, atol=1e-15, rtol=1e-15)
                for length in (0, 3, 24):
                    policy.history.clear()
                    policy.history.extend([encode_aligned(state)] * length)
                    sequences = [window(list(policy.history) + [encode_aligned(state), token], policy.length)
                                 for token in scalar_tokens]
                    with torch.inference_mode():
                        values = policy.model(torch.as_tensor(np.array(sequences))).numpy()
                    raw = np.array(scalar_rewards) + (policy.gamma * torch.as_tensor(values)).numpy()
                    expected = raw.copy()
                    expected[0] -= 1e-3
                    policy.set_phase("train")
                    actual = policy.score(state)
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
                    self.assertEqual(np.argmax(actual), np.argmax(expected))
                    policy.set_phase("test")
                    distance = np.array(scalar_clearances)
                    expected = raw.copy()
                    if np.any(distance >= 0.2):
                        expected[distance < 0.2] = -1e9
                    expected -= 0.8 * np.maximum(0.2 - distance, 0)
                    expected[0] -= 1e-3
                    np.testing.assert_allclose(policy.score(state), expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
