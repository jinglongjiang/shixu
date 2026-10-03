"""The batched consumer must preserve the scalar successor implementation."""

import configparser
from pathlib import Path
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, ObservableState
from shixu.features import encode_aligned, window
from shixu.model import OrderedValueModel
from shixu.observations import TrackedState
from shixu.policy import ValuePolicy, successor


class BatchParityTests(unittest.TestCase):
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
