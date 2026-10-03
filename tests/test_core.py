import configparser
from pathlib import Path
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
from shixu.features import encode_state, window
from shixu.model import ValueModel
from shixu.policy import ValuePolicy, actions, successor
from shixu.replay import Replay, returns
from shixu.runner import environment, orca_teacher, run_episode


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = configparser.ConfigParser()
        self.cfg.read(str(ROOT / "shixu/default.ini"))
        self.model = ValueModel(width=32, layers=1)
        self.policy = ValuePolicy(self.model, self.cfg)
        self.state = JointState(FullState(0, -4, 0, 0, 0.3, 0, 4, 1, np.pi / 2),
                                [ObservableState(1, 0, -0.3, 0.2, 0.3)])

    def test_action_support(self):
        grid = np.array([[a.vx, a.vy] for a in actions(self.cfg)])
        self.assertEqual(grid.shape, (80, 2))
        self.assertGreater(np.linalg.norm(grid, axis=1).min(), 0)
        self.assertAlmostEqual(np.linalg.norm(grid, axis=1).max(), 1)

    def test_successor_keeps_observed_information_only(self):
        future = successor(self.state, actions(self.cfg)[0], 0.25)
        self.assertAlmostEqual(future.human_states[0].px, 0.925)
        self.assertFalse(hasattr(future.human_states[0], "gx"))

    def test_token_shape(self):
        tokens = encode_state(self.state)
        self.assertEqual(tokens.shape, (8, 13))
        self.assertTrue(np.isfinite(tokens).all())

    def test_history_left_padding(self):
        frames = [np.ones((8, 13)), np.ones((8, 13)) * 2]
        result = window(frames, 4)
        np.testing.assert_array_equal(result[:, 0, 0], [1, 1, 1, 2])

    def test_history_is_past_only(self):
        frames = [np.ones((8, 13)) * index for index in range(10)]
        np.testing.assert_array_equal(window(frames[:4], 3)[:, 0, 0], [1, 2, 3])

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            window([], 24)

    def test_value_shape(self):
        self.assertEqual(self.model(torch.zeros(3, 24, 8, 13)).shape, (3,))

    def test_invalid_substrate_rejected(self):
        with self.assertRaises(ValueError):
            ValueModel(backbone="imaginary")

    def test_no_auxiliary_heads(self):
        self.assertFalse(hasattr(self.model, "q_head"))
        self.assertFalse(hasattr(self.model, "future_pred_head"))

    def test_score_is_not_a_history_write(self):
        scores = self.policy.score(self.state)
        self.assertEqual(scores.shape, (80,))
        self.assertEqual(len(self.policy.history), 0)

    def test_one_history_write_per_control(self):
        self.policy.predict(self.state)
        self.assertEqual(len(self.policy.history), 1)

    def test_exploration_does_not_skip_observation(self):
        self.policy.set_phase("train")
        self.policy.predict(self.state, epsilon=1)
        self.assertEqual(len(self.policy.history), 1)

    def test_reset_clears_episode_state(self):
        self.policy.predict(self.state)
        self.policy.reset()
        self.assertEqual(len(self.policy.history), 0)
        self.assertIsNone(self.policy.last_action)

    def test_mc_returns(self):
        np.testing.assert_allclose(returns([0, 0, 1], 0.9), [0.81, 0.9, 1])

    def test_replay_capacity(self):
        buffer = Replay(capacity=2, length=3)
        buffer.add({"tokens": [np.ones((8, 13)) * index for index in range(4)], "rewards": [0, 0, 0, 1]})
        self.assertEqual(len(buffer.samples), 2)
        x, y = buffer.sample(5, "cpu", np.random.default_rng(0))
        self.assertEqual(x.shape, (5, 3, 8, 13))
        self.assertTrue(torch.isfinite(y).all())

    def test_replay_does_not_cross_episode_boundaries(self):
        buffer = Replay(length=4)
        for value in (1, 10):
            buffer.add({"tokens": [np.full((8, 13), value)], "rewards": [1]})
        x, _ = buffer.sample(20, "cpu", np.random.default_rng(0))
        self.assertTrue(all(torch.unique(sample).numel() == 1 for sample in x))

    def test_value_loss_has_finite_gradients_without_parameter_updates(self):
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        prediction = self.model(torch.zeros(2, 24, 8, 13))
        torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction)).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, self.model.parameters())))

    def test_native_one_step(self):
        env = environment(self.cfg, self.policy)
        env.reset(options={"test_case": 0})
        _, reward, done, truncated, info = env.step(actions(self.cfg)[0])
        self.assertTrue(np.isfinite(reward))
        self.assertIn("event", info)
        self.assertFalse(done or truncated)

    def test_legal_actor_observations_retained(self):
        env = environment(self.cfg, self.policy)
        record = run_episode(env, self.policy, 0, teacher=orca_teacher(self.cfg))
        self.assertEqual(len(record["tokens"]), len(record["rewards"]))
        humans = record["observations"][0]["humans"]
        self.assertEqual(len(humans), 5)
        self.assertEqual(set(humans[0]), {"track_id", "state"})
        self.assertEqual(len(humans[0]["state"]), 5)


if __name__ == "__main__":
    unittest.main()
