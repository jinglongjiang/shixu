"""Native-critic parity and physical/value gradient boundaries."""

import copy
import unittest

import numpy as np
import torch

from shixu.forecast import ForecastReplay, ForecastSuccessorValueModel
from shixu.model import OcclusionValueModel
from shixu.replay import Replay
from shixu.training import update
from tests import test_forecast


class ForecastBridgeTests(unittest.TestCase):
    history = test_forecast.ForecastTests.history

    def test_cv_and_zero_residual_preserve_exact_native_scores(self):
        history = self.history()
        query = history[:, -1:].repeat(1, 80, 1, 1)
        query[:, :, 0, 0] += torch.linspace(-.25, .25, 80)
        torch.manual_seed(11)
        parent = OcclusionValueModel("gru", 16, 2, interaction_order="write").eval()
        expected = parent.score_candidates(history[:, 1:], query)
        for kind in ("cv", "current", "gru", "kda"):
            torch.manual_seed(11)
            model = ForecastSuccessorValueModel(kind, 16, 2).eval()
            torch.testing.assert_close(model.critic(history), parent(history), atol=0, rtol=0)
            torch.testing.assert_close(model.score_candidates(history, query), expected, atol=0, rtol=0)

    def test_forecast_changes_queries_not_real_history_or_native_metadata(self):
        history = self.history()
        original = history.clone()
        query = history[:, -1:].repeat(1, 4, 1, 1)
        model = ForecastSuccessorValueModel("current", 16).eval()
        positions = model.forecast_positions(history)
        positions[:, :, 1, 0] += .5
        corrected = model.corrected_queries(history, query, positions)
        torch.testing.assert_close(history, original, atol=0, rtol=0)
        torch.testing.assert_close(corrected[:, :, 0], query[:, :, 0], atol=0, rtol=0)
        torch.testing.assert_close(corrected[:, :, 1:, [3, 4, 5, 6, 9, 10, 11, 12]],
                                   query[:, :, 1:, [3, 4, 5, 6, 9, 10, 11, 12]], atol=0, rtol=0)
        torch.testing.assert_close(corrected[:, :, 1:, 0], query[:, :, 1:, 0] + .5)
        self.assertFalse(torch.allclose(model._value(history, query, positions, 1),
                                       model.score_candidates(history, query)))

    def test_forecast_gradient_is_independent_of_value_gradient(self):
        history = self.history()
        model = ForecastSuccessorValueModel("kda", 16)
        model(history).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.critic.parameters()))
        model.zero_grad(set_to_none=True)
        positions = model.forecast_positions(history)
        future = torch.randn(2, 3, 9, 2)
        valid = torch.ones(2, 3, 9, dtype=torch.bool)
        model.predictor.motion_loss(history, positions, future, valid).backward()
        self.assertGreater(float(model.predictor.decoder.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))

    def test_shared_replay_auxiliary_does_not_change_critic_updates(self):
        frames = self.history().numpy()[0]
        episode = {"tokens": list(frames), "rewards": [0.] * len(frames)}
        ordinary = Replay(length=24, left_pad="zero")
        auxiliary = ForecastReplay(length=24, left_pad="zero")
        ordinary.add(episode)
        auxiliary.add(episode)
        model = ForecastSuccessorValueModel("gru", 16)
        parent = copy.deepcopy(model.critic)
        a = torch.optim.AdamW(parent.parameters(), lr=1e-4)
        b = torch.optim.AdamW(model.parameters(), lr=1e-4)
        first, second = np.random.default_rng(19), np.random.default_rng(19)
        for _ in range(3):
            update(parent, ordinary, a, 4, "cpu", first)
            update(model, auxiliary, b, 4, "cpu", second)
        for name, expected in parent.state_dict().items():
            torch.testing.assert_close(model.critic.state_dict()[name], expected, atol=0, rtol=0)

    def test_reload_and_unknown_padding(self):
        history = self.history()
        for kind in ("cv", "current", "gru", "kda"):
            model = ForecastSuccessorValueModel(kind, 16).eval()
            padded = torch.nn.functional.pad(history, (0, 0, 0, 4))
            torch.testing.assert_close(model(history), model(padded))
            restored = ForecastSuccessorValueModel(kind, 16).eval()
            restored.load_state_dict(copy.deepcopy(model.state_dict()))
            torch.testing.assert_close(model(history), restored(history), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
