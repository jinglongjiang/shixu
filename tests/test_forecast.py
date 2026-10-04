"""The new mechanism has an explicit causal prediction/control boundary."""

import copy
import unittest

import numpy as np
import torch

from shixu.forecast import ForecastReplay, ForecastValueModel
from shixu.replay import Replay


class ForecastTests(unittest.TestCase):
    def history(self, people=3):
        rows = torch.zeros(2, 24, 1 + people, 13)
        rows[:, :, 0, :9] = torch.tensor([0., -3., 0., .3, .3, 0., 3., 1., 0.])
        rows[:, :, 1:, :2] = torch.randn(2, 24, people, 2)
        rows[:, :, 1:, 3:5] = torch.tensor([.2, -.3])
        rows[:, :, 1:, 6] = .3
        rows[:, :, 1:, 10] = rows[:, :, 1:, 12] = 1
        return rows

    def test_all_arms_initially_produce_same_cv_forecasts_and_values(self):
        history = self.history()
        forecasts, values = [], []
        for kind in ("cv", "current", "gru", "kda"):
            torch.manual_seed(3)
            model = ForecastValueModel(kind, 16, 2).eval()
            forecasts.append(model.forecast_positions(history))
            values.append(model(history))
        for predicted, value in zip(forecasts[1:], values[1:]):
            torch.testing.assert_close(predicted, forecasts[0])
            torch.testing.assert_close(value, values[0])

    def test_forecast_is_candidate_independent_but_consumer_is_not(self):
        history = self.history()
        model = ForecastValueModel("kda", 16, 2).eval()
        query = history[:, -1:].repeat(1, 4, 1, 1)
        query[:, 1:, 0, 0] += .4
        query[:, 1:, 0, 2] += .5
        score = model.score_candidates(history, query)
        self.assertEqual(tuple(score.shape), (2, 4))
        self.assertFalse(torch.allclose(score[:, 0], score[:, 1]))
        changed = query.clone()
        changed[:, :, 1:] = -999
        torch.testing.assert_close(score, model.score_candidates(history, changed))

    def test_forecast_change_reaches_candidate_values_without_hidden_bypass(self):
        history = self.history()
        model = ForecastValueModel("current", 16).eval()
        query = history[:, -1:].repeat(1, 4, 1, 1)
        positions = model.forecast_positions(history)
        before = model._value(history, query, positions, 1)
        positions[:, :, 2:, 0] += 1
        after = model._value(history, query, positions, 1)
        self.assertFalse(torch.allclose(before, after))

    def test_actor_permutation(self):
        history = self.history()
        model = ForecastValueModel("gru", 16).eval()
        with torch.no_grad():
            model.decoder.weight.normal_(0, .01)
        original = model.forecast_positions(history)
        changed = history[:, :, [0, 3, 1, 2]]
        torch.testing.assert_close(model.forecast_positions(changed), original[:, [2, 0, 1]])

    def test_current_forecaster_reads_legal_neighbour_information(self):
        history = self.history()
        model = ForecastValueModel("current", 16).eval()
        with torch.no_grad():
            model.decoder.weight.normal_(0, .1)
        before = model.forecast_positions(history)[:, 0]
        history[:, :, 2:, :2] += 2
        after = model.forecast_positions(history)[:, 0]
        self.assertFalse(torch.allclose(before, after))

    def test_hidden_history_does_not_write_memory(self):
        history = self.history()
        history[:, 5:9, 1:, 10] = 0
        changed = history.clone()
        changed[:, 5:9, 1:, :9] = 999
        model = ForecastValueModel("kda", 16).eval()
        with torch.no_grad():
            model.decoder.weight.normal_(0, .01)
        torch.testing.assert_close(model.forecast_positions(history), model.forecast_positions(changed))

    def test_native_query_offsets(self):
        model = ForecastValueModel("cv", 16)
        torch.testing.assert_close(model.times[[0, 1, 2, 4, 8]], torch.tensor([0., .25, .5, 1., 2.]))
        torch.testing.assert_close(model.times[[1, 2, 3, 5, 9]], torch.tensor([.25, .5, .75, 1.25, 2.25]))

    def test_value_gradient_reaches_predictor(self):
        history = self.history()
        model = ForecastValueModel("gru", 16)
        model(history).sum().backward()
        self.assertGreater(float(model.decoder.weight.grad.norm()), 0)

    def test_replay_matches_old_samples_and_future_is_label_only(self):
        frames = self.history().numpy()[0]
        episode = {"tokens": list(frames), "rewards": [0.] * len(frames)}
        ordinary = Replay(length=24, left_pad="zero")
        replay = ForecastReplay(length=24, left_pad="zero")
        ordinary.add(episode)
        replay.add(episode)
        before, labels = ordinary.sample(3, "cpu", np.random.default_rng(41))
        current, targets, future, valid = replay.sample_with_forecasts(3, "cpu", np.random.default_rng(41))
        torch.testing.assert_close(before, current)
        torch.testing.assert_close(labels, targets)
        self.assertEqual(tuple(future.shape), (3, 3, 9, 2))
        self.assertEqual(tuple(valid.shape), (3, 3, 9))

    def test_replay_does_not_label_hidden_or_newborn_actors(self):
        frames = self.history().numpy()[0]
        frames[:, 1:, 10] = 0
        episode = {"tokens": list(frames), "rewards": [0.] * len(frames)}
        replay = ForecastReplay(length=24, left_pad="zero")
        replay.add(episode)
        _, _, _, valid = replay.sample_with_forecasts(16, "cpu", np.random.default_rng(4))
        self.assertFalse(bool(valid.any()))

    def test_reload_and_empty_scene(self):
        history = self.history(1)
        history[:, :, 1:] = 0
        model = ForecastValueModel("kda", 16).eval()
        self.assertTrue(bool(torch.isfinite(model(history)).all()))
        restored = ForecastValueModel("kda", 16).eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        torch.testing.assert_close(model(history), restored(history))


if __name__ == "__main__":
    unittest.main()
