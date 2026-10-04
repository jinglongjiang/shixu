"""Diagnostic controls must preserve reward, identity and root selection."""

import configparser
import unittest

import numpy as np
import torch

from experiments.forecast_evidence import choose_primary, execute, stats
from shixu.forecast import ForecastSuccessorValueModel


class EvidenceTests(unittest.TestCase):
    def test_selection_is_outcome_blind_and_case_balanced(self):
        rows = [dict(episode=0, tick=4, current_clearance=.01),
                dict(episode=0, tick=8, current_clearance=.7),
                dict(episode=0, tick=12, current_clearance=.1),
                dict(episode=1, tick=8, current_clearance=2.),
                dict(episode=1, tick=12, current_clearance=1.)]
        self.assertEqual(choose_primary(rows), [1, 4])

    def test_metric_uses_training_loss_units_and_valid_labels(self):
        errors = np.ones((2, 9))
        squared = np.full((2, 9), 8.)
        valid = np.ones((2, 9), bool)
        valid[1] = False
        result = stats(errors, squared, valid)
        self.assertEqual(result["targets"], 9)
        self.assertEqual(result["velocity_squared_sum"] / (2 * result["targets"]), 4.)
        self.assertEqual(result["0.25"]["targets"], 1)

    def test_smoothing_reuses_common_previous_executed_action(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"eval_protocol": {"action_smoothing": ".3"}})
        np.testing.assert_allclose(execute([1., 0.], [0., 1.], cfg), [.7, .3])
        np.testing.assert_array_equal(execute([1., 0.], None, cfg), [1., 0.])

    def test_truth_swap_reaches_fixed_cv_critic_only_via_positions(self):
        torch.manual_seed(1)
        model = ForecastSuccessorValueModel("cv", 16, 1).eval()
        history = torch.zeros(1, 24, 3, 13)
        history[:, :, 0, :9] = torch.tensor([0., -3., 0., .5, .3, 0., 3., 1., 0.])
        history[:, :, 1:, :2] = torch.tensor([[.4, .9], [-.8, 1.5]])
        history[:, :, 1:, 6] = .3
        history[:, :, 1:, 10] = history[:, :, 1:, 12] = 1
        queries = history[:, -1:].repeat(1, 2, 1, 1)
        queries[:, 1, 0, 0] += .2
        positions = model.forecast_positions(history)
        original = queries.clone()
        torch.testing.assert_close(model.corrected_queries(history, queries, positions), queries)
        positions[:, :, 1, 0] += .3
        corrected = model.corrected_queries(history, queries, positions)
        torch.testing.assert_close(queries, original)
        torch.testing.assert_close(corrected[..., 0, :], queries[..., 0, :])
        torch.testing.assert_close(corrected[..., 1:, 3:7], queries[..., 1:, 3:7])
        torch.testing.assert_close(corrected[..., 1:, 9:], queries[..., 1:, 9:])
        with torch.inference_mode():
            before = model.critic.score_candidates(history[:, 1:], queries)
            after = model.critic.score_candidates(history[:, 1:], corrected)
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
