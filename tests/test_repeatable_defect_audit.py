"""Method-blind selection, command parity and localized-evidence contracts."""

import configparser
from pathlib import Path
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.action_ranking_attribution import stages
from experiments.repeatable_defect_audit import attribution, failure_label, low_progress_start, root_tick, selected_groups
from shixu.forecast import ForecastSuccessorValueModel
from shixu.observations import ObservedTracks
from shixu.policy import ValuePolicy


class RepeatableDefectAuditTests(unittest.TestCase):
    def test_low_progress_is_not_a_feasibility_claim(self):
        self.assertEqual(low_progress_start([5.] * 41), 8)
        self.assertIsNone(low_progress_start([5.] * 40))
        self.assertIsNone(low_progress_start(np.linspace(8., 0., 60)))
        self.assertEqual(failure_label("timeout", [5.] * 41), "low-progress-timeout")
        self.assertEqual(failure_label("collision", [5.] * 41), "collision")
        self.assertEqual(failure_label("reach_goal", [5.] * 41), "success")

    def test_seeds_are_not_independent_case_ids(self):
        records = [dict(behavior="collision", people=20, geometry="square", seed=s, case=c)
                   for s in (419, 443, 467, 491) for c in (80000, 80001)]
        self.assertEqual(selected_groups(records), [])
        records.append(dict(behavior="collision", people=20, geometry="square", seed=467, case=80002))
        groups = selected_groups(records)
        self.assertEqual(groups[0]["cases"], 3)
        self.assertEqual(groups[0]["multiseed_cases"], 2)
        self.assertEqual([r["seed"] for r in groups[0]["selected"]], [419, 419, 467])

    def test_root_uses_fixed_legal_rule(self):
        trace = dict(actions=np.zeros((80, 2)), goal_distance=np.linspace(10., 3., 80),
                     legal_clearance=np.ones(80))
        trace["legal_clearance"][49] = .7
        self.assertEqual(root_tick(dict(behavior="collision"), trace), 49)
        as_lists = {k: v.tolist() for k, v in trace.items()}
        self.assertEqual(root_tick(dict(behavior="collision"), as_lists), 49)
        trace["legal_clearance"][49] = 1.
        self.assertEqual(root_tick(dict(behavior="collision"), trace), 40)

    def test_fragile_rescue_is_reported_but_not_qualified(self):
        result = attribution({}, {}, [dict(terminal="collision", minimum_clearance=-.1),
                                      dict(terminal="reach_goal", minimum_clearance=.004)], 0)
        self.assertEqual(result["stage"], "no-safety-qualified-single-step-rescue")
        self.assertEqual(result["successful_commands"], 1)

    def test_truth_negative_only_localizes_value_consumer(self):
        outcomes = [dict(terminal="collision", minimum_clearance=-.1, return_=-.2),
                    dict(terminal="reach_goal", minimum_clearance=.3, return_=.5)]
        parts = dict(raw_top=0, raw_rank=[1, 53], final_rank=[1, 53], blocked=[False, False])
        result = attribution(parts, dict(final_top=0, final_rank=[1, 53]), outcomes, 0)
        self.assertEqual(result["stage"], "value-ranking-residual")
        self.assertAlmostEqual(result["delta_q"], .7)
        self.assertNotIn("training_cause", result)

    def test_successful_raw_top_is_postprocessing_evidence(self):
        outcomes = [dict(terminal="collision", minimum_clearance=-.1, return_=-.2),
                    dict(terminal="reach_goal", minimum_clearance=.3, return_=.5)]
        parts = dict(raw_top=1, raw_rank=[2, 1], final_rank=[1, 2], blocked=[False, True])
        result = attribution(parts, dict(final_top=0, final_rank=[1, 2]), outcomes, 0)
        self.assertEqual(result["stage"], "postprocessing")

    def test_logging_hook_preserves_native_scoring_and_smoothing_once(self):
        torch.set_num_threads(1)
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1]/"shixu/default.ini")
        cfg.set("model", "representation", "tracks")
        model = ForecastSuccessorValueModel("cv", width=16, layers=1)
        policy = ValuePolicy(model, cfg)
        policy.last_action = ActionXY(.4, -.2)
        state = ObservedTracks(FullState(0, -1, .2, .1, .3, 0, 4, 1, 0),
                               [ObservableState(.5, -.2, -.2, 0, .3)], (0,), (True,), (.5,), 1)
        commands = np.asarray(policy.candidate_actions())
        expected = policy.score(state)
        captured = []
        hook = model.critic.value_head.register_forward_hook(lambda m, args, out: captured.append(out.detach()))
        try:
            action = policy.predict(state)
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        _, rewards, clearances = policy.aligned_candidates(state, commands)
        logged = stages(rewards, captured[0].reshape(-1), clearances, cfg)
        np.testing.assert_array_equal(logged["final"], expected)
        np.testing.assert_array_equal(action, commands[int(np.argmax(expected))])


if __name__ == "__main__":
    unittest.main()
