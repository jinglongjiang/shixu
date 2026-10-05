"""Frozen sequence coverage, evidence selection and native memory semantics."""

import configparser
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crowd_sim.envs.utils.action import ActionXY
from experiments.multistep_headroom_decision_test import backup_roots, neighbors, run_sequence, schedules, summarize


class MultistepHeadroomTests(unittest.TestCase):
    def test_neighbors_follow_heading_major_native_grid(self):
        self.assertEqual(neighbors(0), [0, 1, 5, 10, 70, 75])
        self.assertEqual(neighbors(79), [4, 9, 69, 74, 78, 79])
        self.assertEqual(neighbors(32), [22, 27, 31, 32, 33, 37, 42])
        with self.assertRaises(ValueError):
            neighbors(80)

    def test_fixed_sequence_coverage_and_no_constant_reruns(self):
        constant, local = schedules("constant"), schedules("local")
        self.assertEqual(len(constant), 160)
        self.assertEqual(len(local), 896)
        self.assertEqual(len(set(constant + local)), 1056)
        for seq in local:
            half = len(seq)//2
            self.assertIn(len(seq), (4, 8))
            self.assertEqual(seq[:half], (seq[0],)*half)
            self.assertEqual(seq[half:], (seq[-1],)*half)
            self.assertIn(seq[-1], neighbors(seq[0]))
            self.assertNotEqual(seq[0], seq[-1])

    def test_backup_selection_uses_parent_behavior_not_method_results(self):
        primary = [dict(case=81000), dict(case=81001), dict(case=81003), dict(case=81006)]
        records = [dict(people=n, geometry="square", case=c, seed=s, behavior="low-progress-timeout")
                   for n in (5, 10) for c in range(81000, 81010) for s in (443, 419)]
        result = backup_roots(records, primary)
        self.assertEqual([(r["people"], r["case"], r["seed"]) for r in result],
                         [(5, 81002, 419), (10, 81004, 419), (5, 81005, 419), (10, 81007, 419)])

    def test_safety_qualification_does_not_hide_collisions_or_fragile_success(self):
        row = dict(people=5, geometry="square", case=1, seed=419)
        name = "5-square-1-seed419"
        records = [dict(root=name, terminal=t, minimum_clearance=c, return_=q, branch=b)
                   for b, (t, c, q) in enumerate([("collision", -.1, -.5),
                       ("reach_goal", .019, .8), ("reach_goal", .02, .7)])]
        result = summarize([row], records, {name: dict(return_=-.1)})[0]
        self.assertEqual(result["counts"], dict(collision=1, reach_goal=2))
        self.assertEqual(result["qualified"], 1)
        self.assertEqual(result["fragile"], 1)
        self.assertAlmostEqual(result["best_delta_q"], .8)

    def test_numpy_scalar_clearance_summary_serializes_without_changing_counts(self):
        row = dict(people=5, geometry="square", case=1, seed=419)
        name = "5-square-1-seed419"
        records = [dict(root=name, terminal="reach_goal", minimum_clearance=np.float64(.019),
                        return_=.8, branch=0)]
        result = summarize([row], records, {name: dict(return_=-.1)})[0]
        self.assertEqual(json.loads(json.dumps(result))["fragile"], 1)

    def world(self, terminal_step=6):
        cfg = configparser.ConfigParser()
        cfg.read_dict(dict(train=dict(gamma="0.99"), eval_protocol=dict(safety_margin="0.2")))
        env = SimpleNamespace(tick=0)
        observer = SimpleNamespace(observe=lambda env: env.tick)
        policy = SimpleNamespace(history=[], last_action=ActionXY(0., 0.))
        policy.encode = lambda state: state
        policy.candidate_actions = lambda: [ActionXY(.3*policy.last_action.vx + .7, 0.)]*80
        policy.aligned_candidates = lambda state, commands: (None, None, np.ones(80))
        def predict(state):
            policy.history.append(policy.encode(state))
            policy.last_action = policy.candidate_actions()[0]
            return policy.last_action
        policy.predict = predict
        def step(command):
            env.tick += 1
            return None, .1, env.tick == terminal_step, False, dict(dmin=.3, event="reach_goal")
        env.step = step
        world = dict(env=env, observer=observer, policy=policy, total=0., discount=1., minimum=1e6,
                     steps=0, commands=[], done=False, event=None)
        return world, cfg

    def test_smoothing_once_and_legal_history_during_intervention(self):
        world, cfg = self.world()
        root = dict(state=0, snapshot=dict(time=2.))
        with patch("experiments.multistep_headroom_decision_test.fork", return_value=world):
            result = run_sequence(root, None, cfg, "cpu", (0,)*4)
        np.testing.assert_allclose(np.array(result["commands"])[:, 0], 1.-.3**np.arange(1, 7))
        self.assertEqual(world["policy"].history, list(range(6)))
        self.assertEqual(result["actual_intervention_steps"], 4)
        self.assertAlmostEqual(result["return_"], sum(.1*.99**i for i in range(6)))
        self.assertEqual(result["absolute_terminal_seconds"], 3.5)

    def test_terminal_stops_forced_sequence(self):
        world, cfg = self.world(terminal_step=2)
        with patch("experiments.multistep_headroom_decision_test.fork", return_value=world):
            result = run_sequence(dict(state=0, snapshot=dict(time=2.)), None, cfg, "cpu", (0,)*8)
        self.assertEqual(result["actual_intervention_steps"], 2)
        self.assertEqual(len(result["commands"]), 2)
        self.assertEqual(world["policy"].history, [0, 1])


if __name__ == "__main__":
    unittest.main()
