"""Every candidate is scored as the command that will actually be executed."""

import configparser
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
from shixu.model import ValueModel, OrderedValueModel, OcclusionValueModel
from shixu.observations import ObservedTracks, TrackedState
from shixu.policy import ValuePolicy, successor


class ExecutionContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = configparser.ConfigParser()
        self.cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        self.robot = FullState(0, -1, .2, .1, .3, 0, 4, 1, 0)
        self.humans = [ObservableState(.5, -.2, -.2, 0, .3)]

    def policy(self, representation="legacy"):
        self.cfg.set("model", "representation", representation)
        model = {"legacy": lambda: ValueModel(width=32, layers=1),
                 "aligned": lambda: OrderedValueModel("actor", width=32, layers=1),
                 "tracks": lambda: OcclusionValueModel("gru", width=32, layers=1)}[representation]()
        return ValuePolicy(model, self.cfg)

    def state(self, representation):
        if representation == "legacy":
            return JointState(self.robot, self.humans)
        if representation == "tracks":
            return ObservedTracks(self.robot, self.humans, (0,), (True,), (.5,), 1)
        return TrackedState(self.robot, self.humans, (0,))

    def test_first_control_and_training_use_unchanged_grid(self):
        policy = self.policy()
        self.assertIs(policy.candidate_actions(), policy.action_space)
        policy.last_action = ActionXY(.4, -.2)
        policy.set_phase("train")
        self.assertIs(policy.candidate_actions(), policy.action_space)
        policy.set_phase("test")
        self.cfg.set("eval_protocol", "action_smoothing", "0")
        self.assertIs(policy.candidate_actions(), policy.action_space)

    def test_candidate_index_maps_once_to_executed_command(self):
        policy = self.policy()
        previous = policy.last_action = ActionXY(.4, -.2)
        grid = np.asarray(policy.action_space).copy()
        commands = policy.candidate_actions()
        np.testing.assert_array_equal(commands, .3 * np.asarray(previous) + .7 * grid)
        np.testing.assert_array_equal(policy.action_space, grid)
        self.assertEqual(policy.last_action, previous)
        self.assertEqual(len(policy.history), 0)

    def test_query_reward_clearance_and_filters_share_executed_commands(self):
        for representation in ("legacy", "aligned", "tracks"):
            policy = self.policy(representation)
            policy.model.eval()
            state = self.state(representation)
            policy.last_action = ActionXY(.4, -.2)
            commands = policy.candidate_actions()
            tokens, rewards, clearance = [], [], []
            for command in commands:
                future = successor(state, command, policy.time_step)
                tokens.append(policy.encode(future))
                reward, distance = policy.immediate_reward(state, future, command)
                rewards.append(reward)
                clearance.append(distance)
            if representation != "legacy":
                actual = policy.aligned_candidates(state)
                np.testing.assert_array_equal(actual[0], tokens)
                np.testing.assert_allclose(actual[1], rewards, atol=1e-15, rtol=1e-15)
                np.testing.assert_allclose(actual[2], clearance, atol=1e-15, rtol=1e-15)
                prefix = np.repeat(policy.encode(state)[None], policy.length - 1, axis=0)
                if representation == "tracks":
                    prefix[:-1] = 0
                with torch.inference_mode():
                    if hasattr(policy.model, "score_candidates"):
                        values = policy.model.score_candidates(torch.as_tensor(prefix[None]),
                                                                torch.as_tensor(np.asarray(tokens)[None]))[0]
                    else:
                        sequences = np.concatenate((np.broadcast_to(prefix, (len(tokens), *prefix.shape)),
                                                    np.asarray(tokens)[:, None]), axis=1)
                        values = policy.model(torch.as_tensor(sequences))
            else:
                from shixu.features import window
                sequences = [window([policy.encode(state), token], policy.length) for token in tokens]
                with torch.inference_mode():
                    values = policy.model(torch.as_tensor(np.asarray(sequences)))
            expected = np.asarray(rewards) + (policy.gamma * values).numpy()
            distance = np.asarray(clearance)
            if np.any(distance >= .2):
                expected[distance < .2] = -1e9
            expected -= .8 * np.maximum(.2 - distance, 0)
            expected[0] -= 1e-3
            np.testing.assert_allclose(policy.score(state), expected, atol=3e-6, rtol=1e-6)
            self.assertEqual(len(policy.history), 0)

    def test_predict_executes_scored_command_without_second_smoothing(self):
        policy = self.policy()
        policy.last_action = ActionXY(.4, -.2)
        commands = policy.candidate_actions()
        captured = []

        def score(state, scored):
            captured.extend(scored)
            result = np.zeros(80)
            result[17] = 1
            return result

        policy.score = score
        result = policy.predict(self.state("legacy"))
        np.testing.assert_array_equal(captured, commands)
        self.assertEqual(result, commands[17])
        self.assertEqual(policy.last_action, result)
        self.assertEqual(len(policy.history), 1)

    def test_exploration_and_reset_keep_training_command_contract(self):
        policy = self.policy()
        policy.set_phase("train")
        policy.last_action = ActionXY(.4, -.2)
        with patch("numpy.random.random", return_value=0), patch("numpy.random.randint", return_value=9):
            selected = policy.predict(self.state("legacy"), epsilon=1)
        self.assertEqual(selected, policy.action_space[9])
        policy.reset()
        self.assertIsNone(policy.last_action)
        self.assertEqual(len(policy.history), 0)


if __name__ == "__main__":
    unittest.main()
