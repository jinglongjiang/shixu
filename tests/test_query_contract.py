"""The diagnostic changes only the query, with a correctly shifted target."""

import configparser
import unittest

import numpy as np

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.query_contract_probe import transitions
from shixu.features import encode_tracks, window
from shixu.observations import ObservedTracks
from shixu.policy import ValuePolicy, successor
from shixu.replay import returns


class QueryContractTests(unittest.TestCase):
    def state(self):
        return ObservedTracks(FullState(0, 0, 0, 0, .3, 0, 4, 1, 0),
                              [ObservableState(1, 2, .2, 0, .3),
                               ObservableState(-1, 1, -.1, .1, .3)],
                              (0, 1), (True, False), (0., .5), 2)

    def record(self, state):
        return {"robot": state.self_state.to_array(),
                "humans": [{"track_id": key, "state": human.to_array()}
                           for key, human in zip(state.track_ids, state.human_states)],
                "observed": state.observed, "ages": state.ages}

    def test_prefix_label_and_fresh_age_shadow(self):
        root = self.state()
        action = ActionXY(.1, .3)
        future = successor(root, action, .25)
        following = future._replace(observed=(True, False), ages=(0., .75))
        frames = [encode_tracks(root), encode_tracks(following)]
        episode = {"case": 13000, "tokens": frames, "rewards": [.1, 1.],
                   "actions": [action, action],
                   "observations": [self.record(root), self.record(following)]}
        records, modes, targets, skipped = transitions({"episodes": [episode]}, 4, .25)
        self.assertEqual(skipped, 0)
        self.assertEqual(records[0]["hidden"], 1)
        self.assertEqual(targets[0], returns(episode["rewards"], .99)[1])
        actual, predicted, shadow = (mode[0] for mode in modes)
        np.testing.assert_array_equal(actual[:-1], window(frames[:1], 4, "zero")[1:])
        np.testing.assert_array_equal(predicted[:-1], actual[:-1])
        np.testing.assert_array_equal(shadow[:-1], actual[:-1])
        np.testing.assert_array_equal(predicted[-1], encode_tracks(future))
        np.testing.assert_array_equal(shadow[-1, :, :9], predicted[-1, :, :9])
        self.assertEqual(shadow[-1, 1, 9], 0)
        self.assertEqual(shadow[-1, 2, 9], .75)
        np.testing.assert_array_equal(shadow[-1, :, 10], predicted[-1, :, 10])

    def test_query_matches_vectorized_candidate_path(self):
        cfg = configparser.ConfigParser()
        cfg.read("shixu/default.ini")
        cfg.set("model", "representation", "tracks")
        policy = ValuePolicy.__new__(ValuePolicy)
        policy.config, policy.time_step = cfg, .25
        policy.encode = encode_tracks
        policy.action_space = [ActionXY(.1, .3), ActionXY(-.2, .4)]
        state = self.state()
        tokens, _, _ = policy.aligned_candidates(state)
        for index, action in enumerate(policy.action_space):
            np.testing.assert_allclose(tokens[index], encode_tracks(successor(state, action, .25)),
                                       rtol=1e-6, atol=1e-6)

    def test_support_change_is_excluded_before_prediction(self):
        root = self.state()
        frame = encode_tracks(root)
        changed = frame.copy()
        changed[2] = 0
        episode = {"tokens": [frame, changed], "rewards": [0, 1]}
        records, modes, targets, skipped = transitions({"episodes": [episode]}, 4, .25)
        self.assertEqual((len(records), len(targets), skipped), (0, 0, 1))
        self.assertTrue(all(not samples for samples in modes))


if __name__ == "__main__":
    unittest.main()
