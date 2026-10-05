"""Training-target and descriptive matching conventions, without training."""

import unittest
import numpy as np

from experiments.cv419_value_contract import local_components, local_distance, nearby, input_stats
from shixu.replay import Replay, returns


class CV419ValueContractTests(unittest.TestCase):
    def test_terminal_targets_and_one_step_discount(self):
        for terminal in (1., -.5):
            np.testing.assert_allclose(returns([0., 0., terminal], .99), [.99 ** 2 * terminal, .99 * terminal, terminal])

    def test_replay_target_is_at_pre_action_state(self):
        replay = Replay(capacity=10, length=2, gamma=.99, left_pad="zero")
        frames = np.zeros((2, 2, 13), np.float32)
        frames[0, 0, 0], frames[1, 0, 0] = 1, 2
        replay.add(dict(tokens=frames, rewards=[0., 1.]))
        self.assertEqual(len(replay.samples), 2)
        self.assertAlmostEqual(replay.samples[0][2], .99, places=6)
        self.assertEqual(replay.samples[0][1], 0)
        self.assertEqual(replay.samples[1][2], 1.)
        self.assertEqual(replay.samples[1][1], 1)

    def token(self):
        token = np.zeros((4, 13), np.float32)
        token[0, :9] = [0, 0, .1, 0, .3, 0, 4, 1, 0]
        token[1, [0, 2, 6, 12]] = [1, 1, .3, 1]
        token[2, [1, 2, 6, 9, 12]] = [2, 2, .3, 1, 1]
        return token

    def test_actor_matching_is_permutation_invariant(self):
        token = self.token()
        shuffled = token[[0, 2, 3, 1]]
        self.assertAlmostEqual(local_distance(local_components(token), local_components(shuffled)), 0.)

    def test_neighbours_are_distinct_episodes_not_error_selected(self):
        token = self.token()
        component = local_components(token)
        rows = [dict(episode=i // 2, case=i // 2, tick=i, target=(-1) ** i) for i in range(20)]
        nearest = nearby(rows, [component] * len(rows), token)
        self.assertEqual(len(nearest), 8)
        self.assertEqual(len({r["episode"] for r in nearest}), 8)
        self.assertEqual([r["index"] for r in nearest], list(range(0, 16, 2)))

    def test_padding_is_not_crowd_density(self):
        token = self.token()
        stats = input_stats(np.pad(token, ((0, 10), (0, 0))))
        self.assertEqual(stats["active"], 2)
        self.assertEqual(stats["slots"], 13)


if __name__ == "__main__":
    unittest.main()
