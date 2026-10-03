"""Legal measurements, missing-track reads, variable populations and reloads."""

import configparser
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from crowd_sim.envs.utils.state import FullState, ObservableState
from shixu.features import encode_tracks, window
from shixu.model import OcclusionValueModel, load_weights
from shixu.observations import OccludedTracks, ObservedTracks
from shixu.policy import ValuePolicy
from shixu.replay import Replay


class Human:
    def __init__(self, x, y, vx=0):
        self.px, self.py, self.vx, self.vy, self.radius = x, y, vx, 0, .3

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)


def robot():
    return FullState(0, 0, 0, 0, .3, 0, 4, 1, 0)


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_unseen_and_hidden_truth_do_not_leak(self):
        front, back = Human(1, 0), Human(2, 0, .2)
        env = SimpleNamespace(robot=SimpleNamespace(get_full_state=robot), humans=[front, back], global_time=0)
        observer = OccludedTracks(2)
        first = observer.observe(env)
        self.assertEqual(first.track_count, 1)
        front.py = 1
        seen = observer.observe(env)
        self.assertEqual(seen.track_count, 2)
        front.py = 0
        env.global_time = .25
        hidden = observer.observe(env)
        self.assertFalse(hidden.observed[1])
        self.assertAlmostEqual(hidden.human_states[1].px, 2.05)
        back.px, back.vx = 3, -7
        changed = observer.observe(env)
        np.testing.assert_array_equal(encode_tracks(hidden), encode_tracks(changed))
        env.global_time = 2.25
        self.assertEqual(observer.observe(env).track_ids, (0,))

    def test_population_padding_and_no_fake_initial_writes(self):
        state = ObservedTracks(robot(), [Human(i + 1, 1).get_observable_state() for i in range(20)],
                               tuple(range(20)), (True,) * 20, (0.,) * 20, 20)
        frame = encode_tracks(state)
        self.assertEqual(frame.shape, (21, 13))
        short = frame[:3]
        history = window([short, frame], 4, "zero")
        self.assertEqual(history.shape, (4, 21, 13))
        self.assertEqual(history[:2].sum(), 0)
        replay = Replay(length=4, left_pad="zero")
        replay.add({"tokens": [short, frame], "rewards": [0, 1]})
        replay.add({"tokens": [short], "rewards": [1]})
        batch, _ = replay.sample(8, "cpu", np.random.default_rng(1))
        self.assertEqual(batch.shape[-2], 21)

    def test_hidden_memory_read_candidate_immutability_and_gradients(self):
        model = OcclusionValueModel("kda", 32, 1)
        tokens = torch.zeros(1, 5, 3, 13)
        tokens[:, :, 0, :9] = torch.tensor(robot().to_array())
        tokens[:, :, 1:, 12] = 1
        tokens[:, :4, 1:, 10] = 1
        tokens[:, :4, 1:, :9] = torch.randn(1, 4, 2, 9)
        first = model(tokens)
        altered = tokens.clone()
        altered[:, :4, 1:, :9] *= -2
        self.assertGreater(float((first - model(altered)).abs().max()), 1e-5)
        memory = model.encode_history(tokens[:, :-1])
        saved = [state.clone() for state in memory[0]]
        model.read_history(memory, tokens[:, -1:].expand(-1, 80, -1, -1))
        for before, after in zip(saved, memory[0]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        first.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        permuted = tokens[:, :, [0, 2, 1]]
        torch.testing.assert_close(model(tokens), model(permuted), atol=1e-5, rtol=1e-5)

    def test_empty_scene_and_checkpoint_reload(self):
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        cfg.set("model", "representation", "tracks")
        state = ObservedTracks(robot(), [], (), (), (), 0)
        for kind in ("current", "gru", "kda"):
            model = OcclusionValueModel(kind, 32, 1)
            policy = ValuePolicy(model, cfg)
            self.assertTrue(np.isfinite(policy.score(state)).all())
            with TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save({"model": model.state_dict()}, path)
                other = OcclusionValueModel(kind, 32, 1)
                load_weights(other, path, "cpu")
                torch.testing.assert_close(model(torch.zeros(1, 4, 2, 13)), other(torch.zeros(1, 4, 2, 13)))


if __name__ == "__main__":
    unittest.main()
