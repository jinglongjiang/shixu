import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pas_temporal_probe import query, tensors


class FakeActor:
    def __init__(self):
        self.base = SimpleNamespace(Sensor_VAE=SimpleNamespace(encode=lambda grid: grid.mean()),
                                    Label_VAE=SimpleNamespace(encode=lambda grid: grid.mean()))
        self.inputs = None

    def act(self, inputs, hidden, masks, deterministic):
        self.inputs = inputs
        self.base.Sensor_VAE.encode(inputs["grid"])
        hidden["policy"].add_(100)
        action = torch.randn(1, 2)
        return torch.zeros(1), action, torch.zeros(1), hidden, inputs["grid"]


class PaSTemporalProbeTests(unittest.TestCase):
    def setUp(self):
        self.observation = dict(vector=np.arange(4, dtype=np.float32),
            grid=np.arange(4, dtype=np.float32)[:, None, None] * np.ones((4, 2, 2), dtype=np.float32),
            label_grid=np.full((2, 2, 2), 99., dtype=np.float32))
        self.hidden = {"policy": torch.ones(1, 1, 2)}
        self.rng = torch.get_rng_state().clone()

    def test_normal_inference_cannot_receive_truth_grid(self):
        self.assertEqual(set(tensors(self.observation)), {"vector", "grid"})

    def test_shortened_grid_preserves_exact_current_observation(self):
        actor = FakeActor()
        for mode in ("full4", "short2", "current_grid"):
            query(actor, self.observation, self.hidden, torch.ones(1, 4), self.rng, mode)
            np.testing.assert_array_equal(actor.inputs["grid"][0, -1].numpy(), self.observation["grid"][-1])
        self.assertEqual(actor.inputs["grid"][0, :, 0, 0].tolist(), [3.] * 4)

    def test_queries_do_not_commit_recurrent_state_and_use_common_noise(self):
        actor = FakeActor()
        before = self.observation["grid"].copy()
        a = query(actor, self.observation, self.hidden, torch.ones(1, 4), self.rng, "full4")
        b = query(actor, self.observation, self.hidden, torch.ones(1, 4), self.rng, "oracle_latent")
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(self.observation["grid"], before)
        self.assertTrue(self.hidden["policy"].eq(1).all())

    def test_oracle_restores_original_encoder_after_call(self):
        actor = FakeActor()
        original = actor.base.Sensor_VAE.encode
        query(actor, self.observation, self.hidden, torch.ones(1, 4), self.rng, "oracle_latent")
        self.assertIs(actor.base.Sensor_VAE.encode, original)


if __name__ == "__main__":
    unittest.main()
