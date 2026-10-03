import unittest

import torch

from shixu.model import OrderedValueModel


class OrderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.scene = OrderedValueModel("scene", width=32, layers=1)
        self.actor = OrderedValueModel("actor", width=32, layers=1)
        self.actor.load_state_dict(self.scene.state_dict(), strict=True)
        self.tokens = torch.randn(2, 6, 8, 13)
        self.tokens[:, :, 3:8, 12] = 1

    def test_same_parameters_and_initialization(self):
        self.assertEqual(sum(p.numel() for p in self.scene.parameters()), sum(p.numel() for p in self.actor.parameters()))
        for a, b in zip(self.scene.parameters(), self.actor.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_one_step_shape(self):
        # One step is not generally equal: GRU and attention need not commute.
        for model in (self.scene, self.actor):
            self.assertEqual(model(self.tokens[:, :1]).shape, (2,))

    def test_consistent_actor_permutation_is_invariant(self):
        permuted = self.tokens.clone()
        permuted[:, :, 3:8] = self.tokens[:, :, [5, 7, 3, 6, 4]]
        for model in (self.scene, self.actor):
            torch.testing.assert_close(model(self.tokens), model(permuted), atol=2e-6, rtol=2e-6)

    def test_empty_observation_is_finite(self):
        self.tokens[:, :, 3:8] = 0
        for model in (self.scene, self.actor):
            self.assertTrue(torch.isfinite(model(self.tokens)).all())

    def test_missing_observation_does_not_write_actor_state(self):
        features = torch.randn(2, 5, 32)
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 2] = False
        changed = features.clone()
        changed[:, 2] += 100
        torch.testing.assert_close(self.actor._actor_last(features, mask), self.actor._actor_last(changed, mask))

    def test_both_orders_backpropagate(self):
        for model in (self.scene, self.actor):
            model(self.tokens).sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
