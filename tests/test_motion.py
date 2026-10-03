import ast
import math
import os
from pathlib import Path
import unittest

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from shixu.motion import MotionKDACell, MotionValueModel, observation_intervals


class MotionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)

    def tokens(self, time=5, people=20):
        tokens = torch.randn(2, time, people + 1, 13)
        tokens[:, :, 1:, 12] = 1
        tokens[:, :, 1:, 10] = 1
        tokens[:, 1:3, 1:, 10] = 0
        return tokens

    def test_clock_preserves_arrival_intervals_and_padding(self):
        mask = torch.tensor([[[0, 0], [1, 0], [0, 0], [0, 0], [1, 0]]], dtype=torch.bool)
        gaps, query = observation_intervals(mask)
        torch.testing.assert_close(gaps[0, :, 0], torch.tensor([1., 1., 1., 2., 3.]))
        torch.testing.assert_close(query, torch.ones(1, 2))
        self.assertEqual(observation_intervals(mask[:, :-1])[1][0, 0], 3)
        empty, query = observation_intervals(mask[:, :0])
        self.assertEqual(empty.shape, (1, 0, 2))
        torch.testing.assert_close(query, torch.ones(1, 2))

    def test_private_successor_matches_real_single_step(self):
        cell = MotionKDACell(32).double()
        query = torch.randn(5, 32, dtype=torch.float64, requires_grad=True)
        state = torch.randn(5, 4, 8, 8, dtype=torch.float64, requires_grad=True)
        elapsed = torch.arange(1, 6, dtype=torch.float64)
        actual = cell.successor(state, query, elapsed)
        expected, _ = cell.encode(query[:, None], torch.ones(5, 1, dtype=torch.bool),
                                  elapsed[:, None], initial_state=state)
        torch.testing.assert_close(actual, expected[:, 0], rtol=1e-10, atol=1e-10)
        first = torch.autograd.grad(actual.square().sum(), (state, query), retain_graph=True)
        second = torch.autograd.grad(expected.square().sum(), (state, query))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)

    def test_official_no_convolution_mixer_parity(self):
        root = Path(os.environ.get("FLA_REFERENCE", "/home/abc/workspace/il_x_rl_candidates_20261003/repos/flash-linear-attention"))
        if not (root / "fla/layers/kda.py").exists():
            self.skipTest("Optional pinned official FLA reference not installed")

        class StripAnnotations(ast.NodeTransformer):
            def visit_arg(self, node):
                node.annotation = None
                return node

            def visit_FunctionDef(self, node):
                node.returns, node.decorator_list = None, []
                return self.generic_visit(node)

        def extract(path, name):
            node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name)
            return compile(ast.fix_missing_locations(ast.Module(body=[StripAnnotations().visit(node)], type_ignores=[])),
                           str(path), "exec")

        class RMSGate(nn.Module):
            def __init__(self, width, activation, eps):
                super().__init__()
                self.weight, self.eps = nn.Parameter(torch.ones(width)), eps

            def forward(self, values, gate):
                return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps) * self.weight * gate.sigmoid()

        namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "rearrange": rearrange,
                     "FusedRMSNormGated": RMSGate, "get_layer_cache": lambda *a: None,
                     "update_layer_cache": lambda *a, **k: None,
                     "unpad_hidden_states": lambda x, *a: (x, None, None),
                     "repad_hidden_states": lambda x, *a: x}
        exec(extract(root / "fla/ops/kda/naive.py", "naive_recurrent_kda"), namespace)
        reference_states = []

        def reference_kernel(q, k, v, g, beta, A_log, dt_bias, **kwargs):
            g = -A_log.exp()[:, None] * F.softplus(g + dt_bias.reshape(g.shape[-2:]))
            result = namespace["naive_recurrent_kda"](F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, g,
                                                      beta.sigmoid(), output_final_state=True)
            reference_states.append(result[1])
            return result

        namespace.update(chunk_kda=reference_kernel, fused_recurrent_kda=reference_kernel)
        exec(extract(root / "fla/layers/kda.py", "KimiDeltaAttention"), namespace)
        official = namespace["KimiDeltaAttention"](hidden_size=32, head_dim=8, num_heads=4, use_short_conv=False)
        cell = MotionKDACell(32)
        for ours, theirs in ((cell.q, official.q_proj), (cell.k, official.k_proj), (cell.v, official.v_proj),
                             (cell.forget, official.f_proj), (cell.strength, official.b_proj),
                             (cell.read_gate, official.g_proj), (cell.output, official.o_proj)):
            theirs.load_state_dict(ours.state_dict())
        with torch.no_grad():
            official.A_log.copy_(cell.log_rate)
            official.dt_bias.copy_(cell.dt_bias.flatten())
            official.o_norm.weight.copy_(cell.read_scale)
        features = torch.randn(3, 11, 32, requires_grad=True)
        mask, gaps = torch.ones(3, 11, dtype=torch.bool), torch.ones(3, 11)
        actual, state = cell.encode(features, mask, gaps)
        expected = official(cell.input_norm(features))[0]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(state, reference_states[-1], rtol=2e-5, atol=2e-6)
        a = torch.autograd.grad(actual.square().sum(), features, retain_graph=True)[0]
        b = torch.autograd.grad(expected.square().sum(), features)[0]
        torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-5)

    def test_memory_is_goal_free_and_does_not_write_hidden_truth(self):
        for kind in ("kda", "gru"):
            model = MotionValueModel(kind, 32, 2)
            tokens = self.tokens()
            changed = tokens.clone()
            changed[:, :, 0, 5:8] *= -5
            changed[:, 1:3, 1:, :10] *= 8
            a, b = model.encode_history(tokens), model.encode_history(changed)
            states_a = a[0] if kind == "kda" else (a[0],)
            states_b = b[0] if kind == "kda" else (b[0],)
            for first, second in zip(states_a, states_b):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_shared_human_successor_and_actor_permutation(self):
        for kind in ("kda", "gru"):
            model = MotionValueModel(kind, 32, 2)
            tokens = self.tokens()
            memory = model.encode_history(tokens[:, :-1])
            queries = tokens[:, -1:].expand(-1, 80, -1, -1).clone()
            shift = torch.randn(2, 80, 2)
            queries[:, :, 0, :2] += shift
            queries[:, :, 1:, :2] -= shift[:, :, None]
            batched = model.read_history(memory, queries)
            separate = torch.cat([model.read_history(memory, queries[:, t:t+1]) for t in range(80)], 1)
            torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-5)
            self.assertGreater(float(batched.std().detach()), 1e-5)
            permuted = tokens[:, :, [0, *range(20, 0, -1)]]
            torch.testing.assert_close(model(tokens), model(permuted), rtol=1e-5, atol=1e-5)
            batched.mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_clock_contrast_is_same_capacity_and_empty_histories_work(self):
        a = MotionValueModel("kda", 32, 2, "observation")
        b = MotionValueModel("kda", 32, 2, "elapsed")
        b.load_state_dict(a.state_dict())
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        tokens = self.tokens()
        self.assertGreater(float((a(tokens) - b(tokens)).abs().max().detach()), 1e-5)
        for model in (a, b, MotionValueModel("gru", 32, 2), MotionValueModel("kda", 32, 2, use_history=False)):
            self.assertTrue(torch.isfinite(model(torch.zeros(2, 1, 2, 13))).all())
            self.assertTrue(torch.isfinite(model(torch.zeros(2, 5, 2, 13))).all())


if __name__ == "__main__":
    unittest.main()
