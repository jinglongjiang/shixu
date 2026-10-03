import ast
import configparser
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from shixu.features import motion_evidence
from shixu.model import MemoryValueModel, OrderedValueModel, build_model
from shixu.temporal import ActorMemory, delta_scan


def recurrence(q, k, v, g, b, w, mask, state):
    output = []
    for t in range(q.shape[1]):
        decayed = state * g[:, t, :, :, None].exp()
        residual = w[:, t] * v[:, t] - ((b[:, t] * k[:, t])[..., None] * decayed).sum(-2)
        proposed = decayed + k[:, t, :, :, None] * residual[..., None, :]
        state = torch.where(mask[:, t, None, None, None], proposed, state)
        read = (q[:, t, :, :, None] * state).sum(-2) * q.shape[-1] ** -.5
        output.append(read * mask[:, t, None, None])
    return torch.stack(output, 1), state


def fixture(dtype=torch.float64):
    torch.manual_seed(41)
    q, k = [F.normalize(torch.randn(2, 23, 2, 4, dtype=dtype), dim=-1) for _ in range(2)]
    v = torch.randn_like(q)
    g = -torch.rand_like(q)
    b, w = [torch.rand_like(q) for _ in range(2)]
    mask = torch.rand(2, 23) > .25
    state = torch.randn(2, 2, 4, 4, dtype=dtype)
    return q, k, v, g, b, w, mask, state


def tokens(batch=2, time=7):
    result = torch.randn(batch, time, 8, 13)
    result[:, :, 3:8, 12] = 1
    return result


class DeltaTests(unittest.TestCase):
    def test_recurrence_and_gradient_parity(self):
        values = fixture()
        left = [v.clone().requires_grad_() for v in values[:6]] + [values[6], values[7].clone().requires_grad_()]
        right = [v.detach().clone().requires_grad_() for v in left[:6]] + [left[6], left[7].detach().clone().requires_grad_()]
        actual = delta_scan(*left[:7], initial_state=left[7])
        expected = recurrence(*right)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)
        sum(x.square().sum() for x in actual).backward()
        sum(x.square().sum() for x in expected).backward()
        for a, e in zip(left[:6] + [left[7]], right[:6] + [right[7]]):
            torch.testing.assert_close(a.grad, e.grad, rtol=1e-9, atol=1e-9)

    def test_mask_holds_state_and_extreme_decay_is_finite(self):
        q, k, v, g, b, w, mask, state = fixture()
        zero, held = delta_scan(q, k, v, g, b, w, torch.zeros_like(mask), initial_state=state)
        self.assertEqual(zero.abs().max().item(), 0)
        torch.testing.assert_close(held, state)
        g = torch.full_like(g, -100)
        actual = delta_scan(q, k, v, g, b, w, mask, initial_state=state)
        expected = recurrence(q, k, v, g, b, w, mask, state)
        for a, e in zip(actual, expected):
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)

    def test_official_reference_parity(self):
        root = Path(os.environ.get("FLA_REFERENCE", "/home/abc/workspace/il_x_rl_candidates_20261003/repos/flash-linear-attention"))
        source = root / "fla/ops/gdn2/naive.py"
        if not source.exists():
            self.skipTest("Optional official FLA reference not installed")
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "naive_recurrent_gdn2")
        for arg in function.args.args:
            arg.annotation = None
        function.returns = None
        namespace = {"torch": torch, "F": F}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
        q, k, v, g, b, w, _, state = fixture(torch.float32)
        mask = torch.ones(q.shape[:2], dtype=torch.bool)
        for erase, write in ((b, w), (b[..., :1].expand_as(b), b[..., :1].expand_as(w))):
            actual = delta_scan(q, k, v, g, erase, write, mask, initial_state=state)
            expected = namespace["naive_recurrent_gdn2"](q, k, v, g, erase, write,
                                                         initial_state=state, output_final_state=True)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)
        source = root / "fla/ops/kda/naive.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "naive_recurrent_kda")
        for arg in function.args.args:
            arg.annotation = None
        function.returns = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
        beta = b[..., 0]
        actual = delta_scan(q, k, v, g, beta[..., None].expand_as(k), beta[..., None].expand_as(v), mask, initial_state=state)
        expected = namespace["naive_recurrent_kda"](q, k, v, g, beta, initial_state=state, output_final_state=True)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_evidence_causality_and_reentry(self):
        x = tokens(time=8)
        earlier = motion_evidence(x[:, :5])
        x[:, 5:, 3:8, 3:5] = 100
        torch.testing.assert_close(earlier, motion_evidence(x)[:, :5])
        x[:, 3, 3, 12] = 0
        evidence = motion_evidence(x)
        self.assertEqual(evidence[:, 3:5, 0].abs().sum().item(), 0)
        self.assertEqual(evidence[:, 0].abs().sum().item(), 0)

    def test_shared_prefix_values_gradients_and_read_only(self):
        for substrate, readout in (("gru", "evidence"), ("kda", "full"), ("kda", "read"),
                                    ("kda", "gate"), ("kda", "evidence"),
                                    ("gdn2", "read"), ("gdn2", "revision")):
            model = MemoryValueModel(substrate, readout, 16, 2).eval()
            x = tokens()
            queries = tokens(time=3)
            memory = model.encode_history(x[:, :-1])
            original = [s.clone() for s in memory[0]]
            shared = model.read_history(memory, queries)
            windows = torch.cat((x[:, None, :-1].expand(-1, 3, -1, -1, -1), queries[:, :, None]), 2)
            ordinary = model(windows.reshape(6, 7, 8, 13)).reshape(2, 3)
            torch.testing.assert_close(shared, ordinary, rtol=2e-5, atol=2e-6)
            for state, before in zip(memory[0], original):
                torch.testing.assert_close(state, before)
            torch.testing.assert_close(model.read_history(memory, queries.flip(1)), shared.flip(1), rtol=2e-5, atol=2e-6)
            model.zero_grad()
            shared.sum().backward()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            model(windows.reshape(6, 7, 8, 13)).sum().backward()
            for name, parameter in model.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=5e-4, atol=2e-5)

    def test_gaps_zero_prefix_and_actor_permutation(self):
        for substrate in ("gru", "kda", "gdn2"):
            model = MemoryValueModel(substrate, "read", 16, 2).eval()
            x = tokens()
            x[:, 2:4, 4, 12] = 0
            result = model(x)
            permuted = x.clone()
            permuted[:, :, 3:8] = x[:, :, [7, 4, 3, 6, 5]]
            torch.testing.assert_close(model(permuted), result, rtol=2e-5, atol=2e-6)
            self.assertTrue(torch.isfinite(model(x[:, -1:])).all())
            x[:, :, 3:8, 12] = 0
            self.assertTrue(torch.isfinite(model(x)).all())

    def test_missing_observations_do_not_write(self):
        for kind in ("gru", "kda", "gdn2"):
            memory = ActorMemory(kind, 16, 2)
            x = torch.randn(2, 4, 16)
            evidence = torch.randn(2, 4, 4)
            mask = torch.tensor([[True, False, False, True], [True, False, False, True]])
            actual = memory.encode(x, mask, evidence)
            expected = memory.encode(x[:, [0, 3]], torch.ones(2, 2, dtype=torch.bool), evidence[:, [0, 3]])
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)

    def test_evidence_controls_are_parameter_matched(self):
        for kind, first, second in (("kda", "gate", "evidence"), ("gdn2", "read", "revision")):
            a, b = [MemoryValueModel(kind, readout, 16, 2) for readout in (first, second)]
            self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
            b.load_state_dict(a.state_dict(), strict=True)

    def test_factory_and_state_dictionary_roundtrip(self):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"model": {"architecture": "memory", "backbone": "gdn2", "readout": "revision", "width": "16", "layers": "2"}})
        model, restored = build_model(cfg), build_model(cfg)
        restored.load_state_dict(model.state_dict(), strict=True)
        x = tokens()
        torch.testing.assert_close(model(x), restored(x))

    def test_cached_gru_compute_control_preserves_values(self):
        from experiments.temporal_memory import cached_actor_control
        model = OrderedValueModel("actor", 16, 2).eval()
        for length in (0, 3, 6):
            prefix, queries = tokens(time=length), tokens(time=3)
            if length > 3:
                prefix[:, 2:4, 3, 12] = 0
                prefix[:, :, 4, 12] = 0
            shared = cached_actor_control(model, prefix, queries)
            windows = torch.cat((prefix[:, None].expand(-1, 3, -1, -1, -1), queries[:, :, None]), 2)
            expected = model(windows.reshape(6, length + 1, 8, 13)).reshape(2, 3)
            torch.testing.assert_close(shared, expected, rtol=2e-5, atol=2e-6)

    def test_shadow_counts_are_json_safe(self):
        from experiments.temporal_memory import shadow_contrasts
        custom = {"root_index": np.int64(2), "progress": np.float64(.2),
                  "minimum_clearance": np.float64(.1), "terminal": "running"}
        control = dict(custom, root_index=np.int64(1), progress=np.float64(.1))
        rows = [{"branches": {"custom": custom, "control": control}},
                {"branches": {"custom": control, "control": custom}}]
        result = json.loads(json.dumps(shadow_contrasts(rows, [("custom", "control")])))
        self.assertEqual(result[0]["different_root_actions"], 2)
        self.assertEqual(result[0]["safe_progress_wins_005m"], 1)
        self.assertEqual(result[0]["safe_progress_losses_005m"], 1)
        self.assertEqual(result[0]["custom_collisions"], 0)


if __name__ == "__main__":
    unittest.main()
