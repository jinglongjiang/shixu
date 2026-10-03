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

    def test_private_successor_update_matches_exact_one_step_and_gradients(self):
        from shixu.temporal import DeltaCell
        cell = DeltaCell(32, "kda").double()
        state = torch.randn(6, 4, 8, 8, dtype=torch.float64, requires_grad=True)
        query = torch.randn(6, 32, dtype=torch.float64, requires_grad=True)
        saved = state.detach().clone()
        direct = cell.imagine(state, query)
        explicit, _ = cell.encode(query[:, None], torch.ones(6, 1, dtype=torch.bool),
                                  query.new_zeros(6, 1, 4), initial_state=state)
        torch.testing.assert_close(direct, explicit[:, 0], rtol=1e-10, atol=1e-10)
        first = torch.autograd.grad(direct.square().sum(), (state, query), retain_graph=True)
        second = torch.autograd.grad(explicit.square().sum(), (state, query))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(state, saved, rtol=0, atol=0)

    def test_candidate_branches_broadcast_without_real_memory_writes(self):
        model = OcclusionValueModel("kda", 32, 2, query_mode="branch")
        tokens = torch.randn(2, 5, 21, 13)
        tokens[:, :, 1:, 12] = 1
        tokens[:, :3, 1:, 10] = 1
        tokens[:, 3:, 1:, 10] = 0
        memory = model.encode_history(tokens[:, :-1])
        saved = [s.clone() for s in memory[0]]
        queries = tokens[:, -1:].expand(-1, 80, -1, -1).clone()
        queries[:, :, 0, :4] += torch.randn(2, 80, 4)
        values = model.read_history(memory, queries)
        sequential = torch.cat([model.read_history(memory, queries[:, tick:tick+1]) for tick in range(80)], 1)
        torch.testing.assert_close(values, sequential, rtol=1e-5, atol=1e-5)
        for before, after in zip(saved, memory[0]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        values.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        torch.testing.assert_close(model(tokens), model(tokens[:, :, [0, *range(20, 0, -1)]]), rtol=1e-5, atol=1e-5)

    def test_retained_writes_are_a_distinct_legal_pseudomeasurement_control(self):
        observed = OcclusionValueModel("kda", 32, 1, query_mode="branch")
        retained = OcclusionValueModel("kda", 32, 1, query_mode="branch", write_mode="retained")
        retained.load_state_dict(observed.state_dict())
        tokens = torch.randn(1, 5, 3, 13)
        tokens[:, :, 1:, 12] = 1
        tokens[:, :, 1:, 10] = 0
        tokens[:, 0, 1:, 10] = 1
        changed = tokens.clone()
        changed[:, 1:4, 1:, :9] *= -2
        torch.testing.assert_close(observed.encode_history(tokens[:, :-1])[0][0],
                                   observed.encode_history(changed[:, :-1])[0][0], rtol=0, atol=0)
        self.assertGreater(float((retained.encode_history(tokens[:, :-1])[0][0]
                                 - retained.encode_history(changed[:, :-1])[0][0]).abs().max()), 1e-5)

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

    def test_observation_clock_read_is_shared_and_parameter_matched(self):
        from unittest.mock import patch
        candidate = OcclusionValueModel("kda", 32, 1)
        real = OcclusionValueModel("kda", 32, 1, "observation")
        real.load_state_dict(candidate.state_dict())
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), sum(p.numel() for p in real.parameters()))
        history = torch.randn(2, 4, 6, 13)
        history[:, :, 1:, 10] = 1
        history[:, :, 1:, 12] = 1
        queries = history[:, -1:].expand(-1, 80, -1, -1).clone()
        queries[:, :, 0, 0] += torch.linspace(-1, 1, 80)
        memory = real.encode_history(history)
        with patch.object(real.temporal_encoder, "read", wraps=real.temporal_encoder.read) as called:
            values = real.read_history(memory, queries)
            self.assertEqual(called.call_args.args[1].shape, (10, 32))
        self.assertTrue(torch.isfinite(values).all())
        self.assertGreater(float(values.std().detach()), 1e-6)
        values.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in real.parameters() if p.grad is not None))
        a = OcclusionValueModel("current", 32, 1, "candidate")
        b = OcclusionValueModel("current", 32, 1, "observation")
        b.load_state_dict(a.state_dict())
        torch.testing.assert_close(a(history), b(history), rtol=0, atol=0)

    def test_final_il_reuse_preserves_rl_sampling_and_update(self):
        import copy
        from unittest.mock import patch
        from shixu.training import train

        class TinyValue(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(13, 1)

            def forward(self, inputs):
                return self.head(inputs[:, -1, 0]).squeeze(-1)

        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
        for key, value in (("il_epochs", 2), ("il_batch_size", 2), ("batch_size", 2), ("updates_per_ep", 2)):
            cfg.set("train", key, str(value))
        cfg.set("buffer", "capacity", "100")
        episode = {"tokens": [np.ones((2, 13), dtype=np.float32), np.full((2, 13), 2., dtype=np.float32)],
                   "rewards": [0., 1.], "terminal": "reach_goal"}
        model = TinyValue()
        policy = SimpleNamespace(model=model, device=torch.device("cpu"), length=2, gamma=.99)
        final_il = {}
        def snapshot(row):
            if row["phase"] == "il" and row["epoch"] == 2:
                final_il.update(copy.deepcopy(model.state_dict()))
        with TemporaryDirectory() as directory, patch("shixu.training.run_episode", return_value=episode):
            train(None, policy, cfg, Path(directory) / "full.pt", 7, 1, 1,
                  demonstrations=[episode], report=snapshot)
            reused = TinyValue()
            reused.load_state_dict(final_il)
            other = SimpleNamespace(model=reused, device=torch.device("cpu"), length=2, gamma=.99)
            train(None, other, cfg, Path(directory) / "reused.pt", 7, 1, 1,
                  demonstrations=[episode], pretrained_il=True)
            for a, b in zip(model.parameters(), reused.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_context_write_retains_other_actor_history_without_extra_parameters(self):
        parent = OcclusionValueModel("kda", 32, 1)
        context = OcclusionValueModel("kda", 32, 1, interaction_order="write")
        context.load_state_dict(parent.state_dict())
        self.assertEqual(sum(p.numel() for p in parent.parameters()), sum(p.numel() for p in context.parameters()))
        tokens = torch.randn(1, 5, 3, 13)
        tokens[:, :, 1:, 10] = 1
        tokens[:, :, 1:, 12] = 1
        changed = tokens.clone()
        changed[:, :3, 2, :9] *= -2
        for model, expected in ((parent, False), (context, True)):
            first = model.encode_history(tokens[:, :-1])[0][0][0]
            other = model.encode_history(changed[:, :-1])[0][0][0]
            self.assertEqual(bool((first - other).abs().max() > 1e-6), expected)
        context(tokens).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in context.parameters() if p.grad is not None))
        torch.testing.assert_close(context(tokens), context(tokens[:, :, [0, 2, 1]]), rtol=1e-5, atol=1e-5)

    def test_context_writes_ignore_unmeasured_neighbour_numeric_fields(self):
        model = OcclusionValueModel("kda", 32, 1, interaction_order="write")
        tokens = torch.randn(1, 4, 3, 13)
        tokens[:, :, 1:, 12] = 1
        tokens[:, :, 1, 10] = 1
        tokens[:, :, 2, 10] = 0
        changed = tokens.clone()
        changed[:, :, 2, :10] *= 10
        first = model.encode_history(tokens)
        other = model.encode_history(changed)
        for left, right in zip(first[0], other[0]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_current_value_and_checkpoint_contract_survive_attention_relocation(self):
        a = OcclusionValueModel("current", 32, 1)
        b = OcclusionValueModel("current", 32, 1, interaction_order="write")
        b.load_state_dict(a.state_dict())
        tokens = torch.randn(3, 6, 21, 13)
        tokens[:, :, 1:, 12] = torch.rand(3, 6, 20) > .4
        tokens[0, :, 1:, 12] = 0
        torch.testing.assert_close(a(tokens), b(tokens), rtol=0, atol=0)
        for kind in ("gru", "kda"):
            model = OcclusionValueModel(kind, 32, 1, interaction_order="write")
            self.assertTrue(torch.isfinite(model(torch.zeros(1, 1, 2, 13))).all())
            with TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save({"model": model.state_dict()}, path)
                other = OcclusionValueModel(kind, 32, 1, interaction_order="write")
                load_weights(other, path, "cpu")
                torch.testing.assert_close(model(tokens), other(tokens))

    def test_local_addresses_preserve_local_motion_with_contextual_content(self):
        vanilla = OcclusionValueModel("kda", 32, 2, interaction_order="residual")
        local = OcclusionValueModel("kda", 32, 2, interaction_order="residual", local_address=True)
        local.load_state_dict(vanilla.state_dict())
        self.assertEqual(sum(p.numel() for p in vanilla.parameters()), sum(p.numel() for p in local.parameters()))
        tokens = torch.randn(2, 5, 21, 13)
        tokens[:, :, 1:, 10] = 1
        tokens[:, :, 1:, 12] = 1
        mask = tokens[:, :-1, 1:, 12] > 0
        features = local._features(tokens[:, :-1])[0]
        content = features + local._attend(features, mask)
        cell = local.temporal_encoder.cells[0]
        shape = features.shape
        flatten = lambda x: x.permute(0, 2, 1, 3).reshape(-1, shape[1], shape[3])
        expected = cell.encode(flatten(content), mask.permute(0, 2, 1).reshape(-1, shape[1]),
                               content.new_zeros(shape[0] * shape[2], shape[1], 4),
                               addresses=flatten(features))[1]
        torch.testing.assert_close(local.encode_history(tokens[:, :-1])[0][0], expected, rtol=0, atol=0)
        local(tokens).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in local.parameters() if p.grad is not None))
        torch.testing.assert_close(local(tokens), local(tokens[:, :, [0, *range(20, 0, -1)]]), atol=1e-5, rtol=1e-5)
        memory = local.encode_history(tokens[:, :-1])
        saved = [s.clone() for s in memory[0]]
        local.read_history(memory, tokens[:, -1:].expand(-1, 80, -1, -1))
        for before, after in zip(saved, memory[0]):
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_residual_writes_obey_measurement_mask(self):
        model = OcclusionValueModel("kda", 32, 1, interaction_order="residual", local_address=True)
        tokens = torch.randn(1, 4, 3, 13)
        tokens[:, :, 1:, 12] = 1
        tokens[:, :, 1, 10] = 1
        tokens[:, :, 2, 10] = 0
        changed = tokens.clone()
        changed[:, :, 2, :10] *= 10
        for a, b in zip(model.encode_history(tokens)[0], model.encode_history(changed)[0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_separate_address_source_equal_to_content_is_exact_default(self):
        from shixu.temporal import ActorMemory
        memory = ActorMemory("kda", 32, 2)
        features = torch.randn(4, 6, 32)
        valid = torch.rand(4, 6) > .3
        evidence = torch.zeros(4, 6, 4)
        regular = memory.encode(features, valid, evidence)
        same_source = memory.encode(features, valid, evidence, addresses=features)
        for a, b in zip(regular, same_source):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_residual_arm_configuration_and_reload(self):
        import json
        from experiments.occlusion import configuration
        from shixu.model import build_model
        protocol = json.loads((Path(__file__).resolve().parents[1] / "experiments/occlusion_context_protocol.json").read_text())
        protocol["interaction_order"] = "residual"
        for arm in ("current", "gru", "kda", "kda_local"):
            cfg = configuration(protocol, arm)
            cfg.set("model", "width", "32")
            cfg.set("model", "layers", "1")
            model = build_model(cfg)
            tokens = torch.zeros(2, 1, 2, 13)
            self.assertTrue(torch.isfinite(model(tokens)).all())
            with TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save({"model": model.state_dict()}, path)
                other = build_model(cfg)
                load_weights(other, path, "cpu")
                torch.testing.assert_close(model(tokens), other(tokens), rtol=0, atol=0)

    def test_broadcast_reads_match_expanded_values_and_gradients(self):
        import copy
        from shixu.temporal import ActorMemory
        for kind in ("gru", "kda"):
            a = ActorMemory(kind, 32, 2)
            b = copy.deepcopy(a)
            batch, people, count = 2, 5, 80
            features = torch.randn(batch * people, 6, 32)
            mask = torch.rand(batch * people, 6) > .2
            evidence = features.new_zeros(batch * people, 6, 4)
            left, right = a.encode(features, mask, evidence), b.encode(features, mask, evidence)
            queries = torch.randn(batch, count, people, 32)
            expanded, shared = [], []
            for x, y in zip(left, right):
                if kind == "gru":
                    expanded.append(x.reshape(2, batch, people, 32)[:, :, None].expand(
                        -1, -1, count, -1, -1).reshape(2, -1, 32))
                    shared.append(y.reshape(2, batch, 1, people, 32))
                else:
                    expanded.append(x.reshape(batch, people, *x.shape[1:])[:, None].expand(
                        -1, count, -1, -1, -1, -1).reshape(-1, *x.shape[1:]))
                    shared.append(y.reshape(batch, 1, people, *y.shape[1:]))
            flat = queries.reshape(-1, 32)
            if kind == "gru":
                old = a.read(tuple(expanded), flat, torch.ones(batch * count * people, dtype=torch.bool))
            else:
                old = flat
                for cell, state in zip(a.cells, expanded):
                    q = torch.nn.functional.normalize(cell._heads(cell.q(cell.input_norm(old))), dim=-1)
                    values = (q[..., None] * state).sum(-2) * (cell.head_width ** -.5)
                    old = old + cell.output(cell.output_norm(values.flatten(-2)))
                old = old - flat
            new = b.read(tuple(shared), queries, torch.ones(batch, count, people, dtype=torch.bool))
            torch.testing.assert_close(old.reshape_as(new), new, rtol=1e-5, atol=1e-5)
            old.square().mean().backward()
            new.square().mean().backward()
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)

    def test_packed_observations_match_held_gru_state_and_gradients(self):
        import copy
        from shixu.temporal import ActorMemory
        a = ActorMemory("gru", 32, 2)
        b = copy.deepcopy(a)
        features = torch.randn(7, 23, 32)
        mask = torch.rand(7, 23) > .3
        mask[0] = False
        mask[1] = True
        old = features.new_zeros(2, 7, 32)
        for tick in range(23):
            _, proposal = a.gru(features[:, tick:tick + 1], old)
            old = torch.where(mask[:, tick][None, :, None], proposal, old)
        new = b.encode(features, mask, features.new_zeros(7, 23, 4))[0]
        torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-5)
        old.square().mean().backward()
        new.square().mean().backward()
        for p, q in zip(a.gru.parameters(), b.gru.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
