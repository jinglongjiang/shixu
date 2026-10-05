"""Frozen two-root score attribution. No history intervention or training."""

import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, restored, weights
from experiments.multihorizon_control import root_commands, run_branches, save_json
from experiments.occlusion import source_hash
from shixu.features import encode_tracks, track_tokens, window
from shixu.policy import ValuePolicy


SOURCE = Path("outputs/temporal_problem_audit")
OUT = Path("outputs/action_ranking_attribution")
SELECTION = {5: (17, 9), 21: (37, 58, 21)}
RULES = {
    "selection": "User-fixed root05/20circle80009/tick18:17,9; root21/20square80005/tick11:37,58,21. "
                 "Action indices are zero-based; ranks are one-based, descending stable order.",
    "frozen": "CV419 checkpoint, native reward/gamma/dynamics, observations, history, action support, "
              "value consumer and continuation. No CC history interventions, new scorer or training.",
    "reference": "Reuse all archived80-action outcomes at each of the two roots. These Q_pi values "
                 "are for smoothed commands, not the grid velocities scored by the policy.",
    "accounting": "Native grid CV -> smoothed-command CV -> same-support true successor -> actual legal "
                  "successor -> archived Q_pi. The sum of adjacent score differences is the total "
                  "native-score minus Q_pi discrepancy, not automatically a pure value error.",
    "truth": "Independently restore each root and advance one native0.25s step for each grid and "
             "smoothed command. This is not a repeat of the full80-action continuation enumeration. "
             "Replace p/v/r only for the same active actors; preserve association, age increment, "
             "observed=false and all historical frames. Separately evaluate the real lawful next "
             "observation (including sensing/mask/admission changes) without rewriting past history. "
             "Truth labels remain offline. No goals/ORCA state enter the network.",
    "stages": "Report native immediate reward, gamma*V, safety mask, risk penalty and index0 bias "
              "separately. Truth-value-only keeps native immediate reward and filters; truth-r+V "
              "uses native simulator immediate reward but still holds original filters fixed. "
              "Report raw and filtered ranks; do not deploy any diagnostic re-scoring.",
    "smoothing": "Only five prespecified action continuations bypass root smoothing; subsequent "
                 "steps retain the identical original policy/smoothing. No alternate continuation.",
    "limits": "Residual value discrepancy is conditional on this fixed continuation and consumer. "
              "It does not locate a defective hidden layer, identify a training cause, prove an "
              "optimal ordering, or establish missing historical information.",
}


def ranks(values):
    order = np.argsort(-np.asarray(values), kind="stable")
    result = np.empty(len(order), dtype=int)
    result[order] = np.arange(1, len(order) + 1)
    return result


def stages(rewards, values, clearances, cfg, done=None):
    """Keep the production float32 gamma multiplication / float64 addition."""
    discounted = cfg.getfloat("train", "gamma") * values
    if done is not None:
        discounted = discounted.masked_fill(torch.as_tensor(done, device=values.device), 0)
    raw = torch.as_tensor(rewards, device=values.device) + discounted
    distance = torch.as_tensor(clearances, device=values.device)
    margin, risk = (cfg.getfloat("eval_protocol", key) for key in ("safety_margin", "risk_lambda"))
    blocked = (distance < margin) if margin > 0 and bool((distance >= margin).any()) else torch.zeros_like(distance, dtype=torch.bool)
    safe = raw.masked_fill(blocked, -1e9)
    threshold = margin if margin > 0 else cfg.getfloat("reward", "discomfort_dist")
    penalty = risk * torch.clamp(threshold - distance, min=0) if risk > 0 else torch.zeros_like(distance)
    final = safe - penalty
    final[0] -= 1e-3
    arrays = dict(immediate_reward=np.asarray(rewards), value=values.cpu().numpy(),
                  discounted_value=discounted.cpu().numpy(), raw=raw.cpu().numpy(),
                  safety_score=safe.cpu().numpy(), risk_penalty=penalty.cpu().numpy(),
                  blocked=blocked.cpu().numpy(), final=final.cpu().numpy(),
                  clearance=np.asarray(clearances))
    result = {k: v.tolist() for k, v in arrays.items()}
    for name in ("value", "raw", "safety_score", "final"):
        result[name + "_rank"] = ranks(arrays[name]).tolist()
        result[name + "_top"] = int(np.argmax(arrays[name]))
    return result


def truth_same_support(root, env):
    """One-step truth changes a query, never the committed observed history."""
    state = root["state"]
    indices = {key: index for index, key, _, _ in root["snapshot"]["tracks"]}
    humans = [ObservableState(env.humans[indices[key]].px, env.humans[indices[key]].py,
                              env.humans[indices[key]].vx, env.humans[indices[key]].vy,
                              env.humans[indices[key]].radius) for key in state.track_ids]
    return state._replace(self_state=env.robot.get_full_state(), human_states=humans,
                          observed=(False,) * len(humans),
                          ages=tuple(age + env.time_step for age in state.ages))


def one_step(root, commands, cfg):
    support, legal, records = [], [], []
    for command in commands:
        env, observer = restored(cfg, root["episode"], root["snapshot"])
        _, reward, done, truncated, info = env.step(ActionXY(*command))
        state = truth_same_support(root, env)
        support.append(encode_tracks(state))
        observed = observer.observe(env)
        legal.append(encode_tracks(observed))
        endpoint = min((np.linalg.norm(np.asarray(h.position) - env.robot.get_position())
                        - h.radius - env.robot.radius for h in state.human_states), default=1e6)
        records.append(dict(reward=reward, done=done or truncated, event=info["event"],
                            native_step_clearance=info["dmin"], same_support_endpoint_clearance=float(endpoint),
                            legal_ids=list(observed.track_ids), legal_observed=list(observed.observed),
                            legal_ages=list(observed.ages)))
    return support, legal, records


def score_queries(model, history, queries, device):
    slots = max(history.shape[1], max(q.shape[0] for q in queries))
    prefix = np.pad(history, ((0, 0), (0, slots - history.shape[1]), (0, 0)))
    queries = np.stack([np.pad(q, ((0, slots - len(q)), (0, 0))) for q in queries])
    return model.score_candidates(torch.as_tensor(prefix[None], device=device),
                                  torch.as_tensor(queries[None], device=device)).squeeze(0)


def discrepancy(native, execution_cv, execution_truth, execution_legal, q):
    parts = dict(grid_to_executed=native - execution_cv,
                 cv_to_truth=execution_cv - execution_truth,
                 truth_to_legal=execution_truth - execution_legal,
                 legal_value_residual=execution_legal - q)
    np.testing.assert_allclose(sum(parts.values()), native - q, atol=1e-12, rtol=1e-12)
    return dict(parts, total=native - q)


def artifacts():
    files = [SOURCE / "cohort.pt", weights(419, "cv")]
    for root in SELECTION:
        files.extend(SOURCE / folder / f"root_{root:02d}.json"
                     for folder in ("native4090_results", "action_headroom"))
    return {str(path): digest(path) for path in files}


def prepare():
    data = torch.load(SOURCE / "cohort.pt", map_location="cpu", weights_only=False)
    assert source_hash() == data["protocol"]["source_sha256"]
    assert digest(weights(419, "cv")) == data["protocol"]["checkpoint_sha256"]
    assert digest(SOURCE / "cohort.pt") == json.loads((SOURCE / "cohort_hash.json").read_text())["sha256"]
    save_json(OUT / "protocol.json", dict(rules=RULES, selection=SELECTION, inputs=artifacts(),
              core_sha256=source_hash(), created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


@torch.inference_mode()
def evaluate(device):
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["inputs"] == artifacts()
    assert protocol["core_sha256"] == source_hash()
    model, cfg = load(weights(419, "cv"), device)
    assert model.kind == "cv" and not cfg.getboolean("robot", "visible")
    manifest = dict(protocol_sha256=digest(OUT / "protocol.json"),
                    script_sha256=digest(Path(__file__)), python=platform.python_version(),
                    torch=torch.__version__, device=device,
                    gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    config={s: dict(cfg[s]) for s in cfg.sections()})
    save_json(OUT / "manifest.json", manifest)
    roots = torch.load(SOURCE / "cohort.pt", map_location="cpu", weights_only=False)["roots"]
    for index, selected in SELECTION.items():
        started = time.perf_counter()
        root = roots[index]
        history_before = root["history"].copy()
        policy = ValuePolicy(model, cfg, device)
        policy.history.extend(root["history"][:-1])
        history = window(list(policy.history) + [policy.encode(root["state"])], policy.length, "zero")
        np.testing.assert_array_equal(history, root["history"])
        reference = json.loads((SOURCE / "native4090_results" / f"root_{index:02d}.json").read_text())
        archived = json.loads((SOURCE / "action_headroom" / f"root_{index:02d}.json").read_text())
        commands, production, _ = root_commands(root, model, cfg, device)
        np.testing.assert_array_equal(commands, archived["commands"])
        np.testing.assert_allclose(production, reference["choices"]["current"]["scores"], atol=1e-7, rtol=0)
        assert int(production.argmax()) == selected[0] == archived["native_action"]
        assert ranks(production)[9 if index == 5 else 58] == (53 if index == 5 else 38)
        grid = np.asarray(policy.action_space)
        q = np.asarray([r["return_"] for r in archived["outcomes"]])
        results, query_artifacts = {}, {}
        for label, velocities in (("grid", grid), ("executed", commands)):
            policy.action_space = [ActionXY(*v) for v in velocities]
            queries, rewards, clearance = policy.aligned_candidates(root["state"])
            values = score_queries(model, history, queries, device)
            results[label + "_cv"] = stages(rewards, values, clearance, cfg)
            if label == "grid":
                np.testing.assert_array_equal(results["grid_cv"]["final"], production)
            truth, legal, steps = one_step(root, velocities, cfg)
            truth_values = score_queries(model, history, truth, device)
            legal_values = score_queries(model, history, legal, device)
            actual_rewards = np.asarray([r["reward"] for r in steps])
            done = [r["done"] for r in steps]
            results[label + "_truth_value_only"] = stages(rewards, truth_values, clearance, cfg)
            results[label + "_truth"] = stages(actual_rewards, truth_values, clearance, cfg, done)
            results[label + "_legal"] = stages(actual_rewards, legal_values, clearance, cfg, done)
            results[label + "_steps"] = steps
            query_artifacts[label] = dict(cv=queries, truth=truth, legal=legal)
        np.testing.assert_array_equal(history_before, root["history"])
        policy.action_space = [ActionXY(*v) for v in grid]
        policy.last_action = ActionXY(*root["previous"])
        actual = policy.predict(root["state"])
        np.testing.assert_array_equal(actual, commands[selected[0]])
        raw_controls = [grid[i] for i in selected]
        bypass = run_branches(root, model, cfg, device, raw_controls)
        selected_rows = []
        for action, outcome in zip(selected, bypass):
            r_true = results["executed_steps"][action]["reward"]
            assert not results["executed_steps"][action]["done"]
            tail = (q[action] - r_true) / policy.gamma
            stages_for_action = {name: {k: v[action] for k, v in row.items() if isinstance(v, list)}
                                 for name, row in results.items() if not name.endswith("_steps")}
            parts = discrepancy(*(results[name]["raw"][action] for name in
                                ("grid_cv", "executed_cv", "executed_truth", "executed_legal")), q[action])
            selected_rows.append(dict(action=action, grid_command=grid[action].tolist(),
                executed_command=commands[action].tolist(), gamma=policy.gamma,
                stages=stages_for_action, actual_step_reward=r_true, actual_q=q[action], actual_tail_value=tail,
                aligned_cv_value_error=results["executed_cv"]["value"][action] - tail,
                same_support_truth_value_error=results["executed_truth"]["value"][action] - tail,
                actual_legal_value_error=results["executed_legal"]["value"][action] - tail,
                discrepancy=parts, archived_outcome={k: v for k, v in archived["outcomes"][action].items()
                                                   if k != "commands"}, root_smoothing_bypass=outcome))
        # Pair differences are native minus good, so positive predicted margins mean wrong preference.
        pair_errors = []
        for good in selected[1:]:
            native = selected[0]
            parts = [next(r for r in selected_rows if r["action"] == a)["discrepancy"] for a in (native, good)]
            pair_errors.append(dict(native=native, good=good, actual_q_margin=float(q[native] - q[good]),
                predicted_raw_margin=results["grid_cv"]["raw"][native] - results["grid_cv"]["raw"][good],
                paired_error_decomposition={k: parts[0][k] - parts[1][k] for k in parts[0]}))
        tensor_path = OUT / f"queries_{index:02d}.pt"
        if tensor_path.exists():
            raise RuntimeError("Refusing to overwrite query evidence")
        torch.save(dict(history=history, queries=query_artifacts, root=index), tensor_path)
        save_json(OUT / f"root_{index:02d}.json", dict(root=index, episode=root["episode"], tick=root["tick"],
            previous=root["previous"], query_sha256=digest(tensor_path), stages=results,
            actual_q=q.tolist(), actual_q_rank=ranks(q).tolist(), selected=selected_rows,
            pairs=pair_errors, production_score_parity=True, command_parity=True, history_unchanged=True,
            elapsed_seconds=time.perf_counter() - started))
        print("ATTRIBUTION", index, "tops", {k: v["raw_top"] for k, v in results.items() if not k.endswith("_steps")},
              "selected", [(r["action"], r["discrepancy"], r["root_smoothing_bypass"]["terminal"]) for r in selected_rows],
              "seconds", time.perf_counter() - started, flush=True)
    assert protocol["inputs"] == artifacts() and protocol["core_sha256"] == source_hash()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare()
    else:
        evaluate(args.device)
