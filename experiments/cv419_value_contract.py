"""Read-only CV419 training contract and frozen80009 checkpoint audit."""

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, kendalltau
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.action_ranking_attribution import ranks, score_queries, stages
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load
from experiments.multihorizon_control import save_json
from experiments.occlusion import source_hash
from shixu.features import encode_tracks, stack_histories, window
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.replay import returns
from shixu.runner import environment


BASE = Path("outputs/forecast_control_b_fresh_cv/419/cv")
DATA = Path("outputs/forecast_control_b/demonstrations.pt")
COHORT = Path("outputs/temporal_problem_audit/cohort.pt")
ATTRIBUTION = Path("outputs/action_ranking_attribution")
HEADROOM = Path("outputs/temporal_problem_audit/action_headroom/root_05.json")
OUT = Path("outputs/cv419_value_contract")
RULES = {
    "scope": "CV419 only. No training, parameter edits, new consumer/history, new scenes or80005 work.",
    "checkpoints": "Same run's il.pt and model.pt only. No intermediate checkpoint exists in run/backup. "
                   "Other seed419 model versions are not a chronological learning curve.",
    "stages": "Exact same80009 root18/history/80actions. Reuse prior native and smoothed CV, same-support "
              "truth and lawful-next-frame query tensors. Reference Q_pi is always the archived FINAL "
              "CV419 continuation, NOT the IL checkpoint's own continuation or an optimal Q.",
    "il": "All128 existing successful ORCA episodes, all6245 recorded states. Recompute original gamma.99 "
          "MC targets and replay existing commands solely to verify rewards/tokens. No new navigation runs.",
    "coverage": "Current root and executed-CV/actual-legal successors17,9. Fixed local distance: robot9 "
                "features scaled by[4,4,1,1,.3,4,4,1,pi]; five nearest active actors matched by Hungarian "
                "assignment using rel-position/2,velocity/1,radius/.3,age/2 and presence/1. Missing slots "
                "cost1, two absent slots cost0. Distance is sqrt(robot mean-square + actor mean-square). "
                "Keep nearest frame per IL episode, then eight nearest distinct episodes, before reading "
                "model residuals. This intentionally omits extra actors: descriptive LOCAL neighbours, "
                "not balanced full-state/history matches or causal evidence. No tuned near threshold.",
    "population": "Compare all IL state statistics to80009. Summarize existing192-case CV419 evaluation "
                  "by geometry/N; no re-evaluation. Four cached5-person roots permit V(H) vs realized "
                  "FINAL-policy returns only, NOT80-action ranking or matched-density conclusions.",
    "missing": "Original3000-episode online replay, optimizer/minibatch histories, intermediate weights "
               "and own-IL continuations are absent. Episode logs contain undiscounted returns and last "
               "minibatch losses, not MC target distributions or visitation coverage.",
}


def distribution(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()),
                q05_q50_q95=np.quantile(x, [.05, .5, .95]).tolist())


def local_components(token):
    active = token[1:][token[1:, 12] > 0]
    active = active[np.argsort(np.linalg.norm(active[:, :2], axis=1), kind="stable")[:5]]
    actor = np.zeros((5, 7))
    actor[:len(active), :6] = active[:, [0, 1, 3, 4, 6, 9]] / [2, 2, 1, 1, .3, 2]
    actor[:len(active), 6] = 1
    robot = token[0, :9] / np.array([4, 4, 1, 1, .3, 4, 4, 1, np.pi])
    return robot, actor


def local_distance(first, second):
    r1, a1 = first
    r2, a2 = second
    cost = ((a1[:, None] - a2[None]) ** 2).mean(-1)
    present1, present2 = a1[:, 6] > 0, a2[:, 6] > 0
    cost[present1[:, None] != present2[None]] = 1.
    cost[~present1[:, None] & ~present2[None]] = 0.
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(np.mean((r1 - r2) ** 2) + cost[rows, cols].mean()))


def input_stats(token):
    active = token[1:][token[1:, 12] > 0]
    distance = np.linalg.norm(active[:, :2], axis=1)
    measured = active[:, 10] > 0
    return dict(slots=len(token) - 1, active=len(active), measured=int(measured.sum()),
                retained=int((~measured).sum()), within2=int((distance < 2).sum()),
                within3=int((distance < 3).sum()), within4=int((distance < 4).sum()),
                minimum_clearance=float(np.min(distance - active[:, 6] - token[0, 4])) if len(active) else 100.,
                max_age=float(active[:, 9].max()) if len(active) else 0.,
                minimum_age=float(active[:, 9].min()) if len(active) else 0.,
                inverse_ttc_max=float(active[:, 7].max()) if len(active) else 0.,
                goal_distance=float(np.linalg.norm(token[0, :2] - token[0, 5:7])),
                robot_speed=float(np.linalg.norm(token[0, 2:4])))


def nearby(rows, components, query):
    target = local_components(query)
    closest = {}
    for index, (row, component) in enumerate(zip(rows, components)):
        distance = local_distance(component, target)
        old = closest.get(row["episode"])
        if old is None or distance < old[0]:
            closest[row["episode"]] = (distance, index)
    return [dict(index=i, distance=d, **rows[i]) for d, i in sorted(closest.values())[:8]]


def input_files():
    return [DATA, COHORT, HEADROOM, ATTRIBUTION / "queries_05.pt", ATTRIBUTION / "root_05.json",
            BASE / "il.pt", BASE / "model.pt", BASE / "learning.jsonl", BASE / "result.json"]


def prepare():
    manifest = {str(p): digest(p) for p in input_files()}
    archived = json.loads((BASE / "result.json").read_text())
    assert archived["source_sha256"] == source_hash()
    assert archived["demonstration_sha256"] == manifest[str(DATA)]
    assert archived["checkpoint_sha256"] == manifest[str(BASE / "model.pt")]
    assert archived["reused_il_sha256"] is None
    save_json(OUT / "protocol.json", dict(rules=RULES, inputs=manifest, source_sha256=source_hash(),
              created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


def verify_demonstrations(episodes, policy, cfg):
    reward_error = token_error = command_error = 0.
    n, off_grid, case_keys = 0, 0, []
    for ep in episodes:
        env = environment(cfg, policy, "circle", 5)
        env.reset(options={"test_case": ep["case"]})
        observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
        totals = []
        assert len(ep["actions"]) == len(ep["tokens"]) == len(ep["rewards"])
        for tick, command in enumerate(ep["actions"]):
            tokens = encode_tracks(observer.observe(env))
            token_error = max(token_error, float(np.max(np.abs(tokens - ep["tokens"][tick]))))
            off_grid += np.min(np.linalg.norm(np.asarray(policy.action_space) - command, axis=1)) > 1e-6
            _, reward, done, truncated, info = env.step(ActionXY(*command))
            reward_error = max(reward_error, abs(reward - ep["rewards"][tick]))
            command_error = max(command_error, float(np.max(np.abs(np.asarray([env.robot.vx, env.robot.vy]) - command))))
            assert (done or truncated) == (tick == len(ep["actions"]) - 1)
            totals.append(reward)
            n += 1
        assert info["event"] == ep["terminal"] == "reach_goal"
        assert ep["rewards"][-1] == 1.
        np.testing.assert_allclose(returns(totals, policy.gamma), returns(ep["rewards"], policy.gamma), atol=1e-7, rtol=0)
        case_keys.append(ep["case"])
    assert reward_error < 1e-7 and token_error < 2e-5 and command_error < 1e-12
    return dict(episodes=len(episodes), states=n, cases=case_keys, reward_max_error=reward_error,
                token_max_error=token_error, command_max_error=command_error, off_grid_teacher_actions=int(off_grid),
                terminal_reward=1., no_post_terminal_training_sample=True)


def log_summary():
    rows = [json.loads(s) for s in (BASE / "learning.jsonl").read_text().splitlines()]
    il = [r for r in rows if r["phase"] == "il"]
    rl = [r for r in rows if r["phase"] == "rl"]
    assert [r["epoch"] for r in il] == list(range(1, 51))
    assert [r["episode"] for r in rl] == list(range(1, 3001))
    blocks = []
    for start in range(0, len(rl), 500):
        batch = rl[start:start + 500]
        blocks.append(dict(first=start + 1, last=start + len(batch), terminals=dict(Counter(r["terminal"] for r in batch)),
                           undiscounted_return=distribution([r["return"] for r in batch]),
                           last_minibatch_loss=distribution([r["value_loss"] for r in batch]),
                           epsilon_first=batch[0]["epsilon"], epsilon_last=batch[-1]["epsilon"]))
    return dict(il_epochs=len(il), rl_episodes=len(rl), il_epoch_losses=il, rl_blocks=blocks,
                rl_terminals=dict(Counter(r["terminal"] for r in rl)),
                source="Logged return is sum(rewards), not G_t. Loss is last minibatch only; replay is absent.")


@torch.inference_mode()
def evaluate(device):
    started = time.perf_counter()
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["inputs"] == {str(p): digest(p) for p in input_files()}
    assert protocol["source_sha256"] == source_hash()
    episodes = torch.load(DATA, map_location="cpu", weights_only=False)["episodes"]
    roots = torch.load(COHORT, map_location="cpu", weights_only=False)["roots"]
    root = roots[5]
    assert root["episode"]["case"] == 80009 and root["tick"] == 18
    evidence = torch.load(ATTRIBUTION / "queries_05.pt", map_location="cpu", weights_only=False)
    previous = json.loads((ATTRIBUTION / "root_05.json").read_text())
    q = np.asarray([r["return_"] for r in json.loads(HEADROOM.read_text())["outcomes"]])
    queries = evidence["queries"]
    model, cfg = load(BASE / "model.pt", device)
    policy = ValuePolicy(model, cfg, device)
    verified = verify_demonstrations(episodes, policy, cfg)
    rows, histories, frames, targets = [], [], [], []
    for index, ep in enumerate(episodes):
        labels = returns(ep["rewards"], policy.gamma)
        for tick, (frame, target) in enumerate(zip(ep["tokens"], labels)):
            rows.append(dict(episode=index, case=ep["case"], tick=tick, target=float(target)))
            histories.append(window(ep["tokens"][:tick + 1], policy.length, "zero"))
            frames.append(frame)
            targets.append(target)
    targets = np.asarray(targets)
    components = [local_components(frame) for frame in frames]
    query_states = {"root": root["history"][-1]}
    for name in ("cv", "legal"):
        for action in (17, 9):
            query_states[f"executed_{name}_{action}"] = queries["executed"][name][action]
    matches = {name: nearby(rows, components, query) for name, query in query_states.items()}
    stats = [input_stats(frame) for frame in frames]
    population = {key: distribution([s[key] for s in stats]) for key in stats[0]}
    query_stats = {name: input_stats(query) for name, query in query_states.items()}
    phases, predictions = {}, {}
    five_roots = [i for i, r in enumerate(roots) if r["episode"]["people"] == 5]
    for phase, filename in (("il50", "il.pt"), ("rl3000", "model.pt")):
        model, phase_cfg = load(BASE / filename, device)
        assert {s: dict(cfg[s]) for s in cfg.sections()} == {s: dict(phase_cfg[s]) for s in phase_cfg.sections()}
        stage_rows = {}
        for label, query_kind in (("grid_cv", "cv"), ("executed_cv", "cv"),
                                  ("executed_truth", "truth"), ("executed_legal", "legal")):
            space = "grid" if label.startswith("grid") else "executed"
            value = score_queries(model, evidence["history"], queries[space][query_kind], device)
            rewards, clearance = (previous["stages"][label][key] for key in ("immediate_reward", "clearance"))
            stage = stages(rewards, value, clearance, cfg)
            if phase == "rl3000":
                np.testing.assert_allclose(stage["raw"], previous["stages"][label]["raw"], atol=1e-7, rtol=0)
            raw, final = np.asarray(stage["raw"]), np.asarray(stage["final"])
            stage_rows[label] = dict(stage, spearman_vs_final_q=float(spearmanr(raw, q).correlation),
                kendall_vs_final_q=float(kendalltau(raw, q).correlation),
                final_score_spearman_vs_final_q=float(spearmanr(final, q).correlation),
                pair_margin_17_minus_9=float(raw[17] - raw[9]),
                errors={str(a): dict(score_minus_final_q=float(raw[a] - q[a]),
                    value_minus_final_tail=(float(value[a].item() - (q[a] - previous["stages"]["executed_steps"][a]["reward"]) / policy.gamma)
                                           if space == "executed" else None))
                        for a in (17, 9)})
        predicted = []
        for start in range(0, len(histories), 64):
            predicted.extend(model(torch.as_tensor(stack_histories(histories[start:start + 64]), device=device)).cpu().tolist())
        predicted = np.asarray(predicted)
        predictions[phase] = predicted
        residual = predicted - targets
        subset = {}
        for name, neighbours in matches.items():
            selected = np.asarray([r["index"] for r in neighbours])
            subset[name] = dict(mse=float(np.mean(residual[selected] ** 2)), mean_residual=float(residual[selected].mean()),
                rows=[dict(row, predicted=float(predicted[row["index"]]), residual=float(residual[row["index"]])) for row in neighbours])
        five = []
        for i in five_roots:
            cached = roots[i]
            v = float(model(torch.as_tensor(cached["history"][None], device=device)).item())
            target = float(sum(policy.gamma ** t * r for t, r in enumerate(cached["archived_rewards"])))
            five.append(dict(root=i, episode=cached["episode"], tick=cached["tick"], predicted=v, final_policy_return=target,
                             residual=v - target))
        root_v = float(model(torch.as_tensor(root["history"][None], device=device)).item())
        root_return = float(sum(policy.gamma ** t * r for t, r in enumerate(root["archived_rewards"])))
        phases[phase] = dict(checkpoint_sha256=digest(BASE / filename), stages=stage_rows,
            all_il=dict(states=len(targets), mse=float(np.mean(residual ** 2)), mae=float(np.mean(np.abs(residual))),
                        bias=float(np.mean(residual)), prediction=distribution(predicted)), neighbours=subset,
            five_cached_roots=five, root_observed_value=root_v, root_final_policy_return=root_return,
            root_observed_value_error=root_v - root_return)
        print("PHASE", phase, "NATIVE", {k: stage_rows["grid_cv"][k] for k in
              ("raw_top", "pair_margin_17_minus_9", "spearman_vs_final_q", "errors")},
              "IL_MSE", phases[phase]["all_il"]["mse"], flush=True)
    evaluation = json.loads((BASE / "result.json").read_text())["episodes"]
    cells = []
    for people in (5, 10, 20):
        for geometry in ("circle", "square"):
            cell = [r for r in evaluation if r["people"] == people and r["geometry"] == geometry]
            cells.append(dict(people=people, geometry=geometry, episodes=len(cell), terminals=dict(Counter(r["terminal"] for r in cell))))
    save_json(OUT / "result.json", dict(protocol_sha256=digest(OUT / "protocol.json"),
        script_sha256=digest(Path(__file__)), source_sha256=source_hash(), python=platform.python_version(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
        il_verification=verified, il_target_distribution=distribution(targets), il_positive_targets=int((targets > 0).sum()),
        checkpoint_phases=phases, coverage=dict(IL_population=population, query_statistics=query_stats,
                                              local_neighbours=matches,
                                              same_active_count_as_root=sum(s["active"] == query_stats["root"]["active"] for s in stats)),
        training_log=log_summary(), existing_evaluation=cells, elapsed_seconds=time.perf_counter() - started))
    np.savez_compressed(OUT / "il_predictions.npz", targets=targets, il50=predictions["il50"],
                        rl3000=predictions["rl3000"], case=np.asarray([r["case"] for r in rows]), tick=np.asarray([r["tick"] for r in rows]))
    assert protocol["inputs"] == {str(p): digest(p) for p in input_files()} and protocol["source_sha256"] == source_hash()
    print("COMPLETE", time.perf_counter() - started, flush=True)


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
