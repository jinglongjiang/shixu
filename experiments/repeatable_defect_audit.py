"""Method-blind failure atlas on the frozen executed-command contract.

No fitting or new consumer. Value-head hooks record the native forward pass;
80-action interventions run only for prespecified repeated failure strata.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.action_ranking_attribution import one_step, ranks, score_queries, stages
from experiments.decision_state import exact_snapshot
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import archive_parity, run_branches, save_json
from experiments.occlusion import source_hash
from shixu.features import window
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/repeatable_defect_audit")
SEEDS = (419, 443, 467, 491)
CELLS = tuple((n, g) for n in (5, 10, 20) for g in ("circle", "square"))
RULES = {
    "parent": "Four existing fresh-CV IL50/RL3000 checkpoints, identical repaired execution contract. "
              "No training, reward/filter/history/model edits. All runs on one local runtime/device.",
    "discovery": "All80000..80031 x5/10/20 people xcircle/square x419/443/467/491;768 episodes. "
                 "No exclusion for success, failure,80009 or prior method results.",
    "behavior": "Collision; persistent-low-progress timeout; other timeout. Low progress means first "
                "32-control-step/8s interval at t>=2s with net goal-distance decrease<=0.8m and "
                "all distances>1m. This is a descriptive label, not proof of feasible alternatives.",
    "root": "Low-progress: start of first qualifying interval. Other failures: first t>=2s legal "
            "clearance<=0.8m within final10s; otherwise beginning of final10s. One root per episode.",
    "repeat": "Stratum=(behavior,population,geometry);at least3 distinct numeric case IDs. "
              "Separate multi-seed repeat counts; do not count seeds as independent cases.",
    "selection": "At most3 qualifying strata, descending distinct-case count, then count of cases "
                 "failed in>=2 seeds, then lexical stratum key. First3 case IDs per stratum, "
                 "lowest failed seed per case. Freeze all roots before any intervention outcome.",
    "headroom": "All80 already-smoothed root commands once for0.25s, then same corrected checkpoint "
                "policy with lawful sensing until termination. No second smoothing. Q^pi, not Q*. "
                "Full native-return and command baseline replay parity required.",
    "attribution": "Raw-top success with final-top failure: postprocessing. Otherwise true-one-step "
                   "same-actor-support query selecting a successful action: successor/query interface. "
                   "Otherwise successful action raw rank>=10: value-ranking residual. "
                   "No successful first command: no-single-step-rescue; other patterns unresolved. "
                   "These localize a consumer stage, not training/representation/temporal causation.",
    "safety": "Keep every actual clearance. Successful branches below0.02m are flagged fragile, "
              "not counted as safety-qualified rescues;0.02m is a diagnostic screen, not a new reward.",
    "nominate": "First selected stratum whose3 independent roots have safety-qualified rescues "
                "and the same one of the three localized stages. Otherwise stop with no nominee.",
    "fresh": "Only after nominee definition frozen:81000..81031 x4 same seeds in its unchanged "
             "population/geometry cell. Same behavior/root/attribution rules. First4 distinct failed "
             "case IDs, lowest failed seed; all80 continuations. Require>=2 independently repeated "
             "safety-qualified localized defects. No threshold or checkpoint selection on fresh data.",
    "limits": "No feasible-path claim from slow motion alone; no global model cause from one root; "
              "no overall navigator gain from one-step oracle rescue. Negative truth intervention "
              "only constrains this frozen consumer. Fresh confirmation remains development evidence.",
}
_MODEL = _CFG = _POLICY = None


def freeze():
    sources = source_hash()
    checkpoints = {}
    configs = []
    for seed in SEEDS:
        p = weights(seed, "cv")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if (ck["seed"], ck["il_episodes"], ck["rl_episodes"]) != (seed, 128, 3000):
            raise ValueError("Unexpected checkpoint budget")
        configs.append(ck["config"])
        checkpoints[str(seed)] = {"path": str(p), "sha256": digest(p)}
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("Paired baseline configs differ")
    probe, cfg = load(weights(SEEDS[0], "cv"), "cpu")
    if probe.kind != "cv" or not hasattr(ValuePolicy, "candidate_actions"):
        raise ValueError("Requires repaired CV execution contract")
    if len(ValuePolicy(probe, cfg).action_space) != 80 or cfg.getfloat("env", "time_step") != .25:
        raise ValueError("Unexpected native actions or clock")
    if cfg.getboolean("robot", "visible"):
        raise ValueError("Do not change native human interaction setting")
    save_json(OUT / "protocol.json", dict(rules=RULES, checkpoints=checkpoints,
        config=configs[0], core_sha256=sources, script_sha256=digest(Path(__file__)),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        corrected_contract_origin="9c7c62a951988cf0d4a158e88acb9910e2197fef",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


def verify():
    p = json.loads((OUT / "protocol.json").read_text())
    if p["rules"] != RULES or p["core_sha256"] != source_hash():
        raise ValueError("Frozen scientific contract changed")
    expected = p["script_sha256"]
    correction = OUT / "implementation-correction.json"
    if correction.exists():
        note = json.loads(correction.read_text())
        if note["protocol_sha256"] != digest(OUT / "protocol.json") or note["original_script_sha256"] != expected:
            raise ValueError("Correction does not match frozen protocol")
        expected = note["corrected_script_sha256"]
    if expected != digest(Path(__file__)):
        raise ValueError("Frozen audit implementation changed")
    for ck in p["checkpoints"].values():
        if digest(Path(ck["path"])) != ck["sha256"]:
            raise ValueError("Frozen checkpoint changed")
    return p


def low_progress_start(distances, width=32):
    for tick in range(8, len(distances) - width):
        span = np.asarray(distances[tick:tick + width + 1])
        if np.all(span > 1.) and span[0] - span[-1] <= .8:
            return tick
    return None


def failure_label(terminal, distances):
    if terminal == "reach_goal":
        return "success"
    if terminal == "collision":
        return "collision"
    if terminal != "timeout":
        raise ValueError("Unexpected terminal " + terminal)
    return "low-progress-timeout" if low_progress_start(distances) is not None else "other-timeout"


def root_tick(record, trace):
    if record["behavior"] == "low-progress-timeout":
        return low_progress_start(trace["goal_distance"])
    start = max(0, len(trace["actions"]) - 40)
    eligible = np.flatnonzero((np.arange(len(trace["actions"])) >= max(8, start)) &
                              (np.asarray(trace["legal_clearance"]) <= .8))
    return int(eligible[0]) if len(eligible) else start


def init_worker(device):
    global _MODEL, _CFG, _POLICY
    torch.set_num_threads(1)
    _MODEL, _CFG, _POLICY = None, None, None
    _POLICY = device


@torch.inference_mode()
def episode(task):
    global _MODEL, _CFG
    split, seed, people, geometry, case = task
    prefix = OUT / split / f"{seed}_{people}_{geometry}_{case}"
    path = prefix.with_suffix(".json")
    if path.exists():
        row = json.loads(path.read_text())
        if digest(prefix.with_suffix(".npz")) != row["trace_sha256"]:
            raise ValueError("Trace changed")
        return row
    started = time.perf_counter()
    device = _POLICY
    if _MODEL is None or getattr(_MODEL, "audit_seed", None) != seed:
        _MODEL, _CFG = load(weights(seed, "cv"), device)
        _MODEL.audit_seed = seed
    model, cfg = _MODEL, _CFG
    policy = ValuePolicy(model, cfg, device)
    policy.reset()
    env = environment(cfg, policy, geometry, people)
    env.reset(options={"test_case": case})
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    captured = []
    hook = model.critic.value_head.register_forward_hook(lambda m, args, out: captured.append(out.detach()))
    rows = defaultdict(list)
    path_length, minimum = 0., 1e6
    try:
        while True:
            state = observer.observe(env)
            commands = np.asarray(policy.candidate_actions())
            grid = np.asarray(policy.action_space)
            previous = None if policy.last_action is None else list(policy.last_action)
            old = np.asarray(env.robot.get_position())
            distances = [np.linalg.norm(old - h.position) - env.robot.radius - h.radius
                         for h in state.human_states]
            captured.clear()
            action = policy.predict(state)
            if len(captured) != 1:
                raise ValueError("Native value-head hook must run exactly once")
            values = captured[0].reshape(-1)
            _, reward, clearance = policy.aligned_candidates(state, commands)
            parts = stages(reward, values, clearance, cfg)
            index = int(np.argmax(parts["final"]))
            np.testing.assert_array_equal(action, commands[index])
            if not rows["actions"]:
                reference = policy.history.pop()
                last = policy.last_action
                policy.last_action = None
                native = policy.score(state, commands)
                np.testing.assert_array_equal(native, parts["final"])
                policy.history.append(reference)
                policy.last_action = last
            rows["actions"].append(list(action))
            rows["grid_action"].append(index)
            rows["grid_command"].append(grid[index].tolist())
            rows["previous"].append([np.nan, np.nan] if previous is None else previous)
            rows["position"].append(old.tolist())
            rows["goal_distance"].append(float(np.linalg.norm(old - env.robot.get_goal_position())))
            rows["legal_clearance"].append(min(distances, default=1e6))
            rows["active_count"].append(len(state.human_states))
            rows["visible_count"].append(sum(state.observed))
            rows["local_density"].append(sum(d <= 2. for d in distances) / (np.pi * 2. ** 2))
            rows["phase"].append(env.global_time / cfg.getfloat("env", "time_limit"))
            for key in ("immediate_reward", "value", "raw", "blocked", "risk_penalty", "final", "clearance"):
                rows[key].append(parts[key])
            _, reward, done, truncated, info = env.step(action)
            rows["rewards"].append(reward)
            rows["actual_clearance"].append(info["dmin"])
            path_length += float(np.linalg.norm(np.asarray(env.robot.get_position()) - old))
            minimum = min(minimum, info["dmin"])
            if done or truncated:
                break
    finally:
        hook.remove()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    if prefix.with_suffix(".npz").exists():
        with np.load(prefix.with_suffix(".npz")) as existing:
            for key, values in arrays.items():
                np.testing.assert_array_equal(existing[key], values)
    else:
        np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    row = dict(split=split, seed=seed, people=people, geometry=geometry, case=case,
        terminal=info["event"], behavior=failure_label(info["event"], rows["goal_distance"]),
        steps=len(rows["actions"]), navigation_time=env.global_time, path=path_length,
        minimum_clearance=minimum, return_=float(sum(rows["rewards"])),
        discounted_return=float(sum(cfg.getfloat("train", "gamma") ** i * r for i, r in enumerate(rows["rewards"]))),
        trace=str(prefix.with_suffix(".npz")), trace_sha256=digest(prefix.with_suffix(".npz")),
        elapsed_seconds=time.perf_counter() - started, device=device,
        checkpoint_sha256=digest(weights(seed, "cv")))
    if row["behavior"] != "success":
        tick = root_tick(row, rows)
        row["root"] = {k: np.asarray(rows[k][tick]).tolist() for k in
                       ("goal_distance", "legal_clearance", "active_count", "visible_count", "local_density", "phase", "grid_action")}
        row["root"].update(tick=tick, time=tick*.25, executed_action=rows["actions"][tick],
                           raw_top=int(np.argmax(rows["raw"][tick])), final_top=int(np.argmax(rows["final"][tick])))
    save_json(path, row)
    return row


def collect(split, workers, device, cell=None):
    verify()
    manifest = dict(device=device, torch=torch.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        protocol_sha256=digest(OUT / "protocol.json"))
    path = OUT / "runtime.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Runtime changed")
    else:
        save_json(path, manifest)
    cases = range(80000, 80032) if split == "discovery" else range(81000, 81032)
    cells = CELLS if cell is None else [cell]
    tasks = [(split, seed, n, g, case) for seed in SEEDS for n, g in cells for case in cases]
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(device,)) as pool:
        for row in pool.map(episode, tasks, chunksize=4):
            records.append(row)
            if len(records) % 16 == 0:
                print("ATLAS", split, len(records), "/", len(tasks), row["terminal"],
                      "elapsed", round(time.perf_counter()-started, 1), flush=True)
    save_json(OUT / (split + "-atlas.json"), dict(records=records, elapsed_seconds=time.perf_counter()-started))


def selected_groups(records):
    groups = defaultdict(list)
    for row in records:
        if row["behavior"] != "success":
            groups[row["behavior"], row["people"], row["geometry"]].append(row)
    eligible = []
    for key, rows in groups.items():
        by_case = defaultdict(list)
        for row in rows:
            by_case[row["case"]].append(row)
        if len(by_case) >= 3:
            repeated = sum(len(values) >= 2 for values in by_case.values())
            chosen = [min(by_case[c], key=lambda r: r["seed"]) for c in sorted(by_case)[:3]]
            eligible.append(dict(key=list(key), cases=len(by_case), multiseed_cases=repeated, selected=chosen))
    return sorted(eligible, key=lambda g: (-g["cases"], -g["multiseed_cases"], tuple(g["key"])))[:3]


def select():
    records = json.loads((OUT / "discovery-atlas.json").read_text())["records"]
    save_json(OUT / "headroom-selection.json", dict(groups=selected_groups(records),
        atlas_sha256=digest(OUT / "discovery-atlas.json"), rules=RULES))


def restore_root(row, model, cfg, device):
    trace = np.load(row["trace"])
    if digest(Path(row["trace"])) != row["trace_sha256"]:
        raise ValueError("Trace changed")
    tick = row["root"]["tick"]
    policy = ValuePolicy(model, cfg, device)
    env = environment(cfg, policy, row["geometry"], row["people"])
    env.reset(options={"test_case": row["case"]})
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    frames = []
    for t in range(tick + 1):
        state = observer.observe(env)
        frames.append(policy.encode(state))
        if t == tick:
            break
        env.step(ActionXY(*trace["actions"][t]))
    return dict(episode={k: row[k] for k in ("people", "geometry", "case", "terminal")},
        tick=tick, state=state, snapshot=exact_snapshot(env, observer),
        history=window(frames, policy.length, "zero"), previous=trace["actions"][tick-1] if tick else None,
        archived_tail_actions=trace["actions"][tick:], archived_rewards=trace["rewards"][tick:],
        archived_remaining_seconds=(len(trace["actions"])-tick)*.25)


def attribution(parts, truth_parts, outcomes, native):
    safe_rescues = [i for i, r in enumerate(outcomes) if r["terminal"] == "reach_goal"
                    and r["minimum_clearance"] >= .02]
    if not safe_rescues:
        return dict(stage="no-safety-qualified-single-step-rescue", good=None,
                    successful_commands=sum(r["terminal"] == "reach_goal" for r in outcomes))
    good = max(safe_rescues, key=lambda i: outcomes[i]["return_"])
    raw_top, truth_top = parts["raw_top"], truth_parts["final_top"]
    if raw_top in safe_rescues:
        stage = "postprocessing"
    elif truth_top in safe_rescues:
        stage = "successor-query-interface"
    elif parts["raw_rank"][good] >= 10:
        stage = "value-ranking-residual"
    else:
        stage = "unresolved-ranking-pattern"
    return dict(stage=stage, good=good, successful_commands=sum(r["terminal"] == "reach_goal" for r in outcomes),
        safe_rescues=safe_rescues, raw_rank=parts["raw_rank"][good], final_rank=parts["final_rank"][good],
        truth_rank=truth_parts["final_rank"][good], blocked=parts["blocked"][good],
        delta_q=outcomes[good]["return_"]-outcomes[native]["return_"],
        native_q=outcomes[native]["return_"], good_q=outcomes[good]["return_"],
        good_clearance=outcomes[good]["minimum_clearance"], raw_top=raw_top, truth_top=truth_top)


@torch.inference_mode()
def headroom(rows, directory, device):
    verify()
    for row in rows:
        path = directory / f"{row['seed']}_{row['people']}_{row['geometry']}_{row['case']}.json"
        if path.exists():
            continue
        started = time.perf_counter()
        model, cfg = load(weights(row["seed"], "cv"), device)
        root = restore_root(row, model, cfg, device)
        policy = ValuePolicy(model, cfg, device)
        policy.history.extend(root["history"][:-1])
        policy.last_action = None if root["previous"] is None else ActionXY(*root["previous"])
        commands = np.asarray(policy.candidate_actions())
        scores = policy.score(root["state"])
        native = int(scores.argmax())
        if native != row["root"]["grid_action"]:
            raise ValueError("Frozen root selection changed")
        baseline = run_branches(root, model, cfg, device, [commands[native]])[0]
        parity = archive_parity(root, baseline, cfg)
        if not parity["passed"]:
            raise ValueError("Native continuation parity failed: " + str(parity))
        outcomes = []
        for action in range(80):
            branch = directory / "branches" / f"{path.stem}_{action:02d}.json"
            if branch.exists():
                outcome = json.loads(branch.read_text())
            else:
                outcome = baseline if action == native else run_branches(root, model, cfg, device, [commands[action]])[0]
                save_json(branch, outcome)
            outcomes.append(outcome)
            if (action+1) % 20 == 0:
                print("HEADROOM", row["case"], row["seed"], action+1, "/80", flush=True)
        trace = np.load(row["trace"])
        parts = {key: trace[key][root["tick"]].tolist() for key in
                 ("immediate_reward", "value", "raw", "blocked", "risk_penalty", "final", "clearance")}
        for key in ("raw", "final"):
            parts[key+"_rank"] = ranks(parts[key]).tolist()
            parts[key+"_top"] = int(np.argmax(parts[key]))
        truth_queries, _, steps = one_step(root, commands, cfg)
        true_values = score_queries(model, root["history"], truth_queries, device)
        truth_parts = stages(parts["immediate_reward"], true_values, parts["clearance"], cfg)
        result = dict(episode=row, native=native, commands=commands.tolist(), parity=parity,
            native_parts=parts, true_successor_parts=truth_parts,
            true_steps=steps, outcomes=outcomes,
            attribution=attribution(parts, truth_parts, outcomes, native),
            elapsed_seconds=time.perf_counter()-started,
            caveat="True successor only changes value queries with same actor support. "
                   "It does not fix native reward/filter approximation or establish an optimal Q.")
        save_json(path, result)
        print("LOCALIZED", row["case"], row["seed"], result["attribution"], flush=True)


def nominate():
    groups = json.loads((OUT / "headroom-selection.json").read_text())["groups"]
    for group in groups:
        results = [json.loads((OUT / "headroom" / f"{r['seed']}_{r['people']}_{r['geometry']}_{r['case']}.json").read_text())
                   for r in group["selected"]]
        labels = [r["attribution"]["stage"] for r in results]
        if len(set(labels)) == 1 and labels[0] in ("postprocessing", "successor-query-interface", "value-ranking-residual"):
            save_json(OUT / "nominee.json", dict(key=group["key"], stage=labels[0], results=[r["episode"] for r in results],
                rules=RULES, frozen_before_fresh=True))
            return
    save_json(OUT / "nominee.json", dict(key=None, reason="No three independent localized safety-qualified rescues"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "collect", "select", "headroom", "nominate", "fresh", "fresh-headroom"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze()
    elif args.stage == "collect":
        collect("discovery", args.workers, args.device)
    elif args.stage == "select":
        select()
    elif args.stage == "headroom":
        groups = json.loads((OUT / "headroom-selection.json").read_text())["groups"]
        headroom([r for g in groups for r in g["selected"]], OUT / "headroom", args.device)
    elif args.stage == "nominate":
        nominate()
    elif args.stage == "fresh":
        nominee = json.loads((OUT / "nominee.json").read_text())
        if nominee["key"] is not None:
            collect("fresh", args.workers, args.device, tuple(nominee["key"][1:]))
    else:
        nominee = json.loads((OUT / "nominee.json").read_text())
        if nominee["key"] is not None:
            records = json.loads((OUT / "fresh-atlas.json").read_text())["records"]
            by_case = defaultdict(list)
            for row in records:
                if row["behavior"] == nominee["key"][0]:
                    by_case[row["case"]].append(row)
            selected = [min(by_case[c], key=lambda r: r["seed"]) for c in sorted(by_case)[:4]]
            save_json(OUT / "fresh-headroom-selection.json", dict(rows=selected, nominee_sha256=digest(OUT/"nominee.json")))
            headroom(selected, OUT / "fresh-headroom", args.device)


if __name__ == "__main__":
    main()
