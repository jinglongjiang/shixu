"""Fresh three-arm test: immutable Parent/V0 and one all-unsafe release."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import save_json
from experiments.occlusion import source_hash
from experiments.online_action_commitment_v0 import CELLS, SEEDS, RULES as V0_RULES, compare
from experiments.repeatable_defect_audit import failure_label
from shixu.commitment import ActionCommitmentPolicy, low_progress
from shixu.commitment_release import AllUnsafeReleasePolicy
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/online-action-commitment-v01")
OLD = Path("outputs/online-action-commitment-v0")
CASES = tuple(range(87000, 87032))
ARMS = ("parent", "v0", "v01")
RULES = dict(
    change="Only when commitment exists (including this tick's new start), no candidate "
           "clearance>=original0.2m means immediate release and same-tick Parent choice. "
           "No rearm while trigger stays true. Trigger, duration, fallback, scoring unchanged.",
    inherited={k: V0_RULES[k] for k in ("scope", "trigger", "hold", "rearm", "randomness", "primary", "guard")},
    test="Unused87000..87031, three arms, four original checkpoints, six cells,32cases:2304episodes. "
         "No tuning, training, new roots or outcome-based stopping; evaluate nine original gates separately for V0/V0.1.",
    diagnostic="Count collisions on terminal steps still held with all-unsafe clearance, and paired "
               "V0/V0.1 outcomes. Eliminating held/all-unsafe labels by definition is not proof of safer outcomes. "
               "Report total collisions, former V0 collision outcomes, rescues and destroyed successes.",
    limits="Full closed-loop single-rule intervention identifies its net effect, not whether all "
           "remaining damage is caused by initiation rather than duration, action choice or their interaction. "
           "No automatic V0.2, duration sweep, adaptive network, literature search or training.")
_DEVICE = None
_MODELS = {}


def read(path):
    return json.loads(path.read_text())


def parent_hash():
    root = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256()
    for folder in ("shixu", "vendor/crowd_sim"):
        for path in sorted((root/folder).rglob("*.py")):
            if path.name in ("commitment.py", "commitment_release.py"):
                continue
            sha.update(str(path.relative_to(root)).encode())
            sha.update(path.read_bytes())
    return sha.hexdigest()


def helpers():
    return {str(p): digest(p) for p in (Path(__file__),
        Path("experiments/online_action_commitment_v0.py"), Path("shixu/commitment.py"),
        Path("experiments/forecast_evidence.py"), Path("experiments/repeatable_defect_audit.py"),
        Path("experiments/multihorizon_control.py"))}


def freeze(device):
    old = read(OLD/"protocol.json")
    if parent_hash() != old["parent_sha256"]:
        raise ValueError("Parent changed since V0")
    for path, sha in old["helpers"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Frozen V0 helper changed")
    # Frozen V0's full source hash also binds its original commitment module.
    root = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256()
    for folder in ("shixu", "vendor/crowd_sim"):
        for path in sorted((root/folder).rglob("*.py")):
            if path.name == "commitment_release.py":
                continue
            sha.update(str(path.relative_to(root)).encode())
            sha.update(path.read_bytes())
    if sha.hexdigest() != old["core_sha256"]:
        raise ValueError("Original V0 source changed")
    for c in old["checkpoints"].values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Checkpoint changed")
    search = subprocess.run(["rg", "-l", "--glob", "*.json",
        r'"case"\s*:\s*870(?:0[0-9]|[12][0-9]|3[01])\b|"(?:test_case_start|eval_case_start|case_start)"\s*:\s*87000\b',
        "outputs"], text=True, capture_output=True)
    if search.returncode != 1:
        raise ValueError("Case block used or search failed: "+search.stdout+search.stderr)
    save_json(OUT/"protocol.json", dict(rules=RULES, cases=CASES, seeds=SEEDS, cells=CELLS, arms=ARMS,
        checkpoints=old["checkpoints"], parent_sha256=parent_hash(), core_sha256=source_hash(), helpers=helpers(),
        inherited_protocol_sha256=digest(OLD/"protocol.json"), case_unused_check=dict(command=search.args,
            exit_code=search.returncode, matches=search.stdout, scope="Existing local JSON case/start fields only; not external assets."),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime=dict(device=device, torch=torch.__version__, python=platform.python_version(),
            gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
            tf32_cudnn=torch.backends.cudnn.allow_tf32, tf32_matmul=torch.backends.cuda.matmul.allow_tf32)))
    print("FROZEN", len(CASES)*len(SEEDS)*len(CELLS)*len(ARMS), "episodes", flush=True)


def verify():
    p = read(OUT/"protocol.json")
    if (p["rules"] != RULES or p["core_sha256"] != source_hash() or
            p["parent_sha256"] != parent_hash() or p["helpers"] != helpers() or
            p["inherited_protocol_sha256"] != digest(OLD/"protocol.json")):
        raise ValueError("Frozen contract changed")
    for c in p["checkpoints"].values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Weights changed")
    return p


def init_worker(device):
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


@torch.inference_mode()
def episode(task):
    seed, people, geometry, case, arm, protocol_sha = task
    prefix = OUT/"episodes"/f"{seed}-{people}-{geometry}-{case}-{arm}"
    path = prefix.with_suffix(".json")
    if path.exists():
        record = read(path)
        if record["protocol_sha256"] != protocol_sha or digest(prefix.with_suffix(".npz")) != record["trace_sha256"]:
            raise ValueError("Cached episode differs")
        return record
    start = time.perf_counter()
    if seed not in _MODELS:
        _MODELS[seed] = load(weights(seed, "cv"), _DEVICE)
    model, cfg = _MODELS[seed]
    policy = {"parent": ValuePolicy, "v0": ActionCommitmentPolicy, "v01": AllUnsafeReleasePolicy}[arm](model, cfg, _DEVICE)
    policy.reset()
    env = environment(cfg, policy, geometry, people)
    env.reset(options={"test_case": case})
    initial = [env.robot.get_full_state().to_array().tolist()]+[h.get_full_state().to_array().tolist() for h in env.humans]
    initial_hash = hashlib.sha256(json.dumps(initial, separators=(",", ":")).encode()).hexdigest()
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    trace = defaultdict(list)
    events, proposals, distances = [], [], []
    while True:
        state = observer.observe(env)
        robot = state.self_state
        distance = float(np.hypot(robot.px-robot.gx, robot.py-robot.gy))
        commands = np.asarray(policy.candidate_actions())
        previous = np.array([np.nan, np.nan]) if policy.last_action is None else np.asarray(policy.last_action)
        tick_start = time.perf_counter()
        action = policy.predict(state)
        latency = time.perf_counter()-tick_start
        selected = np.flatnonzero(np.all(commands == np.asarray(action), axis=1))
        if len(selected) != 1:
            raise ValueError("Action not a once-smoothed native command")
        index = int(selected[0])
        trace["tokens"].append(policy.encode(state))
        trace["robot"].append(robot.to_array())
        humans = np.zeros((people, 5))
        present, measured, age = np.zeros(people, bool), np.zeros(people, bool), np.zeros(people)
        for key, human, observed, stamp in zip(state.track_ids, state.human_states, state.observed, state.ages):
            humans[key], present[key], measured[key], age[key] = human.to_array(), True, observed, stamp
        for k, v in dict(humans=humans, present=present, observed=measured, ages=age).items():
            trace[k].append(v)
        for k, v in dict(track_count=state.track_count, position=robot.position, time=env.global_time,
            previous=previous, actions=action, grid_action=index, goal_distance=distance,
            inference_seconds=latency).items():
            trace[k].append(v)
        distances.append(distance)
        if arm != "parent":
            decision = policy.last_decision.copy()
            if decision["selected_grid"] != index:
                raise ValueError("Log differs from execution")
            events.append(decision)
            proposals.append(decision["proposal"])
        else:
            proposals.append(index)
        old = np.asarray(env.robot.get_position())
        _, reward, done, truncated, info = env.step(action)
        trace["rewards"].append(reward)
        trace["actual_clearance"].append(info["dmin"])
        trace["path_increment"].append(float(np.linalg.norm(np.asarray(env.robot.get_position())-old)))
        if done or truncated:
            break
    max_actors = max(len(t) for t in trace["tokens"])
    trace["tokens"] = [np.pad(t, ((0, max_actors-len(t)), (0, 0))) for t in trace["tokens"]]
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz = prefix.with_suffix(".npz")
    if npz.exists():
        with np.load(npz) as existing:
            for k, values in arrays.items():
                if k != "inference_seconds":
                    np.testing.assert_array_equal(existing[k], values)
            arrays["inference_seconds"] = existing["inference_seconds"]
    else:
        np.savez_compressed(npz, **arrays)
    result = dict(seed=seed, people=people, geometry=geometry, case=case, arm=arm,
        terminal=info["event"], behavior=failure_label(info["event"], distances), steps=len(distances),
        navigation_time=env.global_time, initial_world_sha256=initial_hash,
        path=float(sum(trace["path_increment"])), minimum_clearance=float(min(trace["actual_clearance"])),
        return_=float(sum(trace["rewards"])),
        discounted_return=float(sum(cfg.getfloat("train", "gamma")**i*r for i, r in enumerate(trace["rewards"]))),
        starts=sum(v["started"] for v in events), held_steps=sum(v["held"] for v in events),
        commitment_seconds=sum(v["held"] for v in events)*.25,
        safety_releases=sum(v["release"] == "safety-blocked" for v in events),
        all_unsafe_releases=sum(v["release"] == "all-unsafe" for v in events),
        completed_budgets=sum(v["release"] == "budget-completed" for v in events),
        no_margin_safe_steps=sum(v["no_margin_safe_candidate"] for v in events),
        held_no_margin_safe_steps=sum(v["held"] and v["no_margin_safe_candidate"] for v in events),
        collision_while_holding=bool(info["event"] == "collision" and events and events[-1]["held"]),
        collision_while_holding_all_unsafe=bool(info["event"] == "collision" and events and
                                             events[-1]["held"] and events[-1]["no_margin_safe_candidate"]),
        decisions=events, proposals=proposals, protocol_sha256=protocol_sha, trace=str(npz),
        trace_sha256=digest(npz), elapsed_seconds=time.perf_counter()-start,
        inference_seconds=float(np.sum(arrays["inference_seconds"])), parameters=sum(p.numel() for p in model.parameters()))
    save_json(path, result)
    return result


def summarize(records):
    grouped = defaultdict(dict)
    for r in records:
        key = (r["seed"], r["people"], r["geometry"], r["case"])
        if r["arm"] in grouped[key]:
            raise ValueError("Duplicate arm")
        grouped[key][r["arm"]] = r
    if len(records) != 2304 or len(grouped) != 768 or any(set(v) != set(ARMS) for v in grouped.values()):
        raise ValueError("Incomplete triple")
    expected = {(s,n,g,c) for s in SEEDS for n,g in CELLS for c in CASES}
    if set(grouped) != expected:
        raise ValueError("Test block differs")
    if any(len({r["initial_world_sha256"] for r in arms.values()}) != 1 for arms in grouped.values()):
        raise ValueError("Unmatched triple")
    results = {}
    for arm in ("v0", "v01"):
        selected = [dict(r, arm="commitment" if r["arm"] == arm else "parent")
                    for r in records if r["arm"] in ("parent", arm)]
        results[arm] = compare(selected)
    contrasts = []
    for key, arms in sorted(grouped.items()):
        a, b = arms["v0"], arms["v01"]
        contrasts.append(dict(seed=key[0], people=key[1], geometry=key[2], case=key[3],
            v0_terminal=a["terminal"], v01_terminal=b["terminal"],
            q_delta=b["discounted_return"]-a["discounted_return"],
            time_delta=b["navigation_time"]-a["navigation_time"],
            clearance_delta=b["minimum_clearance"]-a["minimum_clearance"]))
    totals = {}
    for g in ("all", "circle", "square"):
        totals[g] = {}
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm and (g == "all" or r["geometry"] == g)]
            success = [r["navigation_time"] for r in rows if r["terminal"] == "reach_goal"]
            totals[g][arm] = dict(episodes=len(rows), terminals=dict(Counter(r["terminal"] for r in rows)),
                low_timeouts=sum(r["behavior"] == "low-progress-timeout" for r in rows),
                mean_q=float(np.mean([r["discounted_return"] for r in rows])),
                success_time_mean=float(np.mean(success)), starts=sum(r["starts"] for r in rows),
                held_seconds=sum(r["commitment_seconds"] for r in rows),
                all_unsafe_releases=sum(r["all_unsafe_releases"] for r in rows),
                held_all_unsafe_collisions=sum(r["collision_while_holding_all_unsafe"] for r in rows),
                holding_collisions=sum(r["collision_while_holding"] for r in rows),
                held_all_unsafe_steps=sum(r["held_no_margin_safe_steps"] for r in rows))
    return dict(results=results, totals=totals, release_contrasts=contrasts,
        v0_collision_followups=dict(Counter(r["v01_terminal"] for r in contrasts if r["v0_terminal"] == "collision")),
        v0_holding_all_unsafe_collision_followups=dict(Counter(arms["v01"]["terminal"] for arms in grouped.values()
            if arms["v0"]["collision_while_holding_all_unsafe"])), limits=RULES["limits"])


def execute(workers):
    p = verify()
    tasks = [(s, n, g, c, arm, digest(OUT/"protocol.json")) for s in SEEDS for n, g in CELLS
             for c in CASES for arm in ARMS]
    start, records = time.perf_counter(), []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(p["runtime"]["device"],)) as pool:
        futures = [pool.submit(episode, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 96 == 0:
                print("CLOSED_LOOP", len(records), "/2304", "seconds", round(time.perf_counter()-start, 1), flush=True)
    records.sort(key=lambda r: (r["seed"], r["people"], r["geometry"], r["case"], r["arm"]))
    save_json(OUT/"episode-summary.json", dict(records=[{k:v for k,v in r.items() if k not in ("decisions", "proposals")}
        for r in records], elapsed_seconds=time.perf_counter()-start))
    result = summarize(records)
    result.update(protocol_sha256=digest(OUT/"protocol.json"), episodes=len(records),
                  elapsed_seconds=time.perf_counter()-start)
    save_json(OUT/"summary.json", result)
    print("VERDICT", result["results"]["v01"]["verdict"], result["results"]["v01"]["gates"], flush=True)


def validate():
    verify()
    records = [read(p) for p in sorted((OUT/"episodes").glob("*.json"))]
    result = summarize(records)
    saved = read(OUT/"summary.json")
    for key, value in result.items():
        if saved[key] != value:
            raise ValueError("Summary differs from full metadata")
    groups = defaultdict(dict)
    steps, parity, smooth_error = Counter(), Counter(), 0.
    for r in records:
        key = (r["seed"], r["people"], r["geometry"], r["case"])
        groups[key][r["arm"]] = r
        if digest(Path(r["trace"])) != r["trace_sha256"] or r["protocol_sha256"] != digest(OUT/"protocol.json"):
            raise ValueError("Trace/protocol hash differs")
        _, cfg = load(weights(r["seed"], "cv"), "cpu") if r["seed"] not in _MODELS else _MODELS[r["seed"]]
        _MODELS[r["seed"]] = (None, cfg)
        from shixu.policy import actions
        grid = np.asarray(actions(cfg))
        with np.load(r["trace"]) as t:
            steps[r["arm"]] += len(t["actions"])
            expected = grid[t["grid_action"]].copy()
            expected[1:] = .3*t["previous"][1:]+.7*expected[1:]
            error = float(np.max(np.abs(expected-t["actions"])))
            smooth_error = max(smooth_error, error)
            if error != 0 or np.any(np.diff(t["time"]) != .25):
                raise ValueError("Execution/clock contract differs")
            q = sum(cfg.getfloat("train", "gamma")**i*v for i, v in enumerate(t["rewards"]))
            if abs(q-r["discounted_return"]) > 1e-12:
                raise ValueError("Reward target differs")
            for tick, d in enumerate(r["decisions"]):
                trigger = low_progress(t["goal_distance"][max(0,tick-8):tick+1])
                if d["tick"] != tick or d["time"] != t["time"][tick] or d["trigger"] != trigger or d["selected_grid"] != t["grid_action"][tick]:
                    raise ValueError("Decision log/causality differs")
                if d["held"] and (d["blocked"] or r["arm"] == "v01" and d["no_margin_safe_candidate"]):
                    raise ValueError("Held blocked/all-unsafe command")
                if d["release"] == "all-unsafe" and (r["arm"] != "v01" or not d["no_margin_safe_candidate"] or d["held"] or d["proposal"] != d["selected_grid"]):
                    raise ValueError("Same-tick release failed")
    for arms in groups.values():
        for left, right, eligible in (("parent", "v0", not arms["v0"]["held_steps"]),
                                     ("v0", "v01", not arms["v01"]["all_unsafe_releases"]),
                                     ("parent", "v01", not arms["v01"]["held_steps"])):
            if eligible:
                with np.load(arms[left]["trace"]) as a, np.load(arms[right]["trace"]) as b:
                    np.testing.assert_array_equal(a["actions"], b["actions"])
                for k in ("terminal", "discounted_return", "minimum_clearance", "navigation_time"):
                    if arms[left][k] != arms[right][k]:
                        raise ValueError("No-intervention parity differs")
                parity[left+"/"+right] += 1
    metadata_hashes = {str(p): digest(p) for p in sorted((OUT/"episodes").glob("*.json"))}
    save_json(OUT/"validation.json", dict(episodes=len(records), matched_initial_worlds=len(groups),
        control_steps=dict(steps), max_smoothing_error=smooth_error, exact_no_intervention_pairs=dict(parity),
        protocol_sha256=digest(OUT/"protocol.json"), summary_sha256=digest(OUT/"summary.json"),
        episode_summary_sha256=digest(OUT/"episode-summary.json"),
        metadata_bundle_sha256=hashlib.sha256(json.dumps(metadata_hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        limits="Float32 state vectors use native to_array; commands/time are float64. Timing is descriptive under concurrent GPU workers, not a matched-state latency benchmark."))
    print("VALIDATED", len(records), dict(parity), dict(steps), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("freeze", "run", "validate"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze(args.device)
    elif args.stage == "run":
        execute(args.workers)
    else:
        validate()


if __name__ == "__main__":
    main()
