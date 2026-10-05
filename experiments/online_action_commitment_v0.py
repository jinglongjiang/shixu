"""One frozen causal commitment rule, matched with the untouched parent."""

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
from experiments.repeatable_defect_audit import failure_label
from shixu.commitment import ActionCommitmentPolicy, low_progress, native_blocked
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/online-action-commitment-v0")
ATLAS = Path("outputs/repeatable_defect_audit")
SEEDS = (419, 443, 467, 491)
CELLS = tuple((n, g) for n in (5, 10, 20) for g in ("circle", "square"))
CASES = tuple(range(86000, 86032))
RULES = {
    "scope": "One evaluation-only wrapper, same existing four CV IL50/RL3000 checkpoints. "
             "No training, new critic, reward, action, observation, history or native filter changes.",
    "trigger": "Nine actual control-frame distances cover exactly2s. Current distance>1m and "
               "d[t-8]-d[t]<=0.2m;1e-12m roundoff tolerance only. Require parent-selected grid "
               "action not blocked by the original hard filter. No future features or old root times.",
    "hold": "At most8 executed0.25s steps including the trigger step; fixed grid index, alpha0.3 "
            "smoothing recomputed once every step. Each real observation updates tracker/history. "
            "Held steps skip critic ranking but recheck the original hard filter on all actual "
            "smoothed candidates. On blocking: release and call parent score on the SAME step.",
    "safety": "Exact existing mask: if any candidate clearance>=0.2m, block those below0.2m; "
              "otherwise preserve parent's no-hard-mask fallback. Risk remains in parent scoring, "
              "not an extra veto. Log fallback, held clearances, releases and actual minimum gaps.",
    "rearm": "After budget exhaustion or safety release, condition must clear and then recur "
             "before another commitment. No immediate recommit while continuously low-progress.",
    "randomness": "Consume the same one NumPy draw as native predict per control tick even when "
                  "holding, so action commitment does not silently change simulator RNG consumption.",
    "preflight": "Old1024 parent traces only. Report first eligible timing for four roots and "
                 "trigger opportunity on success traces/current-high-speed frames. No outcomes "
                 "under the new wrapper, no rule selection. Stop full run if zero of four roots "
                 "has an eligible trigger at or before its old diagnostic root.",
    "test": "Unused86000..86031 x4 seeds x5/10/20 xcircle/square x2 arms =1536 native closed-loop "
            "episodes. Freeze before running either arm; paired initial-world hash required. "
            "Same local GPU/native scalar inference; no result-based stopping or parameter changes.",
    "primary": "Square low-progress timeouts: original offline32-step/8s definition unchanged. "
               "Require strictly fewer;>=4 distinct numeric case IDs rescued failure->success "
               "with new full-episode minimum clearance>=0.02m; positive low-timeout and SR net "
               "change in>=2 population cells;>=3/4 seeds nonnegative on both low-timeout and SR; "
               "square collision increase<=1pp. Counts, not significance claims.",
    "guard": "In each geometry, parent-success->failure count<=2pp of all384 paired episodes "
             "(<=7); circle net SR loss<=1pp and collision increase<=1pp (<=3 each). Frozen "
             "operational meaning of no large damage, not post-hoc safety tuning. Report all gaps.",
    "decision": "All gates pass: A_FIXED_COMMITMENT_EFFECTIVE; square rescues but safety/guard "
                "fails: B_RESCUE_WITH_UNACCEPTABLE_DAMAGE; otherwise C_FIXED_RULE_NOT_CONFIRMED. "
                "B does not prove adaptive commitment will work. C closes this fixed rule only. "
                "No retuning or automatic new-method development after this experiment.",
}
_DEVICE = None
_MODELS = {}


def read(path):
    return json.loads(path.read_text())


def parent_hash():
    root = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256()
    for folder in ("shixu", "vendor/crowd_sim"):
        for path in sorted((root/folder).rglob("*.py")):
            if path == root/"shixu/commitment.py":
                continue
            sha.update(str(path.relative_to(root)).encode())
            sha.update(path.read_bytes())
    return sha.hexdigest()


def experiment_hashes():
    return {str(path): digest(path) for path in [Path(__file__), Path("experiments/forecast_evidence.py"),
        Path("experiments/repeatable_defect_audit.py"), Path("experiments/multihorizon_control.py")]}


def trigger_ticks(trace):
    distances = trace["goal_distance"]
    return [t for t in range(8, len(distances)) if low_progress(distances[t-8:t+1]) and
            not bool(trace["blocked"][t, int(trace["grid_action"][t])])]


def preflight():
    paths = [ATLAS/"discovery-atlas.json", ATLAS/"exploratory-fresh-square-5-atlas.json",
             ATLAS/"exploratory-fresh-square-10-atlas.json", ATLAS/"pooled-fresh-selection.json"]
    records = [r for p in paths[:3] for r in read(p)["records"]]
    primary = read(paths[3])["rows"]
    roots, counts = [], Counter()
    for row in records:
        with np.load(row["trace"]) as trace:
            if digest(Path(row["trace"])) != row["trace_sha256"]:
                raise ValueError("Old trace hash differs")
            ticks = trigger_ticks(trace)
            counts["episodes"] += 1
            counts["eligible_frames"] += len(ticks)
            counts["episodes_with_trigger"] += bool(ticks)
            if row["terminal"] == "reach_goal":
                counts["successful_episodes"] += 1
                counts["successful_episodes_with_trigger"] += bool(ticks)
            for t in range(8, len(trace["goal_distance"])):
                progress = trace["goal_distance"][t-8]-trace["goal_distance"][t]
                counts["examined_frames"] += 1
                if progress/2 > .25:
                    counts["fast_goal_progress_frames"] += 1
                    counts["fast_goal_progress_trigger_frames"] += t in ticks
                if t in ticks and np.linalg.norm(trace["previous"][t]) > .5:
                    counts["trigger_while_speed_above_05"] += 1
            matched = next((r for r in primary if (r["seed"], r["people"], r["case"]) ==
                            (row["seed"], row["people"], row["case"]) and row["geometry"] == "square"), None)
            if matched:
                roots.append(dict(seed=row["seed"], people=row["people"], case=row["case"],
                    old_root_seconds=matched["root"]["time"],
                    first_eligible_seconds=ticks[0]*.25 if ticks else None,
                    trigger_at_or_before_root=bool(ticks and ticks[0] <= matched["root"]["tick"])))
    if len(roots) != 4:
        raise ValueError("Preflight did not find all four primary roots")
    save_json(OUT/"preflight.json", dict(roots=roots, counts=dict(counts),
        passed=any(r["trigger_at_or_before_root"] for r in roots),
        archives={str(p): digest(p) for p in paths},
        limitation="Archived eligibility only, not new-policy trigger/outcome. High actual speed "
                   "with low net goal progress is not automatically a false trigger. Zero triggers "
                   "at>0.25m/s net goal progress follows the rule and does not prove selectivity."))
    print("PREFLIGHT", roots, dict(counts), flush=True)


def freeze(device):
    old = read(Path("outputs/multistep-headroom-decision-test/protocol.json"))
    if parent_hash() != old["core_sha256"]:
        raise ValueError("Native parent changed")
    ck = old["checkpoints"]
    for c in ck.values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Existing checkpoint changed")
    # Search exact existing case fields, not numbers coincidentally in metrics.
    search = subprocess.run(["rg", "-l", "--glob", "*.json",
        r'"case"\s*:\s*860(?:0[0-9]|[12][0-9]|3[01])\b|"(?:test_case_start|eval_case_start|case_start)"\s*:\s*86000\b',
        "outputs"], text=True, capture_output=True)
    if search.returncode != 1:
        raise ValueError("New case block not clean or search failed: "+search.stdout+search.stderr)
    save_json(OUT/"protocol.json", dict(rules=RULES, cases=CASES, seeds=SEEDS, cells=CELLS,
        checkpoints=ck, parent_sha256=parent_hash(), core_sha256=source_hash(), helpers=experiment_hashes(),
        preflight_sha256=digest(OUT/"preflight.json"),
        case_unused_check=dict(command=search.args, exit_code=search.returncode, matches=search.stdout,
            scope="Exact case/start fields in existing local JSON evaluation assets; old training "
                  "protocols use12000 IL and20000 RL, not this test block. No claim about external assets."),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime=dict(device=device, torch=torch.__version__, python=platform.python_version(),
            gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
            tf32_cudnn=torch.backends.cudnn.allow_tf32, tf32_matmul=torch.backends.cuda.matmul.allow_tf32)))


def verify():
    p = read(OUT/"protocol.json")
    if p["rules"] != RULES or p["core_sha256"] != source_hash() or p["parent_sha256"] != parent_hash():
        raise ValueError("Frozen scientific contract changed")
    if p["helpers"] != experiment_hashes() or p["preflight_sha256"] != digest(OUT/"preflight.json"):
        raise ValueError("Frozen diagnostic changed")
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
    name = f"{seed}-{people}-{geometry}-{case}-{arm}"
    prefix = OUT/"episodes"/name
    path = prefix.with_suffix(".json")
    if path.exists():
        record = read(path)
        if record["protocol_sha256"] != protocol_sha or digest(prefix.with_suffix(".npz")) != record["trace_sha256"]:
            raise ValueError("Completed episode differs")
        return record
    start = time.perf_counter()
    if seed not in _MODELS:
        _MODELS[seed] = load(weights(seed, "cv"), _DEVICE)
    model, cfg = _MODELS[seed]
    policy = (ValuePolicy if arm == "parent" else ActionCommitmentPolicy)(model, cfg, _DEVICE)
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
            raise ValueError("Executed action must match one once-smoothed native candidate")
        index = int(selected[0])
        token = policy.encode(state)
        trace["tokens"].append(token)
        trace["robot"].append(robot.to_array())
        humans, present, measured, age = np.zeros((people, 5)), np.zeros(people, bool), np.zeros(people, bool), np.zeros(people)
        for key, human, observed, stamp in zip(state.track_ids, state.human_states, state.observed, state.ages):
            humans[key], present[key], measured[key], age[key] = human.to_array(), True, observed, stamp
        for k, v in dict(humans=humans, present=present, observed=measured, ages=age).items():
            trace[k].append(v)
        trace["track_count"].append(state.track_count)
        trace["position"].append(robot.position)
        trace["time"].append(env.global_time)
        trace["previous"].append(previous)
        trace["actions"].append(action)
        trace["grid_action"].append(index)
        trace["goal_distance"].append(distance)
        trace["inference_seconds"].append(latency)
        distances.append(distance)
        if arm == "commitment":
            decision = policy.last_decision.copy()
            if decision["selected_grid"] != index:
                raise ValueError("Commitment log differs from execution")
            proposals.append(decision["proposal"])
            events.append(decision)
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
        arrays["inference_seconds"] = np.load(npz)["inference_seconds"]
    else:
        np.savez_compressed(npz, **arrays)
    result = dict(seed=seed, people=people, geometry=geometry, case=case, arm=arm,
        terminal=info["event"], behavior=failure_label(info["event"], distances),
        steps=len(distances), navigation_time=env.global_time, initial_world_sha256=initial_hash,
        path=float(sum(trace["path_increment"])), minimum_clearance=float(min(trace["actual_clearance"])),
        return_=float(sum(trace["rewards"])),
        discounted_return=float(sum(cfg.getfloat("train", "gamma")**i*r for i, r in enumerate(trace["rewards"]))),
        starts=sum(v["started"] for v in events), held_steps=sum(v["held"] for v in events),
        commitment_seconds=sum(v["held"] for v in events)*.25,
        safety_releases=sum(v["release"] == "safety-blocked" for v in events),
        completed_budgets=sum(v["release"] == "budget-completed" for v in events),
        no_margin_safe_steps=sum(v["no_margin_safe_candidate"] for v in events),
        decisions=events, proposals=proposals, protocol_sha256=protocol_sha,
        trace=str(npz), trace_sha256=digest(npz), elapsed_seconds=time.perf_counter()-start,
        inference_seconds=float(np.sum(arrays["inference_seconds"])), parameters=sum(p.numel() for p in model.parameters()))
    save_json(path, result)
    return result


def compare(records):
    grouped = defaultdict(dict)
    for r in records:
        grouped[r["seed"], r["people"], r["geometry"], r["case"]][r["arm"]] = r
    pairs = []
    for key, arms in sorted(grouped.items()):
        if set(arms) != {"parent", "commitment"}:
            raise ValueError("Incomplete pair")
        parent, custom = arms["parent"], arms["commitment"]
        if parent["initial_world_sha256"] != custom["initial_world_sha256"]:
            raise ValueError("Paired worlds differ")
        pairs.append(dict(seed=key[0], people=key[1], geometry=key[2], case=key[3],
            parent_terminal=parent["terminal"], custom_terminal=custom["terminal"],
            parent_behavior=parent["behavior"], custom_behavior=custom["behavior"],
            rescued=parent["terminal"] != "reach_goal" and custom["terminal"] == "reach_goal",
            qualified_rescue=parent["terminal"] != "reach_goal" and custom["terminal"] == "reach_goal" and custom["minimum_clearance"] >= .02,
            damaged=parent["terminal"] == "reach_goal" and custom["terminal"] != "reach_goal",
            q_delta=custom["discounted_return"]-parent["discounted_return"],
            parent_time=parent["navigation_time"], custom_time=custom["navigation_time"],
            parent_clearance=parent["minimum_clearance"], custom_clearance=custom["minimum_clearance"],
            starts=custom["starts"], commitment_seconds=custom["commitment_seconds"]))
    square = [v for v in pairs if v["geometry"] == "square"]
    circle = [v for v in pairs if v["geometry"] == "circle"]
    def net(rows, label):
        if label == "low":
            return (sum(v["parent_behavior"] == "low-progress-timeout" for v in rows)-
                    sum(v["custom_behavior"] == "low-progress-timeout" for v in rows))
        return (sum(v["custom_terminal"] == "reach_goal" for v in rows)-
                sum(v["parent_terminal"] == "reach_goal" for v in rows))
    def collision_delta(rows):
        return sum(v["custom_terminal"] == "collision" for v in rows)-sum(v["parent_terminal"] == "collision" for v in rows)
    population_stats = {str(n): dict(low_timeout_reduction=net([v for v in square if v["people"] == n], "low"),
        success_gain=net([v for v in square if v["people"] == n], "success")) for n in (5, 10, 20)}
    seed_stats = {str(s): dict(low_timeout_reduction=net([v for v in square if v["seed"] == s], "low"),
        success_gain=net([v for v in square if v["seed"] == s], "success")) for s in SEEDS}
    gates = dict(square_low_timeout_net_reduction=net(square, "low") > 0,
        four_distinct_qualified_rescue_cases=len({v["case"] for v in square if v["qualified_rescue"]}) >= 4,
        two_population_net_improvements=sum(v["low_timeout_reduction"] > 0 and v["success_gain"] > 0 for v in population_stats.values()) >= 2,
        three_seed_nonnegative=sum(v["low_timeout_reduction"] >= 0 and v["success_gain"] >= 0 for v in seed_stats.values()) >= 3,
        square_collision_guard=collision_delta(square)/len(square) <= .01,
        square_success_damage_guard=sum(v["damaged"] for v in square)/len(square) <= .02,
        circle_success_damage_guard=sum(v["damaged"] for v in circle)/len(circle) <= .02,
        circle_net_success_guard=-net(circle, "success")/len(circle) <= .01,
        circle_collision_guard=collision_delta(circle)/len(circle) <= .01)
    unsafe = not all(gates[k] for k in gates if "guard" in k)
    verdict = ("A_FIXED_COMMITMENT_EFFECTIVE" if all(gates.values()) else
               "B_RESCUE_WITH_UNACCEPTABLE_DAMAGE" if any(v["rescued"] for v in square) and unsafe else
               "C_FIXED_RULE_NOT_CONFIRMED")
    cells = []
    for n, geometry in CELLS:
        cell = dict(people=n, geometry=geometry)
        for arm in ("parent", "commitment"):
            r = [v for v in records if (v["people"], v["geometry"], v["arm"]) == (n, geometry, arm)]
            times = [v["navigation_time"] for v in r if v["terminal"] == "reach_goal"]
            cell[arm] = dict(episodes=len(r), terminals=dict(Counter(v["terminal"] for v in r)),
                low_timeouts=sum(v["behavior"] == "low-progress-timeout" for v in r),
                success_time_mean=float(np.mean(times)) if times else None,
                clearance_mean=float(np.mean([v["minimum_clearance"] for v in r])),
                clearance_min=float(min(v["minimum_clearance"] for v in r)),
                starts=sum(v["starts"] for v in r), commitment_seconds=sum(v["commitment_seconds"] for v in r),
                inference_seconds=sum(v["inference_seconds"] for v in r))
        cells.append(cell)
    return dict(verdict=verdict, gates=gates, cells=cells, population_stats=population_stats,
        seed_stats=seed_stats, square_low_timeout_reduction=net(square, "low"),
        square_success_gain=net(square, "success"), circle_success_gain=net(circle, "success"),
        square_collision_delta=collision_delta(square), circle_collision_delta=collision_delta(circle),
        square_damaged=sum(v["damaged"] for v in square), circle_damaged=sum(v["damaged"] for v in circle),
        square_qualified_rescue_cases=sorted({v["case"] for v in square if v["qualified_rescue"]}),
        pairs=pairs, limits="Fixed-rule development test, not paper-level validation or proof of "
                            "frequent replanning as unique cause. This test block is consumed, not fresh again.")


def execute(workers):
    p = verify()
    if not read(OUT/"preflight.json")["passed"]:
        print("PREFLIGHT_FAILED_NO_TEST", flush=True)
        return
    tasks = [(seed, n, g, c, arm, digest(OUT/"protocol.json")) for seed in SEEDS for n, g in CELLS
             for c in CASES for arm in ("parent", "commitment")]
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(p["runtime"]["device"],)) as pool:
        futures = [pool.submit(episode, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 64 == 0:
                print("CLOSED_LOOP", len(records), "/1536", "seconds", round(time.perf_counter()-started, 1), flush=True)
    records.sort(key=lambda r: (r["seed"], r["people"], r["geometry"], r["case"], r["arm"]))
    save_json(OUT/"episode-summary.json", dict(records=[{k:v for k,v in r.items() if k not in ("decisions", "proposals")}
                                                       for r in records], elapsed_seconds=time.perf_counter()-started))
    result = compare(records)
    result.update(protocol_sha256=digest(OUT/"protocol.json"), episodes=len(records),
                  elapsed_seconds=time.perf_counter()-started)
    save_json(OUT/"summary.json", result)
    print("VERDICT", result["verdict"], result["gates"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "freeze", "run"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "preflight":
        preflight()
    elif args.stage == "freeze":
        freeze(args.device)
    else:
        execute(args.workers)


if __name__ == "__main__":
    main()
