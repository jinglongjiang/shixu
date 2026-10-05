"""Frozen five-person-trained initiation gates, four arms and a fresh block."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import save_json
from experiments.occlusion import source_hash
from experiments.online_action_commitment_v0 import CELLS, SEEDS, compare
from experiments.repeatable_defect_audit import failure_label
from shixu.commit_advantage import AdvantageEstimator, CommitAdvantagePolicy
from shixu.commitment_release import AllUnsafeReleasePolicy
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy, actions
from shixu.runner import environment


OUT = Path("outputs/online-commit-advantage")
LABELS = Path("outputs/commit-advantage-probe")
CASES = tuple(range(99000, 99032))
ARMS = ("parent", "v01", "linear", "mlp")
RULES = dict(
    scope="Freeze original four critics. Train/validation labels only native5-person circle. "
          "Deploy existing case-only ridge/MLP fitted on90000..90063, selected on90064..90079. "
          "No test refit, reward/action/filter/critic/duration changes. Shared gate across four critics, "
          "not four independently trained gates; MLP retained as an unsuccessful seed-heldout control.",
    initiation="Use the frozen feature extractor, same18 semantic/128 latent features. At inactive "
               "ticks after8 previous real frames and own-goal distance>1m, A>0 may start current "
               "Parent argmax grid if safe. No low-progress prerequisite. No case/seed/geometry input.",
    option="Original smoothing0.3 each real0.25s tick. Max8steps including initiation. Original "
           "V0.1 clearance0.2m release on held-grid unsafe OR all-unsafe, same-tick Parent choice. "
           "No learned release/duration. After completion/release A>0 must clear and recur before "
           "restart, as V0.1 trigger rearming. No gate reevaluation while safely holding; real "
           "observations/history update everytick. One native NumPy draw everytick.",
    test="Fresh99000..99031 x4critics x5/10/20 xcircle/square x4arms=3072episodes. Complete matched "
         "block, no threshold/checkpoint selection, no outcome-dependent stopping.",
    gates="Each method vsParent: original nine V0 gates unchanged. For a learned-method development "
          "positive additionally require square success count and mean original Q strictly above "
          "fixedV0.1; MLP must also exceed semantic ridge on both for evidence of nonlinear/latent increment.",
    limits="One-shot supervised labels under Parent continuation do not train a repeated-option critic. "
           "Online repeated interventions/changed visitation are explicitly tested here. Four Parent "
           "training seeds, one fitted ridge and one fitted MLP: not four independently trained gates. "
           "Timing under concurrent GPU workers is descriptive, not isolated latency measurement. "
           "No automatic retuning, additional rescue, or adaptive-duration development.")
_DEVICE = None
_MODELS = {}
_GATES = {}


def read(path):
    return json.loads(path.read_text())


def label_core_hash():
    root, h = Path(__file__).resolve().parents[1], hashlib.sha256()
    for folder in ("shixu", "vendor/crowd_sim"):
        for p in sorted((root/folder).rglob("*.py")):
            if p == root/"shixu/commit_advantage.py":
                continue
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def helpers():
    return {str(p): digest(p) for p in (Path(__file__), Path("experiments/online_action_commitment_v0.py"),
        Path("experiments/forecast_evidence.py"), Path("experiments/repeatable_defect_audit.py"))}


def freeze(device):
    p, learned = read(LABELS/"protocol.json"), read(LABELS/"learnability.json")
    if learned["verdict"] != "GO_LEARNABILITY" or p["core_sha256"] != label_core_hash():
        raise ValueError("No GO or frozen label source changed")
    for name, sha in p["helpers"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Frozen label helper changed")
    search = subprocess.run(["rg", "-l", "--glob", "*.json",
        r'"case"\s*:\s*990(?:0[0-9]|[12][0-9]|3[01])\b|"(?:case_start|eval_case_start|test_case_start)"\s*:\s*99000\b',
        "outputs"], capture_output=True, text=True)
    if search.returncode != 1:
        raise ValueError("Fresh block already used or check failed: "+search.stdout+search.stderr)
    gates = {m["kind"]:m["folds"][0] for m in learned["models"]}
    for m in gates.values():
        if digest(Path(m["checkpoint"])) != m["checkpoint_sha256"]:
            raise ValueError("Probe weight changed")
    save_json(OUT/"protocol.json", dict(rules=RULES, cases=CASES, seeds=SEEDS, cells=CELLS, arms=ARMS,
        checkpoints=p["checkpoints"], gates=gates, core_sha256=source_hash(), helpers=helpers(),
        label_protocol_sha256=digest(LABELS/"protocol.json"), dataset_sha256=digest(LABELS/"dataset.json"),
        learnability_sha256=digest(LABELS/"learnability.json"),
        case_unused_check=dict(exit_code=search.returncode, matches=search.stdout), device=device,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    print("FROZEN_ONLINE", len(CASES)*len(SEEDS)*len(CELLS)*len(ARMS), flush=True)


def verify():
    p = read(OUT/"protocol.json")
    if p["rules"] != RULES or p["core_sha256"] != source_hash() or p["helpers"] != helpers():
        raise ValueError("Online contract changed")
    for key, file in (("label_protocol_sha256", "protocol.json"), ("dataset_sha256", "dataset.json"),
                      ("learnability_sha256", "learnability.json")):
        if p[key] != digest(LABELS/file):
            raise ValueError("Learning assets changed")
    for c in p["checkpoints"].values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Parent changed")
    for g in p["gates"].values():
        if digest(Path(g["checkpoint"])) != g["checkpoint_sha256"]:
            raise ValueError("Gate changed")
    return p


def init_worker(device):
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


@torch.inference_mode()
def episode(task):
    seed, people, geometry, case, arm, sha = task
    prefix = OUT/"episodes"/f"{seed}-{people}-{geometry}-{case}-{arm}"
    path = prefix.with_suffix(".json")
    if path.exists():
        r = read(path)
        if r["protocol_sha256"] != sha or digest(prefix.with_suffix(".npz")) != r["trace_sha256"]:
            raise ValueError("Cached episode differs")
        return r
    start = time.perf_counter()
    if seed not in _MODELS:
        _MODELS[seed] = load(weights(seed, "cv"), _DEVICE)
    model, cfg = _MODELS[seed]
    if arm in ("linear", "mlp"):
        if arm not in _GATES:
            _GATES[arm] = AdvantageEstimator(torch.load(LABELS/"probes"/f"{arm}-case-only.pt", map_location="cpu", weights_only=False))
        policy = CommitAdvantagePolicy(model, cfg, _GATES[arm], _DEVICE)
    else:
        policy = (ValuePolicy if arm == "parent" else AllUnsafeReleasePolicy)(model, cfg, _DEVICE)
    env = environment(cfg, policy, geometry, people)
    env.reset(options={"test_case":case})
    initial = [env.robot.get_full_state().to_array().tolist()]+[h.get_full_state().to_array().tolist() for h in env.humans]
    initial_hash = hashlib.sha256(json.dumps(initial, separators=(",", ":")).encode()).hexdigest()
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    trace, events, distances = defaultdict(list), [], []
    while True:
        state = observer.observe(env)
        robot = state.self_state
        distance = float(np.hypot(robot.px-robot.gx, robot.py-robot.gy))
        commands = np.asarray(policy.candidate_actions())
        previous = np.array([np.nan, np.nan]) if policy.last_action is None else np.asarray(policy.last_action)
        tick_start = time.perf_counter()
        action = policy.predict(state)
        latency = time.perf_counter()-tick_start
        indices = np.flatnonzero(np.all(commands == np.asarray(action), axis=1))
        if len(indices) != 1:
            raise ValueError("Action is not a once-smoothed candidate")
        index = int(indices[0])
        features = policy.last_features if arm in ("linear", "mlp") else None
        for k, v in dict(time=env.global_time, robot=robot.to_array(), tokens=policy.encode(state),
            actions=action, previous=previous, grid_action=index, goal_distance=distance,
            inference_seconds=latency, features=features if features is not None else np.full(146, np.nan)).items():
            trace[k].append(v)
        distances.append(distance)
        if arm != "parent":
            d = policy.last_decision.copy()
            if d["selected_grid"] != index:
                raise ValueError("Execution and log differ")
            events.append(d)
        _, reward, done, truncated, info = env.step(action)
        trace["rewards"].append(reward)
        trace["actual_clearance"].append(info["dmin"])
        if done or truncated:
            break
    longest = max(len(t) for t in trace["tokens"])
    trace["tokens"] = [np.pad(t, ((0, longest-len(t)), (0, 0))) for t in trace["tokens"]]
    arrays = {k:np.asarray(v) for k, v in trace.items()}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz = prefix.with_suffix(".npz")
    if npz.exists():
        raise ValueError("Unfinished trace exists; do not overwrite")
    np.savez_compressed(npz, **arrays)
    r = dict(seed=seed, people=people, geometry=geometry, case=case, arm=arm,
        terminal=info["event"], behavior=failure_label(info["event"], distances),
        steps=len(distances), navigation_time=env.global_time, initial_world_sha256=initial_hash,
        minimum_clearance=float(min(trace["actual_clearance"])),
        discounted_return=float(sum(cfg.getfloat("train", "gamma")**i*r for i, r in enumerate(trace["rewards"]))),
        starts=sum(d["started"] for d in events), commitment_seconds=.25*sum(d["held"] for d in events),
        all_unsafe_releases=sum(d["release"] == "all-unsafe" for d in events),
        gate_parameters=policy.gate.parameters if arm in _GATES else 0,
        inference_seconds=float(sum(trace["inference_seconds"])),
        parent_parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-start, decisions=events,
        trace=str(npz), trace_sha256=digest(npz), protocol_sha256=sha)
    save_json(path, r)
    return r


def summarize(records):
    groups = defaultdict(dict)
    for r in records:
        key = r["seed"], r["people"], r["geometry"], r["case"]
        if r["arm"] in groups[key]:
            raise ValueError("Duplicate episode")
        groups[key][r["arm"]] = r
    expected = {(s, n, g, c) for s in SEEDS for n, g in CELLS for c in CASES}
    if set(groups) != expected or any(set(g) != set(ARMS) for g in groups.values()):
        raise ValueError("Incomplete block")
    if any(len({r["initial_world_sha256"] for r in g.values()}) != 1 for g in groups.values()):
        raise ValueError("Unmatched worlds")
    contrasts = {}
    for arm in ARMS[1:]:
        contrasts[arm] = compare([dict(r, arm="parent" if r["arm"] == "parent" else "commitment")
            for r in records if r["arm"] in ("parent", arm)])
    totals = {}
    for geometry in ("all", "circle", "square"):
        totals[geometry] = {}
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm and (geometry == "all" or r["geometry"] == geometry)]
            success_times = [r["navigation_time"] for r in rows if r["terminal"] == "reach_goal"]
            totals[geometry][arm] = dict(episodes=len(rows), terminals=dict(Counter(r["terminal"] for r in rows)),
                low_timeouts=sum(r["behavior"] == "low-progress-timeout" for r in rows),
                mean_q=float(np.mean([r["discounted_return"] for r in rows])),
                success_time_mean=float(np.mean(success_times)) if success_times else None,
                clearance_mean=float(np.mean([r["minimum_clearance"] for r in rows])),
                clearance_min=float(min(r["minimum_clearance"] for r in rows)),
                starts=sum(r["starts"] for r in rows), held_seconds=sum(r["commitment_seconds"] for r in rows),
                all_unsafe_releases=sum(r["all_unsafe_releases"] for r in rows),
                mean_inference_ms=1000*sum(r["inference_seconds"] for r in rows)/sum(r["steps"] for r in rows))
    square, passes = totals["square"], {}
    for arm in ("linear", "mlp"):
        comparators = ("v01",) if arm == "linear" else ("v01", "linear")
        passes[arm] = dict(parent_guards=all(contrasts[arm]["gates"].values()),
            better_simple_success=all(square[arm]["terminals"].get("reach_goal", 0) > square[c]["terminals"].get("reach_goal", 0) for c in comparators),
            better_simple_q=all(square[arm]["mean_q"] > square[c]["mean_q"] for c in comparators))
    return dict(contrasts=contrasts, totals=totals, learned_gates=passes,
        verdict="DEVELOPMENT_POSITIVE" if any(all(v.values()) for v in passes.values()) else "LEARNED_INITIATION_NOT_CONFIRMED",
        limits=RULES["limits"])


def execute(workers):
    p, records, start = verify(), [], time.perf_counter()
    tasks = [(s, n, g, c, a, digest(OUT/"protocol.json")) for s in SEEDS for n, g in CELLS for c in CASES for a in ARMS]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(p["device"],)) as pool:
        for future in as_completed([pool.submit(episode, t) for t in tasks]):
            records.append(future.result())
            if len(records)%128 == 0:
                print("ONLINE", len(records), "/3072", round(time.perf_counter()-start, 1), flush=True)
    records.sort(key=lambda r:(r["seed"], r["people"], r["geometry"], r["case"], r["arm"]))
    elapsed = time.perf_counter()-start
    save_json(OUT/"episode-summary.json", dict(records=[{k:v for k, v in r.items() if k != "decisions"} for r in records], elapsed_seconds=elapsed))
    result = summarize(records)
    result.update(protocol_sha256=digest(OUT/"protocol.json"), episodes=len(records), elapsed_seconds=elapsed)
    save_json(OUT/"summary.json", result)
    print("ONLINE_VERDICT", result["verdict"], result["learned_gates"], flush=True)


def validate():
    p = verify()
    records = [read(f) for f in sorted((OUT/"episodes").glob("*.json"))]
    records.sort(key=lambda r:(r["seed"], r["people"], r["geometry"], r["case"], r["arm"]))
    if summarize(records) != {k:v for k, v in read(OUT/"summary.json").items() if k not in ("protocol_sha256", "episodes", "elapsed_seconds")}:
        raise ValueError("Summary differs")
    groups, steps, parity = defaultdict(dict), Counter(), Counter()
    for r in records:
        groups[r["seed"], r["people"], r["geometry"], r["case"]][r["arm"]] = r
        if digest(Path(r["trace"])) != r["trace_sha256"] or r["protocol_sha256"] != digest(OUT/"protocol.json"):
            raise ValueError("Trace hash changed")
        if r["seed"] not in _MODELS:
            _MODELS[r["seed"]] = load(weights(r["seed"], "cv"), "cpu")
        cfg = _MODELS[r["seed"]][1]
        grid = np.asarray(actions(cfg))
        with np.load(r["trace"]) as t:
            expected = grid[t["grid_action"]].copy()
            expected[1:] = .3*t["previous"][1:]+.7*expected[1:]
            np.testing.assert_array_equal(expected, t["actions"])
            if np.any(np.diff(t["time"]) != .25):
                raise ValueError("Clock changed")
            q = sum(cfg.getfloat("train", "gamma")**i*v for i, v in enumerate(t["rewards"]))
            if abs(q-r["discounted_return"]) > 1e-12:
                raise ValueError("Return mismatch")
            steps[r["arm"]] += r["steps"]
            for tick, d in enumerate(r["decisions"]):
                if d["held"] and (d["blocked"] or d["no_margin_safe_candidate"]):
                    raise ValueError("Unsafe hold")
                if d["selected_grid"] != t["grid_action"][tick] or d["tick"] != tick:
                    raise ValueError("Grid log mismatch")
                if d["release"] in ("all-unsafe", "safety-blocked") and (d["held"] or d["proposal"] != d["selected_grid"]):
                    raise ValueError("Same-tick release mismatch")
                if r["arm"] in ("linear", "mlp") and d["advantage"] is not None:
                    if r["arm"] not in _GATES:
                        _GATES[r["arm"]] = AdvantageEstimator(torch.load(LABELS/"probes"/f"{r['arm']}-case-only.pt", map_location="cpu", weights_only=False))
                    if abs(_GATES[r["arm"]](t["features"][tick])-d["advantage"]) > 1e-7:
                        raise ValueError("Gate prediction mismatch")
                    if d["started"] and (tick < 8 or d["distance"] <= 1 or d["advantage"] <= 0):
                        raise ValueError("Noncausal/unready initiation")
    for g in groups.values():
        for arm in ARMS[1:]:
            if not g[arm]["commitment_seconds"]:
                with np.load(g["parent"]["trace"]) as a, np.load(g[arm]["trace"]) as b:
                    np.testing.assert_array_equal(a["actions"], b["actions"])
                for k in ("terminal", "discounted_return", "minimum_clearance", "navigation_time"):
                    if g[arm][k] != g["parent"][k]:
                        raise ValueError("No-intervention parity mismatch")
                parity[arm] += 1
    save_json(OUT/"validation.json", dict(episodes=len(records), matched_worlds=len(groups),
        control_steps=dict(steps), exact_no_intervention_pairs=dict(parity),
        protocol_sha256=digest(OUT/"protocol.json"), summary_sha256=digest(OUT/"summary.json"),
        validated_smoothing=True, validated_rewards=True, validated_gate_inputs=True,
        validated_same_tick_release=True))
    print("VALIDATED_ONLINE", len(records), dict(parity), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "validate"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"freeze":lambda:freeze(args.device), "run":lambda:execute(args.workers), "validate":validate}[args.stage]()


if __name__ == "__main__":
    main()
