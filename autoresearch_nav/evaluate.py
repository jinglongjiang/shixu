"""Frozen offline screening, bounded development and exactly one final test."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch

from autoresearch_nav.prepare import OUT, SEEDS, CELLS, DEV, FINAL, ProposalPolicy, read, verify
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import save_json
from experiments.online_action_commitment_v0 import compare
from experiments.repeatable_defect_audit import failure_label
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy, actions
from shixu.runner import environment


BUDGET = dict(epochs=300, lr=.003, weight_decay=.001, seed=616, ridge=1.)
_DEVICE = None
_MODELS = {}
_METHODS = {}


def module(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        if any(n not in ("numpy", "torch", "math") for n in names):
            raise ValueError("Method may not import simulator/filesystem/evaluator")
    spec = importlib.util.spec_from_file_location("frozen_autonav_method_"+digest(path)[:8], path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    if value.SPEC["family"] not in ("rule", "linear", "logistic", "mlp"):
        raise ValueError("Unsupported method family")
    if len(value.SPEC.get("hidden", [])) > 2 or any(n > 32 for n in value.SPEC.get("hidden", [])):
        raise ValueError("Architecture exceeds frozen bounds")
    return value


def log(tag, stage, status, metrics):
    path = OUT/"results.tsv"
    fields = ("tag", "commit", "parent", "stage", "status", "parameters", "training_seconds",
              "advantage", "coverage", "positive_coverage", "rescue", "damage", "collision", "description")
    meta = read(OUT/"experiments"/tag/"experiment.json")
    row = dict(tag=tag, commit=meta["commit"], parent=meta["spec"].get("parent"), stage=stage, status=status,
        parameters=meta["parameters"], training_seconds=meta["training_seconds"],
        advantage=metrics.get("advantage", metrics.get("mean_q_delta")), coverage=metrics.get("coverage"),
        positive_coverage=metrics.get("positive_coverage"), rescue=metrics.get("rescues"),
        damage=metrics.get("damages"), collision=metrics.get("collisions_added"), description=meta["spec"]["hypothesis"])
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t")
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def balance(rows):
    counts = Counter((r["seed"], r["case"]) for r in rows)
    return np.array([1/counts[r["seed"], r["case"]] for r in rows])


def offline_metrics(rows, accepted):
    w, yes = balance(rows), np.asarray(accepted, bool)
    y = np.array([r["y"] for r in rows])
    rescue, harm, collision = (np.array([r[k] for r in rows], bool) for k in ("rescue", "damage", "collision_added"))
    positive = y > .005
    def average(x):
        return float(np.average(x, weights=w))
    def coverage(mask):
        return float(np.sum(w*yes*mask)/np.sum(w*mask)) if mask.any() else None
    return dict(rows=len(rows), cases=len({r["case"] for r in rows}), accepted=int(yes.sum()),
        coverage=average(yes), advantage=average(yes*y), all_advantage=average(y),
        positive_coverage=coverage(positive), rescue_retention=coverage(rescue),
        damage_rate=average(yes*harm), all_damage_rate=average(harm),
        collision_rate=average(yes*collision), all_collision_rate=average(collision),
        rescues=int((yes&rescue).sum()), damages=int((yes&harm).sum()), collisions_added=int((yes&collision).sum()),
        positive_cases=len({r["case"] for r in rows if r["y"]>.005}),
        rescue_cases=len({r["case"] for r in rows if r["rescue"]}))


def offline(tag):
    verify()
    if (OUT/"final-lock.json").exists():
        raise ValueError("Final locked: no further method search")
    existing = list((OUT/"experiments").glob("*/experiment.json"))
    if len(existing) >= 60:
        raise ValueError("Offline cap reached")
    if not re.fullmatch(r"[a-z0-9-]+", tag):
        raise ValueError("Invalid tag")
    source = Path("autoresearch_nav/method.py")
    if subprocess.check_output(["git", "status", "--porcelain", "--", str(source)], text=True).strip():
        raise ValueError("Commit method before experiment")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    committed = subprocess.check_output(["git", "show", commit+":"+str(source)])
    if hashlib.sha256(committed).hexdigest() != digest(source):
        raise ValueError("Method does not match commit")
    method = module(source)
    dataset = read(OUT/"dataset.json")
    train = [r for r in dataset["rows"] if r["case"] < 130384]
    dev = [r for r in dataset["rows"] if r["case"] >= 130384]
    if not train or not dev or {r["case"] for r in train}&{r["case"] for r in dev}:
        raise ValueError("Missing data or case leakage")
    torch.set_num_threads(1)
    base = OUT/"experiments"/tag
    if base.exists():
        raise ValueError("Do not overwrite experiment")
    base.mkdir(parents=True)
    (base/"method.py").write_bytes(committed)
    start = time.perf_counter()
    try:
        fitted = method.fit(train, BUDGET.copy())
        seconds = time.perf_counter()-start
        parameters = int(method.parameters(fitted))
        torch.save(fitted, base/"weights.pt")
        infer_start = time.perf_counter()
        predictions = [bool(method.accept(np.array(r["features"], np.float64), fitted)) for r in dev]
        inference_seconds = time.perf_counter()-infer_start
    except Exception as exc:
        save_json(base/"experiment.json", dict(tag=tag, commit=commit, spec=method.SPEC,
            parameters=None, training_seconds=time.perf_counter()-start,
            method_sha256=digest(base/"method.py"), protocol_sha256=digest(OUT/"protocol.json"),
            dataset_sha256=digest(OUT/"dataset.json")))
        save_json(base/"offline.json", dict(passed=False, error=repr(exc), reason="implementation failure"))
        log(tag, "offline", "crash-discard", {})
        raise
    metrics = offline_metrics(dev, predictions)
    gates = dict(positive_support=metrics["positive_cases"] >= 4 and metrics["rescue_cases"] >= 2,
        coverage=metrics["coverage"] >= .05,
        positive_coverage=metrics["positive_coverage"] is not None and metrics["positive_coverage"] >= .10,
        rescue_retention=metrics["rescue_retention"] is not None and metrics["rescue_retention"] >= .25,
        advantage=metrics["advantage"] > 0,
        damage_reduction=metrics["damage_rate"] <= .75*metrics["all_damage_rate"],
        collision=metrics["collision_rate"] <= metrics["all_collision_rate"]+1e-12 and metrics["collision_rate"] <= .005)
    meta = dict(tag=tag, commit=commit, spec=method.SPEC, parameters=parameters, training_seconds=seconds,
        method_sha256=digest(base/"method.py"), weights_sha256=digest(base/"weights.pt"),
        protocol_sha256=digest(OUT/"protocol.json"), dataset_sha256=digest(OUT/"dataset.json"))
    save_json(base/"experiment.json", meta)
    save_json(base/"offline.json", dict(metrics=metrics, gates=gates, passed=all(gates.values()), accepted=predictions,
        inference_seconds=inference_seconds, mean_inference_us=1e6*inference_seconds/len(dev)))
    log(tag, "offline", "keep" if all(gates.values()) else "discard", metrics)
    print("OFFLINE", tag, "KEEP" if all(gates.values()) else "DISCARD", metrics, gates, flush=True)


def frozen_method(tag):
    base = OUT/"experiments"/tag
    meta = read(base/"experiment.json")
    if digest(base/"method.py") != meta["method_sha256"] or digest(base/"weights.pt") != meta["weights_sha256"]:
        raise ValueError("Frozen method changed")
    return module(base/"method.py"), torch.load(base/"weights.pt", map_location="cpu", weights_only=False), meta


def init_worker(device):
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


@torch.inference_mode()
def episode(task):
    tag, stage, seed, people, geometry, case, arm, sha = task
    base = OUT/"experiments"/tag/stage/"episodes"
    prefix = base/f"{seed}-{people}-{geometry}-{case}-{arm}"
    path = prefix.with_suffix(".json")
    if path.exists():
        r = read(path)
        if r["protocol_sha256"] != sha or digest(Path(r["trace"])) != r["trace_sha256"]:
            raise ValueError("Cached episode changed")
        return r
    if seed not in _MODELS:
        _MODELS[seed] = load(weights(seed, "cv"), _DEVICE)
    model, cfg = _MODELS[seed]
    if arm == "candidate":
        if tag not in _METHODS:
            _METHODS[tag] = frozen_method(tag)
        method, fitted, meta = _METHODS[tag]
        policy = ProposalPolicy(model, cfg, lambda x:method.accept(x, fitted), _DEVICE)
    else:
        policy = ValuePolicy(model, cfg, _DEVICE) if arm == "parent" else ProposalPolicy(model, cfg, None, _DEVICE)
    env = environment(cfg, policy, geometry, people)
    env.reset(options={"test_case":case})
    initial = [env.robot.get_full_state().to_array().tolist()]+[h.get_full_state().to_array().tolist() for h in env.humans]
    initial_hash = hashlib.sha256(json.dumps(initial, separators=(",", ":")).encode()).hexdigest()
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    trace, decisions, distances = defaultdict(list), [], []
    start = time.perf_counter()
    while True:
        state = observer.observe(env)
        distance = float(np.hypot(state.self_state.px-state.self_state.gx, state.self_state.py-state.self_state.gy))
        candidates = np.asarray(policy.candidate_actions())
        previous = np.full(2, np.nan) if policy.last_action is None else np.asarray(policy.last_action)
        clock = time.perf_counter()
        command = policy.predict(state)
        latency = time.perf_counter()-clock
        index = np.flatnonzero(np.all(candidates == command, axis=1))
        if len(index) != 1:
            raise ValueError("Command differs from native once-smoothed grid")
        for k,v in dict(time=env.global_time, previous=previous, actions=command, grid_action=int(index[0]),
            goal_distance=distance, inference_seconds=latency).items():
            trace[k].append(v)
        distances.append(distance)
        if arm != "parent":
            decisions.append(policy.last_decision.copy())
            trace["features"].append(policy.last_features if policy.last_features is not None else np.full(24, np.nan))
        _, reward, done, truncated, info = env.step(command)
        trace["rewards"].append(reward)
        trace["actual_clearance"].append(info["dmin"])
        if done or truncated:
            break
    base.mkdir(parents=True, exist_ok=True)
    npz = prefix.with_suffix(".npz")
    if npz.exists():
        raise ValueError("Partial trace cannot be overwritten")
    np.savez_compressed(npz, **{k:np.asarray(v) for k,v in trace.items()})
    r = dict(seed=seed, people=people, geometry=geometry, case=case, arm=arm,
        terminal=info["event"], behavior=failure_label(info["event"], distances),
        steps=len(distances), navigation_time=env.global_time, initial_world_sha256=initial_hash,
        minimum_clearance=float(min(trace["actual_clearance"])),
        discounted_return=float(sum(cfg.getfloat("train", "gamma")**i*r for i,r in enumerate(trace["rewards"]))),
        starts=sum(d["started"] for d in decisions), commitment_seconds=.25*sum(d["held"] for d in decisions),
        offers=sum(d["offered"] for d in decisions), rejections=sum(d["accepted"] is False for d in decisions),
        inference_seconds=float(sum(trace["inference_seconds"])), decisions=decisions,
        trace=str(npz), trace_sha256=digest(npz), protocol_sha256=sha, elapsed_seconds=time.perf_counter()-start)
    save_json(path, r)
    return r


def old_baselines(stage):
    name, path, cases = next(x for x in DEV if x[0] == stage)
    old = read(path/"protocol.json")
    p = verify()
    if old["checkpoints"] != p["checkpoints"]:
        raise ValueError("Old dev checkpoint mismatch")
    rows = []
    for seed in SEEDS:
        for n,g in CELLS:
            for case in cases:
                for arm in ("parent", "v01"):
                    r = read(path/"episodes"/f"{seed}-{n}-{g}-{case}-{arm}.json")
                    if digest(Path(r["trace"])) != r["trace_sha256"] or r["protocol_sha256"] != digest(path/"protocol.json"):
                        raise ValueError("Old development cache changed")
                    rows.append(r)
    return rows


def summary(records):
    groups = defaultdict(dict)
    for r in records:
        key = r["seed"], r["people"], r["geometry"], r["case"]
        if r["arm"] in groups[key]:
            raise ValueError("Duplicate arm")
        groups[key][r["arm"]] = r
    if len(groups) != 768 or any(set(g) != {"parent", "v01", "candidate"} for g in groups.values()):
        raise ValueError("Incomplete matched block")
    if any(len({r["initial_world_sha256"] for r in g.values()}) != 1 for g in groups.values()):
        raise ValueError("Unmatched worlds")
    contrasts = {}
    for arm in ("v01", "candidate"):
        contrasts[arm] = compare([dict(r, arm="parent" if r["arm"] == "parent" else "commitment")
            for r in records if r["arm"] in ("parent", arm)])
    candidate, baseline = contrasts["candidate"], contrasts["v01"]
    def rescue(c, geo=None):
        return sum(r["rescued"] for r in c["pairs"] if geo is None or r["geometry"]==geo)
    gates = dict(square_damage_reduction=candidate["square_damaged"] <= .75*baseline["square_damaged"],
        circle_damage_reduction=candidate["circle_damaged"] <= .75*baseline["circle_damaged"],
        square_rescue_retention=rescue(candidate, "square") >= .5*rescue(baseline, "square"),
        total_rescue_retention=rescue(candidate) >= .5*rescue(baseline),
        four_qualified_cases=len(candidate["square_qualified_rescue_cases"]) >= 4,
        square_low_progress=candidate["square_low_timeout_reduction"] >= 0,
        square_collision=candidate["square_collision_delta"] <= 3,
        circle_collision=candidate["circle_collision_delta"] <= 3)
    totals = {}
    for geometry in ("all", "circle", "square"):
        totals[geometry] = {}
        for arm in ("parent", "v01", "candidate"):
            rows = [r for r in records if r["arm"]==arm and (geometry=="all" or r["geometry"]==geometry)]
            totals[geometry][arm] = dict(terminals=dict(Counter(r["terminal"] for r in rows)),
                low_timeouts=sum(r["behavior"]=="low-progress-timeout" for r in rows),
                mean_q=float(np.mean([r["discounted_return"] for r in rows])),
                minimum_clearance_mean=float(np.mean([r["minimum_clearance"] for r in rows])),
                minimum_clearance_min=float(min(r["minimum_clearance"] for r in rows)),
                mean_inference_ms=1000*sum(r["inference_seconds"] for r in rows)/sum(r["steps"] for r in rows))
    return dict(contrasts=contrasts, totals=totals, dev_gates=gates, dev_pass=all(gates.values()),
        rescues=rescue(candidate), damages=candidate["square_damaged"]+candidate["circle_damaged"],
        collisions_added=candidate["square_collision_delta"]+candidate["circle_collision_delta"],
        mean_q_delta=totals["all"]["candidate"]["mean_q"]-totals["all"]["parent"]["mean_q"])


def run(tag, stage, workers):
    p = verify()
    meta = read(OUT/"experiments"/tag/"experiment.json")
    if meta["protocol_sha256"] != digest(OUT/"protocol.json"):
        raise ValueError("Protocol mismatch")
    frozen_method(tag)
    if stage == "final":
        lock = read(OUT/"final-lock.json")
        if lock["tag"] != tag or lock["method_sha256"] != meta["method_sha256"] or lock["weights_sha256"] != meta["weights_sha256"]:
            raise ValueError("Final candidate mismatch")
        if (OUT/"final-result.json").exists():
            raise ValueError("Final already completed; no rerun")
        cases, records, arms = FINAL, [], ("parent", "v01", "candidate")
    else:
        if (OUT/"final-lock.json").exists():
            raise ValueError("Final locked, no development feedback")
        previous = list((OUT/"experiments").glob(f"*/{stage}/started.json"))
        if len(previous) >= (8 if stage == "dev1" else 3) and not (OUT/"experiments"/tag/stage/"started.json").exists():
            raise ValueError("Development cap reached")
        if not read(OUT/"experiments"/tag/"offline.json")["passed"]:
            raise ValueError("Offline candidate did not pass")
        if stage == "dev2" and not read(OUT/"experiments"/tag/"dev1"/"summary.json")["dev_pass"]:
            raise ValueError("Dev1 did not pass")
        cases = next(x[2] for x in DEV if x[0]==stage)
        records, arms = old_baselines(stage), ("candidate",)
    base = OUT/"experiments"/tag/stage
    if (base/"summary.json").exists():
        raise ValueError("Do not rerun completed stage")
    if not (base/"started.json").exists():
        save_json(base/"started.json", dict(tag=tag, method_sha256=meta["method_sha256"],
            weights_sha256=meta["weights_sha256"], stage=stage))
    start = time.perf_counter()
    tasks = [(tag, stage, s, n, g, c, arm, digest(OUT/"protocol.json")) for s in SEEDS for n,g in CELLS for c in cases for arm in arms]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(p["device"],)) as pool:
        for future in as_completed([pool.submit(episode, t) for t in tasks]):
            records.append(future.result())
            if (len(records)-(1536 if stage!="final" else 0))%128 == 0:
                print("NAV", tag, stage, len(records), round(time.perf_counter()-start, 1), flush=True)
    records.sort(key=lambda r:(r["seed"], r["people"], r["geometry"], r["case"], r["arm"]))
    result = summary(records)
    result.update(seconds=time.perf_counter()-start, tag=tag, stage=stage, method_sha256=meta["method_sha256"], weights_sha256=meta["weights_sha256"])
    save_json(base/"summary.json", result)
    save_json(base/"episode-summary.json", dict(records=[{k:v for k,v in r.items() if k!="decisions"} for r in records]))
    passed = all(result["contrasts"]["candidate"]["gates"].values()) if stage=="final" else result["dev_pass"]
    log(tag, stage, "keep" if passed else "discard", result)
    if stage=="final":
        save_json(OUT/"final-result.json", dict(verdict="METHOD_CANDIDATE_FOUND" if passed else "FINAL_CONFIRMATION_FAILED",
            tag=tag, summary_sha256=digest(base/"summary.json")))
    print("NAV_RESULT", tag, stage, passed, result["dev_gates"], result["contrasts"]["candidate"]["gates"], flush=True)


def lock(tag):
    verify()
    method, fitted, meta = frozen_method(tag)
    for stage in ("dev1", "dev2"):
        if not read(OUT/"experiments"/tag/stage/"summary.json")["dev_pass"]:
            raise ValueError("Both development blocks must pass")
    if (OUT/"final-lock.json").exists():
        raise ValueError("Final already locked")
    search = subprocess.run(["rg", "-l", "--glob", "*.json", r'"case"\s*:\s*1400(?:0[0-9]|[12][0-9]|3[01])\b', "outputs"], text=True, capture_output=True)
    if search.returncode != 1:
        raise ValueError("Final cases consumed before lock")
    save_json(OUT/"final-lock.json", dict(tag=tag, method_sha256=meta["method_sha256"], weights_sha256=meta["weights_sha256"],
        unused_check=dict(exit_code=search.returncode, matches=search.stdout)))


def validate(tag, stage):
    verify()
    method, fitted, meta = frozen_method(tag)
    records = [read(p) for p in sorted((OUT/"experiments"/tag/stage/"episodes").glob("*.json"))]
    _, cfg = load(weights(419, "cv"), "cpu")
    grid, steps = np.asarray(actions(cfg)), Counter()
    for r in records:
        if digest(Path(r["trace"])) != r["trace_sha256"]:
            raise ValueError("Trace hash mismatch")
        with np.load(r["trace"]) as t:
            expected = grid[t["grid_action"]].copy()
            expected[1:] = .3*t["previous"][1:]+.7*expected[1:]
            np.testing.assert_array_equal(expected, t["actions"])
            if np.any(np.diff(t["time"]) != .25):
                raise ValueError("Clock mismatch")
            if abs(sum(.99**i*v for i,v in enumerate(t["rewards"]))-r["discounted_return"])>1e-12:
                raise ValueError("Reward mismatch")
            for i,d in enumerate(r["decisions"]):
                if d["held"] and (d["blocked"] or d["no_margin_safe_candidate"]):
                    raise ValueError("Unsafe hold")
                if d["started"] and (not d["trigger"] or not d["offered"] or d["accepted"] is not True):
                    raise ValueError("Non-proposal initiation")
                if d["offered"] and r["arm"]=="candidate" and bool(method.accept(t["features"][i], fitted)) != d["accepted"]:
                    raise ValueError("Method/log mismatch")
                if d["release"] in ("all-unsafe", "safety-blocked") and (d["held"] or d["proposal"]!=d["selected_grid"]):
                    raise ValueError("Release differs")
        steps[r["arm"]] += r["steps"]
    save_json(OUT/"experiments"/tag/stage/"validation.json", dict(episodes=len(records), control_steps=dict(steps),
        method_sha256=meta["method_sha256"], protocol_sha256=digest(OUT/"protocol.json"),
        proposal_only=True, execution=True, reward=True, release=True, gate_outputs=True))
    print("VALIDATED", tag, stage, len(records), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("offline", "dev1", "dev2", "lock", "final", "validate"))
    p.add_argument("tag")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--block", choices=("dev1", "dev2", "final"))
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.stage=="offline": offline(a.tag)
    elif a.stage=="lock": lock(a.tag)
    elif a.stage=="validate": validate(a.tag, a.block)
    else: run(a.tag, a.stage, a.workers)


if __name__=="__main__":
    main()
