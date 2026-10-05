"""Bounded native multi-step interventions, followed by the frozen policy."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import advance, archive_parity, fork, run_branches, save_json
from experiments.occlusion import source_hash
from experiments.repeatable_defect_audit import OUT as ATLAS, restore_root, root_tick, verify as verify_atlas
from shixu.policy import ValuePolicy


OUT = Path("outputs/multistep-headroom-decision-test")
RULES = {
    "scope": "Four unchanged fresh roots. No training, critic/reward/filter/history/model changes; "
             "an offline control-space diagnostic, not a deployable policy or causal attribution.",
    "constant": "All80 native grid actions held4 or8 control steps (1s or2s). Each step applies "
                "the unchanged alpha=0.3 smoothing once to the previous executed action. "
                "Then restore the same checkpoint's native scalar predict() until terminal.",
    "sequence": "If fewer than2 primary roots have qualified constant-action rescues: allA in80 "
                "and B in deduplicated same/heading+/-1,+/-2/speed+/-1 neighbors. Headings wrap; "
                "speed boundaries clip by omission. 2stepsA+2stepsB or4stepsA+4stepsB. No search.",
    "memory": "Observe legally every0.25s; append each actually observed frame once, including "
              "during forced controls. Keep executed last_action for smoothing and continuation. "
              "Do not write hypothetical candidate states. Abort intervention at native terminal.",
    "safety": "Record all collisions/clearances. ReachGoal with full root-to-terminal minimum "
              "surface clearance>=0.02m is qualified. Below0.02m is fragile. This does not change "
              "the parent's0.2m filter or reward. Forced interventions bypass action selection, "
              "not dynamics; log predicted CV clearance and whether native filter would block.",
    "metric": "Native gamma0.99 discounted original rewards from root to termination, actual "
              "commands, minimum clearance, terminal, remaining/absolute arrival time. No progress proxy.",
    "backup": "Freeze4 before outcomes from existing81000..81031 parent-only square5/10 atlas: "
              "roundrobin populations, ascending case IDs, cross-population unique IDs, exclude "
              "all4 primary cases, lowest failed seed. Same low-progress root rule. Activate only "
              "if exactly1/4 primary roots qualifies after applicable layers. Establish80-step1 "
              "controls for backups before multistep; single-step rescuable backups cannot qualify "
              "as specifically multistep defects. Complete fixed blocks, not favorable branches only.",
    "decision": ">=2 independent case IDs with qualified multi-step rescues and zero qualified "
                "single-step rescues: MULTISTEP_HEADROOM_CONFIRMED. Zero primary rescues: "
                "NO_BOUNDED_MULTISTEP_HEADROOM_FOUND. Exactly1: use frozen backups; if still<2, "
                "INSUFFICIENT_REPEATABLE_MULTISTEP_HEADROOM. Stop tested route; no80x80 expansion. "
                "No root/threshold/checkpoint changes after outcomes. A positive is development "
                "headroom, not overall SR, unique cause, or method novelty. Do not implement Ours here.",
}
_DEVICE = None
_CACHE = {}


def read(path):
    return json.loads(path.read_text())


def key(row):
    return f"{row['people']}-{row['geometry']}-{row['case']}-seed{row['seed']}"


def backup_roots(records, primary):
    used = {r["case"] for r in primary}
    choices = {}
    for people in (5, 10):
        grouped = {}
        for row in records:
            if row["people"] == people and row["geometry"] == "square" and row["behavior"] == "low-progress-timeout":
                grouped.setdefault(row["case"], []).append(row)
        choices[people] = [min(grouped[c], key=lambda r: r["seed"]) for c in sorted(grouped) if c not in used]
    result = []
    while len(result) < 4:
        previous = len(result)
        for people in (5, 10):
            choices[people] = [r for r in choices[people] if r["case"] not in used]
            if choices[people] and len(result) < 4:
                row = choices[people].pop(0)
                result.append(row)
                used.add(row["case"])
        if len(result) == previous:
            raise ValueError("Fewer than four independent parent-only backup cases")
    return result


def neighbors(action, speeds=5, headings=16):
    heading, speed = divmod(action, speeds)
    if not 0 <= action < speeds * headings:
        raise ValueError("Grid index out of range")
    result = {action}
    result.update(((heading + d) % headings) * speeds + speed for d in (-2, -1, 1, 2))
    result.update(heading * speeds + s for s in (speed - 1, speed + 1) if 0 <= s < speeds)
    return sorted(result)


def schedules(layer):
    if layer == "single":
        return [(a,) for a in range(80)]
    if layer == "constant":
        return [(a,) * steps for steps in (4, 8) for a in range(80)]
    if layer == "local":
        # Same-action branches already ran in the constant layer.
        return [(a,) * half + (b,) * half for half in (2, 4) for a in range(80)
                for b in neighbors(a) if b != a]
    raise ValueError("Unknown layer")


def helper_hashes():
    paths = [Path(__file__), Path("experiments/repeatable_defect_audit.py"),
             Path("experiments/multihorizon_control.py"), Path("experiments/forecast_evidence.py"),
             Path("experiments/decision_state.py"), Path("experiments/forecast_control_diagnostic.py")]
    return {str(p): digest(p) for p in paths}


def freeze(device):
    old = verify_atlas()
    primary = read(ATLAS / "pooled-fresh-selection.json")["rows"]
    expected = [(5, 81000, 491, 29), (10, 81001, 419, 33), (5, 81003, 419, 8), (10, 81006, 443, 45)]
    if [(r["people"], r["case"], r["seed"], r["root"]["tick"]) for r in primary] != expected:
        raise ValueError("Primary roots changed")
    paths = [ATLAS / "pooled-fresh-selection.json", ATLAS / "exploratory-fresh-square-5-atlas.json",
             ATLAS / "exploratory-fresh-square-10-atlas.json"]
    records = [r for p in paths[1:] for r in read(p)["records"]]
    backups = backup_roots(records, primary)
    references = {}
    for row in primary:
        p = ATLAS / "pooled-fresh-headroom" / f"{row['seed']}_{row['people']}_square_{row['case']}.json"
        data = read(p)
        if len(data["outcomes"]) != 80 or any(r["terminal"] != "timeout" for r in data["outcomes"]):
            raise ValueError("Expected all80 archived first-action timeouts")
        if np.ptp([r["return_"] for r in data["outcomes"]]) != 0 or not data["parity"]["passed"]:
            raise ValueError("Archived first-action contract changed")
        references[str(p)] = digest(p)
    for row in backups:
        with np.load(row["trace"]) as trace:
            if root_tick(row, trace) != row["root"]["tick"]:
                raise ValueError("Backup root rule changed")
    _, cfg = load(weights(419, "cv"), "cpu")
    if (cfg.getint("policy", "n_speeds"), cfg.getint("policy", "n_headings")) != (5, 16):
        raise ValueError("Unexpected grid")
    if cfg.getboolean("policy", "include_stop") or cfg.getfloat("eval_protocol", "action_smoothing") != .3:
        raise ValueError("Unexpected execution contract")
    manifest = dict(rules=RULES, primary=primary, backups=backups, old_single_step_results=references,
        archives={str(p): digest(p) for p in paths}, core_sha256=source_hash(), helpers=helper_hashes(),
        checkpoints=old["checkpoints"], created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime=dict(device=device, torch=torch.__version__, python=platform.python_version(),
                     gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32, tf32_matmul=torch.backends.cuda.matmul.allow_tf32))
    save_json(OUT / "protocol.json", manifest)
    print("FROZEN", dict(primary=[key(r) for r in primary], backups=[key(r) for r in backups]), flush=True)


def verify():
    p = read(OUT / "protocol.json")
    expected = p["helpers"].copy()
    correction = OUT / "implementation-correction.json"
    if correction.exists():
        note = read(correction)
        script = str(Path(__file__))
        if script not in expected:
            script = "experiments/multistep_headroom_decision_test.py"
        if note["protocol_sha256"] != digest(OUT / "protocol.json") or note["original_script_sha256"] != expected[script]:
            raise ValueError("Correction does not match original freeze")
        expected[script] = note["corrected_script_sha256"]
    if p["rules"] != RULES or p["core_sha256"] != source_hash() or expected != helper_hashes():
        raise ValueError("Frozen source or protocol changed")
    for group in ("checkpoints",):
        for ck in p[group].values():
            if digest(Path(ck["path"])) != ck["sha256"]:
                raise ValueError("Checkpoint changed")
    for group in ("archives", "old_single_step_results"):
        for path, sha in p[group].items():
            if digest(Path(path)) != sha:
                raise ValueError("Frozen evidence changed")
    for row in p["primary"] + p["backups"]:
        if digest(Path(row["trace"])) != row["trace_sha256"]:
            raise ValueError("Root trace changed")
    return p


def init_worker(device):
    global _DEVICE
    torch.set_num_threads(1)
    _DEVICE = device


def context(row):
    name = key(row)
    if name not in _CACHE:
        model, cfg = load(weights(row["seed"], "cv"), _DEVICE)
        root = restore_root(row, model, cfg, _DEVICE)
        _CACHE[name] = (root, model, cfg)
    return _CACHE[name]


@torch.inference_mode()
def run_sequence(root, model, cfg, device, sequence):
    world = fork(root, model, cfg, device)
    intervention = []
    for tick, index in enumerate(sequence):
        if world["done"]:
            break
        policy = world["policy"]
        state = root["state"] if tick == 0 else world["observer"].observe(world["env"])
        commands = policy.candidate_actions()
        _, _, clearance = policy.aligned_candidates(state, commands)
        margin = cfg.getfloat("eval_protocol", "safety_margin")
        blocked = margin > 0 and bool((clearance >= margin).any()) and clearance[index] < margin
        command = commands[index]
        policy.history.append(policy.encode(state))
        policy.last_action = command
        advance(world, command, cfg)
        intervention.append(dict(grid_index=index, command=list(command),
            predicted_cv_clearance=float(clearance[index]), native_filter_would_block=bool(blocked)))
    while not world["done"]:
        state = world["observer"].observe(world["env"])
        advance(world, world["policy"].predict(state), cfg)
    return dict(terminal=world["event"], return_=world["total"], seconds=world["steps"]*.25,
        absolute_terminal_seconds=root["snapshot"]["time"]+world["steps"]*.25,
        minimum_clearance=world["minimum"], commands=world["commands"], intervention=intervention,
        planned_grid_sequence=list(sequence), actual_intervention_steps=len(intervention))


@torch.inference_mode()
def block(task):
    row, layer, numbered, protocol_sha = task
    root, model, cfg = context(row)
    records = []
    for number, sequence in numbered:
        path = OUT / "branches" / key(row) / f"{layer}-{number:04d}.json"
        if path.exists():
            result = read(path)
            if result["protocol_sha256"] != protocol_sha or result["planned_grid_sequence"] != list(sequence):
                raise ValueError("Resumed branch differs")
        else:
            started = time.perf_counter()
            result = run_sequence(root, model, cfg, _DEVICE, sequence)
            result.update(root=key(row), layer=layer, branch=number, protocol_sha256=protocol_sha,
                          elapsed_seconds=time.perf_counter()-started)
            save_json(path, result)
        records.append(dict(path=str(path), sha256=digest(path), root=key(row), layer=layer,
            branch=number, sequence=list(sequence), terminal=result["terminal"], return_=result["return_"],
            seconds=result["seconds"], absolute_terminal_seconds=result["absolute_terminal_seconds"],
            minimum_clearance=result["minimum_clearance"], elapsed_seconds=result["elapsed_seconds"]))
    return records


def summarize(rows, records, baselines):
    result = []
    for row in rows:
        branches = [r for r in records if r["root"] == key(row)]
        qualified = [r for r in branches if r["terminal"] == "reach_goal" and r["minimum_clearance"] >= .02]
        best = max(qualified, key=lambda r: (r["return_"], -r["branch"])) if qualified else None
        result.append(dict(root=key(row), counts=dict(Counter(r["terminal"] for r in branches)),
            branches=len(branches), qualified=len(qualified),
            fragile=int(sum(r["terminal"] == "reach_goal" and r["minimum_clearance"] < .02 for r in branches)),
            best=best, baseline=baselines[key(row)],
            best_delta_q=None if best is None else best["return_"]-baselines[key(row)]["return_"],
            q_span=float(np.ptp([r["return_"] for r in branches])) if branches else None))
    return result


def execute(workers):
    p = verify()
    device, protocol_sha = p["runtime"]["device"], digest(OUT / "protocol.json")
    init_worker(device)
    baselines = {}
    for row in p["primary"] + p["backups"]:
        path = OUT / "parity" / (key(row)+".json")
        if path.exists():
            parity_record = read(path)
        else:
            root, model, cfg = context(row)
            policy = ValuePolicy(model, cfg, device)
            policy.history.extend(root["history"][:-1])
            policy.last_action = None if root["previous"] is None else ActionXY(*root["previous"])
            commands = policy.candidate_actions()
            native = int(policy.score(root["state"]).argmax())
            if native != row["root"]["grid_action"]:
                raise ValueError("Root native action changed")
            expected = run_branches(root, model, cfg, device, [commands[native]])[0]
            check = run_sequence(root, model, cfg, device, (native,))
            parity = archive_parity(root, check, cfg)
            np.testing.assert_array_equal(check["commands"], expected["commands"])
            if not parity["passed"] or check["return_"] != expected["return_"]:
                raise ValueError("Native continuation/multistep step1 parity failed")
            parity_record = dict(root=key(row), native=native, baseline=check, parity=parity,
                                 protocol_sha256=protocol_sha)
            save_json(path, parity_record)
        if not parity_record["parity"]["passed"] or parity_record["protocol_sha256"] != protocol_sha:
            raise ValueError("Saved parity invalid")
        baselines[key(row)] = {k: parity_record["baseline"][k] for k in
                              ("terminal", "return_", "seconds", "minimum_clearance")}
    started = time.perf_counter()
    all_records, layers = [], []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(device,)) as pool:
        def run_layer(rows, layer, label):
            path = OUT / (label+".json")
            if path.exists():
                data = read(path)
                for r in data["records"]:
                    if digest(Path(r["path"])) != r["sha256"]:
                        raise ValueError("Completed branch changed")
            else:
                layer_start, records = time.perf_counter(), []
                seq = schedules(layer)
                tasks = [(r, layer, list(enumerate(seq))[i:i+10], protocol_sha)
                         for r in rows for i in range(0, len(seq), 10)]
                futures = [pool.submit(block, task) for task in tasks]
                for future in as_completed(futures):
                    records.extend(future.result())
                    if len(records) % 40 == 0:
                        print("PROGRESS", label, len(records), "/", len(rows)*len(seq),
                              "seconds", round(time.perf_counter()-layer_start, 1), flush=True)
                records.sort(key=lambda r: (r["root"], r["branch"]))
                data = dict(label=label, protocol_sha256=protocol_sha, records=records,
                    roots=summarize(rows, records, baselines), elapsed_seconds=time.perf_counter()-layer_start)
                save_json(path, data)
            all_records.extend(data["records"])
            layers.append(dict(label=label, roots=data["roots"], elapsed_seconds=data["elapsed_seconds"]))
            print("LAYER", label, [(r["root"], r["counts"], r["qualified"]) for r in data["roots"]], flush=True)
            return data

        run_layer(p["primary"], "constant", "primary-constant")
        primary = summarize(p["primary"], all_records, baselines)
        if sum(r["qualified"] > 0 for r in primary) < 2:
            run_layer(p["primary"], "local", "primary-local")
            primary = summarize(p["primary"], all_records, baselines)
        count = sum(r["qualified"] > 0 for r in primary)
        eligible_backups = []
        if count == 1:
            single = run_layer(p["backups"], "single", "backup-single")
            eligible_backups = [row for row in p["backups"] if not next(
                r for r in single["roots"] if r["root"] == key(row))["qualified"]]
            run_layer(p["backups"], "constant", "backup-constant")
            backup_multi = [r for r in all_records if r["root"] in {key(v) for v in eligible_backups} and r["layer"] != "single"]
            if not any(r["qualified"] for r in summarize(eligible_backups, backup_multi, baselines)):
                run_layer(p["backups"], "local", "backup-local")
    multi = [r for r in all_records if r["layer"] != "single"]
    eligible = p["primary"] + eligible_backups
    results = summarize(eligible, multi, baselines)
    rescued_cases = {row["case"] for row, r in zip(eligible, results) if r["qualified"]}
    verdict = ("MULTISTEP_HEADROOM_CONFIRMED" if len(rescued_cases) >= 2 else
               "NO_BOUNDED_MULTISTEP_HEADROOM_FOUND" if count == 0 else
               "INSUFFICIENT_REPEATABLE_MULTISTEP_HEADROOM")
    save_json(OUT / "summary.json", dict(verdict=verdict, protocol_sha256=protocol_sha,
        layers=layers, primary=primary, eligible_results=results, rescued_cases=sorted(rescued_cases),
        backup_activated=count==1, branches=len(all_records), elapsed_seconds=time.perf_counter()-started,
        limits="Finite oracle intervention headroom only; not a deployable navigator, overall SR, "
               "shared root cause, planning necessity, or new-method novelty."))
    print("VERDICT", verdict, "cases", sorted(rescued_cases), "branches", len(all_records), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze(args.device)
    else:
        execute(args.workers)


if __name__ == "__main__":
    main()
