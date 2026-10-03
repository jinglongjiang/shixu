"""Auditable occlusion IL/RL loop: shared demonstrations and reloaded weights."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.temporal_order import summarize
from shixu.model import build_model, load_weights
from shixu.policy import ValuePolicy
from shixu.runner import environment, orca_teacher, run_episode
from shixu.training import train


PROTOCOL = Path(__file__).with_name("occlusion_protocol.json")


def configuration(protocol, arm):
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
    cfg.add_section("observation")
    for section, key, value in (
        ("model", "architecture", "occlusion"), ("model", "representation", "tracks"),
        ("model", "backbone", arm), ("model", "width", protocol["width"]),
        ("model", "layers", protocol["layers"]), ("buffer", "seq_len", protocol["history"]),
        ("observation", "retention_seconds", protocol["retention_seconds"]),
        ("train", "il_epochs", protocol["il_epochs"]),
        ("train", "batch_size", protocol["batch_size"]),
        ("train", "il_batch_size", protocol["batch_size"]),
        ("train", "updates_per_ep", protocol["rl_updates_per_episode"])
    ):
        cfg.set(section, key, str(value))
    return cfg


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("shixu", "vendor/crowd_sim"):
        for file in sorted((root / folder).rglob("*.py")):
            digest.update(str(file.relative_to(root)).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def collect(path, protocol):
    cfg = configuration(protocol, "current")
    policy = ValuePolicy(build_model(cfg), cfg)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    teacher, episodes, attempts = orca_teacher(cfg), [], []
    for case in range(protocol["il_case_start"], protocol["il_case_start"] + cfg.getint("imitation_learning", "max_il_prefill")):
        episode = run_episode(env, policy, case, teacher=teacher, phase="train")
        attempts.append({"case": case, "terminal": episode["terminal"]})
        if episode["terminal"] == "reach_goal":
            episodes.append(episode)
        if len(episodes) == protocol["il_episodes"]:
            break
    if len(episodes) != protocol["il_episodes"]:
        raise RuntimeError("Insufficient successful legal-observation ORCA demonstrations")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": protocol, "episodes": episodes, "attempts": attempts}, path)
    print("COLLECTED", len(episodes), "successful /", len(attempts), "attempts", flush=True)


def exposure(episode):
    rows = episode["observations"]
    reentries, previous, seen = 0, set(), set()
    for row in rows:
        visible = {key["track_id"] for key, measured in zip(row["humans"], row["observed"]) if measured}
        reentries += len((visible - previous) & seen)
        seen.update(visible)
        previous = visible
    return {"person_frames": sum(row["population"] for row in rows),
            "hidden_person_frames": sum(row["population"] - row["visible"] for row in rows),
            "retained_hidden_person_frames": sum(row["retained_hidden"] for row in rows),
            "reentries": reentries}


def evaluate_weights(path, protocol, device, cases, seed):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    policy = ValuePolicy(build_model(cfg), cfg, device)
    load_weights(policy.model, path, device)
    np.random.seed(seed)
    records, timings = [], []
    score = policy.score

    def timed_score(state):
        start = time.perf_counter()
        value = score(state)
        timings.append(1000 * (time.perf_counter() - start))
        return value

    policy.score = timed_score
    for people in protocol["people"]:
        for geometry in protocol["geometries"]:
            env = environment(cfg, policy, geometry, people)
            for case in cases:
                episode = run_episode(env, policy, case)
                row = {key: episode[key] for key in ("case", "terminal", "navigation_time", "path", "minimum_clearance")}
                row.update(people=people, geometry=geometry, return_=float(sum(episode["rewards"])),
                           exposure=exposure(episode), actions=episode["actions"])
                records.append(row)
            print("EVALUATED", path.parent.name, people, geometry, summarize(records[-len(cases):]), flush=True)
    return {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_phase": checkpoint.get("phase", "final"), "episodes": records,
            "summary": summarize(records),
            "score_median_ms": float(np.median(timings)), "score_p95_ms": float(np.quantile(timings, .95)),
            "timing_scope": "Complete 80-action scoring during evaluation, includes transfers; concurrent training may contend",
            "cells": [{"people": people, **summarize([r for r in records if r["people"] == people])}
                      for people in protocol["people"]]}


def run(root, protocol, arm, seed, data_path, device):
    output = root / str(seed) / arm
    if (output / "result.json").exists():
        raise RuntimeError("A completed result already exists; do not overwrite evidence")
    demonstrations = torch.load(data_path, map_location="cpu", weights_only=False)
    if demonstrations["protocol"] != protocol:
        raise ValueError("Demonstration collection protocol mismatch")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = configuration(protocol, arm)
    policy = ValuePolicy(build_model(cfg), cfg, device)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(policy.device)
    with (output / "learning.jsonl").open("w") as log:
        def report(row):
            if not np.isfinite(row["loss"]):
                raise RuntimeError("Nonfinite training loss")
            row["elapsed_seconds"] = time.perf_counter() - started
            log.write(json.dumps(row) + "\n")
            log.flush()
            if row["phase"] == "il" and row["epoch"] == protocol["il_epochs"]:
                torch.save({"model": policy.model.state_dict(), "seed": seed, "phase": "il",
                            "config": {s: dict(cfg[s]) for s in cfg.sections()}}, output / "il.pt")
            if row["phase"] == "il" and row["epoch"] % 10 == 0 or row["phase"] == "rl" and row["episode"] % 100 == 0:
                print(arm, seed, row, flush=True)
        train(env, policy, cfg, output / "model.pt", seed, protocol["il_episodes"], protocol["rl_episodes"],
              demonstrations=demonstrations["episodes"], rl_case_start=protocol["rl_case_start"], report=report)
    training_seconds = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(policy.device) / 1024 ** 2 if policy.device.type == "cuda" else None
    parameters = sum(p.numel() for p in policy.model.parameters())
    del policy, env
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    start = time.perf_counter()
    result = evaluate_weights(output / "model.pt", protocol, device, protocol["development_cases"], seed)
    result.update(protocol=protocol, source_sha256=source_hash(), seed=seed, arm=arm,
                  demonstration_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                  parameters=parameters, training_seconds=training_seconds,
                  evaluation_seconds=time.perf_counter() - start, peak_allocated_mib=peak_mib,
                  torch=torch.__version__, host=platform.node(), device=str(device),
                  hardware=torch.cuda.get_device_name() if str(device).startswith("cuda") else "CPU")
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", arm, seed, result["summary"], flush=True)


def queue(args, protocol):
    pending = [(seed, arm) for seed in (args.seeds or protocol["seeds"])
               for arm in (args.arms or protocol["arms"])]
    active, failed = [], []
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < args.workers:
            seed, arm = pending.pop(0)
            if (root / str(seed) / arm / "result.json").exists():
                print("ALREADY_COMPLETE", seed, arm, flush=True)
                continue
            log = (root / (str(seed) + "_" + arm + ".log")).open("w")
            command = [sys.executable, "-m", "experiments.occlusion", "run", "--protocol", args.protocol,
                       "--root", args.root, "--data", args.data, "--device", args.device,
                       "--seed", str(seed), "--arm", arm]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active.append((child, log, seed, arm))
            print("STARTED", seed, arm, child.pid, flush=True)
        remaining = []
        for child, log, seed, arm in active:
            status = child.poll()
            if status is None:
                remaining.append((child, log, seed, arm))
            else:
                log.close()
                print("EXIT", seed, arm, status, flush=True)
                if status:
                    failed.append((seed, arm, status))
        active = remaining
        time.sleep(2)
    if failed:
        raise RuntimeError("Failed arms: " + repr(failed))


def comparison(root, protocol):
    models = {}
    for arm in protocol["arms"]:
        rows = [json.loads((root / str(seed) / arm / "result.json").read_text()) for seed in protocol["seeds"]]
        if any(r["protocol"] != protocol for r in rows):
            raise ValueError("Result protocol mismatch")
        models[arm] = {"summary": summarize([e for r in rows for e in r["episodes"]]),
                       "primary_by_seed": [summarize([e for e in r["episodes"] if e["people"] > 5]) for r in rows],
                       "cells": [{"people": n, **summarize([e for r in rows for e in r["episodes"] if e["people"] == n])}
                                 for n in protocol["people"]],
                       "parameters": rows[0]["parameters"],
                       "training_seconds": [r["training_seconds"] for r in rows],
                       "inference_median_ms": [r["score_median_ms"] for r in rows],
                       "exposure": {key: sum(e["exposure"][key] for r in rows for e in r["episodes"])
                                    for key in rows[0]["episodes"][0]["exposure"]}}
    contrasts = []
    for control in ("gru", "current"):
        a, b = models["kda"]["primary_by_seed"], models[control]["primary_by_seed"]
        changes = {key: [100 * (x[key] - y[key]) for x, y in zip(a, b)] for key in ("sr", "cr", "tr")}
        time_change = np.mean([x["success_time"] / y["success_time"] - 1 for x, y in zip(a, b)
                               if x["success_time"] and y["success_time"]])
        positive = (np.mean(changes["sr"]) >= protocol["meaningful_gain_pp"]
                    and sum(d > 0 for d in changes["sr"]) >= protocol["required_same_direction_seeds"]
                    and np.mean(changes["cr"]) <= protocol["maximum_collision_increase_pp"]
                    and np.mean(changes["tr"]) <= protocol["maximum_timeout_increase_pp"]
                    and np.isfinite(time_change) and time_change <= protocol["maximum_success_time_increase_fraction"])
        contrasts.append({"control": control, "seed_changes_pp": changes,
                          "mean_changes_pp": {k: float(np.mean(v)) for k, v in changes.items()},
                          "success_time_change_fraction": float(time_change),
                          "development_positive": bool(positive)})
    result = {"protocol": protocol, "models": models, "contrasts": contrasts,
              "status": "DEVELOPMENT_POSITIVE_REQUIRES_FRESH_CONFIRMATION" if all(c["development_positive"] for c in contrasts)
                        else "VERSION_REQUIRES_DIAGNOSIS_NOT_DIRECTION_REJECTED"}
    (root / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "run", "evaluate", "queue", "summarize"))
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--data")
    parser.add_argument("--root")
    parser.add_argument("--arm", choices=("current", "gru", "kda"))
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("current", "gru", "kda"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(Path(args.protocol).read_text())
    if args.mode == "collect":
        collect(Path(args.data), protocol)
    elif args.mode == "run":
        run(Path(args.root), protocol, args.arm, args.seed, args.data, args.device)
    elif args.mode == "queue":
        queue(args, protocol)
    elif args.mode == "summarize":
        comparison(Path(args.root), protocol)
    else:
        rows = evaluate_weights(args.weights, protocol, args.device,
                                protocol["confirmation_cases"] if args.confirmation else protocol["development_cases"], args.seed)
        destination = Path(args.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
