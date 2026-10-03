"""Controlled memory training and evidence-specific paired contrasts."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import MethodType

import numpy as np
import torch

from shixu.model import build_model, load_weights
from shixu.observations import TrackKeys
from shixu.policy import ValuePolicy
from shixu.runner import environment, run_episode
from shixu.training import train
from experiments.temporal_order import evaluate, summarize


PROTOCOL = Path(__file__).with_name("temporal_memory_protocol.json")


def configuration(protocol, arm):
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
    settings = dict(protocol["arms"][arm], representation="aligned", feature_contract="observed",
                    width=protocol["width"], layers=protocol["layers"])
    if "readout" in settings:
        settings["architecture"] = "memory"
    for key, value in settings.items():
        cfg.set("model", key, str(value))
    for section, key, value in (("buffer", "seq_len", protocol["history"]),
                                ("train", "il_epochs", protocol["il_epochs"]),
                                ("train", "updates_per_ep", protocol["rl_updates_per_episode"]),
                                ("train", "batch_size", protocol["batch_size"]),
                                ("train", "il_batch_size", protocol["batch_size"])):
        cfg.set(section, key, str(value))
    return cfg


def load_data(path, protocol):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != protocol["demonstration_sha256"]:
        raise ValueError("Demonstration hash differs from frozen cohort")
    data = torch.load(path, map_location="cpu", weights_only=False)
    original = json.loads(PROTOCOL.with_name("temporal_protocol.json").read_text())
    if data["protocol"] != original:
        raise ValueError("Demonstration provenance differs from original collection")
    return data["episodes"]


def run_arm(root, protocol, arm, seed, data, device):
    output = root / str(seed) / arm
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        if result["protocol"] != protocol:
            raise ValueError("Existing result uses a different protocol")
        print("COMPLETE_ALREADY", seed, arm, flush=True)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = configuration(protocol, arm)
    model = build_model(cfg)
    policy = ValuePolicy(model, cfg, device)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    demonstrations = load_data(data, protocol)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(policy.device) if policy.device.type == "cuda" else None
    with (output / "learning.jsonl").open("w") as log:
        def report(row):
            if not np.isfinite(row["loss"]):
                raise RuntimeError("Nonfinite loss; stop this arm rather than hide instability")
            row["elapsed_seconds"] = time.perf_counter() - started
            log.write(json.dumps(row) + "\n")
            log.flush()
            if (row["phase"] == "il" and row["epoch"] % 10 == 0
                    or row["phase"] == "rl" and row["episode"] % 100 == 0):
                print(seed, arm, row, flush=True)
        train(env, policy, cfg, output / "model.pt", seed, protocol["il_episodes"], protocol["rl_episodes"],
              demonstrations=demonstrations, rl_case_start=protocol["rl_case_start"], report=report)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    records = evaluate(policy, cfg, protocol)
    result = {"protocol": protocol, "seed": seed, "arm": arm, "summary": summarize(records),
              "episodes": records, "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": training_seconds, "evaluation_seconds": time.perf_counter() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated(policy.device) / 1024 ** 2 if policy.device.type == "cuda" else None,
              "torch": torch.__version__, "device": str(policy.device), "host": platform.node(),
              "hardware": torch.cuda.get_device_name(policy.device) if policy.device.type == "cuda" else "CPU"}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", seed, arm, result["summary"], flush=True)


def paired_summary(root, protocol):
    results = {arm: [json.loads((root / str(seed) / arm / "result.json").read_text())
                     for seed in protocol["seeds"]] for arm in protocol["arms"]}
    means = {}
    for arm, records in results.items():
        episodes = [ep for record in records for ep in record["episodes"]]
        means[arm] = {"summary": summarize(episodes), "seeds": [r["summary"] for r in records],
                      "parameters": records[0]["parameters"],
                      "training_seconds": [r["training_seconds"] for r in records],
                      "cells": [{"people": people, "geometry": geometry,
                                 **summarize([e for e in episodes if e["people"] == people and e["geometry"] == geometry])}
                                for people in protocol["people"] for geometry in protocol["geometries"]]}
    contrasts = []
    for custom, control in protocol["contrast_pairs"]:
        differences = {key: [100 * (a["summary"][key] - b["summary"][key])
                             for a, b in zip(results[custom], results[control])] for key in ("sr", "cr", "tr")}
        custom_time, control_time = [means[arm]["summary"]["success_time"] for arm in (custom, control)]
        time_change = custom_time / control_time - 1 if custom_time and control_time else None
        positive = (np.mean(differences["sr"]) >= protocol["meaningful_success_gain_pp"]
                    and all(d > 0 for d in differences["sr"])
                    and np.mean(differences["cr"]) <= protocol["maximum_collision_increase_pp"]
                    and np.mean(differences["tr"]) <= protocol["maximum_timeout_increase_pp"]
                    and time_change is not None and time_change <= protocol["maximum_success_time_increase_fraction"])
        contrasts.append({"custom": custom, "control": control, "seed_differences_pp": differences,
                          "mean_differences_pp": {key: float(np.mean(value)) for key, value in differences.items()},
                          "success_time_change_fraction": time_change,
                          "verdict": "PILOT_POSITIVE" if positive else "NO_CONSISTENT_PILOT_GAIN"})
    result = {"protocol": protocol, "models": means, "contrasts": contrasts,
              "scope": protocol["scope"], "method_entry": False}
    (root / "paired_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"models": {arm: value["summary"] for arm, value in means.items()}, "contrasts": contrasts}), flush=True)


def cached_actor_control(model, prefix, queries):
    """Same trained actor GRU, reusable prefix; compute control, not a new arm."""
    features, visible = model._features(prefix)
    batch, length, people, width = features.shape
    sequences = features.permute(0, 2, 1, 3).reshape(batch * people, length, width)
    masks = visible.permute(0, 2, 1).reshape(batch * people, length)
    gru = model.temporal_encoder.backend
    hidden = features.new_zeros(gru.num_layers, batch * people, width)
    if length and bool((masks.all(1) | ~masks.any(1)).all()):
        _, hidden = gru(sequences)
        hidden = hidden * masks.any(1)[None, :, None]
    else:
        for tick in range(length):
            _, proposed = gru(sequences[:, tick:tick + 1], hidden)
            hidden = torch.where(masks[:, tick][None, :, None], proposed, hidden)
    current, observed = model._features(queries)
    count = queries.shape[1]
    hidden = hidden.reshape(gru.num_layers, batch, people, width)[:, :, None].expand(
        -1, -1, count, -1, -1).reshape(gru.num_layers, batch * count * people, width)
    _, proposed = gru(current.reshape(batch * count * people, 1, width), hidden)
    hidden = torch.where(observed.reshape(1, -1, 1), proposed, hidden)
    actor = model.temporal_encoder.norm(hidden[-1]).reshape(batch, count, people, width)
    return model.value_head(model._pool(actor, observed)).squeeze(-1)


def latency(root, protocol, seed, device):
    rows = {}
    for label in list(protocol["arms"]) + ["actor_gru_cached"]:
        arm = "actor_gru" if label == "actor_gru_cached" else label
        cfg = configuration(protocol, arm)
        policy = ValuePolicy(build_model(cfg), cfg, device)
        load_weights(policy.model, root / str(seed) / arm / "model.pt", device)
        if label == "actor_gru_cached":
            policy.model.score_candidates = MethodType(cached_actor_control, policy.model)
        env = environment(cfg, policy)
        env.reset(options={"test_case": protocol["evaluation_cases"][0]})
        state = TrackKeys().observe(env)
        policy.history.extend([policy.encode(state)] * protocol["history"])
        samples = []
        for repetition in range(120):
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            started = time.perf_counter()
            policy.score(state)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            if repetition >= 20:
                samples.append(1000 * (time.perf_counter() - started))
        rows[label] = {"median_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, .95)), "samples_ms": samples}
    result = {"hardware": torch.cuda.get_device_name() if device.startswith("cuda") else "CPU", "torch": torch.__version__,
              "host": platform.node(), "threads": torch.get_num_threads(),
              "seed": seed, "repetitions": 100, "warmup": 20,
              "scope": "Complete 80-action score, shared real prefix, no simulator or smoothing", "arms": rows}
    (root / "latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({arm: row["median_ms"] for arm, row in rows.items()}), flush=True)


def shadow_contrasts(events, pairs):
    contrasts = []
    for custom, control in pairs:
        paired = [(row["branches"][custom], row["branches"][control]) for row in events]
        contrasts.append({"custom": custom, "control": control,
                          "different_root_actions": sum(int(a["root_index"] != b["root_index"]) for a, b in paired),
                          "safe_progress_wins_005m": sum(int(a["progress"] - b["progress"] >= .05 and a["minimum_clearance"] >= 0) for a, b in paired),
                          "safe_progress_losses_005m": sum(int(b["progress"] - a["progress"] >= .05 and b["minimum_clearance"] >= 0) for a, b in paired),
                          "custom_collisions": sum(int(a["terminal"] == "collision") for a, _ in paired),
                          "control_collisions": sum(int(b["terminal"] == "collision") for _, b in paired)})
    return contrasts


def event_shadow(root, protocol, seed, device):
    from experiments.temporal_revision_shadow import branch, observed_event
    native = json.loads(PROTOCOL.with_name("temporal_protocol.json").read_text())
    policies = {}
    for arm in protocol["arms"]:
        cfg = configuration(protocol, arm)
        policy = ValuePolicy(build_model(cfg), cfg, device)
        load_weights(policy.model, root / str(seed) / arm / "model.pt", device)
        policies[arm] = policy
    parent = policies["actor_gru"]
    episodes, events = [], []
    for geometry in protocol["geometries"]:
        for case in native["headroom_cases"]:
            record = run_episode(environment(parent.config, parent, geometry, 5), parent, case)
            event = None
            for tick in range(12, len(record["tokens"])):
                key = observed_event(record, tick)
                if key is not None:
                    event = (tick, key)
                    break
            episodes.append({"geometry": geometry, "case": case, "frames": len(record["tokens"]),
                             "terminal": record["terminal"], "eligible_event": event})
            if event is not None:
                tick, key = event
                branches = {arm: branch(policy, policy.config, geometry, record, tick, key, "full", 12)
                            for arm, policy in policies.items()}
                events.append({"geometry": geometry, "case": case, "tick": tick, "actor_key": key, "branches": branches})
    result = {"seed": seed, "native_episodes": len(episodes), "eligible_first_events": len(events),
              "uniform_person_frames": 5 * sum(ep["frames"] for ep in episodes),
              "episodes": episodes, "events": events, "contrasts": shadow_contrasts(events, protocol["contrast_pairs"]),
              "scope": "Exploratory common-root 3-second continuations. First legal near motion event per parent episode; no favorable-root selection. Does not establish motion/context disentanglement or population recovery."}
    (root / "event_shadow.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("native_episodes", "eligible_first_events", "uniform_person_frames", "contrasts")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "queue", "summarize", "latency", "events"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", choices=tuple(json.loads(PROTOCOL.read_text())["arms"]))
    parser.add_argument("--seeds", type=int, nargs="+", default=[191, 223])
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol, root = json.loads(PROTOCOL.read_text()), Path(args.root)
    if args.mode == "summarize":
        paired_summary(root, protocol)
    elif args.mode == "latency":
        latency(root, protocol, args.seeds[0], args.device)
    elif args.mode == "events":
        event_shadow(root, protocol, args.seeds[0], args.device)
    elif args.mode == "queue":
        for arm in args.arms or protocol["arms"]:
            for seed in args.seeds:
                subprocess.run([sys.executable, "-m", "experiments.temporal_memory", "run", "--root", str(root),
                                "--arm", arm, "--seeds", str(seed), "--data", args.data, "--device", args.device], check=True)
    else:
        if not args.arm or not args.data or len(args.seeds) != 1 or args.seeds[0] not in protocol["seeds"]:
            parser.error("Run needs one frozen seed, one arm and the fixed IL data")
        import fcntl
        output = root / str(args.seeds[0]) / args.arm
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run_arm(root, protocol, args.arm, args.seeds[0], args.data, args.device)


if __name__ == "__main__":
    main()
