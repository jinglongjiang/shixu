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


def gate_probe(policy, state, mode="native", constant=None, shuffled=None):
    """Read-gate intervention only: weights, memory writes and state stay frozen."""
    from torch.nn import functional as F
    model, stats = policy.model, {}

    def intervene(layer, inputs):
        x = inputs[0]
        if mode == "zero_evidence":
            evidence = torch.zeros_like(x[..., -4:])
        elif mode == "validity_only":
            evidence = torch.cat((torch.zeros_like(x[..., -4:-1]), x[..., -1:]), -1)
        elif mode == "shuffled_motion":
            evidence = torch.cat((shuffled[None, None].expand(*x.shape[:-1], 3), x[..., -1:]), -1)
        else:
            return
        return (torch.cat((x[..., :-4], evidence), -1),)

    def capture(layer, inputs, output):
        x, width = inputs[0], model.temporal_encoder.width
        f, m = x[..., :width], x[..., width:2 * width]
        motion = F.linear(x[..., -4:-1], layer.weight[:, -4:-1])
        valid = F.linear(x[..., -1:], layer.weight[:, -1:])
        p = output.sigmoid()
        stats.update(gate_mean=float(p.mean()), gate_std=float(p.std(unbiased=False)),
                     gate_low_fraction=float((p < .1).float().mean()),
                     gate_high_fraction=float((p > .9).float().mean()),
                     candidate_gate_std=float(p.std(1, unbiased=False).mean()),
                     motion_logit_mae=float(motion.abs().mean()),
                     validity_logit_mae=float(valid.abs().mean()),
                     motion_gate_effect=float((p - (output - motion).sigmoid()).abs().mean()),
                     all_evidence_gate_effect=float((p - (output - motion - valid).sigmoid()).abs().mean()),
                     gated_memory_to_current=float((p * m).norm() / f.norm().clamp_min(1e-8)),
                     channel_means=p.mean((0, 1, 2)).cpu().tolist())
        if mode == "half_gate":
            return torch.zeros_like(output)
        if mode == "ungated":
            return torch.full_like(output, torch.inf)
        if mode == "constant_gate":
            return torch.logit(constant.to(output).clamp(1e-6, 1 - 1e-6)).expand_as(output)

    before = model.evidence_gate.register_forward_pre_hook(intervene)
    after = model.evidence_gate.register_forward_hook(capture)
    try:
        scores = policy.score(state)
    finally:
        before.remove()
        after.remove()
    return scores, stats


def gate_diagnostic(root, protocol, device):
    from collections import deque
    from crowd_sim.envs.utils.state import FullState, ObservableState
    from shixu.features import motion_evidence
    from shixu.observations import TrackedState
    from experiments.temporal_revision_shadow import observed_event
    cfg = configuration(protocol, "actor_gru")
    parent = ValuePolicy(build_model(cfg), cfg, device)
    load_weights(parent.model, root / "191/actor_gru/model.pt", device)
    roots, episodes = [], []
    for geometry in protocol["geometries"]:
        for people in protocol["people"]:
            for case in protocol["evaluation_cases"][:2]:
                record = run_episode(environment(cfg, parent, geometry, people), parent, case)
                ticks = set(t for t in (12, 24, 36) if t < len(record["tokens"]))
                first_event = next((t for t in range(12, len(record["tokens"])) if observed_event(record, t) is not None), None)
                if first_event is not None:
                    ticks.add(first_event)
                episodes.append({"geometry": geometry, "people": people, "case": case, "terminal": record["terminal"], "frames": len(record["tokens"]), "first_event": first_event})
                for tick in sorted(ticks):
                    observation = record["observations"][tick]
                    state = TrackedState(FullState(*observation["robot"]),
                                         [ObservableState(*h["state"]) for h in observation["humans"]],
                                         tuple(h["track_id"] for h in observation["humans"]))
                    roots.append(({"geometry": geometry, "people": people, "case": case, "tick": tick,
                                   "motion_event": tick == first_event}, state, record["tokens"][max(0, tick - parent.length):tick]))
    results = []
    for seed in protocol["seeds"]:
        for arm in ("kda_gate", "kda_evidence"):
            cfg = configuration(protocol, arm)
            policy = ValuePolicy(build_model(cfg), cfg, device)
            load_weights(policy.model, root / str(seed) / arm / "model.pt", device)
            native, changes = [], []
            for info, state, history in roots:
                policy.history = deque(history, maxlen=policy.length)
                scores, stats = gate_probe(policy, state)
                native.append((scores, stats))
                prefix = torch.as_tensor(np.asarray(history + [policy.encode(state)])[None], device=policy.device)
                changes.append(motion_evidence(prefix)[0, -1, :, :3])
            constant = torch.tensor(np.mean([s["channel_means"] for _, s in native], axis=0),
                                    device=policy.device, dtype=policy.model.evidence_gate.weight.dtype)
            modes = ("half_gate", "constant_gate", "ungated") if arm == "kda_gate" else ("zero_evidence", "validity_only", "shuffled_motion")
            rows = []
            for index, (info, state, history) in enumerate(roots):
                policy.history = deque(history, maxlen=policy.length)
                scores, stats = native[index]
                variants = {}
                for mode in modes:
                    other, _ = gate_probe(policy, state, mode, constant, changes[(index + 1) % len(roots)])
                    variants[mode] = {"action": int(np.argmax(other)), "action_changed": bool(np.argmax(other) != np.argmax(scores)),
                                      "score_mae": float(np.abs(other - scores).mean())}
                rows.append({**info, "native_action": int(np.argmax(scores)), "stats": {k: v for k, v in stats.items() if k != "channel_means"}, "interventions": variants})
            results.append({"seed": seed, "arm": arm, "roots": len(rows),
                            "channel_mean_root_std": float(np.std([s["channel_means"] for _, s in native], axis=0).mean()),
                            "constant_channel_gate": constant.cpu().tolist(),
                            "mean_stats": {k: float(np.mean([r["stats"][k] for r in rows])) for k in rows[0]["stats"]},
                            "changed_actions": {mode: sum(r["interventions"][mode]["action_changed"] for r in rows) for mode in modes}, "records": rows})
    result = {"episodes": episodes, "roots": len(roots), "results": results,
              "scope": "Frozen read-gate diagnostic, no training and no outcome improvement claim. Uniform ticks plus first arrived near motion event from pre-fixed parent episodes. Constant gates estimated on these roots; interventions can be out of distribution and do not isolate training causality."}
    (root / "gate_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"roots": len(roots), "results": [{k: v for k, v in r.items() if k not in ("records", "constant_channel_gate")} for r in results]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "queue", "summarize", "latency", "events", "gate-diagnostic"))
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
    elif args.mode == "gate-diagnostic":
        gate_diagnostic(root, protocol, args.device)
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
