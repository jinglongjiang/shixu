"""Bounded full IL/MC-RL experiment using the shared navigation evaluator."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.occlusion import comparison, evaluate_weights, queue, run, source_hash


PROTOCOL = Path(__file__).with_name("forecast_control_protocol.json")


def prepare(source, destination, protocol):
    if destination.exists():
        raise RuntimeError("Do not overwrite the shared demonstration archive")
    data = torch.load(source, map_location="cpu", weights_only=False)
    shared = ("il_episodes", "il_case_start", "training_people", "training_geometry", "retention_seconds")
    if any(data["protocol"][key] != protocol[key] for key in shared):
        raise ValueError("Existing demonstration collection does not match this task")
    if len(data["episodes"]) != protocol["il_episodes"]:
        raise ValueError("Incomplete shared demonstration cohort")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**data, "protocol": protocol, "original_protocol": data["protocol"],
                "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_sha256": source_hash()}, destination)
    print("REUSED", len(data["episodes"]), "byte-identical episode contents, no new demonstrations", flush=True)


def parent(root, prior_root, protocol, device, seeds):
    for seed in seeds:
        weights = prior_root / str(seed) / "gru_context" / "model.pt"
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if (checkpoint["seed"] != seed or checkpoint["il_episodes"] != protocol["il_episodes"]
                or checkpoint["rl_episodes"] != protocol["rl_episodes"]):
            raise ValueError("Parent checkpoint does not match the fixed training budget")
        destination = root / str(seed) / "parent_gru" / "result.json"
        if destination.exists():
            raise RuntimeError("Do not overwrite a completed parent evaluation")
        result = evaluate_weights(weights, protocol, device, protocol["development_cases"], seed)
        result.update(seed=seed, arm="parent_gru", protocol=protocol, reference=str(weights),
                      scope="Frozen completed V8 parent, re-evaluated on identical new cases, not retrained")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "queue", "parent", "summarize"))
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--root", default="outputs/forecast_control_a")
    parser.add_argument("--data", default="outputs/forecast_control_a/demonstrations.pt")
    parser.add_argument("--source", default="outputs/occlusion_v8/demonstrations.pt")
    parser.add_argument("--prior-root", default="outputs/occlusion_v8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.set_defaults(il_root=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(Path(args.protocol).read_text())
    if args.mode == "prepare":
        prepare(Path(args.source), Path(args.data), protocol)
    elif args.mode in ("queue", "run"):
        data = torch.load(args.data, map_location="cpu", weights_only=False)
        if data["source_sha256"] != source_hash():
            raise RuntimeError("Training implementation changed after protocol/data freeze")
        if args.mode == "queue":
            queue(args, protocol)
        else:
            run(Path(args.root), protocol, args.arm, args.seed, args.data, args.device)
    elif args.mode == "parent":
        parent(Path(args.root), Path(args.prior_root), protocol, args.device, args.seeds or protocol["seeds"])
    else:
        comparison(Path(args.root), protocol)


if __name__ == "__main__":
    main()
