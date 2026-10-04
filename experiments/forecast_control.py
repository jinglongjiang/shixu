"""Bounded full IL/MC-RL experiment using the shared navigation evaluator."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from experiments.occlusion import comparison, configuration, evaluate_weights, queue, run, source_hash
from shixu.model import build_model
from shixu.policy import ValuePolicy
from shixu.observations import OccludedTracks
from shixu.runner import environment


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


def reference(root, prior_root, protocol, data_path, device, seeds):
    """Reuse an exactly equivalent CV critic; never fabricate training logs."""
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    for seed in seeds:
        original = prior_root / str(seed) / "gru_context/model.pt"
        checkpoint = torch.load(original, map_location="cpu", weights_only=False)
        archived = json.loads(original.with_name("result.json").read_text())
        if (checkpoint["seed"] != seed or checkpoint["il_episodes"] != protocol["il_episodes"]
                or checkpoint["rl_episodes"] != protocol["rl_episodes"]
                or archived["demonstration_sha256"] != data["original_sha256"]):
            raise ValueError("Reference data or training budget differs")
        cfg = configuration(protocol, "cv")
        model = build_model(cfg)
        model.critic.load_state_dict(checkpoint["model"], strict=True)
        output = root / str(seed) / "cv"
        output.mkdir(parents=True, exist_ok=True)
        if (output / "result.json").exists():
            raise RuntimeError("Do not overwrite a completed reference")
        weights = output / "model.pt"
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
        torch.save({**checkpoint, "model": model.state_dict(),
                    "config": {s: dict(cfg[s]) for s in cfg.sections()},
                    "reference_sha256": original_sha,
                    "framework": "Exact native-CV wrapper; reused original trained critic, no new training"}, weights)
        result = evaluate_weights(weights, protocol, device, protocol["development_cases"], seed)
        baseline = json.loads((root.parent / "forecast_control_a" / str(seed) / "parent_gru/result_gpu.json").read_text())
        for actual, expected in zip(result["episodes"], baseline["episodes"]):
            for field in ("people", "geometry", "case", "terminal", "navigation_time", "path",
                          "minimum_clearance", "return_", "actions"):
                if actual[field] != expected[field]:
                    raise ValueError("CV wrapper violates native GPU parity: " + field)
        result.update(protocol=protocol, source_sha256=source_hash(), seed=seed, arm="cv",
                      demonstration_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                      reference=str(original), reference_sha256=original_sha, reference_reused=True,
                      original_demonstration_sha256=data["original_sha256"], parameters=sum(p.numel() for p in model.parameters()),
                      training_seconds=archived["training_seconds"], new_training_seconds=0,
                      torch=torch.__version__, host=platform.node(), device=device,
                      scope="Reused verified full-budget Parent critic; exact command/outcome parity on192 cases")
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print("REFERENCE_PARITY", seed, len(result["episodes"]), flush=True)


def parity(prior_root, protocol, device):
    import configparser
    import numpy as np
    checkpoint = torch.load(prior_root / "419/gru_context/model.pt", map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    parent = ValuePolicy(build_model(cfg), cfg, device)
    parent.model.load_state_dict(checkpoint["model"])
    bridge_cfg = configuration(protocol, "cv")
    bridge = ValuePolicy(build_model(bridge_cfg), bridge_cfg, device)
    bridge.model.critic.load_state_dict(checkpoint["model"])
    env = environment(cfg, parent, "circle", 5)
    env.reset(options={"test_case": protocol["development_cases"][0]})
    tracks = OccludedTracks(protocol["retention_seconds"])
    for tick in range(30):
        state = tracks.observe(env)
        np.testing.assert_array_equal(parent.score(state), bridge.score(state))
        first, second = parent.predict(state), bridge.predict(state)
        if first != second:
            raise ValueError("Native execution parity failed")
        _, _, done, truncated, _ = env.step(first)
        if done or truncated:
            break
    print("CV_PARITY", tick + 1, "native steps, identical80 scores and executed commands", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "queue", "parent", "reference", "parity", "smoke", "summarize"))
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
    elif args.mode == "parity":
        parity(Path(args.prior_root), protocol, args.device)
    elif args.mode == "reference":
        reference(Path(args.root), Path(args.prior_root), protocol, Path(args.data), args.device,
                  args.seeds or protocol["seeds"])
    elif args.mode == "smoke":
        protocol.update(il_episodes=2, il_epochs=1, rl_episodes=2, people=[5], geometries=["circle"],
                        development_cases=protocol["development_cases"][:1])
        data = torch.load(args.source, map_location="cpu", weights_only=False)
        output = Path(args.root)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "demonstrations.pt"
        torch.save(dict(protocol=protocol, episodes=data["episodes"][:2]), path)
        for arm in ("current", "gru", "kda"):
            run(output, protocol, arm, args.seed, path, args.device)
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
