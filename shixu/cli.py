import argparse
import configparser
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import build_model, load_weights
from .policy import ValuePolicy
from .runner import environment, orca_teacher, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "evaluate", "collect", "train"))
    parser.add_argument("--config")
    parser.add_argument("--backbone", choices=("gru", "mamba", "kda", "gdn2"))
    parser.add_argument("--readout", choices=("full", "read", "static", "gate", "evidence", "revision"))
    parser.add_argument("--order", choices=("scene", "actor"))
    parser.add_argument("--weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--people", type=int, default=5)
    parser.add_argument("--geometry", choices=("circle", "square"), default="circle")
    parser.add_argument("--cases", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--il-episodes", type=int, default=5)
    parser.add_argument("--rl-episodes", type=int, default=0)
    parser.add_argument("--output", default="outputs/result.json")
    args = parser.parse_args()
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config or Path(__file__).with_name("default.ini")):
        parser.error("Config not found")
    if args.weights and args.config is None:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        if "config" in checkpoint:
            cfg.read_dict(checkpoint["config"])
            if "order" in checkpoint["config"].get("model", {}):
                cfg.set("model", "feature_contract", checkpoint["config"]["model"].get("feature_contract", "legacy"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    backbone = args.backbone or cfg.get("model", "backbone")
    cfg.set("model", "backbone", backbone)
    order = args.order or cfg.get("model", "order", fallback=None)
    if backbone in ("kda", "gdn2") or args.readout:
        cfg.set("model", "architecture", "memory")
    if args.readout:
        cfg.set("model", "readout", args.readout)
    if order or cfg.get("model", "architecture", fallback=None) == "memory":
        cfg.set("model", "representation", "aligned")
        if order:
            cfg.set("model", "order", order)
    model = build_model(cfg)
    if args.weights:
        load_weights(model, args.weights, args.device)
    elif args.command == "evaluate":
        parser.error("Evaluation requires trained weights; use smoke for an untrained interface check")
    policy = ValuePolicy(model, cfg, args.device)
    env = environment(cfg, policy, args.geometry, args.people)
    if args.command == "train":
        from .training import train
        train(env, policy, cfg, args.output, args.seed, args.il_episodes, args.rl_episodes)
        return
    teacher = orca_teacher(cfg) if args.command == "collect" else None
    started = time.perf_counter()
    episodes = [run_episode(env, policy, case, teacher=teacher) for case in args.cases]
    elapsed = time.perf_counter() - started
    summary = {"command": args.command, "backbone": backbone, "trained_weights": bool(args.weights),
               "parameters": sum(p.numel() for p in model.parameters()), "episodes": len(episodes),
               "terminals": [row["terminal"] for row in episodes], "elapsed_seconds": elapsed,
               "scope": "untrained interface check" if args.command == "smoke" else args.command}
    if args.command == "evaluate":
        summary.update(success_rate=np.mean([row["terminal"] == "reach_goal" for row in episodes]),
                       collision_rate=np.mean([row["terminal"] == "collision" for row in episodes]),
                       timeout_rate=np.mean([row["terminal"] == "timeout" for row in episodes]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "episodes": episodes},
                                 default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
