"""One immutable ORCA demonstration set shared by all temporal-order arms."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import torch

from shixu.model import ValueModel
from shixu.policy import ValuePolicy
from shixu.runner import environment, orca_teacher, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(Path(__file__).with_name("temporal_protocol.json").read_text())
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
    cfg.set("model", "representation", "aligned")
    policy = ValuePolicy(ValueModel(width=32, layers=1), cfg)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    teacher = orca_teacher(cfg)
    episodes, attempts = [], 0
    while len(episodes) < protocol["il_episodes"]:
        case = protocol["il_case_start"] + attempts
        record = run_episode(env, policy, case, teacher=teacher)
        attempts += 1
        if record["terminal"] == "reach_goal":
            episodes.append({key: record[key] for key in ("case", "tokens", "rewards", "terminal")})
        if attempts >= cfg.getint("imitation_learning", "max_il_prefill"):
            raise RuntimeError("Insufficient successful teacher demonstrations")
        if len(episodes) and len(episodes) % 32 == 0:
            print("COLLECT", len(episodes), "attempts", attempts, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol": protocol, "episodes": episodes, "attempts": attempts,
                "information": "Observed robot/human features and association masks; no human goals or future input"}, output)
    print(json.dumps({"episodes": len(episodes), "attempts": attempts,
                      "frames": sum(len(row["tokens"]) for row in episodes),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
