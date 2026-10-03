"""Frozen-protocol processing-order training, evaluation and paired summaries."""

import argparse
import configparser
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from shixu.model import OrderedValueModel, load_weights
from shixu.policy import ValuePolicy
from shixu.runner import environment, run_episode
from shixu.training import train


PROTOCOL = Path(__file__).with_name("temporal_protocol.json")


def configuration(protocol, order):
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / "shixu/default.ini")
    for section, key, value in (("model", "representation", "aligned"), ("model", "order", order),
                                ("model", "width", protocol["width"]), ("model", "layers", protocol["layers"]),
                                ("buffer", "seq_len", protocol["history"]), ("train", "il_epochs", protocol["il_epochs"]),
                                ("train", "updates_per_ep", protocol["rl_updates_per_episode"]),
                                ("train", "batch_size", protocol["batch_size"]),
                                ("train", "il_batch_size", protocol["batch_size"])):
        cfg.set(section, key, str(value))
    return cfg


def evaluate(policy, cfg, protocol):
    records = []
    policy.model.eval()
    for geometry in protocol["geometries"]:
        for people in protocol["people"]:
            env = environment(cfg, policy, geometry, people)
            for case in protocol["evaluation_cases"]:
                episode = run_episode(env, policy, case)
                records.append({key: episode[key] for key in
                                ("case", "terminal", "navigation_time", "path", "minimum_clearance")})
                records[-1].update(geometry=geometry, people=people,
                                   return_=float(sum(episode["rewards"])),
                                   moving_steps=len(episode["actions"]))
    return records


def summarize(records):
    successful = [row for row in records if row["terminal"] == "reach_goal"]
    return {"episodes": len(records), "sr": len(successful) / len(records),
            "cr": np.mean([row["terminal"] == "collision" for row in records]),
            "tr": np.mean([row["terminal"] == "timeout" for row in records]),
            "success_time": float(np.mean([row["navigation_time"] for row in successful])) if successful else None,
            "success_path": float(np.mean([row["path"] for row in successful])) if successful else None,
            "minimum_clearance": float(np.mean([row["minimum_clearance"] for row in records]))}


def paired_summary(root, protocol):
    seeds = protocol["seeds"]
    rows = []
    for seed in seeds:
        arms = {order: json.loads((root / str(seed) / order / "result.json").read_text()) for order in ("scene", "actor")}
        row = {"seed": seed, "scene": arms["scene"]["summary"], "actor": arms["actor"]["summary"]}
        row["sr_delta_pp"] = 100 * (row["actor"]["sr"] - row["scene"]["sr"])
        row["cr_delta_pp"] = 100 * (row["actor"]["cr"] - row["scene"]["cr"])
        rows.append(row)
    differences = np.array([row["sr_delta_pp"] for row in rows])
    cells = []
    for people in protocol["people"]:
        for geometry in protocol["geometries"]:
            arms = {}
            for order in ("scene", "actor"):
                selected = []
                for seed in seeds:
                    result = json.loads((root / str(seed) / order / "result.json").read_text())
                    selected.extend(row for row in result["episodes"] if row["people"] == people and row["geometry"] == geometry)
                arms[order] = summarize(selected)
            cells.append({"people": people, "geometry": geometry, **arms})
    floor = np.mean([cell["scene"]["sr"] for cell in cells if cell["people"] == 5])
    cr_delta = float(np.mean([row["cr_delta_pp"] for row in rows]))
    times = [(row["actor"]["success_time"] / row["scene"]["success_time"] - 1)
             for row in rows if row["actor"]["success_time"] and row["scene"]["success_time"]]
    healthy = floor >= protocol["health_floor_five_person_sr"]
    positive = (healthy and differences.mean() >= protocol["meaningful_success_gain_pp"]
                and sum(differences > 0) >= protocol["direction_agreement_required"]
                and cr_delta <= protocol["maximum_collision_increase_pp"]
                and bool(times) and np.mean(times) <= protocol["maximum_success_time_increase_fraction"])
    result = {"protocol": protocol, "seeds": rows, "cells": cells,
              "paired_sr_gain_pp": float(differences.mean()), "paired_cr_change_pp": cr_delta,
              "nominal_sr_95ci_pp": (differences.mean() + np.array([-1, 1]) * 3.182446 * differences.std(ddof=1) / 2).tolist(),
              "same_direction_seeds": int(sum(differences > 0)), "parent_five_person_sr": float(floor),
              "verdict": "PROTOTYPE_POSITIVE" if positive else "PARENT_UNHEALTHY" if not healthy else "NO_STABLE_GAIN",
              "scope": "Actor/scene processing-order evidence only, not a novel selective-revision method"}
    (root / "paired_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("paired_sr_gain_pp", "paired_cr_change_pp", "same_direction_seeds", "parent_five_person_sr", "verdict")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "summarize"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--order", choices=("scene", "actor", "pair"))
    parser.add_argument("--data")
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    root = Path(args.root)
    if args.mode == "summarize":
        paired_summary(root, protocol)
        return
    if args.order is None or args.data is None:
        parser.error("Run requires order and fixed demonstration data")
    if args.order == "pair":
        for order in ("scene", "actor"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "run", "--seed", str(args.seed),
                            "--order", order, "--data", args.data, "--root", str(root),
                            "--device", args.device], check=True)
        return
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg = configuration(protocol, args.order)
    model = OrderedValueModel(args.order, protocol["width"], protocol["layers"])
    policy = ValuePolicy(model, cfg, args.device)
    env = environment(cfg, policy, protocol["training_geometry"], protocol["training_people"])
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    if data["protocol"] != protocol:
        raise ValueError("IL data and training protocol differ")
    output = root / str(args.seed) / args.order
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (output / "learning.jsonl").open("w") as log:
        def report(row):
            row["elapsed_seconds"] = time.perf_counter() - started
            log.write(json.dumps(row) + "\n")
            log.flush()
            if row["phase"] == "il" and row["epoch"] % 10 == 0 or row["phase"] == "rl" and row["episode"] % 100 == 0:
                print(args.seed, args.order, row, flush=True)
        train(env, policy, cfg, output / "model.pt", args.seed, protocol["il_episodes"], protocol["rl_episodes"],
              demonstrations=data["episodes"], rl_case_start=protocol["rl_case_start"], report=report)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    records = evaluate(policy, cfg, protocol)
    result = {"protocol": protocol, "seed": args.seed, "order": args.order,
              "summary": summarize(records), "episodes": records,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "training_seconds": training_seconds, "evaluation_seconds": time.perf_counter() - started,
              "torch": torch.__version__, "device": str(policy.device), "host": platform.node()}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("FINISHED", args.seed, args.order, result["summary"], flush=True)


if __name__ == "__main__":
    main()
