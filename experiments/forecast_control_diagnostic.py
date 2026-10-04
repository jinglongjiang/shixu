"""Fixed-cohort forecast validation and frozen-weight consumption checks.

Replay the first four development cases per cell from an existing parent result.
No navigation retraining, hidden labels, new scenes or outcome-selected states.
"""

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.occlusion import source_hash
from shixu.features import stack_histories, window
from shixu.model import build_model, load_weights
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root, destination):
    if destination.exists():
        raise RuntimeError("Do not overwrite the fixed diagnostic cohort")
    result_path = root / "419/parent_gru/result.json"
    result = json.loads(result_path.read_text())
    checkpoint_path = root.parent / "occlusion_v8/419/gru_context/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    policy = ValuePolicy(build_model(cfg), cfg)
    cases = result["protocol"]["development_cases"][:4]
    records = []
    for row in result["episodes"]:
        if row["case"] not in cases:
            continue
        env = environment(cfg, policy, row["geometry"], row["people"])
        env.reset(options={"test_case": row["case"]})
        tracks = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
        frames, roots, path, clearances = [], [], 0.0, []
        for tick, command in enumerate(row["actions"]):
            state = tracks.observe(env)
            frames.append(policy.encode(state))
            if tick % 4 == 0:
                queries, rewards, clearance = policy.aligned_candidates(state)
                roots.append(dict(tick=tick, queries=queries, rewards=rewards, clearance=clearance))
            old = np.array(env.robot.get_position())
            _, _, done, truncated, info = env.step(ActionXY(*command))
            path += float(np.linalg.norm(np.array(env.robot.get_position()) - old))
            clearances.append(info["dmin"])
            if (done or truncated) != (tick == len(row["actions"]) - 1):
                raise ValueError("Archived action replay has different termination")
        if (info["event"] != row["terminal"] or abs(env.global_time - row["navigation_time"]) > 1e-8
                or abs(path - row["path"]) > 1e-6
                or abs(min(clearances) - row["minimum_clearance"]) > 1e-6):
            raise ValueError("Archived action replay no longer matches the native episode")
        records.append(dict(people=row["people"], geometry=row["geometry"], case=row["case"],
                            terminal=row["terminal"], tokens=frames, roots=roots))
    if len(records) != 24:
        raise ValueError("The preselected six-cell/four-case cohort is incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(episodes=records, protocol=result["protocol"], source_sha256=source_hash(),
                    parent_result_sha256=digest(result_path), cases=cases,
                    scope="Replay archived parent commands; labels are later visible observations only; "
                          "first four development cases in each cell, every fourth frame, no outcome selection"), destination)
    print("COLLECTED", len(records), "episodes", sum(len(r["roots"]) for r in records), "roots", flush=True)


def examples(data):
    rows = []
    for episode_index, episode in enumerate(data["episodes"]):
        frames = episode["tokens"]
        for root in episode["roots"]:
            tick = root["tick"]
            current = frames[tick]
            count = current.shape[0] - 1
            future = np.zeros((count, 9, 2), np.float32)
            valid = np.zeros((count, 9), bool)
            for horizon in range(1, 10):
                if tick + horizon < len(frames):
                    later = frames[tick + horizon]
                    future[:, horizon - 1] = later[1:count + 1, :2] + later[0, :2]
                    valid[:, horizon - 1] = ((current[1:, 12] > 0) & (later[1:count + 1, 12] > 0)
                                            & (later[1:count + 1, 10] > 0))
            rows.append(dict(episode=episode_index, tick=tick,
                             history=window(frames[:tick + 1], data["protocol"]["history"], "zero"),
                             future=future, valid=valid, **{k: root[k] for k in ("queries", "rewards", "clearance")}))
    return rows


def filter_scores(scores, clearances, cfg):
    result = scores.clone()
    margin = cfg.getfloat("eval_protocol", "safety_margin")
    if margin > 0:
        result = result.masked_fill((clearances < margin) & (clearances >= margin).any(-1, keepdim=True), -1e9)
    risk = cfg.getfloat("eval_protocol", "risk_lambda")
    threshold = margin if margin > 0 else cfg.getfloat("reward", "discomfort_dist")
    if risk > 0:
        result -= risk * torch.clamp(threshold - clearances, min=0)
    result[:, 0] -= 1e-3
    return result


@torch.inference_mode()
def compare(weights, cohort, destination, device):
    if destination.exists():
        raise RuntimeError("Do not overwrite an existing checkpoint diagnostic")
    data = torch.load(cohort, map_location="cpu", weights_only=False)
    if data["source_sha256"] != source_hash():
        raise ValueError("Diagnostic scientific source changed")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    model = build_model(cfg).to(device).eval()
    load_weights(model, weights, device)
    rows, records = examples(data), []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        histories = torch.as_tensor(stack_histories([r["history"] for r in batch]), device=device)
        people = histories.shape[2] - 1
        queries, future, valid = [], [], []
        for row in batch:
            queries.append(np.pad(row["queries"], ((0, 0), (0, people + 1 - row["queries"].shape[1]), (0, 0))))
            future.append(np.pad(row["future"], ((0, people - row["future"].shape[0]), (0, 0), (0, 0))))
            valid.append(np.pad(row["valid"], ((0, people - row["valid"].shape[0]), (0, 0))))
        queries = torch.as_tensor(np.stack(queries), device=device)
        positions = model.forecast_positions(histories)
        humans, world = model._physical(histories)
        cv = world[:, -1, :, None] + humans[:, -1, :, None, 3:5] * model.times[None, None, :, None]
        values = model._value(histories, queries, positions, 1)
        cv_values = model._value(histories, queries, cv, 1)
        rewards = torch.as_tensor(np.stack([r["rewards"] for r in batch]), device=device)
        clearance = torch.as_tensor(np.stack([r["clearance"] for r in batch]), device=device)
        raw = rewards + cfg.getfloat("train", "gamma") * values
        cv_raw = rewards + cfg.getfloat("train", "gamma") * cv_values
        filtered, cv_filtered = filter_scores(raw, clearance, cfg), filter_scores(cv_raw, clearance, cfg)
        errors = (positions[:, :, 1:] - torch.as_tensor(np.stack(future), device=device)).norm(dim=-1).cpu().numpy()
        cv_errors = (cv[:, :, 1:] - torch.as_tensor(np.stack(future), device=device)).norm(dim=-1).cpu().numpy()
        for i, row in enumerate(batch):
            mask = valid[i]
            near = humans[i, -1, :, 2].cpu().numpy() < 2
            hidden = (humans[i, -1, :, 12] > 0).cpu().numpy() & ~(humans[i, -1, :, 10] > 0).cpu().numpy()
            subsets = {"all": mask, "near": mask & near[:, None], "currently_retained_hidden": mask & hidden[:, None]}
            accuracy = {name: dict(targets=int(m.sum()), error_sum=float(errors[i][m].sum()),
                                   cv_error_sum=float(cv_errors[i][m].sum())) for name, m in subsets.items()}
            for step in (4, 8):
                m = mask[:, step - 1]
                accuracy[str(step * .25) + "s"] = dict(targets=int(m.sum()), error_sum=float(errors[i, :, step - 1][m].sum()),
                                                       cv_error_sum=float(cv_errors[i, :, step - 1][m].sum()))
            records.append(dict(episode=row["episode"], tick=row["tick"], prediction=accuracy,
                                raw_action=int(raw[i].argmax()), cv_swap_raw_action=int(cv_raw[i].argmax()),
                                filtered_action=int(filtered[i].argmax()), cv_swap_filtered_action=int(cv_filtered[i].argmax()),
                                mean_abs_value_change=float((values[i] - cv_values[i]).abs().mean())))
    result = dict(checkpoint_sha256=digest(weights), cohort_sha256=digest(cohort), source_sha256=source_hash(),
                  seed=checkpoint["seed"], phase=checkpoint.get("phase", "final"), kind=model.kind, roots=records,
                  episodes=[{k: r[k] for k in ("people", "geometry", "case", "terminal")} for r in data["episodes"]],
                  scope="Same legal parent-state cohort for all models. CV swap is a frozen-weight intervention, "
                        "not a retrained baseline or proof of better navigation. Future accuracy covers later "
                        "visible actors only; hidden trajectories and unavailable terminal tails are not labelled.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print("DIAGNOSED", checkpoint["seed"], model.kind, result["phase"], len(records), "roots", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "compare", "batch-compare"))
    parser.add_argument("--root", default="outputs/forecast_control_a")
    parser.add_argument("--cohort", default="outputs/forecast_control_a/shared_diagnostic.pt")
    parser.add_argument("--weights")
    parser.add_argument("--destination")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("il", "model"), default="model")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "collect":
        collect(Path(args.root), Path(args.cohort))
    elif args.mode == "batch-compare":
        weights = sorted(Path(args.root).glob("*/*/" + args.phase + ".pt"))
        if not weights:
            raise ValueError("No completed checkpoint of the requested phase")
        for path in weights:
            destination = path.with_name("forecast_diagnostic_" + args.phase + ".json")
            if not destination.exists():
                compare(path, Path(args.cohort), destination, args.device)
    else:
        if not args.weights or not args.destination:
            parser.error("Checkpoint and destination required")
        compare(Path(args.weights), Path(args.cohort), Path(args.destination), args.device)


if __name__ == "__main__":
    main()
