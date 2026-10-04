"""Fixed-cohort forecast validation and frozen-weight consumption checks.

Replay the first four development cases per cell from an existing parent result.
No navigation retraining, hidden labels, new scenes or outcome-selected states.
"""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState
from experiments.occlusion import source_hash
from shixu.features import stack_histories, window
from shixu.forecast import ForecastReplay
from shixu.model import build_model, load_weights
from shixu.observations import ObservedTracks, OccludedTracks
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


def state_from_tokens(frame):
    keys = np.flatnonzero(frame[1:, 12] > 0)
    robot = FullState(*frame[0, :9])
    humans = [ObservableState(*(frame[1 + key, :2] + frame[0, :2]),
                               *frame[1 + key, 3:5], frame[1 + key, 6]) for key in keys]
    return ObservedTracks(robot, humans, tuple(int(k) for k in keys),
                          tuple(bool(frame[1 + k, 10]) for k in keys),
                          tuple(float(frame[1 + k, 9]) for k in keys), len(frame) - 1)


def latency(weights, cohort, destination, device):
    if destination.exists():
        raise RuntimeError("Do not overwrite a standardized timing replay")
    data = torch.load(cohort, map_location="cpu", weights_only=False)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    policy = ValuePolicy(build_model(cfg), cfg, device)
    load_weights(policy.model, weights, device)
    rows = examples(data)[::10]
    prepared = [(state_from_tokens(row["history"][-1]), row["history"][:-1]) for row in rows]
    for i in range(20):
        state, history = prepared[i * len(prepared) // 20]
        policy.history.clear()
        policy.history.extend(history)
        policy.score(state)
    timings = []
    for _ in range(3):
        for state, history in prepared:
            policy.history.clear()
            policy.history.extend(history)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            started = time.perf_counter()
            policy.score(state)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            timings.append(1000 * (time.perf_counter() - started))
    result = dict(seed=checkpoint["seed"], checkpoint_sha256=digest(weights), cohort_sha256=digest(cohort),
                  device=device, torch=torch.__version__, host=platform.node(),
                  hardware=torch.cuda.get_device_name(policy.device) if policy.device.type == "cuda" else "CPU",
                  states=len(prepared), repetitions=3, median_ms=float(np.median(timings)),
                  p95_ms=float(np.quantile(timings, .95)), timings_ms=timings,
                  scope="Complete native 80-action score/filter/transfers, same legal workload, every tenth root, "
                        "20 warmups, three repetitions. Run only when navigation training is idle. "
                        "This timing replay is not a navigation outcome experiment.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print("TIMED", checkpoint["seed"], cfg.get("model", "backbone"), result["median_ms"], flush=True)


def gradients(weights, demonstrations, destination, device):
    if destination.exists():
        raise RuntimeError("Do not overwrite an objective-gradient diagnostic")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    data = torch.load(demonstrations, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    model = build_model(cfg).to(device).eval()
    load_weights(model, weights, device)
    if model.kind == "cv":
        raise ValueError("Fixed CV has no forecast parameters")
    replay = ForecastReplay(cfg.getint("buffer", "capacity"), cfg.getint("buffer", "seq_len"),
                            cfg.getfloat("train", "gamma"), "zero")
    for episode in data["episodes"]:
        replay.add(episode)
    rng, records = np.random.default_rng(20261004), []
    for _ in range(8):
        history, returns, future, valid = replay.sample_with_forecasts(64, device, rng)
        values, predicted = model.value_and_forecast(history)
        value_loss = (values - returns).square().mean()
        anchor = model._physical(history)[1][:, -1]
        error = ((predicted[:, :, 1:] - anchor[:, :, None] - future) / model.times[None, None, 1:, None]).square().sum(-1)
        forecast_loss = (error * valid).sum() / (2 * valid.sum().clamp_min(1))
        parameters = tuple(model.decoder.parameters())
        a = torch.cat([g.flatten() for g in torch.autograd.grad(value_loss, parameters, retain_graph=True)])
        b = torch.cat([g.flatten() for g in torch.autograd.grad(model.prediction_weight * forecast_loss, parameters)])
        records.append(dict(value_loss=float(value_loss.detach()), forecast_loss=float(forecast_loss.detach()),
                            value_gradient_norm=float(a.norm()), weighted_forecast_gradient_norm=float(b.norm()),
                            cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0))))
    result = dict(seed=checkpoint["seed"], kind=model.kind, phase=checkpoint.get("phase", "final"),
                  checkpoint_sha256=digest(weights), demonstrations_sha256=digest(demonstrations), batches=records,
                  scope="Eight identical64-sample batches from shared successful IL data. Local decoder-gradient "
                        "conflict, not reconstruction of past AdamW updates, not proof of online-RL causation, "
                        "and not a navigation improvement experiment. No parameters are updated.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print("GRADIENTS", checkpoint["seed"], model.kind, result["phase"], flush=True)


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
    parser.add_argument("mode", choices=("collect", "compare", "batch-compare", "latency", "batch-latency", "gradients"))
    parser.add_argument("--root", default="outputs/forecast_control_a")
    parser.add_argument("--cohort", default="outputs/forecast_control_a/shared_diagnostic.pt")
    parser.add_argument("--weights")
    parser.add_argument("--destination")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("il", "model"), default="model")
    parser.add_argument("--data", default="outputs/forecast_control_a/demonstrations.pt")
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
    elif args.mode == "batch-latency":
        root = Path(args.root)
        weights = [(path, path.with_name("standardized_timing.json")) for path in sorted(root.glob("*/*/model.pt"))]
        for seed in (419, 443, 467, 491):
            weights.append((root.parent / "occlusion_v8" / str(seed) / "gru_context/model.pt",
                            root / str(seed) / "parent_gru/standardized_timing.json"))
        for path, destination in weights:
            if not destination.exists():
                latency(path, Path(args.cohort), destination, args.device)
    else:
        if not args.weights or not args.destination:
            parser.error("Checkpoint and destination required")
        function = {"latency": latency, "gradients": gradients}.get(args.mode, compare)
        data = args.data if args.mode == "gradients" else args.cohort
        function(Path(args.weights), Path(data), Path(args.destination), args.device)


if __name__ == "__main__":
    main()
