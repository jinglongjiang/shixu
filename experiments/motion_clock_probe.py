"""Legal observation-clock exposure and frozen trained-memory decay checks."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from shixu.features import stack_histories, window
from shixu.model import build_model, load_weights
from shixu.motion import observation_intervals


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None,
            "p99": float(np.percentile(values, 99)) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None}


def probe(data_path, weights=None, device="cpu"):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    intervals, returning, missing = [], [], []
    writes = 0
    for episode in data["episodes"]:
        tokens = torch.as_tensor(window(episode["tokens"], len(episode["tokens"]), "zero"))[None]
        active = tokens[:, :, 1:, 12] > 0
        measured = (tokens[:, :, 1:, 10] > 0) & active
        gaps, _ = observation_intervals(measured)
        values = gaps[measured].numpy()
        intervals.extend(values.tolist())
        returning.extend(values[values > 1].tolist())
        missing.extend(tokens[:, :, 1:, 9][active & ~measured].numpy().tolist())
        writes += int(measured.sum())
    result = {"demonstration_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
              "episodes": len(data["episodes"]), "measured_writes": writes,
              "measurement_interval_ticks": distribution(intervals), "return_interval_ticks": distribution(returning),
              "retained_hidden_age_seconds": distribution(missing),
              "fraction_of_writes_after_gap": len(returning) / max(writes, 1),
              "scope": "All natural legal ORCA demonstrations, without strong-event selection. Clock exposure is not action value."}
    if weights:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        cfg = configparser.ConfigParser()
        cfg.read_dict(checkpoint["config"])
        model = build_model(cfg).to(device).eval()
        load_weights(model, weights, device)
        if cfg.get("model", "architecture") != "motion" or cfg.get("model", "backbone") != "kda":
            raise ValueError("Decay diagnostics require the physical KDA model")
        windows = [window(ep["tokens"][:tick + 1], 24, "zero") for ep in data["episodes"]
                   for tick in range(0, len(ep["tokens"]), 8)]
        rates, real_factors, gap_factors = [], [], []
        with torch.inference_mode():
            for start in range(0, len(windows), 32):
                tokens = torch.as_tensor(stack_histories(windows[start:start + 32]), device=device)
                prefix = tokens[:, :-1]
                active = prefix[:, :, 1:, 12] > 0
                measured = (prefix[:, :, 1:, 10] > 0) & active
                gaps, _ = observation_intervals(measured)
                features = model.physical_features(prefix, gaps)
                cell = model.temporal_encoder.cells[0]
                _, _, _, _, log_decay, _ = cell.parameters_for(features, torch.ones_like(gaps))
                rates.extend(log_decay[measured].exp().mean((-2, -1)).cpu().tolist())
                scaled = (log_decay * gaps[..., None, None]).exp().mean((-2, -1))
                real_factors.extend(scaled[measured].cpu().tolist())
                gap_factors.extend(scaled[measured & (gaps > 1)].cpu().tolist())
        result.update(checkpoint_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                      checkpoint_phase=checkpoint.get("phase", "final"),
                      uniform_window_count=len(windows), unit_tick_retention=distribution(rates),
                      hypothetical_elapsed_retention=distribution(real_factors),
                      hypothetical_return_gap_retention=distribution(gap_factors),
                      trained_clock=model.clock,
                      interpretation="First-layer channel-decay factors on uniformly sampled legal windows. "
                                     "Elapsed factors are hypothetical for observation-clock models; this is not a runtime intervention or navigation result.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = probe(args.data, args.weights, args.device)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
