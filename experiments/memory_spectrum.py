"""Inspect associative rank and read amplitude in frozen, legally observed histories."""

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from shixu.features import stack_histories, window
from shixu.model import build_model, load_weights


def inspect(weights, data_path):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    model = build_model(cfg).eval()
    load_weights(model, weights, "cpu")
    if model.substrate != "kda":
        raise ValueError("Spectrum inspection requires KDA")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    histories = [window(ep["tokens"][:tick + 1], 24, "zero") for ep in data["episodes"]
                 for tick in range(0, len(ep["tokens"]), 8)]
    measurements = {key: [] for key in ("first_rank1_energy", "last_rank1_energy",
                                        "raw_read_norm", "normalized_read_norm", "projection_amplification")}
    with torch.inference_mode():
        for start in range(0, len(histories), 64):
            tokens = torch.as_tensor(stack_histories(histories[start:start + 64]))
            states, seen, _ = model.encode_history(tokens[:, :-1])
            own, active, _ = model._features(tokens[:, -1:])
            query = own
            if model.interaction_order == "write":
                query = model._attend(own, active)
            elif model.interaction_order == "residual":
                query = own + model._attend(own, active)
            if model.local_address:
                query = own
            usable = (active[:, 0] & seen).reshape(-1)
            for name, state in (("first_rank1_energy", states[0]), ("last_rank1_energy", states[-1])):
                singular = torch.linalg.svdvals(state[usable])
                energy = singular.square()
                ratio = energy[..., 0] / energy.sum(-1).clamp_min(1e-12)
                measurements[name].extend(ratio.flatten().tolist())
            cell = model.temporal_encoder.cells[0]
            q = F.normalize(cell._heads(cell.q(cell.input_norm(query.reshape(-1, query.shape[-1])))), dim=-1)
            raw = ((q[..., None] * states[0]).sum(-2) * cell.head_width ** -.5).flatten(-2)
            normalized = cell.output_norm(raw)
            measurements["raw_read_norm"].extend(raw.norm(dim=-1)[usable].tolist())
            measurements["normalized_read_norm"].extend(normalized.norm(dim=-1)[usable].tolist())
            amplification = cell.output(normalized).norm(dim=-1) / cell.output(raw).norm(dim=-1).clamp_min(1e-8)
            measurements["projection_amplification"].extend(amplification[usable].tolist())
    return {"checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            "phase": checkpoint.get("phase", "final"), "states": len(histories),
            "measurements": {key: {"count": len(rows), "mean": float(np.mean(rows)),
                                    "p10_p50_p90": np.quantile(rows, [.1, .5, .9]).tolist()}
                             for key, rows in measurements.items()},
            "scope": "Uniform every-eighth-frame successful-demonstration histories. Matrices belong to active "
                     "previously measured actors; all inputs are legal. Rank-1 energy is per head. Read values "
                     "are from the first cell. Amplitude is not calibrated confidence; neither normalization "
                     "amplification nor low rank alone proves a bad action or useful alternative architecture."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = inspect(args.weights, args.data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
