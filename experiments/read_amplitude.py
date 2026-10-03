"""Frozen read-normalization intervention; not a retrained method comparison."""

import argparse
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments import occlusion


def evaluate(weights, protocol, device, seed, mode):
    original = occlusion.build_model

    def build(config):
        model = original(config)
        if model.substrate != "kda":
            raise ValueError("Read-amplitude diagnostic requires KDA")
        if mode == "raw":
            def read(cell, state, query):
                q = F.normalize(cell._heads(cell.q(cell.input_norm(query))), dim=-1)
                values = torch.einsum("...hk,...hkv->...hv", q, state) * cell.head_width ** -.5
                return cell.output(values.flatten(-2))
            for cell in model.temporal_encoder.cells:
                cell.read = MethodType(read, cell)
        return model

    with patch.object(occlusion, "build_model", build):
        result = occlusion.evaluate_weights(weights, protocol, device, protocol["development_cases"], seed)
    result.update(mode=mode, scope="Identical frozen weights, observation contract, cases and controller. "
                  "Only retrieval bypasses output normalization; writes remain unchanged. Distribution-shifting "
                  "intervention, not evidence that a trained replacement wins or that amplitude is confidence.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--mode", choices=("raw", "parent"), default="raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(args.weights, json.loads(Path(args.protocol).read_text()), args.device, args.seed, args.mode)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
