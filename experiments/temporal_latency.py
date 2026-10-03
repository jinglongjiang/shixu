"""Same-device end-to-end scoring latency, including legal candidate encoding."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from shixu.model import OrderedValueModel, load_weights
from shixu.observations import TrackKeys
from shixu.policy import ValuePolicy
from shixu.runner import environment
from experiments.temporal_order import configuration, PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol, root = json.loads(PROTOCOL.read_text()), Path(args.root)
    result = {"device": args.device, "torch": torch.__version__, "repetitions": 100, "warmup": 20,
              "scope": "Complete 80-candidate score call, not simulator or execution smoothing"}
    if args.device.startswith("cuda"):
        result["hardware"] = torch.cuda.get_device_name(torch.device(args.device))
    for order in ("scene", "actor"):
        contract = json.loads((root / str(args.seed) / order / "result.json").read_text()).get("feature_contract", "legacy")
        cfg = configuration(protocol, order, contract)
        policy = ValuePolicy(OrderedValueModel(order, protocol["width"], protocol["layers"], contract), cfg, args.device)
        load_weights(policy.model, root / str(args.seed) / order / "model.pt", args.device)
        env = environment(cfg, policy)
        env.reset(options={"test_case": protocol["evaluation_cases"][0]})
        state = TrackKeys().observe(env)
        policy.history.extend([policy.encode(state)] * protocol["history"])
        samples = []
        for repetition in range(120):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(policy.device)
            started = time.perf_counter()
            policy.score(state)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(policy.device)
            if repetition >= 20:
                samples.append(1000 * (time.perf_counter() - started))
        result[order] = {"median_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, .95)),
                         "samples_ms": samples, "parameters": sum(p.numel() for p in policy.model.parameters())}
    (root / "latency.json").write_text(json.dumps(result, indent=2))
    display = {key: value for key, value in result.items() if key not in ("scene", "actor")}
    display.update({order: {key: value for key, value in result[order].items() if key != "samples_ms"}
                    for order in ("scene", "actor")})
    print(json.dumps(display), flush=True)


if __name__ == "__main__":
    main()
