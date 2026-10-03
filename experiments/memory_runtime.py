"""Synthetic shape-level runtime checks, never navigation performance evidence."""

import argparse
import json
import time

import numpy as np
import torch

from shixu.model import OcclusionValueModel


def benchmark(people, device, repeats):
    torch.manual_seed(47)
    model = OcclusionValueModel("kda", 128, 2).to(device).eval()
    prefix = torch.randn(1, 23, people + 1, 13, device=device)
    prefix[:, :, 1:, 10] = 1
    prefix[:, :, 1:, 12] = 1
    queries = prefix[:, -1:].expand(-1, 80, -1, -1).clone()
    queries[:, :, 0, 2:4] = torch.randn(1, 80, 2, device=device)
    times = []
    with torch.inference_mode():
        memory = model.encode_history(prefix)
        for _ in range(5):
            model.read_history(memory, queries)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            started = time.perf_counter()
            result = model.read_history(memory, queries)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
    return {"people": people, "median_read_ms": float(np.median(times)),
            "p95_read_ms": float(np.quantile(times, .95)), "values": result.cpu().tolist(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.startswith("cuda") else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps({"torch": torch.__version__, "device": args.device,
                      "rows": [benchmark(n, args.device, args.repeats) for n in (5, 20)],
                      "scope": "Identically seeded synthetic 80-candidate read shapes. "
                               "No training, inference quality or navigation conclusion."}))


if __name__ == "__main__":
    main()
