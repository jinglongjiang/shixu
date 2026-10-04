"""Verify full-budget artifacts before computing paired navigation contrasts."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from experiments.occlusion import configuration, evaluate_weights, source_hash
from experiments.temporal_order import summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_gpu(root, prior_root, protocol_path, seeds, device):
    protocol = json.loads(protocol_path.read_text())
    for seed in seeds:
        weights = prior_root / str(seed) / "gru_context/model.pt"
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if (checkpoint["seed"] != seed or checkpoint["il_episodes"] != protocol["il_episodes"]
                or checkpoint["rl_episodes"] != protocol["rl_episodes"]):
            raise ValueError("Parent checkpoint budget differs")
        destination = root / str(seed) / "parent_gru/result_gpu.json"
        if destination.exists():
            raise RuntimeError("Do not overwrite a completed GPU parent reference")
        result = evaluate_weights(weights, protocol, device, protocol["development_cases"], seed)
        result.update(protocol=protocol, seed=seed, arm="parent_gru", evaluation_host=platform.node(),
                      evaluation_device=device, evaluation_torch=torch.__version__, reference=str(weights),
                      scope="Frozen V8 final weights evaluated on the paired training GPU/software; no new training")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2))
        print("GPU_PARENT", seed, result["summary"], flush=True)


def aggregate(root, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    expected = {(n, g, c) for n in protocol["people"] for g in protocol["geometries"]
                for c in protocol["development_cases"]}
    data_path = root / "demonstrations.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if data["protocol"] != protocol or data["source_sha256"] != source_hash():
        raise ValueError("Source or shared demonstration protocol changed")
    data_sha = sha(data_path)
    model_records, audits = {}, []
    for arm in protocol["arms"] + ["parent_gru"]:
        results = []
        for seed in protocol["seeds"]:
            folder = root / str(seed) / arm
            result_path = folder / "result.json"
            if arm == "parent_gru":
                result_path = folder / "result_gpu.json"
                if not result_path.exists():
                    raise ValueError("Matched-device parent evaluation must finish before final comparison")
            result = json.loads(result_path.read_text())
            if result["protocol"] != protocol or result["seed"] != seed:
                raise ValueError("Incorrect seed/protocol in result")
            keys = [(e["people"], e["geometry"], e["case"]) for e in result["episodes"]]
            if len(keys) != len(expected) or set(keys) != expected:
                raise ValueError("Missing, repeated or mismatched evaluation cases")
            if arm != "parent_gru":
                weights = folder / "model.pt"
                checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
                logs = [json.loads(line) for line in (folder / "learning.jsonl").read_text().splitlines()]
                il = [r["epoch"] for r in logs if r["phase"] == "il"]
                rl = [r["episode"] for r in logs if r["phase"] == "rl"]
                cfg = configuration(protocol, arm)
                if (il != list(range(1, protocol["il_epochs"] + 1))
                        or rl != list(range(1, protocol["rl_episodes"] + 1))
                        or checkpoint["config"] != {s: dict(cfg[s]) for s in cfg.sections()}
                        or checkpoint["seed"] != seed or checkpoint["rl_episodes"] != protocol["rl_episodes"]
                        or checkpoint["il_episodes"] != protocol["il_episodes"]
                        or result["checkpoint_sha256"] != sha(weights)
                        or result["source_sha256"] != source_hash()
                        or result["demonstration_sha256"] != data_sha
                        or any(not np.isfinite(r["loss"]) for r in logs)):
                    raise ValueError("Incomplete or altered fixed-budget training artifact")
                audits.append(dict(seed=seed, arm=arm, il_epochs=len(il), rl_episodes=len(rl),
                                   checkpoint_sha256=result["checkpoint_sha256"], training_host=result["host"],
                                   torch=result["torch"], device=result["device"], parameters=result["parameters"]))
            else:
                prior = root.parent / "occlusion_v8" / str(seed) / "gru_context"
                archived = json.loads((prior / "result.json").read_text())
                checkpoint = torch.load(prior / "model.pt", map_location="cpu", weights_only=False)
                if (archived["demonstration_sha256"] != data["original_sha256"]
                        or checkpoint["seed"] != seed or checkpoint["il_episodes"] != protocol["il_episodes"]
                        or checkpoint["rl_episodes"] != protocol["rl_episodes"]
                        or result["checkpoint_sha256"] != sha(prior / "model.pt")):
                    raise ValueError("The frozen parent has different demonstrations/budget/weights")
                result.update(parameters=archived["parameters"], training_seconds=archived["training_seconds"])
            results.append(result)
        model_records[arm] = results
    for seed in protocol["seeds"]:
        runs = [r for r in audits if r["seed"] == seed]
        if len({(r["training_host"], r["torch"], r["device"]) for r in runs}) != 1:
            raise ValueError("Paired arms used different training hosts/software")
        parent = model_records["parent_gru"][protocol["seeds"].index(seed)]
        reference = runs[0]
        if ((parent["evaluation_host"], parent["evaluation_torch"], parent["evaluation_device"])
                != (reference["training_host"], reference["torch"], reference["device"])):
            raise ValueError("Parent and new arm numerical evaluation conditions differ")
    models = {}
    for arm, results in model_records.items():
        episodes = [e for r in results for e in r["episodes"]]
        primary = [e for e in episodes if e["people"] > 5]
        models[arm] = dict(overall=summarize(episodes), primary=summarize(primary),
                           primary_by_seed=[summarize([e for e in r["episodes"] if e["people"] > 5]) for r in results],
                           cells=[dict(people=n, geometry=g, **summarize([e for e in episodes if e["people"] == n
                                                                         and e["geometry"] == g]))
                                  for n in protocol["people"] for g in protocol["geometries"]],
                           parameters=results[0].get("parameters"),
                           training_seconds=[r.get("training_seconds") for r in results],
                           score_median_ms=[r["score_median_ms"] for r in results])
    contrasts = []
    for candidate in ("current", "gru", "kda"):
        for control in ("cv", "current", "gru", "parent_gru"):
            if control == candidate:
                continue
            a, b = models[candidate]["primary_by_seed"], models[control]["primary_by_seed"]
            changes = {key: [100 * (x[key] - y[key]) for x, y in zip(a, b)] for key in ("sr", "cr", "tr")}
            time_changes = [x["success_time"] / y["success_time"] - 1 for x, y in zip(a, b)
                            if x["success_time"] is not None and y["success_time"] is not None]
            time_change = float(np.mean(time_changes)) if len(time_changes) == len(a) else None
            gain = np.asarray(changes["sr"])
            positive = (gain.mean() >= protocol["meaningful_gain_pp"]
                        and sum(gain > 0) >= protocol["required_same_direction_seeds"]
                        and np.mean(changes["cr"]) <= protocol["maximum_collision_increase_pp"]
                        and np.mean(changes["tr"]) <= protocol["maximum_timeout_increase_pp"]
                        and time_change is not None and time_change <= protocol["maximum_success_time_increase_fraction"])
            contrasts.append(dict(candidate=candidate, control=control, seed_changes_pp=changes,
                                  mean_changes_pp={k: float(np.mean(v)) for k, v in changes.items()},
                                  nominal_sr_95ci_pp=(gain.mean() + np.array([-1, 1]) * 3.182446 * gain.std(ddof=1) / 2).tolist(),
                                  same_direction_seeds=int(sum(gain > 0)), success_time_change_fraction=time_change,
                                  development_positive=bool(positive)))
    kda = [c for c in contrasts if c["candidate"] == "kda"]
    status = ("DEVELOPMENT_POSITIVE_REQUIRES_FRESH_CONFIRMATION" if all(c["development_positive"] for c in kda)
              else "NO_VALIDATED_KDA_ADVANTAGE_THIS_VERSION")
    result = dict(protocol=protocol, protocol_sha256=sha(protocol_path), source_sha256=source_hash(),
                  demonstration_sha256=data_sha, audits=audits, models=models, contrasts=contrasts, status=status,
                  scope="Four paired training seeds are the replication units. Nominal intervals are exploratory, "
                        "not multiplicity-adjusted. Conditional successful-time averages do not replace timeout "
                        "rates. Failure to pass is not equivalence or a family-level rejection. Concurrent cross-host "
                        "timings are descriptive only. Parent evaluation must be checked for numerical device effects.")
    (root / "matched_results.json").write_text(json.dumps(result, indent=2))
    for name, row in models.items():
        print(name, "primary SR/CR/TO", [round(100 * row["primary"][k], 2) for k in ("sr", "cr", "tr")], flush=True)
    print(status, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/forecast_control_a")
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("forecast_control_protocol.json")))
    parser.add_argument("--parent-seeds", nargs="+", type=int)
    parser.add_argument("--prior-root", default="outputs/occlusion_v8")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.parent_seeds:
        parent_gpu(Path(args.root), Path(args.prior_root), Path(args.protocol), args.parent_seeds, args.device)
    else:
        aggregate(Path(args.root), Path(args.protocol))
