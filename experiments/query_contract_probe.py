"""Observed-state versus prospective-query inputs on legal expert transitions.

Only final-IL weights are eligible: the labels are expert Monte Carlo returns,
not the value of a different learned continuation policy.
"""

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState
from shixu.features import encode_tracks, stack_histories, window
from shixu.model import build_model, load_weights
from shixu.observations import ObservedTracks
from shixu.policy import successor
from shixu.replay import returns


def observed_state(row, track_count):
    return ObservedTracks(FullState(*row["robot"]),
                          [ObservableState(*human["state"]) for human in row["humans"]],
                          tuple(human["track_id"] for human in row["humans"]),
                          tuple(row["observed"]), tuple(row["ages"]), track_count)


def transitions(data, length, dt):
    records, actual, predicted, fresh_age, targets = [], [], [], [], []
    skipped = 0
    for episode in data["episodes"]:
        labels = returns(episode["rewards"], .99)
        frames = episode["tokens"]
        for tick in range(0, len(frames) - 1, 4):
            root, following = frames[tick], frames[tick + 1]
            if root.shape != following.shape or not np.array_equal(root[1:, 12], following[1:, 12]):
                skipped += 1
                continue
            state = observed_state(episode["observations"][tick], root.shape[0] - 1)
            np.testing.assert_allclose(encode_tracks(state), root, rtol=1e-6, atol=1e-6)
            query = encode_tracks(successor(state, ActionXY(*episode["actions"][tick]), dt))
            prefix = window(frames[:tick + 1], length, "zero")[1:]
            shadow = query.copy()
            measured = root[1:, 10] > 0
            shadow[1:, 9][measured] = 0
            actual.append(np.concatenate((prefix, following[None]), axis=0))
            predicted.append(np.concatenate((prefix, query[None]), axis=0))
            fresh_age.append(np.concatenate((prefix, shadow[None]), axis=0))
            targets.append(float(labels[tick + 1]))
            records.append({"case": episode["case"], "tick": tick,
                            "hidden": sum(not flag for flag in state.observed),
                            "query_geometry_change": float(np.abs(query[:, :9] - following[:, :9]).mean())})
    return records, (actual, predicted, fresh_age), np.asarray(targets), skipped


def probe(data_path, weights, device):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != "il":
        raise ValueError("Expert-return regression diagnostics require final-IL weights")
    cfg = configparser.ConfigParser()
    cfg.read_dict(checkpoint["config"])
    if cfg.get("model", "representation") != "tracks" or cfg.getfloat("train", "gamma") != .99:
        raise ValueError("This diagnostic requires the frozen legal-track/gamma contract")
    model = build_model(cfg).to(device).eval()
    load_weights(model, weights, device)
    records, modes, targets, skipped = transitions(data, cfg.getint("buffer", "seq_len"),
                                                   cfg.getfloat("env", "time_step"))
    outputs = []
    with torch.inference_mode():
        for samples in modes:
            values = []
            for start in range(0, len(samples), 32):
                inputs = torch.as_tensor(stack_histories(samples[start:start + 32]), device=device)
                values.extend(model(inputs).cpu().tolist())
            outputs.append(np.asarray(values))
    groups = {"all": np.ones(len(records), dtype=bool),
              "retained_hidden": np.asarray([row["hidden"] > 0 for row in records]),
              "root_all_visible": np.asarray([row["hidden"] == 0 for row in records])}
    summaries = {}
    for name, selected in groups.items():
        summaries[name] = {"transitions": int(selected.sum())}
        for mode, values in zip(("observed_next", "cv_next", "cv_fresh_age_shadow"), outputs):
            error = values[selected] - targets[selected]
            summaries[name][mode] = {"mse": float(np.mean(error ** 2)) if len(error) else None,
                                    "bias": float(np.mean(error)) if len(error) else None}
    for row, target, a, b, c in zip(records, targets, *outputs):
        row.update(target=float(target), observed_next=float(a), cv_next=float(b), cv_fresh_age_shadow=float(c))
    return {"checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "seed": checkpoint["seed"], "config": checkpoint["config"],
            "episodes": len(data["episodes"]), "support_change_skips": skipped,
            "summaries": summaries, "records": records,
            "scope": "Uniform every-fourth transition of pre-fixed held-out legal-observation ORCA episodes. "
                     "Cases where active support changes are excluded before model evaluation. Same legal "
                     "prefix and expert continuation target; only observed-next versus planner-CV query differs. "
                     "The fresh-age shadow fictitiously resets currently measured actors' query age, without "
                     "changing predicted geometry or writing memory. It is a diagnostic, not a deployable fix. "
                     "Regression error does not establish action quality or closed-loop improvement."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = probe(args.data, args.weights, args.device)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "config")}, indent=2))


if __name__ == "__main__":
    main()
