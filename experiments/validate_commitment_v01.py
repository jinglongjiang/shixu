"""Independent post-run checks; keep the frozen experiment source unchanged."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import save_json
from experiments.online_action_commitment_v01 import OUT, read, summarize, verify
from shixu.commitment import low_progress
from shixu.policy import actions


def validate():
    verify()
    records = [read(p) for p in (OUT/"episodes").glob("*.json")]
    records.sort(key=lambda r: (r["seed"],r["people"],r["geometry"],r["case"],r["arm"]))
    saved = read(OUT/"summary.json")
    for key,value in summarize(records).items():
        if saved[key] != value:
            raise ValueError("Metadata and summary differ")
    compact = read(OUT/"episode-summary.json")["records"]
    if compact != [{k:v for k,v in r.items() if k not in ("decisions","proposals")} for r in records]:
        raise ValueError("Compact metadata differ")
    configs, groups, steps, parity = {}, defaultdict(dict), Counter(), Counter()
    releases, smooth_error, prefix_checks = Counter(), 0., 0
    for r in records:
        groups[r["seed"],r["people"],r["geometry"],r["case"]][r["arm"]] = r
        if digest(Path(r["trace"])) != r["trace_sha256"] or r["protocol_sha256"] != digest(OUT/"protocol.json"):
            raise ValueError("Frozen hash differs")
        if r["seed"] not in configs:
            _,configs[r["seed"]] = load(weights(r["seed"],"cv"),"cpu")
        cfg = configs[r["seed"]]
        with np.load(r["trace"]) as t:
            steps[r["arm"]] += len(t["actions"])
            expected = np.asarray(actions(cfg))[t["grid_action"]].copy()
            expected[1:] = .3*t["previous"][1:]+.7*expected[1:]
            error = float(np.max(np.abs(expected-t["actions"])))
            smooth_error = max(smooth_error,error)
            if error != 0 or np.any(np.diff(t["time"]) != .25):
                raise ValueError("Smoothing or clock differs")
            q = sum(cfg.getfloat("train","gamma")**i*v for i,v in enumerate(t["rewards"]))
            if abs(q-r["discounted_return"]) > 1e-12:
                raise ValueError("Original reward differs")
            active,remaining,armed = None,0,True
            for tick,d in enumerate(r["decisions"]):
                trigger = low_progress(t["goal_distance"][max(0,tick-8):tick+1])
                if d["tick"] != tick or d["time"] != t["time"][tick] or d["trigger"] != trigger or d["selected_grid"] != t["grid_action"][tick]:
                    raise ValueError("Log/causal trigger differs")
                if active is None and not trigger:
                    armed = True
                if d["release"] == "safety-blocked":
                    if active is None or d["held"]:
                        raise ValueError("Original release contract differs")
                    active,remaining = None,0
                if d["started"]:
                    if active is not None or not trigger or not armed or d["release"] == "safety-blocked":
                        raise ValueError("Start/rearm contract differs")
                    active,remaining,armed = d["selected_grid"],8,False
                if d["release"] == "all-unsafe":
                    if active is None or r["arm"] != "v01" or not d["no_margin_safe_candidate"] or d["held"] or d["proposal"] != d["selected_grid"]:
                        raise ValueError("New same-tick release differs")
                    active,remaining = None,0
                if d["held"] != (active is not None):
                    raise ValueError("Held state differs")
                if d["held"]:
                    if d["selected_grid"] != active or d["blocked"] or r["arm"] == "v01" and d["no_margin_safe_candidate"]:
                        raise ValueError("Held command differs")
                    remaining -= 1
                    if remaining == 0:
                        if d["release"] != "budget-completed":
                            raise ValueError("Budget failed to expire")
                        active = None
                    elif d["release"] is not None:
                        raise ValueError("Premature budget expiration")
                if d["release"]:
                    releases[r["arm"]+":"+d["release"]] += 1
    for arms in groups.values():
        first = next((i for i,d in enumerate(arms["v01"]["decisions"]) if d["release"] == "all-unsafe"),None)
        if first is not None:
            with np.load(arms["v0"]["trace"]) as a,np.load(arms["v01"]["trace"]) as b:
                np.testing.assert_array_equal(a["actions"][:first],b["actions"][:first])
            prefix_checks += 1
        for left,right,eligible in (("parent","v0",not arms["v0"]["held_steps"]),
                                   ("v0","v01",not arms["v01"]["all_unsafe_releases"]),
                                   ("parent","v01",not arms["v01"]["held_steps"])):
            if eligible:
                with np.load(arms[left]["trace"]) as a,np.load(arms[right]["trace"]) as b:
                    np.testing.assert_array_equal(a["actions"],b["actions"])
                for k in ("terminal","discounted_return","minimum_clearance","navigation_time"):
                    if arms[left][k] != arms[right][k]:
                        raise ValueError("No-intervention parity differs")
                parity[left+"/"+right] += 1
    hashes = {str(p):digest(p) for p in sorted((OUT/"episodes").glob("*.json"))}
    result = dict(episodes=len(records),matched_initial_worlds=len(groups),control_steps=dict(steps),
        max_smoothing_error=smooth_error,exact_no_intervention_pairs=dict(parity),
        identical_command_prefix_before_first_release=prefix_checks,releases=dict(releases),
        parameters=sorted({r["parameters"] for r in records}),
        protocol_sha256=digest(OUT/"protocol.json"),summary_sha256=digest(OUT/"summary.json"),
        episode_summary_sha256=digest(OUT/"episode-summary.json"),validator_sha256=digest(Path(__file__)),
        metadata_bundle_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        correction="Frozen runner's validation sorted filenames lexically, producing one1.1e-16 mean_q difference. This independent checker uses the original numeric execution-summary order. No source used for control, protocol, episode or summary changed; no reruns.",
        limits="State vectors are native to_array float32; commands/time float64. GPU concurrent timings are descriptive, not matched-state latency benchmarks.")
    save_json(OUT/"validation.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    validate()
