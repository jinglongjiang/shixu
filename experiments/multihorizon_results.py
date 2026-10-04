"""Summarize frozen first-action repairs without treating roots as full episodes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = json.loads((folder.parent / "protocol.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("root_*.json"))]
    cohort = torch.load(folder.parent / "inputs/cohort.pt", map_location="cpu", weights_only=False)
    if len(records) != len(protocol["selection"]):
        raise ValueError("Incomplete prespecified cohort")
    for index, (record, expected) in enumerate(zip(records, protocol["selection"])):
        actual = {**record["episode"], "group": record["group"], "tick": record["tick"]}
        if actual != expected or record["root"] != index or not record["archive_parity"]["passed"]:
            raise ValueError("Root or baseline parity differs from frozen protocol")
    methods = list(records[0]["choices"])
    def progress(record, outcome):
        robot = cohort["roots"][record["root"]]["snapshot"]["robot"]
        start, goal = np.asarray(robot[:2]), np.asarray(robot[5:7])
        end = start + .25 * np.asarray(outcome["commands"]).sum(0)
        return float(np.linalg.norm(goal - start) - np.linalg.norm(goal - end))
    groups = {name: [r for r in records if r["group"] == name]
              for name in ("failure", "success_guard")}
    outcomes = []
    for group, rows in groups.items():
        for name in methods:
            choices = [r["choices"][name] for r in rows]
            episode_key = lambda r: (r["episode"]["people"], r["episode"]["geometry"], r["episode"]["case"])
            successes = [r for r, c in zip(rows, choices) if c["realized"]["terminal"] == "reach_goal"]
            outcomes.append(dict(group=group, method=name, roots=len(rows),
                terminals={terminal: sum(c["realized"]["terminal"] == terminal for c in choices)
                           for terminal in ("reach_goal", "collision", "timeout")},
                rescued_roots=sum(c["rescued"] for c in choices),
                distinct_cases_with_any_rescue=len({episode_key(r) for r, c in zip(rows, choices) if c["rescued"]}),
                damaged_roots=sum(c["damaged_success"] for c in choices),
                delta_return_mean=float(np.mean([c["delta_return"] for c in choices])),
                return_mean=float(np.mean([c["realized"]["return_"] for c in choices])),
                success_seconds_mean=float(np.mean([r["choices"][name]["realized"]["seconds"]
                                                     for r in successes])) if successes else None,
                minimum_clearance_mean=float(np.mean([c["realized"]["minimum_clearance"] for c in choices])),
                goal_distance_reduction_mean=float(np.mean([progress(r, c["realized"])
                                                             for r, c in zip(rows, choices)])),
                shadow_terminal_disagreements=sum(c["predicted_terminal"] != c["realized"]["terminal"]
                                                  for c in choices),
                shadow_success_actual_failure=sum(c["predicted_terminal"] == "reach_goal"
                                                  and c["realized"]["terminal"] != "reach_goal" for c in choices),
                selected_return_prediction_error_mean=float(np.mean([
                    c["raw_returns"][c["action"]] - c["realized"]["return_"] for c in choices])),
                scoring_seconds_median=float(np.median([c["scoring_seconds"] for c in choices])),
                scoring_seconds_total=sum(c["scoring_seconds"] for c in choices),
                selected_commands_changed=sum(not np.allclose(c["realized"]["commands"][0],
                                                               r["baseline"]["commands"][0], atol=1e-8, rtol=0)
                                               for r, c in zip(rows, choices)),
                no_raw_score_separation=sum(c["raw_range"] <= 1e-8 for c in choices)))
    contrasts = []
    for seed in (419, 443, 467, 491):
        for reference in ("cv", "truth", f"{seed}_current", f"{seed}_gru"):
            target = f"{seed}_kda"
            for group, rows in groups.items():
                pairs = [(r["choices"][target]["realized"], r["choices"][reference]["realized"]) for r in rows]
                contrasts.append(dict(seed=seed, target=target, reference=reference, group=group, roots=len(rows),
                    target_only_success=sum(a["terminal"] == "reach_goal" and b["terminal"] != "reach_goal"
                                            for a, b in pairs),
                    reference_only_success=sum(b["terminal"] == "reach_goal" and a["terminal"] != "reach_goal"
                                               for a, b in pairs),
                    target_collisions=sum(a["terminal"] == "collision" for a, _ in pairs),
                    reference_collisions=sum(b["terminal"] == "collision" for _, b in pairs),
                    delta_return_mean=float(np.mean([a["return_"] - b["return_"] for a, b in pairs]))))
    result = dict(manifest=manifest, roots=len(records), failure_cases=len({
        (r["episode"]["people"], r["episode"]["geometry"], r["episode"]["case"])
        for r in groups["failure"]}), groups=outcomes, contrasts=contrasts,
        total_seconds=sum(r["elapsed_seconds"] for r in records),
        all_archive_parities_pass=True,
        native_shadow_fallbacks=sum(c["native_shadow_fallback"] for r in records for c in r["choices"].values()),
        maximum_batch_score_error=max(c["max_error"] for r in records for c in r["batch_validation"]),
        scope="Twenty roots in ten existing failed cases plus six success guards. Two roots from one case "
              "are correlated. Four predictor training seeds share one frozen navigation policy; not four "
              "independent navigation evaluations. CV/truth controls are shared, not replicated. "
              "Case-with-any-rescue uses alternative anchor interventions and is not a deployed-policy SR.")
    path = folder.parent / "summary.json"
    if path.exists():
        raise RuntimeError("Refusing to overwrite final summary")
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    for row in outcomes:
        print(row["group"], row["method"], row["terminals"], "rescues", row["rescued_roots"],
              "damages", row["damaged_roots"], "delta", round(row["delta_return_mean"], 5))
    print("TOTAL_SECONDS", round(result["total_seconds"], 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
