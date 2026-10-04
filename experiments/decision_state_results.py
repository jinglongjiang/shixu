"""Aggregate the bounded current-state audit without inventing unrun arms."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.decision_state import distribution
from experiments.forecast_control_diagnostic import digest
from experiments.multihorizon_control import save_json


def summarize(folder):
    data = torch.load(folder / "cohort.pt", map_location="cpu", weights_only=False)
    if digest(folder / "cohort.pt") != json.loads((folder / "cohort_hash.json").read_text())["sha256"]:
        raise ValueError("Cohort checksum mismatch")
    natural = json.loads((folder / "natural_events.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((folder / "results").glob("root_*.json"))]
    if len(records) != len(data["roots"]):
        raise ValueError("Incomplete paired blocks")
    for index, (record, root) in enumerate(zip(records, data["roots"])):
        if (record["root"] != index or record["episode"] != root["episode"]
                or record["tick"] != root["tick"] or record["group"] != root["group"]
                or not record["archive_parity"]["passed"]):
            raise ValueError("Selection or native continuation differs")
    outcomes = []
    for group in ("failure", "success_guard"):
        rows = [r for r in records if r["group"] == group]
        for name in ("current", "simple", "truth"):
            choices = [r["choices"][name] for r in rows]
            progress = []
            for record, choice in zip(rows, choices):
                robot = data["roots"][record["root"]]["snapshot"]["robot"]
                end = robot[:2] + .25 * np.asarray(choice["actual"]["commands"]).sum(0)
                progress.append(float(np.linalg.norm(robot[5:7] - robot[:2]) - np.linalg.norm(robot[5:7] - end)))
            outcomes.append(dict(group=group, method=name, roots=len(rows),
                terminals={t: sum(c["actual"]["terminal"] == t for c in choices)
                           for t in ("reach_goal", "collision", "timeout")},
                action_changes=sum(c["command_changed"] for c in choices),
                rescued=sum(c["rescued"] for c in choices), damaged=sum(c["damaged"] for c in choices),
                delta_return_mean=float(np.mean([c["delta_return"] for c in choices])),
                position_error_mean=float(np.mean([c["hidden_position_error_mean"] for c in choices])),
                velocity_error_mean=float(np.mean([c["hidden_velocity_error_mean"] for c in choices])),
                minimum_clearance_mean=float(np.mean([c["actual"]["minimum_clearance"] for c in choices])),
                goal_distance_reduction_mean=float(np.mean(progress)),
                successful_seconds_mean=float(np.mean([c["actual"]["seconds"] for c in choices
                                                     if c["actual"]["terminal"] == "reach_goal"]))
                                        if any(c["actual"]["terminal"] == "reach_goal" for c in choices) else None,
                scoring_seconds=distribution([c["scoring_seconds"] for c in choices])))
    natural_groups = []
    episodes = natural["episodes"]
    for people in (5, 10, 20):
        for geometry in ("circle", "square"):
            selected = [r for r in episodes if r["people"] == people and r["geometry"] == geometry]
            natural_groups.append(dict(people=people, geometry=geometry, episodes=len(selected),
                frames=sum(r["control_frames"] for r in selected),
                hidden_frames=sum(r["hidden_control_frames"] for r in selected),
                near_hidden_frames=sum(r["near_hidden_control_frames"] for r in selected),
                episodes_with_near_hidden=sum(r["near_hidden_control_frames"] > 0 for r in selected),
                failure_episodes=sum(r["terminal"] != "reach_goal" for r in selected)))
    errors = {key: distribution([v for e in episodes for v in e[key]]) for key in (
        "hidden_position_error", "hidden_velocity_error", "simple_position_error", "simple_velocity_error",
        "near_hidden_position_error", "near_hidden_velocity_error")}
    result = dict(protocol=data["protocol"], cohort_sha256=digest(folder / "cohort.pt"),
        all_archive_parities_pass=True, episodes=len(episodes),
        frames=sum(r["control_frames"] for r in episodes),
        person_frames=sum(r["person_frames"] for r in episodes),
        reentries=sum(r["reentries"] for r in episodes), natural_cells=natural_groups, errors=errors,
        roots=len(records), outcomes=outcomes, first_paired_block_seconds=records[0]["elapsed_seconds"],
        block_seconds=distribution([r["elapsed_seconds"] for r in records]),
        unrun=dict(actor_gru="No compatible current-p/v estimation head; local training allowed but not initiated "
                             "because no concrete recoverable action/outcome deficit was established.",
                   actor_kda="Same limitation and stopping reason; no random-head navigation comparison.",
                   confirmation="No stable positive action/outcome signal; independent cases left unopened."),
        scope="A root-command intervention with one frozen native consumer; not overall SR, trained-state-head "
              "comparison, proof of zero history value or a co-trained representation upper bound.")
    save_json(folder / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
