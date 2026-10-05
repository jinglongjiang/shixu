"""Native PaS event coverage, simple legal memory and action headroom.

Run standalone without shixu/vendor on PYTHONPATH. No navigation training.
Ground-truth actor IDs and grids are labels only; the tracker reads sensor
occupancy and the robot's own coordinate map, not pedestrian truth.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pas_temporal_probe import (
    CHECKPOINT, PARENT, branch, clone_hidden, digest, new_world, query, save,
    tensors, world_state,
)

OUT = ROOT / "outputs/pas_problem_discovery"
OLD = ROOT / "outputs/temporal_problem_audit/pas"
RULES = {
    "scope": "Problem discovery only. Existing adapted PaS seed1207; no training or new temporal architecture.",
    "existing": "Replay the eight existing91000/91001 native command trajectories first. "
                "Earlier formal evaluation files contain episode summaries, not command/observation traces.",
    "new_cases": [[n, g, c] for n in (10, 20) for g in ("circle_crossing", "square_crossing")
                  for c in range(92000, 92004)],
    "native": "Keep checkpoint, reward, dynamics, visibility, grid resolution, standard sensor, "
              "0.25s clock and continuous action support. Do not add noise, delay or occlusion.",
    "root": "First t>=2s with an invisible actor within3m and absent from last4 visibility records. "
            "No root substitution based on action or intervention outcome. Truth is event enrichment only.",
    "selection": "After collection, first three failure roots and first three success roots in queue order. "
                 "One root per episode. Failures are not KDA-selected. Selection uses parent terminal only.",
    "memory": "Grid components of occupied cells>=0.9, area>=3pixels. Coordinate map is own-robot geometry. "
              "CV velocity from consecutive associated centres; Hungarian predicted-position association "
              "gate1m; maximum legal memory4s. No simulator identity or true human velocity is consumed. "
              "Compare0.75s CV,4s CV and4s position-hold, without tuning.",
    "consumer": "Native full4, short2 and current-grid interventions retain the same existing policy GRU. "
                "Separately use a common Label-VAE read of native decoded occupancy, with short/long "
                "tracker additions in currently unknown cells, or GT occupancy. Label-VAE projection "
                "and probability-map inputs may shift distribution; this is a diagnostic, not a new method.",
    "headroom": "Failure roots only: native action plus stop and32 native-support velocities "
                "(16headings, speeds0.5/1). One root step, then identical frozen full4 continuation. "
                "Finite candidate headroom is not a global oracle. Never redefine reward as progress.",
    "continuation": "Use the original full4 root recurrent update and saved next RNG for all branches, "
                    "not an intervened memory write. Subsequent observations/RNG naturally depend on motion. "
                    "Report original gamma.99 return, terminal, duration and swept clearance.",
    "kill_boundary": "An incompatible frozen consumer's negative result cannot kill all history problems. "
                     "GT benefit is positive headroom. Simple recovery of the same gain weakens complex "
                     "memory entry. Prediction/action changes alone are insufficient. No KDA restart.",
}


class GridTracks:
    """Simple anonymous CV tracks from measured occupied grid components."""

    def __init__(self):
        self.tracks = []

    def update(self, grid, coordinates, now):
        components, count = label(np.asarray(grid) >= .9)
        centres = []
        for index in range(1, count + 1):
            pixels = components == index
            if pixels.sum() >= 3:
                centres.append([np.asarray(axis)[pixels].mean() for axis in coordinates])
        centres = np.asarray(centres, dtype=float).reshape(-1, 2)
        self.tracks = [t for t in self.tracks if now - t["time"] <= 4.]
        matched = set()
        if self.tracks and len(centres):
            predicted = np.asarray([t["position"] + (now - t["time"]) * t["velocity"] for t in self.tracks])
            distance = np.linalg.norm(predicted[:, None] - centres[None], axis=-1)
            for track, measurement in zip(*linear_sum_assignment(distance)):
                if distance[track, measurement] > 1.:
                    continue
                row = self.tracks[track]
                elapsed = now - row["time"]
                if elapsed > 0:
                    row["velocity"] = (centres[measurement] - row["position"]) / elapsed
                row.update(position=centres[measurement], time=now, measurements=row["measurements"] + 1)
                matched.add(measurement)
        for index, centre in enumerate(centres):
            if index not in matched:
                self.tracks.append(dict(position=centre.copy(), velocity=np.zeros(2), time=now, measurements=1))

    def occupancy(self, base, sensor, coordinates, now, ttl=4., cv=True):
        result = np.asarray(base).copy()
        unknown = np.asarray(sensor) == .5
        for row in self.tracks:
            age = now - row["time"]
            if age <= 0 or age > ttl:
                continue
            centre = row["position"] + age * row["velocity"] if cv else row["position"]
            footprint = sum((np.asarray(axis) - centre[k]) ** 2 for k, axis in enumerate(coordinates)) <= .3 ** 2
            result[footprint & unknown] = 1.
        return result


def events(env, seen, last):
    visible = set(map(int, env.visible_ids[-1]))
    seen.update(visible)
    tick = int(round(env.global_time / env.time_step))
    for actor in visible:
        last[actor] = tick
    close = [i for i, h in enumerate(env.humans)
             if i not in visible and np.linalg.norm(np.asarray(h.get_position())-env.robot.get_position()) <= 3]
    return dict(invisible_close=close, never_seen=[i for i in close if i not in seen],
                outside4=[i for i in close if i not in last or tick-last[i] >= 4],
                seen_outside4=[i for i in close if i in last and tick-last[i] >= 4],
                seen_hidden2s=[i for i in close if i in last and tick-last[i] >= 8])


def build_actor():
    sys.path.insert(0, str(PARENT))
    from crowd_nav.configs.config import Config
    from rl.model import Policy
    cfg = Config()
    env, _ = new_world(cfg, 5, "circle_crossing", 91000)
    args = SimpleNamespace(num_steps=30, num_processes=12, num_mini_batch=2, rnn_input_size=64,
                           rnn_hidden_size=128, rnn_output_size=128, rnn_embedding_size=64,
                           vae_weights=str(OLD / "extracted_existing_label_vae.pt"))
    actor = Policy(env.action_space, config=cfg, base="pas_rnn", base_kwargs=args).eval()
    actor.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False), strict=True)
    return actor, cfg


def prepare():
    metadata = ROOT.parent / "pas_formal_20260922_weights/pas_1207/metadata.json"
    source = json.loads(metadata.read_text())
    assert digest(CHECKPOINT) == source["final_checkpoint_sha256"]
    assert digest(PARENT / "crowd_nav/configs/config.py") == source["config_sha256"]
    existing = [{"record": json.loads(path.read_text())["record"], "path": str(path), "sha256": digest(path)}
                for path in sorted(OLD.glob("case_*.json"))]
    assert len(existing) == 8
    save(OUT / "protocol.json", dict(rules=RULES, existing=existing,
        checkpoint_sha256=digest(CHECKPOINT), config_sha256=digest(PARENT / "crowd_nav/configs/config.py"),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PARENT, text=True).strip(),
        parent_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PARENT)),
        source_sha256={str(p.relative_to(PARENT)): digest(p) for p in sorted(PARENT.rglob("*.py"))},
        script_sha256=digest(Path(__file__)), torch=torch.__version__))


@torch.inference_mode()
def collect(actor, cfg, people, geometry, case, original=None):
    env, observation = new_world(cfg, people, geometry, case)
    torch.manual_seed(case)
    hidden = {k: torch.zeros(1, 1, 128) for k in ("pas", "policy")}
    masks, tracker, seen, last = torch.ones(1, 4), GridTracks(), set(), {}
    counts = {k: 0 for k in ("invisible_close", "never_seen", "outside4", "seen_outside4", "seen_hidden2s")}
    commands, rewards, executed, grids, robot_positions, root = [], [], [], [], [], None
    for tick in range(300):
        tracker.update(observation["grid"][-1], env.xy_local_grid[-1], env.global_time)
        flags = events(env, seen, last)
        for key, actors in flags.items():
            counts[key] += bool(actors)
        grids.append((observation["grid"][-1] * 2).astype(np.uint8))
        robot_positions.append(list(env.robot.get_position()))
        if root is None and tick >= 8 and flags["outside4"]:
            root = dict(tick=tick, observation=copy.deepcopy(observation), hidden=clone_hidden(hidden),
                        rng=torch.get_rng_state().clone(), world=world_state(env), events=copy.deepcopy(flags),
                        coordinates=copy.deepcopy(env.xy_local_grid[-1]), tracker=copy.deepcopy(tracker),
                        time=env.global_time, past_grid_count=len(grids))
        action, hidden, after_rng, decoded = query(actor, observation, hidden, masks, torch.get_rng_state(), "full4")
        if original is not None:
            np.testing.assert_allclose(action, original["actions"][tick], atol=1e-7, rtol=0)
            action = np.asarray(original["actions"][tick])
        if root is not None and tick == root["tick"]:
            root.update(after_hidden=clone_hidden(hidden), after_rng=after_rng,
                        native_action=action.copy(), decoded=decoded.copy())
        observation, reward, done, info = env.step(action)
        commands.append(action.tolist())
        rewards.append(float(reward))
        executed.append([env.robot.vx, env.robot.vy])
        if done:
            break
    else:
        raise RuntimeError("Native episode did not terminate")
    record = dict(people=people, geometry=geometry, case=case, terminal=type(info["info"]).__name__,
                  actions=commands, rewards=rewards, executed=executed, counts=counts,
                  root_tick=None if root is None else root["tick"], frames=len(commands),
                  original_archive_replay=original is not None)
    if original is not None:
        assert record["terminal"] == original["terminal"] and len(commands) == len(original["actions"])
        np.testing.assert_array_equal(rewards, original["rewards"])
    return record, root, dict(sensor_grids=np.asarray(grids), robot_positions=np.asarray(robot_positions))


def verify_inputs(protocol):
    assert digest(CHECKPOINT) == protocol["checkpoint_sha256"]
    assert digest(Path(__file__)) == protocol["script_sha256"]
    assert all(digest(PARENT / p) == h for p, h in protocol["source_sha256"].items())


def collection():
    protocol = json.loads((OUT / "protocol.json").read_text())
    verify_inputs(protocol)
    actor, cfg = build_actor()
    queue = [(row["record"]["people"], row["record"]["geometry"], row["record"]["case"], row["record"])
             for row in protocol["existing"]]
    queue += [(*case, None) for case in RULES["new_cases"]]
    for index, (people, geometry, case, original) in enumerate(queue):
        destination = OUT / f"case_{index:02d}.json"
        if destination.exists():
            continue
        started = time.perf_counter()
        record, root, legal = collect(actor, cfg, people, geometry, case, original)
        torch.save(dict(record=record, root=root, legal_trace=legal), OUT / f"trace_{index:02d}.pt")
        save(destination, dict(record=record, root_events=None if root is None else root["events"],
             elapsed_seconds=time.perf_counter()-started))
        print("PAS_NATIVE_COVERAGE", index, people, geometry, case, record["terminal"],
              "root", record["root_tick"], "counts", record["counts"], flush=True)
    verify_inputs(protocol)


@torch.inference_mode()
def occupancy_query(actor, root, occupancy):
    torch.set_rng_state(root["rng"].clone())
    before = actor.base.Sensor_VAE.encode
    grid = torch.as_tensor(occupancy, dtype=torch.float32)[None, None]
    actor.base.Sensor_VAE.encode = lambda _: actor.base.Label_VAE.encode(grid)
    try:
        _, action, _, _, _ = actor.act(tensors(root["observation"]), clone_hidden(root["hidden"]),
                                       torch.ones(1, 4), deterministic=True)
    finally:
        actor.base.Sensor_VAE.encode = before
    return action[0].numpy().copy()


def grid_error(probability, root):
    sensor = root["observation"]["grid"][-1]
    truth = root["observation"]["label_grid"][0]
    unknown = sensor == .5
    occupied = (truth > .5) & unknown
    return dict(unknown_cells=int(unknown.sum()), hidden_occupied_cells=int(occupied.sum()),
        unknown_brier=float(np.mean((probability[unknown]-truth[unknown])**2)) if unknown.any() else None,
        hidden_occupied_mse=float(np.mean((probability[occupied]-truth[occupied])**2)) if occupied.any() else None,
        hidden_occupied_recall=float(np.mean(probability[occupied] >= .5)) if occupied.any() else None)


@torch.inference_mode()
def diagnose():
    protocol = json.loads((OUT / "protocol.json").read_text())
    verify_inputs(protocol)
    actor, cfg = build_actor()
    rows = [(int(p.stem.split("_")[-1]), json.loads(p.read_text())) for p in sorted(OUT.glob("case_*.json"))]
    assert len(rows) == 24
    eligible = [(i, r) for i, r in rows if r["record"]["root_tick"] is not None]
    failures = [i for i, r in eligible if r["record"]["terminal"] != "ReachGoal"][:3]
    successes = [i for i, r in eligible if r["record"]["terminal"] == "ReachGoal"][:3]
    selection = dict(failures=failures, successes=successes,
                     rule=RULES["selection"], case_hashes={str(i): digest(OUT/f"case_{i:02d}.json") for i, _ in rows})
    selection_path = OUT / "selection.json"
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selection
    else:
        save(selection_path, selection)
    for index in failures + successes:
        destination = OUT / f"diagnostic_{index:02d}.json"
        if destination.exists():
            continue
        started = time.perf_counter()
        stored = torch.load(OUT/f"trace_{index:02d}.pt", weights_only=False)
        record, root = stored["record"], stored["root"]
        sensor = root["observation"]["grid"][-1]
        native = root["decoded"].reshape(sensor.shape)
        probability = {"projected_parent": native,
            "short_cv": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"], ttl=.75),
            "long_cv": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"]),
            "long_hold": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"], cv=False),
            "oracle_occupancy": root["observation"]["label_grid"][0].copy()}
        arms = {}
        for mode in ("full4", "short2", "current_grid"):
            action, _, _, _ = query(actor, root["observation"], root["hidden"], torch.ones(1,4), root["rng"], mode)
            actual = branch(actor, cfg, record, root, action)
            arms[mode] = dict(action=action.tolist(), actual=actual)
        baseline = arms["full4"]["actual"]
        np.testing.assert_array_equal(baseline["commands"], record["actions"][root["tick"]:])
        assert baseline["terminal"] == record["terminal"]
        for mode, grid in probability.items():
            action = occupancy_query(actor, root, grid)
            actual = branch(actor, cfg, record, root, action)
            arms[mode] = dict(action=action.tolist(), actual=actual, occupancy=grid_error(grid, root),
                             changed_cells_from_projected=int(np.count_nonzero(grid != native)))
        for arm in arms.values():
            arm["delta_return_from_native"] = arm["actual"]["return_"] - baseline["return_"]
        headroom = []
        if index in failures:
            candidates = [root["native_action"], np.zeros(2)]
            candidates += [speed*np.asarray([np.cos(theta),np.sin(theta)]) for speed in (.5,1.)
                           for theta in np.arange(16)*2*np.pi/16]
            for action in candidates:
                headroom.append(dict(action=action.tolist(), actual=branch(actor,cfg,record,root,action)))
        save(destination, dict(index=index, record={k:v for k,v in record.items() if k not in ("actions","rewards","executed")},
            root_events=root["events"], arms=arms, headroom=headroom,
            elapsed_seconds=time.perf_counter()-started,
            limits="One root action, frozen native continuation; shared occupancy projection may be distribution-shifted. "
                   "Current-grid/short2 retain policy GRU; they are not trained memory-free controls."))
        print("PAS_DECISION_DIAGNOSTIC", index, "baseline",record["terminal"],
              {k:(v["actual"]["terminal"], round(v["delta_return_from_native"],6)) for k,v in arms.items()},
              "headroom_successes",sum(x["actual"]["terminal"]=="ReachGoal" for x in headroom),flush=True)
    verify_inputs(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "collect", "diagnose"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"prepare": prepare, "collect": collection, "diagnose": diagnose}[args.mode]()
