"""Independent native-case confirmation of the frozen PaS history diagnostic."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pas_problem_discovery as parent
from experiments.pas_long_occlusion import restore


OUT = parent.ROOT / "outputs/pas_history_confirmation"
CASES = [(n, g, c) for n in (10, 20)
         for g in ("circle_crossing", "square_crossing") for c in range(93000, 93008)]
ARMS = ("full4", "projected_parent", "short_cv", "long_cv", "long_hold")
ACTOR = CONFIG = None


def prepare():
    original = json.loads((parent.OUT / "protocol.json").read_text())
    parent.verify_inputs(original)
    previous = [json.loads(p.read_text())["record"] for p in parent.OUT.glob("case_*.json")]
    assert not set(CASES) & {(r["people"], r["geometry"], r["case"]) for r in previous}
    parent.save(OUT / "protocol.json", dict(
        cases=CASES, arms=ARMS, origin=original,
        script_sha256=parent.digest(Path(__file__)),
        restore_sha256=parent.digest(Path(restore.__code__.co_filename)),
        rule="All32 independently numbered native cases; first t>=2s with a previously seen "
             "invisible actor within3m whose last legal sighting was>=2s ago. "
             "Do not select episodes by failure or root by intervention benefit. "
             "No eligible root means no intervention, not a replaced case.",
        frozen="Same checkpoint, native Standard sensor, reward, dynamics and clipping; "
               "same anonymous tracker,1m gate,0.75s/4s CV and4s hold; same Label-VAE consumer. "
               "One root action only, then original full4 continuation and recurrent update. "
               "No oracle, training, tuning, action sweep or92001 re-analysis.",
        decision="Repeated rescue requires>=2 different confirmation cases where native, "
                 "projected-parent and short-CV fail but long-CV succeeds. "
                 "Require no new collision on native-success cases, no aggregate collision "
                 "increase against native/short-CV, and net success gain against both. "
                 "Any harmful transitions and clearance decreases must be reported. "
                 "If fewer than2 native-failure eligible cases and no safety veto, "
                 "failure coverage is insufficient, not evidence that history has no value. "
                 "This is small-sample problem confirmation, not method success.",
        resources="Four independent CPU workers, one Torch thread per worker. "
                  "Save each entire paired block atomically after all arms; resume completed cases."))


def verify():
    protocol = json.loads((OUT / "protocol.json").read_text())
    parent.verify_inputs(protocol["origin"])
    assert parent.digest(Path(__file__)) == protocol["script_sha256"]
    assert parent.digest(Path(restore.__code__.co_filename)) == protocol["restore_sha256"]
    assert [list(c) for c in CASES] == protocol["cases"]
    return protocol


def initialize():
    global ACTOR, CONFIG
    torch.set_num_threads(1)
    verify()
    ACTOR, CONFIG = parent.build_actor()


def information(short, long, root):
    added = (long > short) & (root["observation"]["grid"][-1] == .5)
    truth = root["observation"]["label_grid"][0] > .5
    ages = [root["time"] - row["time"] for row in root["tracker"].tracks]
    return dict(added_cells=int(added.sum()), added_true_occupied=int((added & truth).sum()),
                added_true_free=int((added & ~truth).sum()),
                older_track_ages=[float(a) for a in ages if .75 < a <= 4.])


@torch.inference_mode()
def run_case(index):
    destination = OUT / f"case_{index:02d}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    started = time.perf_counter()
    people, geometry, case = CASES[index]
    record, _, legal = parent.collect(ACTOR, CONFIG, people, geometry, case)
    # The inherited collector's outside4 root is not this experiment's root.
    record["root_tick"] = None
    root = restore(ACTOR, CONFIG, record) if record["counts"]["seen_hidden2s"] else None
    if root is not None:
        record["root_tick"] = root["tick"]
    torch.save(dict(record=record, root=root, legal_trace=legal), OUT / f"trace_{index:02d}.pt")
    arms, extra = {}, None
    if root is not None:
        baseline = parent.branch(ACTOR, CONFIG, record, root, root["native_action"])
        np.testing.assert_array_equal(baseline["commands"], record["actions"][root["tick"]:])
        assert baseline["terminal"] == record["terminal"]
        arms["full4"] = dict(action=root["native_action"].tolist(), actual=baseline)
        sensor = root["observation"]["grid"][-1]
        native = root["decoded"].reshape(sensor.shape)
        grids = {"projected_parent": native,
                 "short_cv": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"], ttl=.75),
                 "long_cv": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"]),
                 "long_hold": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"], cv=False)}
        extra = information(grids["short_cv"], grids["long_cv"], root)
        for name, grid in grids.items():
            before = time.perf_counter()
            action = parent.occupancy_query(ACTOR, root, grid)
            query_seconds = time.perf_counter() - before
            arms[name] = dict(action=action.tolist(), query_seconds=query_seconds,
                              actual=parent.branch(ACTOR, CONFIG, record, root, action),
                              occupancy=parent.grid_error(grid, root))
        for arm in arms.values():
            arm["delta_return_from_native"] = arm["actual"]["return_"] - baseline["return_"]
    result = dict(index=index, record=record, eligible=root is not None,
                  root_events=None if root is None else root["events"], arms=arms,
                  additional_history=extra, elapsed_seconds=time.perf_counter()-started,
                  limits="New native trajectories, not original RL replay; frozen root-action "
                         "intervention, not a continuously deployed memory policy. "
                         "Label-VAE projection can shift inputs; prior policy GRU retained. "
                         "Query timings are single CPU observations, not deployment benchmarks.")
    parent.save(destination, result)
    return result


def evaluate(workers):
    verify()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for result in pool.map(run_case, range(len(CASES)), chunksize=1):
            print("PAS_CONFIRM", result["index"], result["record"]["people"],
                  result["record"]["geometry"], result["record"]["case"],
                  "native", result["record"]["terminal"], "root", result["record"]["root_tick"],
                  {k: v["actual"]["terminal"] for k, v in result["arms"].items()},
                  "seconds", round(result["elapsed_seconds"], 2), flush=True)
    verify()
    parent.save(OUT / "completion.json", dict(cases=len(CASES), workers=workers,
                invocation_wall_seconds=time.perf_counter()-started,
                result_sha256={str(p.name): parent.digest(p) for p in sorted(OUT.glob("case_*.json"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 4))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else evaluate(args.workers)
