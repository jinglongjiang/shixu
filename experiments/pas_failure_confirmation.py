"""Parent-only failure enrichment, followed by a frozen five-arm diagnostic."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pas_problem_discovery as parent
from experiments.pas_history_confirmation import ARMS, information
from experiments.pas_long_occlusion import restore


OUT = parent.ROOT / "outputs/pas_failure_confirmation"
ACTOR = CONFIG = None


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported JSON value: " + type(value).__name__)


def save(path, value):
    if path.exists():
        raise RuntimeError("Refusing to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=numpy_scalar))
    temporary.replace(path)


def parent_fields(record):
    return dict(people=record["people"], geometry=record["geometry"], case=record["case"],
                terminal=record["terminal"], long_hidden=record["counts"]["seen_hidden2s"] > 0)


def rank_configs(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["people"], record["geometry"])].append(record)
    ranking = []
    for (people, geometry), rows in groups.items():
        failures = sum(r["terminal"] != "ReachGoal" for r in rows)
        hidden = sum(r["long_hidden"] for r in rows)
        joint = sum(r["terminal"] != "ReachGoal" and r["long_hidden"] for r in rows)
        ranking.append(dict(people=people, geometry=geometry, episodes=len(rows), failures=failures,
                            long_hidden=hidden, joint_failures=joint, joint_rate=joint/len(rows)))
    return sorted(ranking, key=lambda r: (-r["joint_rate"], -r["joint_failures"],
                  -r["failures"]/r["episodes"], -r["long_hidden"]/r["episodes"],
                  r["people"], r["geometry"]))


def prepare():
    original = json.loads((parent.OUT / "protocol.json").read_text())
    parent.verify_inputs(original)
    files = [p for d in (parent.OUT, parent.ROOT/"outputs/pas_history_confirmation")
             for p in sorted(d.glob("case_*.json"))]
    records = [parent_fields(json.loads(p.read_text())["record"]) for p in files]
    assert len(records) == 56
    ranking = rank_configs(records)
    configs = [r for r in ranking if r["joint_failures"] > 0][:2]
    assert configs, "No parent-only enriched configuration exists"
    queue = [(configs[i % len(configs)]["people"], configs[i % len(configs)]["geometry"], 95000+i)
             for i in range(160)]
    assert not {r["case"] for r in records} & {r[2] for r in queue}
    save(OUT/"protocol.json", dict(
        origin=original, parent_manifest=records, ranking=ranking, configs=configs, queue=queue,
        prior_files_sha256={str(p): parent.digest(p) for p in files},
        source_sha256={str(Path(p)): parent.digest(Path(p)) for p in
                       (__file__, restore.__code__.co_filename, information.__code__.co_filename)},
        rule="First t>=2s, previously seen invisible actor within3m, last sighting>=2s ago. "
             "First8 parent-failure eligible episodes in index order. Stop submitting new "
             "parent batches once8 are collected, or160 episodes. Four-case CPU batches may "
             "finish up to3 in-flight extras after the8th failure; retain but exclude them "
             "from selection. No method arm runs until selection is saved and hashed.",
        protection="First eligible parent successes in queue order, matching each failure configuration. "
                   "Take the same count as failures per configuration, then fill to8 in queue order "
                   "from selected configurations if the cap yields fewer than8 failures. "
                   "Insufficient protection does not authorize additional screening.",
        contract=parent.RULES["consumer"]+" Only the five previously frozen arms are evaluated. "
                 "No oracle arm, training, parameter tuning or action sweep.",
        continuation=parent.RULES["continuation"],
        safety_margin_m=.05, safety_decrease_m=.05,
        decision="With8 failure roots and8 protection roots, require>=2 cases where full4, "
                 "projected-parent and short-CV fail but long-CV succeeds; protected native "
                 "successes must all remain success. Long-CV collisions must not exceed "
                 "short-CV or full4 over all selected roots. Counted rescues must have>=.05m "
                 "swept clearance. No protection root may lose>=.05m clearance against "
                 "either native or short-CV. Report all Q/clearance differences, even if "
                 "the terminal gate passes. With8 failure opportunities and0/1 rescue, "
                 "close this frozen use, not all PaS/shixu temporal approaches. "
                 "Missing failure/protection quota means insufficient coverage; stop at cap."))


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text())
    parent.verify_inputs(protocol["origin"])
    hashes = dict(protocol["source_sha256"])
    correction_path = OUT/"serialization-correction.json"
    if correction_path.exists():
        correction = json.loads(correction_path.read_text())
        assert correction["protocol_sha256"] == parent.digest(OUT/"protocol.json")
        assert parent.digest(OUT/"frozen-initial-source.py") == hashes[str(Path(__file__))]
        hashes[str(Path(__file__))] = correction["script_sha256"]
    replay_path = OUT/"replay-dtype-correction.json"
    if replay_path.exists():
        correction = json.loads(replay_path.read_text())
        assert correction["protocol_sha256"] == parent.digest(OUT/"protocol.json")
        assert parent.digest(OUT/"frozen-serialized-source.py") == hashes[str(Path(__file__))]
        hashes[str(Path(__file__))] = correction["script_sha256"]
    assert all(parent.digest(Path(p)) == h for p, h in hashes.items())
    assert all(parent.digest(Path(p)) == h for p, h in protocol["prior_files_sha256"].items())
    return protocol


def initialize():
    global ACTOR, CONFIG
    torch.set_num_threads(1)
    verify()
    ACTOR, CONFIG = parent.build_actor()


@torch.inference_mode()
def parent_case(index):
    path = OUT/"screening"/f"case_{index:03d}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.perf_counter()
    people, geometry, case = verify()["queue"][index]
    record, _, legal = parent.collect(ACTOR, CONFIG, people, geometry, case)
    record["root_tick"] = None  # The inherited collector's outside4 root is not selected.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(record=record, legal_trace=legal), path.with_suffix(".pt"))
    result = dict(index=index, record=record, elapsed_seconds=time.perf_counter()-started)
    save(path, result)
    return result


def select_roots(rows):
    rows = sorted(rows, key=lambda r: r["index"])
    failures = [r for r in rows if r["record"]["counts"]["seen_hidden2s"] > 0
                and r["record"]["terminal"] != "ReachGoal"][:8]
    cutoff = failures[-1]["index"] if len(failures) == 8 else rows[-1]["index"]
    usable = [r for r in rows if r["index"] <= cutoff]
    successes = [r for r in usable if r["record"]["counts"]["seen_hidden2s"] > 0
                 and r["record"]["terminal"] == "ReachGoal"]
    quota = defaultdict(int)
    for row in failures:
        r = row["record"]
        quota[(r["people"], r["geometry"])] += 1
    protection, taken = [], defaultdict(int)
    for row in successes:
        r = row["record"]
        config = (r["people"], r["geometry"])
        if taken[config] < quota[config]:
            protection.append(row["index"])
            taken[config] += 1
    if len(protection) < 8:
        protection += [r["index"] for r in successes if r["index"] not in protection][:8-len(protection)]
    return dict(failures=[r["index"] for r in failures], protection=protection,
                cutoff=cutoff, in_flight_excluded=[r["index"] for r in rows if r["index"] > cutoff])


def screen(workers):
    verify()
    selection_path = OUT/"selection.json"
    if selection_path.exists():
        print("Frozen selection already exists", flush=True)
        return
    started, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for start in range(0, 160, 4):
            block = list(pool.map(parent_case, range(start, min(start+4, 160))))
            rows.extend(block)
            selection = select_roots(rows)
            print("PAS_PARENT_SCREEN", len(rows), "eligible_failures", len(selection["failures"]),
                  "eligible_successes", sum(r["record"]["counts"]["seen_hidden2s"] > 0
                  and r["record"]["terminal"] == "ReachGoal" for r in rows), flush=True)
            if len(selection["failures"]) == 8:
                break
    selection.update(screened=len(rows), screening_wall_seconds=time.perf_counter()-started,
                     protocol_sha256=parent.digest(OUT/"protocol.json"),
                     inputs_sha256={str(OUT/"screening"/f"case_{r['index']:03d}.json"):
                        parent.digest(OUT/"screening"/f"case_{r['index']:03d}.json") for r in rows})
    verify()
    save(selection_path, selection)


def native_record(record):
    # JSON has no dtype: restore the Policy.act float32 execution contract.
    return dict(record, actions=[np.asarray(a, dtype=np.float32) for a in record["actions"]])


def verify_baseline(result, record):
    baseline = result["arms"]["full4"]["actual"]
    tick = result["tick"]
    np.testing.assert_array_equal(baseline["commands"], record["actions"][tick:])
    np.testing.assert_array_equal(baseline["executed"], record["executed"][tick:])
    np.testing.assert_allclose(baseline["return_"], sum(.99**i*r for i, r in
                               enumerate(record["rewards"][tick:])), atol=1e-12, rtol=0)
    assert baseline["terminal"] == record["terminal"]


@torch.inference_mode()
def evaluate_root(index):
    path = OUT/"evaluation"/f"root_{index:03d}.json"
    record = json.loads((OUT/"screening"/f"case_{index:03d}.json").read_text())["record"]
    if path.exists():
        result = json.loads(path.read_text())
        verify_baseline(result, record)
        return result
    started = time.perf_counter()
    record = native_record(record)
    root = restore(ACTOR, CONFIG, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(root, path.with_suffix(".pt"))
    baseline = parent.branch(ACTOR, CONFIG, record, root, root["native_action"])
    np.testing.assert_array_equal(baseline["commands"], record["actions"][root["tick"]:])
    assert baseline["terminal"] == record["terminal"]
    arms = {"full4": dict(action=root["native_action"].tolist(), actual=baseline)}
    sensor = root["observation"]["grid"][-1]
    native = root["decoded"].reshape(sensor.shape)
    grids = {"projected_parent": native,
             "short_cv": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"], ttl=.75),
             "long_cv": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"]),
             "long_hold": root["tracker"].occupancy(native, sensor, root["coordinates"], root["time"], cv=False)}
    for name, grid in grids.items():
        action = parent.occupancy_query(ACTOR, root, grid)
        arms[name] = dict(action=action.tolist(), actual=parent.branch(ACTOR, CONFIG, record, root, action),
                          occupancy=parent.grid_error(grid, root))
    result = dict(index=index, people=record["people"], geometry=record["geometry"], case=record["case"],
                  tick=root["tick"], events=root["events"], arms=arms,
                  additional_history=information(grids["short_cv"], grids["long_cv"], root),
                  elapsed_seconds=time.perf_counter()-started)
    verify_baseline(result, record)
    save(path, result)
    return result


def evaluate(workers):
    verify()
    selection_path = OUT/"selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["protocol_sha256"] == parent.digest(OUT/"protocol.json")
    assert all(parent.digest(Path(p)) == h for p, h in selection["inputs_sha256"].items())
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for result in pool.map(evaluate_root, selection["failures"]+selection["protection"]):
            print("PAS_FAILURE_CONFIRM", result["index"], result["case"], "tick", result["tick"],
                  {k: v["actual"]["terminal"] for k, v in result["arms"].items()}, flush=True)
    verify()
    save(OUT/"completion.json", dict(selection_sha256=parent.digest(selection_path),
                evaluation_wall_seconds=time.perf_counter()-started,
                result_sha256={str(p): parent.digest(p) for p in sorted((OUT/"evaluation").glob("*.json"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "screen", "evaluate"))
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 4))
    args = parser.parse_args()
    {"prepare": prepare, "screen": lambda: screen(args.workers),
     "evaluate": lambda: evaluate(args.workers)}[args.mode]()
