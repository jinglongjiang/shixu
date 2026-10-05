"""Read-only admission headroom on archived native trajectories.

This is not a new navigator. Oracle reveals are offline interventions; neither
they nor longer-retention queries update the real tracker or policy history.
"""

import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState
from experiments.decision_state import ARCHIVE, exact_snapshot
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import archive_parity, root_commands, run_branches, save_json
from experiments.occlusion import source_hash
from shixu.features import window
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/temporal_problem_audit")
DEADLINE = "2026-10-05T10:23:42+00:00"
RULES = {
    "start_utc": "2026-10-05T02:23:42+00:00", "deadline_utc": DEADLINE,
    "scope": "New missing-actor annotation of192 existing CV419 evaluation trajectories. No new training, "
             "scenarios, noise, rewards, action support or dynamics.",
    "strata": "never_seen: absent from lifetime legal track store; expired: missing >2s but a real "
              "measurement remains within the existing24-frame window (age<=5.75s).",
    "selection": "First t>=2s per episode and stratum with true missing-actor centre distance<=3m. "
                 "Truth selects events only; this is an enriched offline cohort, not a deployable trigger. "
                 "At most12 roots per stratum, round-robin across N=5/10/20, circle/square, "
                 "failure/success; ascending case within cell. No action or new-return selection.",
    "simple": "Read last real measurement from the same legal24-frame history; extend its CV admission "
              "to5.75s without altering measurements. Cannot infer never-seen humans.",
    "oracle": "Separately reveal current p/v/r of expired-window or never-seen actors as an offline "
              "sensor reveal. Added tokens have age0/observed=true but no fabricated past. "
              "Do not reveal goals, ORCA state or future. Do not commit any reveal to real memory.",
    "consumer": "Frozen CV419 history-conditioned value/lookahead/risk/smoothing. All root queries "
                "retain its original history. Extra actors/ages can cause distribution shift; a negative "
                "result applies only to this frozen consumer and cohort, not every memory mechanism.",
    "continuation": "One selected smoothed root command for0.25s, then identical original native policy "
                    "under legal sensing until the original terminal condition. Original reward/gamma. "
                    "Deduplicate identical commands; verify full archived current continuation.",
    "kill_test": "Simple admission reproducing oracle benefits weakens a complex memory claim. "
                 "No oracle action/outcome benefit closes this consumer test only. An oracle gain is "
                 "positive headroom, not proof of lawful history recoverability or KDA superiority.",
}


def expired_measurements(snapshot, history, retention=2., step=.25):
    """No truth access: historical measurements and timestamps only."""
    now = snapshot["time"]
    result = []
    for _, key, measurement, stamp in snapshot["tracks"]:
        age = now - stamp
        if retention < age <= (len(history) - 1) * step and 1 + key < history.shape[1]:
            row = len(history) - 1 - round(age / step)
            if row >= 0 and history[row, 1 + key, 10] > 0 and history[row, 1 + key, 12] > 0:
                result.append((key, measurement, age))
    return result


def append_actors(state, rows):
    """Append copies, retaining association keys without changing the source."""
    humans, keys, observed, ages = map(list, (state.human_states, state.track_ids, state.observed, state.ages))
    for key, measurement, measured, age in rows:
        if key in keys:
            raise ValueError("Admission cannot replace an already active actor")
        humans.append(ObservableState(*measurement))
        keys.append(key)
        observed.append(measured)
        ages.append(age)
    return state._replace(human_states=humans, track_ids=tuple(keys), observed=tuple(observed),
                          ages=tuple(ages), track_count=max([state.track_count] + [k + 1 for k in keys]))


def query_states(root):
    state, saved = root["state"], root["snapshot"]
    expired = expired_measurements(saved, root["history"])
    simple_rows = []
    for key, measurement, age in expired:
        projected = measurement.copy()
        projected[:2] += age * projected[2:4]
        simple_rows.append((key, projected, False, age))
    key_to_human = {key: index for index, key, _, _ in saved["tracks"]}
    expired_rows = [(key, saved["humans"][key_to_human[key]][:5], True, 0.) for key, _, _ in expired]
    known = set(key_to_human.values())
    unseen = [index for index in range(len(saved["humans"])) if index not in known]
    unseen_rows = [(state.track_count + offset, saved["humans"][index][:5], True, 0.)
                   for offset, index in enumerate(unseen)]
    return dict(current=state, legal_long_cv=append_actors(state, simple_rows),
                oracle_expired=append_actors(state, expired_rows),
                oracle_unseen=append_actors(state, unseen_rows))


def choose_roots(candidates, cap=12):
    result = []
    for stratum in ("expired", "never_seen"):
        buckets = defaultdict(list)
        for root in candidates:
            if root["stratum"] == stratum:
                r = root["episode"]
                buckets[(r["people"], r["geometry"], root["group"])].append(root)
        for rows in buckets.values():
            rows.sort(key=lambda root: root["episode"]["case"])
        count = 0
        while count < cap and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and count < cap:
                    result.append(buckets[key].pop(0))
                    count += 1
    return result


def prepare():
    started = time.perf_counter()
    checkpoint = weights(419, "cv")
    model, cfg = load(checkpoint, "cpu")
    policy = ValuePolicy(model, cfg)
    protocol = dict(rules=RULES, archive_sha256=digest(ARCHIVE), checkpoint_sha256=digest(checkpoint),
                    source_sha256=source_hash(), script_sha256=digest(Path(__file__)))
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text())
        if {k: v for k, v in frozen.items() if k != "script_sha256"} != {
                k: v for k, v in protocol.items() if k != "script_sha256"}:
            raise ValueError("Frozen rules or scientific inputs changed")
        save_json(OUT / "preparation_implementation.json", dict(script_sha256=protocol["script_sha256"],
            frozen_script_sha256=frozen["script_sha256"],
            correction="Use Human.get_position() instead of nonexistent Human.position; no result existed."))
        protocol = frozen
    else:
        save_json(protocol_path, protocol)
    episodes, candidates = [], []
    for record in json.loads(ARCHIVE.read_text())["episodes"]:
        env = environment(cfg, policy, record["geometry"], record["people"])
        env.reset(options={"test_case": record["case"]})
        observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
        frames, rewards, roots, counts, event_ticks = [], [], {}, Counter(), defaultdict(list)
        path, minimum = 0., 1e6
        for tick, command in enumerate(record["actions"]):
            state = observer.observe(env)
            frames.append(policy.encode(state))
            history = window(frames, 24, "zero")
            kinds = defaultdict(list)
            for i, human in enumerate(env.humans):
                if human not in observer.tracks:
                    kind = "never_seen"
                else:
                    age = env.global_time - observer.tracks[human][2]
                    kind = "expired" if 2. < age <= 5.75 else "older_expired" if age > 5.75 else "active"
                kinds[kind].append(i)
            for kind in ("never_seen", "expired", "older_expired"):
                close = [i for i in kinds[kind] if np.linalg.norm(np.asarray(env.humans[i].get_position())
                         - state.self_state.position) <= 3.]
                counts[kind + "_person_frames"] += len(kinds[kind])
                counts[kind + "_frames"] += bool(kinds[kind])
                counts[kind + "_near_frames"] += bool(close)
                if close:
                    event_ticks[kind].append(tick)
                if kind != "older_expired" and close and tick >= 8 and kind not in roots:
                    roots[kind] = dict(stratum=kind, group="success" if record["terminal"] == "reach_goal" else "failure",
                        episode={k: record[k] for k in ("people", "geometry", "case", "terminal")},
                        tick=tick, state=state, history=history, snapshot=exact_snapshot(env, observer),
                        target_human_indices=close, previous=record["actions"][tick - 1],
                        archived_tail_actions=record["actions"][tick:],
                        archived_remaining_seconds=(len(record["actions"]) - tick) * .25)
            old = np.asarray(env.robot.get_position())
            _, reward, done, truncated, info = env.step(ActionXY(*command))
            path += float(np.linalg.norm(np.asarray(env.robot.get_position()) - old))
            minimum = min(minimum, info["dmin"])
            rewards.append(reward)
            if (done or truncated) != (tick == len(record["actions"]) - 1):
                raise ValueError("Archive episode length changed")
        if (info["event"] != record["terminal"] or abs(path - record["path"]) > 1e-6
                or abs(minimum - record["minimum_clearance"]) > 1e-6
                or abs(sum(rewards) - record["return_"]) > 1e-6
                or abs(env.global_time - record["navigation_time"]) > 1e-8):
            raise ValueError("Archived world replay parity failed")
        for root in roots.values():
            root["archived_rewards"] = rewards[root["tick"]:]
            candidates.append(root)
        episodes.append(dict(**{k: record[k] for k in ("people", "geometry", "case", "terminal")},
            frames=len(frames), person_frames=len(frames) * record["people"], counts=dict(counts),
            event_ticks=dict(event_ticks), eligible_roots={k: r["tick"] for k, r in roots.items()}, archive_parity=True))
        print("MISSING_ANNOTATION", record["people"], record["geometry"], record["case"],
              record["terminal"], episodes[-1]["eligible_roots"], flush=True)
    selected = choose_roots(candidates)
    save_json(OUT / "natural_events.json", dict(protocol=protocol, episodes=episodes,
              eligible_roots=len(candidates), elapsed_seconds=time.perf_counter() - started))
    torch.save(dict(roots=selected, protocol=protocol), OUT / "cohort.pt")
    save_json(OUT / "cohort_hash.json", dict(sha256=digest(OUT / "cohort.pt")))
    print("PREPARED", len(selected), "roots", time.perf_counter() - started, flush=True)


@torch.inference_mode()
def evaluate(device, limit=None):
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    if digest(OUT / "cohort.pt") != json.loads((OUT / "cohort_hash.json").read_text())["sha256"]:
        raise ValueError("Cohort changed")
    if data["protocol"]["source_sha256"] != source_hash():
        raise ValueError("Scientific core changed")
    if data["protocol"]["checkpoint_sha256"] != digest(weights(419, "cv")):
        raise ValueError("Consumer weights changed")
    model, cfg = load(weights(419, "cv"), device)
    for index, root in enumerate(data["roots"]):
        if limit is not None and index >= limit:
            break
        path = OUT / "results" / f"root_{index:02d}.json"
        if path.exists():
            continue
        if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(DEADLINE) - dt.timedelta(minutes=30):
            break
        started = time.perf_counter()
        commands, scores, _ = root_commands(root, model, cfg, device)
        parent_index = int(scores.argmax())
        baseline = run_branches(root, model, cfg, device, [commands[parent_index]])[0]
        parity = archive_parity(root, baseline, cfg)
        if not parity["passed"]:
            raise ValueError("Archived native continuation changed: " + str(parity))
        choices = {}
        for name, state in query_states(root).items():
            policy = ValuePolicy(model, cfg, device)
            policy.history.extend(root["history"][:-1])
            before = time.perf_counter()
            score = policy.score(state)
            chosen = int(score.argmax())
            choices[name] = dict(action=chosen, scores=score.tolist(), scoring_seconds=time.perf_counter() - before,
                admitted_ids=list(state.track_ids), added_ids=sorted(set(state.track_ids) - set(root["state"].track_ids)))
        if choices["current"]["action"] != parent_index:
            raise ValueError("Current query parity failed")
        realized = {parent_index: baseline}
        other = sorted({c["action"] for c in choices.values()} - {parent_index})
        realized.update(zip(other, run_branches(root, model, cfg, device, [commands[i] for i in other])))
        for choice in choices.values():
            chosen = choice["action"]
            actual = realized[chosen]
            choice.update(actual=actual, delta_return=actual["return_"] - baseline["return_"],
                rescued=baseline["terminal"] != "reach_goal" and actual["terminal"] == "reach_goal",
                damaged=baseline["terminal"] == "reach_goal" and actual["terminal"] != "reach_goal",
                command_changed=not np.allclose(commands[chosen], commands[parent_index], atol=1e-8, rtol=0))
        save_json(path, dict(root=index, stratum=root["stratum"], group=root["group"], episode=root["episode"],
            tick=root["tick"], choices=choices, archive_parity=parity, elapsed_seconds=time.perf_counter() - started,
            helper_sha256=digest(Path(__file__)), device=device, torch=torch.__version__))
        print("ADMISSION_BLOCK", index, root["stratum"], root["episode"], root["tick"],
              {n: (c["action"], c["actual"]["terminal"], round(c["delta_return"], 6)) for n, c in choices.items()},
              "seconds", round(time.perf_counter() - started, 2), flush=True)


@torch.inference_mode()
def action_headroom(device):
    """Outcome-only oracle over the unchanged support, not a new selector."""
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    if digest(OUT / "cohort.pt") != json.loads((OUT / "cohort_hash.json").read_text())["sha256"]:
        raise ValueError("Cohort changed")
    if data["protocol"]["source_sha256"] != source_hash():
        raise ValueError("Scientific core changed")
    selected, used = [], set()
    for index, root in enumerate(data["roots"]):
        episode = root["episode"]
        key = (episode["people"], episode["geometry"], episode["case"])
        if root["group"] == "failure" and key not in used and len(selected) < 4:
            selected.append(index)
            used.add(key)
    rules = dict(selection="First4 failure roots in frozen cohort order, distinct episodes; "
                 "no selection on intervention benefit. All80 native commands, no new actions.",
                 roots=selected, source_sha256=source_hash(), checkpoint_sha256=digest(weights(419, "cv")),
                 script_sha256=digest(Path(__file__)),
                 interpretation="Q under the fixed native continuation, not Q*. A rescue proves "
                 "local action headroom, not that history can predict the better action. "
                 "No rescue limits a single root-action intervention, not sustained memory use.")
    path = OUT / "action_headroom_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != rules:
            raise ValueError("Action headroom protocol changed")
    else:
        save_json(path, rules)
    model, cfg = load(weights(419, "cv"), device)
    for index in selected:
        path = OUT / "action_headroom" / f"root_{index:02d}.json"
        if path.exists():
            continue
        started = time.perf_counter()
        root = data["roots"][index]
        commands, scores, _ = root_commands(root, model, cfg, device)
        if len(commands) != 80:
            raise ValueError("Native action support changed")
        rows = []
        for start in range(0, len(commands), 8):
            rows.extend(run_branches(root, model, cfg, device, commands[start:start + 8]))
            print("ACTION_HEADROOM", index, len(rows), flush=True)
        native = rows[int(scores.argmax())]
        parity = archive_parity(root, native, cfg)
        if not parity["passed"]:
            raise ValueError("Headroom continuation parity failed")
        save_json(path, dict(root=index, episode=root["episode"], tick=root["tick"],
            native_action=int(scores.argmax()), commands=commands.tolist(), outcomes=rows,
            archive_parity=parity, elapsed_seconds=time.perf_counter() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate", "headroom"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "evaluate":
        evaluate(args.device, args.limit)
    else:
        action_headroom(args.device)
