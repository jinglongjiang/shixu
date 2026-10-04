"""Current retained-state correction, not another future-prediction bridge.

Reconstructs existing archived commands. Hidden simulator states are labels
and offline interventions only; the deployable baseline uses measured history.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import (archive_parity, root_commands,
                                              run_branches, save_json)
from experiments.forecast_control_diagnostic import digest
from experiments.occlusion import source_hash
from shixu.features import window
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/decision_state_rebuild")
ARCHIVE = Path("outputs/forecast_control_b_fresh_cv/419/cv/result.json")
START = "2026-10-04T18:48:39+00:00"
DEADLINE = "2026-10-05T02:48:39+00:00"
RULES = {
    "start_utc": START, "deadline_utc": DEADLINE,
    "scope": "New annotation of existing archived commands, not original online RL replay or new episodes.",
    "population": "All192 existing fresh-CV419 evaluation episodes for natural-event frequencies.",
    "cases": "All25 existing failed episodes and25 distinct successful cases, matched by population/geometry "
             "and nearest case index without replacement. No new outcome-dependent selection.",
    "root": "First t>=2s with a retained hidden actor at legal estimated centre distance<=2m. "
            "One root per selected case; no truth/error/action/gain threshold. Cases lacking such a root are recorded.",
    "interface": "Replace only current position/velocity of already-known, retained hidden actors before "
                 "native candidate construction. Radius/visibility/age/presence/expiry and action support stay fixed.",
    "consumer": "Frozen fresh-CV419 native contextual-GRU value/lookahead/risk/smoothing, including original23 "
                "history frames, legal current velocities/neighbours/ages/measurement flags.",
    "simple": "OLS velocity trend from last3 real visible measurements in24-frame window; acceleration "
              "clipped componentwise to[-1,1]m/s2 and output speed capped at1.5m/s, as in earlier legal-CA baseline. "
              "p'=p_CV+0.5*a*age^2;v'=v_last+a*age. Not a new method or tuned to this cohort.",
    "truth": "Offline current p/v only for retained hidden actors. Never new actors/goals/ORCA state; "
             "never written to real tracker/history. Not an optimally trained consumer or a family upper bound.",
    "execution": "Only selected root command executes for0.25s; same original legal history and native "
                 "policy continue to original terminal. Root state corrections do not persist in real memory.",
    "gate": "Quantify state error, action and actual original-reward outcomes together. No full IL/RL. "
            "Train local GRU/KDA state heads only if a concrete action/outcome deficit is supported; "
            "no gain closes this frozen physical-state consumer, not all latent representations.",
}


def history_correction(state, history):
    """Read real measurements; do not feed CV pseudo-measurements into a fit."""
    humans = list(state.human_states)
    for i, (key, measured, age) in enumerate(zip(state.track_ids, state.observed, state.ages)):
        if measured or age <= 0:
            continue
        rows = history[:, 1 + key]
        ticks = np.flatnonzero((rows[:, 10] > 0) & (rows[:, 12] > 0))[-3:]
        if len(ticks) < 2:
            continue
        times = (ticks - ticks[-1]) * .25
        design = np.column_stack((times, np.ones_like(times)))
        acceleration = np.linalg.lstsq(design, rows[ticks, 3:5], rcond=None)[0][0].clip(-1, 1)
        human = humans[i]
        old = np.array([human.px, human.py, human.vx, human.vy, human.radius], dtype=np.float64)
        old[:2] += .5 * acceleration * age ** 2
        old[2:4] += acceleration * age
        old[2:4] *= min(1., 1.5 / max(np.linalg.norm(old[2:4]), 1e-12))
        humans[i] = ObservableState(*old)
    return state._replace(human_states=humans)


def truth_correction(state, labels):
    humans = [human if measured else ObservableState(*labels[key][:4], human.radius)
              for key, human, measured in zip(state.track_ids, state.human_states, state.observed)]
    return state._replace(human_states=humans)


def exact_snapshot(env, observer):
    fields = ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta")
    states = [np.array([getattr(a, f) for f in fields], dtype=np.float64) for a in [env.robot] + env.humans]
    indices = {human: i for i, human in enumerate(env.humans)}
    return dict(robot=states[0], humans=states[1:], time=env.global_time,
        preferred=[None if h.policy._last_pref_vel is None else h.policy._last_pref_vel.copy() for h in env.humans],
        tracks=[(indices[h], key, np.array([m.px, m.py, m.vx, m.vy, m.radius], dtype=np.float64), stamp)
                for h, (key, m, stamp) in observer.tracks.items()])


def matched_cases(episodes):
    selected, used = [], set()
    failures = sorted((r for r in episodes if r["terminal"] != "reach_goal"),
                      key=lambda r: (r["people"], r["geometry"], r["case"]))
    for failure in failures:
        cell = (failure["people"], failure["geometry"])
        available = [r for r in episodes if (r["people"], r["geometry"]) == cell
                     and r["terminal"] == "reach_goal" and (*cell, r["case"]) not in used]
        success = min(available, key=lambda r: (abs(r["case"] - failure["case"]), r["case"]))
        used.add((*cell, success["case"]))
        selected.extend((dict(group="failure", **{k: failure[k] for k in ("people", "geometry", "case")}),
                         dict(group="success_guard", **{k: success[k] for k in ("people", "geometry", "case")})))
    return selected


def distribution(values):
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                p50_p90_p99=np.quantile(values, [.5, .9, .99]).tolist() if values else None)


def prepare():
    started = time.perf_counter()
    archive = json.loads(ARCHIVE.read_text())
    checkpoint = weights(419, "cv")
    model, cfg = load(checkpoint, "cpu")
    policy = ValuePolicy(model, cfg)
    selection = matched_cases(archive["episodes"])
    protocol = dict(rules=RULES, selected_cases=selection, archive_sha256=digest(ARCHIVE),
                    checkpoint_sha256=digest(checkpoint), source_sha256=source_hash())
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Selection changed after preparation started")
    else:
        save_json(protocol_path, protocol)
    selected = {(r["people"], r["geometry"], r["case"]): r["group"] for r in selection}
    episodes, roots = [], []
    for record in archive["episodes"]:
        env = environment(cfg, policy, record["geometry"], record["people"])
        env.reset(options={"test_case": record["case"]})
        observer = OccludedTracks(2.)
        frames, rewards, path, minimum, root = [], [], 0., 1e6, None
        hidden_errors, hidden_v_errors, ca_errors, ca_v_errors = [], [], [], []
        near_errors, near_v_errors, visible_errors = [], [], []
        hidden_frames, near_frames, person_frames, reentries = 0, 0, 0, 0
        previous_visible, seen = set(), set()
        for tick, command in enumerate(record["actions"]):
            state = observer.observe(env)
            frames.append(policy.encode(state))
            history = window(frames, 24, "zero")
            simple = history_correction(state, history)
            labels = {entry[0]: np.array([h.px, h.py, h.vx, h.vy, h.radius], dtype=np.float64)
                      for h, entry in observer.tracks.items()}
            visible = {key for key, measured in zip(state.track_ids, state.observed) if measured}
            reentries += len((visible - previous_visible) & seen)
            seen.update(visible)
            previous_visible = visible
            person_frames += len(env.humans)
            has_hidden, has_near = False, False
            for key, human, corrected, measured in zip(state.track_ids, state.human_states,
                                                       simple.human_states, state.observed):
                error = float(np.linalg.norm(np.asarray(human.position) - labels[key][:2]))
                v_error = float(np.linalg.norm(np.asarray(human.velocity) - labels[key][2:4]))
                if measured:
                    visible_errors.append(error + v_error)
                    if error + v_error > 1e-10:
                        raise ValueError("Visible current state is not an immediate measurement")
                    continue
                has_hidden = True
                hidden_errors.append(error)
                hidden_v_errors.append(v_error)
                ca_errors.append(float(np.linalg.norm(np.asarray(corrected.position) - labels[key][:2])))
                ca_v_errors.append(float(np.linalg.norm(np.asarray(corrected.velocity) - labels[key][2:4])))
                if np.linalg.norm(np.asarray(human.position) - state.self_state.position) <= 2:
                    has_near = True
                    near_errors.append(error)
                    near_v_errors.append(v_error)
            hidden_frames += has_hidden
            near_frames += has_near
            key = (record["people"], record["geometry"], record["case"])
            if key in selected and root is None and tick >= 8 and has_near:
                root = dict(group=selected[key], episode={k: record[k] for k in ("people", "geometry", "case", "terminal")},
                    tick=tick, state=state, snapshot=exact_snapshot(env, observer), history=history,
                    previous=record["actions"][tick - 1], truth_current=labels,
                    archived_tail_actions=record["actions"][tick:],
                    archived_remaining_seconds=(len(record["actions"]) - tick) * .25)
            old = np.asarray(env.robot.get_position())
            _, reward, done, truncated, info = env.step(ActionXY(*command))
            path += float(np.linalg.norm(np.asarray(env.robot.get_position()) - old))
            minimum = min(minimum, info["dmin"])
            rewards.append(reward)
            if (done or truncated) != (tick == len(record["actions"]) - 1):
                raise ValueError("Archived episode termination differs")
        total_return = sum(rewards)
        if (info["event"] != record["terminal"] or abs(path - record["path"]) > 1e-6
                or abs(env.global_time - record["navigation_time"]) > 1e-8
                or abs(minimum - record["minimum_clearance"]) > 1e-6
                or abs(total_return - record["return_"]) > 1e-6):
            raise ValueError("Native archived trajectory parity failed")
        if root is not None:
            root["archived_rewards"] = rewards[root["tick"]:]
            roots.append(root)
        episodes.append(dict(**{k: record[k] for k in ("people", "geometry", "case", "terminal")},
            control_frames=len(frames), person_frames=person_frames, hidden_control_frames=hidden_frames,
            near_hidden_control_frames=near_frames, reentries=reentries, archive_parity=True,
            visible_error=distribution(visible_errors), hidden_position_error=hidden_errors,
            hidden_velocity_error=hidden_v_errors, simple_position_error=ca_errors,
            simple_velocity_error=ca_v_errors, near_hidden_position_error=near_errors,
            near_hidden_velocity_error=near_v_errors,
            root_tick=None if root is None else root["tick"], selected_group=selected.get(key)))
        print("ANNOTATED", *key, record["terminal"], "near-hidden", near_frames, "root",
              None if root is None else root["tick"], flush=True)
    save_json(OUT / "natural_events.json", dict(protocol=protocol, episodes=episodes,
              elapsed_seconds=time.perf_counter() - started))
    torch.save(dict(roots=roots, protocol=protocol,
                    config={s: dict(cfg[s]) for s in cfg.sections()}), OUT / "cohort.pt")
    save_json(OUT / "cohort_hash.json", dict(sha256=digest(OUT / "cohort.pt")))
    print("PREPARED", len(roots), "roots", "seconds", round(time.perf_counter() - started, 2), flush=True)


@torch.inference_mode()
def evaluate(device):
    data = torch.load(OUT / "cohort.pt", map_location="cpu", weights_only=False)
    if digest(OUT / "cohort.pt") != json.loads((OUT / "cohort_hash.json").read_text())["sha256"]:
        raise ValueError("Frozen cohort changed")
    if data["protocol"]["source_sha256"] != source_hash():
        raise ValueError("Frozen navigation code changed")
    checkpoint = weights(419, "cv")
    if digest(checkpoint) != data["protocol"]["checkpoint_sha256"]:
        raise ValueError("Frozen weights changed")
    model, cfg = load(checkpoint, device)
    for index, root in enumerate(data["roots"]):
        path = OUT / "results" / f"root_{index:02d}.json"
        if path.exists():
            continue
        if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(DEADLINE) - dt.timedelta(minutes=30):
            print("REPORT_RESERVE_REACHED", index, flush=True)
            break
        started = time.perf_counter()
        commands, scores, _ = root_commands(root, model, cfg, device)
        parent_index = int(scores.argmax())
        baseline = run_branches(root, model, cfg, device, [commands[parent_index]])[0]
        parity = archive_parity(root, baseline, cfg)
        if not parity["passed"]:
            raise ValueError("Original policy no longer reproduces archived continuation")
        states = dict(current=root["state"], simple=history_correction(root["state"], root["history"]),
                      truth=truth_correction(root["state"], root["truth_current"]))
        selected, choices = {}, {}
        for name, state in states.items():
            scoring_start = time.perf_counter()
            policy = ValuePolicy(model, cfg, device)
            policy.history.extend(root["history"][:-1])
            score = policy.score(state)
            chosen = int(score.argmax())
            selected[name] = chosen
            _, _, clearance = policy.aligned_candidates(state)
            hidden = [i for i, observed in enumerate(state.observed) if not observed]
            choices[name] = dict(action=chosen, scores=score.tolist(), clearances=clearance.tolist(),
                scoring_seconds=time.perf_counter() - scoring_start,
                hidden_position_error_mean=float(np.mean([np.linalg.norm(np.asarray(state.human_states[i].position)
                    - root["truth_current"][state.track_ids[i]][:2]) for i in hidden])),
                hidden_velocity_error_mean=float(np.mean([np.linalg.norm(np.asarray(state.human_states[i].velocity)
                    - root["truth_current"][state.track_ids[i]][2:4]) for i in hidden])))
        if selected["current"] != parent_index:
            raise ValueError("Current scoring parity failed")
        realized = {parent_index: baseline}
        other = sorted(set(selected.values()) - {parent_index})
        realized.update(zip(other, run_branches(root, model, cfg, device, [commands[i] for i in other])))
        for name, chosen in selected.items():
            actual = realized[chosen]
            choices[name].update(actual=actual, delta_return=actual["return_"] - baseline["return_"],
                rescued=baseline["terminal"] != "reach_goal" and actual["terminal"] == "reach_goal",
                damaged=baseline["terminal"] == "reach_goal" and actual["terminal"] != "reach_goal",
                command_changed=not np.allclose(commands[chosen], commands[parent_index], atol=1e-8, rtol=0))
        save_json(path, dict(root=index, group=root["group"], episode=root["episode"], tick=root["tick"],
                  choices=choices, archive_parity=parity, elapsed_seconds=time.perf_counter() - started,
                  torch=torch.__version__, device=device, helper_sha256=digest(Path(__file__))))
        print("STATE_BLOCK", index, root["group"], root["episode"], root["tick"],
              {name: (c["action"], c["actual"]["terminal"], round(c["delta_return"], 6))
               for name, c in choices.items()}, "seconds", round(time.perf_counter() - started, 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare() if args.mode == "prepare" else evaluate(args.device)
