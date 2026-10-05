"""Frozen proposal state machine, data generation and research contract."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from crowd_sim.envs.utils.action import ActionXY

from autoresearch_nav.features import NAMES, build
from experiments.commit_advantage_probe import branch
from experiments.decision_state import exact_snapshot
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load, weights
from experiments.multihorizon_control import save_json
from shixu.commitment import low_progress, native_blocked
from shixu.commitment_release import AllUnsafeReleasePolicy
from shixu.features import window
from shixu.observations import OccludedTracks
from shixu.runner import environment


OUT = Path("outputs/autonav-overnight")
SEEDS = (419, 443, 467, 491)
CELLS = tuple((n, g) for n in (5, 10, 20) for g in ("circle", "square"))
CASES = tuple(range(130000, 130512))
FINAL = tuple(range(140000, 140032))
DEV = (("dev1", Path("outputs/online-action-commitment-v01"), tuple(range(87000, 87032))),
       ("dev2", Path("outputs/online-commit-advantage"), tuple(range(99000, 99032))))
RULES = dict(
    scope="Only method.py may change after freeze. Gate only accepts/rejects originalV0.1 low-progress "
          "proposal; never generates navigation actions or initiates elsewhere. Max2s, release, Parent, "
          "reward/filter/observer/history/execution/training contracts unchanged.",
    rejection="Consume the original one-shot initiation opportunity on accept AND reject. Rearm only "
              "when original low-progress clears then recurs. Hold steps identical toV0.1.",
    data="Once:512 new cases130000..130511 x4Parents, native5circle only. Collect every proposal on "
         "a reject-all/Parent trajectory, without looking at terminal/method results. Restore world/history/RNG "
         "for Parent forever vs oneV0.1 commit<=8ticks then Parent forever. Original discount/reward. "
         "Train130000..130383, dev130384..130511; allseeds/allroots of samecase stay together. "
         "No square/10/20 training. All future fields labels only, no new navigation training.",
    offline="At most60 commits. Episode-balanced selected mean deltaQ>0; accept coverage>=5%; "
            "positive y>0.005 coverage>=10%; retain>=25% of labeled failure->success opportunity weight; "
            "damage<=75% of all-accept weighted damage; extra collision<=all-accept and<=0.5pp. "
            "Require positive/rescue label support; reject never-commit. Keep/discard not method verdict.",
    dev="Max8 method commits enter olddev1; max3 enter olddev2. Eachblock requires square/circle "
        "Parent-success damage individually reduced>=25% relative fixedV0.1, >=50% squareV0.1 "
        "rescue retained and>=50% totalV0.1rescue retained, >=4qualified square rescue caseIDs, "
        "square low-timeouts<=Parent and circle/square collision<=Parent+1pp. No new thresholds. "
        "Rank eligible by minimum square success net gain across availableblocks, then lower "
        "combineddamage, then fewerparameters, then higherQ. Same effects prefer simpler.",
    final="Exactly one chosen candidate only after passing BOTH devblocks. Lock code/weights. "
          "Fresh140000..140031 x4Parents x6cells xParent/V0.1/Candidate=2304episodes. "
          "Use exact old nineGates. Pass=METHOD_CANDIDATE_FOUND; fail=FINAL_CONFIRMATION_FAILED. "
          "No final feedback/refit/newcandidate. No proposal expansion.",
    models="Rules, linear/logistic or MLP max2hidden layers,width<=32. No temporal backbone, "
           "newcritic/reward/action/duration/filter. Train-only normalization. Fixed CPUtorch1thread. "
           "Every experiment committed; negative artifacts retained. No threshold sweep.")
_DEVICE = None
_MODELS = {}


def read(path):
    return json.loads(path.read_text())


def bindings():
    files = [p for directory in ("shixu", "vendor/crowd_sim") for p in sorted(Path(directory).rglob("*.py"))]
    files += [Path("autoresearch_nav")/name for name in ("prepare.py", "features.py", "evaluate.py", "program.md")]
    files += [Path("experiments")/name for name in ("commit_advantage_probe.py", "decision_state.py", "multihorizon_control.py",
        "forecast_evidence.py", "forecast_control_diagnostic.py", "repeatable_defect_audit.py", "online_action_commitment_v0.py")]
    return {str(p): digest(p) for p in files}


def verify():
    p = read(OUT/"protocol.json")
    if p["rules"] != RULES or p["bindings"] != bindings():
        raise ValueError("Frozen harness/core changed")
    for c in p["checkpoints"].values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Frozen Parent changed")
    return p


def freeze(device):
    p = read(Path("outputs/online-action-commitment-v01/protocol.json"))
    search = subprocess.run(["rg", "-l", "--glob", "*.json",
        r'"case"\s*:\s*(130[0-4][0-9][0-9]|13050[0-9]|13051[01]|1400(?:0[0-9]|[12][0-9]|3[01]))\b|"(?:case_start|eval_case_start|test_case_start)"\s*:\s*(130000|140000)\b',
        "outputs"], capture_output=True, text=True)
    if search.returncode != 1:
        raise ValueError("Cases already used: "+search.stdout+search.stderr)
    save_json(OUT/"protocol.json", dict(rules=RULES, train_cases=CASES[:384], dev_cases=CASES[384:],
        final_cases=FINAL, seeds=SEEDS, cells=CELLS, checkpoints=p["checkpoints"], bindings=bindings(), device=device,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        unused_check=dict(exit_code=search.returncode, matches=search.stdout),
        inherited_nine_gate_code_sha256=digest(Path("experiments/online_action_commitment_v0.py"))))


class ProposalPolicy(AllUnsafeReleasePolicy):
    """Frozen V0.1 with a boolean veto at its original initiation point."""

    def __init__(self, model, config, gate=None, device="cpu"):
        self.gate = gate
        super().__init__(model, config, device)

    def reset(self):
        super().reset()
        self.recent_commands, self.recent_grids = deque(maxlen=8), deque(maxlen=8)
        self.last_features = None

    @torch.inference_mode()
    def predict(self, state, epsilon=0.0):
        if self.phase == "train" or epsilon != 0:
            raise ValueError("Frozen evaluation wrapper only")
        robot = state.self_state
        distance = float(np.hypot(robot.px-robot.gx, robot.py-robot.gy))
        if distance < self.config.getfloat("robot", "success_radius"):
            return ActionXY(0, 0)
        self.distances.append(distance)
        trigger = low_progress(self.distances)
        progress = self.distances[0]-distance if len(self.distances) == 9 else None
        if self.active_grid is None and not trigger:
            self.armed = True
        commands = self.candidate_actions()
        _, _, clearances = self.aligned_candidates(state, commands)
        margin = self.config.getfloat("eval_protocol", "safety_margin")
        blocked = native_blocked(clearances, margin)
        no_safe = not bool((clearances >= margin).any())
        released, started, offered, accepted = None, False, False, None
        self.last_features = None
        if self.active_grid is not None and blocked[self.active_grid]:
            self.active_grid, self.remaining = None, 0
            released = "safety-blocked"
        proposal = None
        if self.active_grid is None:
            scores = self.score(state, commands)
            proposal = int(np.argmax(scores))
            if trigger and self.armed and released is None and not blocked[proposal]:
                self.last_features = build(self, state, commands, scores, clearances, proposal)
                self.proposal_index = proposal
                offered = True
                accepted = True if self.gate is None else bool(self.gate(self.last_features.copy()))
                self.armed = False
                if accepted:
                    self.active_grid, self.remaining = proposal, 8
                    self.starts += 1
                    started = True
        if self.active_grid is not None and no_safe:
            self.active_grid, self.remaining = None, 0
            released = "all-unsafe"
            if proposal is None:
                proposal = int(np.argmax(self.score(state, commands)))
        np.random.random()
        held = self.active_grid is not None
        index = self.active_grid if held else proposal
        selected = commands[index]
        self.history.append(self.encode(state))
        self.last_action = selected
        self.recent_commands.append(selected)
        self.recent_grids.append(index)
        if held:
            self.held_steps += 1
            self.remaining -= 1
            if self.remaining == 0:
                self.active_grid = None
                released = "budget-completed"
        self.last_decision = dict(tick=self.tick, time=self.tick*.25, distance=distance,
            past_two_second_progress=progress, trigger=bool(trigger), proposal=proposal,
            selected_grid=int(index), held=held, started=started, release=released,
            predicted_cv_clearance=float(clearances[index]), blocked=bool(blocked[index]),
            no_margin_safe_candidate=no_safe, offered=offered, accepted=accepted)
        self.tick += 1
        return selected


def init_worker(device):
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


@torch.inference_mode()
def collect(task):
    seed, case, sha = task
    path = OUT/"data"/f"{seed}-{case}.json"
    if path.exists():
        r = read(path)
        if r["protocol_sha256"] != sha or digest(Path(r["inputs"])) != r["inputs_sha256"]:
            raise ValueError("Cached input differs")
        return r
    if seed not in _MODELS:
        _MODELS[seed] = load(weights(seed, "cv"), _DEVICE)
    model, cfg = _MODELS[seed]
    roots, commands, rewards = [], [], []
    def reject(features):
        roots.append(dict(episode=dict(people=5, geometry="circle", case=case), tick=len(commands),
            state=state, snapshot=exact_snapshot(env, observer),
            history=window(list(policy.history)+[policy.encode(state)], 24, "zero"),
            previous=policy.last_action, grid=policy.proposal_index,
            features=features, rng=np.random.get_state()))
        return False
    policy = ProposalPolicy(model, cfg, reject, _DEVICE)
    env = environment(cfg, policy, "circle", 5)
    env.reset(options={"test_case":case})
    observer = OccludedTracks(cfg.getfloat("observation", "retention_seconds"))
    while True:
        state = observer.observe(env)
        command = policy.predict(state)
        _, reward, done, truncated, event = env.step(command)
        commands.append(list(command))
        rewards.append(reward)
        if done or truncated:
            break
    inputs = path.with_suffix(".pt")
    inputs.parent.mkdir(parents=True, exist_ok=True)
    if inputs.exists():
        raise ValueError("Unfinished inputs exist")
    torch.save(dict(roots=roots, parent_commands=commands, rewards=rewards, terminal=event["event"]), inputs)
    labels = []
    for root in roots:
        a = branch(root, model, cfg, False, _DEVICE)
        np.testing.assert_array_equal(a["commands"], commands[root["tick"]:])
        original = sum(cfg.getfloat("train", "gamma")**i*r for i, r in enumerate(rewards[root["tick"]:]))
        if abs(original-a["q"]) > 1e-12 or a["terminal"] != event["event"]:
            raise ValueError("Restoration parity failed")
        b = branch(root, model, cfg, True, _DEVICE)
        labels.append(dict(seed=seed, case=case, tick=root["tick"], features=root["features"].tolist(),
            y=b["q"]-a["q"], damage=a["terminal"]=="reach_goal" and b["terminal"]!="reach_goal",
            rescue=a["terminal"]!="reach_goal" and b["terminal"]=="reach_goal",
            collision_added=a["terminal"]!="collision" and b["terminal"]=="collision",
            replan={k:v for k,v in a.items() if k!="commands"}, commit={k:v for k,v in b.items() if k!="commands"}))
    r = dict(seed=seed, case=case, parent_terminal=event["event"], steps=len(commands), labels=labels,
        inputs=str(inputs), inputs_sha256=digest(inputs), protocol_sha256=sha)
    save_json(path, r)
    return r


def collect_all(workers):
    p, start, records = verify(), time.perf_counter(), []
    sha = digest(OUT/"protocol.json")
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(p["device"],)) as pool:
        for future in as_completed([pool.submit(collect, (s, c, sha)) for s in SEEDS for c in CASES]):
            records.append(future.result())
            if len(records)%64 == 0:
                print("PROPOSAL_DATA", len(records), "/2048", round(time.perf_counter()-start, 1), flush=True)
    records.sort(key=lambda r:(r["seed"], r["case"]))
    save_json(OUT/"dataset.json", dict(rows=[x for r in records for x in r["labels"]],
        episodes=[{k:v for k,v in r.items() if k!="labels"} for r in records],
        seconds=time.perf_counter()-start, protocol_sha256=sha, feature_names=NAMES))
    print("DATA_DONE", sum(len(r["labels"]) for r in records), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("freeze", "collect"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze(args.device)
    else:
        collect_all(args.workers)


if __name__ == "__main__":
    main()
