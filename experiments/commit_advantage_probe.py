"""Bounded five-circle labels, followed by episode/seed-disjoint learnability."""

import argparse
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor,as_completed
import datetime as dt
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from experiments.decision_state import exact_snapshot
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load,weights
from experiments.multihorizon_control import advance,fork,save_json
from experiments.occlusion import source_hash
from shixu.commit_features import SCALARS,extract
from shixu.features import window
from shixu.observations import OccludedTracks
from shixu.policy import ValuePolicy
from shixu.runner import environment


OUT = Path("outputs/commit-advantage-probe")
SEEDS = (419,443,467,491)
CASES = tuple(range(90000,90096))
RULES = dict(
    domain="Only native5-person circle; no square or10/20 labels. Original four frozen CV critics.",
    roots="Before outcomes: ticks8,24,40 (2/6/10s) plus first causal past2s<=0.2m low-progress tick; "
          "deduplicate, require own-goal distance>1m and live episode. No baseline failure selection.",
    label="Restore identical float64 world/history/RNG. Replan follows original Parent to terminal. "
          "Commit holds current Parent grid at most8steps, original mask/all-unsafe same-tick release, "
          "then permanently Parent. Original gamma/rewards; y=Gcommit-Gparent. No future gate in either tail.",
    inputs="18 legal semantic scalars plus128 selected-candidate pre-value-head features already computed "
           "by Parent. No seed/case/geometry label, goal of human, ORCA internal state or future enters inputs.",
    split="90000-90063 train,90064-90079 validation,90080-90095 test. All roots per case stay together. "
          "Four leave-checkpoint-out folds train/validate on three seeds and test only held-out seed plus "
          "unseen test cases. One case-only fold uses all four seeds, same unseen-case test. No refit on test.",
    models="Semantic ridge(alpha1) and18+128->16ReLU->1 MLP; full-batch Adam lr0.001,200epochs, "
           "train-only normalization/target scale, best validation MSE. Episode-balanced weights. "
           "RNG1027+fold. No feature/hyperparameter search or outcome-based sampling.",
    go="For either preregistered model: case-only and combined seed+case-heldout mean selected one-shot "
       "advantage>0.001; >=3/4 seed folds mean selected advantage>=0; both test sign classes "
       "(|y|>0.005) each occur in>=4distinct cases. Selected interventions add<=1pp collision "
       "versus replan labels. GO is development learnability, not online repeated-option performance. "
       "If support absent: INSUFFICIENT_LABEL_SUPPORT. Otherwise no qualifying model: NO_OOS_SIGNAL.",
    stop="No automatic navigation/gate training unless GO. Negative closes only this feature/data/model "
         "protocol, not action persistence generally. Positive needs original nine fresh closed-loop gates.",
    prior="Adaptive navigation execution-time/SMDP and option initiation are prior art; no first-ever claim. "
          "This task does not establish mechanism novelty.")
_DEVICE = None
_MODELS = {}


def read(path):
    return json.loads(path.read_text())


def helpers():
    return {str(p):digest(p) for p in (Path(__file__),Path("experiments/multihorizon_control.py"),
        Path("experiments/decision_state.py"),Path("experiments/forecast_evidence.py"))}


def freeze(device):
    old = read(Path("outputs/online-action-commitment-v01/protocol.json"))
    for c in old["checkpoints"].values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Frozen critic changed")
    search = subprocess.run(["rg","-l","--glob","*.json",
        r'"case"\s*:\s*900(?:[0-8][0-9]|9[0-5])\b|"(?:case_start|eval_case_start|test_case_start)"\s*:\s*90000\b',
        "outputs"],capture_output=True,text=True)
    if search.returncode != 1:
        raise ValueError("Label cases already used or search failed: "+search.stdout+search.stderr)
    save_json(OUT/"protocol.json",dict(rules=RULES,cases=CASES,seeds=SEEDS,checkpoints=old["checkpoints"],
        core_sha256=source_hash(),helpers=helpers(),parent_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),device=device,
        prior_url="https://arxiv.org/abs/2108.06161",case_unused_check=dict(exit_code=search.returncode,matches=search.stdout)))


def verify():
    p = read(OUT/"protocol.json")
    if p["rules"] != RULES or p["core_sha256"] != source_hash() or p["helpers"] != helpers():
        raise ValueError("Probe contract changed")
    for c in p["checkpoints"].values():
        if digest(Path(c["path"])) != c["sha256"]:
            raise ValueError("Critic changed")
    return p


def init_worker(device):
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


@torch.inference_mode()
def branch(root,model,cfg,commit,device):
    world = fork(root,model,cfg,device)
    np.random.set_state(root["rng"])
    holding = bool(commit)
    while not world["done"]:
        tick = world["steps"]
        state = root["state"] if tick==0 else world["observer"].observe(world["env"])
        policy = world["policy"]
        if holding and tick<8:
            commands = policy.candidate_actions()
            _,_,clearance = policy.aligned_candidates(state,commands)
            safe = clearance>=.2
            if not safe.any() or not safe[root["grid"]]:
                holding = False
        else:
            holding = False
        if holding:
            command = commands[root["grid"]]
            np.random.random()
            policy.history.append(policy.encode(state))
            policy.last_action = command
        else:
            command = policy.predict(state)
        advance(world,command,cfg)
    return dict(terminal=world["event"],q=world["total"],minimum=world["minimum"],
                seconds=world["steps"]*.25,commands=world["commands"])


@torch.inference_mode()
def collect(task):
    seed,case,sha = task
    path = OUT/"cases"/f"{seed}-{case}.json"
    if path.exists():
        record = read(path)
        if record["protocol_sha256"] != sha or digest(Path(record["inputs"])) != record["inputs_sha256"]:
            raise ValueError("Cached probe differs")
        return record
    if seed not in _MODELS:
        _MODELS[seed] = load(weights(seed,"cv"),_DEVICE)
    model,cfg = _MODELS[seed]
    policy = ValuePolicy(model,cfg,_DEVICE)
    env = environment(cfg,policy,"circle",5)
    env.reset(options={"test_case":case})
    observer = OccludedTracks(cfg.getfloat("observation","retention_seconds"))
    distances,grids,commands = deque(maxlen=8),deque(maxlen=8),[]
    roots,rewards,clearances,first_low = [],[],[],False
    tick = 0
    while True:
        state = observer.observe(env)
        info = extract(policy,state,distances,grids,commands[-8:])
        distance = float(np.hypot(state.self_state.px-state.self_state.gx,state.self_state.py-state.self_state.gy))
        past = list(distances)+[distance]
        low = len(past)==9 and distance>1 and past[0]-past[-1]<=.2+1e-12
        new_low = low and not first_low
        first_low |= low
        if (tick in (8,24,40) or new_low) and distance>1:
            roots.append(dict(episode=dict(people=5,geometry="circle",case=case),tick=tick,state=state,
                snapshot=exact_snapshot(env,observer),history=window(list(policy.history)+[policy.encode(state)],24,"zero"),
                previous=policy.last_action,grid=info["index"],features=info["features"],
                fixed_trigger=low,rng=np.random.get_state()))
        np.random.random()
        action = info["command"]
        policy.history.append(policy.encode(state))
        policy.last_action = action
        _,reward,done,truncated,event = env.step(action)
        distances.append(distance)
        grids.append(info["index"])
        commands.append(list(action))
        rewards.append(reward)
        clearances.append(event["dmin"])
        tick += 1
        if done or truncated:
            break
    path.parent.mkdir(parents=True,exist_ok=True)
    inputs = path.with_suffix(".pt")
    if inputs.exists():
        raise ValueError("Unfinished inputs exist; do not overwrite")
    torch.save(dict(roots=roots,parent_commands=commands,rewards=rewards,clearances=clearances,
        terminal=event["event"],protocol_sha256=sha),inputs)
    labels = []
    for root in roots:
        a = branch(root,model,cfg,False,_DEVICE)
        expected = commands[root["tick"]:]
        np.testing.assert_array_equal(a["commands"],expected)
        original_q = sum(cfg.getfloat("train","gamma")**i*r for i,r in enumerate(rewards[root["tick"]:]))
        if a["terminal"] != event["event"] or abs(a["q"]-original_q)>1e-12:
            raise ValueError("Root restoration/native continuation differs")
        b = branch(root,model,cfg,True,_DEVICE)
        labels.append(dict(seed=seed,case=case,tick=root["tick"],features=root["features"].tolist(),
            fixed_trigger=root["fixed_trigger"],y=b["q"]-a["q"],
            replan={k:v for k,v in a.items() if k!="commands"},commit={k:v for k,v in b.items() if k!="commands"},
            commit_commands=b["commands"],restored_parent_parity=True))
    record = dict(seed=seed,case=case,parent_terminal=event["event"],steps=tick,labels=labels,
        inputs=str(inputs),inputs_sha256=digest(inputs),protocol_sha256=sha)
    save_json(path,record)
    return record


def collect_all(workers):
    p = verify()
    tasks = [(s,c,digest(OUT/"protocol.json")) for s in SEEDS for c in CASES]
    start,records = time.perf_counter(),[]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context("spawn"),initializer=init_worker,
                             initargs=(p["device"],)) as pool:
        jobs = [pool.submit(collect,t) for t in tasks]
        for future in as_completed(jobs):
            records.append(future.result())
            if len(records)%16==0:
                print("LABEL_CASES",len(records),"/384",round(time.perf_counter()-start,1),flush=True)
    records.sort(key=lambda r:(r["seed"],r["case"]))
    labels = [x for r in records for x in r["labels"]]
    save_json(OUT/"dataset.json",dict(labels=[{k:v for k,v in x.items() if k!="commit_commands"} for x in labels],
        cases=[{k:v for k,v in r.items() if k!="labels"} for r in records],elapsed_seconds=time.perf_counter()-start,
        protocol_sha256=digest(OUT/"protocol.json")))
    print("LABELS_COMPLETE",len(labels),flush=True)


def episode_weights(rows):
    counts = Counter((r["seed"],r["case"]) for r in rows)
    return np.array([1./counts[r["seed"],r["case"]] for r in rows],np.float64)


def fit(train,val,kind,seed):
    dim = len(SCALARS) if kind=="linear" else len(train[0]["features"])
    x = np.array([r["features"][:dim] for r in train])
    y = np.array([r["y"] for r in train])
    w = episode_weights(train)
    mean = np.average(x,axis=0,weights=w)
    std = np.sqrt(np.average((x-mean)**2,axis=0,weights=w)).clip(1e-6)
    xn = (x-mean)/std
    if kind=="linear":
        design = np.column_stack((xn,np.ones(len(xn))))
        penalty = np.eye(dim+1)
        penalty[-1,-1] = 0
        coefficients = np.linalg.solve(design.T@(w[:,None]*design)+penalty,design.T@(w*y))
        return dict(kind=kind,dim=dim,mean=mean,std=std,coefficients=coefficients)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(dim,16),nn.ReLU(),nn.Linear(16,1))
    scale = max(float(np.sqrt(np.average(y*y,weights=w))),.001)
    xt,yt,wt = torch.tensor(xn,dtype=torch.float32),torch.tensor(y/scale,dtype=torch.float32),torch.tensor(w,dtype=torch.float32)
    xv = torch.tensor((np.array([r["features"][:dim] for r in val])-mean)/std,dtype=torch.float32)
    yv = torch.tensor([r["y"]/scale for r in val],dtype=torch.float32)
    wv = torch.tensor(episode_weights(val),dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    best,best_epoch,parameters = float("inf"),0,None
    for epoch in range(200):
        loss = (wt*(model(xt).squeeze(-1)-yt)**2).sum()/wt.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            score = float((wv*(model(xv).squeeze(-1)-yv)**2).sum()/wv.sum())
        if score<best:
            best,best_epoch = score,epoch+1
            parameters = {k:v.detach().clone() for k,v in model.state_dict().items()}
    return dict(kind=kind,dim=dim,mean=mean,std=std,scale=scale,parameters=parameters,best_epoch=best_epoch)


def predict(fitted,rows):
    x = (np.array([r["features"][:fitted["dim"]] for r in rows])-fitted["mean"])/fitted["std"]
    if fitted["kind"]=="linear":
        return np.column_stack((x,np.ones(len(x))))@fitted["coefficients"]
    model = nn.Sequential(nn.Linear(fitted["dim"],16),nn.ReLU(),nn.Linear(16,1))
    model.load_state_dict(fitted["parameters"])
    with torch.no_grad():
        return model(torch.tensor(x,dtype=torch.float32)).squeeze(-1).numpy()*fitted["scale"]


def metrics(rows,predictions):
    y = np.array([r["y"] for r in rows])
    w = episode_weights(rows)
    selected = np.asarray(predictions)>0
    meaningful = np.abs(y)>.005
    positive,negative = y>.005,y<-.005
    sensitivity = float(np.mean(selected[positive])) if positive.any() else None
    specificity = float(np.mean(~selected[negative])) if negative.any() else None
    gain = float(np.average(selected*y,weights=w))
    collision_delta = np.array([(r["commit"]["terminal"]=="collision")-(r["replan"]["terminal"]=="collision") for r in rows])
    return dict(rows=len(rows),cases=len({r["case"] for r in rows}),selected=int(selected.sum()),
        mae=float(np.average(np.abs(predictions-y),weights=w)),rmse=float(np.sqrt(np.average((predictions-y)**2,weights=w))),
        selected_advantage=gain,always_commit_advantage=float(np.average(y,weights=w)),
        fixed_trigger_advantage=float(np.average(y*np.array([r["fixed_trigger"] for r in rows]),weights=w)),
        positive_cases=len({r["case"] for r in rows if r["y"]>.005}),
        negative_cases=len({r["case"] for r in rows if r["y"]<-.005}),
        balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
        selected_collision_delta=float(np.average(selected*collision_delta,weights=w)),
        positive_selected=int((selected&positive).sum()),negative_selected=int((selected&negative).sum()),
        meaningful=int(meaningful.sum()))


def learn():
    verify()
    labels = read(OUT/"dataset.json")["labels"]
    rows = []
    for kind in ("linear","mlp"):
        folds,all_joint,all_predictions = [],[],[]
        for fold,heldout in enumerate((None,*SEEDS)):
            train = [r for r in labels if r["case"]<90064 and r["seed"]!=heldout]
            val = [r for r in labels if 90064<=r["case"]<90080 and r["seed"]!=heldout]
            test = [r for r in labels if r["case"]>=90080 and (heldout is None or r["seed"]==heldout)]
            if {r["case"] for r in train}&{r["case"] for r in test} or heldout is not None and any(r["seed"]==heldout for r in train+val):
                raise ValueError("Group leakage")
            fitted = fit(train,val,kind,1027+fold)
            target = OUT/"probes"/f"{kind}-{heldout or 'case-only'}.pt"
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise ValueError("Do not overwrite probe")
            torch.save(fitted,target)
            estimates = predict(fitted,test)
            report = dict(heldout_seed=heldout,train_rows=len(train),validation_rows=len(val),
                checkpoint=str(target),checkpoint_sha256=digest(target),metrics=metrics(test,estimates))
            folds.append(report)
            if heldout is not None:
                all_joint.extend(test)
                all_predictions.extend(estimates)
        joint = metrics(all_joint,np.array(all_predictions))
        support = joint["positive_cases"]>=4 and joint["negative_cases"]>=4
        gates = dict(label_support=support,case_only_gain=folds[0]["metrics"]["selected_advantage"]>.001,
            joint_gain=joint["selected_advantage"]>.001,
            three_seed_nonnegative=sum(f["metrics"]["selected_advantage"]>=0 for f in folds[1:])>=3,
            case_collision_guard=folds[0]["metrics"]["selected_collision_delta"]<=.01,
            joint_collision_guard=joint["selected_collision_delta"]<=.01)
        rows.append(dict(kind=kind,folds=folds,joint=joint,gates=gates,passed=all(gates.values())))
    verdict = "GO_LEARNABILITY" if any(r["passed"] for r in rows) else (
        "INSUFFICIENT_LABEL_SUPPORT" if not any(r["gates"]["label_support"] for r in rows) else "NO_OOS_SIGNAL")
    save_json(OUT/"learnability.json",dict(verdict=verdict,models=rows,labels=len(labels),
        dataset_sha256=digest(OUT/"dataset.json"),protocol_sha256=digest(OUT/"protocol.json"),
        limits="One-shot labels under Parent continuation and Parent visitation, not repeated-gate Q or online success proof. Test cases have been consumed, not used to refit a winning model."))
    print("LEARNABILITY",verdict,[(r["kind"],r["joint"],r["gates"]) for r in rows],flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage",choices=("freeze","collect","learn"))
    p.add_argument("--device",default="cuda")
    p.add_argument("--workers",type=int,default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.stage=="freeze":
        freeze(a.device)
    elif a.stage=="collect":
        collect_all(a.workers)
    else:
        learn()


if __name__=="__main__":
    main()
