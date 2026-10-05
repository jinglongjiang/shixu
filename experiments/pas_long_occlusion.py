"""Prespecified >=2s-hidden roots in the already collected PaS trajectories."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import pas_problem_discovery as parent


OUT = parent.OUT / "long_occlusion"


def prepare():
    rows = [(int(p.stem.split("_")[-1]), json.loads(p.read_text())["record"])
            for p in sorted(parent.OUT.glob("case_*.json"))]
    eligible = [(i, r) for i, r in rows if r["counts"]["seen_hidden2s"] > 0]
    failures = [i for i, r in eligible if r["terminal"] != "ReachGoal"][:3]
    successes = [i for i, r in eligible if r["terminal"] == "ReachGoal"][:3]
    parent.save(OUT / "protocol.json", dict(
        selection=failures+successes, failures=failures, successes=successes,
        rule="First t>=2s with a previously seen, currently invisible actor within3m "
             "and last legal sighting>=2s ago. First3 failure and first3 success episodes "
             "in the existing fixed queue. No new scenes or outcome-dependent root replacement.",
        inputs={str(parent.OUT/f"case_{i:02d}.json"):parent.digest(parent.OUT/f"case_{i:02d}.json")
                for i,_ in rows}, checkpoint_sha256=parent.digest(parent.CHECKPOINT),
        helper_sha256=parent.digest(Path(parent.__file__)), script_sha256=parent.digest(Path(__file__)),
        comparison=parent.RULES["consumer"], continuation=parent.RULES["continuation"],
        headroom=parent.RULES["headroom"]))


@torch.inference_mode()
def restore(actor, cfg, record):
    env, observation = parent.new_world(cfg, record["people"], record["geometry"], record["case"])
    torch.manual_seed(record["case"])
    hidden = {k:torch.zeros(1,1,128) for k in ("pas","policy")}
    masks, seen, last, tracker = torch.ones(1,4), set(), {}, parent.GridTracks()
    for tick, command in enumerate(record["actions"]):
        tracker.update(observation["grid"][-1], env.xy_local_grid[-1], env.global_time)
        flags = parent.events(env,seen,last)
        root = None
        if tick>=8 and flags["seen_hidden2s"]:
            root = dict(tick=tick, observation=copy.deepcopy(observation), hidden=parent.clone_hidden(hidden),
                        rng=torch.get_rng_state().clone(), world=parent.world_state(env), events=flags,
                        coordinates=copy.deepcopy(env.xy_local_grid[-1]), tracker=copy.deepcopy(tracker),
                        time=env.global_time)
        action,hidden,after_rng,decoded = parent.query(actor,observation,hidden,masks,torch.get_rng_state(),"full4")
        np.testing.assert_allclose(action,command,atol=1e-7,rtol=0)
        if root is not None:
            root.update(after_hidden=parent.clone_hidden(hidden), after_rng=after_rng,
                        native_action=np.asarray(command), decoded=decoded.copy())
            return root
        observation,reward,done,info = env.step(np.asarray(command))
        assert not done
    raise RuntimeError("Recorded long-occlusion event could not be recovered")


@torch.inference_mode()
def evaluate():
    protocol=json.loads((OUT/"protocol.json").read_text())
    assert all(parent.digest(Path(p))==h for p,h in protocol["inputs"].items())
    assert parent.digest(Path(__file__))==protocol["script_sha256"]
    assert parent.digest(Path(parent.__file__))==protocol["helper_sha256"]
    actor,cfg=parent.build_actor()
    for index in protocol["selection"]:
        destination=OUT/f"diagnostic_{index:02d}.json"
        if destination.exists():
            continue
        started=time.perf_counter()
        record=json.loads((parent.OUT/f"case_{index:02d}.json").read_text())["record"]
        root=restore(actor,cfg,record)
        torch.save(root,OUT/f"root_{index:02d}.pt")
        sensor=root["observation"]["grid"][-1]
        native=root["decoded"].reshape(sensor.shape)
        grids={"projected_parent":native,
            "short_cv":root["tracker"].occupancy(native,sensor,root["coordinates"],root["time"],ttl=.75),
            "long_cv":root["tracker"].occupancy(native,sensor,root["coordinates"],root["time"]),
            "long_hold":root["tracker"].occupancy(native,sensor,root["coordinates"],root["time"],cv=False),
            "oracle_occupancy":root["observation"]["label_grid"][0].copy()}
        arms={}
        for mode in ("full4","short2","current_grid"):
            action,_,_,_=parent.query(actor,root["observation"],root["hidden"],torch.ones(1,4),root["rng"],mode)
            arms[mode]=dict(action=action.tolist(),actual=parent.branch(actor,cfg,record,root,action))
        baseline=arms["full4"]["actual"]
        np.testing.assert_array_equal(baseline["commands"],record["actions"][root["tick"]:])
        assert baseline["terminal"]==record["terminal"]
        for mode,grid in grids.items():
            action=parent.occupancy_query(actor,root,grid)
            arms[mode]=dict(action=action.tolist(),actual=parent.branch(actor,cfg,record,root,action),
                           occupancy=parent.grid_error(grid,root))
        for arm in arms.values():
            arm["delta_return_from_native"]=arm["actual"]["return_"]-baseline["return_"]
        headroom=[]
        if index in protocol["failures"]:
            commands=[root["native_action"],np.zeros(2)]
            commands += [speed*np.asarray([np.cos(theta),np.sin(theta)]) for speed in (.5,1.)
                         for theta in np.arange(16)*2*np.pi/16]
            headroom=[dict(action=a.tolist(),actual=parent.branch(actor,cfg,record,root,a)) for a in commands]
        parent.save(destination,dict(index=index,case=record["case"],people=record["people"],
            geometry=record["geometry"],tick=root["tick"],root_events=root["events"],arms=arms,headroom=headroom,
            elapsed_seconds=time.perf_counter()-started,
            limits="Coverage follow-up, not independent confirmation; same episodes and frozen consumer. "
                   "Projection inputs can shift distribution; memory-free controls were not trained."))
        print("PAS_LONG_OCCLUSION",index,"tick",root["tick"],
              {k:(v["actual"]["terminal"],round(v["delta_return_from_native"],6)) for k,v in arms.items()},
              "headroom_successes",sum(v["actual"]["terminal"]=="ReachGoal" for v in headroom),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","evaluate"))
    args=parser.parse_args()
    torch.set_num_threads(1)
    {"prepare":prepare,"evaluate":evaluate}[args.mode]()
