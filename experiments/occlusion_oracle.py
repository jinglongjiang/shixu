"""Frozen-consumer retained-track truth intervention; not a deployable oracle."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.occlusion import evaluate_weights
from shixu.observations import OccludedTracks


def extend_expired(observer, env, state, maximum_age):
    """Offline truth for already-seen expired actors; never update their stored evidence."""
    humans, keys, flags, ages = map(list, (state.human_states, state.track_ids, state.observed, state.ages))
    retained = set(keys)
    for human, (key, _, stamp) in observer.tracks.items():
        age = env.global_time - stamp
        if key not in retained and age <= maximum_age:
            humans.append(human.get_observable_state())
            keys.append(key)
            flags.append(False)
            ages.append(age)
    observer.counts["retained_hidden"] = sum(not flag for flag in flags)
    return state._replace(human_states=humans, track_ids=tuple(keys), observed=tuple(flags), ages=tuple(ages))


def evaluate(weights, protocol, seed, device, mode):
    original = OccludedTracks.observe
    # Prefix has T-1 real frames including the root: its oldest measurement is T-2 ticks old.
    maximum_age = (protocol["history"] - 2) * .25

    def observe(observer, env):
        if mode == "cv_extended":
            observer.retention_seconds = maximum_age
        state = original(observer, env)
        if mode == "truth_expired":
            return extend_expired(observer, env, state, maximum_age)
        if mode == "truth_retained":
            truth = {row[0]: human.get_observable_state() for human, row in observer.tracks.items()}
            humans = [estimated if measured else truth[key]
                      for key, estimated, measured in zip(state.track_ids, state.human_states, state.observed)]
            return state._replace(human_states=humans)
        if mode == "visible_only":
            indices = [i for i, measured in enumerate(state.observed) if measured]
            return state._replace(human_states=[state.human_states[i] for i in indices],
                                  track_ids=tuple(state.track_ids[i] for i in indices),
                                  observed=tuple(state.observed[i] for i in indices),
                                  ages=tuple(state.ages[i] for i in indices))
        return state

    with patch.object(OccludedTracks, "observe", observe):
        result = evaluate_weights(weights, protocol, device, protocol["development_cases"], seed)
    result.update(mode=mode, seed=seed, maximum_expired_age_seconds=maximum_age,
                  scope="Same frozen consumer, cases and controller. truth_retained changes only currently retained "
                        "hidden states. truth_expired adds current truth only for previously measured expired actors "
                        "whose last observation is still inside the policy's legal root prefix; retained states remain CV. "
                        "cv_extended changes only the fixed retention deadline to the same legal-history span. "
                        "No mode introduces never-seen people or writes hidden truth into the tracker. "
                        "Frozen-consumer interventions, not a full-future upper bound or trained method comparison.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("truth_retained", "truth_expired", "cv_extended", "visible_only", "parent"),
                        default="truth_retained")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(args.weights, json.loads(Path(args.protocol).read_text()), args.seed, args.device, args.mode)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
