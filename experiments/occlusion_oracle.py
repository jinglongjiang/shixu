"""Frozen-consumer retained-track truth intervention; not a deployable oracle."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.occlusion import evaluate_weights
from shixu.observations import OccludedTracks


def evaluate(weights, protocol, seed, device, mode):
    original = OccludedTracks.observe

    def observe(observer, env):
        state = original(observer, env)
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
    result.update(mode=mode, seed=seed, scope="Same frozen consumer, cases, retention lifecycle and controller. "
                  "Truth substitution changes only currently retained hidden states, never unseen/expired actors "
                  "or the track store. Not a full-future oracle, upper bound on achievable learning, or deployed "
                  "method. Visible-only is an input intervention, not a newly trained reference.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("truth_retained", "visible_only", "parent"), default="truth_retained")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(args.weights, json.loads(Path(args.protocol).read_text()), args.seed, args.device, args.mode)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
