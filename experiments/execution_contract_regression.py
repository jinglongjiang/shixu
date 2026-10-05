"""Cached80009 regression for the score/execution fix; no new rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crowd_sim.envs.utils.action import ActionXY
from experiments.forecast_control_diagnostic import digest
from experiments.forecast_evidence import load
from experiments.multihorizon_control import save_json
from shixu.policy import ValuePolicy


OUT = Path("outputs/execution_contract_fix")
WEIGHTS = Path("outputs/forecast_control_b_fresh_cv/419/cv/model.pt")
COHORT = Path("outputs/temporal_problem_audit/cohort.pt")
CACHED = Path("outputs/action_ranking_attribution/root_05.json")
QUERIES = Path("outputs/action_ranking_attribution/queries_05.pt")
HEADROOM = Path("outputs/temporal_problem_audit/action_headroom/root_05.json")


@torch.inference_mode()
def main(device):
    files = [WEIGHTS, COHORT, CACHED, QUERIES, HEADROOM, Path("shixu/policy.py"), Path(__file__)]
    hashes = {str(p): digest(p) for p in files}
    save_json(OUT / "protocol.json", dict(inputs=hashes,
        scope="Engineering regression only. Fixed80009, frozen weights/history. No training or new continuation.",
        checks="Scored queries/reward/clearance/filter/risk equal archived executed-CV. "
               "Command mapping occurs once; train phase/first control keep grid commands."))
    root = torch.load(COHORT, map_location="cpu", weights_only=False)["roots"][5]
    assert (root["episode"]["case"], root["tick"]) == (80009, 18)
    cached = json.loads(CACHED.read_text())
    queries = torch.load(QUERIES, map_location="cpu", weights_only=False)["queries"]["executed"]["cv"]
    headroom = json.loads(HEADROOM.read_text())
    model, cfg = load(WEIGHTS, device)
    policy = ValuePolicy(model, cfg, device)
    policy.history.extend(root["history"][:-1])
    policy.last_action = ActionXY(*root["previous"])
    commands = np.asarray(policy.candidate_actions())
    np.testing.assert_array_equal(commands, headroom["commands"])
    before = list(policy.history)
    tokens, rewards, clearances = policy.aligned_candidates(root["state"])
    np.testing.assert_array_equal(tokens, queries)
    reference = cached["stages"]["executed_cv"]
    np.testing.assert_array_equal(rewards, reference["immediate_reward"])
    np.testing.assert_array_equal(clearances, reference["clearance"])
    scores = policy.score(root["state"])
    np.testing.assert_allclose(scores, reference["final"], atol=1e-7, rtol=0)
    for first, second in zip(before, policy.history):
        np.testing.assert_array_equal(first, second)
    index = int(scores.argmax())
    actual = policy.predict(root["state"])
    np.testing.assert_array_equal(actual, commands[index])
    policy.last_action = None
    np.testing.assert_array_equal(policy.candidate_actions(), policy.action_space)
    policy.last_action = ActionXY(*root["previous"])
    policy.set_phase("train")
    np.testing.assert_array_equal(policy.candidate_actions(), policy.action_space)
    assert hashes == {str(p): digest(p) for p in files}
    save_json(OUT / "result.json", dict(inputs=hashes, torch=torch.__version__, device=device,
        maximum_score_error=float(np.max(np.abs(scores - reference["final"]))),
        query_reward_clearance_parity=True, score_does_not_write_history=True,
        no_double_smoothing=True, training_commands_unchanged=True, first_control_commands_unchanged=True,
        selected_action=index, executed_command=list(actual), scores=scores.tolist(),
        archived_outcome=headroom["outcomes"][index],
        interpretation="Selection remains17. Archived outcome uses the old frozen continuation; "
                       "this is not a newly evaluated corrected-policy closed-loop outcome."))
    print("EXECUTION_CONTRACT_REGRESSION", index, "max_error", float(np.max(np.abs(scores-reference["final"]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.device)
