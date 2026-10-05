"""Only initiation can change; accept-all and reject-all have exact parents."""

import unittest
from unittest.mock import patch

import numpy as np
from autoresearch_nav.prepare import ProposalPolicy
from autoresearch_nav.evaluate import offline_metrics
from shixu.commitment_release import AllUnsafeReleasePolicy
from shixu.policy import ValuePolicy
from tests import test_online_commitment as fixtures


class AutonavTests(unittest.TestCase):
    def setup_policy(self, gate):
        helper = fixtures.OnlineCommitmentTests()
        parent = helper.policy()
        return ProposalPolicy(parent.model, parent.config, gate), helper

    def test_all_accept_matches_v01_commands_history_and_rng(self):
        candidate, helper = self.setup_policy(None)
        baseline = AllUnsafeReleasePolicy(candidate.model, candidate.config)
        for i in range(40):
            state = helper.state(5. if i<25 else 3.)
            rng = np.random.get_state()
            expected = baseline.predict(state)
            expected_rng = np.random.get_state()
            np.random.set_state(rng)
            actual = candidate.predict(state)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(candidate.history, baseline.history)
            for a,b in zip(expected_rng, np.random.get_state()):
                np.testing.assert_array_equal(a,b)
            self.assertEqual(candidate.last_decision["held"], baseline.last_decision["held"])
            self.assertEqual(candidate.last_decision["release"], baseline.last_decision["release"])

    def test_all_reject_matches_parent_and_consumes_opportunity(self):
        calls = []
        policy, helper = self.setup_policy(lambda x:calls.append(x) or False)
        parent = ValuePolicy(policy.model, policy.config)
        for _ in range(30):
            state = helper.state()
            rng = np.random.get_state()
            expected = parent.predict(state)
            np.random.set_state(rng)
            np.testing.assert_array_equal(policy.predict(state), expected)
            np.testing.assert_array_equal(policy.history, parent.history)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape, (24,))
        self.assertEqual(policy.starts, 0)

    def test_unsafe_release_unchanged(self):
        policy, helper = self.setup_policy(None)
        original = policy.aligned_candidates
        def candidates(state, commands, clearance):
            tokens, rewards, _ = original(state, commands)
            return tokens, rewards, np.full(80, clearance)
        with patch.object(policy, "aligned_candidates", side_effect=lambda s,c:candidates(s,c,1.)):
            for _ in range(9): policy.predict(helper.state())
        with patch.object(policy, "aligned_candidates", side_effect=lambda s,c:candidates(s,c,.1)):
            policy.predict(helper.state())
        self.assertEqual(policy.last_decision["release"], "all-unsafe")
        self.assertFalse(policy.last_decision["held"])

    def test_never_commit_is_zero_coverage_not_a_success(self):
        rows = [dict(seed=419,case=i//2,y=.1 if i%2 else -.1,rescue=bool(i%2),damage=not bool(i%2),collision_added=False) for i in range(8)]
        result = offline_metrics(rows,[False]*8)
        self.assertEqual(result["coverage"], 0.)
        self.assertEqual(result["rescue_retention"], 0.)
        self.assertEqual(result["advantage"], 0.)


if __name__=="__main__": unittest.main()
