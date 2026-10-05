"""One-rule release: same-tick fallback, no hidden reset or trigger change."""

import unittest
from unittest.mock import patch

import numpy as np

from shixu.commitment import ActionCommitmentPolicy
from shixu.commitment_release import AllUnsafeReleasePolicy
from tests import test_online_commitment as fixtures


class CommitmentReleaseTests(unittest.TestCase):
    def fixture(self):
        helper = fixtures.OnlineCommitmentTests()
        old = helper.policy()
        new = AllUnsafeReleasePolicy(old.model, old.config)
        return helper, old, new

    def test_safe_paths_have_exact_v0_parity(self):
        h, old, new = self.fixture()
        for policy in (old, new):
            self.assertIsInstance(policy, ActionCommitmentPolicy)
        with patch.object(old, "score", return_value=h.scores(24)), \
             patch.object(new, "score", return_value=h.scores(24)), \
             patch.object(old, "aligned_candidates", return_value=(None,None,np.ones(80))), \
             patch.object(new, "aligned_candidates", return_value=(None,None,np.ones(80))):
            for tick in range(32):
                rng = np.random.get_state()
                a = old.predict(h.state())
                expected_rng = np.random.get_state()
                np.random.set_state(rng)
                b = new.predict(h.state())
                np.testing.assert_array_equal(a,b)
                self.assertEqual(old.last_decision, new.last_decision)
                for x,y in zip(expected_rng,np.random.get_state()):
                    np.testing.assert_array_equal(x,y)

    def test_existing_commitment_releases_and_replans_same_tick(self):
        h, old, new = self.fixture()
        with patch.object(new, "score", return_value=h.scores(24)), \
             patch.object(new, "aligned_candidates", return_value=(None,None,np.ones(80))):
            for _ in range(9):
                new.predict(h.state())
        previous = np.asarray(new.last_action)
        with patch.object(new, "score", return_value=h.scores(34)) as score, \
             patch.object(new, "aligned_candidates", return_value=(None,None,np.full(80,.1))):
            b = new.predict(h.state())
            np.testing.assert_array_equal(b,.3*previous+.7*np.asarray(new.action_space[34]))
            score.assert_called_once()
            self.assertEqual(new.last_decision["release"],"all-unsafe")
            self.assertFalse(new.last_decision["held"])
            self.assertEqual(new.last_decision["selected_grid"],34)
            self.assertIsNone(new.active_grid)
            for _ in range(9):
                new.predict(h.state())
                self.assertFalse(new.last_decision["held"])
        self.assertEqual(new.starts,1)

    def test_new_start_also_releases_without_changing_parent_fallback(self):
        h, old, new = self.fixture()
        with patch.object(new,"score",return_value=h.scores(34)), \
             patch.object(new,"aligned_candidates",return_value=(None,None,np.full(80,.1))):
            for _ in range(9):
                new.predict(h.state())
        self.assertTrue(new.last_decision["started"])
        self.assertEqual(new.last_decision["release"],"all-unsafe")
        self.assertFalse(new.last_decision["held"])
        self.assertEqual(new.last_decision["proposal"],34)
        self.assertFalse(new.last_decision["blocked"])
        self.assertFalse(new.armed)

    def test_exact_margin_is_safe_and_original_blocking_still_releases(self):
        h, old, new = self.fixture()
        clearance = np.full(80,.1)
        clearance[24] = .2
        with patch.object(new,"score",return_value=h.scores(24)), \
             patch.object(new,"aligned_candidates",return_value=(None,None,clearance)):
            for _ in range(9):
                new.predict(h.state())
        self.assertTrue(new.last_decision["held"])
        self.assertFalse(new.last_decision["no_margin_safe_candidate"])
        clearance[24],clearance[34] = .1,.2
        with patch.object(new,"score",return_value=h.scores(34)), \
             patch.object(new,"aligned_candidates",return_value=(None,None,clearance)):
            new.predict(h.state())
        self.assertEqual(new.last_decision["release"],"safety-blocked")
        self.assertEqual(new.last_decision["selected_grid"],34)


if __name__ == "__main__":
    unittest.main()
