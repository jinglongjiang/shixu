"""Input parity, feature legality, group weights and honest sign diagnostics."""

import unittest

import numpy as np
import torch

from experiments.commit_advantage_probe import episode_weights,fit,metrics,predict
from shixu.commit_features import SCALARS,extract
from tests import test_online_commitment as fixtures


class CommitAdvantageProbeTests(unittest.TestCase):
    def test_extract_does_not_change_parent_history_rng_or_scores(self):
        h = fixtures.OnlineCommitmentTests()
        p = h.policy()
        s = h.state()
        rng = np.random.get_state()
        result = extract(p,s,[],[],[])
        self.assertEqual(result["index"],int(p.score(s).argmax()))
        self.assertEqual(len(p.history),0)
        self.assertIsNone(p.last_action)
        self.assertEqual(len(result["features"]),len(SCALARS)+16)
        self.assertTrue(np.isfinite(result["features"]).all())
        for a,b in zip(rng,np.random.get_state()):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(len(p.model.critic.value_head._forward_pre_hooks),0)

    def rows(self):
        return [dict(seed=419,case=i//2,features=[float(i%2)]*len(SCALARS),y=.1 if i%2 else -.1,
                     fixed_trigger=bool(i%2),commit=dict(terminal="reach_goal"),
                     replan=dict(terminal="reach_goal")) for i in range(8)]

    def test_each_episode_has_equal_weight(self):
        rows = self.rows()+[dict(self.rows()[0])]
        w = episode_weights(rows)
        totals = {i:sum(x for x,r in zip(w,rows) if r["case"]==i) for i in range(4)}
        self.assertEqual(totals,dict.fromkeys(range(4),1.))

    def test_accuracy_does_not_replace_decision_value(self):
        rows = self.rows()
        zero = metrics(rows,np.zeros(len(rows)))
        good = metrics(rows,np.array([r["y"] for r in rows]))
        self.assertEqual(zero["selected_advantage"],0.)
        self.assertEqual(good["balanced_accuracy"],1.)
        self.assertAlmostEqual(good["selected_advantage"],.05)
        self.assertEqual(good["positive_cases"],4)
        self.assertEqual(good["negative_cases"],4)

    def test_constant_feature_normalization_stays_finite(self):
        rows = self.rows()
        fitted = fit(rows,rows,"linear",1)
        self.assertTrue(np.isfinite(predict(fitted,rows)).all())

    def test_negative_class_absence_is_not_balanced_accuracy_one(self):
        rows = [dict(r,y=.1) for r in self.rows()]
        result = metrics(rows,np.ones(len(rows)))
        self.assertIsNone(result["balanced_accuracy"])
        self.assertEqual(result["negative_cases"],0)


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
