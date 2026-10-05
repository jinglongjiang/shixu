# AUTONAV COMMIT-GATE AUTORESEARCH

Only method.py may be edited after the harness freeze. prepare.py, features.py,
evaluate.py, Parent, reward, observer, history, release, duration and actions are fixed.

Gate accepts/rejects the original V0.1 low-progress proposal. Rejection consumes
that opportunity; condition must clear/recur before another. No autonomous starts.

One dataset: new native five-person circle trajectories, counterfactual Parent
continuation vs one V0.1 commitment. Future data are labels only. Case-grouped
train/development; cross-seed roots of a case cannot split. No final feedback.

Allowed: interpretable rules, linear/logistic, MLP at most two hidden layers,
width <=32. fit(rows,budget) receives train rows only. accept(feature,fitted)
receives lawful fixed features only. No new imports beyond numpy/torch/math.

Every experiment is committed before evaluation and retained in results.tsv.
Cheap offline first. Max60 offline, max8 dev1, max3 dev2. No threshold sweep.
Each change has one mechanism hypothesis or simplification. Negative results stay.

Only a candidate passing both dev blocks may be locked. Fresh final is exactly
one candidate against Parent/V0.1, four checkpoints, six cells,32cases. The original
nine gates decide. Final pass: METHOD_CANDIDATE_FOUND. Final fail:
FINAL_CONFIRMATION_FAILED. No retry/refit after final. Exhausted allowed search:
NO_LEARNABLE_COMMIT_GATE, a bounded-search verdict rather than impossibility proof.

Effect-equivalent methods prefer simplicity. One final AUTONAV_OVERNIGHT_REPORT.md
contains method, experiment tree, keep/discard, two dev blocks, final if performed,
failures and next decision. Do not create per-experiment Markdown reports.
