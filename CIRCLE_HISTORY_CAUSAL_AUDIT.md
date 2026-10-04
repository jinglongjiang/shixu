# Circle History: Generator Coupling and Residual Temporal Information

Date: 2026-10-04. Completed offline controlled audit; not KDA V9, navigation-policy training, or a new SR claim.

## 1. Direct Answer

**The proposed either/or is false. Circle has a provable generator shortcut, and observations beyond the current frame also contain useful predictive information. The latter has not produced a stable additional navigation benefit in this audit.**

Three distinct conclusions follow:

1. **Generator-specific goal memory is established.** For native-circle pedestrians visible at reset, two legal first-sighting coordinates determine the hidden goal exactly. Extra ordered history cannot add information about that already-determined goal. The legal first-sighting rule attains the true-world maximum return at all 16 selected circle roots under both continuations.
2. **Not all historical prediction gains are an antipodal-goal artifact.** After breaking that coupling while retaining starts and the goal set, historical observations still improve held-out goal and motion prediction. The same is true when fitting and testing within native square.
3. **A need for complex ordered memory is not established.** In circle, adding 24-frame history beyond the first-sighting anchor has practically equivalent goal error; short-window and summary controls have practically equivalent two-second prediction error. None of the three tasks shows a stable navigation advantage for ordered history over the simple controls. These are task/probe-specific conclusions, not a proof that all temporal methods are useless.

Remembering an initial observation is genuine memory, not fabricated evidence. What fails is the inference from this particular benefit to generalizable, complex temporal modeling or KDA necessity.

## 2. Evidence Stronger Than the Previous OOD Comparison

### 2.1 Exact source property

The frozen native generator in vendor/crowd_sim/envs/crowd_sim.py, function generate_circle_crossing_human, calls:

```python
human.set(px, py, -px, -py, 0, 0, 0)
```

Consequently, if a pedestrian is first legally observed at reset, its first observed position B is its spawn position and its goal is exactly G = -B. This result follows from the generating code, not from statistical significance. It does not apply exactly to pedestrians first observed after they have moved, and it does not determine all ORCA internal state or future interactions.

On the 64 fresh native-circle test episodes:

- 290/320 pedestrian tracks were legally visible at reset: 90.625% of tracks.
- Those tracks contribute 2,846/3,061 eligible actor-frame prediction samples: 92.976% of samples.
- All 2,846 corresponding goals are recovered within 1e-6m; maximum numerical discrepancy is 2.35e-7m.

The track and actor-frame denominators are different. The earlier progress update's roughly 93% refers to eligible actor-frame samples, not 93% of unique pedestrians.

### 2.2 Controlled goal-ownership intervention

Three tasks are evaluated separately:

| Regime | Initial positions | Goal set | Goal ownership | Dynamics / reward / observations |
| --- | --- | --- | --- | --- |
| Native circle | Native | Native | Own antipodal goal | Frozen parent |
| Circle-permuted | Identical paired native starts | Identical paired native set | Rotated, no actor keeps its own goal | Frozen parent |
| Native square | Native square | Native square | Native square | Frozen parent |

The rotation is fixed by case ID before motion. All 144 circle/circle-permuted pairs pass exact start/set identity checks. Only goal ownership is intervened upon; changed routes and later interactions are consequences. Later states are not claimed to remain geometrically identical across regimes.

The antipodal rule then fails decisively: for pedestrians visible at reset, mean goal error is 5.192m in circle-permuted and 3.870m in square, with zero exact recoveries in either task. This verifies that the intervention removes the original shortcut.

### 2.3 Fixed splits and controlled probes

Each regime uses 64 fitting cases 72000-72063, 16 validation cases 72064-72079 and 64 fresh test cases 72080-72143. Total: **432 trajectories, 21,579 control frames**. All unsuccessful trajectories are retained; no test case is used to choose a model or hyperparameter.

| Regime | Frames | Goal / collision / timeout across all 144 collection trajectories |
| --- | ---: | ---: |
| Circle | 7,484 | 127 / 12 / 5 |
| Circle-permuted | 6,959 | 137 / 1 / 6 |
| Square | 7,136 | 137 / 1 / 6 |

These are collection-teacher outcomes, not performance comparisons between the predictive probes.

Probe inputs contain only legal robot/current-crowd observations, legal actor history and, where enabled, the first legal sighting. Goals, previous ORCA preferred velocity and future positions are labels only. Track IDs are association keys, not numeric learned features. Cases and regime labels are not model inputs.

All seven probe arms use the same padded 235-dimensional layout, 256 shared fitting-row landmarks, eight regression outputs, 3,928 fitted coefficients plus eight intercepts, and the same validation grid. Each regime is fitted independently. Padding equalizes the coefficient layout, not effective feature complexity or every optimization difficulty.

| Arm | Additional information beyond current crowd state |
| --- | --- |
| Current | None |
| Birth | Two first-sighting position coordinates |
| History24 | Previous 23 actor rows, in arrival order |
| Birth + History3 | Birth and previous two actor rows |
| Birth + History24 | Birth and previous 23 actor rows |
| Birth + Bag24 | Birth and the same previous rows canonically sorted by content |
| Birth + Statistics | Birth and 16 cheap history-summary features |

Bag24 removes explicit arrival order, not every possible temporal cue: positions, velocities and observation ages can indirectly reveal order. Statistics also contain temporal summaries. Neither is a strict information-free control.

Future labels require another two seconds of archived trajectory, excluding terminal tails. Results concern these eligible samples, not every failure-critical frame.

## 3. Held-Out Prediction Findings

Tables show pooled actor-frame means. Confidence intervals below instead give equal weight to each test case and resample cases, not correlated frames. Each interval uses 5,000 paired bootstrap draws and 98.3333% confidence, correcting across three regimes for that contrast/endpoint; this is not a blanket correction over all reported analyses.

### 3.1 Goal recovery

| Regime | Current error (m) | Birth error | History24 error | Birth + History24 error |
| --- | ---: | ---: | ---: | ---: |
| Circle | 1.241 | 0.316 | 0.791 | 0.331 |
| Circle-permuted | 1.150 | 1.149 | 0.928 | 0.944 |
| Square | 1.333 | 1.311 | 1.106 | 1.116 |

| Case-weighted error reduction | Circle | Circle-permuted | Square |
| --- | --- | --- | --- |
| Current minus Birth | +0.859 [0.758, 0.968] | +0.010 [0.002, 0.019] | +0.040 [0.020, 0.061] |
| Current minus History24 | +0.490 [0.420, 0.561] | +0.171 [0.124, 0.227] | +0.187 [0.145, 0.230] |
| Birth minus Birth + History24 | -0.010 [-0.037, 0.016] | +0.156 [0.110, 0.208] | +0.144 [0.104, 0.184] |

The prospectively specified local goal-error margin is 0.1m. In circle, Birth and Birth + History24 meet practical equivalence on this endpoint; their interval lies entirely inside +/-0.1m. In both non-antipodal tasks, history beyond Birth improves goal recovery by more than 0.1m even at the lower confidence bound.

**Established interpretation:** simple anchor memory accounts for the tested circle goal-recovery benefit without requiring a long ordered sequence. Nevertheless, additional observed history remains predictively useful when this exact rule is removed. This does not prove that every stronger current-only estimator would fail.

### 3.2 Future motion and the ordered-history boundary

| Regime | Current 2s error (m) | History24 | Birth + History24 | Birth + Bag24 | Birth + Statistics | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Circle | 0.308 | 0.298 | 0.262 | 0.273 | 0.267 | 0.304 |
| Circle-permuted | 0.313 | 0.257 | 0.260 | 0.271 | 0.285 | 0.222 |
| Square | 0.298 | 0.270 | 0.273 | 0.279 | 0.271 | 0.252 |

Current minus History24 case-weighted 2s-error reductions:

- Circle: +0.0224m [0.0042, 0.0377].
- Circle-permuted: +0.0386m [0.0266, 0.0571].
- Square: +0.0205m [0.0118, 0.0295].

The positive improvements in the non-antipodal tasks refute attributing **all observed-history prediction value** to G = -B. However, history does not beat the strong CV forecast there on the pooled endpoint.

The local 2s-error equivalence margin was fixed at 0.02m. In circle:

- Birth + Bag24 minus Birth + History24: +0.0118m [0.0051, 0.0187], statistically positive but inside the practical-equivalence margin.
- Birth + Statistics minus Birth + History24: +0.0047m [-0.0059, 0.0139], practically equivalent.
- Birth + History3 minus Birth + History24: +0.0016m [-0.0142, 0.0142], practically equivalent.

Thus a measurable order effect can coexist with practical equivalence. Nonsignificance is not used as evidence of equivalence. In square, all three simple controls also meet this endpoint's equivalence criterion against Birth + History24. Circle-permuted equivalence is established for Bag24 but unresolved for Statistics and History3. No universal ordering-independence claim follows.

### 3.3 Correction of the earlier square interpretation

An additional, explicitly post-hoc evaluation applies already-frozen fits to the **same** fresh square test samples:

| Fitting regime | Current 2s error | History24 2s error |
| --- | ---: | ---: |
| Circle | 0.3819m | 0.4876m |
| Square | 0.2977m | 0.2698m |

The circle-fit minus square-fit History24 difference is +0.2025m [0.1767, 0.2289] with equal case weighting. No test-driven refitting was performed.

**Correction:** the prior circle-trained/square-tested failure establishes a transfer failure of that predictor, not an intrinsic absence of useful history in square. Using it to conclude that historical information itself cannot generalize was too strong and is withdrawn. These results concern the offline probes, not a demonstrated cause of the old KDA IL/RL failures.

## 4. Actual Action Consequences, Not Only Prediction Error

Selection was fixed before method outcomes: the first eligible root in each case, first 16 eligible test cases per regime, time >=2s, all five pedestrians visible, robot goal distance >1m and minimum surface clearance <=0.8m. Total: **48 distinct roots**. All root one-step snapshot restores match archived states within 4.77e-7 and exactly match rewards.

For each root, nine prediction/reference worlds enumerate the same native 80 commands. The root command executes for **one 0.25s step**; all candidates then share a legal ORCA continuation to native termination. A second continuation only disables the robot teacher's TTC brake. Human dynamics, action support and original reward remain frozen; gamma=0.99 and progress_reward=0.

A predicted world selects the action; its return is evaluated in the corresponding true world. Regret is max_a Q^pi(s,a) - Q^pi(s,a_selected), not Q*. Estimated worlds replace only the hidden goal and previous preferred velocity; CV uses constant-velocity humans. The first-sighting rule combines its goal rule with the legally current velocity, not true preferred state.

**69,120 candidate branches are computation, not 69,120 independent episodes.** Statistical units are the 16 roots per regime. Both continuations reuse each root and are sensitivity checks, not independent replications.

### 4.1 Mean native-return regret

| Regime / continuation | Current | Birth | History24 | Birth + History24 | Birth + Bag24 | Birth + Statistics | CV | Circle birth rule | Truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Circle / inherited | .00189 | .00202 | .00545 | .00186 | .00196 | .00253 | .00242 | .00000 | .00000 |
| Circle / unbraked | .00234 | .00143 | .00539 | .00238 | .00278 | .00243 | .00323 | .00000 | .00000 |
| Permuted / inherited | .03545 | .03637 | .03392 | .03392 | .03541 | .03638 | .02554 | .03039 | .00000 |
| Permuted / unbraked | .03034 | .03097 | .02988 | .03037 | .03028 | .03126 | .02103 | .02451 | .00000 |
| Square / inherited | .01161 | .00888 | .01111 | .01198 | .00099 | .00885 | .01170 | .02635 | .00000 |
| Square / unbraked | .01104 | .01900 | .01351 | .01431 | .00198 | .00931 | .00229 | .02526 | .00000 |

In native circle the legal birth rule achieves zero regret at **32/32 root-continuation combinations**. Identical return does not imply identical action or clearance. Some circle roots have unsafe alternatives: successful true-world commands range from 53/80 to 80/80. The result is not based solely on states where every command works.

Prespecified Birth minus Birth + History24 regret differences:

| Regime | Inherited continuation | Unbraked continuation |
| --- | --- | --- |
| Circle | +.00016 [-.00350, .00375] | -.00095 [-.00446, .00293] |
| Circle-permuted | +.00245 [.00000, .00542] | +.00060 [-.00574, .00522] |
| Square | -.00310 [-.01606, .00457] | +.00469 [-.00344, .02191] |

The local regret margin is 0.01. Circle and circle-permuted meet practical equivalence for these two probe arms under each continuation; square does not have enough precision to establish equivalence. Circle's remaining-time differences also meet the prospectively fixed +/-0.25s equivalence criterion. This is stronger than merely reporting nonsignificance, but remains local to the specified roots, probe family and continuations.

### 4.2 Safety, completion and simple alternatives

- Circle and square: every evaluated selector succeeds at 16/16 roots under each continuation, with zero collisions and timeouts. This small success-only cohort cannot prove SR/CR equivalence across the benchmark.
- Circle-permuted: all eight non-oracle selectors succeed at 15/16 roots and timeout on the same case, 72086; truth succeeds at 16/16. That root has only two successful true-world commands under inherited continuation and one under unbraked continuation. Latent full-information headroom exists there, but the tested history probes do not recover it.
- Native-circle rule versus truth mean remaining time: 7.4844s versus 7.4844s inherited, 7.6562s versus 7.6562s unbraked. Clearance is not identical; no safety-dominance claim is made.
- Birth + History24 versus CV regret wins/ties/losses: circle 3/10/3 and 2/13/1; circle-permuted 2/10/4 and 2/8/6; square 2/11/3 and 0/10/6. No stable history advantage over CV is demonstrated.
- A post-hoc case-paired comparison in square/unbraked finds Birth + History24 worse than CV by +.01202 regret [.00100, .03168]. This unfavorable result is retained, not selected away.
- In square/unbraked the prespecified Bag24 minus ordered-history difference is -.01233 [-.03165, -.00066]; the simpler content-sorted representation performs better in this diagnostic. That is evidence against an automatic benefit from arrival order, not proof that order is inherently harmful.

The action test applies a predictive latent-state consumer, not the learned Mamba-VL or KDA value head. It does not measure full-policy IL/RL performance. Roots are currently fully visible, so this test isolates hidden motion state rather than directly evaluating actions during occlusion.

## 5. What Is Settled, and What Is Not

| Question | Evidence-based answer |
| --- | --- |
| Does native circle contain an exact first-position/goal shortcut? | **Yes: source proof plus numerical verification.** |
| Does long ordered history add goal information for actors visible at reset, beyond their first-position anchor? | **No for that goal variable: it is already determined.** This is not a statement about every future-motion latent. |
| Is all usable history outside the current frame just that shortcut? | **No for the tested predictive estimators:** residual gains survive the coupling intervention and within-square fitting. Information-theoretic insufficiency of all possible current-only models is not established. |
| Does ordered long history improve the tested circle decisions beyond simple anchor memory? | **No additional practical benefit established; the tested probe contrast meets regret/time equivalence and the analytic rule is locally return-optimal.** |
| Does history establish an independent navigation advantage in the other tasks? | **No:** prediction improves, but simple controls absorb or outperform the observed decision gains. |
| Does this explain the original Mamba-VL no-history versus T=24 SR difference, or KDA V1-V8 losses? | **Not established.** Those are different trained-model comparisons; these probes do not causally decompose their SR. |
| Does this justify restarting KDA V9? | **No evidence-based justification yet.** Nor does it permanently reject KDA or all temporal modeling. |

**Operational conclusion:** native-circle goal recovery must not be used as the main justification for a complex temporal navigation mechanism. The appropriate claim is narrower: legal historical observations can improve latent/motion prediction beyond this generator rule, but a deployable navigation benefit beyond strong simple memory/CV remains unproved. No percentage of the old navigation SR gain can honestly be assigned to the shortcut from this audit alone.

## 6. Reproducibility and Frozen Boundaries

Only offline experiment helpers and tests are added. Existing production navigation sources, simulator, reward, checkpoints and V1-V8 results are unchanged. The optional audit dependency is not a deployment dependency. Local execution used four CPU workers, NumPy1.23.5, SciPy1.3.3, scikit-learn1.0.2 and PyTorch2.1.0; no GPU training or server downloads.

Frozen scientific source SHA256:

fb1cfdc5abda8b0af9df66aeae4bd4d1674894ea6b43eb5c55de3265f6889bd5

Protocol was saved with the collected data before fitting or decision outcomes. Its SHA256:

1a4034d2fa886e403293053d7e71e4cf2b51d2eb7820b19b3e13c080e4b4a8a7

Raw local artifacts remain in the ignored directory:

/home/abc/workspace/shixu/outputs/circle_coupling/

| Artifact | SHA256 |
| --- | --- |
| episodes.pt | fac9e6ad3be51c33ec2c62435e67f2ce49f368d619872fe111dfacd8c7dea1ea |
| models.pt | 2de129de8b94075ce1ad06a9805f0bfa7f8a5ea5e00e3626a0875f93819cb959 |
| decision_branches.pt | 89e623b39b51950e46a4156da1aae78e4e58168c74c37348401dbb6a69855b92 |
| probe_results.json | a2629ec95a7c17b1c062124d5525805a296fcc38357d18c1671e9a5aae338a14 |
| decision_results.json | ffc0d5a5032533bc3b3b40ece4ade08440acd03ba19e0c4e4149e7c4fa42d626 |
| cross_distribution_results.json | 9dcacd898e108dcb3aeb0154ac9630671bc673eb7b88cf02f75ce21592ca8cbe |

To reproduce without overwriting these archived results, use a different --output directory consistently:

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling collect --workers 4 --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling probe --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling cross --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling decisions --workers 4 --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. python -m unittest discover -s tests -v
```

Verification: 109 tests, 106 passed and three existing optional legacy-asset skips. Nine new tests check the intervention, legal feature separation, order control, shared input layout, case-paired inference and disjoint splits. No automatic new navigation training follows this audit.
