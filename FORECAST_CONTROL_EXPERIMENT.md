# Forecast-to-Control Navigation Experiment

Date: 2026-10-04. Status: COMPLETE; NO_VALIDATED_KDA_ADVANTAGE_THIS_VERSION.

## Final Decision

The bounded navigation attempt is complete, including its one permitted common
interface repair. Both A and B underwent actual IL and online MC training,
saved-weight reload and native closed-loop evaluation; these are not forward
fixtures or surrogate action tests. B has16 newly trained runs, four reused
strong-reference evaluations and768 test episodes per arm. Training is exclusively
five-person circle; evaluation includes5/10/20 people in circle and square.

B's primary10/20-person SR is86.52% for KDA,83.40% for GRU,87.11% for Current,
86.13% for fresh CV and84.96% for the reused original Parent. KDA does not beat
Current, and its+0.39pp over fresh CV accompanies+1.56pp collision. It fails the
fixed gain, seed-consistency and safety criteria. No fresh confirmation, third
interface rescue, KDA V9/V10 or parameter sweep is started.

This rejects the tested forecast-to-native-query method as a demonstrated KDA
advantage, not temporal modeling or the entire KDA family. All B arms retain the
original contextual-GRU critic: their names distinguish the motion estimators,
not pure GRU-versus-KDA navigation backbones. The connection consumes only the
native .25s forecast; it does not test a planner consuming longer-horizon history
information. Prediction-error and CV-swap diagnostics explain the tested contract
but cannot replace its navigation outcomes or establish a unique failure cause.

## Fixed Question

Can lawful history-derived motion forecasts improve a trained navigation policy
when those forecasts explicitly enter its 80-action value evaluation?

This is a complete bounded method attempt, not KDA V9, another goal-recovery
audit, or a claim that only KDA can solve the task. Circle coupling, the previous
predictive audit and V1-V8 negative results remain closed and unchanged.

## Prototype-A Architecture (Completed)

```text
legal actor observations (24 control frames)
       -> CV / current-frame / GRU / KDA motion estimator
       -> predicted positions over the next 2.25 seconds
       -> relative geometry for each native candidate action
       -> shared spatial attention and scalar value
       -> inherited reward + gamma * successor value
       -> inherited filtering / smoothing / execution

shared ORCA data -> IL value initialization -> online MC refinement
```

There is one motion estimator and one value consumer. No gate variants, hidden
goal head, second memory, world-model search, latent-feature bypass or new risk
penalty. Actor motion is predicted once per real observed window and shared by
all candidates. Candidate robot motion does not write into actor memory.

Observed-state value training consumes geometry at 0/.25/.5/1/2 seconds. Native
one-step successor evaluation shifts those points by .25 seconds. The predictor
supplies offsets through 2.25 seconds, avoiding untrained horizon extrapolation.
Extrapolating candidate robot velocity constructs features; it does not force
that command to execute for two seconds or define optimal action labels.

Motion inputs use actor displacement relative to its latest legally represented
position, velocity, radius, observation age/validity and presence. No pedestrian
goal, simulator spawn, numeric ID or hidden ORCA state is available to the model.
All learned forecasters also receive the same legal current absolute actor
position and pooled current relative-neighbour geometry/velocity/radius. This
does not require initial spawn/goal access and prevents a deliberately weak
current-frame prediction baseline. No translation-invariance claim is made for
this complete current-conditioned estimator.
The current robot/spatial consumer retains the parent's legal state convention.

## Training and Fair Comparison

- Four common-interface arms: fixed CV, learned current-frame, GRU and KDA.
- Same value-consumer architecture and initialization within each paired seed.
  Learned predictors initially reproduce CV exactly; they must learn a useful
  difference. Temporal parameter counts differ and will be reported, not hidden.
- Same original 128 successful five-person circle ORCA episodes, no new data
  collection or privileged demonstration labels.
- 50 IL epochs, 3,000 online MC episodes, batch256 and four updates per episode.
- Prediction supervision uses only subsequently visible positions of previously
  known actors in the same completed episode. Hidden or not-yet-born actors,
  episode boundaries and unavailable terminal tails are masked out.
- Loss: original MC-return MSE plus0.1 times mean squared velocity-normalized
  forecast-displacement error. CV has no learned forecast loss. The decoder is
  part of the actual deployed control path, not an unused auxiliary head.
- Fixed seeds419/443/467/491. All arms within a seed use the same host/software.
- Only circle/5-person training. Held-out native circle/square and5/10/20 people,
  cases80000-80031, with no outcome-based selection. Cases81000-81031 and fresh
  seeds are reserved for confirmation, not used to tune this version.
- Final fixed-budget checkpoints are reloaded before evaluation. Existing V8
  contextual-GRU final weights are separately re-evaluated on the same new cases;
  old-case SR is never compared with new-case SR.

No simulator, action support, reward, control/observation clock, inherited CV
immediate-risk filter or smoothing setting is altered. Legal future measurements
are labels only and never supplied during action selection.

## Prior Boundary

Prediction/value coupling already exists in
[Relational Graph Learning](https://arxiv.org/abs/1909.13165) and its
[official code](https://github.com/ChanganVR/RelationalGraphLearning).
Future features and policy learning also appear in
[NavThinker](https://arxiv.org/abs/2603.15359).
Therefore this shared interface is an experimental baseline, not a novelty claim.
The experiment asks whether a specific temporal estimator has an empirical
navigation advantage here. Any eventual paper must identify a further supported
mechanism difference; replacing GRU with KDA alone is not that contribution.

## Decisions and One Allowed Rescue

Primary endpoint: equal-weight10/20-person SR across both geometries. A candidate
requires >=3pp mean gain, positive paired differences in at least3/4 seeds, <=1pp
collision increase, <=2pp timeout increase and <=10% successful-time increase.
All safety/progress endpoints and individual cells are reported even when the
primary endpoint improves. These are development criteria, not final proof.

1. GRU and KDA improve against shared simple controls and the original parent:
   the forecast-to-control route earns further study; assess KDA's extra value.
2. Only prediction improves: diagnose forecast consumption/value training, not
   automatically the memory unit. An action change alone is not success.
3. Simple controls match navigation: stop this complex mechanism version.
4. KDA retains a paired advantage over every control: perform fresh confirmation
   before organizing a method claim or expanding the benchmark.

At most one rescue is permitted, only after all Prototype-A arms finish and a
specific failed contract/mechanism is identified. It must name what failed and
what is changed. Any revised consumer applies to all common-interface arms and
is retrained under a new frozen protocol. No KDA-only gate sweep, outcome-driven
hyperparameter search or automatic V9/V10 series.

## Prototype-A Evidence

Prototype-A is complete: all16 runs have50 IL epochs,3000 RL episodes and192
reloaded-weight evaluations. The completed contract passes124 tests,
121 passed and three existing optional legacy-asset
skips. Each arm completed a short two-demonstration, one-IL-epoch, two-RL-episode
training/save/reload/native-episode smoke. All four smoke evaluations timed out;
these intentionally tiny pipeline checks are not method-performance evidence.
Full16-run training used419/443/491 on the4090 and467 on the3060.

The A control connection was committed and uploaded to shixu as f3aaae9. Its
completed results, scientific source and protocol are archived; the live source
now implements the one permitted B repair. Offline diagnostics do not alter
either training run.

The diagnostic cohort is fixed independently of method outcomes: the first four
development cases per geometry/population cell, replaying the archived commands
of parent seed419. All24 replays reproduce terminal event, time, path and minimum
clearance. There are591 uniformly sampled control states and34,990 available
later-visible actor/time labels. All learned arms and both IL/final phases use
these identical legal histories. It measures prediction accuracy and the
effect of substituting CV forecasts while holding each model's value weights
fixed. A substitution changes the model's input distribution; action differences
are not an improvement estimate or a replacement for trained closed-loop results.

Initial IL-only diagnostics for seeds419/443/467 show pooled forecast ADE of
0.2737/0.2729/0.2758m for Current,0.3378/0.3402/0.3427m for GRU and
0.3278/0.3370/0.3353m for KDA, versus the common CV0.1537m. This does not establish
the final RL result or its cause. Predictions do enter action selection: replacing
them with CV changes119-269 of591 filtered candidate selections in these models.
The fixed source/protocol is not revised in response to these intermediate data.

| Common-interface arm | Parameters |
| --- | ---: |
| CV | 103,169 |
| Current-frame | 109,879 |
| GRU | 308,279 |
| KDA | 275,839 |

The laptop completed the frozen parent re-evaluation:768 episodes across the
same new cases, all four seeds and all six geometry/population cells. Pooled
SR/CR/timeout are86.85/2.34/10.81%; primary10/20-person SR is85.16%. Primary SR
by seed is86.72/81.25/95.31/77.34%. This is a reference result, not a forecast
method result. It used CPU/PyTorch2.4.1; new arms use the original per-seed GPU
hosts, so absolute timing must not be compared across these hosts. Numerical
CPU/GPU evaluation sensitivity must be checked before a close method claim.

Matched-GPU parent re-evaluation is now complete. Seeds419/443/467 retain all
CPU terminal outcomes; seed491 changes one20-person square case from success to
timeout. The final reference primary SR is84.96%, not the preliminary CPU85.16%.
Both raw evaluations are retained. No threshold, test case or training budget is
changed. Each parent and new arm is evaluated using the same seed-specific GPU,
PyTorch version and native evaluator.

Bounded failure diagnosis: on eight identical64-sample shared-IL batches, the
completed seed467 GRU forecast decoder has value-versus-weighted-prediction
gradient cosines from-0.829 to-0.467; value-gradient norms are81-137 times the
weighted prediction norms. Current has six of eight negative cosines and norm
ratios30-65. These are local gradient measurements, not reconstruction of the
past AdamW trajectory or proof that this caused the navigation loss. KDA has
eight of eight negative cosines on the same batches (-0.733 to-0.480).
These measurements preceded the B repair and did not establish its outcome.

## Completed Prototype-A Result

Primary endpoint:512 episodes per arm,10/20 people, both native geometries,
four paired training seeds. All768 episodes per arm are preserved separately.

| Model | Primary SR (%) | Collision (%) | Timeout (%) | SR by seed419/443/467/491 (%) |
| --- | ---: | ---: | ---: | --- |
| Original contextual-GRU Parent | 84.96 | 3.52 | 11.52 | 86.72 /81.25 /95.31 /76.56 |
| A: CV | 75.20 | 14.45 | 10.35 | 79.69 /73.44 /78.12 /69.53 |
| A: Current | 75.59 | 13.48 | 10.94 | 80.47 /75.00 /71.09 /75.78 |
| A: GRU | 80.86 | 7.42 | 11.72 | 82.03 /78.12 /78.12 /85.16 |
| A: KDA | 75.20 | 6.05 | 18.75 | 76.56 /82.03 /62.50 /79.69 |

Verdict: NO_VALIDATED_KDA_ADVANTAGE_THIS_VERSION. Not equivalence, not a rejection
of temporal modeling. The source, protocol, demonstrations, checkpoint hashes,
complete epoch/episode logs and exact evaluation-case sets passed the artifact
audit. Raw results and frozen_source.tar.gz are retained under forecast_control_a.

### Correction to Our Experimental Interpretation

A kept the environment and training/action pipeline but **did not retain the
original temporal value representation**. It replaced that representation with
a forecast-coordinate bottleneck and a new19-feature consumer. Its CV arm also
underperformed the strong Parent. Consequently A cannot settle whether improved
forecasts help when connected to the inherited value network.

The sole permitted rescue restores the original contextual-GRU critic unchanged.
CV must reproduce its candidate scores and executed actions. Learned motion
estimates only correct pedestrian positions in the native one-step hypothetical
successor queries; the original immediate reward, CV safety/risk filter, smoothing
and action support stay frozen. Real observed histories still train the original
critic with scalar MC returns. Physical forecasts train only on later legal
motion labels, so value gradients cannot repurpose coordinates as latent codes.
Critic and predictor gradient clipping is separate to preserve the critic's IL
updates. There is no new gate, hidden-goal input or latent-feature bypass.

This is **one interface repair**, applied to Current/GRU/KDA under the same data
and budgets. It changes both the consumer and gradient routing, so an A-to-B
improvement cannot isolate gradient conflict as the cause. CV may reuse the
verified original Parent checkpoints only after exact interface-parity checks;
that reuse must not be mislabeled as new training. No second rescue is allowed.
If B retains no practically useful advantage over strong controls, this bounded
attempt stops; further architecture versions are not automatically authorized.

## Prototype-B Frozen Contract

```text
lawful actor observations -> CV / Current / GRU / KDA physical forecast
                                       -> .25s human-position residual
native candidate successor ------------> corrected hypothetical geometry
real23-frame prefix -------------------> unchanged contextual-GRU critic
                                       -> original scalar value/lookahead
                                       -> original filter/smoothing/action
```

The physical head still predicts and receives supervision through2.25s, but
**the inherited one-step decision consumes only its .25s estimate**. This is not
a new multi-step planner. A negative result must be scoped to this connection,
not all possible uses of improved longer-horizon forecasts. All arms retain the
same original value memory; GRU/KDA names now refer to the added motion estimator,
not replacement of that critic. No pure-backbone or parameter-matching claim.

| B model | Total parameters | Newly trained predictor parameters |
| --- | ---: | ---: |
| Native CV/original Parent | 333,313 | 0 |
| Current | 340,023 | 6,710 |
| GRU | 538,423 | 205,110 |
| KDA | 505,983 | 172,670 |

The original forecast-input contract is unchanged. KDA explicitly decays over
elapsed observation gaps, whereas GRU compacts measured frames without an
explicit gap feature. A future positive KDA mechanism claim would need a strong
gap-aware GRU comparison; this bounded attempt does not silently treat their
clock handling as equivalent. Neither unit gets hidden future measurements.

Preflight:129 tests,126 passed and three existing optional-asset skips. Unit
checks include exact CV/zero-residual candidate-score parity, unchanged critic
AdamW updates on shared replay despite auxiliary prediction gradients, no
value-to-predictor gradient, no mutation of real history and save/load parity.
With the original trained seed419 critic,30 native steps reproduce all80 scores
and executed commands exactly. Full192-case GPU parity is required for each of
the four reused CV reference checkpoints. Three2-episode smoke runs all time
out; these only validate train/save/reload/evaluation plumbing, not performance.

B trains Current/GRU/KDA and a fresh CV control from scratch for the unchanged
50 IL +3000 RL budget, four paired seeds. A separate strong CV reference reuses
verified original full-budget trained weights; its new training time is zero
and original training time is recorded separately. No reference learning log
is invented. All final B weights are reloaded for the same192 cases per seed.
The4090 hosts419/443/491 in an owned RAM directory and
the3060 hosts467. A premature remote launch occurred before data transfer ended;
it failed before any worker/training and was relaunched after transfer completion.

B scientific source SHA256:

ca46069a6d8f70e5881b8ac227be7fbd6a54f710620d000825ee7fd2a62b1671

B protocol SHA256:

cbbf9f397ea8effcfa1f868619d6c367a10754eab7b0b04c3f5267dd8b7f636a

B shared demonstration archive SHA256 (same original episode contents):

2b8b7adfa38e34a4d53dbdcf8931c5270716a047b7bb2785d27449899bcd00ff

The scientific source and protocol remained frozen throughout every B run.
Only evidence collection, report updates and non-training diagnostic utilities
changed. This consumed the sole rescue; no additional version is authorized.

Fairness addendum before any B navigation result: within seed467 the completed
Current and GRU IL critic weights are exactly equal, but differ from the reused
original Parent IL weights (largest parameter difference0.00449). The old Parent
used a reused prior-IL checkpoint. That difference cannot be assumed harmless or
attributed to prediction. It also does not show auxiliary gradients altered the
new critic: the new-arm equality and controlled optimizer test show otherwise.
Therefore an additional **fresh CV control**, initialized and fully trained with
the same B code/data/seed/budget, is required. It uses no new mechanism and cannot
overwrite the reused strong Parent. Four fresh CV runs are saved separately under
outputs/forecast_control_b_fresh_cv; their50 IL and3000 RL episodes are audited
like the learned arms. B must beat both fresh CV and the original strong Parent.
This is control completion, not another rescue or changed scientific source.

The four reused CV references completed all192 cases with exact archived GPU
command/outcome parity, not merely matched success percentages. New Current,
GRU and KDA IL critics also have exactly equal tensors within each seed. For
all four fresh-CV IL runs, their critic tensors are exactly equal to the
corresponding new learned-arm critics; the final artifact audit checks this
again and records the IL checkpoint hashes.
This checks initialization/update fairness; it does not imply final trained
policies or online replay remain equal.

## Completed Prototype-B Result

All16 new runs completed50 IL epochs and3000 RL episodes with finite logged
losses. The final audit verifies the frozen scientific source, protocol, shared
demonstrations, checkpoint hashes, exact192-case sets, common seed-specific
host/software, IL critic tensor equality and original-reference command parity.
Primary results below cover512 episodes per arm; overall results cover768.

| B arm | Primary SR (%) | Collision (%) | Timeout (%) | Overall SR (%) | Primary SR by seed419/443/467/491 (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| Reused original Parent / native CV | 84.96 | 3.52 | 11.52 | 86.72 | 86.72 /81.25 /95.31 /76.56 |
| Fresh matched CV | 86.13 | 1.95 | 11.91 | 87.50 | 85.94 /80.47 /93.75 /84.38 |
| Current motion estimator | 87.11 | 2.54 | 10.35 | 88.93 | 85.16 /87.50 /92.19 /83.59 |
| GRU motion estimator | 83.40 | 2.15 | 14.45 | 85.81 | 82.81 /78.91 /91.41 /80.47 |
| KDA motion estimator | 86.52 | 3.52 | 9.96 | 87.50 | 92.97 /80.47 /93.75 /78.91 |

Population-specific SR pools the two geometries equally,256 episodes per entry.
No result is selected by density or geometry.

| B arm | 5-person SR (%) | 10-person SR (%) | 20-person SR (%) |
| --- | ---: | ---: | ---: |
| Reused original Parent / native CV | 90.23 | 88.67 | 81.25 |
| Fresh matched CV | 90.23 | 89.45 | 82.81 |
| Current motion estimator | 92.58 | 91.41 | 82.81 |
| GRU motion estimator | 90.63 | 87.89 | 78.91 |
| KDA motion estimator | 89.45 | 90.63 | 82.42 |

KDA-minus-control primary SR differences, paired by training seed:

| Control | Mean SR difference (pp) | Nominal95% seed-paired interval (pp) | Positive seeds | Collision difference (pp) | Development pass |
| --- | ---: | --- | ---: | ---: | --- |
| Original Parent | +1.56 | [-4.09,7.21] | 2/4 | 0.00 | No |
| Fresh CV | +0.39 | [-7.76,8.54] | 1/4, two ties | +1.56 | No |
| Current | -0.59 | [-11.20,10.03] | 2/4 | +0.98 | No |
| GRU motion estimator | +3.13 | [-4.80,11.05] | 3/4 | +1.37 | No |

Against GRU, KDA reduces timeout by4.49pp but exceeds the allowed collision
increase of1pp. Against fresh CV, timeout falls1.95pp while collision rises1.56pp.
Neither is an acceptable safety-progress improvement under the frozen criteria.
Current is+0.98pp versus fresh CV with only1/4 positive seeds, and+2.15pp versus
the original Parent with2/4 positive seeds. No learned forecast arm establishes
the required advantage over both CV controls. These nominal intervals use four
training seeds as units, are exploratory and are not multiplicity-adjusted;
failure to pass is not statistical equivalence or proof of zero true effect.

### Prediction and Actual Consumption

The same preselected24-episode,591-root cohort is used for every final checkpoint.
Labels remain later-visible positions only; unavailable hidden/terminal futures
are not silently filled with simulator truth. The four-seed pooled ADE below
does not claim accuracy on all occluded trajectories.

| Motion estimator | Consumed .25s ADE (m) | 1s ADE (m) | 2s ADE (m) | All future offsets ADE (m) | CV-swap filtered selections by seed, out of591 |
| --- | ---: | ---: | ---: | ---: | --- |
| CV | 0.0161 | 0.1037 | 0.2899 | 0.1537 | 0 /0 /0 /0 |
| Current | 0.0216 | 0.1346 | 0.3514 | 0.1908 | 7 /15 /12 /10 |
| GRU | 0.0221 | 0.1321 | 0.3427 | 0.1862 | 26 /10 /16 /23 |
| KDA | 0.0224 | 0.1217 | 0.3100 | 0.1701 | 23 /11 /25 /11 |

KDA predicts longer futures more accurately than the learned Current/GRU
estimators on this cohort, but not more accurately than CV. At the .25s horizon
actually consumed by B, all three learned estimators have greater ADE than CV.
CV substitution changes some filtered candidate selections, proving the forecast
is consumed, not that those changes are beneficial. This does not establish
that forecast error alone caused the navigation result: the four policies also
collect different online trajectories and train their critics on those returns.

### Cost and Reproducibility

Standardized timing uses the same local RTX3060/PyTorch2.1.0,60 fixed legal roots,
20 warmups and three repetitions. It measures complete80-action scoring/filtering
and transfers, not simulator time or smoothing. No navigation training or other
timing worker runs concurrently. Desktop graphics remain present; absolute
numbers are descriptive. One overlapping timing launch was preserved as
standardized_timing_overlap.json and excluded in favor of serial remeasurement.

| B arm | Parameters | Range of seed median scoring time (ms) | Sum of worker training wall hours |
| --- | ---: | --- | ---: |
| Reused original Parent / native CV | 333,313 | 3.05-3.10 | 0 new;3.02 historical |
| Fresh matched CV | 333,313 | 3.02-3.17 | 4.49 |
| Current motion estimator | 340,023 | 3.84-4.08 | 5.47 |
| GRU motion estimator | 538,423 | 4.80-4.93 | 6.59 |
| KDA motion estimator | 505,983 | 10.57-10.84 | 8.53 |

The16 new B runs total25.08 worker wall hours, not25.08 elapsed hours or measured
GPU hours. Concurrent training loads differ, so these training times cannot
isolate architecture compute. The standardized inference replay shows no KDA
efficiency advantage. KDA additionally holds8192 state floats per actor versus
GRU's256; neither parameter count nor memory capacity is claimed matched.
The full suite finishes129 tests:126 passes and three existing optional-asset
skips. Raw final weights, IL weights, complete learning logs, episode records,
prediction diagnostics and timing arrays remain local in the ignored output
directories. Scientific source remains unchanged from the B freeze above.

### Stop Decision

The one permitted repair has been spent and none of the learned estimators
passes against both strong CV controls. Stop this bounded version. Reserved
fresh seeds521/547/569/593 and cases81000-81031 remain unused. No KDA-only rescue,
new gate, additional scene or retrospectively relaxed threshold follows.

Established: a common lawful forecast interface can be trained and connected to
the unchanged navigation critic; it changes some candidate decisions; this full
matched experiment does not validate an acceptable KDA navigation advantage.
Not established: KDA is generally incapable, history is valueless, MC training
is the unique bottleneck, or a planner consuming longer-horizon forecasts would
have the same result. A future proposal would require a distinct, justified
mechanism and new authorization, not renaming this failed attempt.

## Prototype-A Archive and Preflight History

A frozen scientific source SHA256:

ae93ca700d995231df0935ec554bdf834c1312c316c5382f83b17b7273212ae3

A frozen protocol SHA256:

7c12cf45a721b0796223b9b503c489c37306c446645bbe90c0b2b615f0a3971c

A shared demonstration archive SHA256:

7c314d387b972037a6ff9971505c70153f19b54f81b35af1837a0f961832c84c

Pre-results interface correction: the first implementation withheld current
neighbour geometry from the forecast head. That could make the Current control
artificially weak. Its incomplete training was interrupted, archived and charged
separately; no final result was used to choose the correction. All common arms
restart after adding identical legal current-crowd conditioning. This is contract
completion; the result-based rescue was subsequently spent on B, not on these
preflight runs.
The12 interrupted runs had about234 wall seconds each, some at IL36 and others
at RL71-204; none reached3000 episodes or produced a final result. Their source
archive, partial IL weights and logs remain in outputs/forecast_control_initial_contract
and outputs/forecast_control_initial_local. They are not reused as final-IL
initialization or counted as completed formal runs.

Artifacts are saved locally under:

/home/abc/workspace/shixu/outputs/forecast_control_a/

B artifacts and the complete remote backup are saved locally under:

/home/abc/workspace/shixu/outputs/forecast_control_b/

/home/abc/workspace/shixu/outputs/forecast_control_b_fresh_cv/

/home/abc/workspace/shixu/outputs/forecast_control_remote_backup_20261004/

The4090 used an owned RAM workspace, without downloads to its system disk. After
all workers exited successfully, all289 files in that workspace were backed up
and individually SHA256-verified against the remote originals. The manifest is
stored as server_backup_manifest.json in the B artifact directory. Only then was
the owned RAM directory removed; no existing remote environment was modified.
The laptop handled diagnostic/reference evaluation and the local3060 handled
one complete paired seed. Frozen-source archives and raw negatives remain local;
large checkpoints are not committed to Git.

## Evidence Follow-Up: Fit, Consumption and Realized Action Value

Completed offline on the existing four IL/final checkpoint pairs. No navigation
training, architecture change, new scenario generator or third repair was made.
The scientific core SHA256 remains the B freeze. The diagnostic source is
experiments/forecast_evidence.py; all raw rows, forecasts' errors, candidate
scores, selections, restoration checks and continuation outcomes remain under:

/home/abc/workspace/shixu/outputs/forecast_control_b/evidence_diagnostic/

### 1. Training Fit Versus Independent Forecast Evaluation

Shared inputs are identical across predictors. Every fourth frame is sampled
from all128 archived successful IL episodes, giving1609 training roots. The
independent case IDs80000-80031 do not overlap the IL cases. Two visitation
controls are reported separately:

- Same ORCA teacher as IL:32 native5-circle cases,401 roots,29 successes and
  three collisions. All cases are retained; a success-only sensitivity summary
  is also archived because IL selected successful demonstrations.
- Original Parent419 commands:32 native5-circle cases plus four cases in each
  of the other five geometry/population cells,1151 roots total. Commands replay
  with identical termination/time/path and matching legal tokens. This is a
  different visitation policy from the ORCA demonstrations, not a clean
  overfitting-only test. These cases were already used for development
  evaluation; they are independent of training, not fresh confirmation cases.

The online MC replay was never serialized. These data therefore cannot measure
fit on the actual3000-episode RL training replay or establish an optimization
failure there. Forecast labels cover later lawful visible measurements;
unavailable terminal tails are masked. Hidden simulator truth is not used for
these fit metrics.

The following are measured means across four paired seeds. Loss is exactly the
forecast training objective: two-coordinate displacement MSE divided by elapsed
time squared, averaged over the nine valid forecast horizons. Its units are
(m/s)^2, not ADE.

| Checkpoint | Predictor | Seen IL loss | Independent ORCA loss | Independent ORCA0.25s ADE(m) | Independent ORCA2s ADE(m) |
| --- | --- | ---: | ---: | ---: | ---: |
| IL | Current | 0.029218 | 0.029510 | 0.028426 | 0.387550 |
| IL | GRU | 0.022803 | 0.023157 | 0.026869 | 0.331141 |
| IL | KDA | 0.018625 | 0.020586 | 0.027745 | 0.319588 |
| Final | Current | 0.034230 | 0.033589 | 0.026949 | 0.385844 |
| Final | GRU | 0.031694 | 0.030953 | 0.025930 | 0.353090 |
| Final | KDA | 0.029683 | 0.028690 | 0.025093 | 0.327661 |

KDA's forecast objective is lower than Current's on both datasets in all four
seeds at both phases. The independent ORCA long-horizon advantage is not just a
training-set effect. Final checkpoints have worse loss on the original IL data
than IL checkpoints in all learned arms; conversely their Parent-visited test
errors decrease. This measures a phase-dependent redistribution of errors. It
does not identify its cause without the missing online replay.

Final forecast errors on the32 independent Parent-visited5-circle cases:

| Predictor | 0.25s ADE(m) | 2s ADE(m) | Forecast loss |
| --- | ---: | ---: | ---: |
| CV | 0.015388 | 0.277600 | 0.027079 |
| Current | 0.019980 | 0.296111 | 0.019765 |
| GRU | 0.019577 | 0.270116 | 0.018223 |
| KDA | 0.018990 | 0.234101 | 0.016668 |

KDA beats Current at both forecast horizons in all four seeds in this cohort.
The previous statement that KDA is worse at0.25s must be restricted to the
earlier six-cell pooled workload. In the20 other-cell episodes, final Current
and KDA0.25s ADE are0.022209 and0.023148m; their2s ADE are0.363932 and0.324216m.
Therefore short-horizon performance depends on the evaluated workload; it is
not uniformly worse for the temporal predictor. CV still has the lowest0.25s
error in the Parent-visited cohorts.

A frozen-weight input intervention removes older lawful frames while leaving
the current frame, current neighbours and labels unchanged:

| Predictor/input | Independent ORCA0.25s/2s ADE(m) | Parent5-circle0.25s/2s ADE(m) |
| --- | --- | --- |
| KDA24 frames | 0.025093 /0.327661 | 0.018990 /0.234101 |
| KDA last3 frames | 0.031678 /0.424772 | 0.024462 /0.329058 |
| KDA current frame only | 0.039645 /0.468802 | 0.029400 /0.349584 |
| GRU24 frames | 0.025930 /0.353090 | 0.019577 /0.270116 |
| GRU current frame only | 0.028116 /0.402207 | 0.022018 /0.330513 |

Current outputs are unchanged by this intervention. This demonstrates useful
history dependence in these frozen predictors; it is not a retrained last-k
baseline, proof of optimal fitting, or proof that KDA matrix memory is necessary.
Measured training/test errors do not support declaring history redundant or
the temporal predictor wholly unlearned.

### 2. Does Forecast Consumption Target High-Error Dangerous States?

The bridge directly reads only0.25s predicted positions. Its2s predictions do
not enter candidate queries, rewards or safety filters. The longer targets
still influence shared predictor parameters during training; they are not
directly consumed by control.

On the original591 shared roots, four KDA-trained critics and four fresh-CV
critics are each frozen. CV, Current, GRU, KDA and true-one-step positions are
substituted into the same critic/history/candidate arrays. Four seeds reuse the
same roots and are not2364 independent states.

| Frozen consumer | KDA/CV raw ranking winner changes | KDA/CV filtered winner changes |
| --- | ---: | ---: |
| KDA-trained critic,591 roots x4 seeds | 77 /2364 | 69 /2364 |
| Fresh-CV-trained critic,591 roots x4 seeds | 113 /2364 | 110 /2364 |

For the KDA-trained consumer, the69 changed root/seed conditions have mean
one-step position error0.050447m versus0.035866m for the2295 unchanged conditions.
These are state means over currently active tracks using diagnostic simulator
positions, including retained hidden actors; they are not the visible-label ADE
from the preceding tables. Current lawful surface clearance<=0.8m occurs in
32/69 changed conditions versus756/2295 unchanged conditions:46.38% versus32.94%.
Only1/69 changed conditions has a CV-selected native clearance below the fixed
0.2m safety margin. For fresh-CV consumers, corresponding mean errors are
0.045774 versus0.035829m and near-state counts42/110 versus746/2254.

Thus changes coincide with larger measured errors on average and are somewhat
more frequent in near encounters, but are not exclusively danger-state changes.
These associations do not prove prediction error caused an episode failure.
The modest raw-to-filtered count reduction also does not identify the safety
filter as the unique cause of missing navigation gains. The direct
long-horizon-to-one-step consumption mismatch is established by the code path;
its share of the observed performance difference is not isolated.

### 3. Fixed Critic, True One-Step Positions and Actual Consequences

Selection rules were saved before calculating any new outcomes:

- Primary: one root per original24 episodes, first sampled root after2s with
  lawful surface clearance<=0.8m; otherwise the closest sampled root after2s.
- Secondary: the first sampled root after2s in each episode where true-position
  and CV substitutions select different actions in any fixed consumer.20 roots
  qualify. This set is explicitly action-selected, not an unbiased population
  estimate. Five roots overlap primary;39 distinct worlds are evaluated.

World restoration was checked against archived commands and rewards at all39
roots. Maximum one-step human-position error is4.47035e-7m and reward errors are
zero. Robot-invisible native dynamics make human future positions independent
of the substituted root command. Truth is supplied only for currently tracked
actors; it does not reveal unseen pedestrians or enter deployed models.

Each alternative uses the identical critic weights, lawful history,80 native
candidates, original reward, CV safety/risk filtering and common previous
executed command for smoothing. Only forecast positions are swapped. The
selected command is executed for one0.25s step, then a common lawful ORCA rule
continues to termination. Unbraked ORCA is a separately reported sensitivity
control.510 unique native continuation branches were executed, reusing exact
duplicates across variants. Discounting uses the original gamma0.99; progress
is recorded but not substituted for the original reward. These are selected
action Q^pi outcomes, not Q*, a full truth-trained navigation benchmark, or
complete deployed-policy success rates.

Primary outcomes under the common ORCA continuation,24 roots x4 seeds per
consumer. Counts repeat root worlds across critic conditions; they must not be
interpreted as96 independent evaluation episodes.

| Frozen consumer | Prediction | Success/collision/timeout | Mean discounted return |
| --- | --- | --- | ---: |
| Fresh CV | CV | 73 /1 /22 | 0.456462 |
| Fresh CV | Current | 73 /1 /22 | 0.456330 |
| Fresh CV | KDA | 73 /1 /22 | 0.456462 |
| Fresh CV | True0.25s positions | 73 /1 /22 | 0.456379 |
| KDA | CV | 70 /4 /22 | 0.417699 |
| KDA | Current | 70 /4 /22 | 0.417699 |
| KDA | KDA | 70 /4 /22 | 0.417615 |
| KDA | True0.25s positions | 70 /4 /22 | 0.417564 |

No primary root/seed condition changes terminal outcome, under either
continuation. Tiny time/return changes remain archived rather than promoted to
a method gain. In the secondary set, fresh-CV consumers retain64 successes,
zero collisions and16 timeouts for all four substitutions. KDA-trained
consumers have64/1/15 under CV and Current versus63/1/16 under KDA and truth.
The changed terminal is one20-person square root,case80002,tick20,critic seed467:
CV's action reaches the goal; KDA and true-position substitutions time out.
The same terminal difference persists with unbraked ORCA. It is not averaged
away or treated as a safety benefit. True-position mean-return changes versus
CV are nonpositive in every primary/secondary consumer aggregate for both
continuations.

This supplies no positive outcome headroom for improving positions within this
specific fixed one-step bridge on these roots. It does not establish that better
prediction is useless generally. Truth is perfect one-step position, not full
future state: velocity/masks/age, immediate reward and safety-filter geometry
remain the parent quantities. Substituted queries are not co-trained with the
fixed critic; distribution compatibility is not independently established.
One-step interventions with ORCA continuation also do not measure repeated
closed-loop forecast use. These are explicit limits, not inferred explanations
for the negative result.

### Answer and Research Boundary

Established: temporal predictors learned measurable forecast improvements,
including on independent lawful histories; removing history worsens their
frozen predictions; the bridge directly consumes only a single short horizon;
and even exact positions at that horizon do not improve terminal outcomes in
the evaluated primary roots. The earlier pooled short-term comparison must not
be generalized to every density/geometry/visitation policy.

Not established: a unique optimization, generalization, memory or critic cause;
matrix associative memory necessity; redundant historical information; or a
universal absence of prediction-to-control value. Continuing to optimize only
this predictor has no demonstrated navigation payoff in this frozen bridge.
No KDA V9/V10, new consumer or new navigation training follows this diagnostic.
All133 tests finish successfully with three existing optional-asset skips.

## Frozen Multihorizon Failure-Repair Experiment (5 October 2026)

The prespecified two-second forecast / CV-tail scorer is complete. One native
root command lasts0.25s; the unchanged fresh-CV419 policy then continues in the
restored real environment to the original deadline. In shadow scoring, the
same policy responds to the forecast for eight steps, followed by a fixed CV
tail. Scores are full original discounted task returns, not two-second
progress or an untrained multihorizon value query. All80 native root actions,
filtering, smoothing, masks and reward remain fixed. Shadow worlds contain
only currently active known actors; even truth supplies only their two-second
positions, not unseen people, hidden goals or perfect long-term dynamics.

Selection was frozen before outcome evaluation: two terminal-relative anchors
in each of ten existing failed cases, plus six separate successful-case guards.
Four predictor training seeds share one frozen navigation policy. Correlated
anchors are not independent episodes; repair counts are not a deployed SR.

| Prediction | Failure roots: success / collision / timeout | Mean return change from native continuation |
| --- | --- | ---: |
| CV | 6 /4 /10 | +0.366347 |
| Two-second position truth | 6 /4 /10 | +0.366949 |
| Current,419/443/467/491 | 1/4/15;5/2/13;5/2/13;3/3/14 | +0.050728;+0.250618;+0.258529;+0.140861 |
| GRU,419/443/467/491 | 2/4/14;4/2/14;1/4/15;4/2/14 | +0.102594;+0.184501;+0.045288;+0.197591 |
| KDA,419/443/467/491 | 3/3/14;2/4/14;3/2/15;4/3/13 | +0.152533;+0.101212;+0.139539;+0.213673 |

CV repairs six anchors in four distinct cases. Truth repairs the same six.
KDA-minus-CV return differences are-0.213815/-0.265135/-0.226808/-0.152674;
all four are negative. KDA-minus-matched-Current success counts are+2/-3/-2/+1.
This is not a stable advantage over the strong simple controls. Every arm
preserves all six guard successes, which is a limited guard, not a broad
non-degradation guarantee.

No known execution/parity defect remains: all26 archived native continuations
match full command traces, return, termination and elapsed simulated time.
Maximum root batch/scalar score discrepancy is2.224e-5; near ties use native
scalar scoring, and every selected shadow is replayed with native predict().
All selected traces agree; no complete native-rescoring fallback is needed.
The initial float32-restoration and numerical-precheck attempts remain separate
and are not pooled into this cohort.

The scorer has demonstrable approximation limits: shadow/actual terminal
disagreements occur in4/20 CV and4/20 truth repairs, and5-6/20 KDA repairs.
Consequently two-second truth is not a full-world upper bound. There is actual
action-improvement space (CV repairs), but no qualified KDA increment in this
specified consumer. Neither result proves history is useless or identifies a
unique KDA failure cause. This forecast-bridge attempt is closed without a new
predictor, reward, scorer rescue or navigation training.

Per-arm failure-root scoring medians are5.28-7.11s on the4090, well beyond the
native0.25s control interval. The sum of recorded root workloads is3598.61s;
the last two independent roots ran concurrently, so this sum is not wall time
and timings are descriptive, not an isolated hardware benchmark. Raw80-action
scores, forecasts, real traces, manifests, checkpoint hashes and logs are in
outputs/multihorizon_control. The scoped suite finishes with136 passes and
three existing optional-asset skips.
