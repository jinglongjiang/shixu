# Forecast Control and State Reconstruction

Forecast consumers and physical-state reconstruction did not establish a validated KDA navigation advantage in these tested implementations. This is not a universal rejection of history or KDA.

## Archive Policy

This is a classification-only consolidation. Original stage text, numbers, negative results and withdrawn claims are preserved byte-for-byte below. Later closure reports supersede earlier proposed next steps. The historical README is retained in thematic sections. Snapshot variants are labeled by their original paths. Frozen protocols and autoresearch_nav/program.md are not changed.

## Stage Index

- [FORECAST_CONTROL_EXPERIMENT.md](#stage-1)
- [TEMPORAL_DECISION_STATE_REBUILD.md](#stage-2)
- [README.md (historical README section 4)](#stage-3)
- [outputs/forecast_control_remote_backup_20261004/FORECAST_CONTROL_EXPERIMENT.md](#stage-4)
- [outputs/forecast_control_remote_backup_20261004/README.md (historical README section 4)](#stage-5)


---

<a id="stage-1"></a>

## Source: FORECAST_CONTROL_EXPERIMENT.md

Original full-source SHA-256: fdeecb188647d55887b698f99b41a120d5012ae9b677f7a388b217c1cf1b85e4

<!-- BEGIN PRESERVED SOURCE -->
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

<!-- END PRESERVED SOURCE -->


---

<a id="stage-2"></a>

## Source: TEMPORAL_DECISION_STATE_REBUILD.md

Original full-source SHA-256: 7156a2c58d78024a8b8ce41d8ee6b742caf393737ff8b1738becec08e45068a9

<!-- BEGIN PRESERVED SOURCE -->
# TEMPORAL DECISION STATE REBUILD

日期：2026-10-05，Asia/Seoul。结论来自冻结权重、原生轨迹回放和真实续跑，不是新导航模型的完整训练结果。

**最终判定：当前母体没有找到值得继续的KDA入口。**

范围：关闭本轮指定的预测消费者和当前物理状态修正入口；不宣布历史无用，不永久否定KDA，也不把冻结消费者当成所有时序表示的上界。

## 1. 启动条件：先完成多时域消费者

原实验已完成，才启动本续跑任务。续跑起点为 2026-10-05 03:48:39 KST，八小时上限为 11:48:39 KST。本轮允许的分支没有留下合格入口，提前进入汇总归档，不为凑满八小时启动随机训练。

### 1.1 冻结的实验是什么

- 一个共同消费者：fresh-CV419 最终导航权重，原 contextual-GRU 价值网络、80动作、奖励、动力学、风险筛选、平滑和观测规则不变。
- 四个预测器训练种子：419/443/467/491。它们共享上述导航消费者，**不是四个独立导航训练种子的比较**。
- 两秒预测，随后由末两点速度形成CV尾部；冻结策略在影子世界反馈续跑到原终止条件，用完整原始折扣奖励评分。
- 每个根动作只在真实环境执行0.25秒，随后共同原策略续跑到终止。不是持续执行根动作两秒，不用进度代替奖励。
- 20个失败根状态来自10个既有失败case，每case两个锚点；另有6个成功保护状态。选择在方法结果产生前保存。相关锚点不当成独立episode，挽救数不当作总体SR。
- 真值只提供当前已知、仍被保留行人的两秒位置；不提供未见行人、目标、完整未来动力学。两秒以后的CV尾部仍是近似。

### 1.2 实际续跑结果

以下每行均为同一20个失败根状态，终止顺序为成功/碰撞/超时；回报差相对原策略原生续跑。

| 预测输入 | 成功/碰撞/超时 | 原始折扣回报平均差 |
| --- | --- | ---: |
| CV | 6/4/10 | +0.366347 |
| 两秒位置真值 | 6/4/10 | +0.366949 |
| Current 419 | 1/4/15 | +0.050728 |
| Current 443 | 5/2/13 | +0.250618 |
| Current 467 | 5/2/13 | +0.258529 |
| Current 491 | 3/3/14 | +0.140861 |
| GRU 419 | 2/4/14 | +0.102594 |
| GRU 443 | 4/2/14 | +0.184501 |
| GRU 467 | 1/4/15 | +0.045288 |
| GRU 491 | 4/2/14 | +0.197591 |
| KDA 419 | 3/3/14 | +0.152533 |
| KDA 443 | 2/4/14 | +0.101212 |
| KDA 467 | 3/2/15 | +0.139539 |
| KDA 491 | 4/3/13 | +0.213673 |

CV和真值挽救同样6个锚点，涉及4个不同case。所有14臂保留6/6成功保护状态，但这不是完整部署后的不退化保证。KDA相对CV的平均回报差四种子均为负：-0.213815/-0.265135/-0.226808/-0.152674；相对同种子Current的成功数差为+2/-3/-2/+1。

**失败分类：存在实际动作改善空间，简单CV消费已取得收益；KDA没有保留超过强简单对照的增量。** 同时评分器存在失真：CV/真值各4/20次影子与真实终止不一致，KDA为5–6/20次。因此不能把真值当作完整上界，也不能把此结果归因于“历史没有信息”。

正确性核查：26/26原策略续跑的完整动作轨迹、终止、回报和时间复现；最大批量/单状态评分差2.224e-5，近似平分使用原生单状态评分，选中影子轨迹再用原生predict核查，未触发全量重评分。早期float32恢复和数值预检记录单独保留，没有并入正式结果。

多时域评分每臂在失败状态上的中位耗时5.28–7.11秒，远超过原0.25秒控制间隔。记录的根状态工作量合计3598.61秒；末两个根状态同卡并行，该和数不是墙钟时长，也不是隔离硬件负载的性能基准。

本消费者封账，不再修改预测桥梁或启动预测器救援。

## 2. 先查了什么旧证据

先读取正式遮挡V1–V8、零历史和stale-memory诊断，没有换名重开旧训练。

- V8主指标：物理KDA、上下文GRU、物理GRU均81.25%；零committed-history KDA为84.38%。汇总持平不是统计等效，KDA没有稳定配对优势。
- V6碰撞回放中碰撞对象在碰撞时均已可见；大量timeout末段附近没有隐藏人。它们不能直接归因于当时缺失的行人位置，也不能因此证明此前历史没有影响。
- V6已测过隐藏当前状态真值替换：小cohort里KDA的40/48成功保持不变，GRU为44/48→45/48。**当前p/v真值干预不是新算法贡献。** 本轮换用已经冻结的新消费者和事前选定的自然早期根状态作诊断，不把旧结果重跑或重命名成V9。
- 原在线MC完整replay没有保存。本轮从已存评估指令重建状态/标签，明确属于新的离线注释，不冒充原RL replay。

## 3. 找到了哪一种具体、自然发生的时序缺陷？

**已证实事实：遮挡期间已见行人的CV状态与真实p/v会有误差；重新可见后当前测量立即更新，没有观察到该跟踪器的可见状态更新迟滞。** 但尚未找到满足“同样完整当前输入、不同合法历史、需要不同合理动作”的配对决策缺陷。

全部192条既有fresh-CV419评估轨迹：167成功、23超时、2碰撞；17,072个控制帧、207,120个人帧。没有生成新场景。回放的终止、路径、时间、最小间距及原存档总回报192/192一致。

注意：原episode记录的return_为未折扣奖励总和；本轮根动作Q^pi诊断使用原gamma=0.99的折扣和。准备阶段曾误用折扣和核查原总和，已在产生方法结果前纠正记录定义，没有改实验数字。

### 3.1 自然频率

| 人数/几何 | episodes | 控制帧 | 有保留隐藏人的帧 | 隐藏人合法估计距离≤2m的帧 | 出现过该近身条件的episodes | 失败episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5/circle | 32 | 2,265 | 796 | 6 | 4 | 1 |
| 5/square | 32 | 2,991 | 766 | 5 | 3 | 6 |
| 10/circle | 32 | 2,434 | 1,634 | 72 | 17 | 0 |
| 10/square | 32 | 3,114 | 1,816 | 92 | 16 | 7 |
| 20/circle | 32 | 2,813 | 2,551 | 347 | 29 | 1 |
| 20/square | 32 | 3,455 | 2,641 | 324 | 29 | 10 |

共有33,319次保留隐藏actor记录、5,971次重新可见事件；846/17,072控制帧满足合法近身隐藏条件，约4.96%。这是该已训练策略访问分布下的频率，不是整个模拟器的普遍发生率，更不是遮挡导致失败的比例。

| 自然样本误差 | 跟踪器CV | 简单历史速度趋势修正 |
| --- | ---: | ---: |
| 所有保留隐藏actor的平均位置误差 | 0.11168m | 0.13573m |
| 所有保留隐藏actor的平均速度误差 | 0.14872m/s | 0.19733m/s |
| 合法近身隐藏actor的平均位置误差 | 0.06554m | 未另作分层结论 |
| 合法近身位置误差P90/P99 | 0.19051/0.56152m | 未另作分层结论 |

有误差和尾部误差，不等于这些误差实际造成了错误动作。此前Current已包含可见真实速度、当前邻居、遮挡年龄、测量/保留标记；原消费者本身也有23帧历史前缀。本轮没有把它人为改成弱无记忆对照。

## 4. 新接口与旧KDA方案的实质差别

只选择一个接口：**在原候选构造之前，替换已见且仍被保留的隐藏actor当前p/v**。半径、age、visibility、presence、保留期限、actor集合均不变。

```text
同一合法观测与逐人历史
        ↓
当前隐藏actor的p/v估计或诊断替换
        ↓
原候选后继、当前特征、即时奖励、风险/安全计算
        ↓
原标量价值与80动作排序
        ↓
原平滑与真实执行
```

旧预测桥梁只替换价值查询里的未来位置，奖励/风险仍消费原状态；本轮替换进入这些共同消费者的上游。没有新增记忆gate、预测器、规划器或奖励，也没有提出一个新的KDA结构。

简单实现只使用最近24帧中最后三次**真实可见**速度测量，用实际测量间隔作OLS速度趋势拟合；沿用既有合法CA基线的逐分量加速度±1m/s²、速度上限1.5m/s，不在结果上调阈值：

\[
\hat p_t=p_{\mathrm{CV},t}+\tfrac12\hat a_i\,\mathrm{age}_i^2,
\qquad \hat v_t=v_{\mathrm{last},i}+\hat a_i\,\mathrm{age}_i.
\]

少于两次测量不修正。缺失帧只用于读取和已有CV时间推进，不伪造测量。每actor按自己的关联槽计算、共享同一规则；actor ID只用于关联。

### 4.1 生命周期和输入合同

- 只有可见真实测量进入拟合；CV保留帧不进入测量样本。
- 80个候选只读取，不改历史、tracker或任何actor记忆。单元测试检查重复调用不变和actor隔离。
- 每episode建立新的observer和history。首次可见时出生；超过原2秒保留期不进入当前候选；重新可见使用原关联key和新测量，过期actor不被修正器复活。
- 私有目标、ORCA持续状态、真实p/v只保存在离线标签/世界恢复数据中，不进入部署修正输入。
- 根状态修正只影响这一步动作选择，不写回真实历史。后续三臂使用完全相同的原策略和各自真实执行产生的合法观测。
- 输出仍为原单位p/v，但纠正后的p/v与旧age/CV历史的联合分布兼容性没有独立认证。冻结消费者结果仅能裁决这一具体用法，不能裁决共同训练后的新状态表示。

## 5. 简单方法、GRU、KDA分别恢复了什么？

### 5.1 完成的配对块

选择原25个失败case及25个同人数/几何的不同成功case，成功case按case索引最近、不重复匹配。不是严格当前几何相同的配对。

事前固定根状态为：t≥2秒后，第一个合法估计的保留隐藏actor中心距离≤2m的控制帧；不按真实误差、动作变化或新方法收益选状态。50个case中28个满足：14失败、14成功保护；其余22个没有合格根状态。5人失败case没有满足这个规则的根状态，不能把该子集称为5人训练入口已验证。

所有28个配对块的全部三臂已完成。原生续跑28/28完整动作轨迹复现。

| 输入 | 失败根状态：成功/碰撞/超时 | 失败动作改变 | 平均折扣回报差 | 成功保护：成功/碰撞/超时 |
| --- | --- | ---: | ---: | --- |
| 当前状态/已有跟踪器 | 0/1/13 | 0/14 | 0 | 14/0/0 |
| 简单历史趋势修正 | 0/1/13 | 0/14 | 0 | 14/0/0 |
| 仅诊断：当前隐藏p/v真值 | 0/2/12 | 2/14 | -0.025352 | 14/0/0 |

简单修正全28个根状态均不改实际动作，未挽救失败。真值共3/28次改动作，其中2次在失败状态、1次在成功状态；也未挽救失败。

失败状态隐藏actor的逐root平均误差：当前位置0.07186m、速度0.10468m/s；简单修正为0.09783m、0.15085m/s；真值为0。真值消除物理状态误差，却未改善这一冻结接口的实际回报。

### 5.2 没有执行的臂，必须明确

**本轮没有新的Actor-GRU/Actor-KDA当前p/v状态估计训练，也没有这两臂的导航结果。** 现有未来位置头将t=0锚定为CV状态，不是可直接复用的当前状态修正权重；把它改名不能算兼容估计器。

局部状态头训练在用户授权范围内，但没有找到支持本接口的具体可恢复动作缺陷，因此没有启动。这里不是“GRU/KDA被状态比较淘汰”，也不是把用户允许的局部训练误说成禁止训练。

已有四种子预测器的比较只属于第1节：GRU和KDA有不同的局部挽救数，但均未超过CV的导航增量。不能将它代替本节未执行的四臂状态估计比较。

## 6. 改善有没有进入真实动作后果？

**本轮当前状态修正没有得到正向后果。** 根动作只执行0.25秒，之后以共同冻结策略在原真实环境续跑到reach_goal/collision/timeout，用原折扣奖励评估，不是只检查预测误差或两秒进度。

需要保留的反例：20人square、case80026、tick16，真值状态把动作21改成26，原timeout变成collision，回报差约-0.336649。20人circle、case80009、tick26，动作52改成51仍然timeout，回报差约-0.018279。

失败子集平均最小间距：当前/简单0.103170m，真值0.102607m；到目标距离减少量：当前/简单1.943778m，真值1.632785m。这些只是原续跑的描述指标，不是新增优化目标。

14个成功保护状态全部成功，平均剩余成功时间均15.446429秒；真值引起的一次动作改变未损坏该episode。没有独立确认集上的正信号，因此没有打开新10/20人确认case来挑结果。

**仍未知：** 重复闭环修正、与修正状态共同训练的消费者、隐意图/响应状态是否有收益。本轮没有做这些实验，不给肯定或否定结论。真值负结果也不能证明不存在某个更合适的消费者。

## 7. 额外计算是否值得？

先实测完整三臂块：首块0.762346秒；28块中位1.097143秒、P90为2.115193秒，全部完成，没有只跑有利臂。

简单OLS修正的CPU算子均值0.34189ms，P50/P90/P99为0.30774/0.57995/0.69015ms，额外参数0、额外持久状态0；复用已有历史最多26,208字节。Python/NumPy临时分配探针峰值5,192字节，不是进程RSS/GPU峰值或整策略延迟。

成本低，但没有动作/回报收益，所以**本轮不值得集成这个修正器为新方法**。新的GRU/KDA状态头没有训练或计时，其额外成本与收益未知，不能编造效率比较。多时域消费者的秒级成本与KDA增量不足也不支持继续部署该桥梁。

## 8. 没有合格机制后的两个限定替代入口

只检查已筛过的两项，没有开新方向队列。

### 8.1 DS-RNN：历史直接形成当前决策表示

[官方代码](https://github.com/Shuijing725/CrowdNav_DSRNN)的generate_ob向策略提供robot状态、robot速度和逐人相对位置，不直接给可见行人速度；srnn_model.py的forward先计算逐edge GRU，再attention和robot-node GRU，直接输出actor/critic：

\[
h_{i,t}=\operatorname{GRU}(\operatorname{embed}(p_{i,t}-p_{r,t}),h_{i,t-1}),
\quad h_{r,t}=\operatorname{GRU}(r_t,\operatorname{Attention}(h_{i,t}),h_{r,t-1}).
\]

局部源码位置：crowd_sim/envs/crowd_sim_dict.py::generate_ob（46–62行）；pytorchBaselines/a2c_ppo_acktr/srnn_model.py::forward（378–453行）；model.py::Policy.act。它确实绕过未来位置桥梁，但普通逐人GRU就是已有强简单实现，不自动构成我们的新增贡献。

本地版本91fb53c0f81964bbce8dd47960419fa8ab2fa810，有既有配置修改，未覆盖。官方示例checkpoint严格加载及CPU单步forward通过：973,983参数，value形状[1,1]、连续action[1,2]，值有限。**仅模型级运行，不是官方完整导航复现或新闭环收益。**

最小替换是官方edge recurrent cell及其共享输入/生命周期合同，但checkpoint是连续PPO，不兼容本母体80离散动作/IL-MC流程。当前母体已提供真实可见速度、逐actor关联和历史；不能削弱它的观测来制造DS-RNN式记忆优势。本轮没有留下可直接实施、已有动作残差支持的KDA差分。

### 8.2 TRACER：执行证据更新当前人类响应状态

[TRACER原论文](https://arxiv.org/html/2609.18776v1)的III-C2、式(20–21)以真实已执行交互后的观测更新identity-bound响应belief，再由III-B的候选响应模型消费：

\[
b_{i,t+1}(z)\propto b_{i,t}(z)\,
p_\theta(o_{i,t+1}\mid H_{i,t},u_{\mathrm{executed}},z).
\]

差分是“历史估计当前响应模式→候选交互动作”，不是“历史预测位置→旧单步价值查询”。标准离散Bayes更新、固定/重置模式权重是已有简单替代；KDA必要性未证明。

当前母体行人看不见robot，不能原样暴露这一robot-response机制。保持任务不变时，缺少其原生交互前提；全文检查未核到可复用的官方代码/checkpoint链接，不能据此声称世界上不存在代码。需要改变交互设定并建立likelihood/消费者，不是今晚允许的一个局部替换。**仅文献/接口证据，未实现、未训练、无本地导航收益。**

## 9. 下一步唯一值得执行的动作

**封存本轮，停止在当前母体上继续KDA结构救援、物理状态头训练和预测桥梁优化。** 不执行V9/V10，不把上述替代接口当成已经找到的新方法，不启动完整IL+3000RL。

重新GO需要一个具体接口在合法历史下显示可恢复的实际动作后果空间，并明确现有Current/CV及原历史消费者没有已经包含该能力。可以允许共同训练和局部新头，但不能把本轮未验证的意图表示补成“已找到的问题”。若需要更换任务或交互设定，应作为另一个明确授权的研究任务，而不是悄悄改本轮科学条件。

## 10. 可复查资产与交付

本轮只新增这一份MD总结；既有FORECAST_CONTROL_EXPERIMENT.md补充原多时域实验封账。核心model/policy/reward/simulator/training没有改动；新增代码仅为experiments诊断及正确性测试。

冻结核心source SHA256：ca46069a6d8f70e5881b8ac227be7fbd6a54f710620d000825ee7fd2a62b1671。

实际运行helper SHA256：

- multihorizon_control.py：a042d99ac307e6bfb949a22a4bb407c5afe7737ad7a91015b6180ad03e61971a
- decision_state.py：65b82c9396cc3b92e69e3758950c1dda611b8bc99f1d01e32126207481ebbc82
- 当前状态cohort：416e8ed6f1a9617c3b79c2c74c6897fd952b90c30d9e506df1e1126333f11438
- DS-RNN参考checkpoint：09eb9965e60159a0f1ab00ecfd420b31fa512fa12953678b796e5448b6bb3499

输入、actor关联、mask、age、时间戳、轨迹、80动作分数、终止、原回报、随机种子与checkpoint哈希已保存。4090的76个代码/输入/原始结果/日志文件与本地SHA256逐一相同；本地自然事件注释、summary和算子计时另存。临时远端RAM工作区在校验回收后清理；没有下载大模型到服务器系统盘。

可直接查看的完整路径：

/home/abc/workspace/shixu/outputs/multihorizon_control/summary.json

/home/abc/workspace/shixu/outputs/multihorizon_control/protocol.json

/home/abc/workspace/shixu/outputs/multihorizon_control/inputs/hashes.json

/home/abc/workspace/shixu/outputs/multihorizon_control/results

/home/abc/workspace/shixu/outputs/decision_state_rebuild/protocol.json

/home/abc/workspace/shixu/outputs/decision_state_rebuild/natural_events.json

/home/abc/workspace/shixu/outputs/decision_state_rebuild/summary.json

/home/abc/workspace/shixu/outputs/decision_state_rebuild/latency.json

/home/abc/workspace/shixu/outputs/decision_state_rebuild/archive_verified.json

/home/abc/workspace/shixu/outputs/decision_state_rebuild/results

原始权重/结果保留本地outputs，不放进Git。Git交付为诊断源代码、测试、既有报告更新和这一总结；不是宣称一个新算法训练完成。

正确性测试使用项目tests目录：142 passed，3个既有可选资产skip。无新训练、无未完成配对测试块、无后台科学计算任务。

**结论再次限定：发现了自然状态误差，没有找到值得继续开发的时序动作缺陷；物理真值纠正在这个冻结消费者中也未转成收益。历史估计其他latent decision state的价值仍未证明，不能用本次负结果替代其验证。**

<!-- END PRESERVED SOURCE -->


---

<a id="stage-3"></a>

## Source: README.md (historical README section 4)

Original full-source SHA-256: 0f5332ec89c79e6ab9fc605bb8e502e3aca61323ac48851518db15cb3f732d9a

<!-- BEGIN PRESERVED SOURCE -->
### Bounded Forecast-to-Control Experiment

FORECAST_CONTROL_EXPERIMENT.md records the next authorized navigation-method
experiment, not another hidden-goal audit. It compares CV, learned current-frame,
GRU and KDA forecasters through a common control interface. No motion-estimator
latent feature bypasses its predicted positions into the value head; B separately
preserves the original critic's legal temporal input. All learned
forecasters see the same legal current crowd; later visible measurements supervise
the actual deployed prediction decoder, without goal/hidden-state input.

The same128 ORCA trajectories,50 IL epochs and3000 online MC episodes are fixed
across four paired seeds. Training is circle/5-person only; final saved weights
are reloaded for held-out circle/square and5/10/20-person navigation. The strong
old GRU is re-evaluated on the identical new cases. A short pipeline check is not
a positive result, and prediction accuracy or action sensitivity cannot pass the
experiment. The single permitted interface rescue has now been completed.

```bash
PYTHONPATH=vendor:. python -m experiments.forecast_control prepare
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.forecast_control queue --workers 4 --device cuda
PYTHONPATH=vendor:. python -m experiments.forecast_control parent --device cpu
PYTHONPATH=vendor:. python -m experiments.forecast_control summarize
```

Raw data, interrupted preflight logs and checkpoints remain under ignored outputs.
Prototype A is complete and negative: primary10/20-person SR is75.20% for CV,
75.59% Current,80.86% GRU and75.20% KDA, versus84.96% for the original Parent.
A replaced the original temporal critic with forecast coordinates, so it does
not settle the more faithful forecast-to-native-critic connection.

The sole interface rescue B restores the original critic, trains forecasts only
from lawful motion labels, and corrects native hypothetical successor positions.
CV is exactly the original trained Parent, verified by score/action parity.
Current/GRU/KDA add motion estimation, not a replacement of the critic. B consumes
only the native .25s forecast for action evaluation; no multi-step planning claim.
Four fresh CV controls also train with exactly the same B code/data/budget and
IL critic tensors, alongside the reused original strong-reference checkpoints.

```bash
PYTHONPATH=vendor:. python -m experiments.forecast_control prepare --protocol experiments/forecast_control_b_protocol.json --data outputs/forecast_control_b/demonstrations.pt
PYTHONPATH=vendor:. python -m experiments.forecast_control parity --protocol experiments/forecast_control_b_protocol.json --device cuda
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.forecast_control queue --protocol experiments/forecast_control_b_protocol.json --root outputs/forecast_control_b --data outputs/forecast_control_b/demonstrations.pt --arms current gru kda --device cuda
PYTHONPATH=vendor:. python -m experiments.forecast_control reference --protocol experiments/forecast_control_b_protocol.json --root outputs/forecast_control_b --data outputs/forecast_control_b/demonstrations.pt --device cuda
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.forecast_control queue --protocol experiments/forecast_control_b_protocol.json --root outputs/forecast_control_b_fresh_cv --data outputs/forecast_control_b/demonstrations.pt --arms cv --device cuda
PYTHONPATH=vendor:. python -m experiments.forecast_control_results --protocol experiments/forecast_control_b_protocol.json --root outputs/forecast_control_b
```

B is complete: primary10/20-person SR is86.13% fresh CV,87.11% Current,83.40% GRU
and86.52% KDA, versus84.96% for the reused original Parent. KDA is only+0.39pp
versus fresh CV with+1.56pp collision and does not beat Current. It fails the
fixed development criteria; no fresh confirmation or further rescue is started.
The16 new runs completed50 IL epochs and3000 RL episodes each, with final-weight
reload and192 held-out evaluations per run. Source/data/checkpoint/case audits
and exact four-seed IL critic parity pass. Both A and B negatives and raw
artifacts are retained. This is a version-scoped negative, not rejection of all
temporal navigation methods. No method-novelty claim is made.

<!-- END PRESERVED SOURCE -->


---

<a id="stage-4"></a>

## Source: outputs/forecast_control_remote_backup_20261004/FORECAST_CONTROL_EXPERIMENT.md

Original full-source SHA-256: 5aad171abdc0b7b35177eb00fd8e48aa37422f675b3d66090bd2ebdf3a2cc49b

<!-- BEGIN PRESERVED SOURCE -->
# Forecast-to-Control Navigation Experiment

Date: 2026-10-04. Status: PROTOTYPE_A_COMPLETE; ONE_INTERFACE_RESCUE_AUTHORIZED.

## Fixed Question

Can lawful history-derived motion forecasts improve a trained navigation policy
when those forecasts explicitly enter its 80-action value evaluation?

This is a complete bounded method attempt, not KDA V9, another goal-recovery
audit, or a claim that only KDA can solve the task. Circle coupling, the previous
predictive audit and V1-V8 negative results remain closed and unchanged.

## Minimal Architecture

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

## Evidence State

Prototype-A is complete: all16 runs have50 IL epochs,3000 RL episodes and192
reloaded-weight evaluations. The completed contract passes124 tests,
121 passed and three existing optional legacy-asset
skips. Each arm completed a short two-demonstration, one-IL-epoch, two-RL-episode
training/save/reload/native-episode smoke. All four smoke evaluations timed out;
these intentionally tiny pipeline checks are not method-performance evidence.
Full16-run training used419/443/491 on the4090 and467 on the3060.

The control connection is committed and uploaded to shixu as f3aaae9. Scientific
source and the training protocol remain frozen while the full runs are active.
The additional diagnostic operates offline and does not alter training.

The diagnostic cohort is fixed independently of method outcomes: the first four
development cases per geometry/population cell, replaying the archived commands
of parent seed419. All24 replays reproduce terminal event, time, path and minimum
clearance. There are591 uniformly sampled control states and34,990 available
later-visible actor/time labels. All learned arms and both IL/final phases will
use these identical legal histories. It measures prediction accuracy and the
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
No rescue has been trained at this point.

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

Frozen scientific source SHA256:

ae93ca700d995231df0935ec554bdf834c1312c316c5382f83b17b7273212ae3

Frozen protocol SHA256:

7c12cf45a721b0796223b9b503c489c37306c446645bbe90c0b2b615f0a3971c

Shared demonstration archive SHA256:

7c314d387b972037a6ff9971505c70153f19b54f81b35af1837a0f961832c84c

Pre-results interface correction: the first implementation withheld current
neighbour geometry from the forecast head. That could make the Current control
artificially weak. Its incomplete training was interrupted, archived and charged
separately; no final result was used to choose the correction. All common arms
restart after adding identical legal current-crowd conditioning. This is contract
completion, not the allowed result-based rescue, which remains unspent.
The12 interrupted runs had about234 wall seconds each, some at IL36 and others
at RL71-204; none reached3000 episodes or produced a final result. Their source
archive, partial IL weights and logs remain in outputs/forecast_control_initial_contract
and outputs/forecast_control_initial_local. They are not reused as final-IL
initialization or counted as completed formal runs.

Artifacts are saved locally under:

/home/abc/workspace/shixu/outputs/forecast_control_a/

The4090 uses an owned RAM workspace, without downloads to its system disk. Raw
artifacts must be retrieved and checked locally before deleting that workspace.
The laptop handles reference evaluation/verification; the local3060 handles one
complete paired seed. No existing remote environment is modified.

<!-- END PRESERVED SOURCE -->


---

<a id="stage-5"></a>

## Source: outputs/forecast_control_remote_backup_20261004/README.md (historical README section 4)

Original full-source SHA-256: d5ac9d5a389c9f6a37c0f9ddc216d8995770f37cc60863cfa476233835b54058

<!-- BEGIN PRESERVED SOURCE -->
### Bounded Forecast-to-Control Experiment

FORECAST_CONTROL_EXPERIMENT.md records the next authorized navigation-method
experiment, not another hidden-goal audit. It compares CV, learned current-frame,
GRU and KDA forecasters through one shared future-geometry value consumer. No
memory feature bypasses the predicted positions into the value head. All learned
forecasters see the same legal current crowd; later visible measurements supervise
the actual deployed prediction decoder, without goal/hidden-state input.

The same128 ORCA trajectories,50 IL epochs and3000 online MC episodes are fixed
across four paired seeds. Training is circle/5-person only; final saved weights
are reloaded for held-out circle/square and5/10/20-person navigation. The strong
old GRU is re-evaluated on the identical new cases. A short pipeline check is not
a positive result, and prediction accuracy or action sensitivity cannot pass the
experiment. At most one evidence-based rescue is reserved after all arms finish.

```bash
PYTHONPATH=vendor:. python -m experiments.forecast_control prepare
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.forecast_control queue --workers 4 --device cuda
PYTHONPATH=vendor:. python -m experiments.forecast_control parent --device cpu
PYTHONPATH=vendor:. python -m experiments.forecast_control summarize
```

Raw data, interrupted preflight logs and checkpoints remain under ignored outputs.
The formal prototype is running; no improvement or method-novelty claim is made.

<!-- END PRESERVED SOURCE -->
