# Forecast-to-Control Navigation Experiment

Date: 2026-10-04. Status: PROTOTYPE_A_COMPLETE; SOLE_INTERFACE_RESCUE_B_RUNNING.

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

B trains Current/GRU/KDA from scratch for the unchanged50 IL +3000 RL budget,
four paired seeds. CV reuses verified original full-budget trained weights; its
new training time is zero and original training time is recorded separately.
No reference learning log is invented. All final B weights are reloaded for the
same192 cases per seed. The4090 hosts419/443/491 in an owned RAM directory and
the3060 hosts467. A premature remote launch occurred before data transfer ended;
it failed before any worker/training and was relaunched after transfer completion.

B scientific source SHA256:

ca46069a6d8f70e5881b8ac227be7fbd6a54f710620d000825ee7fd2a62b1671

B protocol SHA256:

cbbf9f397ea8effcfa1f868619d6c367a10754eab7b0b04c3f5267dd8b7f636a

B shared demonstration archive SHA256 (same original episode contents):

2b8b7adfa38e34a4d53dbdcf8931c5270716a047b7bb2785d27449899bcd00ff

The scientific source and protocol are now frozen until every B run finishes.
Only evidence collection, report updates and non-training diagnostic utilities
may change. This consumes the sole rescue; no additional version is authorized.

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
