# shixu

A small temporal crowd-navigation research framework extracted from the user's
local camrl Mamba-VL project. It contains the cleaned parent and explicit
temporal architecture experiments, not an established new-method claim.

## One Main Path

```text
legal observations -> frame encoder -> temporal encoder -> scalar value
                    -> original candidate successor/value lookahead -> action

ORCA demonstrations -> Monte Carlo value initialization -> online MC refinement
```

GRU is the default temporal baseline. Mamba is optional and retained only for
legacy comparison. There are no Double-Q/PPO/SAC branches, auxiliary prediction
heads, Bayesian modules, teacher networks at deployment, or fallback backbones.

The explicit actor-first alternative moves temporal encoding before crowd pooling:

```text
identity-bound human histories -> shared temporal encoder -> crowd pooling
                              -> scalar value -> unchanged lookahead
```

The processing-order prototype now compares scene-first and actor-first GRU
using identical aligned observations and exactly the same parameters. Every
human uses the same GRU weights, with independent histories. That trial adds no
dual-memory system, contradiction detector or additional prediction head.
That processing-order experiment remains a baseline, not a selective-revision method.

The observation-only contract consumes robot raw state9 and human observed
motion9 plus presence. It removes redundant legacy relation features rather
than adding a predictor or another memory. The explicit legacy feature contract
remains only for reproducing the first trial and loading its checkpoints.

## Layout

| File | Responsibility |
| --- | --- |
| shixu/features.py | Legacy observation contract and history windows |
| shixu/observations.py | Episode-local observed association keys, never numeric ID features |
| shixu/model.py | Frame encoder, replaceable temporal encoder, value head |
| shixu/temporal.py | Compact GRU/KDA/GDN2 actor memories; real-write/candidate-read interface |
| shixu/policy.py | Original action support and successor-value evaluation |
| shixu/replay.py | Episode-safe windows and MC targets; no duplicated window archive |
| shixu/runner.py | One runner for collection, training and evaluation |
| shixu/training.py | ORCA value initialization and online MC refinement |
| shixu/cli.py | Explicit collection/evaluation/training commands |
| vendor/crowd_sim | Frozen local simulator dependency |
| experiments/ | Frozen protocol, immutable ORCA collection and paired processing-order trial |

## Occlusion Development Loop

The new `occlusion` architecture removes the five-person input cap. It uses
episode-local identity slots, actual-measurement write masks and a separate
retained-track read mask. A previously observed actor can remain relevant while
occluded; an actor that has never been seen cannot enter the model. Missing
positions use the last legally measured velocity for at most two seconds.
Predicted positions and candidate successors are read-only queries, not new
measurements. There is one shared actor memory and no external memory gate,
auxiliary predictor, dual-memory branch or changed reward.

```text
body-occluded observations -> legal track histories -> shared KDA memory
                          -> retained-track candidate reads -> scalar value
                          -> inherited 80-action value lookahead
```

This is a functional research prototype, not an established novelty or
performance claim. The common retention interface is also used by the
current-state and recurrent comparisons. Previously reported full-observation
results are not occlusion results.

```bash
# Pin the vendored simulator when another CrowdNav is installed locally.
export PYTHONPATH=vendor:.
python -m unittest discover -s tests -v
python -m experiments.occlusion collect \
  --data outputs/occlusion_v1/demonstrations.pt
python -m experiments.occlusion queue \
  --root outputs/occlusion_v1 --data outputs/occlusion_v1/demonstrations.pt \
  --device cuda --workers 4
python -m experiments.occlusion summarize --root outputs/occlusion_v1
```

`experiments/occlusion_protocol.json` freezes shared demonstrations, four
paired development seeds, 50 IL epochs, 1,000 online MC-RL episodes and
5/10/20-person evaluation. Every reported model is reconstructed and loaded
from its final saved checkpoint before evaluation. Development cases guide
diagnosis; fresh seeds and separate confirmation cases remain reserved until
an architecture is selected. A negative version is diagnosed, not relabeled
as a failed research family. Raw weights, logs and episodes remain local.

The second frozen version, `occlusion_observation_protocol.json`, changes only
the read clock. Memory is read from the latest legal history frame once per
actor, then shared by all candidate actions. Candidate geometry remains in the
spatial value encoder, but it no longer changes the actor's temporal read vector.
The weights and parameter count are unchanged by this switch. Both versions
receive byte-identical episode observations and rewards. The current-track
reference is reused because it has no learned temporal read; the temporal
arms are retrained through the complete IL/MC-RL schedule.

```bash
python -m experiments.occlusion collect \
  --protocol experiments/occlusion_observation_protocol.json \
  --data outputs/occlusion_v2/demonstrations.pt
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_observation_protocol.json \
  --root outputs/occlusion_v2 --data outputs/occlusion_v2/demonstrations.pt \
  --arms gru kda --device cuda --workers 4
python -m experiments.occlusion summarize \
  --protocol experiments/occlusion_observation_protocol.json \
  --root outputs/occlusion_v2
```

The observation-clock design is a hypothesis under test, not a correctness
repair. Both versions leave the stored matrix state unchanged during candidate
evaluation. Action-conditioned reads can legitimately retrieve different
information for different decisions even when humans do not react to the
robot. The new bias separates a shared temporal read vector from
candidate-conditioned geometry evaluation and avoids replicating each actor's
matrix memory 80 times. Frozen-weight read ablations are diagnostic, not
substitutes for retraining or proof of better navigation.

The first full four-seed occlusion trial is complete (384 held-out development
episodes per arm, not the earlier full-observation trial):

| Model | Overall SR | CR | Timeout | 10/20-person SR |
|---|---:|---:|---:|---:|
| CV-track current-value | 81.77% | 4.95% | 13.28% | 79.30% |
| Actor-GRU | 77.34% | 6.25% | 16.41% | 75.78% |
| Actor-KDA, candidate read | 78.39% | 11.20% | 10.42% | 75.78% |

KDA does not pass: primary SR is unchanged against GRU, while primary collision
increases by 5.86 percentage points. A frozen-weight switch to observation-clock
reads also worsens seed 443 overall SR from 70.83% to 56.25%; this is a
distribution-shifting diagnostic, not a trained comparison. The second version
must therefore earn its own result through matched IL and RL.

Completed final-IL checkpoints may be moved to a faster host using `--il-root`.
The experiment verifies seed/configuration and all IL log epochs, restores the
replay sampling stream, then starts a fresh, full-budget RL run. Interrupted RL
work is archived and charged separately; it is not used for checkpoint selection.
An exact CPU pipeline test checks identical full-run versus final-IL-reuse RL
updates. Different GPU/software environments can still introduce numerical
differences and are recorded with each run.

The complete observation-read trial is negative: overall SR is 63.54% for
KDA versus 79.43% for GRU; primary 10/20-person SR is 64.45% versus 76.56%.
KDA loses primary SR in all four paired seeds. Fewer collisions come with
substantially more timeouts (30.47% overall), so this version is not retained
as the candidate architecture. Shared observation reads are not an established
correction to the original action-dependent read design.

The next structural test, `occlusion_context_protocol.json`, returns to the
original candidate clock and moves the existing attention before actor memory:

```text
real actor observations -> attention (actor outputs, no scene pooling)
                        -> shared, measurement-masked KDA writes
candidate geometry     -> same attention -> read each actor memory
                        -> masked max -> scalar value -> original lookahead
```

There is still one attention and one temporal operator. The parameter counts
are unchanged. Prefix attention keys use actual measurements; hidden tracks
cannot supply fresh write evidence. Query attention can use legally retained
CV tracks. The non-temporal current-value arm is mathematically unchanged;
its existing saved weights reproduce every executed action and terminal outcome
in all 96 seed-419 development episodes after the helper refactor.

This tests whether historical neighbour context is useful. The old actor
memory is exactly insensitive to another actor's past when its own history is
fixed. The replacement removes that insensitivity in a fixture, but this is
not proof of navigation headroom or novelty. A separate episode-disjoint
linear probe of 887 natural re-entries does not improve average velocity
prediction over CV, so richer context is not presumed useful in every task.
Archived collision replays also show visible colliders and no margin-safe
action at the final step; they do not establish that occlusion caused the
earlier poor decisions. DS-RNN and [PaS](https://github.com/yejimun/PaS_CrowdNav)
already study temporal/social inference. Moving attention is an experimental
representation choice, not a standalone new-method claim.

The complete context-write trial is negative for KDA: overall SR/CR/timeout
are 62.76/12.24/25.00%, versus 85.94/5.99/8.07% for GRU. Primary 10/20-person
SR is 60.16% versus 85.55%, with KDA losses in all four paired seeds.
On 836 uniformly sampled demonstration windows, seed-443 final KDA keys have
mean inter-actor cosine 0.99926 in this version versus 0.77600 in V1.
Separate per-actor matrices do not prevent homogenization when their inputs
are almost identical. High cosine is a representation diagnostic, not a proof
that it causes all observed navigation losses.

The next test, `occlusion_address_protocol.json`, preserves local features:

```text
actor feature u_i -> u_i + attention(u, measured actors) -> memory content
actor feature u_i -------------------------------------> KDA q/k (custom)
candidate feature + same attention -> retained-memory read -> max -> value
```

All four arms share the residual spatial path. Current-value, contextual GRU,
ordinary residual KDA and actor-addressed KDA are retrained; the old current
reference is not reused because its query representation changes too. The
two KDA arms have exactly the same parameters and initialization. Only q/k
source differs; values, decay and write strength consume contextual content.
At deeper layers both streams receive the same recurrent output. There is
one attention, one shared actor memory and no additional loss or output head.
Existing read/write versions remain available for their archived checkpoints.

This is a mechanism test, not an established new method. Separate-source
delta memory already appears in [DRAM](https://arxiv.org/abs/2609.32453), and
retrieval design is studied in
[Advantage-Driven Explicit Memory](https://arxiv.org/abs/2608.25610).
The hypothesis here is narrower: retaining actor-local addresses while
remembering observed social context helps legal hidden-track action evaluation.
That claim still needs navigation gains and an occlusion-specific analysis.

```bash
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_address_protocol.json \
  --root outputs/occlusion_v4 --data outputs/occlusion_v4/demonstrations.pt \
  --device cuda --workers 4
python -m experiments.occlusion summarize \
  --protocol experiments/occlusion_address_protocol.json \
  --root outputs/occlusion_v4
```

Demonstrations are the same saved observations/rewards as V1, with protocol
metadata retargeted for the matched run; there is no extra training data.
Final IL and RL weights, all failed versions, raw episode controls and consumed
compute are retained. Fresh seeds and unseen confirmation cases are not used
to develop this version.

The complete V4 result is negative. Overall SR/CR/timeout are 68.75/11.72/19.53%
for actor-addressed KDA, 72.40/10.94/16.67% for vanilla residual KDA and
77.08/5.47/17.45% for residual GRU. Primary SR falls by 11.72 pp against GRU;
actor-local addresses do not rescue this version. The final seed-443 address
probe does recover distinct keys (mean inter-actor cosine 0.8144), so loss of
actor addresses alone is not an adequate explanation of the navigation failure.

`occlusion_branch_protocol.json` tests the inherited successor-value interface,
not a new gate or a claim that private branches are novel:

```text
actual measured actor frames -> shared KDA -> real-history matrix
candidate CV successor      -> private one-step KDA calculation
                            -> existing attention/max/value -> original lookahead
```

Candidate computations never commit into real history. The single-step read
is algebraically identical to an explicit private matrix update, but avoids
80 replicated matrices. Its paired control additionally accepts legally
propagated CV tracks as history pseudomeasurements; it receives no hidden truth.
Both new arms retain identical parameters, initialization and IL/RL budgets.
The unchanged V1 current-value and read-only KDA references and the stronger V3
context-GRU reference remain controls, with configuration and action-parity
checks before reuse. Any positive development result still requires new seeds
and unseen cases, with all confirmation arms trained anew.

Engineering optimizations share matrix reads across candidates and pack measured
GRU frames without changing their recurrence. An exclusive laptop CPU read-only
benchmark falls from 5.91 to 2.37 ms for five actors and 27.67 to 9.90 ms for
twenty actors; these are not full-controller latency or navigation gains.
Frozen checkpoint actions and gradient/recurrence tests validate the changes.

## Installation

Install a PyTorch build appropriate for your machine first. Then:

```bash
git clone https://github.com/jinglongjiang/shixu.git
cd shixu
python -m pip install -e .
python -m pip install 'git+https://github.com/sybrenstuvel/Python-RVO2.git'
```

Python-RVO2 needs its normal native build prerequisites. The GRU path does not
require Mamba, Transformers or custom CUDA kernels. Legacy Mamba checkpoints
require the optional mamba-ssm 1.2.0 dependency and a matching CUDA/PyTorch wheel;
do not silently substitute another network if it fails to import.

## Commands

```bash
# Interface checks only; this is not a trained policy result.
python -m shixu.cli smoke --output outputs/smoke.json
python -m unittest discover -s tests -v

# Collect legal, identity-tagged observations and ORCA returns without training.
python -m shixu.cli collect --cases 0 1 --output data/orca.json

# Trained-checkpoint evaluation. Weights are intentionally not uploaded.
python -m shixu.cli evaluate --backbone mamba --device cuda \
  --weights /path/to/rl_model_ep10000_T24.pth --cases 0 1

# Explicitly opt into training.
python -m shixu.cli train --il-episodes 5 --rl-episodes 10 \
  --device cuda --output weights/gru.pt
```

Models must use the same config when comparing them. The optional local-source
regression tests use environment variables CAMRL_PARENT and CAMRL_CHECKPOINT;
they check features, value outputs, actions and history against the original
source. They skip explicitly when those local assets are unavailable.
New checkpoints include their model/observation configuration; evaluation uses
it automatically unless an explicit --config override is supplied.

The matched trial is driven by experiments/temporal_protocol.json, not test
results: four paired seeds, a shared 128-episode successful ORCA dataset,
50 IL epochs, 1,000 MC-RL episodes per arm, and fixed circle/square cases at
5/10/20 humans. The initial legacy contract has 302,337 parameters per arm;
the observation-only contract has 300,417 at the same width 128 and depth 2.
Only the final-budget checkpoint is evaluated. Processing-order prototype
results cannot be represented as a new algorithm or proof of selective memory.

```bash
python experiments/temporal_collect.py --output data/demonstrations.pt
python experiments/temporal_order.py run --seed 17 --order pair \
  --data data/demonstrations.pt --root outputs/temporal_v1 --device cuda
python experiments/temporal_order.py summarize --root outputs/temporal_v1

# One shared interface rescue: same data/budget, subtract legacy derived inputs.
python experiments/temporal_order.py run --seed 17 --order pair \
  --feature-contract observed --data data/demonstrations.pt \
  --root outputs/temporal_v2 --device cuda
```

Run the other seeds in the protocol before requesting the paired summary.
Native experiments assume perfect observed association and retain the original
five-human neural input cap even when the simulator contains 10/20 humans.
Missing observation masks preserve actor state; association errors and
real-world re-identification are not solved by this interface.

## Initial Matched Result

Four paired seeds completed 50 IL epochs + 1,000 online MC-RL episodes per arm,
followed by 96 fixed native evaluations each (768 total).

| Model | SR | Collision | Timeout | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Scene-first GRU | 75.52% | 9.90% | 14.58% | 302,337 |
| Actor-first GRU | 78.39% | 8.33% | 13.28% | 302,337 |

The +2.86 pp mean SR change has only 2/4 positive seed pairs and does not meet
the frozen +3 pp / 3-of-4 direction gate: NO_STABLE_GAIN. Pooled square gains
and smaller actor seed dispersion are exploratory, not a new-method claim.
Same-device RTX 3060 scoring medians are 3.110 / 4.418 ms for scene / actor;
actor-first is not a computation-saving result. No GDN/KDA/revision cell is
claimed successful on the strength of these mixed outcomes.

```bash
python -m experiments.temporal_latency --root outputs/temporal_v1 --seed 17
python -m experiments.temporal_revision_shadow --root outputs/temporal_v1 --seed 17
```

The shadow uses arrived motion evidence and native scene replay. A masked-prefix
intervention is an offline diagnostic, not a trained or deployable revision
policy. Full results/checkpoints stay local under outputs; weights and data are
not committed. The unchanged fresh follow-up used seeds 103/137: SR changes
were +7.29 / -12.50 pp, so the initial seed-dispersion signal did not replicate.
The common observation-only rescue is frozen separately in
experiments/temporal_rescue_protocol.json; its results must not be pooled with
the legacy-contract cohort.

## Completed Observation-Only Rescue

The one permitted rescue subtracts the inherited redundant/incorrect derived
inputs for **both** arms, without changing data, reward, budget or network size.
Four paired seeds again completed 50 IL epochs, 1,000 MC-RL episodes and 96
fixed evaluations per arm (768 evaluations).

| Model | SR | Collision | Timeout | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Scene-first GRU | 75.52% | 11.20% | 13.28% | 300,417 |
| Actor-first GRU | 75.00% | 6.25% | 18.75% | 300,417 |

SR differences are +3.13, +5.21, -5.21 and -5.21 pp across seeds
17/29/43/71. The mean is -0.52 pp with 2/4 positive pairs:
**NO_STABLE_GAIN** under the unchanged gate. Collision decreases in all four
pairs, but timeout increases; this is a safety-progress operating-point signal,
not proof of better navigation. Pooled 20-human gains also remain only 2/4
seed-positive. Same-device scoring medians are 3.012 / 4.163 ms (scene / actor),
so actor-first is about 38% more expensive in this workload.

The legal observed-change shadow finds four first events in 12 native
episodes: targeted actor-history truncation changes no root rankings and gives
no safe progress gain >=0.05 m. Selective-revision headroom remains unproven;
the small masked-prefix intervention does not reject the research family.
Attention and pooling both move relative to recurrence, so this comparison
does not isolate identity continuity alone.

Reserved fresh rescue seeds 191/223 were not run within that study because the
primary gate failed. That study added no GDN/KDA, new reward or extra teacher.
Across the separate initial, fresh and rescue cohorts, 20 models and 1,920
matched evaluation episodes are retained locally. None is relabeled as a new
method. All 38 local tests pass with the original comparison assets configured;
the laptop passes 35 tests with three explicit original-asset skips.

```bash
python experiments/temporal_order.py summarize --root outputs/temporal_v2
python -m experiments.temporal_latency --root outputs/temporal_v2 --seed 17
python -m experiments.temporal_revision_shadow --root outputs/temporal_v2 --seed 17
python -m shixu.cli evaluate --weights outputs/temporal_v2/17/actor/model.pt \
  --device cuda --cases 0 1
```

## Baseline Boundary

The source baseline comes from the user's
CrowdNav(20260511_last_version_mamba_vl).zip, not the later Bayesian-replaced
active camrl directory. The simulator preserves that archive's behavior;
only trailing whitespace is cleaned.
Its CrowdNav foundation is attributed in vendor/CROWDNAV_LICENSE.

The inherited deterministic baseline uses 80 moving actions, dt=0.25 s,
24-frame history, and r+0.99V lookahead. The archive's evaluation settings also
include clearance filtering, a risk penalty and action smoothing. They are
retained explicitly in shixu/default.ini; this is not a reproduction of paper
statistics based on a few episodes.

Legacy metadata indices and spatial relational-coordinate conventions are
preserved for checkpoint parity. Their audit is separate from method novelty;
changing them together with a new memory would confound that comparison.

The new training runner preserves the IL-to-MC-value-learning formulation, not
every historical launcher's behavior: observations are recorded even during
exploratory controls, teacher state is cleared between episodes, and test-case
scheduling is explicit. All new training arms must share this runner. Legacy
training numbers cannot be attributed to this cleanup without matched reruns.

Simulator IDs are association keys attached to observed states, not neural
features. Human goals/future states are not written into deployable inputs.
Weights, data, videos, credentials and old experiment artifacts are excluded
from version control.

## Explicit Memory Architecture Pilot

A separately authorized pilot compares two mechanisms without assuming the
newer operator is better:

```text
observed actor prefix -> one shared GRU/KDA/GDN2 -> per-actor state
candidate successor  -> query that state       -> current feature + memory
                     -> original attention/max pool -> scalar value/lookahead
```

Training uses the first T-1 observed-history slots as the prefix and the last
real frame as the query. Episode starts inherit first-frame replication padding.
In inference, the query is an analytic candidate successor. It
never changes the persistent observation history. All 80 queries share one
prefix encoding. The full-window control updates a disposable state copy with
the query; it also never persists hypothetical observations.

KDA evidence fusion compares `f + gate(f,m,e)*m` with the same-capacity generic
gate using zero evidence. GDN2 evidence revision supplies `e` to the existing
channel-wise erase/write projections, compared with zero evidence at exactly
the same parameter count. Here `e` is causal observed velocity innovation,
signed speed change and a validity bit, computed only from real prefix frames.
It is not a hidden intent, goal change timestamp or future truth.

There is one actor memory, not separate motion/context networks. Channel-wise
gates do not guarantee semantic motion/context separation or safe forgetting;
that is a hypothesis to test, not an architectural property already proved.

The compact cells implement the exact MIT FLA reference recurrence and omit
language-model convolutions, hybrid attention and large decoders. They do not
claim to reproduce the full Kimi Linear or GDN2 language-model architecture.
They need no additional CUDA package. Credit/license: vendor/FLA_LICENSE;
reference commit 9f38d24980c46d46bd38614e743cdacd21906578.

| Arm | Temporal/read interface | Parameters |
| --- | --- | ---: |
| actor_gru | Original actor-first GRU | 300,417 |
| gru_evidence | GRU prefix/read and evidence fusion | 366,593 |
| kda_full | Compact KDA with disposable query write | 268,177 |
| kda_read | KDA read-only query, current residual | 268,177 |
| kda_gate | KDA generic gated residual | 301,585 |
| kda_evidence | KDA evidence-gated residual | 301,585 |
| gdn2_read | GDN2 read, zero evidence at write gates | 335,241 |
| gdn2_revision | GDN2 evidence-conditioned write gates | 335,241 |

The two evidence-specific contrasts are parameter matched; comparisons between
different substrates are not. GRU evidence fusion is the strong cheap control.
Matrix-state capacity is also different: at these dimensions KDA/GDN2 store
40,960 floats versus the original GRU's 1,280, not a matched state-size control.
The frozen protocol uses seeds 191/223, the same immutable 128-episode ORCA
dataset, 50 IL epochs, 1,000 online MC episodes, four updates/episode, width128,
depth2, T24, reward/actions/simulator and 96 development cases/model. These
seeds are a new architecture pilot, not fresh confirmation of earlier trials.
Two seeds and reused cases cannot establish METHOD_ENTRY_FOUND.

```bash
python -m experiments.temporal_memory queue --data data/demonstrations.pt \
  --root outputs/memory_pilot --device cuda
python -m experiments.temporal_memory summarize --root outputs/memory_pilot
python -m experiments.temporal_memory latency --root outputs/memory_pilot \
  --seeds 191 --device cuda
python -m experiments.temporal_memory events --root outputs/memory_pilot \
  --seeds 191 --device cuda
```

Tests compare recurrence and gradients, official reference equations, causal
evidence, masks/re-entry, read-only candidate queries, shared-prefix versus
full-window values/gradients, and native candidate scores. Operator provenance
is not novelty: actor memory, separate current/history consumption and generic
gating have close priors, including ReCAT (https://intuitive-robots.github.io/ReCAT/).
TRACER (https://arxiv.org/html/2609.18776v1) also separates executed evidence
updates from candidate-trajectory queries in social navigation. That principle
is not a novel claim of this implementation.
Navigation results and evidence-specific ablations must justify any narrower
claim before the architecture is selected as a paper method.

The latency replay also measures the original trained actor GRU with a shared
prefix computation, preserving its value function. This prevents attributing
generic prefix reuse to a new memory operator. Natural-event shadow comparisons
use common roots from the first legal near-motion event in each pre-fixed parent
episode, not the best events for a new arm. They remain exploratory supporting
evidence, not a replacement for a negative paired SR result.

## Completed Memory Pilot

All eight arms finished both paired seeds (191/223): 16 final checkpoints,
50 IL epochs and 1,000 online MC episodes each, with 1,536 fixed evaluation
episodes in total. This cohort is separate from the older processing-order
experiments. No reward, action support, demonstration data or training budget
was changed after observing outcomes.

| Arm | SR | Collision | Timeout | Successful time (s) | RTX 4090 score (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| actor_gru | 82.81% | 6.25% | 10.94% | 18.20 | 3.013 |
| gru_evidence | 78.12% | 7.29% | 14.58% | 17.56 | 3.839 |
| kda_full | 74.48% | 15.62% | 9.90% | 19.51 | 11.678 |
| kda_read | 80.21% | 6.77% | 13.02% | 22.56 | 9.590 |
| kda_gate | 84.90% | 7.81% | 7.29% | 20.56 | 9.726 |
| kda_evidence | 75.52% | 6.25% | 18.23% | 23.47 | 9.716 |
| gdn2_read | 77.08% | 8.85% | 14.06% | 20.46 | 9.788 |
| gdn2_revision | 75.00% | 7.29% | 17.71% | 22.70 | 9.799 |

The parameter-matched mechanism tests are negative in both seeds:

- KDA evidence versus generic gate: SR -8.33 / -10.42 pp; mean -9.38 pp,
  timeout +10.94 pp. Adding motion evidence does not justify this gate.
- GDN2 evidence revision versus zero-evidence update: SR -1.04 / -3.13 pp;
  mean -2.08 pp, timeout +3.65 pp.
- Against the original actor GRU, the custom KDA/GDN2 arms lose 7.29 / 7.81 pp
  mean SR. Neither beats the GRU evidence control either.

Generic KDA gating has the highest mean SR, but its gain over actor GRU is only
+2.08 pp with one positive seed and one tie. Successful-episode time rises
about 13%; different success sets make this a descriptive, not causal, time
comparison. Its pooled 20-human SR is 76.56% versus 65.63% for actor GRU, but
this secondary reused-case slice does not rescue the failed primary gate or
establish a social-specific mechanism.

Every frozen contrast returns NO_CONSISTENT_PILOT_GAIN. This is
**VERSION_NEGATIVE, not FAMILY_NEGATIVE**; two seeds cannot establish permanent
dominance or a paper-ready method. No extra fresh training was launched.

### Cost and Validation

The timing table measures the complete 80-action score on an otherwise idle
RTX 4090, PyTorch 2.9.1+cu128, one CPU thread, 20 warmups and 100 synchronized
samples. The output-equivalent cached actor GRU takes 3.373 ms, so generic
prefix reuse is not a GPU speedup in this workload. KDA/GDN2 are roughly three
times slower than the original GRU here. These compact PyTorch cells are not
optimized official FLA kernels; this result does not benchmark those kernels.

On the i7-1165G7 laptop (PyTorch 2.4.1, one thread), actor GRU / cached GRU
take 64.032 / 5.530 ms. KDA evidence / GDN2 revision take 11.403 / 11.276 ms.
Thus the CPU caching benefit is already available without a new operator.
The local RTX 3060 replay is supplemental only: an unrelated RustDesk compute
process was active, so it is not an idle-device performance claim. Timings
across different devices/PyTorch versions are not pooled.

Training wall times per model are 573-778 s for the GRU arms and 1,446-2,063 s
for the matrix-memory arms. Varying concurrent worker counts and episode lengths
make these descriptive resource records, not matched throughput estimates.
Peak allocated memory is 692-724 MiB / 1,889-1,980 MiB respectively.

The common-root shadow covers 12 native parent episodes, 4,365 person-frames
and six first legal near-motion events. Over three-second continuations,
KDA evidence versus generic gate has three progress wins and three losses;
GDN2 revision versus its matched control has zero wins and four losses
(>=0.05 m). All branches are collision-free. Changed root actions therefore
do not establish recovery value or selective motion/context retention.

All 16 artifacts were checked for finite weights/losses, 50 IL epochs,
1,000 RL episodes, identical case sets and the shared data checksum. Source
and result/checkpoint/log hashes were compared with the training host. Normal
CLI loading was also checked for both custom checkpoints, not used as extra
performance evidence. The final local suite passes 50 tests, including legacy
Mamba parity and the NumPy-to-JSON shadow-export regression. The laptop runs
50 tests with 46 passing and four explicit optional-asset skips. All remote
artifacts were retrieved and checksum-verified before this run's temporary
4090 workspace was removed; existing environments were left untouched.

Full checkpoints and records remain local in outputs/memory_pilot, excluded
from Git. The existing strategy report contains the detailed paired contrasts.
The useful delivered result is a tested, compact architecture and reproducible
negative mechanism comparison, not a successful new navigation algorithm.

## Frozen KDA Gate Diagnostic

```bash
python -m experiments.temporal_memory gate-diagnostic \
  --root outputs/memory_pilot --device cuda
```

No new training: 38 common roots from 12 fixed native parent episodes, using
uniform ticks plus six first arrived near-motion events; both trained seeds.
KDA gating changes memory **readout**, not erase/write. The GDN2 update
mechanism is not tested by this read-gate diagnostic.

Removing only explicit motion evidence changes 0/76 candidate selections;
removing motion and validity changes 1/76. The direct mean gate change from
motion is about 0.00063. This does not support attributing the 9.38 pp SR gap
to harmful runtime motion gating on these states. Entire trained models differ,
and online MC refinement collects policy-dependent trajectories.

In the generic model, constant per-channel gates change 7/76 selections, a
uniform 0.5 gate changes 18/76, and no attenuation changes 35/76. Its mean gate
is 0.56, without broad saturation. This suggests readout scale calibration,
not demonstrated semantic stale-motion erasure. Constants use this same root
cohort; interventions are potentially out of distribution, final rankings
include the inherited safety filter, and no closed-loop improvement is claimed.
Raw diagnostics remain in outputs/memory_pilot/gate_diagnostic.json. A new
read-only intervention/restoration regression brings the local suite to 51
passing tests.

## Frozen Static-versus-Dynamic Follow-up

This follow-up trains read-only KDA with coefficient1, 128 learned
state-independent sigmoid channel scales, or the existing generic dynamic
gate. Actor GRU remains an external reference. Static scales initialize at0.5;
all shared KDA weights have identical initialization for a paired seed.
Parameters: 268,177 / 268,305 / 301,585; actor GRU has300,417. Capacity
differences are reported, not hidden using unused new parameters.

The separate frozen protocol uses four new seeds307/331/359/383 and cases
400-415 in circle/square with5/10/20 humans. Data, reward, actions,
50 IL epochs and1,000 MC-RL episodes are unchanged. Diagnostic IL50/RL500
snapshots are retained, but only the final checkpoint is eligible for the
primary comparison. Online trajectories still depend on the learned policy.

```bash
python -m experiments.temporal_memory queue \
  --protocol experiments/temporal_scale_protocol.json \
  --data data/demonstrations.pt --root outputs/scale_followup --device cuda
python -m experiments.temporal_memory summarize \
  --protocol experiments/temporal_scale_protocol.json --root outputs/scale_followup
```

Dynamic versus static is the primary contrast. A meaningful gain is at least
3 pp SR with3/4 positive seed pairs and the unchanged safety/progress limits.
Practical equivalence requires the paired90% t interval inside +/-3 pp for
aggregate SR only; failure to find a gain is not equivalence. The protocol was
frozen before any outcomes were inspected. Ordinary dynamic gating is not
automatically a new social-navigation mechanism.

### Four-seed Results (4 October 2026)

All16 models completed the frozen budget and1,536 evaluations. Only final
checkpoints are compared; neither intermediate snapshots nor the earlier
two-seed pilot are pooled into these results.

| Readout/reference | SR % | CR % | Timeout % | Successful time s | Successful path m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor GRU | 79.17 | 10.16 | 10.68 | 16.62 | 11.69 |
| KDA read, coefficient1 | 76.82 | 11.72 | 11.46 | 22.21 | 16.12 |
| KDA static channel scale | 79.43 | 9.64 | 10.94 | 20.96 | 14.98 |
| KDA dynamic gate | 79.69 | 10.94 | 9.38 | 19.43 | 14.56 |

The primary dynamic-minus-static SR differences for307/331/359/383 are
-1.04 /0.00 /-7.29 /+9.38 pp. Mean +0.26 pp; paired90% interval
[-7.83,+8.35] pp. Only one positive pair, two negative and one tie:
**NO_CONSISTENT_PILOT_GAIN**, and practical SR equivalence is **not** established.
All five pre-fixed contrasts fail the pilot-gain rule. Dynamic-minus-GRU is
only +0.52 pp with one positive pair and16.90% longer successful time;
static-minus-GRU is +0.26 pp with26.09% longer successful time. Successful
time/path averages concern different surviving episode sets, not paired
progress equivalence. Six-cell supporting results remain in the raw summary.

| Complete80-action score | Idle4090 median ms | Laptop CPU median ms |
| --- | ---: | ---: |
| Actor GRU, original batched implementation | 3.02 | 56.43 |
| Actor GRU, mathematically equivalent prefix reuse | 3.37 | 5.38 |
| KDA read | 9.65 | 11.00 |
| KDA static | 9.65 | 11.07 |
| KDA dynamic | 9.74 | 11.47 |

These are100 repetitions after20 warmups, one pre-fixed five-human root,
T24 and no simulator/smoothing time. Server timing starts after all training
processes exit; CPU timing uses the laptop. KDA's apparent CPU advantage over
the unreused GRU is absorbed by prefix reuse; no efficiency advantage is found
over the stronger compute control. This compact recurrence is not the optimized
FLA kernel. KDA actor state is160 KiB versus5 KiB for GRU at this configuration.

Actual process training time is11.77-15.15 min for GRU,34.80-39.29 for KDA
read,30.06-36.32 for static and28.30-41.77 for dynamic. Concurrent load varies
from six to eight jobs; these are recorded costs, not isolated throughput
benchmarks. Summed overlapping training/evaluation times are7.94/0.79 process
hours, not GPU-hours. Peak allocated memory per training process is724 MiB
for GRU and1,890 MiB for KDA.

All48 checkpoints reload with exact configuration/parameter counts and finite
weights. Each log contains50 IL epochs and1,000 RL episodes; every model has
the same96 expected cases. The learned static coefficients finish near0.501,
with the full four-seed range0.4982-0.5051, so this control is close to uniform
attenuation rather than a strongly differentiated channel calibration.

**Interpretation:** the old two-seed dynamic-gate advantage does not replicate
as a stable gain here. This neither proves static/dynamic equivalence nor
rejects temporal navigation, actor memory or KDA as a family. It does not
support selective motion-evidence revision or a new method claim. Keep GRU
as the health/reference baseline. The next justified diagnosis is to locate
the divergence using retained IL50/RL500 snapshots under the same evaluator,
then test one identified replay/readout-contract issue; do not search hundreds
of outcome-selected gate variants or rescue a favorable seed.

Scientific source is frozen at6dde31e. Local results are in
/home/abc/workspace/shixu/outputs/scale_followup, including the protocol/source
manifest, paired summary, full episode records, learning logs, three checkpoints
per model and GPU/CPU latency arrays. Code is versioned; weights are not added
to Git. All98 remote raw artifacts and nine scientific source files match
local SHA256 checksums; laptop timing also matches its original checksum.
The server-only temporary workspace is removed after verification, with the
installed environment left intact. No additional training or architecture
is started by this analysis.

## Occlusion Research Loop

The occlusion experiments use legal measured/retained tracks, five-person
ORCA IL and online MC refinement, followed by reloaded-final-weight tests on
5/10/20 people in circle and square. The frozen primary endpoint is equally
weighted 10/20-person success, with collision, timeout and progress checks.
The teacher may use simulator truth; the student receives only legal input.
Temporary server outputs are copied locally before deletion. Development
results are not final evidence: a promising mechanism needs unseen seeds and
cases, with all controls retrained on the same host.

The V5 private-successor comparison completed all four seeds419/443/467/491.
Overall SR/CR/timeout are75.00/10.16/14.84% for measured-only KDA branches,
77.34/10.68/11.98% for CV-pseudowrite branches, and85.94/5.99/8.07% for the
strongest completed context-GRU reference. Their primary SRs are71.48/74.61/
85.55%. Measured-only branching loses to that reference in4/4 pairs. Neither
private branching nor excluding legal CV writes establishes a navigation gain.
Results remain in outputs/occlusion_v5; this is a version-level negative result.

A separate frozen-consumer shadow replaces only currently retained hidden
positions/velocities with current simulator truth, without introducing unseen
people or modifying tracker memory. Four-seed primary gain is only0.39 pp
(one positive, two negative, one tie). This is not a full-future upper bound:
it shows no large demonstrated hidden-state accuracy headroom for that frozen
consumer, not that temporal reasoning or occlusion handling is unnecessary.
Records remain in outputs/occlusion_v1/*/current/truth_retained.json.

V6 tests physical actor memory before current candidate geometry/goal fusion.
It uses the official-shaped KDA no-short-convolution mixer, verified against
the pinned FLA layer, rather than claiming that mixer or its output gate as new.
All motion-family models receive the same physical features and legal elapsed
interval input. Only the custom elapsed-clock arm scales channel log-decay by
the real interval; vanilla KDA uses unit decay per measurement. Controls include
same-placement GRU, CV pseudowrites, identical-capacity zero motion history,
current-only and the strong completed context-GRU. There is no extra loss or
prediction model. The clock mechanism is a hypothesis, not a demonstrated win.

```bash
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_motion_protocol.json \
  --root outputs/occlusion_v6 --data outputs/occlusion_v6/demonstrations.pt \
  --arms motion_gru motion_kda motion_elapsed motion_imputed motion_nohistory \
  --device cuda --workers 2
```

Generic time-aware recurrence already exists in GRU-D and time-aware LSTM;
actor memory and delta-rule erase/write are also existing mechanisms. The
remaining question is whether legal observation-time semantics and this
physical/current fusion improve closed-loop navigation beyond those controls.
No novelty or safety guarantee is earned by passing numerical tests.

V6 scientific source is frozen at54d19cc. The 4090 runs PyTorch2.9.1/cu128,
and the3060 runs2.1.0/cu121; development comparisons retain this host boundary.
The CUDA packed-GRU test on2.9.1 differed from explicit stepping by2.36e-4
with default cuDNN TF32, and3.86e-6 with TF32 disabled. Float64 validates the
recurrence separately; production float32 kernels are not bitwise identical.
The archived optimized seed419 GRU replay changes controls in9/96 episodes
but changes no terminal outcomes. These checks cannot justify universal action
parity. Fresh confirmation must use one frozen implementation on one host.

### Physical-memory Results and the Next Readout Test

All20 V6 models completed50 IL epochs,1,000 online MC episodes and96
reloaded-final-weight evaluations each. Source, data, finite weights and the
complete episode keys are checked; all77 server artifacts match local checksums
before the temporary V6 workspace is removed.

| Model | Overall SR % | CR % | Timeout % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: |
| Current legal CV-track reference | 81.77 | 4.95 | 13.28 | 79.30 |
| Context-GRU reference | 85.94 | 5.99 | 8.07 | 85.55 |
| Physical-memory GRU | 87.50 | 4.43 | 8.07 | 83.59 |
| Physical KDA, unit observation clock | 80.99 | 3.65 | 15.36 | 76.56 |
| Physical KDA, elapsed clock | 80.99 | 2.86 | 16.15 | 76.95 |
| Physical KDA, CV pseudo-writes | 83.07 | 5.47 | 11.46 | 78.13 |
| KDA capacity control, zero committed history | 86.72 | 6.51 | 6.77 | 83.59 |

Elapsed-minus-unit primary differences are-9.38/+4.69/+7.81/-1.56 pp:
mean+0.39 pp, two positive and two negative. Elapsed-minus-physical-GRU is
-6.64 pp, with two losses and two ties. Fewer collisions are accompanied by
more timeouts, so this version does not meet the fixed success/progress rule.
The zero-history arm still receives legal interval/age input. These are
development results, not a family rejection or evidence that history is useless.

On836 uniformly sampled demonstration windows per checkpoint, first-layer
cross-actor key cosine averages0.297-0.327 across the four trained unit-clock
KDA models. Thus the earlier near-identical-address symptom is no longer
observed here; this statistic does not certify useful retrieval. Removing all
committed memory changes scalar values substantially, including hidden-track
windows, but value sensitivity does not establish better action ranking.

A separate four-seed CPU shadow checks the fixed2-second retention deadline.
Only previously seen expired actors still inside the legal history horizon
are eligible: current truth adds0.39 pp primary SR, while extending ordinary
CV retention adds0.78 pp. Of602 expired person-frames with a last measurement
still inside the legal prefix,31 are within2 m of the robot. This demonstrates
a lifecycle limitation but no large frozen-consumer headroom or residual beyond
the simple CV control. It does not justify another expiration architecture.
Raw records are in outputs/occlusion_v6/expiry_shadow.

V7 therefore replaces the readout interface, not another gate. Measured physical
actor streams write the same KDA memory; existing current robot/human fusion
moves before access and creates a different query for each candidate action.
The read-only arm contracts that query with the pre-existing matrix. It never
treats a hypothetical successor as a new measurement. A private-update KDA
control uses exactly the same fusion placement and parameters; a same-placement
GRU and a zero-committed-history control are also trained. No extra network,
prediction target, reward or data is introduced. Physical V6 references preserve
their original readout and budgets.

The mechanism hypothesis is that action-relevant retrieval can improve the
use of physical history beyond a candidate-independent actor summary. It is
not presumed true. The proposed read-before-write primitive is already present
in [DRAM](https://arxiv.org/abs/2609.32453), and task-conditioned retrieval is
not new. [Advantage-Driven Explicit Memory](https://arxiv.org/abs/2608.25610)
retrieves recurrent navigation experiences across episodes, unlike this bounded
per-actor measured stream. [Kimi Linear](https://arxiv.org/abs/2510.26692)
provides the mixer. A possible paper claim must concern the candidate-query/
occluded-actor interface and validated navigation benefit, not qS, KDA, actor
identity or generic read/write separation alone. Novelty remains unearned.

V7 is frozen at36d1702 with the same128 demonstrations, four paired seeds,
5-person training and5/10/20-person evaluation. Its pre-outcome protocol is
experiments/occlusion_query_protocol.json; results are in outputs/occlusion_v7.
All controls must remain in the comparison. Fresh seeds/cases remain untouched
until an eligible development winner is chosen.
