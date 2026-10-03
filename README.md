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

Training uses the first T-1 real frames as the prefix and the last real frame
as the query. In inference, the query is an analytic candidate successor. It
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
Navigation results and evidence-specific ablations must justify any narrower
claim before the architecture is selected as a paper method.

The latency replay also measures the original trained actor GRU with a shared
prefix computation, preserving its value function. This prevents attributing
generic prefix reuse to a new memory operator. Natural-event shadow comparisons
use common roots from the first legal near-motion event in each pre-fixed parent
episode, not the best events for a new arm. They remain exploratory supporting
evidence, not a replacement for a negative paired SR result.
