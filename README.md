# shixu

A small temporal crowd-navigation research framework extracted from the user's
local camrl Mamba-VL project. This is a cleaned baseline, not a new algorithm or
a claim that actor-specific memory has already improved navigation.

## One Main Path

```text
legal observations -> frame encoder -> temporal encoder -> scalar value
                    -> original candidate successor/value lookahead -> action

ORCA demonstrations -> Monte Carlo value initialization -> online MC refinement
```

GRU is the default temporal baseline. Mamba is optional and retained only for
legacy comparison. There are no Double-Q/PPO/SAC branches, auxiliary prediction
heads, Bayesian modules, teacher networks at deployment, or fallback backbones.

The intended next change is to move temporal encoding before crowd pooling:

```text
identity-bound human histories -> shared temporal encoder -> crowd pooling
                              -> scalar value -> unchanged lookahead
```

That actor-specific variant is not implemented or trained in this initial
baseline. No dual-memory system or contradiction detector is being added in
advance of the problem/headroom audit.

## Layout

| File | Responsibility |
| --- | --- |
| shixu/features.py | Legacy observation contract and history windows |
| shixu/model.py | Frame encoder, replaceable temporal encoder, value head |
| shixu/policy.py | Original action support and successor-value evaluation |
| shixu/replay.py | Episode-safe windows and MC targets; no duplicated window archive |
| shixu/runner.py | One runner for collection, training and evaluation |
| shixu/training.py | ORCA value initialization and online MC refinement |
| shixu/cli.py | Explicit collection/evaluation/training commands |
| vendor/crowd_sim | Frozen local simulator dependency |

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

# Explicitly opt into training; not executed during this repository cleanup.
python -m shixu.cli train --il-episodes 5 --rl-episodes 10 \
  --device cuda --output weights/gru.pt
```

Models must use the same config when comparing them. The optional local-source
regression tests use environment variables CAMRL_PARENT and CAMRL_CHECKPOINT;
they check features, value outputs, actions and history against the original
source. They skip explicitly when those local assets are unavailable.

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
