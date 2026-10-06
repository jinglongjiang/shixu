# shixu

A compact crowd-navigation research project extracted from the local camrl
Mamba-VL parent. It retains the cleaned value-learning framework, temporal
experiments and their reproducibility assets. A runnable experiment is not a
validated new navigation method.

## Core Path

```text
legal observations -> spatial/actor features -> temporal encoder -> scalar value
                   -> candidate successor/value lookahead -> executed action

ORCA demonstrations -> Monte Carlo value initialization -> online MC refinement
```

GRU is the default baseline. Optional Mamba supports legacy comparison;
KDA/GDN2 actor-memory variants remain available for reproducing the completed
experiments. Forecasting, occlusion and action-commitment experiments are
separate research paths, not components that must all be stacked together.

## Code Map

| Location | Responsibility |
| --- | --- |
| shixu/model.py, shixu/temporal.py | Value models and shared actor-memory operators |
| shixu/features.py, shixu/observations.py | Legal inputs, history alignment and actor association |
| shixu/policy.py | Candidate actions, successor scoring and execution contract |
| shixu/replay.py, shixu/training.py | Episode-safe history windows and MC value training |
| shixu/runner.py, shixu/cli.py | Collection, evaluation and explicit training entry points |
| shixu/motion.py, shixu/forecast.py | Preserved motion/forecast experiment modules |
| shixu/commitment*.py, shixu/commit*.py | Commitment, release and initiation experiments |
| experiments/ | Diagnostic scripts and frozen experiment protocols |
| autoresearch_nav/ | Bounded commit-gate harness and frozen instructions |
| tests/ | Correctness, contract and reproducibility checks |
| vendor/crowd_sim/ | Local simulator dependency |
| outputs/ | Local checkpoints, datasets, logs, traces and evaluation evidence |
| reports/ | Six consolidated report categories and an integrity manifest |

## Report Index

| Category | Consolidated Report | Scope |
| --- | --- | --- |
| KDA and temporal architectures | [Temporal experiments](reports/kda-temporal-experiments.md) | Processing order, memory comparisons and occlusion V1-V8 |
| Latent/history information | [Information audits](reports/latent-history-information.md) | Hidden-state recovery, circle shortcuts and history controls |
| Prediction and state consumption | [Forecast and state](reports/forecast-control-and-state.md) | Forecast-to-control consumers and current-state reconstruction |
| Value contracts and repeatable defects | [Value and defect audits](reports/value-and-defect-audits.md) | Action attribution, CV419 training contract and failure atlas |
| External temporal problems | [External problem checks](reports/external-temporal-problems.md) | PaS discovery/confirmation and interaction-response audit |
| Multistep control and commitment | [Commitment experiments](reports/action-commitment-experiments.md) | Headroom, V0/V0.1, advantage gate and bounded autoresearch |

Original stage text, numbers, negative results and withdrawn claims are kept
unchanged inside the corresponding source sections. The previous long README
is archived by topic rather than discarded. Identical remote report snapshots
are deduplicated; differing snapshots are retained with their original paths.
[The manifest](reports/archive-manifest.json) records source SHA-256 hashes and
byte ranges, allowing every original report to be reconstructed and checked.

Earlier proposed next steps are historical, not authorization to restart
closed experiments. Later closure results take precedence. In particular,
the KDA trials did not establish stable extra navigation value; observed
commitment/release effects did not make the final learned gate a validated
method. The bounded gate search ended in FINAL_CONFIRMATION_FAILED.

The frozen autoresearch_nav/program.md still names its original final report,
AUTONAV_OVERNIGHT_REPORT.md. That report is now a preserved source section in
the commitment archive; the frozen program and harness were not rewritten.

## Install and Verify

```bash
python -m pip install -e .
python -m pip install 'git+https://github.com/sybrenstuvel/Python-RVO2.git'
export PYTHONPATH=vendor:.
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

GRU does not require Mamba or custom CUDA kernels. Legacy Mamba comparison
requires the optional dependencies and matching CUDA setup. Three legacy
parity tests require CAMRL_PARENT and CAMRL_CHECKPOINT; otherwise they skip.

To evaluate existing trained weights, use a checkpoint compatible with its
saved configuration:

```bash
export PYTHONPATH=vendor:.
python -m shixu.cli evaluate --weights /path/to/checkpoint.pt \
  --device cpu --people 5 --geometry circle --cases 0 1 \
  --output outputs/local-evaluation.json
```

Evaluation requires weights. Smoke checks are untrained interface checks,
not success-rate evidence. Training runs only through explicit train or
experiment commands; this documentation cleanup starts no training.

## Cleanup Boundary

Only regenerable Python/pytest caches and an obsolete build wheel were
deleted as disposable artifacts. Individual Markdown sources were removed
only after exact archive reconstruction passed. Core code, tests, frozen
protocols, checkpoints, datasets, raw trajectories and learning logs were
preserved, including negative experiments and unfinished historical runs.
