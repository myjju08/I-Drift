# I-Drift

### MAE drift, Replay, Double Drift, force balance, and adversarial supervision

ReplayDrift is a follow-up to [Drifting](https://github.com/lambertae/drifting)
and DualDrift. It lets a generator learn not only from real data, but also from
where its previous selves have already been: class-conditioned samples from an
earlier generator state are replayed as detached repulsive particles.

The project also collects the efficiency controls developed while making
Drifting practical with smaller generator token counts, narrower feature
encoders, fewer GPUs, and lower-cost loss construction.

This repository now includes the adversarial extensions and the DINO/MAE
experiment runtimes used by the source I-Drift instance on **2026-09-10**.
The implementation is local to this checkout; datasets, pretrained weights
and checkpoints are supplied separately.

Replay memory policies, Double Drift, and attraction/repulsion balance are ported from
[`cosmosjhj/I-Drift` at `27a0e4f`](https://github.com/cosmosjhj/I-Drift/commit/27a0e4fb54c8ed04e2538831a9df86a85c8c25ef).
The [Corrective Field experiment guide](docs/CORRECTIVE_FIELD.md) describes
the matched S4 / MAE-256 comparison, exact assets, equations, and launch steps.
[The upstream comparison](docs/UPSTREAM_PARITY.md) records what matches the source
and the remaining differences beyond DINO/GAN.

## Corrective Field: three S4 experiments

The current six-GPU deployment uses **srv06 / RTX A5000**, with the same
scientific settings and copied assets. See [the srv06 guide](docs/SRV06_EXPERIMENTS.md)
and `configs/corrective_field_srv06/`. The srv02 configuration remains available.

All three configurations train from scratch for 40 generated-sample epochs
with the same FP32 latent cache, frozen MAE-256 checkpoint, seed 43, and
global generated batch 512. Each job uses two GPUs on the selected Slurm node.

| Configuration | Historical replay | Double Drift | GAN |
| --- | --- | --- | --- |
| [baseline](configs/corrective_field/baseline.yaml) | Off | Off | Off |
| [replay_double](configs/corrective_field/replay_double.yaml) | `rho=0.35`, bank frozen at epoch 10 | Feature, `c0=0.75`, `c1=0.25` | Off |
| [replay_only](configs/corrective_field/replay_only.yaml) | Same | Off | Off |

For the selected feature-space method, normalized features move according to
`u1 = u + 0.75*V(u)` and `u2 = u1 + 0.25*V(u1)`. Both generated queries and generated
negative particles move for the second field evaluation; real and historical
references stay fixed. The final displacement is not normalized again.
All three arms have GAN supervision disabled.

The repository also exposes the upstream sample-space gradient-probe method
through `double_drift_mode: sample`; its coefficients and latent probe RMS
have a different interpretation. See the guide before changing modes.
New run metrics are logged online to W&B project **Corrective Field**.
The historical result tables below are separate experiments.

## Replay and attraction/repulsion controls

These settings go under `train:` in YAML, using the upstream names:

```yaml
train:
  historical_gen_replay: true
  historical_gen_replay_ratio: 0.35
  historical_gen_replay_count: 16
  historical_gen_replay_bank_count: 16
  historical_gen_replay_start_generated_epochs: 10.0
  historical_gen_replay_source: frozen_snapshot
  historical_gen_replay_policy: frozen  # frozen | fifo | reservoir | usage_budget
  historical_gen_replay_update_count: 2
  historical_gen_replay_update_interval_steps: 1
  historical_gen_replay_usage_budget: 4
  double_drift_mode: feature            # off | feature | sample
  double_drift_c0: 0.75
  double_drift_c1: 0.25
  rev_drift_balance_delta: 0.0
  rev_drift_balance_anneal_steps: 0
  rev_drift_balance_diagnostics: false
```

The coefficients are **attraction = 1 + delta**, **repulsion = 1 - delta**, with
`-1 < delta < 1`. For example, `delta: -0.05` selects 0.95 and 1.05. They scale
the positive and negative force components **after** mutual-mass coupling and
before per-temperature RMS normalization. Both Double Drift field evaluations
receive the same coefficients. `rev_drift_force_multiplier` separately scales
the resulting force; it does not change the attraction/repulsion balance.

A positive `rev_drift_balance_anneal_steps` linearly reduces delta to zero using
the absolute training step. The default zero keeps delta constant. Optional
diagnostics report radial alignment, centered feature RMS, and component RMS.
Nonunit balance and Double Drift require reverse drift. Double Drift with
learned feature adapters or GAN/adversarial systems is still unsupported and
raises an error; the separate DINO/GAN implementations remain available.

`frozen` retains the epoch-10 bank. `fifo`, `reservoir`, and `usage_budget` start
from that bank and stream detached candidates from the existing generator
forward pass on the configured cadence. These policies, their sampling and
serialized state match upstream. The current S4 comparison uses **frozen** and
**delta 0** for both replay arms, so the comparison changes only Double Drift.

## Implemented experiment families

| Family | Generator supervision | Entry point |
| --- | --- | --- |
| Reverse / forward / dual drift and historical replay | Frozen encoder drift, optional detached historical repulsion | `train_imagenet_gen.py` and `scripts/` |
| Double Drift with replay | Two detached feature-field evaluations with fixed real/history references | `configs/corrective_field/`, `drifting_core/double_drift.py` |
| DINO Double Drift | Frozen DINO reverse drift with upstream feature-space or sample-space correction | [Three matched configs and guide](experiments/dino/DOUBLE_DRIFT.md) |
| DINO raw conditional GAN | Frozen DINO drift + non-saturating logistic GAN loss | `experiments.dino.train` |
| DINO adversarial feature drift | Frozen DINO drift + drift in learned discriminator coordinates | `experiments.dino.train` |
| DINO mixed objective | Frozen DINO drift + GAN loss + learned-feature drift | `experiments.dino.train --config ...` |
| MAE same-cache control | Frozen MAE drift on 4×32×32 latents | `experiments.mae.runtime` |
| MAE spatial latent GAN / feature drift | MAE drift + GAN loss or learned drift from an 8×8/4×4/4×4 latent discriminator | `experiments.mae.runtime` |
| Frozen-feature GAN heads and feature adapters | Earlier hinge-loss / adapter ablations | `models/feature_gan.py`, `models/feature_adapter.py`, existing ablation scripts |

Read [the objective and update definitions](docs/ADVERSARIAL_METHODS.md),
[DINO experiments](DINO_EXPERIMENTS.md),
[MAE experiments](experiments/mae/README.md), and
[port validation](docs/PORT_VALIDATION.md) for details.

The active DINO recipe applies replay only to frozen DINO drift; its learned
CNN drift receives neither replay samples nor replay weights. The archived
MAE presets under `experiments/mae/configs/` have replay disabled. Their latent discriminator and loss weights
are different from the RGB DINO discriminator. The supplied experiment
presets preserve these distinctions.

## Core idea

For each current generated query, the reverse-drift target pool contains

```text
[current generated | real negative | historical generated | real positive]
```

The historical samples come from a class-conditioned generator bank,
frozen at the configured epoch by default; rolling policies optionally replace
entries after that boundary. At replay ratio `rho`, generated repulsion is divided as

```text
current weight = 1 - rho
history weight = rho * G / H
```

where `G` is the number of current generated particles and `H` is the replay
count. Historical particles are detached targets: they do not retain a
generator or feature-encoder backward graph.

## Preliminary ImageNet-256 result

These are earlier B/4 replay results, not results of the newly ported
adversarial DINO or MAE experiments.

The controlled B/4 MAE-256 comparison below starts every continuation from the
same epoch-10 checkpoint. Evaluation uses 50,000 samples at CFG 1.4.

| Method at epoch 40 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|
| Reverse drift, no replay | 18.510 | 81.35 | 0.7431 | **0.4060** |
| ReplayDrift, `H=16`, `rho=0.5` | **11.822** | **122.53** | **0.7839** | 0.3305 |

Replay improves convergence, FID, and precision, while the lower recall exposes
an important quality–coverage trade-off. A lower replay ratio (`rho=0.35`) is a
useful balanced setting.

## Setup

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the CPU regression suite, also install `requirements-dev.txt`.
The packed DINO native color decoder additionally needs a C compiler and
an x86 CPU with SSE2; build it with
`python -m experiments.dino.io.build_native`. The documented NumPy decoder
fallback preserves decoded bytes and sampling RNG without the native build.

ImageNet, latent caches, pretrained MAE weights, FID statistics, checkpoints,
and generated samples are intentionally not stored in Git. Supply their paths
locally through the launch-script environment variables.

Example B/4 reverse-drift launch:

```bash
TORCHRUN_BIN="$(command -v torchrun)" \
IMAGENET_PATH=/path/to/ILSVRC2012 \
IMAGENET_CACHE_PATH=/path/to/image_latents \
MAE_CKPT=/path/to/mae_latent_256.pt \
GPU_IDS=0,1 NPROC_PER_NODE=2 \
LAUNCH_MODE=background \
bash scripts/run_B4_rev-drift_mae256.sh
```

The historical-replay causal and efficiency launchers are under `scripts/`.
Their configurations keep the baseline and replay comparisons matched in
checkpoint, seed, target count, and evaluation protocol.

## DINO: inspect and reproduce the active recipe

Run from this repository root. A DINO command without `--train` performs a
CPU preflight and does not launch training:

```bash
python -m experiments.dino.train \
  --preset dino_only_replay_feature
```

Use `--check-assets --check-generator` with explicit local assets for
additional checks. The other presets are `replay_feature_hardcopy`,
`replay_raw_gan_d32`, and `replay_raw_gan_d128`. The first is the earlier
both-branches-replay feature experiment; the latter two reproduce the
replay-plus-logistic-GAN recipes.

Example launch on two available GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  --module experiments.dino.train \
  --preset dino_only_replay_feature --train \
  --imagenet-path /path/to/raw_ilsvrc2012 \
  --feature-checkpoint /path/to/dino_resnet50_pretrain.pth \
  --workdir runs/dino_only_replay_feature_new
```

The existing `configs/gen/*raw-conditional-gan*.yaml`,
`*adversarial-feature-drift.yaml`, and
`*mixed-gan-feature-drift*.yaml` retain the older ablation settings. Pass
one with `--config` to explore those objectives; do not substitute it for
the active-run preset and expect identical EMA, structure or replay behavior.
Dataset and weight paths in copied S4 configs are repository-relative;
the calibration JSON is included under `experiments/dino/calibration/`.

## MAE: same-cache control and spatial latent discriminator

Supply the exact frozen MAE checkpoint, VAE assets, and complete packed
latent cache. For example, to reuse the assets already on this instance:

```bash
export IDRIFT_IMAGENET_PATH=/workspace/I-Drift/data/imagenet/raw_ilsvrc2012
export IDRIFT_MAE_CACHE=/workspace/I-Drift/data/imagenet/latent_cache_256_mae_gan_20260910
export IDRIFT_MAE_CHECKPOINT=/workspace/mae256_gan_prepare_20260910/artifacts/mae_latent_256/ckpt_ema.pt
export IDRIFT_MAE_VAE=/workspace/mae256_gan_prepare_20260910/artifacts/sd-vae-ft-mse

python -m experiments.mae.runtime \
  --config experiments/mae/configs/raw_gan.yaml \
  --preflight --verify-asset-hashes
```

Those are explicit **asset** locations, not source-code dependencies.
On another machine, set the variables to the corresponding local paths.
The MAE guide documents cache construction and exact asset provenance.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  --module experiments.mae.runtime \
  --config experiments/mae/configs/raw_gan.yaml \
  --workdir runs/mae_spatial_raw_gan_new
```

Select `control.yaml` or `feature_drift.yaml` for the other current MAE
recipes. Raw GAN weight is **0.15**; learned-feature drift weight is **0.1**.
Both use D32, hard-copy target updates, no structure penalty and no replay.
The latent architecture is part of checkpoint compatibility.

Use fresh work directories for new comparisons; the DINO wrapper requires
one, while the MAE trainer resumes an existing latest checkpoint and W&B
identity where applicable. W&B remains enabled in the reproduction presets.
For another account, copy a DINO preset to a custom YAML, change
`logging.entity`, and select it with `--config`; matched DINO runs retain
the source guard requiring `logging.project: Feature encoder - S4 model`
and `use_wandb: true`. MAE logging account/project can be configured in YAML.
For sustained runs on this Vast instance, run
the command in a supervisor wrapper following the instance guide. The port
does not register or start additional services.

## Validation and experiment state

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
```

Tests cover the logistic signs, generator/discriminator gradient isolation,
lazy R1, distributed synchronization, target EMA/hard copy, replay routing,
checkpoint validation, latent spatial features and lossless data loading.
The port also has numerical comparisons with the original local runtimes;
these source-comparison tests skip when the original workspace is absent.
The ordinary regression tests run from this checkout alone.

[The validation report](docs/PORT_VALIDATION.md) records the executed checks,
source/config provenance and limitations. Existing training processes and
source experiment files were not changed by this port. Full GPU retraining
and final-metric reproduction are separate from the CPU parity checks.

## Repository policy

The following stay local and are ignored by Git:

- datasets and latent caches;
- pretrained weights and checkpoints;
- generated samples and evaluation archives;
- experiment runs, logs, PID files, and W&B state;
- Python, test, profiler, and editor caches.

## Attribution

ReplayDrift builds on the public Drifting implementation and the DualDrift code
lineage. Please preserve upstream attribution when redistributing this work.

## Status

This is research code under active development. Reproducibility scripts and
ablation configurations are included, but paths to datasets and pretrained
weights must be configured for each environment.
