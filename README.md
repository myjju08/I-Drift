# I-Drift

## Corrective Field: S4 / MAE256

The active three-run suite uses this repository's reverse drift with the same
existing ImageNet latent cache and frozen MAE256 checkpoint on `srv02`.

| Run | Generator supervision |
| --- | --- |
| `baseline` | MAE reverse drift |
| `replay_double` | MAE + frozen epoch-10 replay (`rho=0.35`) + double drift |
| `replay_double_gan` | The same replay and double drift + `0.1` non-saturating GAN loss |

Double drift constructs the detached feature target `x + V(x) + V(x + V(x))`.
The second field evaluation uses the same reference particles and calibration.
Each run trains S4 for 40 generated epochs, with P32/N32/G32, seed43, and two
RTX3090 GPUs. W&B records all three under **Corrective Field**.

[Run instructions, exact settings, validation and evaluation limits](docs/CORRECTIVE_FIELD.md)
are kept together with the [three matched configs](configs/corrective_field).
Submission freezes the committed code, queues a two-GPU validation job, and
starts the three training jobs only after that validation succeeds.

```bash
python scripts/preflight_corrective_field.py --config-only
python scripts/submit_corrective_field.py          # inspect the commands
python scripts/submit_corrective_field.py --submit # queue validation and training
```

The earlier experiment families and their original presets are documented below.

### Learning from Its Own Past through Repulsive Generative Replay

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

## Implemented experiment families

| Family | Generator supervision | Entry point |
| --- | --- | --- |
| Reverse / forward / dual drift and historical replay | Frozen encoder drift, optional detached historical repulsion | `train_imagenet_gen.py` and `scripts/` |
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
CNN drift receives neither replay samples nor replay weights. Current MAE
recipes have replay disabled. Their latent discriminator and loss weights
are different from the RGB DINO discriminator. The supplied experiment
presets preserve these distinctions.

## Core idea

For each current generated query, the reverse-drift target pool contains

```text
[current generated | real negative | historical generated | real positive]
```

The historical samples come from a frozen, class-conditioned generator
snapshot. At replay ratio `rho`, generated repulsion is divided as

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
