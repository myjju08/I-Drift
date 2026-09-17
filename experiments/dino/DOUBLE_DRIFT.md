# DINO Double Drift

These configurations apply the Double Drift methods from
[`cosmosjhj/I-Drift` at `27a0e4fb54c8ed04e2538831a9df86a85c8c25ef`](https://github.com/cosmosjhj/I-Drift/commit/27a0e4fb54c8ed04e2538831a9df86a85c8c25ef)
to the frozen DINO ResNet-50 encoder. They use the shared implementation in
[`drifting_core/imagenet_loss.py`](../../drifting_core/imagenet_loss.py),
[`drifting_core/double_drift.py`](../../drifting_core/double_drift.py), and
[`train_imagenet_gen.py`](../../train_imagenet_gen.py). Double Drift performs
two successive reverse-field evaluations; it is distinct from
`drift_matching: dual-drift`, which mixes forward and reverse drift.

## Matched comparison

| Configuration | `double_drift_mode` | `(c0, c1)` | Replay / GAN |
| --- | --- | --- | --- |
| [Baseline](configs/double_drift_baseline.yaml) | `off` | `(1.0, 0.0)` | Disabled |
| [Feature Double Drift](configs/double_drift_feature.yaml) | `feature` | `(0.75, 0.25)` | Disabled |
| [Sample Double Drift](configs/double_drift_sample.yaml) | `sample` | `(0.75, 0.25)` | Disabled |

Each YAML is self-contained. The three arms differ only in their logging
names, Double Drift mode, and coefficients. Their common recipe comes from
the existing
[`S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched.yaml`](../../configs/gen/S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched.yaml)
baseline; the original archived presets and their provenance hashes remain
unchanged.

| Setting | Value in all three arms |
| --- | --- |
| Generator | S/4, RGB 256×256, patch 32, hidden/condition 384, depth 12, heads 6 |
| Encoder | Frozen DINO ResNet-50, stage 3/4 + global + `norm_x` |
| Feature weights | Stage 1/2 excluded; `norm_x` × 2; group normalization |
| Drift | `rev-drift`, `R_list: [0.2, 0.05, 0.02]`; balanced attraction/repulsion |
| Sampling per rank | 4 labels × G32, P64 real positive and N32 real negative per label |
| Global generated batch | 256 images/update with 2 ranks |
| Seed | 43 + rank |
| Duration | 40 generated epochs = 200,183 updates for 1,281,167 training images |
| Optimizer | AdamW, LR 0.0004, betas 0.9/0.95, weight decay 0, warmup 2,000 |
| EMA / gradient clipping | 0.999 / 2.0 |
| Evaluation | At initialization and every 10 generated epochs; 1,024 images, CFG 1/2/3 |
| Checkpoints | Every 10 generated epochs |
| I/O / precision | Standard raw ImageNet loader/bank; BF16 generator and DINO |

`total_generated_epochs` determines the training length using the actual
dataset size and global generated batch. The YAML's `total_steps` is the
matching two-rank value. These are new DINO experiments, with no reported
training result or measured runtime attached to the configurations.

## Feature-space method

Let `z` be a generated DINO feature tensor, `s` the first evaluation's input
scale, and `u = z/s`. `V` is the existing reverse-drift field in normalized
feature coordinates, including its per-temperature force RMS normalization.
The target is:

```text
u1 = stopgrad(u + c0 * V(u))
u2 = stopgrad(u1 + c1 * V(u1))
Lfeature = mean((u - u2)^2)
```

The second field moves both generated queries and their generated negative
particles. Real positive/negative references, labels, CFG weights, and the
first distance scale stay fixed. If replay were enabled, historical
references would also stay fixed; this comparison disables replay. Each
field receives its own per-temperature RMS normalization; the final summed
displacement is not normalized again. Targets are detached and no gradient
passes through either field evaluation.

This happens separately for each active DINO feature tensor, followed by
the baseline's feature-group weighting. It adds a field evaluation without
another DINO encoder forward. Stage 1/2 features remain excluded.

## Sample-space method

Here `x0` is the generated RGB tensor before DINO preprocessing and
`Ldrift` is the ordinary weighted DINO reverse-drift loss:

```text
g0 = dLdrift(x0)/dx0
a  = sample_step_rms / max(RMS(g0), 1e-12)
x1 = stopgrad(x0 - c0*a*g0)
g1 = dLdrift(x1)/dx1
x2 = stopgrad(x0 - a*(c0*g0 + c1*g1))
Lsample = sum((x0 - x2)^2) / (2*a)
```

The sample gradient is exactly `c0*g0 + c1*g1`. The scale `a` is computed
once and reused. These configs use `double_drift_sample_step_rms: 0.1`, so
the first probe displacement has global RMS 0.075 when `RMS(g0) >= 1e-12`.
The configured global statistics reduce the squared gradient sum and count
across ranks. Sample-space coefficients act on RGB gradients; they do not
have the same interpretation as feature-space coefficients.

The probe is detached from the generator. DINO evaluates it with gradients
enabled only for its input; real features, labels, CFG weights and feature
selection are reused, and current generated negatives are rebuilt from the
probe features. The second loss recomputes the ordinary drift normalization
at the probe. No Hessian or generator rerun is required. The final backward
updates the generator through `x0`; DINO stays frozen.

Sample mode adds a DINO generated-feature forward and input-gradient
calculation. It therefore costs more compute than feature mode. The probe
uses the configured generated-feature microbatch size; resource use and
throughput need measurement on the selected hardware. Both Double Drift
modes require `rev-drift` and reject learned feature adapters or GAN/
adversarial branches.

## Assets and commands

Install the repository requirements in the selected Python environment and
run these commands from the repository root. Dataset and official DINO
weights are supplied externally. Defaults are repository-relative:
`data/imagenet/raw_ilsvrc2012` and
`weights/pretrained/dino_resnet50_pretrain.pth`. The supplied calibration
JSON, raw ImageNet manifest, and DINO checkpoint hashes must match the
existing calibrated DINO recipe; see [the asset guide](../../DINO_EXPERIMENTS.md#cpu-checks-and-assets).

CPU preflight does not initialize CUDA or launch training:

```bash
python -m experiments.dino.train \
  --config experiments/dino/configs/double_drift_feature.yaml
```

Add `--check-assets` to validate externally supplied assets before launch:

```bash
python -m experiments.dino.train \
  --config experiments/dino/configs/double_drift_feature.yaml \
  --check-assets \
  --imagenet-path /path/to/raw_ilsvrc2012 \
  --feature-checkpoint /path/to/dino_resnet50_pretrain.pth
```

Launch on two available GPUs with a fresh work directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  --module experiments.dino.train \
  --config experiments/dino/configs/double_drift_feature.yaml \
  --train --workdir runs/dino_double_drift_feature_new \
  --imagenet-path /path/to/raw_ilsvrc2012 \
  --feature-checkpoint /path/to/dino_resnet50_pretrain.pth
```

Choose the baseline or sample YAML and a separate workdir for the other
arms. `--config` uses standard I/O by default and records its effective
configuration in the workdir. The wrapper requires exactly two ranks and a
new workdir. On a Vast instance, register the foreground command with
supervisor using the instance's environment/logging wrapper and select
unoccupied GPUs.

All three YAMLs retain W&B project `Feature encoder - S4 model`, as required
by the calibration guard. Set `logging.entity` in a copied/custom config to
an accessible account if needed; logging must initialize successfully.

## Validation

The DINO entry point validates Double Drift mode, coefficients, sample probe
RMS, reverse-drift selection, frozen DINO selection, and disabled learned
adapters/adversarial branches when loading the config, before allocating
CUDA resources. CPU preflight passes for all three supplied configurations;
their generated global batch is 256, and every scientific setting apart
from the Double Drift controls is identical.

For numerical source parity, supply an independent checkout at the pinned
upstream commit:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/verify_double_drift_port.py \
  --source /path/to/cosmosjhj-I-Drift-at-27a0e4f \
  --target . --output-dir runs/dino_double_drift_parity
```

The [checked-in parity report](../../docs/DOUBLE_DRIFT_PARITY.json) records
360 deterministic CPU cases and 25,080 exact comparisons with zero maximum
absolute error. It covers field/loss values and gradients, feature
aggregation, sample-space correction, optional history, and force-balance
settings. These numerical checks do not measure full DINO training speed
or predict final image quality.

The focused CPU regression suite is:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_dino_double_drift_config.py \
  tests/test_dino_double_drift.py tests/test_double_drift.py \
  tests/test_dino_experiments.py tests/test_ssl_resnet.py \
  tests/test_required_wandb.py
```

Validation on this change passed **78 tests and 10 subtests**. The new DINO
integration tests exercise the production RGB normalization, feature
selection, multi-chunk extraction, and rematerialization with a small
nonlinear backbone. They verify frozen encoder parameters and buffers,
unchanged real/history references, generator gradients, a single generator
forward, sample-probe re-encoding, and exact baseline reduction at
`c0=1, c1=0`. These are CPU checks; pretrained 256-pixel BF16 multi-GPU
training has not been run for the new configurations.
