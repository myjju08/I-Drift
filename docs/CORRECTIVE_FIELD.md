# Corrective Field: S4 / MAE256 / 40 generated epochs

This suite runs the requested three-arm comparison from `myjju08/I-Drift`
(base commit `c6e8280`), using the existing FP32 latent cache and frozen MAE256
checkpoint. The launcher archives the final committed source and runs every arm
from that same immutable archive.

| Configuration | Replay | Double drift | Generator GAN term |
| --- | --- | --- | --- |
| `configs/corrective_field/baseline.yaml` | Disabled | Disabled | None |
| `configs/corrective_field/replay_double.yaml` | rho 0.35, freeze at epoch 10 | Enabled | None |
| `configs/corrective_field/replay_double_gan.yaml` | rho 0.35, freeze at epoch 10 | Enabled | `0.1 * softplus(-D(G(z), y)).mean()` |

Each YAML is self-contained. Only the run name, replay flag, double-drift flag,
GAN mode, and GAN coefficient differ. `scripts/preflight_corrective_field.py`
rejects accidental differences in all other settings.

## Objectives and the epoch-10 freeze

The baseline keeps the existing reverse-drift objective on frozen MAE features.
With double drift, the target is

```text
x1 = x + V(x)
target = stop_gradient(x1 + V(x1))
       = stop_gradient(x + V(x) + V(x + V(x)))
```

The drift target is constructed in the MAE feature space. The second field
evaluation uses the shifted query and the same original reference pools,
distance/input scales, per-temperature force normalizers, and bandwidths.
Gradients update the generator through the original generated features; the
MAE weights remain frozen. Double drift applies from the first step in arms 2
and 3.

Replay retains 16 recent generated latents per class, in FP16 storage, during
the first 10 generated epochs. At the boundary, each rank freezes its class
bank and saves `historical_gen_replay_rank*.npz`. These are the latest per-class
samples collected before the boundary, not samples regenerated from an EMA
checkpoint. Replay starts on the following training step and the bank is no
longer refreshed. The generator continues training to epoch 40; the MAE remains
frozen throughout all 40 epochs.

For `G=32` current generated samples and `H=16` historical samples per label,
the generated-negative pool uses current weight `1-rho = 0.65` and historical
weight `rho*G/H = 0.70` per sample. This preserves total mass `G` with 35% of
that mass assigned to history. The original positive and unconditional-negative
sampling settings remain identical across the three arms.

Arm 3 adds a class-conditional discriminator on the original `4 x 32 x 32`
latent tensors. Its non-saturating generator loss is added to the replay +
double-drift loss with coefficient 0.1. The discriminator is trained separately
with real/fake logistic loss and lazy R1 (`gamma=1`, every 16 steps), using base
width 32, Adam learning rate `1e-4`, and betas `(0, 0.99)`. It does not replace
the frozen MAE feature metric.

## Shared settings and reused assets

| Setting | Value |
| --- | --- |
| Generator | S4, latent `4 x 32 x 32`, patch 4, width 384, depth 12, heads 6 |
| Precision | BF16 generator and MAE compute; original FP32 latent inputs |
| Batch | 8 labels/rank x 2 ranks x 32 generations = 512 generated images/step |
| Sampling | P32, N32, G32; positive bank 128/class; seed 43 |
| Optimizer | AdamW `lr=4e-4`, betas `(0.9, 0.95)`, weight decay 0, 2,000-step warmup |
| Duration | 40 generated epochs = 100,092 updates for 1,281,168 rows |
| Replay boundary | 10 generated epochs = 25,023 completed updates |
| Save cadence | Every 0.5 generated epoch; latest checkpoint and 4 recent checkpoints |
| Evaluation | At epoch 0 and every 10 generated epochs, 1,024 samples, CFG 1/2/3 |
| Resources per arm | `srv02`, 2 RTX3090 GPUs, 6 CPUs, 80 GB, 3-day time limit |
| W&B | Online project `Corrective Field`, entity `a01065522071-kaist-digital-humanities-and-social-science` |

The generated-epoch clock is `completed_updates * 512 / 1_281_168`.
The last update slightly exceeds exactly 40 epochs because updates are whole
batches. The Slurm time limit is a limit on an individual allocation, not an
estimate of training completion time.

All paths below refer to existing assets on `srv02`:

- Latent mmap cache: `/home/juhyeong/replay-drift-cache/imagenet-latents-f32-v1`.
- Original latent source: `/data/juhyeong/imagenet/imagenet_latents`.
- Frozen MAE256: `/home/juhyeong/idrift-data/imagenet/mae_latent_256/ckpt_latest.pt`,
  SHA256 `59c269f99d83645b6c7bb2bf832711aa83d894998259a1ada16c0c9ed7836081`.
- Decoder: existing `stabilityai/sd-vae-ft-mse` snapshot
  `31f26fdeee1355a5c34592e401dd41e45d25a493`; weights SHA256
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`.
- Evaluation RGB: `/data/juhyeong/imagenet/ILSVRC2012/val`.

Preflight checks asset hashes, all 1,000 labels, the full row count, and bitwise
equality of 32 spread-out mmap/source latent rows. Training does not re-encode
images or rescale the cached latents. Evaluation alone decodes generated
latents after dividing by `0.18215`. Its inherited 1,024-image ordered
validation prefix is a small monitoring metric, not ImageNet FID50k. All arms
use the same evaluation setup and preserve training RNG state during evaluation.

## Validate and launch

The submission host needs Python and PyYAML. GPU work uses the existing
`/home/juhyeong/.venvs/replay-drift/bin/python` environment on `srv02`.

```bash
python3 scripts/preflight_corrective_field.py --config-only
python3 -m unittest tests.test_corrective_field_suite
python3 scripts/submit_corrective_field.py
```

The last command prints the plan. After committing the final code and configs,
submit the validation job and all three dependent production jobs:

```bash
python3 scripts/submit_corrective_field.py --submit
```

The helper refuses dirty or untracked source changes. It creates
`runs/corrective-field-submissions/<UTC-id>/source/` with `git archive HEAD`,
records the full commit and SHA256 of every source/config file in
`source-manifest.json`, and removes write permissions from the archive source.
Every job uses this source even if the checkout is edited while waiting in the
queue. `submission.json` records the validation ID and each production job ID.

The validation job runs the actual cached-latent/MAE two-GPU smoke check for all
three arms, including active replay, double drift, GAN updates, and decoding.
Production jobs depend on its successful exit through Slurm `afterok`.
A validation failure leaves the production jobs unable to start; inspect the
validation log before submitting a corrected, freshly committed suite.

The three production jobs request six GPUs in total and start as resources
become available on `srv02`. Each initially creates a unique fresh directory under
`/home/juhyeong/corrective-field-runs/<variant>/job<id>-<suffix>/`; it never
silently resumes a previous arm. Source/config/input provenance is saved in its
`run_metadata/` directory. W&B groups all three runs by their submission ID.
Preflight creates no W&B run; production requires successful online logging.

Five minutes before the three-day allocation limit, Slurm sends `USR1`. The
launcher stops both training ranks, checks the latest completed checkpoint and
replay sidecars, and requeues the same job. A restart must match its original
commit, config checksum, variant, and workdir binding in the submission's
`workdirs/` directory. It resumes model, optimizer, EMA, GAN state where present,
and replay state in the same workdir and W&B run. Missing or mismatched state
fails instead of starting another experiment. Generic training errors are not
automatically requeued.

Up to one 0.5-epoch checkpoint interval may be repeated after a time-limit
restart. The inherited training code does not checkpoint the real memory banks
or exact training RNG state, so resumed training continues the objective but
does not promise bitwise equivalence to one uninterrupted allocation. Before
epoch 10, step-aligned replay-capture sidecars preserve accumulated history;
after epoch 10, the frozen replay banks are reused unchanged.

```bash
squeue -u "$USER" -o '%.18i %.28j %.9T %.10M %.25R'
```

Shared Slurm logs are in the submission directory. A partially failed submission
is recorded explicitly in `submission.json`; inspect its existing job IDs before
rerunning the submit command, which always creates a new suite.

## Preparation validation (2026-09-11)

The CPU regression suite passed 472 tests and 175 subtests; 15 tests requiring
CUDA or explicitly supplied historical source fixtures were skipped. This
covers the nonlinear double-field target and detached gradients, single-step
parity, MAE context/gradient preservation, exact latent loading, GAN integration,
replay capture before and after its freeze boundary, resume, and strict online
logging/evaluation. Python/shell syntax and diff whitespace checks passed.
An actual-data asset preflight on srv02 verified both pretrained weight hashes
and 32 bitwise-identical mmap/source rows. GPU validation is a separate queued
prerequisite; these CPU results do not establish GPU throughput or convergence.

Runtime-only latent loading, decoder, frozen-MAE microbatching, and evaluation
guards were adapted from the existing local `replay-drift` runtime so the assets
remain unchanged. Generator, MAE feature definitions, reverse-field math, and
other experiment presets retain the requested `myjju08/I-Drift` code lineage.
The optional archived-source hash audit now requires `IDRIFT_LIVE_WORKSPACE`
explicitly, avoiding accidental comparisons with an unrelated sibling checkout.
