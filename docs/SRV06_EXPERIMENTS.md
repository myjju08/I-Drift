# Corrective Field on srv06

The three S4 experiments use six RTX A5000 GPUs, two per experiment, through
Slurm on `srv06`. The scientific configurations match the original srv02
suite: 40 generated-sample epochs, seed 43, global generated batch 512,
frozen MAE-256, unchanged FP32 latents, and no GAN loss.

| Arm | Replay | Double Drift |
| --- | --- | --- |
| `baseline` | Off | Off |
| `replay_only` | rho 0.35, frozen bank at epoch 10 | Off |
| `replay_double` | rho 0.35, frozen bank at epoch 10 | Feature `(c0,c1)=(1,1)` |

Attraction/repulsion balance stays at delta 0 in all arms. The DINO/GAN
implementations and optional balance/Replay policies remain available in the
repository; they are not enabled in this comparison.

## Assets and runtime

Node-local assets are under `/data/juhyeong/corrective-field-assets` and the
worker interpreter is `/data/juhyeong/venvs/replay-drift/bin/python`.
The copied runtime uses the same package versions as srv02. Training requires
no RGB training-set copy because inputs come from the existing latent cache;
RGB validation images remain necessary for the same monitoring protocol.

The MAE checkpoint and local SD-VAE-MSE decoder are copied unchanged. The
latent tensor and label payloads retain their recorded SHA-256 hashes; no
VAE encoding, augmentation, scaling, precision conversion, or row reordering
is performed during migration. Migration evidence stays with local assets.

The copied `metadata.json` is unchanged. `relocation.json` records full checksums
of `latents.npy`, `labels.npy`, and `source_stats.npy`, binds them to their local
file identities, and points to 32 authentic original source rows for startup
bitwise checks. The source-evidence directory contains only those audit rows;
all 1,281,168 training examples are read from the complete copied mmap. This
avoids copying millions of redundant small files. The original cache retains
its strict source-directory validation when no relocation certificate exists.

After copying unchanged payloads and audit rows with nanosecond mtimes:

```bash
python scripts/certify_latent_cache_relocation.py \
  --cache /data/juhyeong/corrective-field-assets/imagenet-latents-f32-v1 \
  --source-evidence /data/juhyeong/corrective-field-assets/imagenet_latents
```

Certification reads every payload byte and rejects any mismatch with the
original manifest. Subsequent loader starts validate the certificate, local
file identities, small inventory hashes, and original source-row metadata;
production preflight also compares those source rows bitwise.

Run outputs go to `/data/juhyeong/corrective-field-runs`. W&B project remains
`Corrective Field`, with one fresh run per production arm.

## Slurm submission

From a clean committed checkout, run:

```bash
python scripts/submit_corrective_field.py \
  --node srv06 --suite-dir configs/corrective_field_srv06 \
  --python /data/juhyeong/venvs/replay-drift/bin/python \
  --run-root /data/juhyeong/corrective-field-runs \
  --variants baseline replay_only replay_double --submit
```

The immutable source manifest binds the selected node, GPU type and suite.
Slurm requests two A5000 GPUs, six CPUs and 80 GiB RAM per production arm.
A two-GPU validation job first checks all three arms using the actual copied
MAE/cache/decoder, before the three dependent production jobs may start.
Validation includes finite losses/gradients, rank agreement, unchanged MAE
weights, active Replay, and the requested Double Drift coefficients.

The original srv02 queued jobs are held during migration and superseded once
the srv06 submission is verified. An idle GPU shown by `nvidia-smi` alone
does not establish Slurm availability; both CPU and GPU allocations must fit.

See [CORRECTIVE_FIELD.md](CORRECTIVE_FIELD.md) for the equations, exact
hyperparameters, checkpoint/requeue behavior, and evaluation limitations.

## Migration verification

The final CPU regression suite passed 583 tests and 179 subtests, with 15
optional CUDA/external-reference skips and one known upstream expected failure.
The srv06 and srv02 scientific settings compare equal; only the five local
asset paths differ. The relocation tests cover a complete 67-row mmap with
only 32 source audit rows, full bit/label preservation, sampling order, and
checksum/identity/provenance corruption rejection.

A bounded Slurm job on srv06 successfully allocated two A5000 GPUs and ran
finite CUDA matrix operations using PyTorch 2.8.0+cu128. The installed local
runtime matches all 96 packages in the srv02 environment and passes CPU
imports, backward execution and `pip check`. W&B project access was checked
without creating a training run. Full model validation is performed by the
mandatory gate in each production submission.
