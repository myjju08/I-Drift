# DINO adversarial experiments

`experiments/dino` packages the DINO experiment configurations and the lossless
I/O used by the September 9–10, 2026 runs. All training imports resolve inside
this checkout. The ImageNet dataset and pretrained DINO checkpoint remain
external assets; they are not copied into Git.

## Reproduced presets

| `--preset` | Generator objective in addition to DINO drift | CNN history | Width | Default I/O | Source status on September 10 |
| --- | --- | --- | --- | --- | --- |
| `dino_only_replay_feature` | CNN feature drift × 1 | Disabled | 32 | Packed native RAM | Running |
| `replay_feature_hardcopy` | CNN feature drift × 1 | Enabled | 32 | Packed native RAM | Earlier experiment |
| `replay_raw_gan_d32` | Conditional raw-image GAN × 0.1 | No CNN drift branch | 32 | Original standard bank/loader | Completed, step 200183 |
| `replay_raw_gan_d128` | Conditional raw-image GAN × 0.1 | No CNN drift branch | 128 | Original standard bank/loader | Completed, step 200183 |

Every preset freezes the pretrained DINO ResNet-50 teacher. DINO features use
stage 3, stage 4, global features and `norm_x`; stage 1/2 weights are zero,
`norm_x` weight is 2, with group normalization. The calibrated stage multipliers
are 1.010352456195719 and 1.0110986372034372. Generator and DINO forwards use
BF16 as configured; the adversarial discriminator retains the shared source
implementation's own arithmetic.

The CNN discriminator is the original conditional multi-scale RGB critic from
`models/adversarial_drift.py`. All four presets train it with logistic real/fake
loss and lazy R1 (`gamma=1`, every 16 steps); structure weight and target EMA
decay are both zero. Its target is an exact copy after each D update. Generator
feature drift reads a frozen snapshot before that update, including real,
negative, generated and, where enabled, historical particles. Only generated
images retain the gradient path into the generator. The feature-drift presets
have no direct generator GAN loss. D learns from detached current fake images;
historical replay changes drift supports, not D's real/fake objective.

The active `dino_only_replay_feature` experiment sets
`adversarial_apply_replay: false`: DINO retains historical particles and replay
weights, while CNN feature drift gets neither historical particles nor replay
weights. Omitting this switch defaults to `true`, preserving the previous
`replay_feature_hardcopy` behavior. `adversarial/replay_enabled` and
`adversarial/history_count` report the CNN branch's actual scope.

The common schedule is two ranks, seed 43 plus rank for G, discriminator seed
43, batch size 4 classes per rank, G32/P64/N32 per class, 256 generated images
per optimizer step globally, and 40 generated epochs. Frozen-snapshot replay
starts at step **50046** (10 generated epochs of 1,281,167 ImageNet images),
uses 16 FP16 historical samples per class and `rho=0.35`. At G32/H16 the
per-current-particle weight is 0.65 and per-history-particle weight is 0.7.
Evaluation and checkpoint cadence is 10 generated epochs; evaluation also runs
at initialization, with 1,024 samples and CFG 1/2/3. G uses AdamW LR 0.0004,
betas 0.9/0.95, 2,000-step warmup and max gradient norm 2. D uses LR 0.0001,
betas 0/0.99, 8 samples per class, D chunks of 8 and G chunks of 16.

## CPU checks and assets

Run these from the repository root with the project Python environment:

```bash
python -m experiments.dino.io.build_native
python -m experiments.dino.train --preset dino_only_replay_feature
python -m unittest discover -s tests -p test_dino_experiments.py -v
```

The default action is CPU preflight, not training. It checks the preset hash,
calibration hash, replay boundary and packed-bank sample/RNG equality. The
native codec builds from the included C source with a local C compiler and
SSE2 on x86; its generated `.so` is ignored by Git. No binary is downloaded.
`--decode-backend numpy` uses the byte-exact NumPy codec when the native build
is unavailable; the implementation and throughput differ, while decoded
samples and RNG behavior remain identical.

Check the actual calibrated assets and both original generator initialization
hashes without initializing CUDA:

```bash
python -m experiments.dino.train \
  --preset dino_only_replay_feature --check-assets --check-generator \
  --imagenet-path /workspace/I-Drift/data/imagenet/raw_ilsvrc2012 \
  --feature-checkpoint /workspace/I-Drift/weights/pretrained/dino_resnet50_pretrain.pth \
  --report /tmp/dino-preflight.json
```

Relative asset paths in packaged presets resolve against `I-Drift-new`,
regardless of the caller's working directory. The explicit paths above reuse
existing local data and weights; they do not import source code from I-Drift.
`--temperature-calibration-artifact` can override the calibration location,
but its original SHA256 must still match. The included calibration JSON is
byte-identical to the source artifact, including historical provenance paths.
ImageNet's manifest content and root are pinned by the source validation code;
moving or rewriting a manifest is not automatically a newly validated dataset.
The asset check validates the pinned manifest and checkpoint, not a fresh hash
of all 1.28 million JPEG payloads. Training also validates ImageFolder counts
and class mappings through the shared trainer.

The production calibration guard requires `use_wandb: true` and the original
project name `Feature encoder - S4 model`. Packaged presets retain the source
W&B entity/name; use a copied YAML with `--config` to select an accessible entity
or a distinct display name. The wrapper requires successful fresh W&B logging;
packaged presets additionally require online mode. It does not resume the
source W&B run IDs.

## Launching a new run

Full training requires an available pair of GPUs, a new workdir, the complete
ImageNet assets and the native build for the default feature presets. On this
Vast instance, run the foreground command below in a supervisor service using
the instance guide's environment/logging wrapper. Choose available GPUs and a
free rendezvous port when registering that service; the currently running
experiment's GPU allocation is not embedded in this port.

```bash
torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29928 --max_restarts=0 \
  --module experiments.dino.train \
  --preset dino_only_replay_feature --train \
  --workdir runs/dino_only_replay_feature_new \
  --imagenet-path /workspace/I-Drift/data/imagenet/raw_ilsvrc2012 \
  --feature-checkpoint /workspace/I-Drift/weights/pretrained/dino_resnet50_pretrain.pth
```

Set `CUDA_VISIBLE_DEVICES` in that service to its assigned devices. Optional
`IDRIFT_RANK_CPUSETS` is a JSON list of two valid CPU-index lists. The wrapper
records the effective config, code root, generator hashes, process identity,
first-step checks and replay-boundary checks inside the new workdir. Existing
workdirs are rejected. This entry starts from scratch; copying a checkpoint or
old W&B ID into a workdir is not an exact-state continuation facility.

The feature presets retain the live packed positive bank: 1000 classes × 128
slots, lossless color/XY Zstandard level 1, 8 codec workers, codec batch 128,
13 GiB arena plus index budget per rank, no disk spill, whole-file PIL reads,
packed ImageFolder metadata and runtime train prefetch factor 1. The YAML's
prefetch factor remains the historical value 2, as in the live run; the wrapper
records the effective override. Negative and frozen replay banks keep their
original storage. Arena capacity excludes model/optimizer memory, decoded
batches, codec workspaces and replay banks; it is not the full job RAM budget.

Raw GAN presets use the source raw launcher's original loader/bank route.
`--io-backend packed` or `--io-backend standard` deliberately overrides the
preset's runtime choice and is recorded. The bank/loader parity tests cover
pixel values and RNG behavior, not throughput equivalence across routes.

The shared `train_imagenet_gen.py` also supports raw GAN, feature drift and
mixed objectives through the original `configs/gen/*raw-conditional-gan*.yaml`,
`*adversarial-feature-drift*.yaml` and `*mixed-gan-feature-drift*.yaml`
configs. Pass a chosen or edited YAML to
`python -m experiments.dino.train --config <file>` for CPU preflight, or use
the same two-rank command with `--config <file> --train --workdir <new-dir>`.
Custom configs use standard I/O by default and skip the bundled preset's
fixed-shape/initialization assertions; core loss and calibration checks remain
active. `--config` is not a claim that custom settings reproduce a named run.

## Verification and provenance

`experiments/dino/provenance.json` records source file paths/hashes, exact
calibration hash and the three relocated config fields. Its paths describe
where the implementation came from; they are never imported by the runtime.
`experiments/dino/source_verification.json` records comparison with the actual
active DINO effective config, prior hard-copy effective config, and completed
raw D32/D128 W&B configs and final checkpoint configs. After restoring only
the three asset locations for comparison, **every expected field matches**:
150 for active DINO-only replay, and 149 for each other preset. Both completed
raw checkpoints are at step 200183; no late config overrides were found.

CPU checks cover native/NumPy codec equality, ring overwrite, empty classes,
sample/RNG identity, JPEG reader equality, path/label order and timing wrappers.
The replay test executes the real generator training branch and verifies that
DINO history/weights remain active while CNN history/weights are independently
disabled or enabled, generated-image gradients remain present, DINO stays
frozen and the post-update critic target is an exact copy. The broader core
tests check GAN losses, lazy R1, mixed objectives and source/live function parity.

These checks and calibrated-asset preflight ran on CPU. They establish source
configuration and tested arithmetic parity; a fresh multi-GPU full training
trajectory and FID reproduction have not been run as part of this migration.
