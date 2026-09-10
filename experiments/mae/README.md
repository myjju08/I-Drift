# MAE256 control, GAN loss and adversarial feature drift

These presets reproduce the **currently running spatial latent MAE experiments
observed on 2026-09-10**, including their same-cache MAE-only control. The training
implementation is contained in this repository. The source was
`mae256_gan_prepare_20260910/latent_spatial`; file hashes and asset identities are
recorded in [provenance.json](provenance.json).

| Preset | Generator objective | Discriminator input |
|---|---|---|
| [control.yaml](configs/control.yaml) | Frozen MAE reverse drift | No discriminator |
| [raw_gan.yaml](configs/raw_gan.yaml) | Frozen MAE reverse drift + `0.15 × mean(softplus(-D_target(fake, class)))` | Scaled 4×32×32 VAE latent |
| [feature_drift.yaml](configs/feature_drift.yaml) | Frozen MAE reverse drift + `0.1 × CNN feature drift` | Scaled 4×32×32 VAE latent |

Feature drift uses the frozen discriminator's stage2/3/4 features as additional
drift coordinates. Its generator has **no scalar GAN loss**. The raw GAN generator
has **no CNN feature drift**. Both retain their original frozen MAE objective.
This adversarial system is separate from the older `feature_gan: true` MAE
feature-space hinge-GAN experiment; that option stays disabled here.

## Discriminator and update order

`latent_spatial_844` adapts the existing conditional D32 CNN to small latent maps:
the stem has no average pooling, and the four blocks' first convolution strides
are `[2, 2, 2, 1]`. Native stage2/3/4 maps are respectively 8×8, 4×4 and 4×4, with
64, 128 and 256 channels. Each map is average-pooled to 4×4 and flattened in spatial
order into 16 tokens, then normalized over channels with
`F.normalize(..., eps=1e-8) * sqrt(channels)`. There is no spatial up-pooling.
The three conditional projection-head logits are averaged before logistic loss.

The geometry change preserves every initialized parameter and optimizer parameter
object. Seeded D initialization runs inside an RNG fork, so constructing D does
not change generator or sampling RNG state. The discriminator uses float32
unclipped scaled latents directly. R1 differentiates those latent coordinates;
there is no GAN-time VAE decode or RGB conversion.

For each iteration, G first backpropagates through the frozen target D and updates
its parameters. D then receives detached samples from that same G forward: the
first 8 real and 8 generated samples per selected class, one D update per G step.
Both modes use the same objective and optimizer:

```text
D loss = mean(softplus(-D(real, class))) + mean(softplus(D(fake, class)))
       + 0.5 × gamma × interval × R1                 [on R1 steps only]
R1     = mean(sum(gradient_real(D(real, class)) ** 2))
Adam   = lr 1e-4, betas (0, 0.99)
gamma  = 1; interval = 16; R1 applies at steps 0, 16, 32, ...
```

Structure preservation has weight zero. Shared trainer gates avoid requesting
DINO terminal maps when that weight is zero, so MAE feature drift uses the normal
trainer without the live run's `inspect`/`exec` patch. The target is an exact hard
copy after each successful D update (`adversarial_ema_decay: 0`), frozen for the
next G step. Replay, feature adapters and mixed supervision are disabled.

Checkpoint state includes the latent coordinate and spatial geometry identity.
It accepts matching live `latent_spatial_844` states and rejects the older RGB or
coarse-grid direct-latent checkpoints, even though the coarse model has compatible
parameter tensor shapes. The contained `latent_direct_gan.py` is the shared base
for the spatial variant and the geometry regression test, not the current preset.

## Matched settings and comparison limits

All three presets retain the live S/4 generator (hidden size 384, depth 12, six
heads, patch size 4), seed 43, G Adam `4e-4/(0.9,0.95)`, 2,000-step warmup, G EMA
0.999 and gradient clipping 2.0. They use G/P/N = 32/64/32 and four selected labels
per rank. The live runs have **two ranks**, hence 256 generated samples per step;
40 generated ImageNet epochs correspond to 200,183 steps. Changing world size
changes this comparison even if the YAML remains the same.

The frozen MAE uses stage3/4 activations, mean/std patch sizes 2 and 4, global
features and `norm_x`, `every_k_block: 2`, and includes terminal blocks. The MAE
feature loss keeps `norm_x` weight 2, skips stages1/2, normalizes feature groups,
uses global scale/fnorm statistics and the `mae_scale_matched` temperature profile.
Reverse drift temperatures are `[0.2, 0.05, 0.02]`; the exponential kernel,
bank capacities/sampling and throughput optimization level 3 match the live
control. The complete settings are in the YAMLs rather than inferred defaults.

Evaluation occurs at epoch 0 and every 10 generated epochs, with 1,024 samples at
CFG 1, 2 and 3. This is the existing **class-limited quick evaluation, not
FID50k**. Compare checkpoints at matching completed updates/generated epochs and
CFG. The latest spatial experiment changed the geometry and auxiliary
coefficients together: older raw/feature weights were 1.5/1.0, now 0.15/0.1.
The current experiment does not isolate architecture from coefficient effects,
and these coefficients are not claimed to match gradient magnitudes.

The control retains two inactive `latent_rgb_*` metadata fields from its original
launcher. With `adversarial_mode: none`, neither affects training or evaluation.

## Assets and execution

Run commands from the repository root. Relative paths inside these presets are
resolved against that root, independently of the current working directory.
Model weights and ImageNet/cache data are external assets, not copied into Git.

| Environment override | Default location relative to repository |
|---|---|
| `IDRIFT_IMAGENET_PATH` | `data/imagenet/raw_ilsvrc2012` |
| `IDRIFT_MAE_CACHE` | `data/imagenet/latent_cache_256_mae_gan_20260910` |
| `IDRIFT_MAE_CHECKPOINT` | `artifacts/mae_latent_256/ckpt_ema.pt` |
| `IDRIFT_MAE_VAE` | `artifacts/sd-vae-ft-mse` |

To explicitly reuse this instance's current immutable assets:

```bash
export IDRIFT_IMAGENET_PATH=/workspace/I-Drift/data/imagenet/raw_ilsvrc2012
export IDRIFT_MAE_CACHE=/workspace/I-Drift/data/imagenet/latent_cache_256_mae_gan_20260910
export IDRIFT_MAE_CHECKPOINT=/workspace/mae256_gan_prepare_20260910/artifacts/mae_latent_256/ckpt_ema.pt
export IDRIFT_MAE_VAE=/workspace/mae256_gan_prepare_20260910/artifacts/sd-vae-ft-mse
python -m experiments.mae.runtime \
  --config experiments/mae/configs/raw_gan.yaml --preflight --verify-asset-hashes
```

Preflight reads only files on CPU. It checks complete cache coverage and metadata,
asset availability, and, with `--verify-asset-hashes`, the live frozen MAE and VAE
hashes. It does not create a CUDA context, start training, or write a workdir.
Explicit reuse keeps the original asset folders necessary; move the assets and
change these variables before removing those folders.

For training after selecting **two available GPUs**, use a fresh workdir and run
the following foreground command through your normal supervisor/job service:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=online HF_HUB_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Set CUDA_VISIBLE_DEVICES to the two allocated GPU indices before launch.
torchrun --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  --module experiments.mae.runtime \
  --config experiments/mae/configs/raw_gan.yaml \
  --workdir runs/mae256_spatial_raw_gan_reproduction
```

Substitute `feature_drift.yaml` or `control.yaml` and a different workdir for the
other runs. The common trainer's flags (including `--max_steps`) remain available.
These presets retain the current W&B project/entity/name; change the logging
section for another account. `use_wandb: false` explicitly disables logging;
enabled W&B initialization failure is fatal, as in the live runtime. Optional
`MAE256_RANK_CPUSETS='[[...],[...]]'` pins each rank within its allowed CPU set;
there is no machine-specific default pinning.

No source service was replaced or restarted by this port. A runnable command is
provided for a future allocation; the ongoing training continues to use its
original code and workdirs.

## Packed cache format

[latent_cache.py](latent_cache.py) contains the original complete preparation,
resumable construction, finalization and loader implementation. Cache tensors
are float32 `[images, 2, 4, 32, 32]`: separate VAE posterior samples for clean and
horizontally flipped RGB views, scaled by 0.18215. The loader chooses one view
with the original `torch.rand(1) < 0.5` draw for **both train and validation**.
It does not resample the posterior at training time, even when `use_aug: false`.
The class ordering, distributed sampler, worker seeding, collation and decoder
postprocess are preserved. Only evaluation/image output lazily loads the VAE.

The full cache includes 1,281,167 training images and 50,000 validation images.
The source data are ImageFolder `train/` and `val/` trees with matching 1,000-class
mappings. Commands, if creating a new cache rather than reusing the live one:

```bash
python -m experiments.mae.latent_cache prepare \
  --raw-root "$IDRIFT_IMAGENET_PATH" --cache-root "$IDRIFT_MAE_CACHE"
# GPU operation: run as a managed foreground job on an allocated GPU.
python -m experiments.mae.latent_cache build \
  --cache-root "$IDRIFT_MAE_CACHE" --vae-path "$IDRIFT_MAE_VAE" \
  --device cuda:0 --batch-size 32 --seed 43
python -m experiments.mae.latent_cache finalize --cache-root "$IDRIFT_MAE_CACHE"
python -m experiments.mae.latent_cache verify --cache-root "$IDRIFT_MAE_CACHE"
```

The VAE is `stabilityai/sd-vae-ft-mse` revision
`31f26fdeee1355a5c34592e401dd41e45d25a493`. The frozen MAE checkpoint was converted
from `Goodeat/drifting` revision `1a5afa9fc22c3beefe06e4689e84d3e6500b5983`,
`models/mae/jax/mae_latent_256`. The cache recipe also pins float32, TF32 cuDNN on,
TF32 matmul off, cuDNN benchmark off, fixed encoding batch size and the per-chunk
CUDA RNG recipe. Reusing the verified live cache gives stronger reproduction
than regenerating it on different hardware/software.

`finalize` hashes complete chunk contents. Routine `verify`/loader startup checks
coverage, index/recipe/marker hashes and chunk sizes, without rehashing every
41 GiB of cached data. Re-run `finalize` for full content verification.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_mae_experiments.py -v
```

The CPU suite checks spatial geometry and token normalization, parameter/RNG
identity, logistic/lazy-R1 math, gradients to G with a frozen target and no VAE,
identical D updates for both modes, optimizer/target checkpoint roundtrips and
rejection of incompatible checkpoints. Cache tests compare clean/flip draws,
RNG state, distributed sampler and loader batches with the original `.pt` cache
loader, and reject incomplete/truncated/changed chunks.

Setting `IDRIFT_MAE_REFERENCE_ROOT` to the original preparation directory enables
additional optional tests that import its literal live modules and compare
forward maps, features/logits, generator input gradients, D updates at steps
0/1/16, optimizer and target state, and checkpoint transfer with zero numerical
tolerance. A complete small CPU G+D step also compares the live patched trainer
with the local runtime for all three modes: losses, updates, optimizer states,
returned extras and final RNG state match exactly. The integrated trainer adds
two diagnostics, `adversarial/replay_enabled=1` (the auxiliary branch's default
eligibility switch) and `adversarial/history_count=0`; the zero count confirms
that these MAE runs use no Replay particles. The test asserts these added values
before comparing all other metrics. Every scientific config field is compared, with only explicit
path relocation excluded. These tests are optional so the repository's normal
tests do not depend on that external directory. Verification evidence and its
CPU-only scope are recorded in [port_verification.json](port_verification.json).
