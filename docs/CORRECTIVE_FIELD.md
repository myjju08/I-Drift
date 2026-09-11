# S4 / MAE-256 Corrective Field experiments

The current deployment uses all three arms on **srv06 / six A5000 GPUs**.
See [SRV06_EXPERIMENTS.md](SRV06_EXPERIMENTS.md) for its copied assets and
Slurm command. The srv02 profile below remains available with the same
scientific settings.

## Source and objective

Replay policies, Double Drift, and attraction/repulsion balance are ported from
[`cosmosjhj/I-Drift`, commit `27a0e4fb54c8ed04e2538831a9df86a85c8c25ef`](https://github.com/cosmosjhj/I-Drift/commit/27a0e4fb54c8ed04e2538831a9df86a85c8c25ef).
Its source files are `drifting_core/double_drift.py`,
`drifting_core/imagenet_loss.py`, and the integration in
`train_imagenet_gen.py`. Existing DINO and MAE experiment entry points remain
available with their own presets.

The three new arms use the root trainer and a common frozen MAE-256:

| Arm | Replay | Double Drift | Adversarial objective |
| --- | --- | --- | --- |
| `baseline` | Disabled | Disabled | Disabled |
| `replay_double` | `rho=0.35`, `H=16`, freeze at generated epoch 10 | Feature, `(c0,c1)=(1,1)` | Disabled |
| `replay_only` | Same | Disabled | Disabled |

Comparing `replay_only` with `baseline` measures replay's contribution.
Comparing `replay_double` with `replay_only` measures the additional Double
Drift contribution. GAN supervision is disabled in all three arms.
These S4 runs have no completed result claims at setup time; earlier B/4
metrics in the main README and upstream reports use different protocols.

## Double Drift definition

The selected method operates on the normalized features of the frozen MAE.
For input scale `s`, write `u=x/s`. The reverse field `V` retains the original
per-temperature RMS normalization. The target is

```text
u1 = stopgrad(u + c0 * V(u))
u2 = stopgrad(u1 + c1 * V(u1))
loss = mean((u - u2)^2)
```

The production settings use `c0=c1=1`, matching
`u + V(u) + V(u + V(u))`. The second evaluation rebuilds both the generated
queries and generated negative particles at the moved features. Real
positive/negative features, replay particles, labels, CFG weights, and the
first distance scale stay fixed. The second field has its own force RMS
normalization; the summed displacement has no additional normalization.
Targets are detached, so training does not differentiate through field
construction. `double_drift_mode: off` keeps ordinary single drift.

The upstream sample-space method is also available for separate studies:

```text
g0 = dL_drift(x0)/dx0
a = sample_step_rms / RMS(g0)
x1 = stopgrad(x0 - c0*a*g0)
g1 = dL_drift(x1)/dx1
x2 = stopgrad(x0 - a*(c0*g0 + c1*g1))
L = sum((x0-x2)^2) / (2*a)
```

Its sample gradient is `c0*g0+c1*g1`, and the first probe distance is
`c0*sample_step_rms`. The upstream reported `(0.75,0.25)` sample-space
experiments are distinct from this requested feature-space `(1,1)` suite.

The enabled trainer modes default to coefficients `(0.75,0.25)` and sample
probe RMS `0.1`; the suite explicitly overrides the feature coefficients to
`(1,1)`. Both Double Drift modes currently require reverse drift without
learned feature adapters or GAN branches. Unsupported combinations raise an
error instead of silently changing the objective.

## Replay updates

Each rank collects detached class-conditioned historical latents during the
first ten generated-sample epochs and freezes its bank at that boundary.
After activation, current generated repulsion has weight `1-rho=0.65`;
historical particles have weight `rho*G/H=0.7`, with `G=32`, `H=16`.
The same replay references and weights are used in both field evaluations.
MAE parameters remain frozen. The three suite configs explicitly use
`historical_gen_replay_policy: frozen` and `rev_drift_balance_delta: 0.0`.
The optional rolling policies and force schedule are documented in
[the upstream comparison](UPSTREAM_PARITY.md) and the main README.

All three arms set `feature_gan: false` and `adversarial_mode: none`.
No discriminator is built or updated by this suite.

## Matched training settings

| Setting | Value |
| --- | --- |
| Generator | S/4, 4×32×32 latent input, patch 4, hidden/condition 384, depth 12, heads 6 |
| Initialization and seed | Scratch, seed 43 |
| Hardware per arm | 2 RTX 3090 GPUs on `srv02` |
| Generated batch | 8 labels/rank × 32 samples/label × 2 ranks = 512 |
| Targets per label | 32 positive, 32 real negative |
| Duration | 40 generated-sample epochs = 100,092 updates |
| Epoch denominator | 1,281,168 cached samples |
| Replay activation | Update boundary `ceil(10*1281168/512)=25023` |
| Adam | LR 0.0004, betas (0.9,0.95), weight decay 0, warmup 2,000 |
| Generator EMA / gradient clipping | 0.999 / 2.0 |
| Feature objective | `no_stage12_norm_x2`, same weights in every arm |
| Training evaluation | Step 0 and every 10 epochs, 1,024 images, CFG 1/2/3 |
| Checkpoints | Every 0.5 generated-sample epochs |
| Logging | Online W&B project `Corrective Field` |

The FP32 cache is read unchanged; feature microbatching controls activation
memory. The frozen encoder runtime is checked for forward and input-gradient
parity and does not train or replace the MAE weights.
During-training evaluation retains the existing ordered 1,024-image RGB
validation prefix. These monitoring metrics are not ImageNet FID50k and must
not be compared directly with the upstream 50,000-sample result tables.

## Existing assets on srv02

Assets are supplied separately and are not committed to Git.

| Asset | Path |
| --- | --- |
| Exact FP32 latent mmap | `/home/juhyeong/replay-drift-cache/imagenet-latents-f32-v1` |
| Original flat latent source | `/data/juhyeong/imagenet/imagenet_latents` |
| Frozen MAE-256 | `/home/juhyeong/idrift-data/imagenet/mae_latent_256/ckpt_latest.pt` |
| ImageNet RGB | `/data/juhyeong/imagenet/ILSVRC2012` |
| SD-VAE-MSE decoder | `/home/juhyeong/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493` |

MAE checkpoint SHA-256:
`59c269f99d83645b6c7bb2bf832711aa83d894998259a1ada16c0c9ed7836081`.
The mmap contains FP32 `[1281168,4,32,32]` latents and integer labels.
The latent scaling factor `0.18215` is used for VAE decoding during image
evaluation; training values are not rescaled or regenerated.

## Launch and validation

Run from this repository with its Python dependencies installed:

```bash
python scripts/preflight_corrective_field.py --config-only
python scripts/submit_corrective_field.py
```

The second command prints the submission plan. On a Slurm host, after the
final source is committed, submit the checked suite:

```bash
python scripts/submit_corrective_field.py --submit --variants replay_only replay_double
```

This selection queues the two requested replay experiments and leaves the
baseline unsubmitted. Omit `--variants` to explicitly select the full three-arm
suite. Validation always covers all three configurations.

The default worker Python is
`/home/juhyeong/.venvs/replay-drift/bin/python`. Override `--python`,
`--run-root`, or `--snapshot-root` for another installation; update all three
asset paths/configurations together when moving the experiment.

Submission writes a read-only Git archive under
`runs/corrective-field-submissions/<UTC-uuid>/source`, with per-file SHA-256
hashes and a submission record. It queues a bounded two-GPU validation job
and the selected production jobs with `afterok` dependencies. Production also
checks the successful validation report's exact source and config hashes.
Each arm receives two Slurm-assigned RTX 3090 GPUs, six CPUs, 80 GB RAM,
and a three-day time limit on `srv02`. The selected pair uses four GPUs;
the full suite uses six GPUs in total;
validation finishes before any production arm starts.

The GPU gate runs six full-batch updates per arm, activates an `H=16` frozen
replay bank after two updates for the replay arms, checks finite gradients,
agreement between ranks, identical initial generator weights, unchanged MAE
weights, and finite VAE-decoded images. It does not create W&B training runs.
The production logger requires online W&B; authentication or initialization
failure stops the job instead of falling back to unlogged training.

New work directories are
`/home/juhyeong/corrective-field-runs/<variant>/job<ID>-<suffix>`.
They contain `train_log.jsonl`, `wandb_run_id.txt`, checkpoints, evaluation
artifacts, and `run_metadata/`. The immutable submission directory contains
`submission.json`, Slurm logs, and the GPU validation result when completed.

Five minutes before a time limit, the job stops its worker process group,
validates its last completed checkpoint and matching replay state, and
requeues the same job/workdir/W&B identity. Up to half an epoch can repeat.
Generator/EMA/optimizer and historical replay state are restored. Replay
state and usage telemetry are saved per completed checkpoint and per rank,
using atomic publication before the matching model checkpoint. Older frozen
checkpoints may use their immutable epoch-boundary bank; rolling policies
require the exact step sidecar. As in the
existing trainer, real-data banks are rebuilt and all sampler/RNG state is
not checkpointed, so this is not a bitwise trajectory-resume guarantee.

## Port verification

The independent [numerical parity report](DOUBLE_DRIFT_PARITY.json) records
360 cases and 25,080 exact scalar/tensor comparisons against the upstream
commit, with maximum absolute difference zero. Cases cover loss, fields,
input gradients, weighted multi-feature aggregation, sample-space probing,
replay off/`rho=0.35`, coefficients `(1,0)`, `(1,1)`, `(0.75,0.25)`, and five
static/annealed attraction-repulsion settings including nonunit balance.
The production feature dtype is FP32; outer sample graphs also cover FP64.
[Replay state parity](REPLAY_PARITY.json) covers all four memory policies,
FP16/FP32 storage, isolated/global host RNG, and midpoint snapshot restoration
with 9,184 exact comparisons. The current CPU regression result and remaining
source differences are recorded in [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md).
These CPU results do not replace the queued two-GPU validation gate.

The existing srv02 asset preflight verified 32 original FP32 cache rows
bitwise, MAE/VAE weight hashes, and authenticated access to `Corrective Field`
without creating a training run.

To repeat the comparison with a checkout of the referenced upstream commit:

```bash
OMP_NUM_THREADS=1 python scripts/verify_double_drift_port.py \
  --source /path/to/cosmosjhj-I-Drift-at-27a0e4f
```

Portable CPU regression tests run with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
```

Tests that inspect the historical external DINO snapshot require an explicit
`IDRIFT_LIVE_WORKSPACE`; an unrelated sibling folder named `I-Drift` is not
treated as that reference. Numerical behavior tests run without the snapshot.
GPU-specific checks remain pending until the scheduled validation executes.
