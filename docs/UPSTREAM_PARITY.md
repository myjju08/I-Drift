# Upstream Replay, Double Drift, and force-balance port

Source: [`cosmosjhj/I-Drift` at `27a0e4fb54c8ed04e2538831a9df86a85c8c25ef`](https://github.com/cosmosjhj/I-Drift/commit/27a0e4fb54c8ed04e2538831a9df86a85c8c25ef).
Target base: `myjju08/I-Drift` at `4c26125a6acdd1cff488b6904025996bbb6b8dc5`.
Both remote main branches were fetched and checked on 2026-09-11.
The target file hashes in [the complete inventory](UPSTREAM_PARITY.json)
identify the final port contents independently of the commit containing this report.

## Result and scope

The requested Replay, Double Drift, and attraction/repulsion implementation
is integrated into the root training code. **The entire repositories are not
identical after excluding DINO/GAN.** Target runtime additions, evaluation
behavior, documentation, and upstream hardware/ablation workflows also differ.
The inventory records every tracked upstream path and every target source
path, including SHA-256 hashes and changed Python definitions; this report and
the inventory itself are excluded to avoid self-referential hashes.

The pre-port comparison had 94 identical files, 19 differing shared files,
59 upstream-only files, and 131 target-only files. The final inventory has
99 identical files, 17 differing shared files, 56 upstream-only files, and
136 target-only files, excluding the two self-report files described above.
File counts measure source identity, not numerical equivalence.

## Integrated source behavior

| Component | What matches upstream |
| --- | --- |
| `drifting_core/force_balance.py` | File copied byte-for-byte; finite `-1 < delta < 1`, coefficients `1+delta` / `1-delta`, optional linear annealing using absolute training step |
| `drifting_core/double_drift.py` | File remains byte-identical; detached feature-field and sample-gradient two-step methods |
| `drifting_core/imagenet_loss.py` | Independent force coefficients applied after positive/negative mass coupling, before each temperature's RMS; both Double Drift stages and source balance diagnostics |
| `memory_bank.py` | `HistoricalReplayMemoryBank` class copied verbatim, including frozen/FIFO/reservoir/usage-budget policies, sampling, replacement, telemetry and NPZ state |
| `train_imagenet_gen.py` | Source balance schedule, Replay configuration validation, cadence, step/rank-isolated update RNG, candidates from the existing generator forward, and CLI overrides |
| `configs/gen/B4_rev-drift_mae256.yaml` and its launch script | Both files now byte-identical to the pinned upstream; optional Double Drift/Replay controls and portable torchrun discovery |
| Data / VAE / official evaluation | Source split-aware flat-cache resolver, mse/ema variant compatibility, environment precedence for dataset/cache paths, flat validation labels, optional within-class diversity report |

For nonunit balance, the root trainer requires reverse drift. Double Drift
continues to reject learned feature adapters and GAN/adversarial systems;
these combinations are not newly supported by this port. Existing DINO/GAN
entry points and the target raw/compressed bank classes are preserved.

The S4 suite explicitly keeps `rho=0.35`, frozen Replay at generated epoch 10,
`delta=0`, and feature Double Drift `(c0,c1)=(1,1)` in the double arm only.
No GAN term is enabled. See [the README controls](../README.md#replay-and-attractionrepulsion-controls)
and [the experiment guide](CORRECTIVE_FIELD.md).

## Remaining differences beyond DINO/GAN

- The target retains exact FP32 mmap loading, local VAE decoding, frozen-MAE
  microbatching, optional SDPA, and diagnostic/normalization optimizations.
  The untouched default force path was compared with the target base in 48
  combinations: loss, gradients and diagnostics were bitwise identical.
- Target MAE exports retain optional stage pruning, `with_norm_x`, and
  terminal-block controls. These options must match when comparing feature
  objectives; arbitrary presets are not claimed numerically equivalent.
- Replay bank and usage telemetry are published atomically per checkpoint
  and rank before the model checkpoint. Unlike upstream's single overwritten
  latest-bank file, this permits selecting an older checkpoint with its own
  bank state. Legacy frozen-bank fallback remains supported. Full training
  trajectories are not guaranteed identical after resume because real-data
  bank and all sampler/RNG state are not checkpointed.
- Target raw validation uses a deterministic center crop; upstream's raw
  validation transform can randomly flip images. The matched S4 monitoring
  uses the existing 1,024-image validation prefix, not upstream FID50k.
- The 56 upstream-only paths are 51 launch/benchmark/asset/Slurm workflows
  and five tests for those workflows. They include srv08/A6000-specific
  ablation pipelines. They were audited but not copied or scheduled on
  srv02. Their Replay/Double Drift/force-balance mechanisms are exposed by
  the integrated trainer. The target keeps its own srv02 Slurm submission,
  two-GPU validation gate, and immutable source snapshots.
- Documentation, dependencies, extended encoder/adapter APIs, experiment
  presets and their regression tests also remain different. The JSON
  inventory lists these without treating all target additions as GAN changes.

## Executed verification

- [Field, feature and sample parity](DOUBLE_DRIFT_PARITY.json): **360 cases,
  25,080 exact comparisons, maximum absolute error 0**. Includes Replay
  off/on, coefficients `(1,0)`, `(1,1)`, `(.75,.25)`, positive/negative delta,
  and annealing at midpoint/end. Features are FP32; outer sample graphs cover
  FP32 and FP64. Global-stat flags are exercised on CPU with world size one.
- [Replay state parity](REPLAY_PARITY.json): **16 cases, 9,184 exact
  comparisons, maximum error 0**. All four policies, FP16/FP32 storage,
  global/isolated NumPy RNG and midstream save/restore are checked.
- Actual `train_step` schedule tests cover all three modes with/without
  Replay at start/midpoint/end/after-end. Scheduled and constant effective
  coefficients match in loss, gradient, and optimizer update.
- Actual `train_gen` checkpoint/resume tests cover every Replay policy,
  retaining newer sidecars while resuming an older model checkpoint; future
  replay samples and final bank/telemetry state match an uninterrupted run.
- Final CPU regression: **569 passed, 179 subtests passed, 15 skipped,
  1 expected failure** (40.20 seconds). Skips require CUDA or unavailable
  external reference assets. The strict expected failure reproduces the
  upstream compact-top-k translation test;
  the dense force path used by this S4 suite passes.
- The two-GPU GPU validation gate remains required before production. CPU
  parity establishes the covered numerical paths, not GPU throughput or
  final image-quality reproduction.

Reproduce from a source checkout containing the pinned commit:

```bash
OMP_NUM_THREADS=1 python scripts/verify_double_drift_port.py --source /path/to/cosmos-I-Drift
OMP_NUM_THREADS=1 python scripts/verify_replay_port.py --source /path/to/cosmos-I-Drift
python scripts/audit_upstream_port.py --source /path/to/cosmos-I-Drift --source-ref 27a0e4f
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests -q
```
