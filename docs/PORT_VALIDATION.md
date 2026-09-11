# I-Drift implementation port — 2026-09-10

This is the historical DINO/GAN port report. The subsequent 2026-09-11
Replay, Double Drift, force-balance changes and current source comparison
are documented in [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md).

The port targets the implementation and effective configurations on the
running instance, including runtime code that was outside the original
repository. It does not copy datasets, model weights, run directories or
W&B identities into Git.

## Source and intentional changes

| Component | Source | Port |
| --- | --- | --- |
| Trainer, losses, encoders, generator, memory banks, data loading | `I-Drift/` | Repository root, `models/`, `drifting_core/`, `train/` |
| DINO-only replay branch | `idrift2_transfer_20260908/dino_only_replay_feature_drift_20260909/trainer_dino_only_replay.py` | Shared trainer `adversarial_apply_replay` gate |
| DINO packed RAM and JPEG input path | `idrift2_transfer_20260908/io_optimization/` | `experiments/dino/runtime.py`, `experiments/dino/io/` |
| DINO replay presets | Active feature experiment, earlier hard-copy feature experiment, completed raw GAN D32/D128 pair | `experiments/dino/configs/` |
| MAE packed cache and latent discriminator | `mae256_gan_prepare_20260910/latent_cache.py`, `latent_direct_gan.py`, `latent_spatial_gan.py` | `experiments/mae/` |
| MAE active configurations | `configs/control.yaml`, `latent_spatial/configs/{raw_gan,feature_drift}.yaml` | `experiments/mae/configs/` |

The shared trainer starts from the current DINO-only replay version. Its
only additional training-step changes are the two `structure_weight > 0`
guards used by the live MAE `trainer_bridge.py`. These are integrated directly
instead of modifying function source text at runtime. Default replay behavior
for earlier configs is preserved.

Runtime imports are package-local. Asset paths are relocatable and CPU
affinity is optional/configurable. These infrastructure changes do not alter
scientific hyperparameters. The DINO runtime reproduces its packed positive
bank, native/NumPy lossless color decoder, compact ImageFolder metadata and
prefetch policy. The historical raw GAN pair uses the standard input path as
its source launcher did. MAE uses the complete packed clean/flip latent cache.

Source hashes and configuration relocation records:

- [Shared source manifest](core_source_manifest.json)
- [DINO provenance](../experiments/dino/provenance.json)
- [MAE provenance and asset hashes](../experiments/mae/provenance.json)

The older source DINO launch report described EMA .99 and structure weight 1.
It is not used as evidence of the current hard-copy/zero-structure runs.
See [the method table](ADVERSARIAL_METHODS.md) for the distinctions.

## Executed verification

CPU validation used `/venv/main/bin/python` with PyTorch `2.11.0+cu128`
and torchvision `0.26.0+cu128`, with CUDA hidden from every test process.
Pytest and its dependencies were installed only in
`/tmp/idrift-port-test-deps`; the environment used by ongoing training was
not upgraded.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/idrift-port-test-deps:. \
IDRIFT_MAE_REFERENCE_ROOT=/workspace/mae256_gan_prepare_20260910 \
python -m pytest tests -q
```

Final result: **290 passed, 2 skipped, 600 subtests passed** in 16.63 seconds.
The two skipped tests require CUDA. The 46 reported warnings are inherited
Python 3.12 multiprocessing fork deprecation warnings. On another machine,
install `requirements-dev.txt`, omit the temporary `PYTHONPATH`, and omit the
MAE reference variable unless that original source tree is available.

Validation includes:

- Source-identical shared loss/model/input modules and normalized trainer AST.
- Exact original/ported DINO train-step comparisons for loss, metrics,
  generator gradients and optimizer state, online/target D and D optimizer
  state, including lazy R1 and `raw_gan`, `feature_drift`, `mixed` branches.
- Replay routing: frozen DINO retains rho .35 history while the active CNN
  branch receives no history and no replay weights; default true preserves
  the original both-branches behavior.
- Exact original/ported MAE native maps, pooled features, logits, generator
  input gradients, R1 updates at steps 0/1/16, optimizer and target state.
- Latent checkpoint architecture identity and incompatible-state rejection.
- Full trainer CPU steps using a latent encoder that cannot provide DINO
  structure-teacher maps, plus lossless cache/sampler/clean-flip checks.
- Two-rank CPU Gloo discriminator synchronization and collective rejection
  of non-finite updates.
- DINO native decoder compilation, ring-buffer/RNG equality, calibration
  hash, raw ImageNet manifest, and initial generator fingerprints for both
  ranks with seeds 43/44.
- MAE cache completeness (1,281,167 training and 50,000 validation examples)
  and SHA256 matches for the actual MAE checkpoint and VAE config/weights.
- Syntax compilation and `git diff --check`.

The final machine-readable [validation result](port_validation_result.json)
records the suite result and the live-process/source preservation audit.

Three inherited test assumptions needed correction, each reproduced in the
untouched original source first: an archived logging threshold was absent
from a config-difference allowlist; a diagnostic used a fixed decimal-place
comparison tighter than its existing float32 loss/gradient tolerance; and
the RAM-mirror fork integration test ran in a parent containing tqdm's
background thread. The last now runs the same HTTP-range/resume/checksum
scenario in a clean subprocess, preserving the production fork-safety guard.
These changes affect tests, not the training implementation.

## Limits and current experiment preservation

The four existing training jobs and monitor were left running. No GPU
training, production evaluation, W&B upload, service restart or source edit
was performed as part of this port. Asset preflight reads shared files.

CPU numerical parity and exact config comparisons establish agreement for
the checked operations. They do not measure final FID, guarantee identical
future GPU trajectories, or establish bitwise checkpoint resumption. The
original trainer does not checkpoint every RNG and raw-memory-bank state.
Perform a separate full-data GPU smoke/reproduction when GPUs are available;
the documented launches use the same scientific configurations.
