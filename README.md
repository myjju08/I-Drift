# ReplayDrift: DINO and adversarial experiments

Direct RGB ImageNet-256 generation with an S4 generator and frozen DINO ResNet-50
features. The code supports the DINO baseline, real/fake tuning of DINO, raw
conditional GAN loss, adversarial feature drifting, their combination, and
historical generated-sample replay.

## Supported experiments

All configurations below are in `configs/gen/` and begin with
`S4_pixel-p32_direct_dino-r50-stage34only-r64g32-mae-matched`.
The historical `mae-matched` filename records the experiment geometry;
these configurations run DINO on raw RGB images. DINO stages 3 and 4 are active.

| Config suffix | Generator objective |
|---|---|
| `.yaml` | Frozen DINO drifting |
| `-tuned-rf1500.yaml`, `-tuned-rf1500-io1.yaml` | Drifting with a separately tuned, then frozen DINO |
| `-raw-conditional-gan.yaml` | DINO drifting + raw conditional GAN, D32 |
| `-raw-conditional-gan-d128.yaml` | DINO drifting + raw conditional GAN, D128 |
| `-adversarial-feature-drift.yaml` | DINO drifting + discriminator-feature drifting |
| `-mixed-gan-feature-drift-d32-opt4.yaml` | Both adversarial terms, D32 |
| `-mixed-gan-feature-drift-d128-opt4.yaml` | Both adversarial terms, D128 |
| `-historical-replay-e10-rho050.yaml` | DINO drifting with a frozen historical sample bank |

`models/adversarial_drift.py` implements the class-conditioned discriminator,
logistic discriminator objective, lazy R1, real-feature structure preservation,
and the target discriminator. `train_imagenet_gen.py` combines the selected
objectives and saves/restores the discriminator, optimizer and target.
In mixed mode the raw GAN and feature-drift terms both contribute to the generator.
The frozen DINO remains part of every generator objective.

Adapter, MoCo, MAE and latent-bridge training implementations have been removed.
Disabled legacy configuration fields and historical calibration provenance are
accepted for existing DINO checkpoints; removed training modes are rejected.

## Local setup and training

Install a PyTorch build suitable for your GPU, then `pip install -r requirements.txt`.
ImageNet, DINO weights, calibration artifacts and evaluation references are external
inputs. The raw archive downloader is `scripts/download_extract_raw_imagenet.py`.

The YAMLs preserve the existing experiments' settings and local paths.
Before using another machine, copy the desired YAML and set `env.imagenet_path`,
`feature.feature_checkpoint`, your W&B settings and the calibration artifact paths.
The raw ImageNet root must contain `train/` and `val/` class directories.
Existing calibrated runs also need their pinned calibration files and SHA256 values;
The tuned DINO variants additionally require their temperature-inheritance sidecar.
The legacy calibration refers to a DINO/MoCo comparison as provenance, without
loading or training MoCo. See `scripts/calibrate_feature_encoder_temperatures.py`
for DINO-only diagnostic capture using an already verified profile. Fitting new
production temperatures requires a separate calibration study.

From the repository root, launch the chosen configuration on two available GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_imagenet_gen.py --config /path/to/local-dino-config.yaml \
  --workdir runs/dino-experiment
```

Use a new work directory for a new experiment. Reusing a work directory resumes
its latest checkpoint. Preserve the existing generator, discriminator, data,
temperature and runtime settings when comparing an existing run.
The mixed configs select throughput opt4; baseline/raw GAN configs retain their
original optimization level. They are not a new matched-runtime ablation.
`scripts/benchmark_s4_mixed_adversarial.py` measures throughput without production
W&B logging, and `scripts/smoke_s4_adversarial.py` checks a short two-rank run/resume.

## Tune a separate DINO

`scripts/prepare_dino_rf_tuning_data.py` makes matched real/generated training and
held-out pairs from a generator checkpoint. Then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/tune_dino_real_fake.py --manifest /path/to/pairs/manifest.json \
  --teacher-checkpoint /path/to/original-dino.pth \
  --workdir runs/dino-rf-tuning --steps 1500
```

This tunes convolution weights in DINO `layer3.5` and `layer4.2` with a class-conditioned
real/fake head and a real-feature preservation term. Other backbone parameters and
batch-normalization state stay frozen. The selected backbone is exported to
`WORKDIR/dino_tuned.pth`; the original teacher checkpoint is preserved.
Generator training loads this export as a frozen feature extractor.

## Evaluation and checks

The NPZ roundtrip evaluator generates a uint8 sample archive, reads it back,
and computes FID, Inception Score, precision and recall. Example matching the
existing 40-epoch protocol:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_npz_roundtrip.py \
  --config /path/to/local-dino-config.yaml --ckpt /path/to/generator.pt \
  --cfg_scale 1.5 --n_samples 50000 --seed 0 --label_source official_val \
  --batch_size 64 --metrics_batch_size 128 \
  --fid_ref_npz /path/to/fid-reference.npz \
  --pr_ref_npz /path/to/reference-images.npz --pr_ref_count 10000 --pr_nhood 3 \
  --out runs/evaluation/metrics.json
```

References and sample protocol must agree across comparisons. The bundled upstream
TensorFlow evaluator is available for independent checking and additionally needs
TensorFlow and requests; the normal evaluation command above uses torch-fidelity.
Run the regression suite with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q`.

## Code-only repository

Data, original/tuned weights, generator checkpoints, calibration outputs, generated
images, metric archives, logs, W&B state and credentials stay outside Git.
`.gitignore` excludes these files. Historical results are not bundled here.

ReplayDrift follows the [Drifting](https://github.com/lambertae/drifting), DualDrift
and [I-Drift](https://github.com/cosmosjhj/I-Drift) code lineage. The bundled
OpenAI guided-diffusion evaluator retains its upstream license in
`third_party/openai_guided_diffusion/LICENSE`.
