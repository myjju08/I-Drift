"""Validate the matched Corrective Field suite without creating a W&B run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ("baseline", "replay_double", "replay_double_gan")
ALLOWED_DIFFERENCES = {
    "name", "historical_gen_replay", "double_drift",
    "adversarial_mode", "adversarial_loss_weight",
}
ENTITY = "a01065522071-kaist-digital-humanities-and-social-science"
DATASET_ROWS = 1_281_168
GENERATED_PER_STEP = 512
MAE_SHA256 = "59c269f99d83645b6c7bb2bf832711aa83d894998259a1ada16c0c9ed7836081"
VAE_SHA256 = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject YAML keys that the trainer would otherwise silently overwrite."""


def _construct_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    raw = yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a sectioned config: {path}")
    cfg = {}
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"Expected mapping for config section {section}")
        for key, value in values.items():
            if key in cfg:
                raise ValueError(f"Duplicate flattened config key: {key}")
            cfg[key] = value
    return cfg


def validate_suite(suite_dir):
    configs = {name: read_config(Path(suite_dir) / f"{name}.yaml") for name in VARIANTS}
    common = {k: v for k, v in configs["baseline"].items() if k not in ALLOWED_DIFFERENCES}
    for variant, cfg in configs.items():
        actual = {k: v for k, v in cfg.items() if k not in ALLOWED_DIFFERENCES}
        if actual != common:
            changed = sorted(k for k in actual.keys() | common.keys() if actual.get(k) != common.get(k))
            raise ValueError(f"Unmatched {variant} configuration: {changed}")
        expected_variant = {
            "historical_gen_replay": variant != "baseline",
            "double_drift": variant != "baseline",
            "adversarial_mode": "raw_gan" if variant == "replay_double_gan" else "none",
            "adversarial_loss_weight": 0.1 if variant == "replay_double_gan" else 0.0,
        }
        for key, value in expected_variant.items():
            if cfg.get(key) != value:
                raise ValueError(f"Wrong {key} for {variant}: expected {value!r}")
    expected = {
        "project": "Corrective Field", "entity": ENTITY,
        "use_wandb": True, "require_wandb": True,
        "resolution": 256, "use_latent": True, "use_cache": True,
        "cache_format": "npy_flat", "latent_scaling_factor": 0.18215,
        "num_classes": 1000, "use_aug": False,
        "input_size": 32, "in_channels": 4, "out_channels": 4,
        "patch_size": 4, "hidden_size": 384, "cond_dim": 384,
        "depth": 12, "num_heads": 6, "use_bf16": True,
        "pos_per_sample": 32, "neg_per_sample": 32,
        "gen_per_label": 32, "batch_size": 8,
        "feature_extractor": "mae", "feature_encoder_only": True,
        "feature_adapter": False, "feature_gan": False,
        "feature_use_bf16": True, "feature_include_norm_x": True,
        "memory_bank_storage_mode": "raw", "positive_memory_bank_backend": "dense",
        "raw_train_uint8": False, "seed": 43, "seed_host_rng": True,
        "total_generated_epochs": 40.0,
        "total_steps": math.ceil(DATASET_ROWS * 40 / GENERATED_PER_STEP),
        "eval_at_start": True, "require_eval_metrics": True, "preserve_rng_during_eval": True,
        "eval_per_generated_epochs": 10.0, "save_per_generated_epochs": 0.5,
        "eval_samples": 1024, "cfg_list": [1.0, 2.0, 3.0],
        "drift_matching": "rev-drift", "historical_gen_replay_ratio": 0.35,
        "historical_gen_replay_count": 16, "historical_gen_replay_bank_count": 16,
        "historical_gen_replay_start_generated_epochs": 10.0,
        "historical_gen_replay_source": "frozen_snapshot",
        "historical_gen_replay_ratio_start": None,
        "historical_gen_replay_ratio_ramp_start_step": 0,
        "historical_gen_replay_ratio_ramp_end_step": 0,
        "historical_gen_current_weight": None, "historical_gen_history_weight": None,
        "adversarial_drift_weight": 0, "adversarial_structure_weight": 0,
        "adversarial_base_channels": 32, "adversarial_seed": 43,
        "adversarial_r1_gamma": 1.0, "adversarial_r1_interval": 16,
    }
    failed = [f"{key}: expected {value!r}, got {common.get(key)!r}"
              for key, value in expected.items() if common.get(key) != value]
    if common.get("feature_spatial_pool", 1) != 1:
        failed.append("feature_spatial_pool must be 1 for latent MAE")
    for key in ("bulk_feature_drift", "grouped_feature_drift", "compile_backbone"):
        if key in common:
            failed.append(f"{key} is not supported in this suite")
    if len({cfg["name"] for cfg in configs.values()}) != len(VARIANTS):
        failed.append("W&B run names must distinguish all three arms")
    if failed:
        raise ValueError("Corrective Field suite checks failed: " + "; ".join(failed))
    return configs


def validate_snapshot(source_root, manifest_path):
    """Check every archived source file before a queued job imports its trainer."""
    source_root = Path(source_root).resolve(strict=True)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("kind") != "corrective_field_source_snapshot":
        raise ValueError("Unexpected source snapshot manifest")
    if not manifest.get("commit") or len(manifest["commit"]) != 40:
        raise ValueError("Snapshot must identify a full Git commit")
    for relative, expected in manifest["files_sha256"].items():
        path = source_root / relative
        if not path.resolve(strict=True).is_relative_to(source_root):
            raise ValueError(f"Source file escapes snapshot: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Source snapshot checksum mismatch: {relative}")
    actual_files = {str(p.relative_to(source_root)) for p in source_root.rglob("*") if p.is_file()}
    if actual_files != set(manifest["files_sha256"]):
        raise ValueError("Source snapshot file set differs from committed archive")
    return manifest


def validate_runtime_environment(cfg):
    if os.environ.get("WANDB_MODE") != "online":
        raise ValueError("Production requires WANDB_MODE=online")
    for name, expected in (("WANDB_PROJECT", cfg["project"]), ("WANDB_ENTITY", cfg["entity"])):
        if os.environ.get(name) != expected:
            raise ValueError(f"{name} must match the checked configuration")
    if os.environ.get("WANDB_DISABLED") or os.environ.get("WANDB_RUN_ID"):
        raise ValueError("Production requires a fresh online W&B run")
    import torch
    if torch.cuda.device_count() != 2:
        raise ValueError(f"Each suite arm requires exactly 2 visible GPUs, got {torch.cuda.device_count()}")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if any("3090" not in name for name in names):
        raise ValueError(f"Expected the requested srv02 RTX3090 GPUs, got {names}")
    return names


def validate_assets(cfg):
    import numpy as np
    checkpoint = Path(cfg["feature_checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != MAE_SHA256:
        raise ValueError("MAE256 checkpoint checksum mismatch")
    decoder = Path(cfg["latent_decoder_path"]).resolve(strict=True)
    if sha256(decoder / "diffusion_pytorch_model.safetensors") != VAE_SHA256:
        raise ValueError("Evaluation VAE checkpoint checksum mismatch")
    from train.latent_data import MmapLatentDataset, read_flat_latent_row
    dataset = MmapLatentDataset(cfg["latent_mmap_cache_path"], source_root=cfg["cache_path"])
    if len(dataset) != DATASET_ROWS:
        raise ValueError(f"Expected exactly {DATASET_ROWS:,} source latent rows")
    counts = dataset.metadata["label_counts"]
    if len(counts) != 1000 or min(counts) <= 0:
        raise ValueError("The latent cache must cover all 1,000 ImageNet classes")
    rows = sorted(set(np.linspace(0, len(dataset) - 1, 32, dtype=np.int64).tolist()))
    for row in rows:
        source, label, _ = read_flat_latent_row(cfg["cache_path"], row)
        packed, packed_label = dataset[row]
        packed = packed.numpy()
        if source.shape != (4, 32, 32) or packed.dtype != np.float32:
            raise ValueError(f"Unexpected latent dtype/shape at row {row}")
        if label != packed_label or not np.array_equal(source.view(np.uint32), packed.view(np.uint32)):
            raise ValueError(f"Packed latent differs from original source bits at row {row}")
        if not np.isfinite(packed).all():
            raise ValueError(f"Non-finite latent values at row {row}")
    if not (Path(cfg["imagenet_path"]) / "val").is_dir():
        raise ValueError("Original ImageNet RGB validation data is required for evaluation")
    return {
        "feature_checkpoint": str(checkpoint), "feature_checkpoint_sha256": MAE_SHA256,
        "evaluation_decoder": {"path": str(decoder), "weights_sha256": VAE_SHA256,
                               "config_sha256": sha256(decoder / "config.json")},
        "data": dataset.metadata, "exact_bitwise_source_rows_checked": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=ROOT / "configs/corrective_field")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--config-only", action="store_true", help="Check suite settings without GPUs/data/W&B.")
    parser.add_argument("--runtime-source", type=Path, default=ROOT)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configs = validate_suite(args.suite_dir)
    if args.config_only:
        print("[preflight] PASS: all three Corrective Field configurations are matched")
        return
    if args.config is None or args.variant is None or args.output is None:
        parser.error("runtime validation requires --config, --variant, and --output")
    cfg = read_config(args.config)
    if cfg != configs[args.variant]:
        raise ValueError("Launched config does not match its named suite arm")
    manifest = (validate_snapshot(args.runtime_source, args.snapshot_manifest)
                if args.snapshot_manifest else None)
    gpu_names = validate_runtime_environment(cfg)
    report = {
        "kind": "corrective_field_preflight", "variant": args.variant,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": manifest["commit"] if manifest else None,
        "config_sha256": sha256(args.config),
        "suite_config_sha256": {name: sha256(args.suite_dir / f"{name}.yaml") for name in VARIANTS},
        "source_sha256": manifest["files_sha256"] if manifest else None,
        "allowed_arm_differences": sorted(ALLOWED_DIFFERENCES),
        "gpu_names": gpu_names, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "generated_per_step": GENERATED_PER_STEP, "total_steps": cfg["total_steps"],
        "replay_freeze_step": math.ceil(DATASET_ROWS * 10 / GENERATED_PER_STEP),
        "training_representation": "Original FP32 4x32x32 latent values; frozen MAE256; no VAE re-encoding",
        "evaluation": {"real": "original ImageNet validation RGB", "generated": "SD-VAE decode(latent / 0.18215)",
                       "samples": 1024, "cfg_scales": cfg["cfg_list"],
                       "limitation": "Inherited 1024-image ordered validation prefix; not ImageNet FID50k"},
        **validate_assets(cfg),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[preflight] PASS: {args.variant}; report={args.output}", flush=True)


if __name__ == "__main__":
    main()
