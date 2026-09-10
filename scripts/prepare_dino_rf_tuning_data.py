#!/usr/bin/env python3
"""Stage a fixed, class-balanced real/fake corpus for offline DINO tuning.

Real examples come exclusively from disjoint paths in ImageNet's training
split. Both primary domains are CHW uint8; optional float16 generator outputs
are retained only for diagnostics. This script never trains an encoder or G.
Run one Python process with two visible GPUs; generation uses DataParallel.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_data import _build_transforms  # noqa: E402


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    encoded = json.dumps([int(seed), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def select_real_examples(
    imagenet_root: str | Path, *, train_per_class: int, validation_per_class: int,
    train_seed: int, validation_seed: int, num_classes: int,
) -> tuple[list[str], dict[str, list[dict]]]:
    root = Path(imagenet_root).resolve()
    train_root = root / "train"
    classes = sorted(path for path in train_root.iterdir() if path.is_dir())
    if len(classes) != num_classes:
        raise ValueError(f"Expected {num_classes} ImageNet training classes, found {len(classes)}")
    if min(train_per_class, validation_per_class) < 1:
        raise ValueError("Both splits need at least one example per class")
    selections = {"train": [], "validation": []}
    extensions = {".jpg", ".jpeg", ".png"}
    for label, class_dir in enumerate(classes):
        candidates = sorted(path for path in class_dir.iterdir()
                            if path.is_file() and path.suffix.lower() in extensions)
        count = train_per_class + validation_per_class
        if len(candidates) < count:
            raise ValueError(f"{class_dir.name} has {len(candidates)} images; need {count}")
        rng = np.random.default_rng(stable_seed(train_seed, "real_selection", class_dir.name))
        selected = rng.choice(len(candidates), size=count, replace=False)
        for split, offset, amount, seed in (
            ("train", 0, train_per_class, train_seed),
            ("validation", train_per_class, validation_per_class, validation_seed),
        ):
            for slot, index in enumerate(selected[offset:offset + amount]):
                path = candidates[int(index)]
                # Path resolution must not permit a train symlink to read val.
                if not path.resolve().is_relative_to(train_root.resolve()):
                    raise ValueError(f"Selected training image resolves outside train/: {path}")
                selections[split].append({
                    "path": str(path), "relative_path": str(path.relative_to(root)),
                    "label": label, "class_name": class_dir.name,
                    "split": split, "slot": slot,
                    "transform_seed": stable_seed(seed, "real_transform", class_dir.name, slot),
                })
    for split, seed in (("train", train_seed), ("validation", validation_seed)):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(selections[split]))
        selections[split] = [selections[split][int(index)] for index in order]
    first = {item["path"] for item in selections["train"]}
    second = {item["path"] for item in selections["validation"]}
    if first & second:
        raise AssertionError("Real training and heldout paths overlap")
    return [path.name for path in classes], selections


def load_real_pixels(record: dict, resolution: int = 256) -> tuple[np.ndarray, str]:
    data = Path(record["path"]).read_bytes()
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    transform = _build_transforms(resolution, use_aug=False, split="train", return_uint8=True)
    # Seed/restore only the CPU RNG used by RandomHorizontalFlip, preserving G's
    # CUDA noise streams even when the two preparation phases are interleaved.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(record["transform_seed"]))
        pixels = transform(image)
    return pixels.numpy(), hashlib.sha256(data).hexdigest()


def quantize_fake_pixels(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"Expected NCHW RGB generator outputs, got {tuple(samples.shape)}")
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Non-finite generator output; refusing to sanitize fake data")
    # Same truncating pixel convention as the existing offline image evaluator.
    return ((samples.float() + 1.0) / 2.0).clamp(0.0, 1.0).mul(255.0).to(torch.uint8)


def open_array(path: Path, shape: tuple, dtype) -> np.memmap:
    if path.exists():
        raise FileExistsError(path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def stage_real_split(directory: Path, records: list[dict], resolution: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    image_path, labels_path = directory / "real_images.npy", directory / "labels.npy"
    images = open_array(image_path, (len(records), 3, resolution, resolution), np.uint8)
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    np.save(labels_path, labels, allow_pickle=False)
    records_path = directory / "real_sources.jsonl"
    content_hash = hashlib.sha256()
    with records_path.open("x") as handle:
        for index, record in enumerate(tqdm(records, desc=f"real-{directory.name}", unit="image")):
            pixels, source_hash = load_real_pixels(record, resolution)
            images[index] = pixels
            content_hash.update(pixels.tobytes())
            handle.write(json.dumps({"index": index, **record, "source_sha256": source_hash}) + "\n")
    images.flush()
    del images
    return {
        "real_images": str(image_path), "real_labels": str(labels_path),
        "fake_images": str(directory / "fake_images.npy"), "fake_labels": str(labels_path),
        "real_sources": str(records_path), "real_sources_sha256": sha256_file(records_path),
        "real_pixels_sha256": content_hash.hexdigest(),
        "labels_sha256": sha256_file(labels_path), "examples_per_domain": len(labels),
        "class_counts": {str(label): count for label, count in sorted(Counter(labels.tolist()).items())},
    }


@torch.no_grad()
def stage_fake_split(
    directory: Path, generator: torch.nn.Module, *, seed: int, cfg_scale: float | None,
    batch_size: int, device: torch.device, resolution: int, save_raw_fake: bool,
    cfg_min: float = 1.0, cfg_max: float = 4.0,
) -> dict:
    labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
    shape = (len(labels), 3, resolution, resolution)
    fake_path, raw_path = directory / "fake_images.npy", directory / "fake_raw_float16.npy"
    quantized = open_array(fake_path, shape, np.uint8)
    raw_array = open_array(raw_path, shape, np.float16) if save_raw_fake else None
    scales_path = directory / "fake_cfg_scales.npy"
    # This independent CPU stream makes CFG values reproducible independently of
    # generator noise and ensures DataParallel receives matching per-image CFG.
    scales_rng = np.random.default_rng(stable_seed(seed, "fake_cfg_scales"))
    scales = (scales_rng.uniform(cfg_min, cfg_max, size=len(labels)).astype(np.float32)
              if cfg_scale is None else np.full(len(labels), cfg_scale, dtype=np.float32))
    np.save(scales_path, scales, allow_pickle=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw_generator = generator.module if hasattr(generator, "module") else generator
    use_bf16 = bool(getattr(raw_generator, "use_bf16", False))
    pixel_hash, raw_hash = hashlib.sha256(), hashlib.sha256()
    minimum, maximum, outside, pixels_seen = float("inf"), float("-inf"), 0, 0
    for start in tqdm(range(0, len(labels), batch_size), desc=f"fake-{directory.name}", unit="batch"):
        end = min(start + batch_size, len(labels))
        class_ids = torch.tensor(np.asarray(labels[start:end]), dtype=torch.long, device=device)
        batch_scales = torch.tensor(scales[start:end], dtype=torch.float32, device=device)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda" else nullcontext()
        with context:
            samples = generator(class_ids, cfg_scale=batch_scales, train=False)["samples"]
        if tuple(samples.shape) != (end - start, *shape[1:]):
            raise ValueError(f"Unexpected generator shape {tuple(samples.shape)}")
        pixels = quantize_fake_pixels(samples).cpu().numpy()
        quantized[start:end] = pixels
        pixel_hash.update(pixels.tobytes())
        raw = samples.float()
        minimum, maximum = min(minimum, float(raw.min())), max(maximum, float(raw.max()))
        outside += int(((raw < -1) | (raw > 1)).sum())
        pixels_seen += raw.numel()
        if raw_array is not None:
            values = raw.to(torch.float16).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("float16 diagnostic storage overflowed")
            raw_array[start:end] = values
            raw_hash.update(values.tobytes())
        del samples, raw, pixels, class_ids, batch_scales
    quantized.flush()
    del quantized
    metadata = {
        "fake_pixels_sha256": pixel_hash.hexdigest(), "fake_seed": int(seed),
        "fake_cfg_scales": str(scales_path), "fake_cfg_scales_sha256": sha256_file(scales_path),
        "raw_fake_min": minimum, "raw_fake_max": maximum,
        "raw_fake_outside_minus1_plus1_fraction": outside / max(1, pixels_seen),
    }
    if raw_array is not None:
        raw_array.flush()
        del raw_array
        metadata.update(raw_fake_images=str(raw_path), raw_fake_dtype="float16",
                        raw_fake_pixels_sha256=raw_hash.hexdigest(),
                        raw_fake_usage="diagnostic_only_not_primary_training")
    return metadata


def publish_manifest(directory: Path, manifest: dict) -> Path:
    for split in ("train", "validation"):
        spec = manifest[split]
        arrays = {key: np.load(spec[key], mmap_mode="r", allow_pickle=False)
                  for key in ("real_images", "fake_images", "real_labels", "fake_labels")}
        if arrays["real_images"].shape != arrays["fake_images"].shape:
            raise ValueError(f"Real/fake shapes disagree for {split}")
        if arrays["real_images"].ndim != 4 or arrays["real_images"].shape[1] != 3:
            raise ValueError(f"Images must be NCHW RGB for {split}")
        if arrays["real_images"].dtype != np.uint8 or arrays["fake_images"].dtype != np.uint8:
            raise ValueError("Both primary domains must use uint8 pixels")
        if not np.array_equal(arrays["real_labels"], arrays["fake_labels"]):
            raise ValueError(f"Real/fake class order differs for {split}")
        if arrays["real_labels"].ndim != 1 or not np.issubdtype(arrays["real_labels"].dtype, np.integer):
            raise ValueError(f"Labels must be a one-dimensional integer array for {split}")
        if arrays["real_images"].shape[0] != arrays["real_labels"].shape[0]:
            raise ValueError(f"Image/label count differs for {split}")
        if "raw_fake_images" in spec:
            raw = np.load(spec["raw_fake_images"], mmap_mode="r", allow_pickle=False)
            if raw.shape != arrays["real_images"].shape or raw.dtype != np.float16:
                raise ValueError(f"Raw diagnostic images must match primary shape with float16 dtype for {split}")
            del raw
        if "fake_cfg_scales" in spec:
            scales = np.load(spec["fake_cfg_scales"], mmap_mode="r", allow_pickle=False)
            if scales.shape != arrays["real_labels"].shape or not np.isfinite(scales).all():
                raise ValueError(f"CFG scales must be finite and match label shape for {split}")
            del scales
        del arrays
    destination, temporary = directory / "manifest.json", directory / "manifest.json.tmp"
    if destination.exists():
        raise FileExistsError(destination)
    with temporary.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imagenet-root", required=True)
    parser.add_argument("--out", required=True, help="Fresh directory, preferably under /dev/shm")
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--validation-per-class", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=43)
    parser.add_argument("--validation-seed", type=int, default=44)
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="Fixed CFG override; default samples a recorded scale uniformly per image")
    parser.add_argument("--cfg-min", type=float, default=1.0)
    parser.add_argument("--cfg-max", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-raw-fake", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.train_seed == args.validation_seed:
        parser.error("Training and validation generator seeds must differ")
    if (args.batch_size < 1 or not np.isfinite([args.cfg_min, args.cfg_max]).all()
            or args.cfg_min > args.cfg_max or args.cfg_min < 1
            or (args.cfg_scale is not None and not np.isfinite(args.cfg_scale))):
        parser.error("Batch size must be positive and CFG finite")
    directory = Path(args.out).resolve()
    if directory.exists():
        raise FileExistsError(f"Refusing an existing preparation directory: {directory}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Run one Python process with exactly two visible CUDA devices")

    from train_imagenet_gen import load_yaml_config
    from scripts.eval_official_imagenet256 import _build_generator, _maybe_dataparallel
    cfg = load_yaml_config(args.config)
    resolution, num_classes = int(cfg.get("resolution", 256)), int(cfg.get("num_classes", 1000))
    if resolution != 256 or bool(cfg.get("use_latent", True)):
        raise ValueError("This corpus requires the original direct-RGB ImageNet256 generator")
    classes, records = select_real_examples(
        args.imagenet_root, train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class, train_seed=args.train_seed,
        validation_seed=args.validation_seed, num_classes=num_classes,
    )
    images_per_domain = sum(map(len, records.values()))
    estimated_bytes = images_per_domain * 3 * resolution**2 * (2 + 2 * args.save_raw_fake)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(directory.parent)
    free_bytes = stats.f_bavail * stats.f_frsize
    if free_bytes < estimated_bytes + 1024**3:
        raise OSError(f"Corpus needs {estimated_bytes / 1024**3:.2f} GiB plus 1 GiB margin; only {free_bytes / 1024**3:.2f} GiB free")
    directory.mkdir()
    manifest = {
        "schema_version": 1, "status": "complete", "classes": classes,
        "num_classes": num_classes, "resolution": resolution,
        "generator_checkpoint": str(Path(args.checkpoint).resolve()),
        "generator_sha256": sha256_file(args.checkpoint), "generator_weight_key": "ema",
        "generator_config": str(Path(args.config).resolve()),
        "generator_config_sha256": sha256_file(args.config),
        "preparation_script_sha256": sha256_file(__file__),
        "source_hashes": {str(path): sha256_file(ROOT / path) for path in (
            "models/imagenet_generator.py", "scripts/eval_official_imagenet256.py", "train/train_data.py")},
        "source_imagenet_root": str(Path(args.imagenet_root).resolve()),
        "source_split": "train", "official_validation_or_fid_reference_used": False,
        "cfg_scale": args.cfg_scale, "generation_batch_size": args.batch_size,
        "cfg_distribution": ("uniform_per_image" if args.cfg_scale is None else "fixed"),
        "cfg_min": args.cfg_min, "cfg_max": args.cfg_max,
        "real_selection_seed": args.train_seed, "train_seed": args.train_seed,
        "validation_seed": args.validation_seed,
        "train_per_class": args.train_per_class, "validation_per_class": args.validation_per_class,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "generation_gpu_count": 2,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "numpy_madvise_hugepage": os.environ.get("NUMPY_MADVISE_HUGEPAGE"),
        "estimated_array_bytes": estimated_bytes,
        "input_policy": {
            "primary_layout": "NCHW", "primary_dtype": "uint8",
            "real_transform": "production ADM center crop256 + deterministic seeded training hflip; source byte hashes recorded",
            "fake_transform": "fixed EMA raw float -> ((x+1)/2).clamp(0,1)*255 -> truncate uint8",
            "tuner_normalization": "float32.div(255).sub(0.5).div(0.5), identically for real and fake",
            "primary_training_uses_raw_fake": False,
            "existing_generator_or_encoder_pipeline_modified": False,
            "future_drift_input_protocol_validated": False,
            "future_drift_requirement": "Validate raw-float versus uint8 robustness before any frozen-encoder continuation or transform change",
            "reproducibility_scope": "Recorded split seeds, generation_batch_size, two visible GPUs, software versions and source checkpoint; different GPU count/batching can change G noise samples",
        },
    }
    for split in ("train", "validation"):
        manifest[split] = stage_real_split(directory / split, records[split], resolution)
    device = torch.device("cuda:0")
    generator, loaded_step = _build_generator(cfg, args.checkpoint, device)
    generator.requires_grad_(False).eval()
    generator = _maybe_dataparallel(generator)
    manifest["generator_step"] = loaded_step
    for split, seed in (("train", args.train_seed), ("validation", args.validation_seed)):
        manifest[split].update(stage_fake_split(
            directory / split, generator, seed=seed, cfg_scale=args.cfg_scale,
            batch_size=args.batch_size, device=device, resolution=resolution,
            save_raw_fake=args.save_raw_fake, cfg_min=args.cfg_min, cfg_max=args.cfg_max,
        ))
    manifest_path = publish_manifest(directory, manifest)
    print(f"[dino-rf-data] Complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
