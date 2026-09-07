#!/usr/bin/env python3
"""Bounded CPU-only real-ImageNet dense versus lossless pixel-bank benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_bank import ArrayMemoryBank, CompressedPixelMemoryBank  # noqa: E402
from train.train_data import _build_transforms  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet", default=str(ROOT / "data/imagenet/raw_ilsvrc2012"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    train_root = Path(args.imagenet) / "train"
    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    selected = [class_dirs[index] for index in (0, len(class_dirs)//3, 2*len(class_dirs)//3, len(class_dirs)-1)]
    transform = _build_transforms(256, use_aug=False, split="train", return_uint8=True)
    source = np.empty((512, 3, 256, 256), dtype=np.uint8)
    labels = np.repeat(np.arange(4, dtype=np.int64), 128)
    source_names = []
    for label, directory in enumerate(selected):
        paths = sorted(directory.glob("*.JPEG"))[:128]
        if len(paths) != 128:
            raise RuntimeError(f"Need 128 real JPEGs for {directory}")
        for slot, path in enumerate(paths):
            with Image.open(path) as im:
                source[label * 128 + slot] = transform(im.convert("RGB")).numpy()
            source_names.append(str(path.relative_to(train_root)))
    dense = ArrayMemoryBank(num_classes=4, max_size=128, storage_mode="pixel_uint8")
    compressed = CompressedPixelMemoryBank(num_classes=4, max_size=128, codec_workers=4)
    banks = {"dense": dense, "zstd_delta": compressed}
    for offset in range(0, 512, 128):
        for bank in banks.values():
            bank.add(source[offset:offset+128], labels[offset:offset+128])
    # Production sampling is B4/P64, with each test class fully populated.
    sample_labels = np.array([3, 0, 2, 1], dtype=np.int64)
    timings = {name: {"add_seconds": [], "sample_cpu_seconds": [],
                     "sample_host_uint8_seconds": []} for name in banks}
    generator_rng_matches = 0
    tensor_matches = 0
    for step in range(args.warmup + args.steps):
        indices = (np.arange(128) * 5 + step * 128) % 512
        batch, batch_labels = source[indices], labels[indices]
        outputs = {}
        next_rng = {}
        order = ("dense", "zstd_delta") if step % 2 == 0 else ("zstd_delta", "dense")
        for name in order:
            bank = banks[name]
            start = time.perf_counter()
            bank.add(batch, batch_labels)
            add_elapsed = time.perf_counter() - start
            rng = np.random.default_rng(43 + step)
            # Observe the exact boundary before CPU-only float expansion.
            # All tensor operations still execute, so value/RNG checks below
            # verify the original sample semantics. Production transfers this
            # uint8 tensor to CUDA and normalizes there instead.
            original_from_numpy = torch.from_numpy
            boundary = []

            class FloatBoundary:
                def __init__(self, tensor):
                    self.tensor = tensor

                def float(self):
                    boundary.append(time.perf_counter())
                    return self.tensor.float()

            def observed_from_numpy(array):
                return FloatBoundary(original_from_numpy(array))

            start = time.perf_counter()
            torch.from_numpy = observed_from_numpy
            try:
                outputs[name] = bank.sample(sample_labels, n_samples=64, rng=rng)
            finally:
                torch.from_numpy = original_from_numpy
            sample_elapsed = time.perf_counter() - start
            if len(boundary) != 1:
                raise RuntimeError("Unexpected bank sample float-conversion path")
            host_uint8_elapsed = boundary[0] - start
            next_rng[name] = rng.integers(0, 2**62, size=16).tolist()
            if step >= args.warmup:
                timings[name]["add_seconds"].append(add_elapsed)
                timings[name]["sample_cpu_seconds"].append(sample_elapsed)
                timings[name]["sample_host_uint8_seconds"].append(host_uint8_elapsed)
        if not torch.equal(outputs["dense"], outputs["zstd_delta"]):
            raise AssertionError(f"Pixel values differ at step {step}")
        if next_rng["dense"] != next_rng["zstd_delta"]:
            raise AssertionError(f"Generator RNG trace differs at step {step}")
        np.testing.assert_array_equal(dense.ptr, compressed.ptr)
        np.testing.assert_array_equal(dense.count, compressed.count)
        tensor_matches += 1
        generator_rng_matches += 1
        del outputs
    np.random.seed(713)
    dense_out = dense.sample(sample_labels, n_samples=64)
    dense_next = np.random.randint(0, 2**31, size=16)
    np.random.seed(713)
    compressed_out = compressed.sample(sample_labels, n_samples=64)
    compressed_next = np.random.randint(0, 2**31, size=16)
    if not torch.equal(dense_out, compressed_out):
        raise AssertionError("Global NumPy RNG sample values differ")
    np.testing.assert_array_equal(dense_next, compressed_next)
    summary = {
        "real_images": len(source), "source_classes": [path.name for path in selected],
        "source_paths": source_names,
        "transform": "Production ADM center crop 256 + seeded training hflip + CHW uint8",
        "seed": 43, "steps": args.steps, "warmup": args.warmup,
        "batch_add": 128, "sample_batch": 4, "samples_per_class": 64,
        "classes": 4, "bank_capacity_per_class": 128,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_num_threads": 1,
        "exact_tensor_checks": tensor_matches + 1,
        "explicit_generator_rng_checks": generator_rng_matches,
        "global_numpy_rng_check": "PASS", "ring_count_pointer_checks": "PASS",
        "dense_bank_gib": dense.bank.nbytes / 1024**3,
        "compressed_payload_gib": compressed.payload_bytes / 1024**3,
        "compression_ratio": compressed.compression_ratio,
        "production_dense_positive_gib_per_rank": 1000 * 128 * 3 * 256**2 / 1024**3,
        "production_four_new_ranks_positive_gib": 4 * 1000 * 128 * 3 * 256**2 / 1024**3,
        "production_four_new_ranks_negative_gib": 4 * 1000 * 3 * 256**2 / 1024**3,
        "timings": {name: {key: {"median": statistics.median(values), "mean": statistics.mean(values),
                                "values": values} for key, values in measures.items()}
                    for name, measures in timings.items()},
        "limitations": [
            "CPU-only sample timing includes float normalization on CPU; production copies uint8 to CUDA before normalization.",
            "Four fully populated classes use bounded RAM and do not reproduce full-bank DRAM locality.",
            "512 actual transformed JPEGs cycle through ring updates; no synthetic input or GPU is used.",
        ],
    }
    compressed.suspend_codec_workers()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "checks": "PASS", "timing_medians_seconds": {
        name: {key: statistics.median(values) for key, values in measures.items()}
        for name, measures in timings.items()}, "dense_production_gib_per_rank": summary["production_dense_positive_gib_per_rank"]}), flush=True)


if __name__ == "__main__":
    main()
