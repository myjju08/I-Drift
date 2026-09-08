#!/usr/bin/env python3
"""Measure real-data S4/DINO and mixed adversarial runs under two-rank torchrun.

The timed window includes bank input/codec work, training, EMA, and the original
logging cadence. CUDA is synchronized only at window boundaries by default.
Model, losses, B/G/P/N, bank capacities, real input, and sample cadence come from
the config. A fresh short run has naturally underfilled banks; this is reported
explicitly and must not be presented as a full-bank steady-state measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_imagenet_gen as trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True,
                        help="New benchmark directory; refuses checkpoints or old logs.")
    parser.add_argument("--output", help="Rank-zero JSON summary; default WORKDIR/benchmark.json")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32, help="Measured steps after warmup.")
    parser.add_argument("--throughput-opt-level", type=int, choices=range(5))
    parser.add_argument("--feature-real-microbatch-size", type=int)
    parser.add_argument("--feature-generated-microbatch-size", type=int)
    parser.add_argument("--adversarial-d-chunk-size", type=int)
    parser.add_argument("--adversarial-g-chunk-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--profile", action="store_true",
                        help="Component profiling adds synchronizations; benchmark separately.")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("warmup must be nonnegative and measured steps positive")

    # Force offline logging even if a parent service uses online W&B.
    os.environ["WANDB_MODE"] = "disabled"
    workdir = Path(args.workdir).resolve()
    if (workdir / "train_log.jsonl").exists() or list(workdir.glob("checkpoints/*.pt")):
        raise RuntimeError(f"Use a fresh benchmark directory: {workdir}")
    cfg = trainer.load_yaml_config(args.config)
    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    source_hashes = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in (
            "train_imagenet_gen.py", "models/imagenet_generator.py",
            "models/adversarial_drift.py", "memory_bank.py",
            "scripts/benchmark_s4_mixed_adversarial.py",
        )
    }
    get_madvise_hugepage = getattr(np._core.multiarray, "_get_madvise_hugepage", None)
    runtime_metadata = {
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "NUMPY_MADVISE_HUGEPAGE": os.environ.get("NUMPY_MADVISE_HUGEPAGE"),
        "numpy_actual_madvise_hugepage": (
            bool(get_madvise_hugepage()) if get_madvise_hugepage is not None else None
        ),
        "transparent_hugepage": {
            option: Path(f"/sys/kernel/mm/transparent_hugepage/{option}").read_text().strip()
            if Path(f"/sys/kernel/mm/transparent_hugepage/{option}").exists() else None
            for option in ("enabled", "defrag")
        },
    }
    if (int(cfg.get("batch_size", 0)), int(cfg.get("gen_per_label", 0)),
            int(cfg.get("pos_per_sample", 0)), int(cfg.get("neg_per_sample", 0))) != (4, 32, 64, 32):
        raise ValueError("Matched S4 benchmark requires B4/G32/P64/N32 per rank")
    overrides = {}
    for key in (
        "throughput_opt_level", "feature_real_microbatch_size",
        "feature_generated_microbatch_size", "adversarial_d_chunk_size",
        "adversarial_g_chunk_size", "num_workers",
    ):
        value = getattr(args, key)
        if value is not None:
            if key != "throughput_opt_level" and value < 1:
                parser.error(f"{key} must be positive")
            overrides[key] = value
    cfg.update(overrides)
    cfg.update(
        use_wandb=False,
        require_raw_temperature_calibration=False,
        eval_at_start=False,
        eval_per_generated_epochs=0.0,
        eval_per_step=1_000_000_000,
        save_per_generated_epochs=0.0,
        save_per_step=1_000_000_000,
        train_max_step_exclusive=args.warmup + args.steps,
        profile_train_step=bool(args.profile),
    )
    if args.profile:
        cfg["log_every_k"] = 1
    rank, world_size, device = trainer.setup_distributed()
    if world_size != 2 or device.type != "cuda":
        raise RuntimeError("Launch with torchrun --nproc_per_node=2 on two CUDA devices")
    workdir.mkdir(parents=True, exist_ok=True)

    original_set_step = trainer.Logger.set_step
    original_finish = trainer.Logger.finish
    original_bank_init = trainer.ArrayMemoryBank.__init__
    banks = []
    state = {"started": None, "ended": None, "boundaries": [], "completed": 0}

    def bank_init(bank, *positional, **keywords):
        original_bank_init(bank, *positional, **keywords)
        if bank.num_classes == int(cfg.get("num_classes", 1000)):
            banks.append(bank)

    def synchronize():
        torch.cuda.synchronize(device)

    def finish_window():
        if state["started"] is not None and state["ended"] is None:
            synchronize()
            state["ended"] = time.perf_counter()

    def set_step(logger, step):
        original_set_step(logger, step)
        if step == args.warmup:
            synchronize()
            dist.barrier()
            synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            state["started"] = time.perf_counter()
        if step >= args.warmup:
            state["boundaries"].append(time.perf_counter())
            state["completed"] += 1

    def finish(logger):
        finish_window()
        original_finish(logger)

    trainer.Logger.set_step = set_step
    trainer.Logger.finish = finish
    trainer.ArrayMemoryBank.__init__ = bank_init
    try:
        trainer.train_gen(cfg, str(workdir), rank, world_size, device)
        finish_window()
    finally:
        trainer.Logger.set_step = original_set_step
        trainer.Logger.finish = original_finish
        trainer.ArrayMemoryBank.__init__ = original_bank_init
    if state["completed"] != args.steps or state["ended"] is None:
        raise RuntimeError(f"Incomplete benchmark: {state}")
    elapsed = state["ended"] - state["started"]
    durations = [b - a for a, b in zip(
        state["boundaries"], state["boundaries"][1:] + [state["ended"]],
    )]
    local = {
        "rank": rank,
        "elapsed_seconds": elapsed,
        "seconds_per_iteration": elapsed / args.steps,
        "host_boundary_iteration_median_seconds": statistics.median(durations),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "gpu": torch.cuda.get_device_name(device),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "runtime_metadata": runtime_metadata,
        "positive_banks": [
            {"class_count_min": int(bank.count.min()),
             "class_count_median": float(statistics.median(bank.count.tolist())),
             "class_count_max": int(bank.count.max()),
             "classes_with_64_distinct_reals": int((bank.count >= 64).sum()),
             "backend": "zstd_delta" if isinstance(bank, trainer.CompressedPixelMemoryBank) else "dense",
             "payload_gib": (bank.payload_bytes if isinstance(bank, trainer.CompressedPixelMemoryBank)
                             else bank.bank.nbytes) / 1024**3}
            for bank in banks
        ],
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank == 0:
        window = max(item["elapsed_seconds"] for item in gathered)
        rows = []
        log_path = workdir / "train_log.jsonl"
        if log_path.exists():
            rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            rows = [row for row in rows if int(row.get("step", -1)) >= args.warmup]
        metric_names = {key for row in rows for key in row
                        if key in {"time/step", "time/iteration"}
                        or key.startswith("prof_ms/") or key.startswith("adversarial/")}
        metrics = {
            key: statistics.mean(float(row[key]) for row in rows if key in row)
            for key in sorted(metric_names)
        }
        summary = {
            "config": str(Path(args.config).resolve()),
            "config_sha256": config_sha256,
            "source_files_at_launch_sha256": source_hashes,
            "runtime_metadata": runtime_metadata,
            "overrides": overrides,
            "mode": cfg.get("adversarial_mode", "baseline"),
            "adversarial_base_channels": cfg.get("adversarial_base_channels"),
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "world_size": world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "b_g_p_n_per_rank": [4, 32, 64, 32],
            "positive_bank_capacity_per_class": cfg.get("positive_bank_size"),
            "positive_memory_bank_backend": cfg.get("positive_memory_bank_backend", "dense"),
            "bank_init_changed": False,
            "input_push_per_step_per_rank": cfg.get("push_per_step"),
            "log_every_k": cfg.get("log_every_k"),
            "profile_synchronizations_enabled": bool(args.profile),
            "seconds_per_iteration": window / args.steps,
            "generated_images_per_second": args.steps * 4 * 32 * world_size / window,
            "logged_metric_means_after_warmup": metrics,
            "ranks": gathered,
            "limitations": [
                "Fresh real-data banks retain production capacity and sampling; a short run has fewer unique reals per class than a mature training run.",
                "Individual host boundaries are asynchronous; use aggregate seconds_per_iteration for comparison.",
                "Two original experiments remain live; use the same GPU pair, CPU affinity and options for all variants.",
                "Rank 1's endpoint includes train_gen return cleanup; rank 0 ends at Logger.finish before cleanup.",
            ],
        }
        output = Path(args.output).resolve() if args.output else workdir / "benchmark.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print("[mixed-benchmark] " + json.dumps({
            "summary": str(output), "mode": summary["mode"],
            "seconds_per_iteration": summary["seconds_per_iteration"],
            "generated_images_per_second": summary["generated_images_per_second"],
            "peak_allocated_gib": max(item["peak_allocated_gib"] for item in gathered),
        }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
