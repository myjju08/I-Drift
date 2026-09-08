#!/usr/bin/env python3
"""Run a short, evaluation-free S4 throughput measurement under torchrun."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train_imagenet_gen import load_yaml_config, setup_distributed, train_gen  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--throughput-opt-level", type=int, default=3)
    parser.add_argument("--diagnostics-every-k", type=int, default=10)
    parser.add_argument("--feature-real-microbatch-size", type=int, default=None)
    parser.add_argument("--feature-generated-microbatch-size", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--raw-train-uint8",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--use-sdpa", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    if hasattr(torch.backends.cudnn, "benchmark_limit"):
        torch.backends.cudnn.benchmark_limit = 0 if args.cudnn_benchmark else 10
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    cfg = load_yaml_config(args.config)
    cfg["batch_size"] = int(args.batch_size)
    cfg["throughput_opt_level"] = int(args.throughput_opt_level)
    cfg["allow_tf32"] = bool(args.allow_tf32)
    cfg["cudnn_benchmark"] = bool(args.cudnn_benchmark)
    cfg["cudnn_benchmark_limit"] = 0 if args.cudnn_benchmark else 10
    cfg["train_max_step_exclusive"] = int(args.steps)
    cfg["eval_at_start"] = False
    cfg["use_wandb"] = False
    # Production runs remain fail-closed on the calibrated raw-input guard.
    # This evaluation-free harness only changes runtime knobs and therefore
    # disables the production-only W&B invariant while benchmarking.
    cfg["require_raw_temperature_calibration"] = False
    cfg["log_every_k"] = 10
    cfg["diagnostics_every_k"] = int(args.diagnostics_every_k)
    cfg["profile_train_step"] = False
    if args.feature_real_microbatch_size is not None:
        cfg["feature_real_microbatch_size"] = int(
            args.feature_real_microbatch_size
        )
    if args.feature_generated_microbatch_size is not None:
        cfg["feature_generated_microbatch_size"] = int(
            args.feature_generated_microbatch_size
        )
    if args.prefetch_factor is not None:
        cfg["prefetch_factor"] = int(args.prefetch_factor)
        cfg["eval_prefetch_factor"] = int(args.prefetch_factor)
    if args.raw_train_uint8 is not None:
        cfg["raw_train_uint8"] = bool(args.raw_train_uint8)
    if args.pin_memory is not None:
        cfg["pin_memory"] = bool(args.pin_memory)
    cfg["_raw"].setdefault("model", {})["use_sdpa"] = bool(args.use_sdpa)

    rank, world_size, device = setup_distributed()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_gen(cfg, args.workdir, rank, world_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_allocated = peak_reserved = 0.0
    print(
        "[benchmark-summary] "
        f"rank={rank} world={world_size} batch={args.batch_size} "
        f"opt={args.throughput_opt_level} sdpa={args.use_sdpa} "
        f"real_mb={cfg.get('feature_real_microbatch_size', 'default')} "
        f"generated_mb={cfg.get('feature_generated_microbatch_size', 'default')} "
        f"raw_train_uint8={cfg.get('raw_train_uint8', False)} "
        f"prefetch={cfg.get('prefetch_factor', 2)} "
        f"tf32={args.allow_tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"peak_allocated_gib={peak_allocated:.3f} "
        f"peak_reserved_gib={peak_reserved:.3f}",
        flush=True,
    )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Prevent benchmark runs from attaching to the production W&B runs even if
    # the parent shell has a different default.
    os.environ.setdefault("WANDB_MODE", "disabled")
    main()
