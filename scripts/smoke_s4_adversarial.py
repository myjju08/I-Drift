#!/usr/bin/env python3
"""Short real-data, two-rank launch/resume check without production W&B writes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_imagenet_gen import load_yaml_config, setup_distributed, train_gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--checkpoint-every", type=int, default=17)
    args = parser.parse_args()
    if args.max_steps < 1 or args.checkpoint_every < 1:
        parser.error("step counts must be positive")
    cfg = load_yaml_config(args.config)
    if cfg.get("adversarial_mode") not in {"raw_gan", "feature_drift", "mixed"}:
        raise ValueError("this smoke harness requires an adversarial experiment")
    # Keep model, optimizer, real data, B/G/P/N and frozen DINO tau unchanged.
    # Bypass only the production-only W&B assertion, not raw dataset validation.
    cfg.update(
        use_wandb=False,
        require_raw_temperature_calibration=False,
        eval_at_start=False,
        eval_per_generated_epochs=0.0,
        eval_per_step=100000000,
        train_max_step_exclusive=args.max_steps,
        save_per_generated_epochs=0.0,
        save_per_step=args.checkpoint_every,
        log_every_k=1,
        profile_train_step=True,
    )
    rank, world_size, device = setup_distributed()
    if world_size != 2:
        raise RuntimeError("launch smoke with torchrun --nproc_per_node=2")
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    print(f"[adversarial-smoke] mode={cfg['adversarial_mode']} rank={rank} max_steps={args.max_steps}", flush=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"[adversarial-smoke] PASS rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled"
    main()
