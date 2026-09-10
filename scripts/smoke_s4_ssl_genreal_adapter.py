#!/usr/bin/env python3
"""Two-step launch smoke for the S4 DINO/MoCo generated-real adapter path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_imagenet_gen import (  # noqa: E402
    is_main_process,
    load_yaml_config,
    setup_distributed,
    train_gen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if cfg.get("feature_adapter_objective") != "gen_real_multipos_infonce":
        raise ValueError("smoke requires the generated-real adapter objective")
    # A two-step smoke cannot fill 64 examples for every ImageNet class. Bypass
    # only the production bank-readiness latch so the adapter/DDP branch runs.
    cfg["feature_adapter_require_unique_reals"] = False
    cfg["feature_adapter_start_step"] = 0
    cfg["train_max_step_exclusive"] = int(args.max_steps)
    cfg["eval_at_start"] = False
    cfg["log_every_k"] = 1
    cfg["profile_train_step"] = True

    rank, world_size, device = setup_distributed()
    if is_main_process(rank):
        print(
            "[adapter-smoke] forcing bank readiness only for launch validation; "
            f"world_size={world_size} max_steps={args.max_steps}",
            flush=True,
        )
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
