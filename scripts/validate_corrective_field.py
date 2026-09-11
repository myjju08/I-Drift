#!/usr/bin/env python3
"""Allocated two-rank correctness gate using the actual MAE and latent cache.

The short gate deliberately activates fixed generated H16 replay immediately.
Production retains its epoch-10 activation. Fixed real inputs exercise complete
B8/P32/N32/G32 compute, not mature-bank coverage or sustained input throughput.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json_new(path, payload):
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(encoded)


def validate_step_metrics(metrics, *, variant, step, r1_interval=16):
    required = {"loss", "drift_loss", "g_norm"}
    gan = variant == "replay_double_gan"
    if gan:
        required.update({"adversarial/d_loss_finite", "adversarial/d_grad_finite",
                         "adversarial/g_loss_finite", "adversarial/g_grad_finite",
                         "adversarial/r1_applied", "adversarial/r1_weighted"})
    missing = required.difference(metrics)
    if missing:
        raise AssertionError(f"Missing gate metrics for {variant} step {step}: {sorted(missing)}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise AssertionError(f"Nonfinite {variant} step {step}: {key}={value}")
        if key.endswith("_finite") and value != 1.0:
            raise AssertionError(f"Failed finite check {variant} step {step}: {key}={value}")
    if gan and metrics["adversarial/r1_applied"] != float(step % r1_interval == 0):
        raise AssertionError(f"R1 cadence mismatch at {variant} step {step}")


def module_hash(module):
    import torch
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_rank_agreement(value, description, world_size):
    import torch.distributed as dist
    values = [None] * world_size
    dist.all_gather_object(values, value)
    if any(item != values[0] for item in values[1:]):
        raise AssertionError(f"Ranks disagree on {description}: {values}")


def assert_gradients(module, *, frozen=False):
    import torch
    parameters = list(module.parameters())
    if frozen:
        if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
            raise AssertionError("Frozen model acquired trainable parameters or gradients")
        return
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
        raise AssertionError("Trainable model has missing or nonfinite gradients")


def load_fixed_inputs(cfg, rank, device):
    import numpy as np
    import torch
    from train.latent_data import MmapLatentDataset
    dataset = MmapLatentDataset(cfg["latent_mmap_cache_path"], source_root=cfg["cache_path"])
    targets = np.load(Path(cfg["latent_mmap_cache_path"]) / "labels.npy", mmap_mode="r")
    rng = np.random.default_rng(int(cfg["seed"]) + rank)
    labels = rng.choice(int(cfg["num_classes"]), size=int(cfg["batch_size"]), replace=False)
    positive_rows, negative_rows = [], []
    for label in labels:
        positive_rows.append(rng.choice(np.flatnonzero(targets == label), cfg["pos_per_sample"], replace=False))
        negative_rows.append(rng.choice(np.flatnonzero(targets != label), cfg["neg_per_sample"], replace=False))

    def read_rows(rows):
        return torch.stack([torch.stack([dataset[int(index)][0] for index in group]) for group in rows]).to(device)

    positive, negative = read_rows(positive_rows), read_rows(negative_rows)
    assert positive.dtype == negative.dtype == torch.float32
    assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
    evidence = {"labels": labels.tolist(), "positive_rows": [r.tolist() for r in positive_rows],
                "negative_rows": [r.tolist() for r in negative_rows], "source_rows": len(dataset)}
    return torch.tensor(labels, device=device, dtype=torch.long), positive, negative, evidence


def fixed_generated_history(generator, labels, cfg, device):
    import torch
    count = int(cfg["historical_gen_replay_count"])
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []), torch.no_grad():
        torch.manual_seed(int(cfg["seed"]) + 1719)
        expanded = labels[:, None].expand(-1, count).reshape(-1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(cfg["use_bf16"])):
            samples = generator(expanded, cfg_scale=torch.full_like(expanded, 2, dtype=torch.float32),
                                train=True)["samples"]
    dtype = {"float16": torch.float16, "float32": torch.float32}[cfg["historical_gen_replay_storage_dtype"]]
    return samples.detach().reshape(len(labels), count, 4, 32, 32).to(dtype=dtype).clone()


def run_arm(trainer, cfg, variant, feature, inputs, *, rank, world_size, device, steps):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    cfg = copy.deepcopy(cfg)
    cfg.update(use_wandb=False, require_wandb=False, log_every_k=1)
    # Keep optimizer/geometry/precision/diagnostic arithmetic from production.
    torch.manual_seed(int(cfg["seed"]) + rank)
    np.random.seed((int(cfg["seed"]) + rank) % (2**32))
    trainer._configure_level4_cuda_runtime(cfg, int(cfg["throughput_opt_level"]), device)
    raw = trainer.build_ditgen_from_config(cfg["_raw"]["model"], cfg["_raw"]["dataset"]).to(device)
    generator = DDP(raw, device_ids=[device.index], find_unused_parameters=False,
                    static_graph=True, gradient_as_bucket_view=True, broadcast_buffers=False)
    initial_hash = module_hash(raw)
    assert_rank_agreement(initial_hash, f"{variant} initial generator", world_size)
    optimizer = trainer.build_optimizer(raw, cfg)
    ema = trainer.EMA(raw, decay=float(cfg["ema_decay"]), foreach=True)
    adversarial = trainer.build_adversarial_system(cfg, device)
    labels, positive, negative = inputs
    history = fixed_generated_history(raw, labels, cfg, device) if cfg["historical_gen_replay"] else None
    history_before = None if history is None else history.clone()
    if history is not None and (history.requires_grad or history.grad_fn is not None):
        raise AssertionError("Historical samples must be detached")
    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations, rows = [], []
    preview = None
    for step in range(steps):
        trainer.set_lr(optimizer, trainer.get_lr(step, int(cfg["warmup_steps"]), float(cfg["lr"]),
                                               float(cfg.get("warmup_init_lr", 1e-6))))
        started = time.perf_counter()
        loss, metrics, extras = trainer.train_step(
            generator, feature, optimizer, labels, positive, negative, device, step, cfg,
            historical_samples=history, adversarial_system=adversarial,
        )
        ema.update(raw)
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        validate_step_metrics(metrics, variant=variant, step=step,
                              r1_interval=int(cfg["adversarial_r1_interval"]))
        if not bool(torch.isfinite(loss.detach()).all()):
            raise AssertionError("Nonfinite returned generator loss")
        assert_gradients(raw)
        assert_gradients(feature, frozen=True)
        if history is not None and (history.grad is not None or not torch.equal(history, history_before)):
            raise AssertionError("Training modified or differentiated historical samples")
        if adversarial is not None:
            assert_gradients(adversarial.online)
            assert_gradients(adversarial.target, frozen=True)
        if step in (0, 16, steps - 1):
            assert_rank_agreement(module_hash(raw), f"{variant} step {step} generator", world_size)
            if adversarial is not None:
                assert_rank_agreement(module_hash(adversarial.online), f"{variant} step {step} discriminator", world_size)
                assert_rank_agreement(module_hash(adversarial.target), f"{variant} step {step} target", world_size)
        rows.append({"step": step, "seconds": duration, **metrics})
        durations.append(duration)
        preview = extras["gen_samples_detached"][:2].float().cpu()
        del extras, loss
        if rank == 0:
            print(f"[validation] {variant} step={step} loss={metrics['loss']:.6g} seconds={duration:.3f}", flush=True)
    result = {"variant": variant, "rank": rank, "steps": steps,
              "initial_generator_sha256": initial_hash, "final_generator_sha256": module_hash(raw),
              "seconds": sum(durations), "seconds_per_step": statistics.mean(durations),
              "seconds_per_step_after_first": statistics.mean(durations[1:]),
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
              "history_count": 0 if history is None else history.shape[1], "metrics": rows}
    if result["initial_generator_sha256"] == result["final_generator_sha256"]:
        raise AssertionError(f"Generator did not update for {variant}")
    del generator, raw, optimizer, ema, adversarial, history, history_before
    gc.collect()
    torch.cuda.empty_cache()
    return result, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, default=ROOT / "configs/corrective_field")
    parser.add_argument("--steps", type=int, default=18)
    args = parser.parse_args()
    if args.steps < 18:
        parser.error("At least 18 steps are required to cover both R1 steps 0 and 16")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("WORLD_SIZE") != "2":
        raise RuntimeError("Run only inside a two-GPU Slurm allocation using two-rank torchrun")
    if os.environ.get("WANDB_MODE") != "disabled":
        raise RuntimeError("The bounded validation must use WANDB_MODE=disabled")
    from scripts.preflight_corrective_field import validate_snapshot, validate_suite, validate_assets, sha256, VARIANTS
    manifest = validate_snapshot(ROOT, args.snapshot_root / "source-manifest.json")
    validate_suite(args.suite_dir)
    # Snapshot validation precedes importing the training implementation.
    import torch
    import torch.distributed as dist
    import train_imagenet_gen as trainer
    from train.latent_decoder import LatentDecoderPostprocessor
    from train.evaluation_runtime import run_rank_zero_evaluation
    rank, world_size, device = trainer.setup_distributed()
    if world_size != 2 or device.type != "cuda" or torch.cuda.device_count() != 2:
        raise RuntimeError("The validation requires exactly two allocated CUDA devices")
    if "3090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Expected an allocated srv02 RTX3090")
    configs = {name: trainer.load_yaml_config(str(args.suite_dir / f"{name}.yaml")) for name in VARIANTS}
    cfg = configs["baseline"]
    # The same assertions are inexpensive enough on local SSD to check per rank.
    assets = validate_assets(cfg)
    args.workdir.mkdir(parents=True, exist_ok=True)
    if ((args.workdir / "validation-success.json").exists()
            or (args.snapshot_root / "validation-success.json").exists()):
        raise FileExistsError("Use a fresh validation workdir")
    labels, positive, negative, input_evidence = load_fixed_inputs(cfg, rank, device)
    with torch.random.fork_rng(devices=[device.index]):
        feature = trainer.load_feature_extractor(cfg, device)
    frozen_hash = module_hash(feature)
    assert_rank_agreement(frozen_hash, "frozen MAE encoder", world_size)
    outcomes, previews = [], {}
    for variant in VARIANTS:
        outcome, preview = run_arm(trainer, configs[variant], variant, feature,
                                   (labels, positive, negative), rank=rank,
                                   world_size=world_size, device=device, steps=args.steps)
        if module_hash(feature) != frozen_hash:
            raise AssertionError(f"Frozen MAE weights changed in {variant}")
        outcomes.append(outcome)
        previews[variant] = preview
        dist.barrier()
    if len({item["initial_generator_sha256"] for item in outcomes}) != 1:
        raise AssertionError("Arms did not start from identical generator parameters")

    decoder_report = {}
    def decode_probe():
        feature.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        decoder = LatentDecoderPostprocessor(model_path=cfg["latent_decoder_path"],
                                             scaling_factor=cfg["latent_scaling_factor"])
        try:
            sources = {"source": positive[0, :2], **previews}
            for name, values in sources.items():
                pixels = decoder(values.to(device))
                if tuple(pixels.shape) != (2, 3, 256, 256) or not bool(torch.isfinite(pixels).all()):
                    raise AssertionError(f"Decoder failed finite RGB256 output for {name}")
                decoder_report[name] = {"shape": list(pixels.shape), "finite": True,
                                        "min": float(pixels.min()), "max": float(pixels.max())}
                del pixels
        finally:
            decoder.release()
        if decoder._device != torch.device("cpu"):
            raise AssertionError("Evaluation decoder did not return to CPU")

    run_rank_zero_evaluation(decode_probe, rank=rank, world_size=world_size, device=device,
                            step=args.steps, require_eval_metrics=True, preserve_rng_during_eval=True)
    local = {"rank": rank, "gpu": torch.cuda.get_device_name(device),
             "input_evidence": input_evidence, "variants": outcomes}
    ranks = [None] * world_size
    dist.all_gather_object(ranks, local)
    if rank == 0:
        report = {
            "kind": "corrective_field_gpu_validation", "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "commit": manifest["commit"], "suite_id": manifest["suite_id"],
            "source_manifest_sha256": sha256(args.snapshot_root / "source-manifest.json"),
            "source_files_sha256": manifest["files_sha256"],
            "config_sha256": {name: sha256(args.suite_dir / f"{name}.yaml") for name in VARIANTS},
            "assets": assets, "world_size": world_size, "steps_per_variant": args.steps,
            "geometry_per_rank": {"B": 8, "P": 32, "N": 32, "G": 32, "H_if_replay": 16},
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "workdir": str(args.workdir),
            "smoke_overrides": {"wandb_mode": "disabled", "log_every_k": 1,
                                "replay_activation": "fixed generated snapshot at step0"},
            "decoder": decoder_report, "ranks": ranks,
            "limitations": ["Fixed actual real rows; no mature-bank/input-throughput claim.",
                            "Forced H16 activation tests replay code; production begins replay at epoch10.",
                            "Decoder correctness checked; production performs its full 1024-image step0 FID/IS."],
        }
        write_json_new(args.workdir / "validation-success.json", report)
        write_json_new(args.snapshot_root / "validation-success.json", report)
        print(f"[validation] PASS report={args.snapshot_root / 'validation-success.json'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
