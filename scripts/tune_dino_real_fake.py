#!/usr/bin/env python3
"""Two-rank offline DINO tuning on fixed, class-matched real/fake arrays.

Manifest schema 1 has train/validation dictionaries containing real_images,
real_labels, fake_images, fake_labels NPY paths. Images are NCHW uint8 or raw
RGB floating tensors in [-1,1]. Optional validation.raw_fake_images measures
the gap to unquantized generator outputs. No generator is instantiated here.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dino_rf_tuning import DinoRealFakeTuner, TRAINABLE_BLOCKS, atomic_torch_save, module_fingerprint


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_images(value):
    tensor = torch.from_numpy(np.array(value, copy=True))
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0).sub_(0.5).div_(0.5)
    return tensor.float()


class MatchedPairDataset(Dataset):
    """Map each fixed fake to a real example of the same class, without caches."""
    def __init__(self, manifest_path, split="train", *, raw_fake=False):
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported tuning manifest schema")
        entry = manifest[split]
        def path(key):
            value = Path(entry[key])
            return value if value.is_absolute() else self.manifest_path.parent / value
        self.real = np.load(path("real_images"), mmap_mode="r", allow_pickle=False)
        self.fake = np.load(path("raw_fake_images" if raw_fake else "fake_images"), mmap_mode="r", allow_pickle=False)
        self.real_labels = np.load(path("real_labels"), mmap_mode="r", allow_pickle=False)
        self.fake_labels = np.load(path("fake_labels"), mmap_mode="r", allow_pickle=False)
        for name, images, labels in (("real", self.real, self.real_labels), ("fake", self.fake, self.fake_labels)):
            if images.ndim != 4 or images.shape[1] != 3 or labels.shape != (images.shape[0],):
                raise ValueError(f"Invalid {name} NCHW images/labels")
            if images.dtype not in (np.uint8, np.float16, np.float32) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"Unsupported {name} image/label dtype")
            if not len(labels) or np.min(labels) < 0:
                raise ValueError("Tuning arrays must be nonempty with nonnegative labels")
        if self.real.shape[1:] != self.fake.shape[1:]:
            raise ValueError("Real/fake image geometry differs")
        labels = np.asarray(self.real_labels)
        real_indices = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
        counters = {}
        pairs = []
        for label in self.fake_labels:
            label = int(label)
            if label not in real_indices:
                raise ValueError(f"No real examples for fake class {label}")
            offset = counters.get(label, 0)
            pairs.append(int(real_indices[label][offset % len(real_indices[label])]))
            counters[label] = offset + 1
        self.real_indices = np.asarray(pairs, dtype=np.int64)

    def __len__(self):
        return len(self.fake_labels)

    def __getitem__(self, index):
        return decode_images(self.real[self.real_indices[index]]), decode_images(self.fake[index]), int(self.fake_labels[index])


def distributed():
    return dist.is_available() and dist.is_initialized()


def all_finite(value):
    flag = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def reduce_metrics(total, count):
    keys = sorted(total)
    packed = torch.stack([total[key].double() for key in keys] + [count.double()])
    if distributed():
        dist.all_reduce(packed)
    denominator = packed[-1].clamp_min(1.0)
    return dict(zip(keys, (packed[:-1] / denominator).cpu().tolist()))


def amp_context(device, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and device.type == "cuda" else contextlib.nullcontext()


def selected_indices(length, limit):
    if int(limit) <= 0 or int(limit) >= length:
        return list(range(length))
    return np.linspace(0, length - 1, int(limit), dtype=np.int64).tolist()


def real_fake_auc(real_scores, fake_scores):
    """Exact Mann-Whitney AUC, assigning average ranks to tied scores."""
    real_scores, fake_scores = np.asarray(real_scores), np.asarray(fake_scores)
    scores = np.concatenate((real_scores, fake_scores))
    if not len(real_scores) or not len(fake_scores) or not np.isfinite(scores).all():
        raise ValueError("AUC requires finite scores from both domains")
    order = np.argsort(scores, kind="stable")
    ranked = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    positive = len(real_scores)
    return float((ranks[:positive].sum() - positive * (positive + 1) / 2) / (positive * len(fake_scores)))


def make_optimizer(model, args, *, head_only=False):
    groups = [{"params": list(model.heads.parameters()), "lr": args.head_lr}]
    if not head_only:
        groups.insert(0, {"params": [p for p in model.student.parameters() if p.requires_grad], "lr": args.backbone_lr})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)


def next_batch(loader, sampler, state):
    if state.get("iterator") is None:
        sampler.set_epoch(state["epoch"])
        state["iterator"] = iter(loader)
    try:
        return next(state["iterator"])
    except StopIteration:
        state["epoch"] += 1
        sampler.set_epoch(state["epoch"])
        state["iterator"] = iter(loader)
        return next(state["iterator"])


def train_update(wrapper, model, optimizer, batch, args, device, *, head_only=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    real, fake, labels = [value.to(device, non_blocking=True) for value in batch]
    if not all_finite(real) or not all_finite(fake):
        raise FloatingPointError("Non-finite offline tuning inputs")
    count = real.shape[0]
    totals = {}
    for start in range(0, count, args.microbatch_pairs):
        end = min(start + args.microbatch_pairs, count)
        context = wrapper.no_sync() if isinstance(wrapper, DDP) and end < count else contextlib.nullcontext()
        with context, amp_context(device, args.bf16):
            output = wrapper(real[start:end], fake[start:end], labels[start:end],
                             preservation_weight=args.preservation_weight, head_only=head_only)
            loss = output["loss"] * ((end - start) / count)
            if not all_finite(loss):
                raise FloatingPointError("Non-finite tuning loss on at least one rank")
            loss.backward()
        for key, value in output.items():
            totals[key] = totals.get(key, torch.zeros((), device=device)) + value.detach() * (end - start)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    if not all_finite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite tuning gradients on at least one rank")
    optimizer.step()
    totals["gradient_norm"] = norm.detach() * count
    return reduce_metrics(totals, torch.tensor(float(count), device=device))


@torch.no_grad()
def validate(model, dataset, args, device, rank, world):
    model.eval()
    # Exact striding avoids sampler padding/duplicated validation examples.
    selected = selected_indices(len(dataset), args.validation_pairs)
    indices = selected[rank::world]
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=args.validation_microbatch_pairs or args.microbatch_pairs,
                        num_workers=0, shuffle=False)
    total, count = {}, torch.zeros((), device=device)
    real_scores, fake_scores, class_labels = [], [], []
    for batch in loader:
        real, fake, labels = [value.to(device) for value in batch]
        with amp_context(device, args.bf16):
            output = model(real, fake, labels, preservation_weight=args.preservation_weight, return_scores=True)
        real_scores.append(output.pop("real_scores").cpu().numpy())
        fake_scores.append(output.pop("fake_scores").cpu().numpy())
        class_labels.append(labels.cpu().numpy())
        for key, value in output.items():
            total[key] = total.get(key, torch.zeros((), device=device)) + value.detach() * len(labels)
        count += len(labels)
    if not total:
        raise ValueError("Validation split needs at least one pair per rank")
    metrics = reduce_metrics(total, count)
    local_scores = (np.concatenate(real_scores), np.concatenate(fake_scores), np.concatenate(class_labels))
    score_parts = [None] * world if distributed() else [local_scores]
    if distributed():
        dist.all_gather_object(score_parts, local_scores)
    real_scores, fake_scores, labels = (np.concatenate([part[index] for part in score_parts]) for index in range(3))
    metrics["roc_auc"] = real_fake_auc(real_scores, fake_scores)
    metrics["same_class_roc_auc"] = float(np.mean([
        real_fake_auc(real_scores[labels == label], fake_scores[labels == label])
        for label in np.unique(labels)
    ]))
    metrics["pairs_evaluated"] = len(selected)
    metrics["classes_evaluated"] = len(np.unique(labels))
    return metrics


@torch.no_grad()
def calibrate(model, dataset, args, device, rank, world):
    # Read unique original TRAIN-real rows directly, never validation or fake.
    indices = selected_indices(len(dataset.real), args.calibration_pairs)[rank::world]
    sums = torch.zeros(len(model.feature_keys), dtype=torch.float64, device=device)
    counts = torch.zeros_like(sums)
    for start in range(0, len(indices), args.microbatch_pairs):
        selected = indices[start:start + args.microbatch_pairs]
        real = decode_images(dataset.real[selected]).to(device)
        with amp_context(device, args.bf16):
            batch_sums, batch_counts = model.scale_statistics(real)
        sums += batch_sums
        counts += batch_counts
    if distributed():
        dist.all_reduce(sums)
        dist.all_reduce(counts)
    model.set_scales(sums, counts)


def rng_state():
    return {"torch": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.random.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def publish_checkpoint(model, optimizer, workdir, step, metadata, head_probe, best_selection, last_validation,
                       *, is_best=False):
    rank = dist.get_rank() if distributed() else 0
    states = [None] * dist.get_world_size() if distributed() else [rng_state()]
    if distributed():
        dist.all_gather_object(states, rng_state())
    if rank == 0:
        payload = {"schema_version": 1, "step": step, "metadata": metadata,
                   "tuning": model.tuning_state_dict(), "optimizer": optimizer.state_dict(),
                   "head_probe": head_probe, "best_selection": best_selection,
                   "last_validation": last_validation, "rank_rng": states}
        latest = workdir / "tuning_latest.pt"
        atomic_torch_save(payload, latest)
        if is_best:
            temporary = workdir / ".best-link"
            temporary.unlink(missing_ok=True)
            os.link(latest, temporary)
            os.replace(temporary, workdir / "tuning_best.pt")
            model.export_backbone(workdir / "dino_tuned.pth")
    if distributed():
        dist.barrier()


def reconcile_selected_export(workdir, latest_state):
    """Repair a preemption between committing latest and its selected export."""
    selection = latest_state.get("best_selection")
    if selection is None:
        return
    best = workdir / "tuning_best.pt"
    if int(selection["step"]) == int(latest_state["step"]):
        temporary = workdir / ".best-link"
        temporary.unlink(missing_ok=True)
        os.link(workdir / "tuning_latest.pt", temporary)
        os.replace(temporary, best)
        best_state = latest_state
    else:
        if not best.is_file():
            raise FileNotFoundError("Recorded best tuning checkpoint is missing")
        best_state = torch.load(best, map_location="cpu", weights_only=False)
        if best_state.get("metadata") != latest_state.get("metadata") or int(best_state["step"]) != int(selection["step"]):
            raise ValueError("Best tuning checkpoint does not match the recorded selection")
    atomic_torch_save({key: value.detach().cpu() for key, value in best_state["tuning"]["student"].items()}, workdir / "dino_tuned.pth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--head-probe-steps", "--probe-steps", dest="head_probe_steps", type=int, default=200)
    parser.add_argument("--pairs-per-rank", type=int, default=16)
    parser.add_argument("--microbatch-pairs", type=int, default=4)
    parser.add_argument("--calibration-pairs", type=int, default=0, help="Limit unique TRAIN reals for smoke; 0 uses all.")
    parser.add_argument("--validation-pairs", type=int, default=0, help="Limit heldout pairs for smoke; 0 uses all.")
    parser.add_argument("--validation-microbatch-pairs", "--validation-microbatch", dest="validation_microbatch_pairs", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--preservation-weight", type=float, default=10.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.pairs_per_rank, args.microbatch_pairs, args.validate_every, args.save_every) < 1 or args.head_probe_steps < 0:
        parser.error("Step counts and batch sizes must be positive; probe steps may be zero")
    if min(args.calibration_pairs, args.validation_pairs, args.validation_microbatch_pairs, args.num_workers) < 0:
        parser.error("Calibration/validation limits and worker counts must be nonnegative")
    if args.validation_pairs == 1:
        parser.error("Two-rank validation requires at least two pairs")
    for name in ("backbone_lr", "head_lr", "preservation_weight", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    world, rank, local_rank = (int(os.environ.get(key, default)) for key, default in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if world != 2 or not torch.cuda.is_available():
        raise RuntimeError("Run offline tuning with torchrun --nproc_per_node=2 on two GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(args.teacher_checkpoint).resolve()
    if teacher_path.parent == workdir:
        raise ValueError("Keep the original teacher checkpoint outside the tuning output directory")
    for name in ("dino_tuned.pth", "tuning_latest.pt", "tuning_best.pt"):
        candidate = workdir / name
        if candidate.resolve() == teacher_path or (candidate.exists() and os.path.samefile(candidate, teacher_path)):
            raise ValueError("A tuning output aliases the original teacher checkpoint")
    latest = workdir / "tuning_latest.pt"
    existing = [name for name in ("tuning_latest.pt", "tuning_best.pt", "dino_tuned.pth", "tuning_log.jsonl", "tuning_summary.json") if (workdir / name).exists()]
    if existing and not args.resume:
        raise FileExistsError(f"Existing tuning artifacts {existing}: use --resume or a fresh workdir")
    if args.resume and not latest.is_file():
        raise FileNotFoundError("--resume requires tuning_latest.pt")
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest.get("generator_sha256") or manifest.get("generator_weight_key") not in {"model", "ema"}:
        raise ValueError("Manifest must identify the fixed generator checkpoint and weight key")
    generator_checkpoint = Path(manifest["generator_checkpoint"])
    if sha256_file(generator_checkpoint) != manifest["generator_sha256"]:
        raise ValueError("Fixed generator checkpoint differs from manifest SHA256")
    settings = {key: value for key, value in vars(args).items() if key not in {"resume", "workdir", "steps", "save_every", "validate_every"}}
    metadata = {"teacher_sha256": sha256_file(teacher_path), "manifest_sha256": sha256_file(args.manifest),
                "generator_sha256": manifest["generator_sha256"], "generator_weight_key": manifest["generator_weight_key"],
                "settings": settings, "world_size": world, "trainable_blocks": list(TRAINABLE_BLOCKS),
                "source_sha256": {name: sha256_file(ROOT / name) for name in ("models/dino_rf_tuning.py", "scripts/tune_dino_real_fake.py")},
                "preservation": "fixed original TRAIN-real teacher RMS; no student normalization",
                "export_selection": "minimum heldout BCE + preservation_weight * heldout preservation"}
    train_set = MatchedPairDataset(args.manifest)
    validation_set = MatchedPairDataset(args.manifest, "validation")
    raw_validation_set = MatchedPairDataset(args.manifest, "validation", raw_fake=True) if manifest["validation"].get("raw_fake_images") else None
    if int(max(np.max(train_set.fake_labels), np.max(validation_set.fake_labels))) >= args.num_classes:
        raise ValueError("Dataset labels exceed --num-classes")
    model = DinoRealFakeTuner.from_checkpoint(teacher_path, num_classes=args.num_classes, head_seed=args.seed).to(device)
    metadata["teacher_state_fingerprint"] = module_fingerprint(model.teacher)
    model.train()
    wrapper = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(train_set, batch_size=args.pairs_per_rank, sampler=sampler,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    if not len(loader):
        raise ValueError("Training split is too small for configured pairs per rank")
    log_path = workdir / "tuning_log.jsonl"
    def log(record):
        if rank == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    optimizer = make_optimizer(model, args)
    start_step, best_selection, last_validation, head_probe, resumed_rng = 0, None, None, None, None
    if args.resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("schema_version") != 1 or state.get("metadata") != metadata:
            raise ValueError("Resume metadata/configuration differs from the saved tuning experiment")
        model.load_tuning_state_dict(state["tuning"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_selection, last_validation, head_probe = int(state["step"]), state["best_selection"], state["last_validation"], state["head_probe"]
        resumed_rng = state["rank_rng"][rank]
        if rank == 0:
            reconcile_selected_export(workdir, state)
        dist.barrier()
        del state
    else:
        calibrate(model, train_set, args, device, rank, world)
        initial_heads = copy.deepcopy(model.heads.state_dict())
        initial_validation = {"phase": "initial_validation", "quantized": validate(model, validation_set, args, device, rank, world)}
        if raw_validation_set is not None:
            initial_validation["raw_float"] = validate(model, raw_validation_set, args, device, rank, world)
        log(initial_validation)
        if args.head_probe_steps:
            probe_optimizer = make_optimizer(model, args, head_only=True)
            probe_data = {"epoch": 0, "iterator": None}
            for step in range(args.head_probe_steps):
                metrics = train_update(wrapper, model, probe_optimizer, next_batch(loader, sampler, probe_data), args, device, head_only=True)
                if (step + 1) % 50 == 0:
                    log({"phase": "head_probe", "step": step + 1, **metrics})
            head_probe = {"quantized": validate(model, validation_set, args, device, rank, world)}
            if raw_validation_set is not None:
                head_probe["raw_float"] = validate(model, raw_validation_set, args, device, rank, world)
            log({"phase": "head_probe_validation", **head_probe})
            del probe_optimizer
        model.heads.load_state_dict(initial_heads)
        del initial_heads
        optimizer = make_optimizer(model, args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
    data_state = {"epoch": start_step // len(loader), "iterator": None}
    # Deterministic sampler and no stochastic augmentation: skip only index rows.
    sampler.set_epoch(data_state["epoch"])
    data_state["iterator"] = iter(loader)
    for _ in range(start_step % len(loader)):
        next(data_state["iterator"])
    if resumed_rng is not None:
        restore_rng(resumed_rng)
    log({"phase": "tuning_start", "step": start_step, "metadata": metadata,
         "backbone_trainable": sum(p.numel() for p in model.student.parameters() if p.requires_grad),
         "head_trainable": sum(p.numel() for p in model.heads.parameters())})
    for step in range(start_step, args.steps):
        before = time.monotonic()
        metrics = train_update(wrapper, model, optimizer, next_batch(loader, sampler, data_state), args, device)
        if (step + 1) % 10 == 0:
            log({"phase": "tuning", "step": step + 1, "seconds": time.monotonic() - before, **metrics})
        is_best = False
        if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
            validation = validate(model, validation_set, args, device, rank, world)
            record = {"phase": "validation", "step": step + 1, "quantized": validation}
            if raw_validation_set is not None:
                record["raw_float"] = validate(model, raw_validation_set, args, device, rank, world)
            log(record)
            last_validation = record
            if not math.isfinite(validation["loss"]):
                raise FloatingPointError("Non-finite heldout tuning objective")
            is_best = best_selection is None or validation["loss"] < best_selection["quantized"]["loss"]
            if is_best:
                best_selection = record
        if is_best or (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            publish_checkpoint(model, optimizer, workdir, step + 1, metadata, head_probe, best_selection, last_validation, is_best=is_best)
    freeze_report = model.frozen_state_report()
    teacher_state_unchanged = module_fingerprint(model.teacher) == metadata["teacher_state_fingerprint"]
    integrity = torch.tensor(int(freeze_report["all_frozen_tensors_unchanged"] and teacher_state_unchanged), device=device)
    dist.all_reduce(integrity, op=dist.ReduceOp.MIN)
    if not bool(integrity):
        raise RuntimeError("Frozen backbone/BN tensors or in-memory teacher changed during tuning")
    if rank == 0:
        original_unchanged = sha256_file(teacher_path) == metadata["teacher_sha256"]
        if not original_unchanged:
            raise RuntimeError("Original DINO checkpoint hash changed during tuning")
        export = workdir / "dino_tuned.pth"
        summary = {"steps": args.steps, "last_step": max(start_step, args.steps),
                   "best_step": best_selection["step"], "best_snapshot_metrics": best_selection,
                   "last_validation": last_validation, "metadata": metadata,
                   "head_probe": head_probe, "original_teacher_unchanged": original_unchanged,
                   "in_memory_teacher_unchanged": teacher_state_unchanged, "last_student_freeze_audit": freeze_report,
                   "export": str(export), "export_sha256": sha256_file(export)}
        (workdir / "tuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log({"phase": "complete", "export": str(export), "steps": args.steps})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
