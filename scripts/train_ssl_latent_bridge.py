#!/usr/bin/env python3
"""Distill a VAE+SSL-R50 prefix into a latent-to-layer2 bridge.

Example (DINO)::

    python scripts/train_ssl_latent_bridge.py \
      --backbone dino_resnet50 \
      --feature-checkpoint weights/pretrained/dino_resnet50_pretrain.pth \
      --cache-path data/imagenet/latent_cache_256 \
      --output weights/pretrained/dino_r50_latent_bridge.pt

The expensive VAE, stem, layer1 and layer2 are teachers only.  The saved
checkpoint contains just the small bridge.  Drift runtime should load it with
``SSLLatentBridgeFeatureExtractor``, which retains only the official frozen
ResNet layer3/layer4 tail.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ssl_latent_bridge import (  # noqa: E402
    LatentBridgeConfig,
    LatentToStage2Bridge,
    SSLLatentBridgeFeatureExtractor,
    compare_reverse_drift_latent_gradients,
    load_bridge_checkpoint,
    save_bridge_checkpoint,
    stage_map_distillation_metrics,
)
from models.ssl_resnet import SSLResNetFeatureExtractor, canonical_ssl_backbone  # noqa: E402
from models.imagenet_generator import build_ditgen_from_config  # noqa: E402
from train.train_data import create_imagenet_split, infinite_sampler  # noqa: E402


DEFAULT_VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
DEFAULT_VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _build_loader(
    cache_path: str,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
):
    loader, preprocess, _ = create_imagenet_split(
        imagenet_path="",
        resolution=256,
        batch_size=batch_size,
        split=split,
        use_aug=False,
        use_latent=True,
        use_cache=True,
        cache_path=cache_path,
        cache_format="pt_imagefolder",
        num_workers=num_workers,
        pin_memory=True,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    return loader, preprocess


def _batch_latents(batch, preprocess, device: torch.device) -> torch.Tensor:
    values = preprocess(batch)["images"]
    return values.to(device=device, dtype=torch.float32, non_blocking=True)


def _load_coverage_generator(
    config_path: str,
    checkpoint_path: str,
    *,
    seed: int,
    device: torch.device,
) -> Optional[nn.Module]:
    """Load/freeze a B/4 generator used only to cover its latent distribution."""
    if not config_path:
        return None
    import yaml

    with Path(config_path).open("r") as handle:
        raw_config = yaml.safe_load(handle)
    # ``torch.device("cuda")`` has no explicit index in ordinary single-GPU
    # launches (unlike torchrun's ``cuda:<local_rank>``).  fork_rng expects a
    # concrete CUDA device index, so resolve the current visible device here.
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        generator = build_ditgen_from_config(
            raw_config["model"], raw_config["dataset"]
        ).to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise ValueError(f"Unsupported generator checkpoint: {checkpoint_path}")
        generator_state = state.get("ema") or state.get("model") or state.get("state_dict")
        if not isinstance(generator_state, Mapping):
            raise ValueError(
                f"Generator checkpoint has no ema/model/state_dict: {checkpoint_path}"
            )
        generator.load_state_dict(dict(generator_state), strict=True)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator


@torch.no_grad()
def _mix_generator_coverage(
    real_latents: torch.Tensor,
    generator: Optional[nn.Module],
    *,
    fraction: float,
    num_classes: int,
    cfg_min: float,
    cfg_max: float,
    use_bf16: bool,
) -> torch.Tensor:
    if generator is None or fraction <= 0:
        return real_latents
    count = min(
        real_latents.shape[0],
        max(1, int(round(real_latents.shape[0] * float(fraction)))),
    )
    device = real_latents.device
    labels = torch.randint(0, int(num_classes), (count,), device=device)
    cfg = torch.empty(count, device=device).uniform_(float(cfg_min), float(cfg_max))
    with _amp_context(device, use_bf16):
        generated = generator(labels, cfg_scale=cfg, train=False)["samples"]
    mixed = torch.cat(
        [real_latents[: real_latents.shape[0] - count], generated.float()], dim=0
    )
    return mixed[torch.randperm(mixed.shape[0], device=device)]


def teacher_stage_maps(
    teacher: SSLResNetFeatureExtractor,
    latents: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute the exact VAE+R50 layer2/3/4 teacher maps."""
    pixels = teacher._decode_and_normalize(latents)
    backbone = teacher.backbone
    x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(pixels))))
    x = backbone.layer1(x)
    layer2 = backbone.layer2(x)
    layer3 = backbone.layer3(layer2)
    layer4 = backbone.layer4(layer3)
    return {"layer2": layer2, "layer3": layer3, "layer4": layer4}


def student_stage_maps(
    bridge: LatentToStage2Bridge,
    teacher: SSLResNetFeatureExtractor,
    latents: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run the trainable bridge followed by the same frozen teacher tail."""
    layer2 = bridge(latents)
    layer3 = teacher.backbone.layer3(layer2)
    layer4 = teacher.backbone.layer4(layer3)
    return {"layer2": layer2, "layer3": layer3, "layer4": layer4}


@torch.no_grad()
def calibrate_bridge_output_rms(
    bridge: LatentToStage2Bridge,
    teacher: SSLResNetFeatureExtractor,
    latents: torch.Tensor,
    *,
    use_bf16: bool,
    min_ratio: float = 1e-3,
    max_ratio: float = 100.0,
) -> Dict[str, float]:
    """Match initial bridge/teacher layer2 RMS channel by channel.

    DINO and MoCo layer2 units differ by more than an order of magnitude.  A
    positive rescale of each output-projection row commutes exactly with the
    final ReLU, removes this avoidable initialization mismatch, and adds no
    runtime operation.  Statistics are globally reduced so every DDP rank
    applies identical parameter updates.
    """
    device = latents.device
    with _amp_context(device, use_bf16):
        target = teacher_stage_maps(teacher, latents)["layer2"].float()
        predicted = bridge(latents).float()
    target_square_sum = target.square().sum(dim=(0, 2, 3))
    predicted_square_sum = predicted.square().sum(dim=(0, 2, 3))
    count = target.new_tensor(float(target.shape[0] * target.shape[2] * target.shape[3]))
    if dist.is_available() and dist.is_initialized():
        channels = target_square_sum.numel()
        packed = torch.cat(
            [target_square_sum, predicted_square_sum, count.view(1)]
        )
        dist.all_reduce(packed)
        target_square_sum = packed[:channels]
        predicted_square_sum = packed[channels : 2 * channels]
        count = packed[-1]
    target_rms = (target_square_sum / count.clamp_min(1.0)).sqrt()
    predicted_rms = (predicted_square_sum / count.clamp_min(1.0)).sqrt()
    ratio = (target_rms / predicted_rms.clamp_min(1e-8)).clamp(
        min=float(min_ratio), max=float(max_ratio)
    )
    bridge.output_projection.weight.mul_(ratio[:, None, None, None])
    if bridge.output_projection.bias is not None:
        bridge.output_projection.bias.mul_(ratio)
    return {
        "initialization/output_rms_ratio_mean": float(ratio.mean().item()),
        "initialization/output_rms_ratio_min": float(ratio.min().item()),
        "initialization/output_rms_ratio_max": float(ratio.max().item()),
        "initialization/teacher_layer2_rms_mean": float(target_rms.mean().item()),
        "initialization/bridge_layer2_rms_mean_before": float(
            predicted_rms.mean().item()
        ),
    }


def normalized_feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Scale-invariant MSE plus a per-sample direction penalty."""
    pred = prediction.float()
    ref = target.detach().float()
    denominator = ref.square().mean().detach().clamp_min(eps)
    normalized_mse = (pred - ref).square().mean() / denominator
    cosine_loss = 1.0 - F.cosine_similarity(
        pred.flatten(1), ref.flatten(1), dim=1, eps=eps
    ).mean()
    loss = normalized_mse + float(cosine_weight) * cosine_loss
    return loss, {
        "normalized_mse": float(normalized_mse.detach().item()),
        "cosine_loss": float(cosine_loss.detach().item()),
    }


def directional_derivative_distillation_loss(
    predicted_delta: Mapping[str, torch.Tensor],
    target_delta: Mapping[str, torch.Tensor],
    *,
    stage_weights: Mapping[str, float],
    cosine_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match already-computed feature-map directional derivatives."""
    stages = tuple(
        stage
        for stage in ("layer2", "layer3", "layer4")
        if stage_weights.get(stage, 0) > 0
    )
    if not stages:
        raise ValueError("at least one finite-difference stage weight must be positive")

    total = predicted_delta[stages[0]].new_zeros((), dtype=torch.float32)
    metrics: Dict[str, float] = {}
    for stage in stages:
        weight = float(stage_weights[stage])
        predicted_stage_delta = predicted_delta[stage].float()
        target_stage_delta = target_delta[stage].detach().float()
        stage_loss, components = normalized_feature_loss(
            predicted_stage_delta,
            target_stage_delta,
            cosine_weight=cosine_weight,
        )
        total = total + weight * stage_loss

        predicted_rms = predicted_stage_delta.detach().square().mean().sqrt()
        target_rms = target_stage_delta.detach().square().mean().sqrt()
        metrics.update(
            {
                f"jacobian/{stage}_{key}": value
                for key, value in components.items()
            }
        )
        metrics[f"jacobian/{stage}_predicted_delta_rms"] = float(
            predicted_rms.item()
        )
        metrics[f"jacobian/{stage}_target_delta_rms"] = float(target_rms.item())
        metrics[f"jacobian/{stage}_delta_rms_ratio"] = float(
            (predicted_rms / target_rms.clamp_min(1e-12)).item()
        )
    return total, metrics


def finite_difference_distillation_loss(
    predicted_base: Mapping[str, torch.Tensor],
    predicted_perturbed: Mapping[str, torch.Tensor],
    target_base: Mapping[str, torch.Tensor],
    target_perturbed: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
    stage_weights: Mapping[str, float],
    cosine_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match local map directional derivatives without a second-order graph."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    predicted_delta = {
        stage: (predicted_perturbed[stage].float() - value.float()) / float(epsilon)
        for stage, value in predicted_base.items()
    }
    target_delta = {
        stage: (target_perturbed[stage].detach().float() - value.detach().float())
        / float(epsilon)
        for stage, value in target_base.items()
    }
    return directional_derivative_distillation_loss(
        predicted_delta,
        target_delta,
        stage_weights=stage_weights,
        cosine_weight=cosine_weight,
    )


def local_finite_difference_distillation(
    bridge: nn.Module,
    teacher: SSLResNetFeatureExtractor,
    reference_latents: torch.Tensor,
    *,
    count: int,
    epsilon: float,
    stage_weights: Mapping[str, float],
    cosine_weight: float,
    use_bf16: bool,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Distill directions around the B/4 initialization point ``z0 = 0``.

    Independent Gaussian directions are normalized to unit RMS, making every
    endpoint exactly ``epsilon`` RMS from zero. Teacher endpoints are constants,
    so the loss takes only first-order gradients with respect to bridge weights.
    """
    if reference_latents.ndim != 4 or reference_latents.shape[0] == 0:
        raise ValueError("reference_latents must be a non-empty BCHW tensor")
    if count <= 0:
        raise ValueError("count must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    actual_count = min(int(count), int(reference_latents.shape[0]))
    shape = (actual_count, *reference_latents.shape[1:])
    generator = torch.Generator(device=reference_latents.device)
    generator.manual_seed(int(seed))
    direction = torch.randn(
        shape,
        generator=generator,
        device=reference_latents.device,
        dtype=torch.float32,
    )
    direction = direction / direction.square().mean(
        dim=(1, 2, 3), keepdim=True
    ).sqrt().clamp_min(1e-12)
    base = torch.zeros(shape, device=reference_latents.device, dtype=torch.float32)
    perturbed = base + float(epsilon) * direction
    paired = torch.cat((base, perturbed), dim=0)

    device = reference_latents.device
    with torch.no_grad(), _amp_context(device, use_bf16):
        target_paired = teacher_stage_maps(teacher, paired)
    with _amp_context(device, use_bf16):
        predicted_paired = student_stage_maps(bridge, teacher, paired)

    target_base = {
        stage: value[:actual_count] for stage, value in target_paired.items()
    }
    target_perturbed = {
        stage: value[actual_count:] for stage, value in target_paired.items()
    }
    predicted_base = {
        stage: value[:actual_count] for stage, value in predicted_paired.items()
    }
    predicted_perturbed = {
        stage: value[actual_count:] for stage, value in predicted_paired.items()
    }
    loss, metrics = finite_difference_distillation_loss(
        predicted_base,
        predicted_perturbed,
        target_base,
        target_perturbed,
        epsilon=epsilon,
        stage_weights=stage_weights,
        cosine_weight=cosine_weight,
    )
    metrics["jacobian/latent_delta_rms"] = float(
        (perturbed - base).square().mean().sqrt().item()
    )
    metrics["jacobian/count_per_rank"] = float(actual_count)
    return loss, metrics


@dataclass(frozen=True)
class LocalFiniteDifferenceTeacherBank:
    """CPU-resident teacher directional derivatives for a fixed local bank."""

    directions: torch.Tensor
    target_delta: Dict[str, torch.Tensor]
    epsilon: float
    seed: int

    @property
    def size(self) -> int:
        return int(self.directions.shape[0])

    @property
    def storage_bytes(self) -> int:
        tensors = (self.directions, *self.target_delta.values())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@torch.no_grad()
def build_local_finite_difference_teacher_bank(
    teacher: SSLResNetFeatureExtractor,
    reference_latents: torch.Tensor,
    *,
    bank_size: int,
    epsilon: float,
    teacher_chunk_size: int,
    use_bf16: bool,
    seed: int,
) -> LocalFiniteDifferenceTeacherBank:
    """Precompute exact-teacher local derivatives and store them as CPU BF16.

    ``F(0)`` is evaluated once. Perturbed endpoints are evaluated in bounded
    chunks, subtracted in FP32, divided by epsilon, and only then quantized for
    storage. Caching derivatives rather than BF16 endpoints avoids cancellation
    when the two nearby maps are subtracted during every training step.
    """
    if reference_latents.ndim != 4 or reference_latents.shape[0] == 0:
        raise ValueError("reference_latents must be a non-empty BCHW tensor")
    if bank_size <= 0:
        raise ValueError("bank_size must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if teacher_chunk_size <= 0:
        raise ValueError("teacher_chunk_size must be positive")

    latent_shape = tuple(int(value) for value in reference_latents.shape[1:])
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    directions = torch.randn(
        (int(bank_size), *latent_shape),
        generator=cpu_generator,
        device="cpu",
        dtype=torch.float32,
    )
    directions = directions / directions.square().mean(
        dim=(1, 2, 3), keepdim=True
    ).sqrt().clamp_min(1e-12)

    device = reference_latents.device
    base = torch.zeros((1, *latent_shape), device=device, dtype=torch.float32)
    with _amp_context(device, use_bf16):
        target_base = teacher_stage_maps(teacher, base)

    delta_chunks: Dict[str, list[torch.Tensor]] = {
        "layer2": [],
        "layer3": [],
        "layer4": [],
    }
    for start in range(0, int(bank_size), int(teacher_chunk_size)):
        stop = min(start + int(teacher_chunk_size), int(bank_size))
        perturbed = (
            directions[start:stop].to(device=device, dtype=torch.float32)
            * float(epsilon)
        )
        with _amp_context(device, use_bf16):
            target_perturbed = teacher_stage_maps(teacher, perturbed)
        for stage in delta_chunks:
            delta = (
                target_perturbed[stage].float() - target_base[stage].float()
            ) / float(epsilon)
            delta_chunks[stage].append(delta.to(device="cpu", dtype=torch.bfloat16))
        del perturbed, target_perturbed

    target_delta = {
        stage: torch.cat(chunks, dim=0).contiguous()
        for stage, chunks in delta_chunks.items()
    }
    return LocalFiniteDifferenceTeacherBank(
        directions=directions.contiguous(),
        target_delta=target_delta,
        epsilon=float(epsilon),
        seed=int(seed),
    )


def local_finite_difference_distillation_from_bank(
    bridge: nn.Module,
    teacher: SSLResNetFeatureExtractor,
    bank: LocalFiniteDifferenceTeacherBank,
    *,
    count: int,
    application_index: int,
    stage_weights: Mapping[str, float],
    cosine_weight: float,
    use_bf16: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Use a deterministic cyclic slice of a cached local teacher bank."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count > bank.size:
        raise ValueError(f"count={count} exceeds teacher bank size={bank.size}")
    if application_index < 0:
        raise ValueError("application_index must be non-negative")

    start = (int(application_index) * int(count)) % bank.size
    indices = torch.tensor(
        [(start + offset) % bank.size for offset in range(int(count))],
        device="cpu",
        dtype=torch.long,
    )
    device = next(bridge.parameters()).device
    directions = bank.directions.index_select(0, indices).to(
        device=device, dtype=torch.float32
    )
    base = torch.zeros(
        (1, *directions.shape[1:]), device=device, dtype=torch.float32
    )
    perturbed = directions * float(bank.epsilon)
    paired = torch.cat((base, perturbed), dim=0)
    with _amp_context(device, use_bf16):
        predicted_paired = student_stage_maps(bridge, teacher, paired)

    predicted_delta = {
        stage: (value[1:].float() - value[:1].float()) / float(bank.epsilon)
        for stage, value in predicted_paired.items()
    }
    target_delta = {
        stage: value.index_select(0, indices).to(device=device)
        for stage, value in bank.target_delta.items()
    }
    loss, metrics = directional_derivative_distillation_loss(
        predicted_delta,
        target_delta,
        stage_weights=stage_weights,
        cosine_weight=cosine_weight,
    )
    metrics["jacobian/latent_delta_rms"] = float(
        perturbed.square().mean().sqrt().item()
    )
    metrics["jacobian/count_per_rank"] = float(count)
    metrics["jacobian/bank_index_start"] = float(start)
    metrics["jacobian/bank_size"] = float(bank.size)
    return loss, metrics


def channel_whitened_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
    cosine_weight: float,
    std_floor_ratio: float,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Robust layer2 loss after whitening by teacher channel statistics.

    DINO and MoCo layer2 amplitudes differ substantially.  Per-channel teacher
    whitening prevents the high-amplitude checkpoint from dominating simply
    because of its units while retaining the exact channelwise geometry the
    frozen layer3 expects.
    """
    pred = prediction.float()
    ref = target.detach().float()
    channel_sum = ref.sum(dim=(0, 2, 3), keepdim=True)
    channel_square_sum = ref.square().sum(dim=(0, 2, 3), keepdim=True)
    count = ref.new_tensor(float(ref.shape[0] * ref.shape[2] * ref.shape[3]))
    if dist.is_available() and dist.is_initialized():
        channels = channel_sum.numel()
        packed = torch.cat(
            [channel_sum.flatten(), channel_square_sum.flatten(), count.view(1)]
        )
        dist.all_reduce(packed)
        channel_sum = packed[:channels].view_as(channel_sum)
        channel_square_sum = packed[channels : 2 * channels].view_as(
            channel_square_sum
        )
        count = packed[-1]
    mean = channel_sum / count.clamp_min(1.0)
    variance = channel_square_sum / count.clamp_min(1.0) - mean.square()
    raw_std = variance.clamp_min(0.0).sqrt()
    # ReLU makes a subset of teacher channels exactly or nearly constant for a
    # small batch.  Do not amplify those dead channels by 1/eps.
    std_floor = raw_std.mean().clamp_min(eps) * float(std_floor_ratio)
    std = raw_std.clamp_min(std_floor)
    pred_white = (pred - mean) / std
    ref_white = (ref - mean) / std
    huber = F.smooth_l1_loss(pred_white, ref_white, beta=float(beta))
    cosine_loss = 1.0 - F.cosine_similarity(
        pred.flatten(1), ref.flatten(1), dim=1, eps=eps
    ).mean()
    loss = huber + float(cosine_weight) * cosine_loss
    return loss, {
        "whitened_huber": float(huber.detach().item()),
        "cosine_loss": float(cosine_loss.detach().item()),
        "teacher_channel_std_mean": float(std.detach().mean().item()),
        "teacher_channel_std_floor": float(std_floor.detach().item()),
    }


@torch.no_grad()
def evaluate_bridge(
    bridge: LatentToStage2Bridge,
    teacher: SSLResNetFeatureExtractor,
    batches: Iterable[torch.Tensor],
    *,
    use_bf16: bool,
) -> Dict[str, float]:
    """Aggregate held-out raw-map fidelity and pair geometry metrics."""
    predicted: Dict[str, list[torch.Tensor]] = {"layer3": [], "layer4": []}
    target: Dict[str, list[torch.Tensor]] = {"layer3": [], "layer4": []}
    device = next(bridge.parameters()).device
    bridge.eval()
    for latents in batches:
        with _amp_context(device, use_bf16):
            teacher_maps = teacher_stage_maps(teacher, latents)
            bridge_maps = student_stage_maps(bridge, teacher, latents)
        for stage in predicted:
            # Keep held-out maps on GPU: exact pair-distance correlation over
            # flattened layer3 maps is billions of multiply-adds at B=128 and
            # is prohibitively slow on CPU, while remaining modest in VRAM.
            predicted[stage].append(bridge_maps[stage].float())
            target[stage].append(teacher_maps[stage].float())
    combined_pred = {key: torch.cat(values) for key, values in predicted.items()}
    combined_target = {key: torch.cat(values) for key, values in target.items()}
    return stage_map_distillation_metrics(combined_pred, combined_target)


def _save(
    args: argparse.Namespace,
    bridge: LatentToStage2Bridge,
    *,
    step: int,
    metrics: Mapping[str, float],
) -> None:
    save_bridge_checkpoint(
        args.output,
        bridge,
        backbone_name=args.backbone,
        teacher_checkpoint_path=args.feature_checkpoint,
        vae_model_id=args.vae_model_id,
        vae_revision=args.vae_revision,
        metrics=metrics,
        training_metadata={
            "step": int(step),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "layer2_weight": float(args.layer2_weight),
            "layer3_weight": float(args.layer3_weight),
            "layer4_weight": float(args.layer4_weight),
            "cosine_weight": float(args.cosine_weight),
            "cache_path": str(Path(args.cache_path).resolve()),
            "overfit_one_batch": bool(args.overfit_one_batch),
            "generator_config": str(Path(args.generator_config).resolve())
            if args.generator_config
            else None,
            "generator_checkpoint": str(Path(args.generator_checkpoint).resolve())
            if args.generator_checkpoint
            else None,
            "generated_fraction": float(args.generated_fraction),
            "calibrate_output_rms": bool(args.calibrate_output_rms),
            "jacobian_distill_weight": float(args.jacobian_distill_weight),
            "jacobian_distill_count": int(args.jacobian_distill_count),
            "jacobian_distill_eps": float(args.jacobian_distill_eps),
            "jacobian_distill_every": int(args.jacobian_distill_every),
            "jacobian_distill_bank_size": int(args.jacobian_distill_bank_size),
            "jacobian_teacher_chunk_size": int(args.jacobian_teacher_chunk_size),
            "jacobian_bank_storage": "cpu_bfloat16_derivatives",
        },
    )


def _distributed_context(requested_device: str) -> Tuple[torch.device, int, int, bool]:
    """Initialize torchrun/NCCL when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed bridge training requires CUDA/NCCL")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device(requested_device)
    return device, rank, world_size, distributed


def _barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("dino_resnet50", "moco_v2_resnet50"))
    parser.add_argument("--feature-checkpoint", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-bridge", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-model-id", default=DEFAULT_VAE_MODEL_ID)
    parser.add_argument("--vae-revision", default=DEFAULT_VAE_REVISION)

    parser.add_argument("--bridge-width", type=int, default=256)
    parser.add_argument("--bridge-depth", type=int, default=8)
    parser.add_argument("--bridge-expansion", type=int, default=2)
    parser.add_argument("--bridge-kernel-size", type=int, default=7)
    parser.add_argument("--bridge-norm-groups", type=int, default=32)
    parser.add_argument("--bridge-layer-scale-init", type=float, default=1e-3)
    parser.add_argument(
        "--calibrate-output-rms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Globally match bridge/teacher layer2 channel RMS before step 1.",
    )

    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--layer2-weight", type=float, default=1.0)
    parser.add_argument("--layer3-weight", type=float, default=0.5)
    parser.add_argument("--layer4-weight", type=float, default=0.5)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--whiten-std-floor-ratio", type=float, default=0.05)
    parser.add_argument("--latent-noise-std", type=float, default=0.05)
    parser.add_argument(
        "--jacobian-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for paired finite-difference map distillation around z=0; "
            "zero (the default) disables it."
        ),
    )
    parser.add_argument(
        "--jacobian-distill-count",
        type=int,
        default=4,
        help="Number of independent z=0 directions per rank when enabled.",
    )
    parser.add_argument(
        "--jacobian-distill-eps",
        type=float,
        default=0.05,
        help="RMS latent displacement for each local finite-difference direction.",
    )
    parser.add_argument(
        "--jacobian-distill-every",
        type=int,
        default=1,
        help="Apply local finite-difference distillation every N optimizer steps.",
    )
    parser.add_argument(
        "--jacobian-distill-bank-size",
        "--jacobian-bank-size",
        dest="jacobian_distill_bank_size",
        type=int,
        default=0,
        help=(
            "Rank-local fixed teacher-direction bank size; zero keeps dynamic "
            "teacher evaluation on every Jacobian step."
        ),
    )
    parser.add_argument(
        "--jacobian-teacher-chunk-size",
        type=int,
        default=16,
        help="Teacher precompute chunk size for a nonzero Jacobian bank.",
    )
    parser.add_argument(
        "--generator-config",
        default=str(REPO_ROOT / "configs/gen/B4_rev-drift_mae256.yaml"),
        help="B/4 generator YAML used for generated-latent coverage; empty disables it.",
    )
    parser.add_argument(
        "--generator-checkpoint",
        default="",
        help="Optional B/4 checkpoint; EMA is preferred. Empty uses seeded initialization.",
    )
    parser.add_argument("--generator-seed", type=int, default=42)
    parser.add_argument("--generated-fraction", type=float, default=0.25)
    parser.add_argument("--generator-cfg-min", type=float, default=1.0)
    parser.add_argument("--generator-cfg-max", type=float, default=4.0)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="Feature encoder - S4 model")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument(
        "--gradient-check",
        action="store_true",
        help="Run an expensive final tiny B=1/G=2/P=2/N=1 teacher-vs-bridge drift-gradient check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.backbone = canonical_ssl_backbone(args.backbone)
    device, rank, world_size, distributed = _distributed_context(args.device)
    is_main = rank == 0
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    if min(args.layer2_weight, args.layer3_weight, args.layer4_weight) < 0:
        raise ValueError("distillation weights must be non-negative")
    if args.layer2_weight + args.layer3_weight + args.layer4_weight <= 0:
        raise ValueError("at least one distillation weight must be positive")
    if not 0.0 <= args.generated_fraction <= 1.0:
        raise ValueError("generated_fraction must be in [0, 1]")
    if args.latent_noise_std < 0:
        raise ValueError("latent_noise_std must be >= 0")
    if args.whiten_std_floor_ratio <= 0:
        raise ValueError("whiten_std_floor_ratio must be > 0")
    if args.jacobian_distill_weight < 0:
        raise ValueError("jacobian_distill_weight must be >= 0")
    if args.jacobian_distill_count <= 0:
        raise ValueError("jacobian_distill_count must be > 0")
    if args.jacobian_distill_eps <= 0:
        raise ValueError("jacobian_distill_eps must be > 0")
    if args.jacobian_distill_every <= 0:
        raise ValueError("jacobian_distill_every must be > 0")
    if args.jacobian_distill_bank_size < 0:
        raise ValueError("jacobian_distill_bank_size must be >= 0")
    if args.jacobian_teacher_chunk_size <= 0:
        raise ValueError("jacobian_teacher_chunk_size must be > 0")
    if (
        args.jacobian_distill_bank_size > 0
        and args.jacobian_distill_bank_size < args.jacobian_distill_count
    ):
        raise ValueError(
            "jacobian_distill_bank_size must be >= jacobian_distill_count"
        )

    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(process_seed)
        torch.backends.cudnn.benchmark = True

    if is_main:
        print(
            f"[bridge] loading teacher={args.backbone} "
            f"checkpoint={args.feature_checkpoint} world_size={world_size}",
            flush=True,
        )
    teacher = SSLResNetFeatureExtractor(
        args.backbone,
        args.feature_checkpoint,
        use_latent=True,
        use_bf16=args.use_bf16,
        use_remat=False,
        vae_gradient_checkpointing=False,
        microbatch_size=0,
        spatial_pool=2,
        include_norm_x=False,
        vae_model_id=args.vae_model_id,
        vae_revision=args.vae_revision,
        device=device,
    ).to(device)
    teacher.eval()

    if args.resume_bridge:
        bridge, metadata = load_bridge_checkpoint(
            args.resume_bridge, expected_backbone_name=args.backbone
        )
        if is_main:
            print(
                f"[bridge] resumed weights from {args.resume_bridge}; "
                f"previous_step={metadata.get('training', {}).get('step')}",
                flush=True,
            )
    else:
        bridge = LatentToStage2Bridge(
            LatentBridgeConfig(
                width=args.bridge_width,
                depth=args.bridge_depth,
                expansion=args.bridge_expansion,
                kernel_size=args.bridge_kernel_size,
                norm_groups=args.bridge_norm_groups,
                layer_scale_init=args.bridge_layer_scale_init,
            )
        )
    bridge = bridge.to(device)
    bridge.train()
    bridge_raw = bridge
    bridge_train: nn.Module = bridge
    if distributed:
        bridge_train = DistributedDataParallel(
            bridge,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    trainable = sum(parameter.numel() for parameter in bridge_raw.parameters())
    if is_main:
        print(f"[bridge] trainable_parameters={trainable:,} config={bridge_raw.config}", flush=True)

    optimizer = torch.optim.AdamW(
        bridge_train.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    coverage_generator = _load_coverage_generator(
        args.generator_config,
        args.generator_checkpoint,
        seed=args.generator_seed,
        device=device,
    )
    if is_main:
        print(
            f"[bridge] B4 generated coverage fraction={args.generated_fraction} "
            f"config={args.generator_config} checkpoint={args.generator_checkpoint or 'seeded-init'}",
            flush=True,
        )
    train_loader, train_preprocess = _build_loader(
        args.cache_path,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    eval_loader = None
    eval_preprocess = None
    if is_main:
        eval_loader, eval_preprocess = _build_loader(
            args.cache_path,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=False,
            rank=0,
            world_size=1,
        )
    train_iterator = infinite_sampler(train_loader)
    fixed_overfit_batch: Optional[torch.Tensor] = None
    initialization_metrics: Dict[str, float] = {}
    if args.calibrate_output_rms and not args.resume_bridge:
        calibration_real = _batch_latents(
            next(train_iterator), train_preprocess, device
        )
        calibration_latents = _mix_generator_coverage(
            calibration_real,
            coverage_generator,
            fraction=args.generated_fraction,
            num_classes=1000,
            cfg_min=args.generator_cfg_min,
            cfg_max=args.generator_cfg_max,
            use_bf16=args.use_bf16,
        )
        if args.latent_noise_std > 0:
            calibration_latents = calibration_latents + (
                torch.randn_like(calibration_latents) * args.latent_noise_std
            )
        initialization_metrics = calibrate_bridge_output_rms(
            bridge_raw,
            teacher,
            calibration_latents,
            use_bf16=args.use_bf16,
        )
        if args.overfit_one_batch:
            fixed_overfit_batch = calibration_real
        if is_main:
            print(json.dumps(initialization_metrics, sort_keys=True), flush=True)

    jacobian_teacher_bank: Optional[LocalFiniteDifferenceTeacherBank] = None
    if (
        args.jacobian_distill_weight > 0
        and args.jacobian_distill_bank_size > 0
    ):
        bank_reference = fixed_overfit_batch
        if bank_reference is None:
            bank_reference = _batch_latents(
                next(train_iterator), train_preprocess, device
            )
            if args.overfit_one_batch:
                fixed_overfit_batch = bank_reference
        bank_seed = args.seed + 10_000_019 * (rank + 1)
        bank_start_time = time.perf_counter()
        jacobian_teacher_bank = build_local_finite_difference_teacher_bank(
            teacher,
            bank_reference,
            bank_size=args.jacobian_distill_bank_size,
            epsilon=args.jacobian_distill_eps,
            teacher_chunk_size=args.jacobian_teacher_chunk_size,
            use_bf16=args.use_bf16,
            seed=bank_seed,
        )
        bank_build_seconds = time.perf_counter() - bank_start_time
        initialization_metrics.update(
            {
                "initialization/jacobian_bank_size_per_rank": float(
                    jacobian_teacher_bank.size
                ),
                "initialization/jacobian_bank_storage_mib_per_rank": float(
                    jacobian_teacher_bank.storage_bytes / (1024**2)
                ),
                "initialization/jacobian_bank_build_seconds": float(
                    bank_build_seconds
                ),
            }
        )
        _barrier(distributed)
        if is_main:
            print(
                "[bridge] cached local teacher bank "
                f"size_per_rank={jacobian_teacher_bank.size} "
                f"chunk={args.jacobian_teacher_chunk_size} "
                f"storage_mib_per_rank={jacobian_teacher_bank.storage_bytes / (1024**2):.1f} "
                f"build_seconds={bank_build_seconds:.2f}",
                flush=True,
            )
    latest_metrics: Dict[str, float] = dict(initialization_metrics)
    start_time = time.perf_counter()
    wandb_run = None
    if is_main and args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or Path(args.output).stem,
            config=vars(args),
        )
        if initialization_metrics:
            wandb_run.log(initialization_metrics, step=0)

    for step in range(1, args.max_steps + 1):
        if fixed_overfit_batch is None or not args.overfit_one_batch:
            fixed_overfit_batch = _batch_latents(
                next(train_iterator), train_preprocess, device
            )
        latents = fixed_overfit_batch
        latents = _mix_generator_coverage(
            latents,
            coverage_generator,
            fraction=args.generated_fraction,
            num_classes=1000,
            cfg_min=args.generator_cfg_min,
            cfg_max=args.generator_cfg_max,
            use_bf16=args.use_bf16,
        )
        if args.latent_noise_std > 0:
            latents = latents + torch.randn_like(latents) * args.latent_noise_std

        optimizer.zero_grad(set_to_none=True)
        jacobian_active = (
            args.jacobian_distill_weight > 0
            and step % args.jacobian_distill_every == 0
        )
        with torch.no_grad(), _amp_context(device, args.use_bf16):
            target_maps = teacher_stage_maps(teacher, latents)

        # Standard DDP gradient accumulation: skip synchronization for the
        # ordinary batch, then synchronize its sum with the bounded local block.
        main_gradient_context = (
            bridge_train.no_sync()
            if distributed and jacobian_active
            else nullcontext()
        )
        with main_gradient_context:
            with _amp_context(device, args.use_bf16):
                predicted_maps = student_stage_maps(bridge_train, teacher, latents)
                total_loss = latents.new_zeros((), dtype=torch.float32)
                step_metrics: Dict[str, float] = {}
                for stage, weight in (
                    ("layer2", args.layer2_weight),
                    ("layer3", args.layer3_weight),
                    ("layer4", args.layer4_weight),
                ):
                    if weight <= 0:
                        continue
                    if stage == "layer2":
                        stage_loss, components = channel_whitened_huber_loss(
                            predicted_maps[stage],
                            target_maps[stage],
                            beta=args.huber_beta,
                            cosine_weight=args.cosine_weight,
                            std_floor_ratio=args.whiten_std_floor_ratio,
                        )
                    else:
                        stage_loss, components = normalized_feature_loss(
                            predicted_maps[stage],
                            target_maps[stage],
                            cosine_weight=args.cosine_weight,
                        )
                    total_loss = total_loss + weight * stage_loss
                    step_metrics.update(
                        {
                            f"train/{stage}_{key}": value
                            for key, value in components.items()
                        }
                    )
            primary_loss_value = float(total_loss.detach().item())
            total_loss.backward()
        del predicted_maps, target_maps, total_loss

        objective_value = primary_loss_value
        if jacobian_active:
            jacobian_stage_weights = {
                "layer2": args.layer2_weight,
                "layer3": args.layer3_weight,
                "layer4": args.layer4_weight,
            }
            if jacobian_teacher_bank is None:
                jacobian_loss, jacobian_metrics = local_finite_difference_distillation(
                    bridge_train,
                    teacher,
                    latents,
                    count=args.jacobian_distill_count,
                    epsilon=args.jacobian_distill_eps,
                    stage_weights=jacobian_stage_weights,
                    cosine_weight=args.cosine_weight,
                    use_bf16=args.use_bf16,
                    seed=args.seed + rank + 1_000_003 * step,
                )
            else:
                jacobian_loss, jacobian_metrics = (
                    local_finite_difference_distillation_from_bank(
                        bridge_train,
                        teacher,
                        jacobian_teacher_bank,
                        count=args.jacobian_distill_count,
                        application_index=(
                            step // args.jacobian_distill_every - 1
                        ),
                        stage_weights=jacobian_stage_weights,
                        cosine_weight=args.cosine_weight,
                        use_bf16=args.use_bf16,
                    )
                )
            weighted_jacobian_loss = (
                float(args.jacobian_distill_weight) * jacobian_loss
            )
            weighted_jacobian_loss.backward()
            objective_value += float(weighted_jacobian_loss.detach().item())
            step_metrics.update(jacobian_metrics)
            step_metrics["jacobian/loss"] = float(jacobian_loss.detach().item())
            step_metrics["jacobian/weighted_loss"] = float(
                weighted_jacobian_loss.detach().item()
            )
            del jacobian_loss, weighted_jacobian_loss

        grad_norm = nn.utils.clip_grad_norm_(bridge_train.parameters(), args.max_grad_norm)
        optimizer.step()
        latest_metrics = {
            "train/loss": objective_value,
            "train/primary_loss": primary_loss_value,
            "train/grad_norm": float(grad_norm.detach().item()),
            **step_metrics,
        }

        if (step == 1 or step % args.log_every == 0) and is_main:
            elapsed = time.perf_counter() - start_time
            payload = {
                "step": step,
                "steps_per_second": step / max(elapsed, 1e-9),
                **latest_metrics,
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(payload, step=step)

        should_eval = args.eval_every > 0 and (
            step % args.eval_every == 0 or step == args.max_steps
        )
        if should_eval:
            _barrier(distributed)
            if is_main:
                assert eval_loader is not None and eval_preprocess is not None
                eval_batches = []
                eval_iterator = iter(eval_loader)
                for _ in range(args.eval_batches):
                    try:
                        batch = next(eval_iterator)
                    except StopIteration:
                        break
                    eval_latents = _batch_latents(batch, eval_preprocess, device)
                    eval_batches.append(
                        _mix_generator_coverage(
                            eval_latents,
                            coverage_generator,
                            fraction=args.generated_fraction,
                            num_classes=1000,
                            cfg_min=args.generator_cfg_min,
                            cfg_max=args.generator_cfg_max,
                            use_bf16=args.use_bf16,
                        )
                    )
                    if args.latent_noise_std > 0:
                        eval_batches[-1] = eval_batches[-1] + (
                            torch.randn_like(eval_batches[-1])
                            * args.latent_noise_std
                        )
                heldout = evaluate_bridge(
                    bridge_raw, teacher, eval_batches, use_bf16=args.use_bf16
                )
                latest_metrics.update(
                    {f"heldout/{key}": value for key, value in heldout.items()}
                )
                print(
                    json.dumps({"step": step, **latest_metrics}, sort_keys=True),
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(latest_metrics, step=step)
                bridge_raw.train()
            _barrier(distributed)

        if is_main and args.save_every > 0 and step % args.save_every == 0:
            _save(args, bridge_raw, step=step, metrics=latest_metrics)

    _barrier(distributed)
    if is_main:
        _save(args, bridge_raw, step=args.max_steps, metrics=latest_metrics)
        print(f"[bridge] saved runtime checkpoint: {args.output}", flush=True)

    if is_main and args.gradient_check:
        runtime = SSLLatentBridgeFeatureExtractor(
            args.backbone,
            args.feature_checkpoint,
            args.output,
            use_bf16=args.use_bf16,
            use_remat=False,
            microbatch_size=2,
            spatial_pool=2,
            include_norm_x=False,
        ).to(device)
        assert eval_loader is not None and eval_preprocess is not None
        gradient_source = iter(eval_loader)
        gradient_latents = _batch_latents(
            next(gradient_source), eval_preprocess, device
        )
        required = 2 + 2 + 1
        if gradient_latents.shape[0] < required:
            raise ValueError(
                f"gradient check needs at least {required} held-out samples"
            )
        if coverage_generator is not None and args.generated_fraction > 0:
            generated_queries = _mix_generator_coverage(
                gradient_latents[:2],
                coverage_generator,
                fraction=1.0,
                num_classes=1000,
                cfg_min=args.generator_cfg_min,
                cfg_max=args.generator_cfg_max,
                use_bf16=args.use_bf16,
            )
        else:
            generated_queries = gradient_latents[:2]
        if args.latent_noise_std > 0:
            generated_queries = generated_queries + (
                torch.randn_like(generated_queries) * args.latent_noise_std
            )
        gradient_metrics = compare_reverse_drift_latent_gradients(
            teacher,
            runtime,
            generated_queries,
            gradient_latents[2:4],
            gradient_latents[4:5],
            batch_size=1,
            generated_count=2,
            positive_count=2,
            negative_count=1,
        )
        latest_metrics.update(gradient_metrics)
        _save(args, bridge_raw, step=args.max_steps, metrics=latest_metrics)
        print(json.dumps({"step": args.max_steps, **gradient_metrics}, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(gradient_metrics, step=args.max_steps)

    _barrier(distributed)
    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
