#!/usr/bin/env python3
"""Validate a latent SSL bridge against its exact VAE+ResNet teacher.

This is intentionally a *launch gate*, not a training-time reconstruction
metric.  Each trial builds the same class-conditional reverse-drift geometry
used by the B/4 experiments (by default B=1, G=32, P=64, N=32), evaluates all
35 stage-3/4 activation objectives, and compares the latent gradient induced by
the exact and bridged encoders.  Clean latents drive the drift comparison;
additional noisy real/generated latents test bridge coverage off the cache
manifold.

The script is single-GPU by design.  The exact teacher uses outer
rematerialization and feature microbatches so its generated-query backward pass
fits on a 32 GiB card.  It writes a self-contained JSON report before returning
a non-zero status when a configured gate fails.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting_core.imagenet_loss import (  # noqa: E402
    _cdist_batched,
    _ratio_of_means,
)
from models.imagenet_generator import build_ditgen_from_config  # noqa: E402
from models.ssl_latent_bridge import (  # noqa: E402
    SSLLatentBridgeFeatureExtractor,
    file_sha256,
)
from models.ssl_resnet import (  # noqa: E402
    SSLResNetFeatureExtractor,
    canonical_ssl_backbone,
)
from train_imagenet_gen import (  # noqa: E402
    compute_drift_loss_from_features,
    sample_cfg,
)


DEFAULT_VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
DEFAULT_VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
ACTIVATION_KWARGS: Dict[str, Any] = {
    "active_stages": ["stage3", "stage4"],
    "patch_mean_size": [2, 4],
    "patch_std_size": [2, 4],
    "use_std": True,
    "use_mean": True,
    "with_global": False,
    "with_norm_x": False,
    "every_k_block": 2,
    "exclude_terminal_block": True,
}


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    ]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Iterable[float]) -> Dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "p10": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "p10": _quantile(finite, 0.10),
        "median": _quantile(finite, 0.50),
        "p90": _quantile(finite, 0.90),
        "min": min(finite),
        "max": max(finite),
    }


def _stage_for_feature(name: str) -> str:
    if name == "layer3" or name.startswith("layer3_"):
        return "stage3"
    if name == "layer4" or name.startswith("layer4_"):
        return "stage4"
    raise ValueError(f"Unexpected non-stage3/4 feature in bridge gate: {name}")


def _sample_flat(values: torch.Tensor, maximum: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if maximum <= 0 or flat.numel() <= maximum:
        return flat
    indices = torch.linspace(
        0,
        flat.numel() - 1,
        steps=maximum,
        device=flat.device,
    ).round().long()
    return flat.index_select(0, indices)


def _pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float(((x * y).sum() / denominator.clamp_min(eps)).item())


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    # Distances are effectively continuous.  Stable ordinal ranks avoid the
    # quadratic memory cost of exact tie averaging while remaining deterministic.
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        order.numel(), device=values.device, dtype=torch.float32
    )
    return ranks


def _distortion_metrics(
    teacher_values: torch.Tensor,
    bridge_values: torch.Tensor,
    *,
    max_values: int,
    min_distance: float,
) -> Dict[str, float]:
    teacher_values = _sample_flat(teacher_values.detach().float(), max_values)
    bridge_values = _sample_flat(bridge_values.detach().float(), max_values)
    if teacher_values.shape != bridge_values.shape:
        raise RuntimeError("Distance sampling produced mismatched shapes")
    # The reference decides which distances are meaningful.  Requiring the
    # candidate to exceed the cutoff too would silently hide bridge collapse:
    # a teacher distance of 0.5 mapped to 0.0 must be a large error, not an
    # excluded term.
    valid = teacher_values >= float(min_distance)
    if valid.any():
        log_ratio = torch.log(
            bridge_values[valid].clamp_min(1e-12) / teacher_values[valid]
        )
        log_rmse = float(log_ratio.square().mean().sqrt().item())
        candidate_below = float(
            (bridge_values[valid] < float(min_distance)).float().mean().item()
        )
    else:
        log_rmse = float("nan")
        candidate_below = float("nan")
    normalized_rmse = float(
        (
            (bridge_values - teacher_values).square().mean().sqrt()
            / teacher_values.square().mean().sqrt().clamp_min(1e-12)
        ).item()
    )
    pearson = _pearson(teacher_values, bridge_values)
    spearman = _pearson(_rankdata(teacher_values), _rankdata(bridge_values))
    return {
        "log_rmse": log_rmse,
        "normalized_rmse": normalized_rmse,
        "pearson": pearson,
        "spearman": spearman,
        "reference_accepted_fraction": float(valid.float().mean().item()),
        "candidate_below_min_fraction_on_reference": candidate_below,
    }


class LatentVariantIndex:
    """Index both cached horizontal variants without stochastic data loading."""

    def __init__(self, cache_path: str | Path, split: str) -> None:
        root = Path(cache_path).resolve() / split
        if not root.is_dir():
            raise FileNotFoundError(f"Latent cache split not found: {root}")
        class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if not class_dirs:
            raise ValueError(f"No class directories found under {root}")
        self.class_names = [path.name for path in class_dirs]
        self.by_class: Dict[int, list[Tuple[Path, bool]]] = {}
        self.all_variants: list[Tuple[int, Path, bool]] = []
        for class_index, class_dir in enumerate(class_dirs):
            variants: list[Tuple[Path, bool]] = []
            for path in sorted(class_dir.glob("*.pt")):
                variants.append((path, False))
                variants.append((path, True))
                self.all_variants.append((class_index, path, False))
                self.all_variants.append((class_index, path, True))
            if not variants:
                raise ValueError(f"No .pt latents found in {class_dir}")
            self.by_class[class_index] = variants

    @staticmethod
    def load(variant: Tuple[Path, bool]) -> torch.Tensor:
        path, flipped = variant
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = "moments_flip" if flipped else "moments"
        if not isinstance(payload, Mapping) or key not in payload:
            raise ValueError(f"Cache entry {path} is missing {key}")
        value = torch.as_tensor(np.asarray(payload[key])).float()
        if tuple(value.shape) != (4, 32, 32):
            raise ValueError(f"Unexpected latent shape in {path}: {tuple(value.shape)}")
        return value

    @staticmethod
    def _draw_unique_or_replace(
        population: Sequence[Any], count: int, rng: random.Random
    ) -> list[Any]:
        if count <= len(population):
            return rng.sample(list(population), count)
        return [rng.choice(population) for _ in range(count)]

    def sample_fixed(
        self,
        labels: Sequence[int],
        *,
        positive_count: int,
        negative_count: int,
        rng: random.Random,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positives: list[torch.Tensor] = []
        negatives: list[torch.Tensor] = []
        for label in labels:
            positive_variants = self._draw_unique_or_replace(
                self.by_class[int(label)], positive_count, rng
            )
            negative_pool = [
                (path, flipped)
                for other_label, path, flipped in self.all_variants
                if other_label != int(label)
            ]
            negative_variants = self._draw_unique_or_replace(
                negative_pool, negative_count, rng
            )
            positives.extend(self.load(value) for value in positive_variants)
            negatives.extend(self.load(value) for value in negative_variants)
        return torch.stack(positives), torch.stack(negatives)


def _load_generator(
    config_path: str,
    checkpoint_path: str,
    *,
    seed: int,
    device: torch.device,
) -> nn.Module:
    import yaml

    with Path(config_path).open("r") as handle:
        raw_config = yaml.safe_load(handle)
    fork_devices = _fork_devices(device)
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        generator = build_ditgen_from_config(
            raw_config["model"], raw_config["dataset"]
        ).to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise ValueError(f"Unsupported generator checkpoint: {checkpoint_path}")
        generator_state = None
        for key in ("ema", "model", "state_dict"):
            candidate = state.get(key)
            if isinstance(candidate, Mapping):
                generator_state = candidate
                break
        if generator_state is None:
            raise ValueError(
                f"Generator checkpoint has no ema/model/state_dict: {checkpoint_path}"
            )
        generator.load_state_dict(dict(generator_state), strict=True)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator


@torch.no_grad()
def _generated_queries(
    generator: nn.Module,
    labels: torch.Tensor,
    cfg_scales: torch.Tensor,
    *,
    generated_count: int,
    seed: int,
    device: torch.device,
    use_bf16: bool,
) -> torch.Tensor:
    expanded_labels = repeat(labels, "b -> (b g)", g=generated_count)
    expanded_cfg = repeat(cfg_scales, "b -> (b g)", g=generated_count)
    fork_devices = _fork_devices(device)
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        with _amp_context(device, use_bf16):
            generated = generator(
                expanded_labels, cfg_scale=expanded_cfg, train=False
            )["samples"]
    return generated.float()


def _split_features(
    features: Mapping[str, torch.Tensor],
    *,
    generated_size: int,
    positive_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    generated = {name: value[:generated_size] for name, value in features.items()}
    positive = {
        name: value[generated_size : generated_size + positive_size]
        for name, value in features.items()
    }
    negative = {
        name: value[generated_size + positive_size :]
        for name, value in features.items()
    }
    return generated, positive, negative


@torch.no_grad()
def _extract_all(
    extractor: nn.Module,
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    device: torch.device,
    use_bf16: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    values = torch.cat([generated, positive, negative], dim=0)
    with _amp_context(device, use_bf16):
        features = extractor.get_activations(values, **ACTIVATION_KWARGS)
    if len(features) != 35:
        raise RuntimeError(f"Expected exactly 35 stage3/4 objectives, got {len(features)}")
    forbidden = {
        name
        for name in features
        if not (name == "layer3" or name.startswith("layer3_") or name == "layer4" or name.startswith("layer4_"))
    }
    if forbidden:
        raise RuntimeError(f"Stage1/2 or unknown objectives escaped pruning: {sorted(forbidden)}")
    return _split_features(
        features,
        generated_size=generated.shape[0],
        positive_size=positive.shape[0],
    )


def _value_metrics(
    teacher_features: Mapping[str, torch.Tensor],
    bridge_features: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if set(teacher_features) != set(bridge_features):
        raise RuntimeError("Teacher/bridge activation key sets differ")
    for name in teacher_features:
        teacher = teacher_features[name].detach().float().flatten(1)
        bridge = bridge_features[name].detach().float().flatten(1)
        if teacher.shape != bridge.shape:
            raise RuntimeError(f"Activation shape mismatch for {name}")
        cosine = F.cosine_similarity(teacher, bridge, dim=1, eps=1e-8)
        per_sample_nrmse = (
            (bridge - teacher).square().mean(dim=1).sqrt()
            / teacher.square().mean(dim=1).sqrt().clamp_min(1e-8)
        )
        result[name] = {
            "cosine_mean": float(cosine.mean().item()),
            "cosine_p10": float(torch.quantile(cosine, 0.10).item()),
            "normalized_rmse_mean": float(per_sample_nrmse.mean().item()),
            "normalized_rmse_p90": float(torch.quantile(per_sample_nrmse, 0.90).item()),
        }
    return result


def _mutual_affinity(
    normalized_distance: torch.Tensor,
    *,
    self_mask: torch.Tensor,
    target_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = -(normalized_distance + self_mask * 100.0) / float(temperature)
    affinity = (
        F.softmax(logits, dim=2) * F.softmax(logits, dim=1)
    ).clamp_min(1e-6).sqrt()
    affinity = affinity * target_weights.unsqueeze(1)
    return affinity / affinity.sum(dim=2, keepdim=True).clamp_min(1e-12)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float().clamp_min(1e-12)
    q = q.float().clamp_min(1e-12)
    midpoint = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - midpoint.log())).sum(dim=2)
    js = js + 0.5 * (q * (q.log() - midpoint.log())).sum(dim=2)
    return float(js.mean().item())


@torch.no_grad()
def _distance_metrics(
    teacher: Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    bridge: Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
    generated_count: int,
    positive_count: int,
    negative_count: int,
    weight_negative: torch.Tensor,
    R_list: Sequence[float],
    stage_temperatures: Mapping[str, float],
    max_values: int,
    min_distance: float,
) -> Dict[str, Dict[str, float]]:
    teacher_gen, teacher_pos, teacher_neg = teacher
    bridge_gen, bridge_pos, bridge_neg = bridge
    if set(teacher_gen) != set(bridge_gen):
        raise RuntimeError("Teacher/bridge activation key sets differ")
    results: Dict[str, Dict[str, float]] = {}
    for name in teacher_gen:
        grids = []
        for generated, positive, negative in (
            (teacher_gen[name], teacher_pos[name], teacher_neg[name]),
            (bridge_gen[name], bridge_pos[name], bridge_neg[name]),
        ):
            token_count = generated.shape[1]
            generated_grid = rearrange(
                generated.detach().float(),
                "(b g) t d -> (b t) g d",
                b=batch_size,
                g=generated_count,
            )
            positive_grid = rearrange(
                positive.detach().float(),
                "(b p) t d -> (b t) p d",
                b=batch_size,
                p=positive_count,
            )
            negative_grid = rearrange(
                negative.detach().float(),
                "(b n) t d -> (b t) n d",
                b=batch_size,
                n=negative_count,
            )
            targets = torch.cat([generated_grid, negative_grid, positive_grid], dim=1)
            weights = repeat(
                torch.cat(
                    [
                        weight_negative.new_ones(batch_size, generated_count),
                        weight_negative,
                        weight_negative.new_ones(batch_size, positive_count),
                    ],
                    dim=1,
                ),
                "b m -> (b t) m",
                t=token_count,
            )
            distance = _cdist_batched(generated_grid, targets)
            scale = _ratio_of_means(
                distance * weights.unsqueeze(1),
                weights,
                use_global_stats=False,
            )
            grids.append((distance / scale.clamp_min(1e-3), weights))

        teacher_distance, target_weights = grids[0]
        bridge_distance, _ = grids[1]
        rows = teacher_distance.shape[0]
        diagonal = torch.eye(
            generated_count,
            device=teacher_distance.device,
            dtype=torch.bool,
        )
        self_mask = F.pad(
            diagonal, (0, negative_count + positive_count)
        ).unsqueeze(0)
        self_mask = self_mask.expand(rows, -1, -1)
        split = generated_count + negative_count
        masks = {
            "nonself": ~self_mask,
            "generated_nonself": F.pad(~diagonal, (0, negative_count + positive_count)).unsqueeze(0).expand(rows, -1, -1),
            "real_negative": torch.zeros_like(self_mask),
            "positive": torch.zeros_like(self_mask),
        }
        masks["real_negative"][:, :, generated_count:split] = True
        masks["positive"][:, :, split:] = True

        feature_result: Dict[str, float] = {}
        for group, mask in masks.items():
            distortion = _distortion_metrics(
                teacher_distance[mask],
                bridge_distance[mask],
                max_values=max_values,
                min_distance=min_distance,
            )
            for metric, value in distortion.items():
                feature_result[f"{group}/{metric}"] = value

        teacher_pos_winner = teacher_distance[:, :, split:].argmin(dim=2)
        bridge_pos_winner = bridge_distance[:, :, split:].argmin(dim=2)
        feature_result["positive_nearest_agreement"] = float(
            teacher_pos_winner.eq(bridge_pos_winner).float().mean().item()
        )
        teacher_repulsive = teacher_distance[:, :, :split].masked_fill(
            self_mask[:, :, :split], float("inf")
        )
        bridge_repulsive = bridge_distance[:, :, :split].masked_fill(
            self_mask[:, :, :split], float("inf")
        )
        feature_result["repulsive_nearest_agreement"] = float(
            teacher_repulsive.argmin(dim=2)
            .eq(bridge_repulsive.argmin(dim=2))
            .float()
            .mean()
            .item()
        )

        stage_temperature = float(stage_temperatures[_stage_for_feature(name)])
        js_values = []
        top1_values = []
        self_mask_float = self_mask.float()
        for base_R in R_list:
            temperature = float(base_R) * stage_temperature
            teacher_affinity = _mutual_affinity(
                teacher_distance,
                self_mask=self_mask_float,
                target_weights=target_weights,
                temperature=temperature,
            )
            bridge_affinity = _mutual_affinity(
                bridge_distance,
                self_mask=self_mask_float,
                target_weights=target_weights,
                temperature=temperature,
            )
            js_values.append(_js_divergence(teacher_affinity, bridge_affinity))
            top1_values.append(
                float(
                    teacher_affinity.argmax(dim=2)
                    .eq(bridge_affinity.argmax(dim=2))
                    .float()
                    .mean()
                    .item()
                )
            )
        feature_result["affinity_js_mean_over_R"] = statistics.fmean(js_values)
        feature_result["affinity_js_max_over_R"] = max(js_values)
        feature_result["affinity_top1_agreement_mean_over_R"] = statistics.fmean(
            top1_values
        )
        results[name] = feature_result
        del teacher_distance, bridge_distance, grids
    return results


def _gradient(
    extractor: nn.Module,
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    batch_size: int,
    generated_count: int,
    positive_count: int,
    negative_count: int,
    weight_negative: torch.Tensor,
    R_list: Tuple[float, ...],
    stage_temperatures: Mapping[str, float],
    device: torch.device,
    use_bf16: bool,
) -> torch.Tensor:
    queries = generated.detach().clone().requires_grad_(True)
    with torch.no_grad(), _amp_context(device, use_bf16):
        fixed_features = extractor.get_activations(
            torch.cat([positive, negative], dim=0), **ACTIVATION_KWARGS
        )
    positive_features = {
        name: value[: batch_size * positive_count]
        for name, value in fixed_features.items()
    }
    negative_features = {
        name: value[batch_size * positive_count :]
        for name, value in fixed_features.items()
    }
    with _amp_context(device, use_bf16):
        generated_features = extractor.get_activations(
            queries, **ACTIVATION_KWARGS
        )
    if len(generated_features) != 35:
        raise RuntimeError(
            f"Gradient gate expected 35 objectives, got {len(generated_features)}"
        )
    loss, _ = compute_drift_loss_from_features(
        gen_feats=generated_features,
        pos_feats=positive_features,
        neg_feats=negative_features,
        B=batch_size,
        G=generated_count,
        P=positive_count,
        N=negative_count,
        weight_neg=weight_negative,
        R_list=R_list,
        drift_matching="rev-drift",
        compute_wpos_stats=False,
        global_scale_stats=False,
        global_fnorm_stats=False,
        collect_diagnostics=False,
        feature_temperature_multipliers=dict(stage_temperatures),
        rev_drift_affinity_kernel="exponential",
    )
    gradient = torch.autograd.grad(loss, queries, create_graph=False)[0]
    return gradient.detach().float()


def _gradient_metrics(
    teacher_gradient: torch.Tensor,
    bridge_gradient: torch.Tensor,
) -> Dict[str, list[float]]:
    teacher = teacher_gradient.flatten(1)
    bridge = bridge_gradient.flatten(1)
    cosine = F.cosine_similarity(teacher, bridge, dim=1, eps=1e-12)
    teacher_norm = teacher.norm(dim=1).clamp_min(1e-12)
    bridge_norm = bridge.norm(dim=1)
    norm_ratio = bridge_norm / teacher_norm
    relative_error = (bridge - teacher).norm(dim=1) / teacher_norm
    return {
        "cosine": cosine.cpu().tolist(),
        "norm_ratio": norm_ratio.cpu().tolist(),
        "relative_error": relative_error.cpu().tolist(),
        "teacher_norm": teacher_norm.cpu().tolist(),
        "bridge_norm": bridge_norm.cpu().tolist(),
    }


def _accumulate_nested(
    destination: Dict[str, Dict[str, list[float]]],
    values: Mapping[str, Mapping[str, float]],
) -> None:
    for feature, metrics in values.items():
        for metric, value in metrics.items():
            destination[feature][metric].append(float(value))


def _summarize_nested(
    values: Mapping[str, Mapping[str, Sequence[float]]]
) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    return {
        feature: {metric: _summary(samples) for metric, samples in metrics.items()}
        for feature, metrics in values.items()
    }


def _feature_medians(
    values: Mapping[str, Mapping[str, Sequence[float]]], metric: str
) -> list[float]:
    result = []
    for metrics in values.values():
        if metric in metrics:
            median = float(_summary(metrics[metric])["median"])
            if math.isfinite(median):
                result.append(median)
    return result


def _evaluate_gates(
    args: argparse.Namespace,
    distance_values: Mapping[str, Mapping[str, Sequence[float]]],
    gradient_values: Mapping[str, Sequence[float]],
) -> Tuple[bool, list[Dict[str, Any]]]:
    checks: list[Dict[str, Any]] = []

    def maximum(name: str, value: float, threshold: float) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "operator": "<=",
                "threshold": threshold,
                "passed": math.isfinite(value) and value <= threshold,
            }
        )

    def minimum(name: str, value: float, threshold: float) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "operator": ">=",
                "threshold": threshold,
                "passed": math.isfinite(value) and value >= threshold,
            }
        )

    log_rmse = _feature_medians(distance_values, "nonself/log_rmse")
    pearson = _feature_medians(distance_values, "nonself/pearson")
    affinity_js = _feature_medians(distance_values, "affinity_js_mean_over_R")
    affinity_top1 = _feature_medians(
        distance_values, "affinity_top1_agreement_mean_over_R"
    )
    maximum(
        "distance_log_rmse_feature_median",
        float(_summary(log_rmse)["median"]),
        args.max_distance_log_rmse_median,
    )
    maximum(
        "distance_log_rmse_worst_feature",
        max(log_rmse, default=float("nan")),
        args.max_distance_log_rmse_worst,
    )
    minimum(
        "distance_pearson_feature_median",
        float(_summary(pearson)["median"]),
        args.min_distance_pearson_median,
    )
    minimum(
        "distance_pearson_worst_feature",
        min(pearson, default=float("nan")),
        args.min_distance_pearson_worst,
    )
    maximum(
        "affinity_js_feature_median",
        float(_summary(affinity_js)["median"]),
        args.max_affinity_js_median,
    )
    maximum(
        "affinity_js_worst_feature",
        max(affinity_js, default=float("nan")),
        args.max_affinity_js_worst,
    )
    minimum(
        "affinity_top1_feature_median",
        float(_summary(affinity_top1)["median"]),
        args.min_affinity_top1_median,
    )

    cosine_summary = _summary(gradient_values["cosine"])
    ratio_summary = _summary(gradient_values["norm_ratio"])
    minimum(
        "gradient_cosine_median",
        float(cosine_summary["median"]),
        args.min_gradient_cosine_median,
    )
    minimum(
        "gradient_cosine_p10",
        float(cosine_summary["p10"]),
        args.min_gradient_cosine_p10,
    )
    minimum(
        "gradient_norm_ratio_median_lower",
        float(ratio_summary["median"]),
        args.min_gradient_norm_ratio_median,
    )
    maximum(
        "gradient_norm_ratio_median_upper",
        float(ratio_summary["median"]),
        args.max_gradient_norm_ratio_median,
    )
    return all(bool(check["passed"]) for check in checks), checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        required=True,
        choices=("dino_resnet50", "moco_v2_resnet50"),
    )
    parser.add_argument("--feature-checkpoint", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--cache-split", default="val")
    parser.add_argument("--generator-config", required=True)
    parser.add_argument("--generator-checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-model-id", default=DEFAULT_VAE_MODEL_ID)
    parser.add_argument("--vae-revision", default=DEFAULT_VAE_REVISION)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-microbatch-size", type=int, default=8)
    parser.add_argument("--bridge-microbatch-size", type=int, default=32)

    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generated-count", type=int, default=32)
    parser.add_argument("--positive-count", type=int, default=64)
    parser.add_argument("--negative-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--generator-seed", type=int, default=43)
    parser.add_argument("--cfg-min", type=float, default=1.0)
    parser.add_argument("--cfg-max", type=float, default=4.0)
    parser.add_argument("--neg-cfg-pw", type=float, default=3.0)
    parser.add_argument("--no-cfg-frac", type=float, default=0.0)
    parser.add_argument("--R-list", type=float, nargs="+", default=(0.2, 0.05, 0.02))
    parser.add_argument("--stage3-temperature", type=float, default=1.0)
    parser.add_argument("--stage4-temperature", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--noisy-value-count",
        type=int,
        default=32,
        help="Clean samples per trial to copy/noise for the off-manifold value gate; 0 disables.",
    )
    parser.add_argument("--max-distance-values", type=int, default=65_536)
    parser.add_argument("--min-distance", type=float, default=0.02)

    parser.add_argument("--max-distance-log-rmse-median", type=float, default=0.10)
    parser.add_argument("--max-distance-log-rmse-worst", type=float, default=0.25)
    parser.add_argument("--min-distance-pearson-median", type=float, default=0.98)
    parser.add_argument("--min-distance-pearson-worst", type=float, default=0.90)
    parser.add_argument("--max-affinity-js-median", type=float, default=0.02)
    parser.add_argument("--max-affinity-js-worst", type=float, default=0.08)
    parser.add_argument("--min-affinity-top1-median", type=float, default=0.90)
    parser.add_argument("--min-gradient-cosine-median", type=float, default=0.95)
    parser.add_argument("--min-gradient-cosine-p10", type=float, default=0.85)
    parser.add_argument("--min-gradient-norm-ratio-median", type=float, default=0.80)
    parser.add_argument("--max-gradient-norm-ratio-median", type=float, default=1.25)
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.backbone = canonical_ssl_backbone(args.backbone)
    if args.trials <= 0 or args.batch_size <= 0:
        raise ValueError("trials and batch-size must be positive")
    if args.generated_count < 32 or args.positive_count < 64 or args.negative_count < 32:
        raise ValueError("The production bridge gate requires at least G=32/P=64/N=32")
    if args.noise_std < 0 or args.noisy_value_count < 0:
        raise ValueError("noise settings must be non-negative")
    if args.teacher_microbatch_size <= 0 or args.bridge_microbatch_size <= 0:
        raise ValueError("feature microbatch sizes must be positive")
    if any(value <= 0 or not math.isfinite(value) for value in args.R_list):
        raise ValueError("R-list values must be finite and positive")
    stage_temperatures = {
        "stage3": float(args.stage3_temperature),
        "stage4": float(args.stage4_temperature),
    }
    if any(value <= 0 or not math.isfinite(value) for value in stage_temperatures.values()):
        raise ValueError("stage temperatures must be finite and positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    started = time.perf_counter()
    cache = LatentVariantIndex(args.cache_path, args.cache_split)
    if len(cache.by_class) != 1000:
        raise ValueError(
            f"Expected 1,000 ImageNet classes, found {len(cache.by_class)}"
        )
    generator = _load_generator(
        args.generator_config,
        args.generator_checkpoint,
        seed=args.generator_seed,
        device=device,
    )
    teacher = SSLResNetFeatureExtractor(
        args.backbone,
        args.feature_checkpoint,
        use_latent=True,
        use_bf16=args.use_bf16,
        use_remat=True,
        vae_gradient_checkpointing=False,
        microbatch_size=args.teacher_microbatch_size,
        spatial_pool=2,
        include_norm_x=False,
        vae_model_id=args.vae_model_id,
        vae_revision=args.vae_revision,
        device=device,
    ).to(device).eval()
    bridge = SSLLatentBridgeFeatureExtractor(
        args.backbone,
        args.feature_checkpoint,
        args.bridge_checkpoint,
        use_bf16=args.use_bf16,
        use_remat=False,
        microbatch_size=args.bridge_microbatch_size,
        spatial_pool=2,
        include_norm_x=False,
        verify_teacher_sha256=True,
    ).to(device).eval()

    forbidden_names = {"vae", "conv1", "bn1", "maxpool", "layer1", "layer2"}
    forbidden_runtime_modules = sorted(
        f"{owner_name}.{name}"
        for owner_name, owner in (("runtime", bridge), ("tail", bridge.tail))
        for name in forbidden_names
        if hasattr(owner, name)
    )
    if forbidden_runtime_modules:
        raise RuntimeError(
            f"Bridge runtime owns forbidden prefix modules: {forbidden_runtime_modules}"
        )

    distance_values: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    clean_generated_values: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    clean_real_values: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    noisy_values: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gradient_values: Dict[str, list[float]] = defaultdict(list)
    trial_records = []

    for trial in range(args.trials):
        trial_seed = int(args.seed) + 1_000_003 * trial
        rng = random.Random(trial_seed)
        labels_list = [rng.randrange(len(cache.by_class)) for _ in range(args.batch_size)]
        labels = torch.tensor(labels_list, device=device, dtype=torch.long)
        positive_cpu, negative_cpu = cache.sample_fixed(
            labels_list,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            rng=rng,
        )
        positive = positive_cpu.to(device, non_blocking=True)
        negative = negative_cpu.to(device, non_blocking=True)

        fork_devices = _fork_devices(device)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(trial_seed + 17)
            cfg_scales = sample_cfg(
                args.batch_size,
                args.cfg_min,
                args.cfg_max,
                args.neg_cfg_pw,
                args.no_cfg_frac,
                device,
            )
        generated = _generated_queries(
            generator,
            labels,
            cfg_scales,
            generated_count=args.generated_count,
            seed=trial_seed + 29,
            device=device,
            use_bf16=args.use_bf16,
        )
        weight_negative = (
            (cfg_scales - 1.0).unsqueeze(1).expand(-1, args.negative_count)
            * float(args.generated_count - 1)
            / float(args.negative_count)
        )

        teacher_features = _extract_all(
            teacher,
            generated,
            positive,
            negative,
            device=device,
            use_bf16=args.use_bf16,
        )
        bridge_features = _extract_all(
            bridge,
            generated,
            positive,
            negative,
            device=device,
            use_bf16=args.use_bf16,
        )
        _accumulate_nested(
            clean_generated_values,
            _value_metrics(teacher_features[0], bridge_features[0]),
        )
        teacher_real = {
            name: torch.cat([teacher_features[1][name], teacher_features[2][name]])
            for name in teacher_features[1]
        }
        bridge_real = {
            name: torch.cat([bridge_features[1][name], bridge_features[2][name]])
            for name in bridge_features[1]
        }
        _accumulate_nested(clean_real_values, _value_metrics(teacher_real, bridge_real))
        trial_distance = _distance_metrics(
            teacher_features,
            bridge_features,
            batch_size=args.batch_size,
            generated_count=args.generated_count,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            weight_negative=weight_negative,
            R_list=args.R_list,
            stage_temperatures=stage_temperatures,
            max_values=args.max_distance_values,
            min_distance=args.min_distance,
        )
        _accumulate_nested(distance_values, trial_distance)
        del teacher_features, bridge_features, teacher_real, bridge_real

        if args.noisy_value_count > 0 and args.noise_std > 0:
            clean_pool = torch.cat([generated, positive, negative], dim=0)
            noisy_count = min(args.noisy_value_count, clean_pool.shape[0])
            # Evenly span generated and real regions rather than taking only
            # the leading generated queries.
            noisy_indices = torch.linspace(
                0, clean_pool.shape[0] - 1, steps=noisy_count, device=device
            ).round().long()
            noisy = clean_pool.index_select(0, noisy_indices)
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(trial_seed + 41)
                noisy = noisy + torch.randn_like(noisy) * args.noise_std
            with torch.no_grad(), _amp_context(device, args.use_bf16):
                teacher_noisy = teacher.get_activations(noisy, **ACTIVATION_KWARGS)
                bridge_noisy = bridge.get_activations(noisy, **ACTIVATION_KWARGS)
            _accumulate_nested(
                noisy_values, _value_metrics(teacher_noisy, bridge_noisy)
            )
            del teacher_noisy, bridge_noisy, noisy

        teacher_gradient = _gradient(
            teacher,
            generated,
            positive,
            negative,
            batch_size=args.batch_size,
            generated_count=args.generated_count,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            weight_negative=weight_negative,
            R_list=tuple(float(value) for value in args.R_list),
            stage_temperatures=stage_temperatures,
            device=device,
            use_bf16=args.use_bf16,
        )
        bridge_gradient = _gradient(
            bridge,
            generated,
            positive,
            negative,
            batch_size=args.batch_size,
            generated_count=args.generated_count,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            weight_negative=weight_negative,
            R_list=tuple(float(value) for value in args.R_list),
            stage_temperatures=stage_temperatures,
            device=device,
            use_bf16=args.use_bf16,
        )
        trial_gradient = _gradient_metrics(teacher_gradient, bridge_gradient)
        for metric, values in trial_gradient.items():
            gradient_values[metric].extend(values)
        trial_records.append(
            {
                "trial": trial,
                "seed": trial_seed,
                "labels": labels_list,
                "cfg_scales": [float(value) for value in cfg_scales.cpu().tolist()],
                "gradient": {
                    metric: _summary(values) for metric, values in trial_gradient.items()
                },
            }
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "trial": trial + 1,
                    "trials": args.trials,
                    "elapsed_seconds": elapsed,
                    "gradient_cosine_median": _summary(trial_gradient["cosine"])["median"],
                    "gradient_norm_ratio_median": _summary(trial_gradient["norm_ratio"])["median"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del teacher_gradient, bridge_gradient, generated, positive, negative
        if device.type == "cuda":
            torch.cuda.empty_cache()

    passed, gate_checks = _evaluate_gates(args, distance_values, gradient_values)
    report = {
        "format": "idrift.ssl_latent_bridge_validation.v1",
        "passed": passed,
        "backbone": args.backbone,
        "feature_checkpoint": str(Path(args.feature_checkpoint).resolve()),
        "feature_checkpoint_sha256": file_sha256(args.feature_checkpoint),
        "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
        "bridge_checkpoint_sha256": file_sha256(args.bridge_checkpoint),
        "bridge_provenance": bridge.bridge_metadata.get("teacher", {}),
        "bridge_training": bridge.bridge_metadata.get("training", {}),
        "generator": {
            "config": str(Path(args.generator_config).resolve()),
            "checkpoint": str(Path(args.generator_checkpoint).resolve())
            if args.generator_checkpoint
            else None,
            "seed": args.generator_seed,
        },
        "cache": {
            "path": str(Path(args.cache_path).resolve()),
            "split": args.cache_split,
            "classes": len(cache.by_class),
            "variants": len(cache.all_variants),
        },
        "geometry": {
            "trials": args.trials,
            "B": args.batch_size,
            "G": args.generated_count,
            "P": args.positive_count,
            "N": args.negative_count,
            "R_list": [float(value) for value in args.R_list],
            "stage_temperatures": stage_temperatures,
            "activation_objectives": 35,
            "activation_kwargs": ACTIVATION_KWARGS,
            "negative_weighting": "(cfg-1)*(G-1)/N",
        },
        "runtime_audit": {
            "forbidden_prefix_modules": forbidden_runtime_modules,
            "teacher_use_remat": True,
            "teacher_microbatch_size": args.teacher_microbatch_size,
            "bridge_microbatch_size": args.bridge_microbatch_size,
        },
        "clean_generated_value": _summarize_nested(clean_generated_values),
        "clean_real_value": _summarize_nested(clean_real_values),
        "noisy_value": _summarize_nested(noisy_values),
        "distance_and_affinity": _summarize_nested(distance_values),
        "gradient": {
            metric: _summary(values) for metric, values in gradient_values.items()
        },
        "gate_checks": gate_checks,
        "trial_records": trial_records,
        "elapsed_seconds": time.perf_counter() - started,
        "args": vars(args),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "passed": passed,
                "output": str(output_path),
                "gradient": report["gradient"],
                "failed_checks": [
                    check["name"] for check in gate_checks if not check["passed"]
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.fail_on_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
