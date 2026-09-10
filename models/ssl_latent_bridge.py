"""Distilled latent-to-ResNet bridge for DINO/MoCo drift features.

The regular :mod:`models.ssl_resnet` path must decode a 4x32x32 Stable
Diffusion latent to a 256px RGB image and then execute the ResNet stem,
``layer1`` and ``layer2`` before it can compute the stage-3/4 drift features.
This module replaces that entire prefix with a small learned bridge which
predicts the *post-layer2* tensor (512x32x32) directly from the latent.

Only the official, frozen ``layer3`` and ``layer4`` modules are retained at
runtime.  In particular, :class:`SSLLatentBridgeFeatureExtractor` never owns a
VAE, ResNet stem, ``layer1`` or ``layer2``.  Its ``get_activations`` output is
compatible with ``SSLResNetFeatureExtractor`` for stage-3/4-only drift runs.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import Bottleneck

from models.mae_resnet import safe_mean, safe_mean_std, safe_rms, safe_std
from models.ssl_resnet import _load_backbone_state, canonical_ssl_backbone


BRIDGE_CHECKPOINT_FORMAT = "idrift.ssl_latent_bridge.v1"
_SUPPORTED_STAGES = ("stage3", "stage4")
# Stable-Diffusion latents are not channelwise standard normal.  These values
# were measured over 16 deterministically spread cache entries in each of the
# 1,000 ImageNet train classes (32,768,000 spatial values/channel, including
# cached flips).  Keeping the affine in the checkpoint makes training/runtime
# input conventions impossible to accidentally diverge.
_IMAGENET256_LATENT_MEAN = (
    0.15693267017015933,
    -0.059107416547183676,
    0.041562387369274804,
    0.0675984769803581,
)
_IMAGENET256_LATENT_STD = (
    0.8843920143059556,
    0.9712291705863965,
    0.7205902124550205,
    0.7278769964073596,
)


def file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 digest of ``path`` without reading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _group_count(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return max(groups, 1)


@dataclass(frozen=True)
class LatentBridgeConfig:
    """Serializable architecture definition for a latent-to-layer2 bridge."""

    in_channels: int = 4
    out_channels: int = 512
    width: int = 256
    depth: int = 8
    expansion: int = 2
    kernel_size: int = 7
    norm_groups: int = 32
    layer_scale_init: float = 1e-3
    output_activation: str = "relu"
    input_mean: Tuple[float, ...] = _IMAGENET256_LATENT_MEAN
    input_std: Tuple[float, ...] = _IMAGENET256_LATENT_STD

    def validate(self) -> "LatentBridgeConfig":
        for name in ("in_channels", "out_channels", "width", "depth", "expansion"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"bridge {name} must be positive")
        if int(self.kernel_size) <= 0 or int(self.kernel_size) % 2 == 0:
            raise ValueError("bridge kernel_size must be a positive odd integer")
        if int(self.norm_groups) <= 0:
            raise ValueError("bridge norm_groups must be positive")
        if not math.isfinite(float(self.layer_scale_init)) or self.layer_scale_init < 0:
            raise ValueError("bridge layer_scale_init must be finite and >= 0")
        if str(self.output_activation) not in {"relu", "identity"}:
            raise ValueError("bridge output_activation must be 'relu' or 'identity'")
        if len(self.input_mean) != int(self.in_channels):
            raise ValueError("bridge input_mean length must match in_channels")
        if len(self.input_std) != int(self.in_channels):
            raise ValueError("bridge input_std length must match in_channels")
        if not all(math.isfinite(float(value)) for value in self.input_mean):
            raise ValueError("bridge input_mean values must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in self.input_std
        ):
            raise ValueError("bridge input_std values must be finite and positive")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LatentBridgeConfig":
        fields = cls.__dataclass_fields__
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"Unknown latent bridge config fields: {sorted(unknown)}")
        normalized = dict(values)
        for name in ("input_mean", "input_std"):
            if name in normalized:
                normalized[name] = tuple(float(value) for value in normalized[name])
        return cls(**normalized).validate()


class _BridgeBlock(nn.Module):
    """Wide-receptive-field residual block with inexpensive channel mixing."""

    def __init__(self, config: LatentBridgeConfig) -> None:
        super().__init__()
        width = int(config.width)
        hidden = width * int(config.expansion)
        kernel = int(config.kernel_size)
        self.depthwise = nn.Conv2d(
            width,
            width,
            kernel,
            padding=kernel // 2,
            groups=width,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            _group_count(width, int(config.norm_groups)),
            width,
            eps=1e-6,
        )
        self.expand = nn.Conv2d(width, hidden, 1)
        self.project = nn.Conv2d(hidden, width, 1)
        self.layer_scale = nn.Parameter(
            torch.full((1, width, 1, 1), float(config.layer_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = F.silu(self.expand(x))
        x = self.project(x)
        return residual + x * self.layer_scale


class LatentToStage2Bridge(nn.Module):
    """Map scaled SD latents directly to the frozen R50 layer2 interface."""

    def __init__(self, config: LatentBridgeConfig | Mapping[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = LatentBridgeConfig()
        elif not isinstance(config, LatentBridgeConfig):
            config = LatentBridgeConfig.from_mapping(config)
        self.config = config.validate()

        width = int(self.config.width)
        self.input_projection = nn.Conv2d(
            int(self.config.in_channels), width, 3, padding=1, bias=False
        )
        self.input_norm = nn.GroupNorm(
            _group_count(width, int(self.config.norm_groups)), width, eps=1e-6
        )
        self.blocks = nn.Sequential(
            *[_BridgeBlock(self.config) for _ in range(int(self.config.depth))]
        )
        self.output_norm = nn.GroupNorm(
            _group_count(width, int(self.config.norm_groups)), width, eps=1e-6
        )
        self.output_projection = nn.Conv2d(
            width, int(self.config.out_channels), 1, bias=True
        )
        self.register_buffer(
            "input_mean",
            torch.tensor(self.config.input_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(self.config.input_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=True,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(
                f"latent bridge expects BCHW input, got shape {tuple(latent.shape)}"
            )
        if latent.shape[1] != int(self.config.in_channels):
            raise ValueError(
                f"latent bridge expects {self.config.in_channels} channels, "
                f"got {latent.shape[1]}"
            )
        x = (latent - self.input_mean) / self.input_std
        x = F.silu(self.input_norm(self.input_projection(x)))
        x = self.blocks(x)
        x = self.output_projection(F.silu(self.output_norm(x)))
        if self.config.output_activation == "relu":
            x = F.relu(x)
        return x


def _resnet50_stage(
    in_channels: int,
    planes: int,
    blocks: int,
    *,
    stride: int,
) -> nn.Sequential:
    """Construct one torchvision-compatible ResNet-50 bottleneck stage."""
    out_channels = planes * Bottleneck.expansion
    downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
    modules: List[nn.Module] = [
        Bottleneck(
            in_channels,
            planes,
            stride=stride,
            downsample=downsample,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        )
    ]
    for _ in range(1, blocks):
        modules.append(
            Bottleneck(
                out_channels,
                planes,
                groups=1,
                base_width=64,
                dilation=1,
                norm_layer=nn.BatchNorm2d,
            )
        )
    return nn.Sequential(*modules)


class FrozenResNet50Stage34(nn.Module):
    """Official R50 layer3/layer4 tail, strict-loaded without early modules."""

    def __init__(self, backbone_name: str, checkpoint_path: str | Path) -> None:
        super().__init__()
        self.backbone_name = canonical_ssl_backbone(backbone_name)
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.layer3 = _resnet50_stage(512, 256, 6, stride=2)
        self.layer4 = _resnet50_stage(1024, 512, 3, stride=2)

        full_state = _load_backbone_state(self.checkpoint_path, self.backbone_name)
        tail_state = {
            key: value
            for key, value in full_state.items()
            if key.startswith("layer3.") or key.startswith("layer4.")
        }
        # strict=True proves every retained stage3/4 tensor came from the
        # official checkpoint and that no retained tensor was silently skipped.
        self.load_state_dict(tail_state, strict=True)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, layer2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        layer3 = self.layer3(layer2)
        return layer3, self.layer4(layer3)


def save_bridge_checkpoint(
    path: str | Path,
    bridge: LatentToStage2Bridge,
    *,
    backbone_name: str,
    teacher_checkpoint_path: str | Path,
    vae_model_id: str,
    vae_revision: Optional[str],
    metrics: Optional[Mapping[str, float]] = None,
    training_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically save a self-describing, runtime-only bridge checkpoint."""
    teacher_path = Path(teacher_checkpoint_path).resolve()
    canonical_name = canonical_ssl_backbone(backbone_name)
    checkpoint_data: Dict[str, Any] = {
        "format": BRIDGE_CHECKPOINT_FORMAT,
        "bridge_config": asdict(bridge.config),
        "bridge_state_dict": {
            key: value.detach().cpu() for key, value in bridge.state_dict().items()
        },
        "teacher": {
            "backbone_name": canonical_name,
            "feature_checkpoint_path": str(teacher_path),
            "feature_checkpoint_sha256": file_sha256(teacher_path),
            "vae_model_id": str(vae_model_id),
            "vae_revision": str(vae_revision) if vae_revision else None,
            "architecture": "torchvision.resnet50",
            "target_interface": "post_layer2_relu",
            "target_shape_256px": [512, 32, 32],
            "runtime_tail": ["layer3", "layer4"],
        },
        "metrics": dict(metrics or {}),
        "training": dict(training_metadata or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(checkpoint_data, temporary)
    os.replace(temporary, destination)
    return checkpoint_data


def load_bridge_checkpoint(
    path: str | Path,
    *,
    expected_backbone_name: Optional[str] = None,
    expected_teacher_sha256: Optional[str] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[LatentToStage2Bridge, Dict[str, Any]]:
    """Strict-load a bridge and validate its provenance metadata."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Latent bridge checkpoint not found: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(data, Mapping) or data.get("format") != BRIDGE_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported latent bridge checkpoint format in {checkpoint_path}"
        )
    if not isinstance(data.get("teacher"), Mapping):
        raise ValueError("Latent bridge checkpoint is missing teacher provenance")
    teacher = dict(data["teacher"])
    saved_backbone = canonical_ssl_backbone(str(teacher.get("backbone_name", "")))
    if expected_backbone_name is not None:
        expected = canonical_ssl_backbone(expected_backbone_name)
        if saved_backbone != expected:
            raise ValueError(
                f"Bridge was trained for {saved_backbone}, not requested {expected}"
            )
    saved_sha = str(teacher.get("feature_checkpoint_sha256", ""))
    if not saved_sha:
        raise ValueError("Latent bridge checkpoint is missing teacher SHA256")
    if expected_teacher_sha256 is not None and saved_sha != expected_teacher_sha256:
        raise ValueError(
            "Bridge teacher checkpoint SHA256 mismatch: "
            f"saved={saved_sha}, expected={expected_teacher_sha256}"
        )
    config_raw = data.get("bridge_config")
    state_raw = data.get("bridge_state_dict")
    if not isinstance(config_raw, Mapping) or not isinstance(state_raw, Mapping):
        raise ValueError("Latent bridge checkpoint is missing config or weights")
    bridge = LatentToStage2Bridge(LatentBridgeConfig.from_mapping(config_raw))
    bridge.load_state_dict(dict(state_raw), strict=True)
    return bridge, dict(data)


class SSLLatentBridgeFeatureExtractor(nn.Module):
    """VAE-free, early-stage-free DINO/MoCo stage3/4 feature extractor."""

    def __init__(
        self,
        backbone_name: str,
        feature_checkpoint_path: str | Path,
        bridge_checkpoint_path: str | Path,
        *,
        use_bf16: bool = True,
        use_remat: bool = False,
        microbatch_size: int = 32,
        spatial_pool: int = 2,
        include_norm_x: bool = False,
        verify_teacher_sha256: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_ssl_backbone(backbone_name)
        self.checkpoint_path = str(Path(feature_checkpoint_path).resolve())
        expected_sha = (
            file_sha256(self.checkpoint_path) if verify_teacher_sha256 else None
        )
        bridge, bridge_metadata = load_bridge_checkpoint(
            bridge_checkpoint_path,
            expected_backbone_name=self.backbone_name,
            expected_teacher_sha256=expected_sha,
        )
        self.bridge = bridge
        self.tail = FrozenResNet50Stage34(
            self.backbone_name, self.checkpoint_path
        )
        self.bridge_checkpoint_path = str(Path(bridge_checkpoint_path).resolve())
        self.bridge_metadata = bridge_metadata
        self.use_bf16 = bool(use_bf16)
        self.use_remat = bool(use_remat)
        self.microbatch_size = int(microbatch_size)
        self.spatial_pool = int(spatial_pool)
        self.include_norm_x = bool(include_norm_x)
        self.base_channels = 256
        self.use_latent = True

        if self.microbatch_size < 0:
            raise ValueError("feature microbatch size must be >= 0")
        if self.spatial_pool < 1:
            raise ValueError("feature spatial_pool must be >= 1")
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _active_stage_set(active_stages: Optional[Sequence[str]]) -> set[str]:
        if active_stages is None:
            return set(_SUPPORTED_STAGES)
        result = {str(stage).lower().strip() for stage in active_stages}
        unsupported = result - set(_SUPPORTED_STAGES)
        if unsupported:
            raise ValueError(
                "Latent bridge runtime supports only stage3/stage4; "
                f"got unsupported stages {sorted(unsupported)}"
            )
        return result

    def _selected_map_names(
        self,
        every_k_block: float,
        active: set[str],
        exclude_terminal_block: bool,
    ) -> Tuple[str, ...]:
        names = [
            f"layer{index}"
            for index in (3, 4)
            if f"stage{index}" in active
        ]
        need_blocks = (
            isinstance(every_k_block, (int, float))
            and not math.isinf(float(every_k_block))
            and every_k_block >= 1
        )
        if need_blocks:
            k = int(every_k_block)
            for stage_index, layer in ((3, self.tail.layer3), (4, self.tail.layer4)):
                if f"stage{stage_index}" not in active:
                    continue
                for block_index in range(1, len(layer) + 1):
                    terminal = block_index == len(layer)
                    if block_index % k == 0 and not (
                        exclude_terminal_block and terminal
                    ):
                        names.append(f"layer{stage_index}_blk{block_index}")
        return tuple(names)

    def _pool_export(self, feature: torch.Tensor) -> torch.Tensor:
        if self.spatial_pool == 1:
            return feature
        height, width = feature.shape[-2:]
        if height % self.spatial_pool or width % self.spatial_pool:
            raise ValueError(
                f"Cannot spatially pool feature map {height}x{width} "
                f"by {self.spatial_pool}"
            )
        return F.avg_pool2d(
            feature, kernel_size=self.spatial_pool, stride=self.spatial_pool
        )

    def _forward_selected_maps(
        self, latent: torch.Tensor, names: Tuple[str, ...]
    ) -> Tuple[torch.Tensor, ...]:
        selected = set(names)
        y = self.bridge(latent)
        terminal: Dict[str, torch.Tensor] = {}
        blocks: Dict[str, torch.Tensor] = {}
        for stage_index, layer in ((3, self.tail.layer3), (4, self.tail.layer4)):
            layer_name = f"layer{stage_index}"
            for block_index, block in enumerate(layer, start=1):
                y = block(y)
                block_name = f"{layer_name}_blk{block_index}"
                if block_name in selected:
                    blocks[block_name] = self._pool_export(y)
            if layer_name in selected:
                terminal[layer_name] = self._pool_export(y)
        merged = {**terminal, **blocks}
        return tuple(merged[name] for name in names)

    @staticmethod
    def _stage_for_name(name: str) -> Optional[str]:
        if name == "layer3" or name.startswith("layer3_"):
            return "stage3"
        if name == "layer4" or name.startswith("layer4_"):
            return "stage4"
        return None

    def _process_feature(
        self,
        name: str,
        feature: torch.Tensor,
        *,
        patch_mean_size: Sequence[int],
        patch_std_size: Sequence[int],
        use_std: bool,
        use_mean: bool,
        stage_adapters: Optional[nn.ModuleDict],
    ) -> Dict[str, torch.Tensor]:
        stage = self._stage_for_name(name)
        if stage_adapters is not None and stage in stage_adapters:
            feature = stage_adapters[stage](feature)
        result: Dict[str, torch.Tensor] = {
            name: rearrange(feature, "b c h w -> b (h w) c")
        }
        _, _, height, width = feature.shape
        if use_mean and use_std:
            mean, std = safe_mean_std(feature, dim=(2, 3))
            result[f"{name}_mean"] = mean.unsqueeze(1)
            result[f"{name}_std"] = std.unsqueeze(1)
        else:
            if use_mean:
                result[f"{name}_mean"] = safe_mean(feature, dim=(2, 3)).unsqueeze(1)
            if use_std:
                result[f"{name}_std"] = safe_std(feature, dim=(2, 3)).unsqueeze(1)

        cached_patch_std: Dict[int, torch.Tensor] = {}
        for raw_size in patch_mean_size:
            size = int(raw_size)
            if size <= 0:
                raise ValueError("patch mean sizes must be positive")
            if height % size == 0 and width % size == 0:
                patches = rearrange(
                    feature,
                    "b c (h ph) (w pw) -> b (h w) (ph pw) c",
                    ph=size,
                    pw=size,
                )
                if use_std and size in patch_std_size:
                    mean, std = safe_mean_std(patches, dim=2)
                    result[f"{name}_mean_{size}"] = mean
                    cached_patch_std[size] = std
                else:
                    result[f"{name}_mean_{size}"] = safe_mean(patches, dim=2)
        for raw_size in patch_std_size:
            size = int(raw_size)
            if size <= 0:
                raise ValueError("patch std sizes must be positive")
            if height % size == 0 and width % size == 0:
                if size in cached_patch_std:
                    result[f"{name}_std_{size}"] = cached_patch_std[size]
                else:
                    patches = rearrange(
                        feature,
                        "b c (h ph) (w pw) -> b (h w) (ph pw) c",
                        ph=size,
                        pw=size,
                    )
                    result[f"{name}_std_{size}"] = safe_std(patches, dim=2)
        return result

    def get_activations(
        self,
        x: torch.Tensor,
        patch_mean_size: Optional[List[int]] = None,
        patch_std_size: Optional[List[int]] = None,
        use_std: bool = True,
        use_mean: bool = True,
        with_global: bool = False,
        with_norm_x: Optional[bool] = None,
        every_k_block: float = 2,
        exclude_terminal_block: bool = False,
        active_stages: Optional[List[str]] = None,
        stage_adapters: Optional[nn.ModuleDict] = None,
        return_stage_features: bool = False,
    ) -> Dict[str, torch.Tensor] | Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """Return the same stage3/4 activation schema as the decoded path."""
        patch_mean_size = [2, 4] if patch_mean_size is None else list(patch_mean_size)
        patch_std_size = [2, 4] if patch_std_size is None else list(patch_std_size)
        active = self._active_stage_set(active_stages)
        names = self._selected_map_names(
            every_k_block, active, bool(exclude_terminal_block)
        )

        output_parts: Dict[str, List[torch.Tensor]] = {}
        stage_parts: Dict[str, List[torch.Tensor]] = {}
        if with_global:
            output_parts["global"] = [rearrange(x, "b c h w -> b 1 (c h w)")]
        if self.include_norm_x if with_norm_x is None else bool(with_norm_x):
            output_parts["norm_x"] = [safe_rms(x, dim=(2, 3)).unsqueeze(1)]

        chunk_size = self.microbatch_size if self.microbatch_size > 0 else int(x.shape[0])
        for start in range(0, int(x.shape[0]), chunk_size):
            chunk = x[start : start + chunk_size]

            def extract(input_chunk: torch.Tensor) -> Tuple[torch.Tensor, ...]:
                return self._forward_selected_maps(input_chunk, names)

            if self.use_remat and torch.is_grad_enabled() and chunk.requires_grad:
                maps = checkpoint(extract, chunk, use_reentrant=False)
            else:
                maps = extract(chunk)
            if isinstance(maps, torch.Tensor):
                maps = (maps,)

            for name, feature in zip(names, maps):
                if return_stage_features and name in {"layer3", "layer4"}:
                    stage_parts.setdefault(name, []).append(feature)
                processed = self._process_feature(
                    name,
                    feature,
                    patch_mean_size=patch_mean_size,
                    patch_std_size=patch_std_size,
                    use_std=use_std,
                    use_mean=use_mean,
                    stage_adapters=stage_adapters,
                )
                for key, value in processed.items():
                    output_parts.setdefault(key, []).append(value)

        output = {
            key: values[0] if len(values) == 1 else torch.cat(values, dim=0)
            for key, values in output_parts.items()
        }
        if return_stage_features:
            stages = {
                key: values[0] if len(values) == 1 else torch.cat(values, dim=0)
                for key, values in stage_parts.items()
            }
            return output, stages
        return output


@torch.no_grad()
def stage_map_distillation_metrics(
    predicted: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    stages: Sequence[str] = ("layer3", "layer4"),
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Cosine, normalized RMSE and pair-distance correlation on held-out maps."""
    metrics: Dict[str, float] = {}
    for stage in stages:
        if stage not in predicted or stage not in target:
            raise KeyError(f"Missing held-out stage map {stage!r}")
        pred = predicted[stage].detach().float().flatten(1)
        ref = target[stage].detach().float().flatten(1)
        if pred.shape != ref.shape:
            raise ValueError(
                f"Held-out map shape mismatch for {stage}: "
                f"{tuple(pred.shape)} vs {tuple(ref.shape)}"
            )
        cosine = F.cosine_similarity(pred, ref, dim=1, eps=eps).mean()
        nrmse = (pred - ref).square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(eps)
        metrics[f"{stage}/raw_cosine"] = float(cosine.item())
        metrics[f"{stage}/normalized_rmse"] = float(nrmse.item())

        if pred.shape[0] >= 3:
            pred_dist = torch.pdist(pred, p=2)
            ref_dist = torch.pdist(ref, p=2)
            pred_dist = pred_dist / pred_dist.mean().clamp_min(eps)
            ref_dist = ref_dist / ref_dist.mean().clamp_min(eps)
            pred_centered = pred_dist - pred_dist.mean()
            ref_centered = ref_dist - ref_dist.mean()
            correlation = (pred_centered * ref_centered).sum() / (
                pred_centered.square().sum().sqrt()
                * ref_centered.square().sum().sqrt()
            ).clamp_min(eps)
            distance_nrmse = (
                (pred_dist - ref_dist).square().mean().sqrt()
                / ref_dist.square().mean().sqrt().clamp_min(eps)
            )
            metrics[f"{stage}/pair_distance_correlation"] = float(correlation.item())
            metrics[f"{stage}/pair_distance_normalized_rmse"] = float(
                distance_nrmse.item()
            )
        else:
            metrics[f"{stage}/pair_distance_correlation"] = float("nan")
            metrics[f"{stage}/pair_distance_normalized_rmse"] = float("nan")
    return metrics


def compare_reverse_drift_latent_gradients(
    teacher: nn.Module,
    bridge_extractor: nn.Module,
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    batch_size: int,
    generated_count: int,
    positive_count: int,
    negative_count: int,
    feature_names: Optional[Sequence[str]] = None,
    R_list: Tuple[float, ...] = (0.2, 0.05, 0.02),
    activation_kwargs: Optional[Mapping[str, Any]] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compare teacher/bridge gradients for a tiny reverse-drift B/G/P/N grid.

    This intentionally calls the same reverse-drift primitive used by the
    generator training.  By default all 35 stage3/4 raw/stat objectives are
    included.  It is a quality gate for the surrogate: feature reconstruction
    alone can look good while inducing a poor latent update direction.
    """
    from drifting_core.imagenet_loss import drift_loss_imagenet

    expected = (
        batch_size * generated_count,
        batch_size * positive_count,
        batch_size * negative_count,
    )
    actual = (generated.shape[0], positive.shape[0], negative.shape[0])
    if actual != expected:
        raise ValueError(f"B/G/P/N flattened sizes mismatch: got {actual}, expected {expected}")
    kwargs = {
        "patch_mean_size": [2, 4],
        "patch_std_size": [2, 4],
        "use_std": True,
        "use_mean": True,
        "with_global": False,
        "with_norm_x": False,
        "every_k_block": 2,
        "exclude_terminal_block": True,
        "active_stages": ["stage3", "stage4"],
    }
    kwargs.update(dict(activation_kwargs or {}))

    def gradient_for(extractor: nn.Module) -> torch.Tensor:
        queries = generated.detach().clone().requires_grad_(True)
        with torch.no_grad():
            fixed = torch.cat([positive, negative], dim=0)
            fixed_features = extractor.get_activations(fixed, **kwargs)
            positive_features = {
                name: value[: batch_size * positive_count]
                for name, value in fixed_features.items()
            }
            negative_features = {
                name: value[batch_size * positive_count :]
                for name, value in fixed_features.items()
            }
        query_features = extractor.get_activations(queries, **kwargs)
        total = queries.new_zeros((), dtype=torch.float32)
        selected_names = (
            tuple(query_features.keys())
            if feature_names is None
            else tuple(feature_names)
        )
        for name in selected_names:
            if name not in positive_features or name not in negative_features:
                raise KeyError(f"Fixed features are missing drift objective {name!r}")
            gen_map = query_features[name]
            pos_map = positive_features[name]
            neg_map = negative_features[name]
            tokens, channels = gen_map.shape[1:]
            gen_grid = rearrange(
                gen_map,
                "(b g) t d -> (b t) g d",
                b=batch_size,
                g=generated_count,
            )
            pos_grid = rearrange(
                pos_map,
                "(b p) t d -> (b t) p d",
                b=batch_size,
                p=positive_count,
            )
            neg_grid = rearrange(
                neg_map,
                "(b n) t d -> (b t) n d",
                b=batch_size,
                n=negative_count,
            )
            if gen_grid.shape != (batch_size * tokens, generated_count, channels):
                raise RuntimeError("Unexpected generated activation reshape")
            loss, _ = drift_loss_imagenet(
                gen_grid,
                pos_grid,
                neg_grid,
                R_list=R_list,
                global_scale_stats=False,
                global_fnorm_stats=False,
                collect_diagnostics=False,
            )
            total = total + loss.mean()
        gradient = torch.autograd.grad(total, queries, create_graph=False)[0]
        return gradient.detach().float()

    teacher_gradient = gradient_for(teacher)
    bridge_gradient = gradient_for(bridge_extractor)
    teacher_flat = teacher_gradient.flatten(1)
    bridge_flat = bridge_gradient.flatten(1)
    cosine = F.cosine_similarity(
        teacher_flat, bridge_flat, dim=1, eps=eps
    ).mean()
    relative_rmse = (
        (teacher_gradient - bridge_gradient).square().mean().sqrt()
        / teacher_gradient.square().mean().sqrt().clamp_min(eps)
    )
    return {
        "drift_gradient/cosine": float(cosine.item()),
        "drift_gradient/normalized_rmse": float(relative_rmse.item()),
        "drift_gradient/teacher_rms": float(
            teacher_gradient.square().mean().sqrt().item()
        ),
        "drift_gradient/bridge_rms": float(
            bridge_gradient.square().mean().sqrt().item()
        ),
    }


__all__ = [
    "BRIDGE_CHECKPOINT_FORMAT",
    "FrozenResNet50Stage34",
    "LatentBridgeConfig",
    "LatentToStage2Bridge",
    "SSLLatentBridgeFeatureExtractor",
    "compare_reverse_drift_latent_gradients",
    "file_sha256",
    "load_bridge_checkpoint",
    "save_bridge_checkpoint",
    "stage_map_distillation_metrics",
]
