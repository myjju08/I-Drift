"""Frozen ImageNet SSL ResNet-50 feature extractors for latent-space drifting.

DINO and MoCo operate on RGB images.  For a latent generator this module keeps
the SD-VAE decoder and ResNet weights frozen while preserving gradients with
respect to generated latents.  Exported feature names and tensor shapes follow
``MAEResNet.get_activations`` so the existing drifting loss can consume either
backbone without special cases.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

from models.mae_resnet import safe_mean, safe_mean_std, safe_rms, safe_std
from vae_imagenet import load_vae


_VALID_STAGES = tuple(f"stage{i}" for i in range(1, 5))
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def derive_ssl_map_features(
    name: str,
    feature: torch.Tensor,
    *,
    patch_mean_size: Sequence[int],
    patch_std_size: Sequence[int],
    use_std: bool,
    use_mean: bool,
) -> Dict[str, torch.Tensor]:
    """Derive the exact drift-feature family for one SSL spatial map.

    This is shared by the frozen generator metric and the online adapter
    objective.  Keeping one implementation is important: the adapter's direct
    field objective must see the same raw/mean/std/patch representations that
    the generator drift loss sees, rather than a terminal-layer proxy.
    """
    result: Dict[str, torch.Tensor] = {
        name: rearrange(feature, "b c h w -> b (h w) c")
    }
    _, _, height, width = feature.shape
    spatial_mean: Optional[torch.Tensor] = None
    spatial_std: Optional[torch.Tensor] = None
    if use_mean and use_std:
        spatial_mean, spatial_std = safe_mean_std(feature, dim=(2, 3))
        spatial_mean = spatial_mean.unsqueeze(1)
        spatial_std = spatial_std.unsqueeze(1)
        result[f"{name}_mean"] = spatial_mean
        result[f"{name}_std"] = spatial_std
    else:
        if use_mean:
            spatial_mean = safe_mean(feature, dim=(2, 3)).unsqueeze(1)
            result[f"{name}_mean"] = spatial_mean
        if use_std:
            spatial_std = safe_std(feature, dim=(2, 3)).unsqueeze(1)
            result[f"{name}_std"] = spatial_std

    cached_patch_std: Dict[int, torch.Tensor] = {}
    for raw_size in patch_mean_size:
        size = int(raw_size)
        if size <= 0:
            raise ValueError("patch mean sizes must be positive")
        if height % size == 0 and width % size == 0:
            full_map_patch = height == size and width == size
            if full_map_patch:
                result[f"{name}_mean_{size}"] = (
                    spatial_mean
                    if spatial_mean is not None
                    else safe_mean(feature, dim=(2, 3)).unsqueeze(1)
                )
                if use_std and size in patch_std_size:
                    cached_patch_std[size] = (
                        spatial_std
                        if spatial_std is not None
                        else safe_std(feature, dim=(2, 3)).unsqueeze(1)
                    )
            else:
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
            elif height == size and width == size and spatial_std is not None:
                result[f"{name}_std_{size}"] = spatial_std
            else:
                patches = rearrange(
                    feature,
                    "b c (h ph) (w pw) -> b (h w) (ph pw) c",
                    ph=size,
                    pw=size,
                )
                result[f"{name}_std_{size}"] = safe_std(patches, dim=2)
    return result


def canonical_ssl_backbone(name: str) -> str:
    """Return a stable backbone name while accepting convenient aliases."""
    normalized = str(name).lower().strip().replace("-", "_")
    aliases = {
        "dino": "dino_resnet50",
        "dino_r50": "dino_resnet50",
        "dino_resnet50": "dino_resnet50",
        "moco": "moco_v2_resnet50",
        "moco_r50": "moco_v2_resnet50",
        "moco_v2": "moco_v2_resnet50",
        "moco_v2_r50": "moco_v2_resnet50",
        "moco_v2_resnet50": "moco_v2_resnet50",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown SSL feature backbone {name!r}; choose dino_resnet50 "
            "or moco_v2_resnet50"
        )
    return aliases[normalized]


def _load_backbone_state(path: str | Path, backbone_name: str) -> Dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Feature checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if backbone_name == "dino_resnet50":
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"Unsupported DINO checkpoint format: {checkpoint_path}")
        result = dict(state)
    elif backbone_name == "moco_v2_resnet50":
        if not isinstance(state, Mapping) or not isinstance(state.get("state_dict"), Mapping):
            raise ValueError(f"Unsupported MoCo checkpoint format: {checkpoint_path}")
        prefix = "module.encoder_q."
        result = {
            key[len(prefix):]: value
            for key, value in state["state_dict"].items()
            if key.startswith(prefix) and not key.startswith(prefix + "fc.")
        }
    else:  # Guard callers that bypass canonical_ssl_backbone.
        raise ValueError(f"Unsupported SSL backbone: {backbone_name}")

    if not result:
        raise ValueError(f"No ResNet backbone weights found in {checkpoint_path}")
    return result


class SSLResNetFeatureExtractor(nn.Module):
    """DINO/MoCo ResNet-50 with optional differentiable latent decoding.

    ``spatial_pool=2`` maps the standard 256px ResNet grids
    ``64/64/32/16/8`` to the latent MAE grids ``32/32/16/8/4``.  This keeps
    token geometry and the 2x2/4x4 patch-stat objectives matched across the
    feature-encoder comparison.
    """

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str | Path,
        *,
        use_latent: bool = True,
        use_bf16: bool = True,
        use_remat: bool = True,
        vae_gradient_checkpointing: Optional[bool] = None,
        microbatch_size: int = 16,
        real_microbatch_size: Optional[int] = None,
        generated_microbatch_size: Optional[int] = None,
        spatial_pool: int = 2,
        include_norm_x: bool = True,
        clip_decoded_pixels: bool = False,
        vae_model_id: str = "stabilityai/sd-vae-ft-mse",
        vae_revision: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_ssl_backbone(backbone_name)
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.use_latent = bool(use_latent)
        self.use_bf16 = bool(use_bf16)
        self.use_remat = bool(use_remat)
        # Backward compatibility: before this option existed, VAE gradient
        # checkpointing was enabled exactly when the outer feature extractor
        # was rematerialized.  An explicit value lets callers keep the VAE's
        # fine-grained checkpointing while disabling the much broader outer
        # checkpoint around VAE + ResNet.
        self.vae_gradient_checkpointing = (
            self.use_remat
            if vae_gradient_checkpointing is None
            else bool(vae_gradient_checkpointing)
        )
        # Keep ``microbatch_size`` as the backward-compatible common default,
        # while allowing the detached real path to use a larger batch than the
        # activation-retaining generated path.  The call site is identified
        # from autograd state in ``get_activations``; callers do not need to
        # select the path explicitly.
        self.microbatch_size = int(microbatch_size)
        self.real_microbatch_size = int(
            self.microbatch_size
            if real_microbatch_size is None
            else real_microbatch_size
        )
        self.generated_microbatch_size = int(
            self.microbatch_size
            if generated_microbatch_size is None
            else generated_microbatch_size
        )
        self.spatial_pool = int(spatial_pool)
        self.include_norm_x = bool(include_norm_x)
        self.clip_decoded_pixels = bool(clip_decoded_pixels)
        self.vae_model_id = str(vae_model_id)
        self.vae_revision = str(vae_revision) if vae_revision else None
        # Bottleneck ResNet stage widths are 256, 512, 1024, 2048.  Existing
        # optional adapter/GAN builders derive those widths from base_channels.
        self.base_channels = 256

        if self.microbatch_size < 0:
            raise ValueError("feature microbatch size must be >= 0")
        if self.real_microbatch_size < 0:
            raise ValueError("real feature microbatch size must be >= 0")
        if self.generated_microbatch_size < 0:
            raise ValueError("generated feature microbatch size must be >= 0")
        if self.spatial_pool < 1:
            raise ValueError("feature spatial_pool must be >= 1")

        backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()
        state_dict = _load_backbone_state(self.checkpoint_path, self.backbone_name)
        backbone.load_state_dict(state_dict, strict=True)
        self.backbone = backbone

        self.vae: Optional[nn.Module]
        if self.use_latent:
            self.vae = load_vae(
                device=device,
                model_id=self.vae_model_id,
                revision=self.vae_revision,
            )
            if self.vae_gradient_checkpointing and hasattr(
                self.vae, "enable_gradient_checkpointing"
            ):
                self.vae.enable_gradient_checkpointing()
            elif hasattr(self.vae, "disable_gradient_checkpointing"):
                # ``load_vae`` is process-global and can return an instance that
                # a previously constructed extractor already enabled.
                self.vae.disable_gradient_checkpointing()
        else:
            self.vae = None

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _active_stage_set(active_stages: Optional[Sequence[str]]) -> Optional[set[str]]:
        if active_stages is None:
            return None
        result = {str(stage).lower().strip() for stage in active_stages}
        unknown = result - set(_VALID_STAGES)
        if unknown:
            raise ValueError(
                f"Unknown active_stages entries: {sorted(unknown)}; "
                f"expected a subset of {list(_VALID_STAGES)}"
            )
        return result

    @staticmethod
    def _stage_for_name(name: str) -> Optional[str]:
        if name == "conv1" or name.startswith("conv1_"):
            return "stage1"
        for index in range(1, 5):
            prefix = f"layer{index}"
            if name == prefix or name.startswith(prefix + "_"):
                return f"stage{index}"
        return None

    def _stage_is_active(self, name: str, active: Optional[set[str]]) -> bool:
        stage = self._stage_for_name(name)
        return active is None or stage is None or stage in active

    def _selected_map_names(
        self,
        every_k_block: float,
        active: Optional[set[str]],
        exclude_terminal_block: bool,
    ) -> Tuple[str, ...]:
        names: List[str] = []
        if self._stage_is_active("conv1", active):
            names.append("conv1")
        for index in range(1, 5):
            name = f"layer{index}"
            if self._stage_is_active(name, active):
                names.append(name)

        need_blocks = (
            isinstance(every_k_block, (int, float))
            and not math.isinf(float(every_k_block))
            and every_k_block >= 1
        )
        if need_blocks:
            k = int(every_k_block)
            for index, layer in enumerate(
                (self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4),
                start=1,
            ):
                layer_name = f"layer{index}"
                if not self._stage_is_active(layer_name, active):
                    continue
                for block_index in range(1, len(layer) + 1):
                    is_terminal = block_index == len(layer)
                    if block_index % k == 0 and not (
                        exclude_terminal_block and is_terminal
                    ):
                        names.append(f"{layer_name}_blk{block_index}")
        return tuple(names)

    def _decode_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_latent:
            if self.vae is None:
                raise RuntimeError("Latent feature extraction requires a VAE decoder")
            scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
            # Do not add torch.no_grad here.  VAE parameters are frozen, but the
            # generated-latent path must remain differentiable.
            pixels = self.vae.decode(x.float() / scaling_factor, return_dict=False)[0]
        else:
            pixels = x
        pixels = (pixels + 1.0) * 0.5
        if self.clip_decoded_pixels:
            pixels = pixels.clamp(0.0, 1.0)
        return (pixels - self.imagenet_mean) / self.imagenet_std

    def _pool_export(self, feature: torch.Tensor) -> torch.Tensor:
        if self.spatial_pool == 1:
            return feature
        height, width = feature.shape[-2:]
        if height % self.spatial_pool != 0 or width % self.spatial_pool != 0:
            raise ValueError(
                f"Cannot spatially pool feature map {height}x{width} by {self.spatial_pool}"
            )
        return F.avg_pool2d(
            feature,
            kernel_size=self.spatial_pool,
            stride=self.spatial_pool,
        )

    def _forward_selected_maps(
        self,
        x: torch.Tensor,
        names: Tuple[str, ...],
    ) -> Tuple[torch.Tensor, ...]:
        selected = set(names)
        pixels = self._decode_and_normalize(x)
        y = self.backbone.conv1(pixels)
        y = self.backbone.bn1(y)
        y = self.backbone.relu(y)
        y = self.backbone.maxpool(y)

        terminal: Dict[str, torch.Tensor] = {}
        blocks: Dict[str, torch.Tensor] = {}
        if "conv1" in selected:
            terminal["conv1"] = self._pool_export(y)

        for stage_index, layer in enumerate(
            (self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4),
            start=1,
        ):
            layer_name = f"layer{stage_index}"
            terminal_block_name = f"{layer_name}_blk{len(layer)}"
            for block_index, block in enumerate(layer, start=1):
                y = block(y)
                block_name = f"{layer_name}_blk{block_index}"
                if block_name in selected:
                    blocks[block_name] = self._pool_export(y)
            if layer_name in selected:
                # A selected terminal block and its stage output are the same
                # activation.  Reuse the pooled tensor while still exporting
                # both logical names; downstream losses therefore retain both
                # independently weighted entries and their gradients add at
                # this shared node exactly as they do at the unpooled map.
                if terminal_block_name in blocks:
                    terminal[layer_name] = blocks[terminal_block_name]
                else:
                    terminal[layer_name] = self._pool_export(y)

        merged = {**terminal, **blocks}
        return tuple(merged[name] for name in names)

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
        return derive_ssl_map_features(
            name,
            feature,
            patch_mean_size=patch_mean_size,
            patch_std_size=patch_std_size,
            use_std=use_std,
            use_mean=use_mean,
        )

    def get_activations(
        self,
        x: torch.Tensor,
        patch_mean_size: Optional[List[int]] = None,
        patch_std_size: Optional[List[int]] = None,
        use_std: bool = True,
        use_mean: bool = True,
        with_global: bool = True,
        with_norm_x: Optional[bool] = None,
        every_k_block: float = 2,
        exclude_terminal_block: bool = False,
        active_stages: Optional[List[str]] = None,
        stage_adapters: Optional[nn.ModuleDict] = None,
        return_stage_features: bool = False,
        return_pre_adapter_maps: bool = False,
    ) -> Dict[str, torch.Tensor] | Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        patch_mean_size = [2, 4] if patch_mean_size is None else list(patch_mean_size)
        patch_std_size = [2, 4] if patch_std_size is None else list(patch_std_size)
        active = self._active_stage_set(active_stages)
        names = self._selected_map_names(
            every_k_block,
            active,
            bool(exclude_terminal_block),
        )

        # ``layerN`` and ``layerN_blk{len(layerN)}`` are duplicate logical
        # views whenever both are requested.  With no adapter, process the
        # shared activation once and publish the same derived tensors under
        # both key families.  Keeping the two keys preserves loss weighting;
        # autograd accumulates both contributions into the shared tensors.
        # When adapters are present, retain separate calls because an adapter
        # may intentionally be stateful or stochastic.
        duplicate_sources: Dict[str, str] = {}
        if stage_adapters is None:
            for stage_index, layer in enumerate(
                (
                    self.backbone.layer1,
                    self.backbone.layer2,
                    self.backbone.layer3,
                    self.backbone.layer4,
                ),
                start=1,
            ):
                layer_name = f"layer{stage_index}"
                terminal_block_name = f"{layer_name}_blk{len(layer)}"
                if layer_name in names and terminal_block_name in names:
                    duplicate_sources[terminal_block_name] = layer_name

        output: Dict[str, torch.Tensor] = {}
        stage_parts: Dict[str, List[torch.Tensor]] = {}
        if with_global:
            output["global"] = rearrange(x, "b c h w -> b 1 (c h w)")
        if self.include_norm_x if with_norm_x is None else bool(with_norm_x):
            output["norm_x"] = safe_rms(x, dim=(2, 3)).unsqueeze(1)

        generated_path = torch.is_grad_enabled() and x.requires_grad
        configured_chunk_size = (
            self.generated_microbatch_size
            if generated_path
            else self.real_microbatch_size
        )
        chunk_size = (
            configured_chunk_size
            if configured_chunk_size > 0
            else int(x.shape[0])
        )
        # Accumulate the selected maps themselves, then compute the per-sample
        # spatial statistics once on the concatenated batch.  Those statistics
        # never reduce across the batch dimension, so this is mathematically
        # identical to computing them chunk-by-chunk, while avoiding hundreds
        # of small reduction kernels and one concatenation per derived key.
        map_parts: Dict[str, List[torch.Tensor]] = {
            name: [] for name in names if name not in duplicate_sources
        }
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
                # Legacy objectives retain only terminal layerN maps. The
                # direct field objective opts into every selected pre-adapter
                # block without changing the old InfoNCE/GAN path.
                if return_stage_features and (
                    return_pre_adapter_maps
                    or name in {"layer1", "layer2", "layer3", "layer4"}
                ):
                    stage_parts.setdefault(name, []).append(feature)
                if name in duplicate_sources:
                    continue
                # Adapters stay inside the original chunk boundary.  This
                # preserves semantics even for a future adapter containing a
                # batch-dependent layer, while still moving all feature-stat
                # work out of the microbatch loop.
                stage = self._stage_for_name(name)
                if stage_adapters is not None and stage in stage_adapters:
                    feature = stage_adapters[stage](feature)
                map_parts[name].append(feature)

        processed_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in names:
            if name in duplicate_sources:
                source_name = duplicate_sources[name]
                source_values = processed_by_name[source_name]
                aliased_values = {
                    f"{name}{key[len(source_name):]}": value
                    for key, value in source_values.items()
                }
                output.update(aliased_values)
                processed_by_name[name] = aliased_values
                continue
            values = map_parts[name]
            feature = values[0] if len(values) == 1 else torch.cat(values, dim=0)
            processed = self._process_feature(
                name,
                feature,
                patch_mean_size=patch_mean_size,
                patch_std_size=patch_std_size,
                use_std=use_std,
                use_mean=use_mean,
                # Applied per chunk above to preserve adapter behavior.
                stage_adapters=None,
            )
            output.update(processed)
            processed_by_name[name] = processed
        if return_stage_features:
            stage_features = {
                key: values[0] if len(values) == 1 else torch.cat(values, dim=0)
                for key, values in stage_parts.items()
            }
            return output, stage_features
        return output


def build_ssl_resnet_from_config(
    backbone_name: str,
    checkpoint_path: str,
    cfg: Mapping[str, object],
    device: torch.device,
) -> SSLResNetFeatureExtractor:
    """Build, freeze, and place an SSL ResNet feature extractor on ``device``."""
    model = SSLResNetFeatureExtractor(
        backbone_name=backbone_name,
        checkpoint_path=checkpoint_path,
        use_latent=bool(cfg.get("use_latent", True)),
        use_bf16=bool(cfg.get("feature_use_bf16", True)),
        use_remat=bool(cfg.get("feature_use_remat", True)),
        vae_gradient_checkpointing=(
            bool(cfg["feature_vae_gradient_checkpointing"])
            if "feature_vae_gradient_checkpointing" in cfg
            and cfg["feature_vae_gradient_checkpointing"] is not None
            else None
        ),
        microbatch_size=int(cfg.get("feature_microbatch_size", 16)),
        real_microbatch_size=(
            int(cfg["feature_real_microbatch_size"])
            if cfg.get("feature_real_microbatch_size") is not None
            else None
        ),
        generated_microbatch_size=(
            int(cfg["feature_generated_microbatch_size"])
            if cfg.get("feature_generated_microbatch_size") is not None
            else None
        ),
        spatial_pool=int(cfg.get("feature_spatial_pool", 2)),
        include_norm_x=bool(cfg.get("feature_include_norm_x", True)),
        clip_decoded_pixels=bool(cfg.get("feature_clip_decoded_pixels", False)),
        vae_model_id=str(cfg.get("feature_vae_model_id", "stabilityai/sd-vae-ft-mse")),
        vae_revision=(
            str(cfg["feature_vae_revision"])
            if cfg.get("feature_vae_revision")
            else None
        ),
        device=device,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "SSLResNetFeatureExtractor",
    "build_ssl_resnet_from_config",
    "canonical_ssl_backbone",
]
