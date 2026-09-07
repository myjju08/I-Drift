"""PyTorch MAEResNet — port of the official JAX MAEResNetJAX.

Architecture:
  - ResNet encoder (4 stages, GroupNorm) → multi-scale feature maps
  - UNet decoder (skip connections) → pixel reconstruction
  - MAE masking at patch level

Input format: BCHW (PyTorch standard).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

GN_EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class _DtypePreservingGroupNorm(nn.GroupNorm):
    """GroupNorm that returns the incoming activation dtype.

    CUDA autocast promotes torch.nn.GroupNorm outputs to fp32. The official
    Flax MAE is constructed with dtype=bf16 when use_bf16=True, so casting the
    output back keeps the local activation stream and returned features closer
    to the official bf16 path.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input).to(dtype=input.dtype)


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return _DtypePreservingGroupNorm(
        _choose_gn_groups(num_channels, max_groups),
        num_channels,
        eps=GN_EPS,
    )


def safe_mean(x: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
    return x.float().mean(dim=dim, keepdim=keepdim).to(x.dtype)


def safe_rms(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    return (x32.square().mean(dim=dim, keepdim=keepdim).clamp_min(0.0) + eps).sqrt().to(x.dtype)


def safe_std(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return (var.clamp_min(0.0) + eps).sqrt()


def safe_mean_std(
    x: torch.Tensor,
    dim,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the existing safe_mean/safe_std pair with one FP32 cast/mean."""
    x32 = x.float()
    mean_keepdim = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean_keepdim) ** 2).mean(dim=dim, keepdim=keepdim)
    mean = mean_keepdim if keepdim else mean_keepdim.squeeze(dim)
    return mean.to(x.dtype), (var.clamp_min(0.0) + eps).sqrt()


def activations_from_feature_map(
    name: str,
    feat: torch.Tensor,
    *,
    patch_mean_size: Optional[List[int]] = None,
    patch_std_size: Optional[List[int]] = None,
    use_std: bool = True,
    use_mean: bool = True,
    fuse_stats: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build the exact drift activations derived from one BCHW feature map.

    This is shared by :meth:`MAEResNet.get_activations` and the separately
    optimized feature adapter. Keeping the transformation in one place makes
    the adapter raw-field objective operate on precisely the same token,
    global-statistic, and patch-statistic features as the generator drift.
    """
    if feat.ndim != 4:
        raise ValueError(f"feature map must be BCHW, got {tuple(feat.shape)}")
    if patch_mean_size is None:
        patch_mean_size = [2, 4]
    if patch_std_size is None:
        patch_std_size = [2, 4]

    _, _, height, width = feat.shape
    out: Dict[str, torch.Tensor] = {
        name: rearrange(feat, "b c h w -> b (h w) c")
    }
    if fuse_stats and use_mean and use_std:
        mean, std = safe_mean_std(feat, dim=(2, 3))
        out[f"{name}_mean"] = mean.unsqueeze(1)
        out[f"{name}_std"] = std.unsqueeze(1)
    else:
        if use_mean:
            out[f"{name}_mean"] = safe_mean(feat, dim=(2, 3)).unsqueeze(1)
        if use_std:
            out[f"{name}_std"] = safe_std(feat, dim=(2, 3)).unsqueeze(1)

    cached_patch_std: Dict[int, torch.Tensor] = {}
    for size in patch_mean_size:
        if height % size == 0 and width % size == 0:
            patches = rearrange(
                feat,
                "b c (h s1) (w s2) -> b (h w) (s1 s2) c",
                s1=size,
                s2=size,
            )
            if fuse_stats and use_std and size in patch_std_size:
                mean, std = safe_mean_std(patches, dim=2)
                out[f"{name}_mean_{size}"] = mean
                cached_patch_std[size] = std
            else:
                out[f"{name}_mean_{size}"] = safe_mean(patches, dim=2)

    for size in patch_std_size:
        if height % size == 0 and width % size == 0:
            if size in cached_patch_std:
                out[f"{name}_std_{size}"] = cached_patch_std[size]
            else:
                patches = rearrange(
                    feat,
                    "b c (h s1) (w s2) -> b (h w) (s1 s2) c",
                    s1=size,
                    s2=size,
                )
                out[f"{name}_std_{size}"] = safe_std(patches, dim=2)
    return out


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    """ResNet basic block: 3x3 → GN → ReLU → 3x3 → GN + skip."""

    def __init__(self, filters: int, stride: int = 1, dropout_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, stride=stride, padding=1, bias=False)
        self.gn1   = _group_norm(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.gn2   = _group_norm(filters)
        self.drop  = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

        self.skip: Optional[nn.Sequential] = None
        if stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(filters, filters, 1, stride=stride, bias=False),
                _group_norm(filters),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.drop(y)
        y = self.gn2(self.conv2(y))
        if self.skip is not None:
            residual = self.skip(x)
        return F.relu(residual + y)


class _BasicBlockTransition(nn.Module):
    """First block in a stage with channel doubling and optional stride."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2   = _group_norm(out_ch)
        self.drop  = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

        # Match official behavior: project skip only when shape changes.
        if in_ch == out_ch and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _group_norm(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.drop(y)
        y = self.gn2(self.conv2(y))
        return F.relu(residual + y)


class _ResNetEncoder(nn.Module):
    """4-stage ResNet encoder producing multi-scale feature maps.

    Returns dict: {conv1, layer1, layer2, layer3, layer4}
    each with shape (B, C_i, H_i, W_i).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        layers: Tuple[int, ...] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        c = base_channels
        # Official MAE-ResNet conv1 consumes patched input channels directly.
        self.conv1 = nn.Conv2d(in_channels, c, 3, stride=1, padding=1, bias=False)
        self.gn1   = _group_norm(c)

        stage_list = []
        in_ch = c
        for i, n_blocks in enumerate(layers):
            out_ch = c * (2 ** i)
            stride = 2 if i > 0 else 1
            blocks = [_BasicBlockTransition(in_ch, out_ch, stride, dropout_prob)]
            for _ in range(1, n_blocks):
                blocks.append(_BasicBlock(out_ch, stride=1, dropout_prob=dropout_prob))
            stage_list.append(nn.Sequential(*blocks))
            setattr(self, f"layer{i+1}_norm",
                    _group_norm(out_ch))
            in_ch = out_ch
        self.stages = nn.ModuleList(stage_list)

    def forward(
        self,
        x: torch.Tensor,
        return_block_outputs: bool = False,
        use_remat: bool = False,
        capture_stages: Optional[Tuple[str, ...]] = None,
        block_output_stages: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, torch.Tensor] | Tuple[Dict, Dict]:
        all_stages = tuple(f"stage{i}" for i in range(1, 5))
        captured = set(all_stages if capture_stages is None else capture_stages)
        captured_blocks = set(
            captured if block_output_stages is None else block_output_stages
        )
        feats: Dict[str, torch.Tensor] = {}
        block_outs: Dict[str, List[torch.Tensor]] = {}

        x = F.relu(self.gn1(self.conv1(x)))
        if "stage1" in captured:
            feats["conv1"] = x

        for i, stage in enumerate(self.stages):
            lname = f"layer{i+1}"
            stage_name = f"stage{i+1}"
            outs = [] if return_block_outputs and stage_name in captured_blocks else None
            for block in stage:
                if use_remat and torch.is_grad_enabled() and x.requires_grad:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
                if outs is not None:
                    outs.append(x)
            if outs is not None:
                block_outs[lname] = outs
            x = getattr(self, f"{lname}_norm")(x)
            if stage_name in captured:
                feats[lname] = x

        if return_block_outputs:
            return feats, block_outs
        return feats


# ---------------------------------------------------------------------------
# Decoder building blocks
# ---------------------------------------------------------------------------

class _ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.gn   = _group_norm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.concat_norm = _group_norm(in_ch + skip_ch)
        self.proj   = _ConvGNReLU(in_ch + skip_ch, out_ch)
        self.refine = _ConvGNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm(x)
        x = self.proj(x)
        return self.refine(x)


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c = base_channels
        # Channel widths at each encoder stage
        c1, c2, c3, c4, c5 = c, c, c * 2, c * 4, c * 8
        self.bridge = _ConvGNReLU(c5, c5)
        self.up43   = _UpBlock(c5, c4, c4)
        self.up32   = _UpBlock(c4, c3, c3)
        self.up21   = _UpBlock(c3, c2, c2)
        self.up10   = _UpBlock(c2, c1, c1)
        self.head   = nn.Conv2d(c1, out_channels, 1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def patch_input(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Rearrange (B, C, H*p, W*p) → (B, C*p*p, H, W).

    Used to downsample the spatial resolution by patch_size before encoding.
    """
    if patch_size == 1:
        return x
    return rearrange(
        x,
        "b c (h ph) (w pw) -> b (c ph pw) h w",
        ph=patch_size,
        pw=patch_size,
    )


def make_patch_mask(
    x: torch.Tensor,
    mask_ratio: torch.Tensor,
    patch_size: int = 4,
) -> torch.Tensor:
    """Create a random patch mask with shape (B, 1, H, W) matching x spatial dims.

    Args:
        x:           (B, C, H, W) tensor (after patch_input).
        mask_ratio:  per-sample mask ratios (B,).
        patch_size:  patch size used for masking (applied on top of patch_input resolution).

    Returns:
        Binary mask (B, 1, H, W); 1 = masked, 0 = kept.
    """
    B, _, H, W = x.shape
    nh, nw = H // patch_size, W // patch_size
    noise = torch.rand(B, nh, nw, device=x.device)
    mask = (noise < mask_ratio[:, None, None]).float()
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return mask.unsqueeze(1)  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MAEResNet(nn.Module):
    """PyTorch MAEResNet for ImageNet feature learning.

    Equivalent to the official JAX MAEResNetJAX but in BCHW format.

    Args:
        num_classes:       Number of ImageNet classes (1000).
        in_channels:       Raw image channels (3 = pixel, 4 = latent).
        base_channels:     Encoder base width.
        patch_size:        MAE masking patch size (applied on the encoded spatial map).
        dropout_prob:      Dropout in residual blocks.
        layers:            ResNet stage depths, e.g. (2,2,2,2) or (3,4,6,3).
        use_bf16:          Run encoder in bfloat16.
        input_patch_size:  Downsampling factor before encoding (1 = none, 8 = pixel→latent scale).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Tuple[int, ...] = (2, 2, 2, 2),
        use_bf16: bool = False,
        input_patch_size: int = 1,
        use_remat: bool = False,
        fuse_stats: bool = False,
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.in_channels    = in_channels
        self.base_channels  = base_channels
        self.patch_size     = patch_size
        self.use_bf16       = use_bf16
        self.input_patch_size = input_patch_size
        self.use_remat      = use_remat
        self.fuse_stats     = fuse_stats

        enc_in_ch = in_channels * input_patch_size * input_patch_size
        # Keep for backward compatibility with older checkpoints (unused, identity).
        self.input_proj = nn.Identity()

        self.encoder = _ResNetEncoder(
            in_channels=enc_in_ch,
            base_channels=base_channels,
            layers=layers,
            dropout_prob=dropout_prob,
        )
        self.decoder = _UNetDecoder(
            base_channels,
            out_channels=in_channels * input_patch_size * input_patch_size,
        )
        self.fc = nn.Linear(base_channels * (2 ** (len(layers) - 1)), num_classes)

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        return x.bfloat16() if self.use_bf16 else x.float()

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        train: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x:              Input images (B, C, H, W).
            labels:         Class labels (B,).
            lambda_cls:     Weight on classification loss (0.0 = pure reconstruction).
            mask_ratio_min: Min mask ratio (uniform random between min and max).
            mask_ratio_max: Max mask ratio.
            train:          Enables dropout.

        Returns:
            loss:    scalar (averaged over batch).
            metrics: dict with per-sample tensors for logging.
        """
        return _mae_forward(
            self,
            x=x,
            labels=labels,
            lambda_cls=lambda_cls,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            train=train,
        )

    def get_activations(
        self,
        x: torch.Tensor,
        patch_mean_size: Optional[List[int]] = None,
        patch_std_size: Optional[List[int]] = None,
        use_std: bool = True,
        use_mean: bool = True,
        with_global: bool = True,
        every_k_block: float = 2,
        stage_adapters: Optional[nn.ModuleDict] = None,
        return_stage_features: bool = False,
        return_all_stage_features: bool = False,
        active_stages: Optional[List[str]] = None,
        with_norm_x: bool = True,
        exclude_terminal_block: bool = False,
    ) -> Dict[str, torch.Tensor] | Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """Extract multi-scale features for drift loss.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Dict of feature tensors each shaped (B, T, D) where T = spatial tokens.
        """
        if patch_mean_size is None:
            patch_mean_size = [2, 4]
        if patch_std_size is None:
            patch_std_size = [2, 4]

        valid_stages = tuple(f"stage{i}" for i in range(1, 5))
        if active_stages is None:
            selected_stages = valid_stages
        else:
            requested_stages = {str(stage).strip() for stage in active_stages}
            unknown_stages = requested_stages.difference(valid_stages)
            if unknown_stages:
                raise ValueError(
                    f"Unknown active feature stages: {sorted(unknown_stages)}; "
                    f"expected a subset of {list(valid_stages)}"
                )
            selected_stages = tuple(
                stage for stage in valid_stages if stage in requested_stages
            )
        selected_stage_set = set(selected_stages)

        out: Dict[str, torch.Tensor] = {}
        adapter_inputs: Dict[str, torch.Tensor] = {}
        if with_global:
            # Match the official JAX wrapper, which adds the raw sample as a
            # single-token feature before merging MAE activations.
            out["global"] = rearrange(x, "b c h w -> b 1 (c h w)")

        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        x = x.to(dtype)
        x_patched = patch_input(x, self.input_patch_size)
        x = self.input_proj(x_patched)

        need_blocks = (
            isinstance(every_k_block, (int, float))
            and not math.isinf(float(every_k_block))
            and every_k_block >= 1
        )
        if need_blocks:
            feats, block_outs = self.encoder(
                x,
                return_block_outputs=True,
                use_remat=self.use_remat,
                capture_stages=selected_stages,
                block_output_stages=selected_stages,
            )
        else:
            feats = self.encoder(
                x,
                use_remat=self.use_remat,
                capture_stages=selected_stages,
            )
            block_outs = {}

        if with_norm_x:
            # Match official behavior: norm_x is computed on patched input.
            # Keep epsilon to avoid sqrt(0) -> inf gradient at exactly-zero activations.
            out["norm_x"] = safe_rms(x_patched, dim=(2, 3)).unsqueeze(1)

        if return_all_stage_features and not return_stage_features:
            raise ValueError(
                "return_all_stage_features requires return_stage_features=true"
            )

        def process_feat(name: str, feat: torch.Tensor) -> None:
            stage = None
            for stage_index in range(1, 5):
                prefix = f"layer{stage_index}"
                if name == prefix or name.startswith(f"{prefix}_"):
                    stage = f"stage{stage_index}"
                    break
            if stage is not None and stage not in selected_stage_set:
                return
            if return_all_stage_features and stage is not None:
                adapter_inputs[name] = feat
            if stage_adapters is not None and stage in stage_adapters:
                feat = stage_adapters[stage](feat)
            out.update(
                activations_from_feature_map(
                    name,
                    feat,
                    patch_mean_size=patch_mean_size,
                    patch_std_size=patch_std_size,
                    use_std=use_std,
                    use_mean=use_mean,
                    fuse_stats=self.fuse_stats,
                )
            )

        if return_stage_features:
            adapter_inputs = {
                name: feat
                for name, feat in feats.items()
                if name.startswith("layer") and name[5:].isdigit()
            }

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                stage_block_outputs = block_outs.get(lname, [])
                for blk_idx, feat_i in enumerate(stage_block_outputs, start=1):
                    is_terminal = blk_idx == len(stage_block_outputs)
                    if blk_idx % k == 0 and not (
                        exclude_terminal_block and is_terminal
                    ):
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        if return_stage_features:
            return out, adapter_inputs
        return out


def _mae_forward(
    self: MAEResNet,
    x: torch.Tensor,
    labels: torch.Tensor,
    lambda_cls: float = 0.0,
    mask_ratio_min: float = 0.75,
    mask_ratio_max: float = 0.75,
    train: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    dtype = torch.bfloat16 if self.use_bf16 else torch.float32
    x = x.to(dtype)
    # Patch aggregation (for pixel → latent scale, input_patch_size=8 etc.)
    x_patched = patch_input(x, self.input_patch_size)   # (B, C*p^2, H, W)
    x_enc = self.input_proj(x_patched)                  # (B, base_ch, H, W)

    # Random mask ratio per sample
    B = x.shape[0]
    mask_ratio = (
        torch.zeros(B, device=x.device).uniform_(mask_ratio_min, mask_ratio_max)
        .to(dtype)
    )
    mask = make_patch_mask(x_enc, mask_ratio, self.patch_size)   # (B,1,H,W)

    x_in = x_enc * (1.0 - mask)
    feats = self.encoder(x_in)

    # Classification
    top = feats["layer4"]
    pooled = top.mean(dim=(2, 3))        # (B, C_top)
    logits = self.fc(pooled.float())

    # Reconstruction
    recon = self.decoder(feats)          # (B, C*p^2, H, W)

    # Losses
    one_hot = F.one_hot(labels, self.num_classes).float()
    cls_loss = -(one_hot * F.log_softmax(logits, dim=-1)).sum(dim=-1)   # (B,)

    mse = (recon - x_patched.float()) ** 2
    recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)

    loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss

    metrics = {
        "loss": loss,
        "cls_loss": cls_loss,
        "recon_loss": recon_loss,
        "accuracy": (logits.argmax(dim=-1) == labels).float(),
        "mask_ratio": mask.mean(dim=(1, 2, 3)),
    }
    return loss, metrics


def build_mae_from_config(config: dict) -> MAEResNet:
    """Instantiate MAEResNet from a config dict (e.g. YAML model section)."""
    return MAEResNet(
        num_classes=config.get("num_classes", 1000),
        in_channels=config.get("in_channels", 3),
        base_channels=config.get("base_channels", 64),
        patch_size=config.get("patch_size", 4),
        dropout_prob=config.get("dropout_prob", 0.0),
        layers=tuple(config.get("layers", [2, 2, 2, 2])),
        use_bf16=bool(config.get("use_bf16", False)),
        input_patch_size=config.get("input_patch_size", 1),
        use_remat=bool(config.get("use_remat", False)),
        fuse_stats=bool(config.get("fuse_stats", False)),
    )


__all__ = [
    "MAEResNet",
    "activations_from_feature_map",
    "build_mae_from_config",
    "patch_input",
    "make_patch_mask",
]
