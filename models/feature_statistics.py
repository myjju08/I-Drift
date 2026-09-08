"""Numerically stable statistics used by DINO feature extraction."""
from __future__ import annotations

from typing import Tuple

import torch


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
