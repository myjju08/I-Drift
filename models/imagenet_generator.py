"""PyTorch DitGen / LightningDiT — port of the official JAX generator for ImageNet.

Key differences from the existing model.py DriftDiT_Small:
  - Works for 256×256 pixel-space (patch_size=16) or 32×32 latent-space (patch_size=2)
  - Noise conditioning: noise_classes + noise_coords for sample diversity
  - n_cls_tokens: class tokens prepended to the sequence
  - CFG scale conditioning via TimestepEmbedder + RMSNorm

Input/output format: BCHW (PyTorch).
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """(grid_size*grid_size, embed_dim)"""
    h = np.arange(grid_size, dtype=np.float32)
    w = np.arange(grid_size, dtype=np.float32)
    gw, gh = np.meshgrid(w, h)
    half = embed_dim // 2
    emb_w = _sincos_1d(half, gw.reshape(-1))
    emb_h = _sincos_1d(half, gh.reshape(-1))
    return np.concatenate([emb_w, emb_h], axis=1)


# ---------------------------------------------------------------------------
# Basic modules (reuse + extend patterns from model.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        rms = (x32.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return (x32 * rms * self.weight.float()).to(x.dtype)


def _init_torchlinear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        for layer in (self.w1, self.w3, self.w2):
            _init_torchlinear(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE for q, k: (B, N, H, D)."""
    B, N, H, D = q.shape
    half = D // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, device=q.device, dtype=rope_dtype) / half))
    t = torch.arange(N, device=q.device, dtype=rope_dtype)
    freqs = torch.outer(t, freqs)
    emb = torch.cat([freqs, freqs], dim=-1)          # (N, D)
    cos = emb.cos()[None, :, None, :]                 # (1, N, 1, D)
    sin = emb.sin()[None, :, None, :]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_qk_norm: bool = True,
        use_rope: bool = False,
        use_rmsnorm: bool = True,
        attn_fp32: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_rope  = use_rope
        self.attn_fp32 = attn_fp32
        self.use_sdpa  = use_sdpa

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        for layer in (self.qkv, self.proj):
            _init_torchlinear(layer)

        if use_qk_norm:
            norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
            self.q_norm = norm_cls(self.head_dim)
            self.k_norm = norm_cls(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)           # each (B, N, H, D)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            rope_dtype = torch.float32 if self.attn_fp32 else q.dtype
            q, k = apply_rope(q, k, rope_dtype=rope_dtype)

        if self.attn_fp32:
            q = q.transpose(1, 2).float() * self.scale  # (B, H, N, D)
            k = k.transpose(1, 2).float()
            v = v.transpose(1, 2).float()
        else:
            q = q.transpose(1, 2) * self.scale
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if self.use_sdpa:
            # q already carries head_dim**-0.5, so disable SDPA's default scale.
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out.to(x.dtype))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        attn_fp32: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = Attention(
            hidden_size,
            num_heads,
            use_qk_norm,
            use_rope,
            use_rmsnorm,
            attn_fp32=attn_fp32,
            use_sdpa=use_sdpa,
        )

        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            inter = (int(2 / 3 * mlp_hidden) + 31) // 32 * 32
            self.mlp = SwiGLUFFN(hidden_size, inter)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden_size),
            )
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    _init_torchlinear(layer)

        # adaLN-Zero: zero-init for training stability
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        chunks = self.adaLN_mod(c.float()).to(x.dtype).chunk(6, dim=1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = chunks

        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = True):
        super().__init__()
        self.norm = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        chunks = self.adaLN_mod(c.float()).to(x.dtype).chunk(2, dim=1)
        shift, scale = chunks
        return self.linear(modulate(self.norm(x), shift, scale))


# ---------------------------------------------------------------------------
# TimestepEmbedder (for CFG scale)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_size: int = 256):
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# LightningDiT backbone
# ---------------------------------------------------------------------------

class LightningDiT(nn.Module):
    """DiT-style backbone for conditional image generation."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 4,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        n_cls_tokens: int = 0,
        use_remat: bool = False,       # gradient checkpointing
        attn_fp32: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.patch_size   = patch_size
        self.input_size   = input_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_cls_tokens = n_cls_tokens
        self.use_remat    = use_remat

        num_patches = (input_size // patch_size) ** 2

        self.patch_embed = nn.Linear(patch_size * patch_size * in_channels, hidden_size, bias=True)
        _init_torchlinear(self.patch_embed)

        # Sincos-initialized positional embedding (trainable, matching Flax self.param).
        pos = get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos).float().unsqueeze(0))  # (1, T, D)

        if n_cls_tokens > 0:
            self.cls_proj  = nn.Linear(hidden_size, hidden_size, bias=True)
            _init_torchlinear(self.cls_proj)
            self.cls_embed = nn.Parameter(torch.zeros(1, n_cls_tokens, hidden_size))
            nn.init.normal_(self.cls_embed, std=0.02)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio,
                use_qk_norm,
                use_swiglu,
                use_rope,
                use_rmsnorm,
                attn_fp32=attn_fp32,
                use_sdpa=use_sdpa,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, use_rmsnorm)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, T, patch_size^2 * C)"""
        B, C, H, W = x.shape
        p = self.patch_size
        gh = H // p
        gw = W // p
        x = rearrange(x, "b c (gh p1) (gw p2) -> b (gh gw) (p1 p2 c)", p1=p, p2=p)
        return x

    def _unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, T, patch_size^2 * C) → (B, C, H, W)"""
        p = self.patch_size
        gh = H // p
        gw = W // p
        return rearrange(
            x, "b (gh gw) (p1 p2 c) -> b c (gh p1) (gw p2)",
            gh=gh, gw=gw, p1=p, p2=p,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noise (B, C, H, W)
            c: conditioning vector (B, D)
        Returns:
            output (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        tokens = self.patch_embed(self._patchify(x))     # (B, T, D)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)

        if self.n_cls_tokens > 0:
            cls = self.cls_proj(c).unsqueeze(1).expand(-1, self.n_cls_tokens, -1)
            cls = cls + self.cls_embed.to(dtype=cls.dtype)
            tokens = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            if self.use_remat and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, c, use_reentrant=False)
            else:
                tokens = block(tokens, c)

        if self.n_cls_tokens > 0:
            tokens = tokens[:, self.n_cls_tokens:]

        out_tokens = self.final_layer(tokens, c)
        return self._unpatchify(out_tokens, H, W)


# ---------------------------------------------------------------------------
# DitGen wrapper
# ---------------------------------------------------------------------------

class DitGen(nn.Module):
    """Generator wrapper: class embed + noise embed + CFG scale embed → LightningDiT.

    Interface matches the official JAX DitGen:
      forward(labels, cfg_scale=1.0, temp=1.0, train=False) -> {"samples": (B,C,H,W)}
    """

    def __init__(
        self,
        cond_dim: int = 768,
        num_classes: int = 1000,
        noise_classes: int = 64,       # num discretized noise labels
        noise_coords: int = 32,        # number of independent noise axes
        input_size: int = 32,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 4,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        n_cls_tokens: int = 16,
        use_bf16: bool = True,
        use_remat: bool = False,
        attn_fp32: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.cond_dim    = cond_dim
        self.num_classes = num_classes
        self.noise_classes = noise_classes
        self.noise_coords  = noise_coords
        self.input_size  = input_size
        self.in_channels = in_channels
        self.use_bf16    = use_bf16

        # Class embedding for real ImageNet classes. CFG is handled by cfg_embedder.
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # Noise diversity embeddings
        if noise_classes > 0:
            self.noise_embeds = nn.ModuleList([
                nn.Embedding(noise_classes, cond_dim) for _ in range(noise_coords)
            ])
            for e in self.noise_embeds:
                nn.init.normal_(e.weight, std=0.02)

        # CFG scale embedding (TimestepEmbedder + RMSNorm)
        self.cfg_embedder = TimestepEmbedder(cond_dim)
        self.cfg_norm     = RMSNorm(cond_dim)

        self.model = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_qk_norm=use_qk_norm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            n_cls_tokens=n_cls_tokens,
            use_remat=use_remat,
            attn_fp32=attn_fp32,
            use_sdpa=use_sdpa,
        )

    def _build_cond(
        self,
        labels: torch.Tensor,
        cfg_scale: torch.Tensor,
        noise_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Combine class + noise + CFG embeddings."""
        device_type = labels.device.type
        if self.use_bf16 and device_type in ("cuda", "cpu"):
            amp_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            cond = self.class_embed(labels)                       # (B, D)
            if self.use_bf16:
                cond = cond.to(torch.bfloat16)

            if self.noise_classes > 0:
                for i, emb in enumerate(self.noise_embeds):
                    noise_emb = emb(noise_labels[:, i])
                    if self.use_bf16:
                        noise_emb = noise_emb.to(torch.bfloat16)
                    cond = cond + noise_emb

            cfg_emb = self.cfg_norm(self.cfg_embedder(cfg_scale))  # (B, D)
            if self.use_bf16:
                cfg_emb = cfg_emb.to(torch.bfloat16)
            cond = cond + cfg_emb * 0.02

        if self.use_bf16:
            cond = cond.to(torch.bfloat16)
        return cond

    def forward(
        self,
        labels: torch.Tensor,
        cfg_scale: float | torch.Tensor = 1.0,
        temp: float = 1.0,
        train: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            labels:    Class indices (B,) in [0, num_classes-1].
            cfg_scale: CFG scale; scalar float or (B,) tensor.
            temp:      Noise temperature.
            train:     Unused (kept for interface compat).

        Returns:
            dict with "samples" key: generated images (B, out_channels, H, W).
        """
        B = labels.shape[0]
        device = labels.device

        # Build CFG scale tensor
        if isinstance(cfg_scale, (float, int)):
            cfg_t = torch.full((B,), float(cfg_scale), device=device)
        else:
            cfg_t = cfg_scale.float().to(device)

        # Sample noise labels for diversity
        if self.noise_classes > 0:
            noise_labels = torch.randint(0, self.noise_classes, (B, self.noise_coords), device=device)
        else:
            noise_labels = torch.zeros(B, max(1, self.noise_coords), device=device, dtype=torch.long)

        cond = self._build_cond(labels, cfg_t, noise_labels)

        # Sample noise
        x = torch.randn(B, self.in_channels, self.input_size, self.input_size, device=device) * temp
        if self.use_bf16:
            x = x.to(torch.bfloat16)

        if self.use_bf16 and device.type in ("cuda", "cpu"):
            amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            samples = self.model(x, cond)
        return {"samples": samples, "noise": {"x": x, "noise_labels": noise_labels}}


def build_ditgen_from_config(model_cfg: dict, dataset_cfg: dict) -> DitGen:
    """Build DitGen from YAML config dicts."""
    model_cfg = dict(model_cfg)
    if "use_qknorm" in model_cfg:
        model_cfg.setdefault("use_qk_norm", model_cfg.pop("use_qknorm"))
    return DitGen(
        num_classes=dataset_cfg.get("num_classes", 1000),
        **model_cfg,
    )


__all__ = ["DitGen", "LightningDiT", "build_ditgen_from_config"]
