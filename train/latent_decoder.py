"""Evaluation-only SD-VAE decoder; no encoder API or training-time GPU state."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _DecoderOnly(nn.Module):
    def __init__(self, vae):
        super().__init__()
        # Retain exactly the pretrained modules used by AutoencoderKL.decode.
        # The encoder and quant_conv remain on the temporary CPU VAE and are
        # released before this decoder is moved to a GPU.
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, unscaled_latents):
        return self.decoder(self.post_quant_conv(unscaled_latents))


def load_local_vae_decoder(model_path="stabilityai/sd-vae-ft-mse"):
    """Load existing local pretrained weights on CPU, retaining only decode."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32,
    )
    decoder = _DecoderOnly(vae).eval()
    decoder.requires_grad_(False)
    del vae
    return decoder


class LatentDecoderPostprocessor:
    """Lazy decoder callable with an explicit evaluation-end CPU release hook.

    Training consumes cached numbers unchanged. Only this evaluation callable
    applies the canonical SD-VAE decode scaling (divide by 0.18215), followed
    by the established [-1,1] -> [0,1] conversion and clamp.
    """

    def __init__(self, *, model_path="stabilityai/sd-vae-ft-mse", scaling_factor=0.18215):
        self.model_path = str(model_path)
        self.scaling_factor = float(scaling_factor)
        if not math.isfinite(self.scaling_factor) or self.scaling_factor <= 0:
            raise ValueError("latent decoder scaling_factor must be finite and positive")
        self._decoder = None
        self._device = None

    def _get_decoder(self, device):
        if self._decoder is None:
            self._decoder = load_local_vae_decoder(self.model_path)
        if self._device != device:
            self._decoder.to(device=device, dtype=torch.float32)
            self._device = device
        return self._decoder

    @torch.no_grad()
    def __call__(self, latents):
        if latents.ndim != 4 or latents.shape[1:] != (4, 32, 32):
            raise ValueError(f"Expected Bx4x32x32 latents for decoding, got {tuple(latents.shape)}")
        decoder = self._get_decoder(latents.device)
        with torch.autocast(device_type=latents.device.type, enabled=False):
            pixels = decoder(latents.to(dtype=torch.float32) / self.scaling_factor)
        return ((pixels + 1) / 2).clamp(0, 1)

    def release(self):
        """Keep reusable decoder weights on CPU between evaluation phases."""
        if self._decoder is not None:
            self._decoder.to(device="cpu")
            self._device = torch.device("cpu")
