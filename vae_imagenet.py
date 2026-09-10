"""PyTorch VAE encode/decode helper for ImageNet latent-space training.

Uses the pretrained Stable Diffusion VAE (sd-vae-ft-mse) via diffusers.
Encodes (B,C,H,W) pixel images in [-1,1] → (B,4,H/8,W/8) latents.
Decodes latents back to pixel images.

Cache: VAE params are loaded once and cached in module-level variables.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

_vae_cache: dict = {}


def load_vae(
    device: Optional[torch.device] = None,
    model_id: str = "stabilityai/sd-vae-ft-mse",
    revision: Optional[str] = None,
):
    """Load and freeze an SD VAE, cached by model id and revision."""
    cache_key = (str(model_id), str(revision) if revision else None)
    if cache_key not in _vae_cache:
        from diffusers import AutoencoderKL
        load_kwargs = {"revision": revision} if revision else {}
        vae = AutoencoderKL.from_pretrained(model_id, **load_kwargs)
        _vae_cache[cache_key] = vae
    vae = _vae_cache[cache_key]
    if device is not None:
        vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_vae(device: Optional[torch.device] = None):
    """Backward-compatible default VAE loader."""
    return load_vae(device=device)


def vae_encode(
    images: torch.Tensor,
    vae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Encode (B, C, H, W) pixel images ∈ [-1,1] to latents (B, 4, H/8, W/8).

    Args:
        images: Float tensor in [-1, 1] with shape (B, 3, H, W).
        vae:    Pre-loaded AutoencoderKL; loaded automatically if None.

    Returns:
        Latents scaled by 0.18215, shape (B, 4, H/8, W/8).
    """
    if vae is None:
        vae = _load_vae(images.device)
    with torch.no_grad():
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample() * 0.18215
    return latents


def vae_decode(
    latents: torch.Tensor,
    vae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Decode latents (B, 4, H/8, W/8) to pixel images (B, 3, H, W) ∈ [-1,1].

    Args:
        latents: Scaled latents, shape (B, 4, H/8, W/8).
        vae:     Pre-loaded AutoencoderKL; loaded automatically if None.

    Returns:
        Pixel images in [-1, 1], shape (B, 3, H, W).
    """
    if vae is None:
        vae = _load_vae(latents.device)
    with torch.no_grad():
        images = vae.decode(latents / 0.18215).sample
    return images


def get_vae_enc_dec(
    device: Optional[torch.device] = None,
    model_id: str = "stabilityai/sd-vae-ft-mse",
    revision: Optional[str] = None,
) -> Tuple[Callable, Callable]:
    """Return (encode_fn, decode_fn) bound to a shared VAE instance.

    Both callables accept torch tensors on `device` and return tensors on the same device.
    """
    vae = load_vae(device=device, model_id=model_id, revision=revision)
    return (
        partial(vae_encode, vae=vae),
        partial(vae_decode, vae=vae),
    )


# ---------------------------------------------------------------------------
# Latent cache builder (pre-encode ImageNet to disk)
# ---------------------------------------------------------------------------

def build_latent_cache(
    data_path: str,
    target_path: str,
    batch_size: int = 64,
    num_workers: int = 8,
    device: Optional[torch.device] = None,
    resolution: int = 256,
) -> None:
    """Encode all ImageNet images to VAE latents and save as .pt files.

    Saved files contain:
        {"moments": latent_sample, "moments_flip": latent_of_hflipped}

    Args:
        data_path:   ImageNet root with train/ and val/ subdirs.
        target_path: Output root; mirrors data_path structure.
        batch_size:  Encoding batch size.
        num_workers: DataLoader workers.
        device:      Compute device (defaults to first available CUDA or CPU).
        resolution:  Center-crop resolution (256 by default).
    """
    import os
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import datasets, transforms
    from tqdm import tqdm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encode_fn, _ = get_vae_enc_dec(device)

    def center_crop(img: Image.Image, size: int) -> Image.Image:
        while min(*img.size) >= 2 * size:
            img = img.resize(tuple(x // 2 for x in img.size), Image.BOX)
        scale = size / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), Image.BICUBIC)
        arr = np.array(img)
        cy = (arr.shape[0] - size) // 2
        cx = (arr.shape[1] - size) // 2
        return Image.fromarray(arr[cy : cy + size, cx : cx + size])

    tf = transforms.Compose([
        transforms.Lambda(lambda img: center_crop(img, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    class _FolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, idx):
            img, lbl = super().__getitem__(idx)
            path, _ = self.samples[idx]
            rel = os.path.join(*path.split(os.sep)[-2:])
            return img, lbl, rel

    for split in ("train", "val"):
        ds = _FolderWithPaths(os.path.join(data_path, split), transform=tf)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        Path(target_path, split).mkdir(parents=True, exist_ok=True)

        for imgs, _, rel_paths in tqdm(loader, desc=f"cache:{split}"):
            imgs = imgs.to(device)
            imgs_flip = torch.flip(imgs, dims=[3])

            latents      = encode_fn(imgs).cpu()
            latents_flip = encode_fn(imgs_flip).cpu()

            for i, rel in enumerate(rel_paths):
                out = Path(target_path, split, rel).with_suffix(".pt")
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "moments": latents[i].numpy(),
                        "moments_flip": latents_flip[i].numpy(),
                    },
                    tmp,
                )
                os.replace(tmp, out)


__all__ = [
    "load_vae",
    "vae_encode",
    "vae_decode",
    "get_vae_enc_dec",
    "build_latent_cache",
]
