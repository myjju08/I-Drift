"""Direct-latent D32 with native stage2/3/4 grids of 8x8, 4x4, 4x4.

The original stem average pool is omitted and the fourth block's first
convolution uses stride one.  All learned tensors, projection heads, feature
pooling/normalization, optimizer settings, and adversarial objectives come from
the existing direct-latent system.  No VAE or RGB conversion is used.
"""
from __future__ import annotations

from types import MethodType
from typing import Dict

import torch
from torch import nn

from .latent_direct_gan import build_adversarial_system as _build_direct_system


ARCHITECTURE = 'latent_spatial_844'
NATIVE_FEATURE_GRIDS = ((8, 8), (4, 4), (4, 4))


def _spatial_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Original feature traversal with an identity stem pool."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (4, 32, 32):
        raise ValueError('Expected N x 4 x 32 x 32 scaled latent samples')
    hidden = self.stem_pool(self.stem(images.float()))
    maps = {}
    for index, block in enumerate(self.blocks, start=1):
        hidden = block(hidden)
        if index >= 2:
            maps[f'stage{index}'] = hidden
    return maps


def _install_spatial_geometry(discriminator: nn.Module) -> None:
    """Change sampling geometry in place, preserving every parameter object."""
    if discriminator.in_channels != 4 or discriminator.base_channels != 32:
        raise ValueError('The spatial latent architecture requires a four-channel D32')
    if len(discriminator.blocks) != 4 or hasattr(discriminator, 'stem_pool'):
        raise ValueError('Expected an unmodified original four-block discriminator')
    for block in discriminator.blocks:
        if not isinstance(block[0], nn.Conv2d) or block[0].stride != (2, 2):
            raise ValueError('Expected the original stride-two first convolution in each block')
    discriminator.stem_pool = nn.Identity()
    discriminator.blocks[3][0].stride = (1, 1)
    discriminator._maps = MethodType(_spatial_maps, discriminator)


def build_adversarial_system(cfg: dict, device: torch.device):
    """Build the explicitly selected spatial latent variant for either G loss.

    The direct builder retains its full validation and RNG isolation.  Installing
    geometry consumes no random numbers, creates no learned tensor, and does not
    replace the optimizer.  The added identity fields reject checkpoints from
    the older 4/2/1 geometry despite their compatible parameter tensor shapes.
    """
    mode = str(cfg.get('adversarial_mode', 'none')).strip().lower()
    if mode not in {'', 'none', 'off'} and cfg.get('adversarial_architecture') != ARCHITECTURE:
        raise ValueError(f'Set adversarial_architecture={ARCHITECTURE}')
    system = _build_direct_system(cfg, device)
    if system is None:
        return None
    _install_spatial_geometry(system.online)
    _install_spatial_geometry(system.target)
    system.target.eval().requires_grad_(False)
    system.input_identity.update({
        'architecture': ARCHITECTURE,
        'native_feature_grids': [list(shape) for shape in NATIVE_FEATURE_GRIDS],
        'stem_pool': 'identity',
        'block_first_conv_strides': [2, 2, 2, 1],
    })
    return system
