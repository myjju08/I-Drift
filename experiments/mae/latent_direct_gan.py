"""Original conditional CNN trained directly on cached/scaled MAE latents.

No VAE import, load, decode, pixel conversion, or activation checkpoint is used
by this branch. The input-domain change is explicit: R1 differentiates four
scaled latent channels, and native CNN stage grids are 4x4, 2x2, and 1x1 for a
32x32 input. Original 4x4 feature pooling is retained, so coarse stages repeat
their available cells; pooling does not create additional spatial information.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping

import torch

from models.adversarial_drift import AdversarialDriftSystem, FEATURE_STAGES


class LatentDirectAdversarialSystem(AdversarialDriftSystem):
    def __init__(self, **kwargs):
        if kwargs.get('mode', 'raw_gan') not in {'raw_gan', 'feature_drift'}:
            raise ValueError('Only raw_gan and feature_drift are supported')
        if int(kwargs.get('in_channels', 0)) != 4:
            raise ValueError('The direct latent CNN requires four input channels')
        if float(kwargs.get('ema_decay', 0.)) != 0.:
            raise ValueError('Both targets must use an exact copy (EMA decay zero)')
        if float(kwargs.get('structure_weight', 0.)) != 0.:
            raise ValueError('Both discriminator objectives omit structure loss')
        kwargs.setdefault('ema_decay', 0.)
        kwargs.setdefault('structure_weight', 0.)
        super().__init__(**kwargs)
        self.input_identity = {
            'version': 1, 'input_space': 'latent', 'in_channels': 4,
            'input_size': [32, 32], 'latent_scale': 0.18215,
            'decoder_used': False, 'precision': 'float32', 'input_clipped': False,
            'r1_coordinates': 'scaled_latent', 'feature_stages': list(FEATURE_STAGES),
            'native_feature_grids': [[4, 4], [2, 2], [1, 1]], 'feature_pool': [4, 4],
        }

    @staticmethod
    def _check_input(images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (4, 32, 32) or images.shape[0] <= 0:
            raise ValueError('Expected nonempty N x 4 x 32 x 32 scaled latent samples')

    def target_logits(self, images, labels, chunk_size=None):
        if self.mode != 'raw_gan':
            raise ValueError('Feature-drift G must not consume discriminator logits')
        self._check_input(images)
        return super().target_logits(images, labels, chunk_size=chunk_size)

    def target_features(self, images, labels=None, chunk_size=None):
        if self.mode != 'feature_drift':
            raise ValueError('Raw GAN G must not consume CNN feature drifting')
        self._check_input(images)
        return super().target_features(images, labels, chunk_size=chunk_size)

    def target_logits_and_features(self, *args, **kwargs):
        raise ValueError('Mixed generator supervision is outside this comparison')

    def discriminator_step(self, real_images, fake_images, real_labels, fake_labels,
                           step, teacher_real_features=None, chunk_size=None):
        self._check_input(real_images)
        self._check_input(fake_images)
        if teacher_real_features is not None:
            raise ValueError('These comparisons do not use a structure teacher')
        return super().discriminator_step(real_images, fake_images, real_labels, fake_labels,
                                          step, teacher_real_features=None, chunk_size=chunk_size)

    def state_dict(self):
        state = super().state_dict()
        state['adversarial_input_identity'] = copy.deepcopy(self.input_identity)
        return state

    def load_state_dict(self, state: Mapping):
        if state.get('adversarial_input_identity') != self.input_identity:
            raise ValueError('Direct latent discriminator input identity differs; RGB checkpoints cannot resume')
        super().load_state_dict(state)


def build_adversarial_system(cfg: dict, device: torch.device):
    """Build the original D32 CNN in the explicitly authorized latent domain."""
    mode = str(cfg.get('adversarial_mode', 'none')).strip().lower()
    if mode in {'', 'none', 'off'}:
        return None
    if mode not in {'raw_gan', 'feature_drift'}:
        raise ValueError('Only raw_gan and feature_drift are supported')
    if cfg.get('adversarial_input_space') != 'latent' or int(cfg.get('adversarial_in_channels', 0)) != 4:
        raise ValueError('Set adversarial_input_space=latent and adversarial_in_channels=4')
    if not bool(cfg.get('use_latent', False)) or int(cfg.get('in_channels', 0)) != 4:
        raise ValueError('The generator must use four-channel latent outputs')
    feature = str(cfg.get('feature_extractor', 'mae')).strip().lower().replace('-', '_')
    if feature not in {'mae', 'mae_resnet', 'mae_resnet256'}:
        raise ValueError('This comparison requires the unchanged frozen MAE encoder')
    if bool(cfg.get('feature_adapter', False)) or bool(cfg.get('feature_gan', False)):
        raise ValueError('MAE adapters and feature_gan must remain disabled')
    if bool(cfg.get('historical_gen_replay', False)):
        raise ValueError('Both direct latent comparisons require Replay disabled')
    inactive = 'adversarial_drift_weight' if mode == 'raw_gan' else 'adversarial_loss_weight'
    for key in ('adversarial_ema_decay', 'adversarial_structure_weight', inactive):
        if float(cfg.get(key, 0.)) != 0.:
            raise ValueError(f'{key} must be zero')
    active = 'adversarial_loss_weight' if mode == 'raw_gan' else 'adversarial_drift_weight'
    weight = float(cfg.get(active, .1 if mode == 'raw_gan' else 1.))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f'{active} must be finite and positive')
    if int(cfg.get('adversarial_samples_per_class', 8)) <= 0:
        raise ValueError('adversarial_samples_per_class must be positive')
    if int(cfg.get('adversarial_base_channels', 32)) != 32:
        raise ValueError('These direct latent comparisons require D32')
    arguments = {name: cfg.get('adversarial_' + name, default)
                 for name, default in {
                     'base_channels': 32, 'lr': 1e-4, 'adam_b1': 0., 'adam_b2': .99,
                     'r1_gamma': 1., 'r1_interval': 16, 'd_chunk_size': 8,
                     'g_chunk_size': 16, 'seed': 43, 'max_grad_norm': 0.,
                     'fuse_grad_reduce': False,
                 }.items()}
    device = torch.device(device)
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == 'cuda' else [])
    with torch.random.fork_rng(devices=cuda_devices):
        system = LatentDirectAdversarialSystem(
            device=device, mode=mode, in_channels=4, num_classes=int(cfg.get('num_classes', 1000)),
            ema_decay=0., structure_weight=0., **arguments)
    system._config_fields = copy.deepcopy({k: v for k, v in cfg.items() if k.startswith('adversarial_')})
    return system
