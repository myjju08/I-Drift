"""The MAE latent GAN works directly in four-channel sample coordinates."""
import copy

import pytest
import torch

from train_imagenet_gen import build_adversarial_system


def config(**changes):
    return dict(feature_extractor="mae", use_latent=True, in_channels=4,
                out_channels=4, adversarial_mode="raw_gan",
                adversarial_base_channels=4, num_classes=4,
                adversarial_r1_gamma=1.0, adversarial_r1_interval=16,
                adversarial_d_chunk_size=2, adversarial_g_chunk_size=2,
                **changes)


def test_latent_gan_finite_r1_frozen_target_and_rng():
    cfg = config()
    rng = torch.get_rng_state().clone()
    system = build_adversarial_system(cfg, torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state())
    assert system.online.stem[0].in_channels == 4
    assert all(not p.requires_grad for p in system.target.parameters())
    real = torch.randn(4, 4, 32, 32) * 2  # unbounded latent values, not RGB
    fake = torch.randn_like(real).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    logits = system.target_logits(fake, labels)
    grad, = torch.autograd.grad(logits.sum(), fake)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    metrics = system.discriminator_step(real, fake.detach(), labels, labels, step=0)
    assert all(torch.isfinite(v).all() for v in metrics.values() if isinstance(v, torch.Tensor))
    restored = build_adversarial_system(cfg, torch.device("cpu"))
    restored.load_state_dict(copy.deepcopy(system.state_dict()))
    assert restored.updates == system.updates


@pytest.mark.parametrize("changes", [
    {"in_channels": 3}, {"out_channels": 3}, {"use_latent": False},
    {"feature_extractor": "dino_resnet50"},
    {"adversarial_mode": "mixed"}, {"adversarial_mode": "feature_drift"},
])
def test_latent_gan_rejects_incompatible_representation(changes):
    cfg = config()
    cfg.update(changes)
    with pytest.raises(ValueError):
        build_adversarial_system(cfg, torch.device("cpu"))
