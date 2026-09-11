"""Check corrective targets against a small explicit pairwise-force oracle."""

from unittest.mock import patch

import pytest
import torch

from drifting_core.imagenet_loss import drift_loss_imagenet


def _inputs(requires_grad=False):
    arrays = (
        [[[0.0, 0.0], [1.0, -0.5], [-0.5, 1.5]]],
        [[[1.5, 1.0], [-1.5, 0.5], [0.5, -1.5]]],
        [[[-1.0, -1.0], [2.0, 0.5]]],
        [[[0.4, 0.7], [-0.5, -1.2]]],
    )
    return tuple(torch.tensor(x, requires_grad=requires_grad) for x in arrays)


@torch.no_grad()
def _explicit_reference(gen, pos, neg, history, rho, multiplier):
    """Independent dense exponential reference, retaining original banks."""
    temperatures = (0.2, 0.6)
    count = gen.shape[1]
    weight_neg = torch.tensor([[0.8, 1.2]])
    base = torch.cat((gen, neg, pos), dim=1)
    base_weight = torch.cat((torch.ones(1, count), weight_neg, torch.ones(1, 3)), dim=1)

    def distances(query, bank):
        return (query[:, :, None] - bank[:, None]).square().sum(-1).clamp_min(1e-8).sqrt()

    scale = (distances(gen, base) * base_weight[:, None]).mean() / base_weight.mean()
    input_scale = (scale / gen.shape[-1] ** 0.5).clamp_min(1e-3)
    history = history if rho else history[:, :0]
    bank = torch.cat((gen, neg, history, pos), dim=1)
    split = bank.shape[1] - pos.shape[1]
    weights = torch.cat((
        torch.full((1, count), 1.0 - rho),
        weight_neg,
        torch.full((1, history.shape[1]), rho * count / max(history.shape[1], 1)),
        torch.ones(1, pos.shape[1]),
    ), dim=1)
    self_mask = torch.zeros(1, count, bank.shape[1])
    self_mask[:, :, :count] = torch.eye(count)

    def fields(query, reference=bank):
        normalized = distances(query, reference) / scale.clamp_min(1e-3) + 100.0 * self_mask
        results = []
        for temperature in temperatures:
            logits = -normalized / temperature
            affinity = (logits.softmax(2) * logits.softmax(1)).clamp_min(1e-6).sqrt()
            affinity = affinity * weights[:, None]
            attractive = affinity[:, :, split:]
            repulsive = affinity[:, :, :split]
            coefficients = torch.cat((
                -repulsive * attractive.sum(2, keepdim=True),
                attractive * repulsive.sum(2, keepdim=True),
            ), dim=2)
            # Explicit weighted displacement sum, rather than production bmm.
            displacement = reference[:, None] - query[:, :, None]
            results.append((coefficients[:, :, :, None] * displacement).sum(2) / input_scale)
        return results

    first_raw = fields(gen)
    normalizers = [field.square().mean().clamp_min(1e-8).sqrt() for field in first_raw]
    first = multiplier * sum(field / norm for field, norm in zip(first_raw, normalizers))
    shifted = gen + first * input_scale
    second = multiplier * sum(field / norm for field, norm in zip(fields(shifted), normalizers))
    # A tempting but incorrect implementation replaces the current bank too.
    moved_bank = torch.cat((shifted, bank[:, count:]), dim=1)
    moved_second = multiplier * sum(
        field / norm for field, norm in zip(fields(shifted, moved_bank), normalizers)
    )
    return first, second, moved_second, input_scale


@pytest.mark.parametrize("rho", [0.0, 0.35])
@pytest.mark.parametrize("multiplier", [0.5, 1.0])
def test_double_target_matches_composed_field_and_detached_gradient(rho, multiplier):
    gen, pos, neg, history = _inputs(requires_grad=True)
    first, second, moved_second, input_scale = _explicit_reference(
        gen, pos, neg, history, rho, multiplier,
    )
    kwargs = dict(
        R_list=(0.2, 0.6),
        weight_neg=torch.tensor([[0.8, 1.2]]),
        global_scale_stats=False,
        global_fnorm_stats=False,
        force_multiplier=multiplier,
    )
    if rho:
        kwargs.update(
            historical_gen=history,
            weight_gen=torch.full((1, 3), 1.0 - rho),
            weight_history=torch.full((1, 2), rho * 3 / 2),
        )
    single, _ = drift_loss_imagenet(gen, pos, neg, **kwargs)
    actual, info = drift_loss_imagenet(gen, pos, neg, double_drift=True, **kwargs)
    expected = (first + second).square().mean((-1, -2))
    torch.testing.assert_close(single, first.square().mean((-1, -2)), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert not torch.allclose(expected, (2 * first).square().mean((-1, -2)))
    assert not torch.allclose(expected, (first + moved_second).square().mean((-1, -2)))
    actual.sum().backward()
    expected_gradient = -2 * (first + second) / (gen.shape[1] * gen.shape[2] * input_scale)
    torch.testing.assert_close(gen.grad, expected_gradient, rtol=3e-5, atol=3e-6)
    assert pos.grad is None
    assert neg.grad is None
    assert history.grad is None
    assert info["double_drift/enabled"] == 1.0
    if rho:
        assert info["history/count"] == 2
        assert info["history/current_mass"] == pytest.approx(3 * 0.65)
        assert info["history/replay_mass"] == pytest.approx(3 * 0.35)


def test_disabled_double_drift_preserves_loss_gradient_and_diagnostics_exactly():
    gen, pos, neg, _ = _inputs(requires_grad=True)
    kwargs = dict(global_scale_stats=False, global_fnorm_stats=False, R_list=(0.2, 0.6))
    with patch("drifting_core.imagenet_loss._reverse_drift_force_at_query", side_effect=AssertionError):
        default, default_info = drift_loss_imagenet(gen, pos, neg, **kwargs)
        disabled, disabled_info = drift_loss_imagenet(gen, pos, neg, double_drift=False, **kwargs)
    default_gradient, = torch.autograd.grad(default.sum(), gen)
    disabled_gradient, = torch.autograd.grad(disabled.sum(), gen)
    assert torch.equal(default, disabled)
    assert torch.equal(default_gradient, disabled_gradient)
    assert default_info == disabled_info
    assert not any(key.startswith("double_drift/") for key in disabled_info)


@pytest.mark.parametrize("kernel", ["exponential", "generalized_exponential"])
def test_double_drift_fused_normalization_matches_sequential_with_replay(kernel):
    gen, pos, neg, history = _inputs(requires_grad=True)
    kwargs = dict(
        historical_gen=history,
        weight_gen=torch.full((1, 3), 0.65),
        weight_history=torch.full((1, 2), 0.525),
        R_list=(0.2, 0.6),
        affinity_kernel=kernel,
        kernel_shape=2.0,
        kernel_adaptive_k_pos=1,
        kernel_adaptive_k_neg=1,
        global_scale_stats=True,
        global_fnorm_stats=True,
        double_drift=True,
    )
    sequential, info = drift_loss_imagenet(gen, pos, neg, **kwargs)
    fused, fused_info = drift_loss_imagenet(gen, pos, neg, fuse_fnorm_across_R=True, **kwargs)
    torch.testing.assert_close(sequential, fused)
    assert info == fused_info
    sequential_gradient, = torch.autograd.grad(sequential.sum(), gen)
    fused_gradient, = torch.autograd.grad(fused.sum(), gen)
    torch.testing.assert_close(sequential_gradient, fused_gradient)


def test_zero_multiplier_remains_zero_for_both_evaluations():
    gen, pos, neg, _ = _inputs(requires_grad=True)
    loss, _ = drift_loss_imagenet(gen, pos, neg, double_drift=True, force_multiplier=0)
    loss.sum().backward()
    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(gen.grad, torch.zeros_like(gen))
