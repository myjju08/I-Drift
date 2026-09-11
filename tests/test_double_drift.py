"""Double Drift tests ported from cosmosjhj/I-Drift 27a0e4f.

The default-call assertion follows this target's retained original loss path.
"""
from unittest import mock
import math

import pytest
import torch

from drifting_core.double_drift import sample_double_drift_loss, validate_double_drift_coefficients
from drifting_core.imagenet_loss import (
    _drift_loss_imagenet_single,
    drift_loss_imagenet,
    reverse_drift_field,
)


LOCAL = dict(global_scale_stats=False, global_fnorm_stats=False)


def inputs():
    torch.manual_seed(19)
    return (torch.randn(2, 4, 5, requires_grad=True),
            torch.randn(2, 3, 5), torch.randn(2, 2, 5))


def test_original_loss_and_gradient_fixture_are_unchanged():
    x, p, n = inputs()
    loss, _ = drift_loss_imagenet(x, p, n, **LOCAL)
    grad = torch.autograd.grad(loss.mean(), x)[0]
    # The upstream literal fixture differs by 1.4e-5 on this CPU/PyTorch too;
    # retain a portable fixture plus exact parity to the original target path.
    torch.testing.assert_close(loss, torch.tensor([2.22752046585083, 13.02928352355957]),
                               rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(grad.square().sum(), torch.tensor(0.6148685216903687))
    torch.testing.assert_close(grad.flatten()[:4], torch.tensor([
        0.08993204683065414, 0.031595055013895035,
        0.010085341520607471, -0.0046698865480721]))
    original, original_info = _drift_loss_imagenet_single(x, p, n, **LOCAL)
    actual, actual_info = drift_loss_imagenet(x, p, n, **LOCAL)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.mean(), x)[0],
                               torch.autograd.grad(original.mean(), x)[0], rtol=0, atol=0)
    assert actual_info == original_info


@pytest.mark.parametrize("c0,c1", [(1, 0), (1, 1), (.9, .1), (.75, .25), (.5, .5), (0, 1)])
def test_feature_matches_manual_two_step_target_and_stopped_gradient(c0, c1):
    x, p, n = inputs()
    p.requires_grad_()
    n.requires_grad_()
    f0, scale, info = reverse_drift_field(x, p, n, **LOCAL)
    probe = x.detach() + c0 * f0 * scale
    f1, scale1, _ = reverse_drift_field(probe, p, n, fixed_distance_scale=info["scale"], **LOCAL)
    assert not f0.requires_grad and not f1.requires_grad
    torch.testing.assert_close(scale, scale1)
    target = (x.detach() / scale + c0 * f0 + c1 * f1).detach()
    expected = (x / scale - target).square().mean(dim=(-1, -2))
    actual, _ = drift_loss_imagenet(x, p, n, double_drift_c0=c0, double_drift_c1=c1, **LOCAL)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.mean(), (x, p, n), allow_unused=True)
    ge = torch.autograd.grad(expected.mean(), x)[0]
    torch.testing.assert_close(ga[0], ge)
    assert ga[1:] == (None, None)


def test_feature_zero_second_coefficient_does_not_evaluate_second_field():
    x, p, n = inputs()
    with mock.patch("drifting_core.imagenet_loss.reverse_drift_field", wraps=reverse_drift_field) as field:
        drift_loss_imagenet(x, p, n, double_drift_c1=0, **LOCAL)
    # The target keeps its original single-step implementation directly.
    assert field.call_count == 0


def test_feature_refreshes_both_queries_and_current_pool():
    x, p, n = inputs()
    with mock.patch("drifting_core.imagenet_loss.reverse_drift_field", wraps=reverse_drift_field) as field:
        drift_loss_imagenet(x, p, n, double_drift_c0=.75, double_drift_c1=.25, **LOCAL)
    assert field.call_count == 2
    first, second = field.call_args_list
    assert not torch.equal(first.args[0], second.args[0])
    # The field derives its current-negative targets from its gen argument.
    assert second.kwargs["fixed_pos"] is p and second.kwargs["fixed_neg"] is n
    assert "fixed_distance_scale" in second.kwargs


def test_sample_zero_second_coefficient_exactly_preserves_baseline():
    x, p, n = inputs()
    original, _ = drift_loss_imagenet(x, p, n, **LOCAL)
    callback = mock.Mock(side_effect=AssertionError("unneeded second pass"))
    actual, _ = sample_double_drift_loss(x, original.mean(), callback, c0=1, c1=0)
    assert actual.item() == original.mean().item()
    torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                               torch.autograd.grad(original.mean(), x)[0])
    callback.assert_not_called()


def test_sample_matches_two_actual_input_steps_and_preserves_generator_graph():
    torch.manual_seed(31)
    generator = torch.nn.Linear(3, 5)
    encoder = torch.nn.Linear(5, 4).requires_grad_(False)
    z = torch.randn(6, 3)
    y = torch.randn(6, 4)
    x = generator(z)
    loss0 = (encoder(x).tanh() - y).square().mean()
    observed = []

    def second(probe):
        assert probe.is_leaf and probe.requires_grad
        observed.append(probe.detach().clone())
        return (encoder(probe).tanh() - y).square().mean(), {}

    # Independent manual construction, retaining no generator graph.
    reference = x.detach().requires_grad_()
    l0 = (encoder(reference).tanh() - y).square().mean()
    g0 = torch.autograd.grad(l0, reference)[0]
    scale = .1 / g0.square().mean().sqrt()
    moved = (reference.detach() - .75 * scale * g0).requires_grad_()
    g1 = torch.autograd.grad((encoder(moved).tanh() - y).square().mean(), moved)[0]
    combined = .75 * g0 + .25 * g1
    loss, info = sample_double_drift_loss(x, loss0, second, global_stats=False)
    torch.testing.assert_close(observed[0], moved)
    loss.backward()
    torch.testing.assert_close(generator.weight.grad, combined.T @ z)
    torch.testing.assert_close(generator.bias.grad, combined.sum(0))
    assert encoder.weight.grad is None
    assert info["double_drift/sample_probe_rms"] == pytest.approx(.075, rel=1e-5)


def test_sample_zero_field_stays_finite():
    x = torch.ones(2, 3, requires_grad=True)
    loss, _ = sample_double_drift_loss(
        x, (x-x.detach()).square().mean(),
        lambda q: ((q-q.detach()).square().mean(), {}), global_stats=False,
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.equal(x.grad, torch.zeros_like(x))


@pytest.mark.parametrize("c0,c1", [(-1, 1), (1, -1), (float("nan"), 1), (1, float("inf")), (0, 0)])
def test_invalid_coefficients(c0, c1):
    with pytest.raises(ValueError):
        validate_double_drift_coefficients(c0, c1)


@pytest.mark.parametrize("mode,encoder_calls", [("off", 2), ("feature", 2), ("sample", 3)])
@pytest.mark.parametrize("replay", [False, True])
def test_full_training_step_and_encoder_call_counts(mode, encoder_calls, replay):
    from train_imagenet_gen import train_step

    class Generator(torch.nn.Module):
        use_bf16 = False

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 16)
            self.calls = 0

        def forward(self, labels, **kwargs):
            self.calls += 1
            return {"samples": self.proj(torch.randn(len(labels), 3)).reshape(-1, 4, 2, 2)}

    class Encoder(torch.nn.Module):
        use_bf16 = False

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 5).requires_grad_(False)
            self.calls = 0

        def get_activations(self, x, **kwargs):
            self.calls += 1
            return {"layer4": self.proj(x.flatten(1)).tanh().unsqueeze(1)}

    torch.manual_seed(79)
    gen, enc = Generator(), Encoder()
    before = gen.proj.weight.detach().clone()
    history = torch.randn(2, 1, 4, 2, 2, requires_grad=True) if replay else None
    history_before = history.detach().clone() if replay else None
    loss, metrics, _ = train_step(
        gen, enc, torch.optim.SGD(gen.parameters(), lr=.01),
        torch.tensor([0, 1]), torch.randn(2, 3, 4, 2, 2),
        torch.randn(2, 2, 4, 2, 2), torch.device("cpu"), 10,
        dict(gen_per_label=4, R_list=[.2], compute_wpos_stats=False,
             global_scale_stats=False, global_fnorm_stats=False,
             double_drift_mode=mode, double_drift_c0=1., double_drift_c1=1.,
             historical_gen_replay=replay, historical_gen_replay_ratio=.35),
        historical_samples=history,
    )
    assert torch.isfinite(loss) and math.isfinite(metrics["g_norm"])
    assert gen.calls == 1 and enc.calls == encoder_calls
    assert not torch.equal(before, gen.proj.weight)
    assert all(p.grad is None for p in enc.parameters())
    if replay:
        assert history.grad is None
        torch.testing.assert_close(history, history_before, rtol=0, atol=0)
