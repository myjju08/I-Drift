"""Upstream force-balance tests plus replay/Double Drift integration contracts."""
import math
from unittest import mock

import pytest
import torch

from drifting_core import imagenet_loss as core
from drifting_core.force_balance import balance_coefficients
from train_imagenet_gen import compute_drift_loss_from_features


@pytest.mark.parametrize("delta,step,anneal,expected", [
    (0, 0, 0, (1, 1)), (.1, 9000, 0, (1.1, .9)),
    (-.1, 0, 0, (.9, 1.1)), (.1, 0, 5005, (1.1, .9)),
    (.1, 2500, 5000, (1.05, .95)), (.1, 5005, 5005, (1, 1)),
    (.1, 10009, 5005, (1, 1)),
])
def test_schedule(delta, step, anneal, expected):
    assert balance_coefficients(delta, step, anneal) == pytest.approx(expected)


@pytest.mark.parametrize("delta,step,anneal", [(float('nan'), 0, 0), (1, 0, 0),
                                              (-1, 0, 0), (.1, -1, 0), (.1, 0, -1)])
def test_schedule_rejects_invalid(delta, step, anneal):
    with pytest.raises(ValueError):
        balance_coefficients(delta, step, anneal)


@pytest.mark.parametrize("a,b", [(1, 1), (1.1, .9), (.9, 1.1)])
def test_weighted_displacements_and_stopgrad(monkeypatch, a, b):
    # Independent explicit pairwise displacement reference, including nonunit
    # group masses: catches both missing x subtraction and upstream scaling.
    x = torch.tensor([[[1., 2.], [-1., 3.]]], requires_grad=True)
    pos = torch.tensor([[[3., -1.], [2., 4.]]], requires_grad=True)
    neg = torch.tensor([[[-2., -3.]]], requires_grad=True)
    affinity = torch.tensor([[[0., .3, .2, .7, .4], [.5, 0., .8, .2, .9]]])
    monkeypatch.setattr(core, '_reverse_mutual_affinity', lambda *args, **kwargs: affinity.clone())
    wp, wn = affinity[:, :, 3:], affinity[:, :, :3]
    neg_targets = torch.cat([x.detach(), neg.detach()], dim=1)
    positive = (wp[..., None] * (pos.detach()[:, None] - x.detach()[:, :, None])).sum(2)
    negative = (wn[..., None] * (neg_targets[:, None] - x.detach()[:, :, None])).sum(2)
    raw = a * wn.sum(2, keepdim=True) * positive - b * wp.sum(2, keepdim=True) * negative
    field, scale, info = core.reverse_drift_field(x, pos, neg, R_list=(.2,),
        global_scale_stats=False, global_fnorm_stats=False,
        attraction_scale=a, repulsion_scale=b, balance_diagnostics=True)
    expected = raw / raw.square().mean().sqrt()
    torch.testing.assert_close(field, expected)
    assert all(math.isfinite(v) for v in info.values())
    loss, _ = core.drift_loss_imagenet(x, pos, neg, R_list=(.2,),
        global_scale_stats=False, global_fnorm_stats=False,
        attraction_scale=a, repulsion_scale=b)
    loss.mean().backward()
    torch.testing.assert_close(x.grad, -2 * expected / (scale * x.numel()))
    assert pos.grad is None and neg.grad is None


@pytest.mark.parametrize("top_k", [0, pytest.param(2, marks=pytest.mark.xfail(
    strict=True, reason="Pre-existing CPU indexed-sum out accumulation breaks translation invariance, including a=b=1; sweep uses dense weights"))])
def test_translation_invariance_common_scale_and_diagnostics(top_k):
    torch.manual_seed(81)
    x, pos, neg = torch.randn(2, 4, 5), torch.randn(2, 3, 5), torch.randn(2, 2, 5)
    options = dict(R_list=(.2,), global_scale_stats=False, global_fnorm_stats=False,
                   top_k_pos=top_k, top_k_neg=top_k)
    base, _, _ = core.reverse_drift_field(x, pos, neg, attraction_scale=1.1, repulsion_scale=.9, **options)
    shift = torch.tensor([2., -3., .5, 1., -2.])
    moved, _, _ = core.reverse_drift_field(x+shift, pos+shift, neg+shift,
        attraction_scale=1.1, repulsion_scale=.9, **options)
    torch.testing.assert_close(base, moved, atol=2e-5, rtol=2e-5)
    scaled, _, _ = core.reverse_drift_field(x, pos, neg,
        attraction_scale=2.2, repulsion_scale=1.8, **options)
    torch.testing.assert_close(base, scaled)
    diagnosed, _, info = core.reverse_drift_field(x, pos, neg,
        attraction_scale=1.1, repulsion_scale=.9, balance_diagnostics=True, **options)
    torch.testing.assert_close(base, diagnosed, rtol=0, atol=0)
    assert info['balance/attraction_scale'] == 1.1


def test_feature_loss_passes_coefficients_and_gradients():
    torch.manual_seed(93)
    x = torch.randn(4, 1, 5, requires_grad=True)
    pos, neg = torch.randn(3, 1, 5), torch.randn(2, 1, 5)
    actual, info = compute_drift_loss_from_features(
        {'global': x}, {'global': pos}, {'global': neg}, B=1, G=4, P=3, N=2,
        weight_neg=None, R_list=(.2,), global_scale_stats=False, global_fnorm_stats=False,
        rev_drift_attraction_scale=1.1, rev_drift_repulsion_scale=.9,
        rev_drift_balance_diagnostics=True)
    expected, _ = core.drift_loss_imagenet(x[:, 0][None], pos[:, 0][None], neg[:, 0][None],
        R_list=(.2,), global_scale_stats=False, global_fnorm_stats=False,
        attraction_scale=1.1, repulsion_scale=.9)
    torch.testing.assert_close(actual, expected.mean())
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected.mean(), x)[0])
    assert info['balance/repulsion_scale/global'] == .9


@pytest.mark.parametrize("a,b", [(1.1, .9), (.9, 1.1)])
def test_replay_weights_preserve_independent_attraction_and_repulsion(monkeypatch, a, b):
    x = torch.tensor([[[1., 2.], [-1., 3.]]], requires_grad=True)
    pos = torch.tensor([[[3., -1.], [2., 4.]]], requires_grad=True)
    neg = torch.tensor([[[-2., -3.]]], requires_grad=True)
    history = torch.tensor([[[4., -2.], [-3., 1.]]], requires_grad=True)
    history_before = history.detach().clone()
    # Pool order is [current gen | real negative | history | real positive].
    affinity = torch.tensor([[[0., .3, .2, .5, .6, .7, .4],
                              [.5, 0., .8, .2, .7, .2, .9]]])
    monkeypatch.setattr(core, '_reverse_mutual_affinity', lambda *args, **kwargs: affinity.clone())
    wg = torch.full((1, 2), .65)
    wh = torch.full((1, 2), .35)
    wn = torch.tensor([[1.3]])
    wp = torch.tensor([[.8, 1.2]])
    weighted = affinity * torch.cat([wg, wn, wh, wp], dim=1)[:, None]
    negative_weights, positive_weights = weighted[:, :, :5], weighted[:, :, 5:]
    negative_targets = torch.cat([x.detach(), neg.detach(), history.detach()], dim=1)
    attractive = (positive_weights[..., None] *
                  (pos.detach()[:, None] - x.detach()[:, :, None])).sum(2)
    repulsive = (negative_weights[..., None] *
                 (negative_targets[:, None] - x.detach()[:, :, None])).sum(2)
    raw = (a * negative_weights.sum(2, keepdim=True) * attractive -
           b * positive_weights.sum(2, keepdim=True) * repulsive)
    expected = raw / raw.square().mean().sqrt()
    options = dict(R_list=(.2,), global_scale_stats=False, global_fnorm_stats=False,
                   historical_gen=history, weight_gen=wg, weight_history=wh,
                   weight_neg=wn, weight_pos=wp, attraction_scale=a, repulsion_scale=b)
    field, scale, info = core.reverse_drift_field(x, pos, neg, **options)
    torch.testing.assert_close(field, expected)
    assert not field.requires_grad
    assert info['history/current_mass'] == pytest.approx(1.3)
    assert info['history/replay_mass'] == pytest.approx(.7)
    loss, _ = core.drift_loss_imagenet(x, pos, neg, **options)
    grads = torch.autograd.grad(loss.mean(), (x, pos, neg, history), allow_unused=True)
    torch.testing.assert_close(grads[0], -2 * expected / (scale * x.numel()))
    assert grads[1:] == (None, None, None)
    torch.testing.assert_close(history, history_before, rtol=0, atol=0)


@pytest.mark.parametrize("c0,c1", [(1., 1.), (.75, .25)])
@pytest.mark.parametrize("fused", [False, True])
def test_balanced_double_drift_uses_both_fields_and_fixed_replay(c0, c1, fused):
    torch.manual_seed(482)
    x = torch.randn(2, 4, 5, requires_grad=True)
    pos = torch.randn(2, 3, 5, requires_grad=True)
    neg = torch.randn(2, 2, 5, requires_grad=True)
    history = torch.randn(2, 3, 5, requires_grad=True)
    options = dict(
        R_list=(.2, .05), historical_gen=history,
        weight_gen=torch.full((2, 4), .65),
        weight_history=torch.full((2, 3), .35 * 4 / 3),
        attraction_scale=1.1, repulsion_scale=.9, balance_diagnostics=True,
        global_scale_stats=True, global_fnorm_stats=True,
        fuse_fnorm_across_R=fused,
    )
    first, scale, first_info = core.reverse_drift_field(x, pos, neg, **options)
    moved = x.detach() + c0 * first * scale
    second, second_scale, _ = core.reverse_drift_field(
        moved, pos, neg, fixed_distance_scale=first_info['scale'], **options)
    torch.testing.assert_close(scale, second_scale, rtol=0, atol=0)
    expected = (x / scale - (x.detach() / scale + c0 * first + c1 * second)).square().mean((-1, -2))
    actual, info = core.drift_loss_imagenet(
        x, pos, neg, double_drift_c0=c0, double_drift_c1=c1, **options)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.mean(), (x, pos, neg, history), allow_unused=True)
    expected_grad = torch.autograd.grad(expected.mean(), x)[0]
    torch.testing.assert_close(actual_grads[0], expected_grad)
    assert actual_grads[1:] == (None, None, None)
    for prefix in ('', 'double_drift/second/'):
        assert info[prefix + 'balance/attraction_scale'] == 1.1
        assert info[prefix + 'balance/repulsion_scale'] == .9


@pytest.mark.parametrize("double", [False, True])
def test_balance_diagnostics_do_not_change_loss_or_gradient_and_can_be_disabled(double):
    torch.manual_seed(37)
    x = torch.randn(2, 4, 5, requires_grad=True)
    pos, neg = torch.randn(2, 3, 5), torch.randn(2, 2, 5)
    options = dict(R_list=(.2,), global_scale_stats=False, global_fnorm_stats=False,
                   attraction_scale=.9, repulsion_scale=1.1,
                   double_drift_c0=1., double_drift_c1=float(double))
    quiet, quiet_info = core.drift_loss_imagenet(x, pos, neg, **options)
    diagnosed, info = core.drift_loss_imagenet(x, pos, neg, balance_diagnostics=True, **options)
    disabled, disabled_info = core.drift_loss_imagenet(
        x, pos, neg, balance_diagnostics=True, collect_diagnostics=False, **options)
    for other in (diagnosed, disabled):
        torch.testing.assert_close(quiet, other, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(quiet.mean(), x, retain_graph=True)[0],
                                   torch.autograd.grad(other.mean(), x)[0], rtol=0, atol=0)
    assert not any('balance/' in key for key in quiet_info)
    assert not any('balance/' in key for key in disabled_info)
    assert info['balance/repulsion_scale'] == 1.1


@pytest.mark.parametrize("entrypoint", [core.reverse_drift_field, core.drift_loss_imagenet])
@pytest.mark.parametrize("key,value", [('attraction_scale', 0), ('repulsion_scale', -1),
                                       ('attraction_scale', float('nan')),
                                       ('repulsion_scale', float('inf'))])
def test_field_and_loss_reject_invalid_independent_scales(entrypoint, key, value):
    with pytest.raises(ValueError, match='finite and positive'):
        entrypoint(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), **{key: value})


def test_balanced_double_drift_coincident_features_preserve_zero_field():
    x = torch.zeros(2, 4, 5, requires_grad=True)
    pos, neg = torch.zeros(2, 3, 5), torch.zeros(2, 2, 5)
    loss, info = core.drift_loss_imagenet(
        x, pos, neg, double_drift_c0=1, double_drift_c1=1,
        attraction_scale=1.1, repulsion_scale=.9,
        global_scale_stats=False, global_fnorm_stats=False)
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss.mean(), x)[0], torch.zeros_like(x), rtol=0, atol=0)
    assert info['scale'] == info['double_drift/second/scale']
    field, input_scale, field_info = core.reverse_drift_field(
        x, pos, neg, fixed_distance_scale=0,
        attraction_scale=1.1, repulsion_scale=.9,
        global_scale_stats=False, global_fnorm_stats=False)
    torch.testing.assert_close(field, torch.zeros_like(field), rtol=0, atol=0)
    assert input_scale.item() == pytest.approx(.001) and field_info['scale'] == 0


class _BalanceGenerator(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.samples = torch.nn.Parameter(torch.randn(8, 4, 2, 2))
        self.calls = 0

    def forward(self, labels, **kwargs):
        self.calls += 1
        assert len(labels) == len(self.samples)
        return {'samples': self.samples}


class _BalanceEncoder(torch.nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(16, 5).requires_grad_(False)
        self.calls = 0

    def get_activations(self, x, **kwargs):
        self.calls += 1
        return {'layer4': self.projection(x.flatten(1)).tanh().unsqueeze(1)}


def _tiny_balance_training_inputs(mode, replay, step, **overrides):
    torch.manual_seed(659)
    generator, encoder = _BalanceGenerator(), _BalanceEncoder()
    optimizer = torch.optim.SGD(generator.parameters(), lr=.01)
    history = torch.randn(2, 1, 4, 2, 2, requires_grad=True) if replay else None
    cfg = dict(
        gen_per_label=4, R_list=[.2], compute_wpos_stats=False,
        global_scale_stats=False, global_fnorm_stats=False,
        double_drift_mode=mode, double_drift_c0=1., double_drift_c1=1.,
        historical_gen_replay=replay, historical_gen_replay_ratio=.35,
        rev_drift_balance_diagnostics=True,
    )
    cfg.update(overrides)
    return dict(
        generator=generator, feature_extractor=encoder, optimizer=optimizer,
        labels=torch.tensor([0, 1]), pos_samples=torch.randn(2, 3, 4, 2, 2),
        neg_samples=torch.randn(2, 2, 4, 2, 2), device=torch.device('cpu'),
        step=step, cfg=cfg, historical_samples=history,
    )


@pytest.mark.parametrize('mode', ['off', 'feature', 'sample'])
@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('step,effective_delta', [(0, .1), (50, .05), (100, 0.), (150, 0.)])
def test_train_step_balance_annealing_matches_constant_effective_delta(mode, replay, step, effective_delta):
    import train_imagenet_gen as trainer

    results = []
    for delta, anneal in ((.1, 100), (effective_delta, 0)):
        args = _tiny_balance_training_inputs(
            mode, replay, step, rev_drift_balance_delta=delta,
            rev_drift_balance_anneal_steps=anneal)
        generator, encoder = args['generator'], args['feature_extractor']
        history = args['historical_samples']
        before = generator.samples.detach().clone()
        history_before = history.detach().clone() if replay else None
        with mock.patch.object(trainer, 'drift_loss_imagenet', wraps=core.drift_loss_imagenet) as field_loss:
            loss, metrics, _ = trainer.train_step(**args)
        expected_scales = (1 + effective_delta, 1 - effective_delta)
        assert field_loss.call_count == (2 if mode == 'sample' else 1)
        for call in field_loss.call_args_list:
            assert call.kwargs['attraction_scale'] == expected_scales[0]
            assert call.kwargs['repulsion_scale'] == expected_scales[1]
            assert call.kwargs['balance_diagnostics']
            assert (call.kwargs['historical_gen'] is not None) == replay
        assert metrics['balance/attraction_scale/layer4'] == expected_scales[0]
        assert metrics['balance/repulsion_scale/layer4'] == expected_scales[1]
        assert torch.isfinite(loss)
        assert not torch.equal(before, generator.samples)
        assert generator.calls == 1
        assert encoder.calls == (3 if mode == 'sample' else 2)
        assert all(parameter.grad is None for parameter in encoder.parameters())
        if replay:
            assert history.grad is None
            torch.testing.assert_close(history, history_before, rtol=0, atol=0)
        results.append((loss.detach(), generator.samples.detach().clone(), generator.samples.grad.clone()))
    # Check the actual optimizer update and outer gradient, including sample
    # Double Drift's second encoder pass, against the constant-delta run.
    for scheduled, constant in zip(*results):
        torch.testing.assert_close(scheduled, constant, rtol=0, atol=0)


@pytest.mark.parametrize('drift_matching', ['fwd-drift', 'dual-drift', 'cf-drift'])
def test_train_step_nonreverse_force_balance_fails_before_forward_or_update(drift_matching):
    from train_imagenet_gen import train_step

    args = _tiny_balance_training_inputs(
        'off', False, 10, drift_matching=drift_matching,
        rev_drift_balance_delta=.1, rev_drift_balance_anneal_steps=100)
    generator, encoder = args['generator'], args['feature_extractor']
    before = generator.samples.detach().clone()
    with pytest.raises(ValueError, match='force balance currently supports rev-drift only'):
        train_step(**args)
    assert generator.calls == 0 and encoder.calls == 0
    assert generator.samples.grad is None
    torch.testing.assert_close(generator.samples, before, rtol=0, atol=0)
