"""CPU integration of upstream Double Drift with the production DINO RGB path.

Only checkpoint loading and the ResNet constructor are substituted.  The real
SSL extractor still normalizes RGB inputs, selects stage 3/4 maps, chunks and
rematerializes nonlinear features, and derives the complete drift feature set.
"""

import math
from unittest import mock

import pytest
import torch
from torch import nn

from models.ssl_resnet import SSLResNetFeatureExtractor
from train_imagenet_gen import train_step


class _ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, x):
        return x + 0.1 * self.conv(x).tanh()


class _TinyResNet(nn.Module):
    """Learned nonlinear blocks with ResNet-50's 3/4/6/3 stage geometry."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        for stage, count in enumerate((3, 4, 6, 3), start=1):
            setattr(self, f"layer{stage}", nn.Sequential(*[
                _ResidualBlock() for _ in range(count)
            ]))
        self.fc = nn.Identity()


class _Generator(nn.Module):
    use_bf16 = False

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 3 * 8 * 8)
        self.calls = 0

    def forward(self, labels, **kwargs):
        self.calls += 1
        noise = torch.randn(len(labels), 5)
        return {"samples": self.proj(noise).tanh().reshape(-1, 3, 8, 8)}


@pytest.fixture(autouse=True)
def _cpu_only_single_thread(monkeypatch):
    def forbid_cuda_init():
        raise AssertionError("DINO Double Drift integration tests must stay on CPU")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_cuda_init)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def _extractor(use_remat):
    backbone = _TinyResNet()
    state = {key: value.clone() for key, value in backbone.state_dict().items()}
    with (
        mock.patch("models.ssl_resnet.resnet50", return_value=backbone),
        mock.patch("models.ssl_resnet._load_backbone_state", return_value=state),
        mock.patch("models.ssl_resnet.load_vae") as load_vae,
    ):
        extractor = SSLResNetFeatureExtractor(
            "dino_resnet50", "/unused/dino.pth", use_latent=False,
            use_bf16=False, use_remat=use_remat, spatial_pool=2,
            real_microbatch_size=3, generated_microbatch_size=2,
            include_norm_x=True, device=torch.device("cpu"),
        )
    load_vae.assert_not_called()
    return extractor


def _run_step(mode, use_remat, *, c0=0.75, c1=0.25):
    torch.manual_seed(719)
    extractor = _extractor(use_remat)
    generator = _Generator()
    encoder_before = {
        key: value.detach().clone() for key, value in extractor.state_dict().items()
    }
    generator_before = generator.proj.weight.detach().clone()
    positive = torch.randn(2, 3, 3, 8, 8, requires_grad=True)
    negative = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
    history = torch.randn(2, 1, 3, 8, 8, requires_grad=True)
    anchors = (positive, negative, history)
    anchors_before = tuple(value.detach().clone() for value in anchors)
    records = []
    get_activations = extractor.get_activations

    def record_activations(samples, **kwargs):
        # Hooks are scoped to the forward so rematerialization during backward
        # does not inflate the observed extraction microbatch sizes.
        normalized_chunks = []
        hook = extractor.backbone.conv1.register_forward_pre_hook(
            lambda _module, args: normalized_chunks.append(args[0].detach().clone())
        )
        try:
            output = get_activations(samples, **kwargs)
        finally:
            hook.remove()
        records.append({
            "samples": samples.detach().clone(),
            "is_leaf": samples.is_leaf,
            "requires_grad": samples.requires_grad,
            "grad_enabled": torch.is_grad_enabled(),
            "normalized_chunks": normalized_chunks,
            "features": {key: value.detach().clone() for key, value in output.items()},
        })
        return output

    config = dict(
        gen_per_label=3, R_list=[0.2, 0.05, 0.02],
        drift_matching="rev-drift", compute_wpos_stats=False,
        global_scale_stats=False, global_fnorm_stats=False,
        double_drift_mode=mode, double_drift_c0=c0, double_drift_c1=c1,
        double_drift_sample_step_rms=0.1,
        historical_gen_replay=True, historical_gen_replay_ratio=0.35,
        # Keep gradients unclipped so comparisons inspect the full method.
        max_grad_norm=1e6,
        activation_kwargs=dict(
            active_stages=["stage3", "stage4"],
            with_global=True, with_norm_x=True,
            every_k_block=2, exclude_terminal_block=False,
            patch_mean_size=[2, 4], patch_std_size=[2, 4],
            use_std=True, use_mean=True,
        ),
    )
    with mock.patch.object(extractor, "get_activations", side_effect=record_activations):
        loss, metrics, _ = train_step(
            generator, extractor, torch.optim.SGD(generator.parameters(), lr=0.01),
            torch.tensor([0, 1]), positive, negative, torch.device("cpu"),
            10, config, historical_samples=history,
        )

    assert torch.isfinite(loss) and math.isfinite(metrics["g_norm"])
    assert generator.calls == 1
    assert not torch.equal(generator_before, generator.proj.weight)
    for parameter in generator.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert not extractor.training
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in extractor.parameters())
    for key, value in extractor.state_dict().items():
        torch.testing.assert_close(value, encoder_before[key], rtol=0, atol=0)
    for value, before in zip(anchors, anchors_before):
        assert value.grad is None
        torch.testing.assert_close(value, before, rtol=0, atol=0)

    expected_names = {"global", "norm_x"}
    for name in ("layer3", "layer4", "layer3_blk2", "layer3_blk4",
                 "layer3_blk6", "layer4_blk2"):
        expected_names.update(name + suffix for suffix in (
            "", "_mean", "_std", "_mean_2", "_mean_4", "_std_2", "_std_4"
        ))
    for record in records:
        assert set(record["features"]) == expected_names
        # Validate that the production RGB normalization actually ran on every
        # chunk; global/norm_x intentionally retain original image coordinates.
        normalized = (0.5 * (record["samples"] + 1) - extractor.imagenet_mean)
        normalized = normalized / extractor.imagenet_std
        torch.testing.assert_close(torch.cat(record["normalized_chunks"]), normalized)
        torch.testing.assert_close(
            record["features"]["global"], record["samples"].flatten(1).unsqueeze(1)
        )
        assert all(torch.isfinite(value).all() for value in record["features"].values())

    assert not records[0]["grad_enabled"]
    assert [len(chunk) for chunk in records[0]["normalized_chunks"]] == [3, 3, 3, 3]
    for record in records[1:]:
        assert record["grad_enabled"] and record["requires_grad"]
        assert [len(chunk) for chunk in record["normalized_chunks"]] == [2, 2, 2]
    return dict(
        loss=loss.detach(), metrics=metrics, records=records,
        gradients={name: value.grad.clone() for name, value in generator.named_parameters()},
        weights={name: value.detach().clone() for name, value in generator.named_parameters()},
    )


@pytest.mark.parametrize("use_remat", [False, True])
@pytest.mark.parametrize("mode", ["off", "feature", "sample"])
def test_dino_rgb_training_step_and_sample_feature_refresh(mode, use_remat):
    result = _run_step(mode, use_remat)
    records = result["records"]
    assert len(records) == (3 if mode == "sample" else 2)
    if mode == "sample":
        generated, moved = records[1:]
        assert not generated["is_leaf"]
        assert moved["is_leaf"]
        assert not torch.equal(generated["samples"], moved["samples"])
        assert not torch.equal(generated["features"]["layer4"],
                               moved["features"]["layer4"])
        assert result["metrics"]["double_drift/sample_field_evaluations"] == 2.0
        assert result["metrics"]["double_drift/sample_probe_rms"] == pytest.approx(0.075, rel=1e-5)


@pytest.mark.parametrize("use_remat", [False, True])
@pytest.mark.parametrize("mode", ["feature", "sample"])
def test_dino_zero_second_step_exactly_preserves_baseline(mode, use_remat):
    baseline = _run_step("off", use_remat)
    actual = _run_step(mode, use_remat, c0=1.0, c1=0.0)
    assert len(actual["records"]) == 2
    torch.testing.assert_close(actual["loss"], baseline["loss"], rtol=0, atol=0)
    for name in baseline["gradients"]:
        torch.testing.assert_close(actual["gradients"][name], baseline["gradients"][name], rtol=0, atol=0)
        torch.testing.assert_close(actual["weights"][name], baseline["weights"][name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["feature", "sample"])
def test_dino_double_drift_rematerialization_preserves_generator_gradient(mode):
    direct = _run_step(mode, False)
    rematerialized = _run_step(mode, True)
    torch.testing.assert_close(rematerialized["loss"], direct["loss"], rtol=0, atol=0)
    for name in direct["gradients"]:
        torch.testing.assert_close(rematerialized["gradients"][name], direct["gradients"][name], rtol=0, atol=0)
