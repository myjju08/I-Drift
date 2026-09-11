"""Encoder-only frozen MAE preserves canonical features and gradient paths."""
import copy
from unittest import mock

import pytest
import torch

from models.mae_resnet import MAEResNet
from models.frozen_mae import FrozenMAEFeatureExtractor


def _model():
    torch.manual_seed(621)
    return MAEResNet(
        num_classes=5, in_channels=4, base_channels=4, input_patch_size=1,
        layers=(3, 4, 6, 3), use_bf16=False,
    ).eval().requires_grad_(False)


# The new base includes conv1 statistics when raw stage maps are requested.
# Preserve its 51-context adapter API; ordinary production extraction has 44.
def _features(model, x):
    return model.get_activations(
        x, active_stages=["stage3", "stage4"], every_k_block=2,
        patch_mean_size=[2, 4], patch_std_size=[2, 4],
        return_stage_features=True,
    )


def _loss(features):
    return sum((index + 1) * value.square().mean() / len(features)
               for index, value in enumerate(features.values()))


def test_production_44_contexts_match_base_without_stage_map_capture():
    source = _model()
    runtime = FrozenMAEFeatureExtractor(copy.deepcopy(source), {})
    values = torch.randn(2, 4, 32, 32)
    options = dict(active_stages=["stage3", "stage4"], every_k_block=2,
                   patch_mean_size=[2, 4], patch_std_size=[2, 4])
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    expected = source.get_activations(first, **options)
    actual = runtime.get_activations(second, **options)
    assert len(expected) == len(actual) == 44
    assert list(expected) == list(actual)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    _loss(expected).backward()
    _loss(actual).backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)


@pytest.fixture
def _native_cpu_convolution():
    # Isolate chunking algebra from oneDNN's shape-dependent CPU algorithms.
    # Actual CUDA NCHW->NHWC/TF32/Inductor rounding has its own fixed gate in
    # scripts/validate_mae_runtime.py; this is not an approval of those options.
    # Other suites legitimately use four CPU threads. Native channels-last
    # BLAS reductions are thread-count dependent, so isolate this fixed-math
    # algebra fixture and restore its caller's settings afterward.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.backends.mkldnn.flags(enabled=False):
            yield
    finally:
        torch.set_num_threads(previous_threads)


def test_encoder_only_default_is_bitwise_reference_with_base_stage_feature_contexts():
    reference = _model()
    runtime = FrozenMAEFeatureExtractor(copy.deepcopy(reference), {})
    values = torch.randn(3, 4, 32, 32)
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    expected, expected_maps = _features(reference, first)
    rng = torch.get_rng_state().clone()
    actual, actual_maps = _features(runtime, second)
    assert len(expected) == len(actual) == 51
    assert list(expected) == list(actual)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in expected_maps:
        torch.testing.assert_close(actual_maps[key], expected_maps[key], rtol=0, atol=0)
    _loss(expected).backward()
    _loss(actual).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert not hasattr(runtime, "decoder") and not hasattr(runtime, "fc")
    assert all(key.startswith("encoder.") for key in runtime.state_dict())
    assert not torch.equal(actual["layer3"], actual["layer3_blk6"])
    runtime.train()
    assert not runtime.training and not runtime.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in runtime.parameters())


@pytest.mark.parametrize("remat,channels_last", [(False, False), (True, False), (False, True), (True, True)])
def test_microbatch_real_and_generated_maps_gradients_state_and_rng(remat, channels_last, _native_cpu_convolution):
    reference = _model()
    runtime = FrozenMAEFeatureExtractor(copy.deepcopy(reference), {
        "feature_real_microbatch_size": 3, "feature_generated_microbatch_size": 2,
        "feature_channels_last": channels_last,
    })
    runtime.use_remat = remat
    values = torch.randn(5, 4, 32, 32)
    if channels_last:
        # Compare canonical and chunked algebra within the same layout. The
        # cross-layout candidate must separately pass the real GPU gate.
        reference.to(memory_format=torch.channels_last)
        values = values.contiguous(memory_format=torch.channels_last)
    state = {key: value.clone() for key, value in runtime.state_dict().items()}
    with torch.no_grad(), mock.patch.object(runtime.encoder, "_forward_chunk", wraps=runtime.encoder._forward_chunk) as forward:
        expected, _ = _features(reference, values)
        actual, _ = _features(runtime, values)
        assert forward.call_count == 2
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=3e-5, atol=5e-6)
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    expected, _ = _features(reference, first)
    rng = torch.get_rng_state().clone()
    with mock.patch.object(runtime.encoder, "_forward_chunk", wraps=runtime.encoder._forward_chunk) as forward:
        actual, _ = _features(runtime, second)
        assert forward.call_count == 3
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=3e-5, atol=5e-6)
    _loss(expected).backward()
    _loss(actual).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=5e-5, atol=1e-5)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in runtime.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_runtime_preserves_maps_gradients_and_deepcopy_bindings(backend):
    torch._dynamo.reset()
    reference = FrozenMAEFeatureExtractor(_model(), {"feature_microbatch_size": 2})
    compiled = copy.deepcopy(reference)
    compiled.encoder.compile_backbone = True
    compiled.encoder.compile_backend = backend
    values = torch.randn(3, 4, 32, 32)
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    expected, _ = _features(reference, first)
    actual, _ = _features(compiled, second)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=3e-5, atol=5e-6)
    _loss(expected).backward()
    _loss(actual).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=5e-5, atol=1e-5)
    cloned = copy.deepcopy(compiled)
    assert cloned.encoder._compiled_encoder_forward is None
    assert list(cloned.state_dict()) == list(reference.state_dict())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(cloned.parameters(), compiled.parameters()))
    torch._dynamo.reset()


def test_runtime_option_defaults_and_inductor_precision_flags():
    runtime = FrozenMAEFeatureExtractor(_model(), {})
    assert runtime.encoder.real_microbatch_size == 0
    assert runtime.encoder.generated_microbatch_size == 0
    assert not runtime.encoder.channels_last
    assert not runtime.encoder.compile_backbone
    assert runtime.encoder.compile_layout_optimization is None
    runtime.encoder.compile_backbone = True
    with mock.patch("models.frozen_mae.torch.compile", side_effect=lambda function, **kwargs: function) as compile_call:
        with torch.no_grad():
            _features(runtime, torch.randn(2, 4, 32, 32))
    assert compile_call.call_args.kwargs["options"] == {
        "triton.cudagraphs": False, "emulate_precision_casts": True,
    }
    assert compile_call.call_args.kwargs["fullgraph"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_compile_layout_policy_is_scoped(enabled):
    import torch._inductor.config as inductor_config
    before = inductor_config.layout_optimization
    runtime = FrozenMAEFeatureExtractor(_model(), {
        "feature_compile_backbone": True,
        "feature_compile_layout_optimization": enabled,
    })
    with mock.patch("models.frozen_mae.torch.compile", side_effect=lambda function, **kwargs: function) as compile_call:
        with torch.no_grad():
            _features(runtime, torch.randn(2, 4, 32, 32))
    assert compile_call.call_args.kwargs["options"]["layout_optimization"] is enabled
    assert inductor_config.layout_optimization is before


@pytest.mark.parametrize("key", ["feature_microbatch_size", "feature_real_microbatch_size", "feature_generated_microbatch_size"])
def test_negative_microbatch_rejected(key):
    with pytest.raises(ValueError, match="nonnegative"):
        FrozenMAEFeatureExtractor(_model(), {key: -1})


@pytest.mark.parametrize("stages", [["stage1", "stage2"], ["stage1"], [], ["stage1", "stage2", "stage3", "stage4"]])
def test_selective_remat_is_bitwise_canonical_features_gradient_and_rng(stages):
    reference = _model()
    reference.use_remat = True
    runtime = FrozenMAEFeatureExtractor(copy.deepcopy(reference), {"feature_remat_stages": stages})
    values = torch.randn(3, 4, 32, 32)
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    expected, expected_maps = _features(reference, first)
    state = {key: value.clone() for key, value in runtime.state_dict().items()}
    rng = torch.get_rng_state().clone()
    actual, actual_maps = _features(runtime, second)
    assert list(actual) == list(expected) and len(actual) == 51
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in expected_maps:
        torch.testing.assert_close(actual_maps[key], expected_maps[key], rtol=0, atol=0)
    _loss(expected).backward()
    _loss(actual).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in runtime.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_selective_remat_only_checkpoints_selected_blocks():
    model = _model()
    model.use_remat = True
    runtime = FrozenMAEFeatureExtractor(model, {"feature_remat_stages": ["stage2", "stage1", "stage1"]})
    assert runtime.encoder.remat_stages == ("stage1", "stage2")
    selected = {id(block) for stage in runtime.encoder.stages[:2] for block in stage}
    import models.frozen_mae as module
    with mock.patch.object(module, "checkpoint", wraps=module.checkpoint) as remat:
        _features(runtime, torch.randn(2, 4, 32, 32, requires_grad=True))
    assert len(remat.call_args_list) == 7
    assert {id(call.args[0]) for call in remat.call_args_list} == selected
    with torch.no_grad(), mock.patch.object(module, "checkpoint") as remat:
        _features(runtime, torch.randn(2, 4, 32, 32))
    remat.assert_not_called()


@pytest.mark.parametrize("invalid", ["stage1", ["stage5"], ["layer1"]])
def test_invalid_selective_remat_policy_is_rejected(invalid):
    with pytest.raises(ValueError):
        FrozenMAEFeatureExtractor(_model(), {"feature_remat_stages": invalid})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_groupnorm_input_layout_output_dtype_and_gradient_are_bitwise_canonical(dtype):
    from models.mae_resnet import _DtypePreservingGroupNorm
    from models.frozen_mae import _CanonicalLayoutGroupNorm
    torch.manual_seed(519)
    reference = _DtypePreservingGroupNorm(2, 8, eps=1e-6).eval().requires_grad_(False)
    hybrid = _CanonicalLayoutGroupNorm(copy.deepcopy(reference)).eval().requires_grad_(False)
    values = torch.randn(3, 8, 8, 8).to(dtype)
    first = values.clone().requires_grad_()
    second = values.contiguous(memory_format=torch.channels_last).requires_grad_()
    expected = reference(first)
    with mock.patch("torch.nn.functional.group_norm", wraps=torch.nn.functional.group_norm) as normalize:
        actual = hybrid(second)
    assert normalize.call_args.args[0].is_contiguous()
    assert actual.dtype == expected.dtype == dtype
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.linspace(-0.25, 0.75, actual.numel()).reshape_as(actual)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)
    for key, value in hybrid.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)


def test_hybrid_runtime_only_replaces_norms_preserves_keys_weights_and_freeze():
    from models.frozen_mae import _CanonicalLayoutGroupNorm
    source = _model()
    expected = {key: value.clone() for key, value in source.encoder.state_dict().items()}
    norm_count = sum(isinstance(module, torch.nn.GroupNorm) for module in source.encoder.modules())
    runtime = FrozenMAEFeatureExtractor(source, {
        "feature_channels_last": True,
        "feature_channels_last_preserve_groupnorm_layout": True,
    })
    assert sum(isinstance(module, _CanonicalLayoutGroupNorm) for module in runtime.encoder.modules()) == norm_count
    assert list(runtime.encoder.state_dict()) == list(expected)
    for key, value in runtime.encoder.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in runtime.parameters())
    with mock.patch("torch.nn.functional.group_norm", wraps=torch.nn.functional.group_norm) as normalize:
        features, _ = _features(runtime, torch.randn(2, 4, 32, 32))
    assert len(features) == 51
    assert all(call.args[0].is_contiguous() for call in normalize.call_args_list)


def test_hybrid_option_requires_channels_last():
    with pytest.raises(ValueError, match="feature_channels_last"):
        FrozenMAEFeatureExtractor(_model(), {"feature_channels_last_preserve_groupnorm_layout": True})


@pytest.mark.parametrize("remat", [False, True])
def test_precast_integration_is_bitwise_for_all_features_bf16_gradients_and_state(remat):
    from models.frozen_conv import FrozenPrecastConv2d
    source = _model()
    source.use_bf16 = True
    source.fuse_stats = True
    source.use_remat = remat
    config = {"feature_real_microbatch_size": 3, "feature_generated_microbatch_size": 2,
              "feature_remat_stages": ["stage1", "stage2"]}
    reference = FrozenMAEFeatureExtractor(copy.deepcopy(source), config)
    parameter_ids = {name: id(value) for name, value in source.encoder.named_parameters()}
    actual = FrozenMAEFeatureExtractor(source, {**config, "feature_precast_conv_weights": True})
    assert {name: id(value) for name, value in actual.encoder.named_parameters()} == parameter_ids
    state = {key: value.clone() for key, value in actual.state_dict().items()}
    real = torch.randn(5, 4, 32, 32)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        real_expected, _ = _features(reference, real)
        real_actual, _ = _features(actual, real)
    for key in real_expected:
        torch.testing.assert_close(real_actual[key], real_expected[key], rtol=0, atol=0)
    convolutions = [module for module in actual.modules() if isinstance(module, FrozenPrecastConv2d)]
    caches = [module._precast_weight for module in convolutions]
    assert caches and all(value is not None and not torch.is_inference(value) for value in caches)
    first = real.bfloat16().requires_grad_()
    second = first.detach().clone().requires_grad_()
    rng = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = _features(reference, first)
        observed, _ = _features(actual, second)
        expected_loss, observed_loss = _loss(expected), _loss(observed)
    assert len(expected) == len(observed) == 51
    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
    expected_loss.backward()
    observed_loss.backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all(module._precast_weight is value for module, value in zip(convolutions, caches))
    assert list(actual.state_dict()) == list(reference.state_dict())
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    actual.cpu()
    assert all(module._precast_weight is None for module in convolutions)


def test_precast_and_compiled_encoder_combination_is_explicitly_rejected():
    with pytest.raises(ValueError, match="feature_compile_backbone=false"):
        FrozenMAEFeatureExtractor(_model(), {
            "feature_precast_conv_weights": True, "feature_compile_backbone": True,
        })
