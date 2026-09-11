"""Exact-output/input-gradient and lifetime gates for frozen autocast casts."""
import copy
import hashlib

import pytest
import torch
from torch import nn

from models.frozen_conv import FrozenPrecastConv2d


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source_conv(*, bias=True, groups=1, padding_mode="zeros", dtype=torch.float32, device="cpu"):
    devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(872)
        return nn.Conv2d(4, 8, 3, padding=1, groups=groups, bias=bias,
                         padding_mode=padding_mode, dtype=dtype, device=device).eval().requires_grad_(False)


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def forward_backward(module, values, *, autocast_dtype=torch.bfloat16, enabled=True):
    x = values.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    with torch.autocast(x.device.type, dtype=autocast_dtype, enabled=enabled):
        y = module(x)
        # Nonconstant, exactly shared upstream signal exercises all gradients.
        upstream = torch.linspace(-.7, .9, y.numel(), device=y.device).reshape(y.shape).to(y.dtype)
    gradient, = torch.autograd.grad(y, x, upstream)
    return y.detach(), gradient


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
def test_exact_output_input_grad_and_canonical_state(bias, groups, padding_mode, autocast_dtype):
    original = source_conv(bias=bias, groups=groups, padding_mode=padding_mode)
    before_rng = torch.get_rng_state().clone()
    wrapped = FrozenPrecastConv2d(original)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert wrapped.weight is original.weight and wrapped.bias is original.bias
    assert state_hash(wrapped) == state_hash(original)
    assert list(wrapped.state_dict()) == list(original.state_dict())
    assert list(wrapped.named_parameters()) == list(original.named_parameters())
    values = torch.linspace(-1, 1, 2 * 4 * 7 * 7).reshape(2, 4, 7, 7)
    reference = forward_backward(original, values, autocast_dtype=autocast_dtype)
    actual = forward_backward(wrapped, values, autocast_dtype=autocast_dtype)
    assert all(torch.equal(left, right) for left, right in zip(reference, actual))
    cached = wrapped._precast_weight
    actual_again = forward_backward(wrapped, values, autocast_dtype=autocast_dtype)
    assert wrapped._precast_weight is cached
    assert all(torch.equal(left, right) for left, right in zip(reference, actual_again))
    assert cached.dtype == autocast_dtype and cached.grad_fn is None and not cached.requires_grad
    assert torch.equal(cached, original.weight.to(autocast_dtype))
    assert state_hash(wrapped) == state_hash(original)
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
def test_input_dtype_layout_and_disabled_nested_autocast(input_dtype, channels_last):
    original = source_conv()
    if channels_last:
        original.to(memory_format=torch.channels_last)
    wrapped = FrozenPrecastConv2d(original)
    values = torch.linspace(-1, 1, 2 * 4 * 7 * 7).reshape(2, 4, 7, 7).to(input_dtype)
    if channels_last:
        values = values.contiguous(memory_format=torch.channels_last)
    assert all(torch.equal(a, b) for a, b in zip(forward_backward(original, values), forward_backward(wrapped, values)))
    if channels_last:
        assert wrapped._precast_weight.is_contiguous(memory_format=torch.channels_last)
    values = values.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = forward_backward(original, values, enabled=False)
        actual = forward_backward(wrapped, values, enabled=False)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def warm(module):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        module(torch.ones(1, 4, 6, 6))
    return module._precast_weight


@pytest.mark.parametrize("mutation", ["weight_add", "bias_add", "replace_weight", "replace_bias", "load", "load_assign", "parent_load", "layout", "cpu_offload", "dtype"])
def test_invalidation(mutation):
    wrapped = FrozenPrecastConv2d(source_conv())
    cached = warm(wrapped)
    if mutation in ("weight_add", "bias_add"):
        with torch.no_grad():
            getattr(wrapped, mutation.split("_")[0]).add_(.125)
    elif mutation in ("replace_weight", "replace_bias"):
        name = mutation.split("_")[1]
        setattr(wrapped, name, nn.Parameter(getattr(wrapped, name).detach() + .125, requires_grad=False))
    elif mutation in ("load", "load_assign"):
        replacement = {name: tensor.detach().clone() + .125 for name, tensor in wrapped.state_dict().items()}
        wrapped.load_state_dict(replacement, strict=True, assign=mutation == "load_assign")
        assert wrapped._precast_weight is None
    elif mutation == "parent_load":
        parent = nn.Sequential(wrapped)
        parent.load_state_dict({name: tensor.detach().clone() + .125 for name, tensor in parent.state_dict().items()})
        assert wrapped._precast_weight is None
    elif mutation == "layout":
        wrapped.to(memory_format=torch.channels_last)
        assert wrapped._precast_weight is None
    elif mutation == "cpu_offload":
        wrapped.cpu()
        assert wrapped._precast_weight is None
    elif mutation == "dtype":
        wrapped.double()
        assert wrapped._precast_weight is None
        wrapped.float()
    refreshed = warm(wrapped)
    assert refreshed is not cached
    assert torch.equal(refreshed, wrapped.weight.to(torch.bfloat16))
    values = torch.ones(1, 4, 6, 6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = wrapped._conv_forward(values, wrapped.weight, wrapped.bias)
        actual = wrapped(values)
    assert torch.equal(expected, actual)


def test_autocast_dtype_switch_refreshes_and_unfreeze_falls_back():
    wrapped = FrozenPrecastConv2d(source_conv())
    old = warm(wrapped)
    with torch.autocast("cpu", dtype=torch.float16):
        wrapped(torch.ones(1, 4, 6, 6))
    assert wrapped._precast_weight.dtype == torch.float16
    assert wrapped._precast_weight is not old
    wrapped.requires_grad_(True)
    x = torch.ones(1, 4, 6, 6, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        wrapped(x).sum().backward()
    assert wrapped._precast_weight is None
    assert wrapped.weight.grad is not None and wrapped.bias.grad is not None
    assert x.grad is not None


def test_inference_warmup_remains_usable_for_later_input_backward():
    original = source_conv()
    wrapped = FrozenPrecastConv2d(original)
    with torch.inference_mode():
        cached = warm(wrapped)
    assert not torch.is_inference(cached)
    values = torch.linspace(-1, 1, 2 * 4 * 7 * 7).reshape(2, 4, 7, 7)
    expected, actual = forward_backward(original, values), forward_backward(wrapped, values)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert wrapped._precast_weight is cached


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Root-authorized GPU allocation required"))])
def test_frozen_saved_weight_storage_is_shared_across_live_graphs(device):
    original = source_conv(bias=False, device=device)
    wrapped = FrozenPrecastConv2d(original)
    values = torch.linspace(-1, 1, 2 * 4 * 7 * 7, device=device).reshape(2, 4, 7, 7)

    def collect(module):
        saved_weights, xs, ys = [], [], []

        def pack(tensor):
            if tensor.shape == module.weight.shape and tensor.dtype == torch.bfloat16:
                saved_weights.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            for index in range(3):
                x = (values + index * .1).detach().requires_grad_(True)
                with torch.autocast(device, dtype=torch.bfloat16):
                    ys.append(module(x))
                xs.append(x)
            loss = sum(y.float().square().sum() for y in ys)
        gradients = torch.autograd.grad(loss, xs)
        assert len(saved_weights) == 3
        return ys, gradients, saved_weights

    reference, actual = collect(original), collect(wrapped)
    assert all(torch.equal(a, b) for a, b in zip(reference[0], actual[0]))
    assert all(torch.equal(a, b) for a, b in zip(reference[1], actual[1]))
    assert len({weight.data_ptr() for weight in reference[2]}) == 3
    assert len({weight.data_ptr() for weight in actual[2]}) == 1
    assert actual[2][0].data_ptr() == wrapped._precast_weight.data_ptr()


def test_non_reentrant_rematerialization_and_mutation_with_live_graphs():
    from torch.utils.checkpoint import checkpoint

    original = source_conv()
    wrapped = FrozenPrecastConv2d(copy.deepcopy(original))
    values = torch.linspace(-1, 1, 2 * 4 * 7 * 7).reshape(2, 4, 7, 7)

    def run(module, use_remat):
        x = values.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            output = checkpoint(module, x, use_reentrant=False) if use_remat else module(x)
        gradient, = torch.autograd.grad(output.float().square().sum(), x)
        return output, gradient

    baseline = run(original, False)
    for use_remat in (False, True):
        assert all(torch.equal(a, b) for a, b in zip(baseline, run(wrapped, use_remat)))

    def mutate_between_forward(module):
        inputs, outputs = [], []
        for _ in range(2):
            x = values.clone().requires_grad_(True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs.append(module(x))
            inputs.append(x)
            with torch.no_grad():
                module.weight.add_(.25)
        gradients = torch.autograd.grad(sum(value.float().square().sum() for value in outputs), inputs)
        return outputs, gradients

    expected, actual = mutate_between_forward(original), mutate_between_forward(wrapped)
    for expected_tensors, actual_tensors in zip(expected, actual):
        assert all(torch.equal(a, b) for a, b in zip(expected_tensors, actual_tensors))


@pytest.mark.parametrize("canonical_dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_non_fp32_canonical_parameters_use_native_path(canonical_dtype):
    original = source_conv(dtype=canonical_dtype)
    wrapped = FrozenPrecastConv2d(original)
    values = torch.ones(1, 4, 6, 6, dtype=canonical_dtype)
    expected, actual = forward_backward(original, values), forward_backward(wrapped, values)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert wrapped._precast_weight is None


def test_warmed_native_frozen_convs_recast_but_wrapper_does_not():
    original = source_conv()
    wrapped = FrozenPrecastConv2d(original)
    values = torch.ones(1, 4, 6, 6, dtype=torch.bfloat16)
    warm(wrapped)

    def casts(module):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                for _ in range(3):
                    module(values)
        return sum(event.count for event in profile.key_averages() if event.key == "aten::_to_copy")

    assert casts(original) == 6  # Separate FP32 weight + bias casts on each call.
    assert casts(wrapped) == 0


def test_deepcopy_clears_derived_buffers_and_state_dict_stays_canonical():
    wrapped = FrozenPrecastConv2d(source_conv())
    warm(wrapped)
    before = state_hash(wrapped)
    duplicated = copy.deepcopy(wrapped)
    assert duplicated._precast_weight is None and duplicated._precast_key is None
    assert state_hash(duplicated) == before
    assert duplicated.weight is not wrapped.weight
    assert set(duplicated.state_dict()) == {"weight", "bias"}
    assert warm(duplicated).data_ptr() != wrapped._precast_weight.data_ptr()


def test_reject_trainable_subclass_or_hooked_sources():
    with pytest.raises(ValueError, match="Freeze"):
        FrozenPrecastConv2d(nn.Conv2d(4, 8, 3))

    class CustomConv(nn.Conv2d):
        pass

    with pytest.raises(TypeError, match="exact"):
        FrozenPrecastConv2d(CustomConv(4, 8, 3).requires_grad_(False))
    hooked = source_conv()
    hooked.register_forward_hook(lambda module, inputs, outputs: outputs)
    with pytest.raises(ValueError, match="hooks"):
        FrozenPrecastConv2d(hooked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Root-authorized GPU allocation required")
@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("channels_last", [False, True])
def test_cuda_exact_output_input_grad_offload_and_inference_warmup(autocast_dtype, channels_last):
    original = source_conv(device="cuda")
    if channels_last:
        original.to(memory_format=torch.channels_last)
    wrapped = FrozenPrecastConv2d(original)
    values = torch.linspace(-1, 1, 2 * 4 * 7 * 7, device="cuda").reshape(2, 4, 7, 7)
    if channels_last:
        values = values.contiguous(memory_format=torch.channels_last)
    rng_before = torch.cuda.get_rng_state().clone()
    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        wrapped(values)
    cached = wrapped._precast_weight
    expected = forward_backward(original, values, autocast_dtype=autocast_dtype)
    actual = forward_backward(wrapped, values, autocast_dtype=autocast_dtype)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert wrapped._precast_weight is cached and not torch.is_inference(cached)
    assert torch.equal(rng_before, torch.cuda.get_rng_state())
    before = state_hash(wrapped)
    wrapped.cpu()
    assert wrapped._precast_weight is None and wrapped.weight.device.type == "cpu"
    wrapped.cuda()
    assert state_hash(wrapped) == before
    actual = forward_backward(wrapped, values, autocast_dtype=autocast_dtype)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
