"""Optional exact autocast-weight reuse for frozen, canonical Conv2d modules.

PyTorch 2.8's autocast ``cached_cast`` only caches FP32 weights when
``requires_grad`` is true. Frozen convolutions therefore recast their weights
on every call, and input-gradient graphs retain each separate low-precision
copy. See the version-pinned implementation:
https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/autocast_mode.cpp#L110-L133

This wrapper preserves the original FP32 Parameter objects and checkpoint
keys. Only nonpersistent BF16/FP16 buffers are cached; native autocast still
handles the input and convolution dispatch. It does not change normalization,
padding, memory format, autocast precision, or parameter requires_grad flags.
As with ordinary version-based caches, untracked ``parameter.data.copy_``
mutations require an explicit ``clear_precast_cache()`` call.
"""
from __future__ import annotations

import torch
from torch import nn


class FrozenPrecastConv2d(nn.Conv2d):
    """Wrap an exact, already-frozen ``nn.Conv2d`` without reinitialization.

    Device/dtype/layout moves and state loads discard the cache. Normal
    in-place updates and Parameter replacement invalidate it lazily. Becoming
    trainable, leaving autocast, and non-FP32 canonical parameters use the
    unchanged native Conv2d path. CPU offload discards derived GPU buffers.
    """

    def __init__(self, source: nn.Conv2d):
        if type(source) is not nn.Conv2d:
            raise TypeError("FrozenPrecastConv2d requires an exact nn.Conv2d, not a subclass")
        if any(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError("Freeze Conv2d parameters before installing the precast wrapper")
        if (source._forward_hooks or source._forward_pre_hooks or source._backward_hooks
                or source._backward_pre_hooks or source._state_dict_hooks
                or source._state_dict_pre_hooks or source._load_state_dict_pre_hooks
                or source._load_state_dict_post_hooks):
            raise ValueError("Install the precast wrapper before attaching module hooks")
        nn.Module.__init__(self)
        for name in ("in_channels", "out_channels", "kernel_size", "stride", "padding",
                     "dilation", "transposed", "output_padding", "groups", "padding_mode",
                     "_reversed_padding_repeated_twice"):
            setattr(self, name, getattr(source, name))
        self.weight = source.weight
        self.bias = source.bias
        self.training = source.training
        self.register_buffer("_precast_weight", None, persistent=False)
        self.register_buffer("_precast_bias", None, persistent=False)
        self._precast_key = None

    def clear_precast_cache(self):
        self._precast_weight = None
        self._precast_bias = None
        self._precast_key = None

    def _apply(self, fn, recurse=True):
        # Do not transport/recast derived buffers when offloading canonical
        # weights or switching their dtype/layout. Rebuild on the next use.
        self.clear_precast_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_precast_cache()
        try:
            return super()._load_from_state_dict(*args, **kwargs)
        finally:
            self.clear_precast_cache()

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["_buffers"] = state["_buffers"].copy()
        state["_buffers"]["_precast_weight"] = None
        state["_buffers"]["_precast_bias"] = None
        state["_precast_key"] = None
        return state

    @staticmethod
    def _parameter_key(parameter):
        if parameter is None:
            return None
        if torch.is_inference(parameter):
            # Inference tensors have no mutation version counter: do not cache.
            return None
        return (id(parameter), parameter._version, parameter.data_ptr(),
                parameter.device, parameter.dtype, tuple(parameter.shape), tuple(parameter.stride()))

    def _autocast_parameters(self, input):
        device_type = input.device.type
        if device_type not in ("cpu", "cuda") or not torch.is_autocast_enabled(device_type):
            return self.weight, self.bias
        target_dtype = torch.get_autocast_dtype(device_type)
        parameters = (self.weight,) if self.bias is None else (self.weight, self.bias)
        if (target_dtype not in (torch.bfloat16, torch.float16)
                or any(parameter.requires_grad or parameter.dtype != torch.float32
                       or parameter.device != input.device or torch.is_inference(parameter)
                       for parameter in parameters)):
            self.clear_precast_cache()
            return self.weight, self.bias
        # A stream-specific key avoids reusing an asynchronously created cast
        # on another stream before its producer finishes; no synchronization is
        # introduced on the normal single-stream training path.
        stream = torch.cuda.current_stream(input.device).cuda_stream if device_type == "cuda" else None
        key = (target_dtype, stream, self._parameter_key(self.weight), self._parameter_key(self.bias))
        if key != self._precast_key:
            # Real-data extraction can warm this cache under inference_mode.
            # Create normal tensors so later generated-data backward may save
            # the same frozen weights and differentiate with respect to input.
            with torch.inference_mode(False), torch.no_grad():
                weight = self.weight.detach().to(dtype=target_dtype)
                bias = None if self.bias is None else self.bias.detach().to(dtype=target_dtype)
            self._precast_weight = weight
            self._precast_bias = bias
            self._precast_key = key
        return self._precast_weight, self._precast_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._autocast_parameters(input)
        # _conv_forward retains native nonzero padding-mode behavior. Leaving
        # autocast enabled preserves its exact input-cast autograd boundary.
        return self._conv_forward(input, weight, bias)
