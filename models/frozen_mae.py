"""Frozen MAE encoder runtime; canonical feature statistics remain unchanged.

The complete pretrained MAE is validated before constructing this object. Only
its encoder is registered here: reconstruction/classification weights never
occupy GPU memory. Runtime options default to the original, unchunked encoder.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.mae_resnet import MAEResNet, _ResNetEncoder, _DtypePreservingGroupNorm


class _CanonicalLayoutGroupNorm(_DtypePreservingGroupNorm):
    """Keep canonical NCHW GroupNorm kernels inside an optional CL encoder."""
    def __init__(self, source: _DtypePreservingGroupNorm):
        nn.Module.__init__(self)
        self.num_groups, self.num_channels = source.num_groups, source.num_channels
        self.eps, self.affine = source.eps, source.affine
        self.weight, self.bias = source.weight, source.bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # No dtype/epsilon/math changes: call the canonical implementation on
        # exactly the memory layout used by the unoptimized MAE reference.
        return super().forward(input.contiguous(memory_format=torch.contiguous_format))


def _preserve_groupnorm_layout(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, _DtypePreservingGroupNorm):
            setattr(module, name, _CanonicalLayoutGroupNorm(child))
        else:
            _preserve_groupnorm_layout(child)


def _install_precast_convolutions(module: nn.Module) -> None:
    from models.frozen_conv import FrozenPrecastConv2d

    for name, child in list(module.named_children()):
        if type(child) is nn.Conv2d:
            setattr(module, name, FrozenPrecastConv2d(child))
        else:
            _install_precast_convolutions(child)


class _FrozenEncoderRuntime(_ResNetEncoder):
    def __init__(self, encoder: _ResNetEncoder, cfg: dict):
        nn.Module.__init__(self)
        # Keep canonical checkpoint key names, not encoder.backbone.* aliases.
        self.conv1 = encoder.conv1
        self.gn1 = encoder.gn1
        for index in range(1, 5):
            setattr(self, f"layer{index}_norm", getattr(encoder, f"layer{index}_norm"))
        self.stages = encoder.stages
        common = int(cfg.get("feature_microbatch_size", 0))
        self.real_microbatch_size = int(
            cfg["feature_real_microbatch_size"]
            if cfg.get("feature_real_microbatch_size") is not None else common
        )
        self.generated_microbatch_size = int(
            cfg["feature_generated_microbatch_size"]
            if cfg.get("feature_generated_microbatch_size") is not None else common
        )
        if min(common, self.real_microbatch_size, self.generated_microbatch_size) < 0:
            raise ValueError("MAE feature microbatch sizes must be nonnegative")
        self.channels_last = bool(cfg.get("feature_channels_last", False))
        self.channels_last_preserve_groupnorm_layout = bool(
            cfg.get("feature_channels_last_preserve_groupnorm_layout", False)
        )
        if self.channels_last_preserve_groupnorm_layout and not self.channels_last:
            raise ValueError("Preserving CL GroupNorm layout requires feature_channels_last=true")
        self.compile_backbone = bool(cfg.get("feature_compile_backbone", False))
        self.precast_conv_weights = bool(cfg.get("feature_precast_conv_weights", False))
        if self.precast_conv_weights and self.compile_backbone:
            raise ValueError("Precast frozen conv weights require feature_compile_backbone=false")
        self.compile_backend = str(cfg.get("feature_compile_backend", "inductor"))
        self.compile_fullgraph = bool(cfg.get("feature_compile_fullgraph", True))
        self.compile_dynamic = bool(cfg.get("feature_compile_dynamic", False))
        self.compile_emulate_precision_casts = bool(
            cfg.get("feature_compile_emulate_precision_casts", True)
        )
        self.compile_layout_optimization = (
            bool(cfg["feature_compile_layout_optimization"])
            if cfg.get("feature_compile_layout_optimization") is not None else None
        )
        stages = cfg.get("feature_remat_stages")
        if stages is not None and not isinstance(stages, (list, tuple)):
            raise ValueError("feature_remat_stages must be a list of stage names")
        valid_stages = tuple(f"stage{index}" for index in range(1, 5))
        requested = None if stages is None else set(stages)
        if requested is not None and requested.difference(valid_stages):
            raise ValueError(f"Unknown MAE rematerialization stages: {sorted(requested.difference(valid_stages))}")
        self.remat_stages = (
            None if requested is None else tuple(stage for stage in valid_stages if stage in requested)
        )
        self._compiled_encoder_forward = None
        if self.channels_last_preserve_groupnorm_layout:
            _preserve_groupnorm_layout(self)
        self.eval().requires_grad_(False)
        if self.channels_last:
            self.to(memory_format=torch.channels_last)
        if self.precast_conv_weights:
            _install_precast_convolutions(self)

    def __getstate__(self):
        state = super().__getstate__()
        # A compiled bound method must not retain another model after deepcopy.
        state["_compiled_encoder_forward"] = None
        return state

    def _forward_reference(self, x, return_block_outputs, use_remat,
                           capture_stages, block_output_stages):
        if self.remat_stages is not None and use_remat:
            return self._forward_selective_remat(
                x, return_block_outputs, capture_stages, block_output_stages,
            )
        return _ResNetEncoder.forward(
            self, x, return_block_outputs=return_block_outputs,
            use_remat=use_remat, capture_stages=capture_stages,
            block_output_stages=block_output_stages,
        )

    def _forward_selective_remat(self, x, return_block_outputs,
                                 capture_stages, block_output_stages):
        # Identical to _ResNetEncoder.forward except for the checkpoint policy.
        # Keep every convolution/GN/ReLU, capture key, and block output order.
        all_stages = tuple(f"stage{index}" for index in range(1, 5))
        captured = set(all_stages if capture_stages is None else capture_stages)
        captured_blocks = set(captured if block_output_stages is None else block_output_stages)
        features, block_outputs = {}, {}
        x = F.relu(self.gn1(self.conv1(x)))
        if "stage1" in captured:
            features["conv1"] = x
        for index, stage in enumerate(self.stages):
            layer_name, stage_name = f"layer{index + 1}", f"stage{index + 1}"
            outputs = [] if return_block_outputs and stage_name in captured_blocks else None
            for block in stage:
                if (stage_name in self.remat_stages and torch.is_grad_enabled()
                        and x.requires_grad):
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
                if outputs is not None:
                    outputs.append(x)
            if outputs is not None:
                block_outputs[layer_name] = outputs
            x = getattr(self, f"{layer_name}_norm")(x)
            if stage_name in captured:
                features[layer_name] = x
        return (features, block_outputs) if return_block_outputs else features

    def _forward_chunk(self, x, return_block_outputs, use_remat,
                       capture_stages, block_output_stages):
        forward = self._forward_reference
        if self.compile_backbone:
            if self._compiled_encoder_forward is None:
                options = (
                    {"triton.cudagraphs": False,
                     "emulate_precision_casts": self.compile_emulate_precision_casts}
                    if self.compile_backend == "inductor" else None
                )
                if options is not None and self.compile_layout_optimization is not None:
                    options["layout_optimization"] = self.compile_layout_optimization
                self._compiled_encoder_forward = torch.compile(
                    self._forward_reference, backend=self.compile_backend,
                    fullgraph=self.compile_fullgraph, dynamic=self.compile_dynamic,
                    options=options,
                )
            forward = self._compiled_encoder_forward
        return forward(x, return_block_outputs, use_remat,
                       capture_stages, block_output_stages)

    def forward(
        self, x: torch.Tensor, return_block_outputs: bool = False,
        use_remat: bool = False, capture_stages: Optional[Tuple[str, ...]] = None,
        block_output_stages: Optional[Tuple[str, ...]] = None,
    ):
        generated = torch.is_grad_enabled() and x.requires_grad
        microbatch = self.generated_microbatch_size if generated else self.real_microbatch_size
        chunk_size = microbatch if microbatch > 0 else int(x.shape[0])
        if chunk_size < 1:
            raise ValueError("MAE feature input batch must not be empty")
        feature_parts, block_parts = {}, {}
        for start in range(0, int(x.shape[0]), chunk_size):
            chunk = x[start:start + chunk_size]
            if self.channels_last:
                chunk = chunk.contiguous(memory_format=torch.channels_last)
            result = self._forward_chunk(
                chunk, return_block_outputs, use_remat,
                capture_stages, block_output_stages,
            )
            if return_block_outputs:
                features, blocks = result
            else:
                features, blocks = result, {}
            for name, value in features.items():
                feature_parts.setdefault(name, []).append(value)
            for name, values in blocks.items():
                block_parts.setdefault(name, []).append(values)

        def concatenate(parts):
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        features = {name: concatenate(parts) for name, parts in feature_parts.items()}
        if not return_block_outputs:
            return features
        blocks = {
            name: [concatenate([chunk[index] for chunk in parts])
                   for index in range(len(parts[0]))]
            for name, parts in block_parts.items()
        }
        return features, blocks


class FrozenMAEFeatureExtractor(MAEResNet):
    """Encoder-only view of an already strictly validated pretrained MAE.

    Inherited get_activations retains the original input patching, BF16 casts,
    terminal GroupNorm, feature names, and feature-statistic arithmetic.
    """
    def __init__(self, validated_mae: MAEResNet, cfg: dict):
        nn.Module.__init__(self)
        for name in ("num_classes", "in_channels", "base_channels", "patch_size",
                     "use_bf16", "input_patch_size", "use_remat", "fuse_stats"):
            setattr(self, name, getattr(validated_mae, name))
        self.input_proj = validated_mae.input_proj
        self.encoder = _FrozenEncoderRuntime(validated_mae.encoder, cfg)
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        # A metric encoder is never trainable, including dropout behavior.
        return super().train(False)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Frozen MAE is feature-only; use get_activations")


__all__ = ["FrozenMAEFeatureExtractor"]
