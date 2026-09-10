"""Small adapters for frozen MAE/DINO/MoCo feature maps.

The online adapter can be trained either from real-only supervised contrastive
examples or from detached generated queries against class-matched real keys.
A separate frozen copy remains available for experiments that want an EMA
metric in the generator path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from models.ssl_resnet import derive_ssl_map_features


_VALID_STAGES = ("stage1", "stage2", "stage3", "stage4")


def _feature_stage_for_name(name: str) -> Optional[str]:
    """Map an exported SSL feature key to its residual stage."""
    for index in range(1, 5):
        prefix = f"layer{index}"
        if name == prefix or name.startswith(prefix + "_"):
            return f"stage{index}"
    return None


def _gather_with_gradient(values: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Concatenate equal-size rank-local statistics without cutting autograd."""
    if (
        not bool(enabled)
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return values
    return torch.cat(tuple(dist_nn.all_gather(values.contiguous())), dim=0)


def _raw_fields_fp32(function, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
    """Keep raw-field bmm/softmax arithmetic FP32 under outer AMP contexts."""
    if not args or not isinstance(args[0], torch.Tensor):
        raise ValueError("raw-field helper requires the query tensor first")
    with torch.autocast(device_type=args[0].device.type, enabled=False):
        return function(*args, **kwargs)


def _class_field_energy(field: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return one raw-field mean-square energy per sampled class."""
    if field.ndim != 3:
        raise ValueError(f"raw drift field must be [B*T,Q,D], got {field.shape}")
    batch_size = int(batch_size)
    if batch_size <= 0 or field.shape[0] % batch_size:
        raise ValueError(
            f"raw field leading dimension {field.shape[0]} is not divisible by "
            f"batch_size={batch_size}"
        )
    token_count = field.shape[0] // batch_size
    return (
        field.float()
        .reshape(batch_size, token_count, field.shape[1], field.shape[2])
        .square()
        .mean(dim=(1, 2, 3))
    )


def _field_vector_energy(field: torch.Tensor) -> torch.Tensor:
    """Return one feature-normalized energy for every raw field vector."""
    if field.ndim != 3:
        raise ValueError(f"raw drift field must be [B*T,Q,D], got {field.shape}")
    if field.shape[-1] <= 0:
        raise ValueError("raw drift field feature dimension must be non-empty")
    return field.float().square().mean(dim=-1).reshape(-1)


def drift_field_snr_v2_statistic(
    signal_field_a: torch.Tensor,
    signal_field_b: torch.Tensor,
    real_null_field: torch.Tensor,
    generated_null_field: torch.Tensor,
    *,
    epsilon: float,
    global_statistics: bool,
) -> Dict[str, torch.Tensor]:
    """Version-2 signal-to-null statistic over individual field energies.

    The two cross-distribution fields estimate ``D_pq`` from independent
    support banks.  RR and QQ energies share the explicitly defined ``D0``
    center and use one pooled unbiased variance.  Nothing in the null branch
    is detached, so the adapter can reduce both null energy and estimator
    variance.  Production Version 2 uses rank-local statistics and lets DDP
    average the resulting adapter gradients; ``global_statistics`` remains an
    explicit test/ablation switch.
    """
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("feature-adapter Version-2 SNR epsilon must be positive")

    energies = [
        _gather_with_gradient(_field_vector_energy(field), global_statistics)
        for field in (
            signal_field_a,
            signal_field_b,
            real_null_field,
            generated_null_field,
        )
    ]
    energy_a, energy_b, energy_rr, energy_qq = energies
    D_a = energy_a.mean()
    D_b = energy_b.mean()
    Drr = energy_rr.mean()
    Dqq = energy_qq.mean()
    Dpq = 0.5 * (D_a + D_b)
    D0 = 0.5 * (Drr + Dqq)
    null_count = int(energy_rr.numel() + energy_qq.numel())
    denominator = max(null_count - 1, 1)
    Var0 = (
        (energy_rr - D0).square().sum()
        + (energy_qq - D0).square().sum()
    ) / denominator
    J = (Dpq - D0) / (Var0 + epsilon).sqrt()
    return {
        "D_a": D_a,
        "D_b": D_b,
        "Dpq": Dpq,
        "Drr": Drr,
        "Dqq": Dqq,
        "D0": D0,
        "Var0": Var0,
        "J": J,
    }


def _reshape_feature_bank(
    feature: torch.Tensor,
    *,
    batch_size: int,
    sample_count: int,
) -> torch.Tensor:
    """Convert [B*C,T,D] activations to drift's [B*T,C,D] layout."""
    if feature.ndim != 3:
        raise ValueError(f"drift feature must be [B*C,T,D], got {feature.shape}")
    expected = int(batch_size) * int(sample_count)
    if feature.shape[0] != expected:
        raise ValueError(
            f"drift feature has {feature.shape[0]} samples, expected {expected} "
            f"for B={batch_size}, C={sample_count}"
        )
    tokens, channels = feature.shape[1:]
    return (
        feature.reshape(batch_size, sample_count, tokens, channels)
        .permute(0, 2, 1, 3)
        .reshape(batch_size * tokens, sample_count, channels)
    )


def _select_class_major_samples(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    sample_count: int,
) -> torch.Tensor:
    """Select class-specific rows from a flattened ``[B*C,...]`` tensor."""
    batch_size = int(batch_size)
    sample_count = int(sample_count)
    if values.shape[0] != batch_size * sample_count:
        raise ValueError(
            f"class-major tensor has {values.shape[0]} rows, expected "
            f"{batch_size * sample_count}"
        )
    if indices.ndim != 2 or indices.shape[0] != batch_size:
        raise ValueError(
            f"class-major indices must be [B,K], got {tuple(indices.shape)}"
        )
    offsets = torch.arange(batch_size, device=indices.device)[:, None] * sample_count
    flat_indices = (indices + offsets).reshape(-1)
    return values.index_select(0, flat_indices)


def _select_feature_bank(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Select per-class supports from drift-layout ``[B*T,C,D]`` features."""
    if values.ndim != 3:
        raise ValueError(f"drift feature bank must be [B*T,C,D], got {values.shape}")
    batch_size = int(batch_size)
    if batch_size <= 0 or values.shape[0] % batch_size:
        raise ValueError("drift feature leading dimension must be divisible by B")
    if indices.ndim != 2 or indices.shape[0] != batch_size:
        raise ValueError(f"feature-bank indices must be [B,K], got {indices.shape}")
    tokens = values.shape[0] // batch_size
    channels = values.shape[-1]
    index = indices[:, None, :, None].expand(
        batch_size, tokens, indices.shape[1], channels
    )
    selected = torch.gather(
        values.reshape(batch_size, tokens, values.shape[1], channels),
        dim=2,
        index=index,
    )
    return selected.reshape(batch_size * tokens, indices.shape[1], channels)


def deterministic_class_partition_indices(
    total: int,
    sizes: Sequence[int],
    *,
    labels: torch.Tensor,
    split_index: int,
    seed: int,
    salt: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Make step-dependent, class-specific disjoint permutations privately."""
    total = int(total)
    normalized_sizes = tuple(int(size) for size in sizes)
    if total <= 0 or any(size <= 0 for size in normalized_sizes):
        raise ValueError("partition total and group sizes must be positive")
    if sum(normalized_sizes) > total:
        raise ValueError(
            f"partition sizes {normalized_sizes} exceed total={total}"
        )
    labels_cpu = labels.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    rows = []
    modulus = 2**63 - 1
    for row_index, label in enumerate(labels_cpu.tolist()):
        generator = torch.Generator(device="cpu")
        mixed_seed = (
            int(seed)
            + 1_000_003 * int(split_index)
            + 97_409 * int(label)
            + 65_537 * int(row_index)
            + int(salt)
        ) % modulus
        generator.manual_seed(mixed_seed)
        rows.append(torch.randperm(total, generator=generator))
    order = torch.stack(rows, dim=0).to(device=device)
    result = []
    offset = 0
    for size in normalized_sizes:
        result.append(order[:, offset : offset + size])
        offset += size
    return tuple(result)


def drift_field_snr_statistic(
    signal_field: torch.Tensor,
    real_null_field: torch.Tensor,
    generated_null_field: torch.Tensor,
    *,
    batch_size: int,
    epsilon: float,
    global_statistics: bool,
    stopgrad_variance: bool,
) -> Dict[str, torch.Tensor]:
    """Compute one feature/temperature null-corrected raw-field statistic.

    Variance is estimated across class-level energies, not across spatial
    coordinates (which would falsely treat correlated tokens as independent
    Monte-Carlo samples).  By default the denominator remains differentiable,
    matching the stated objective exactly; the optional stop-gradient variant
    is an explicit stability ablation.
    """
    epsilon = float(epsilon)
    if not epsilon > 0.0:
        raise ValueError("feature-adapter SNR epsilon must be positive")
    energy_pq = _gather_with_gradient(
        _class_field_energy(signal_field, batch_size), global_statistics
    )
    energy_pp = _gather_with_gradient(
        _class_field_energy(real_null_field, batch_size), global_statistics
    )
    energy_qq = _gather_with_gradient(
        _class_field_energy(generated_null_field, batch_size), global_statistics
    )

    Dpq = energy_pq.mean()
    Dpp = energy_pp.mean()
    Dqq = energy_qq.mean()
    D0 = 0.5 * (Dpp + Dqq)
    correction = 1 if energy_pp.numel() > 1 else 0
    var_pp = energy_pp.var(correction=correction)
    correction = 1 if energy_qq.numel() > 1 else 0
    var_qq = energy_qq.var(correction=correction)
    var0 = 0.5 * (var_pp + var_qq)
    denominator_variance = var0.detach() if bool(stopgrad_variance) else var0
    J = (Dpq - D0) / (denominator_variance + epsilon).sqrt()
    return {
        "Dpq": Dpq,
        "Dpp": Dpp,
        "Dqq": Dqq,
        "D0": D0,
        "Var0": var0,
        "J": J,
    }


def drift_direction_consistency(
    online_field: torch.Tensor,
    target_field: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Per-query/channel direction agreement with a stopped EMA target."""
    if online_field.shape != target_field.shape:
        raise ValueError(
            "online/target raw fields must have identical shapes, got "
            f"{online_field.shape} and {target_field.shape}"
        )
    return 1.0 - F.cosine_similarity(
        online_field.float(),
        target_field.detach().float(),
        dim=-1,
        eps=float(epsilon),
    ).mean()


def canonical_adapter_stages(keys: Iterable[str]) -> Tuple[str, ...]:
    """Normalize ``layer3``/``stage3`` spellings and preserve stage order."""
    selected = set()
    for raw in keys:
        key = str(raw).strip().lower()
        if key.startswith("layer"):
            key = f"stage{key[5:]}"
        if key not in _VALID_STAGES:
            raise ValueError(
                f"Unknown feature-adapter key {raw!r}; expected layer1..4 or stage1..4"
            )
        selected.add(key)
    return tuple(stage for stage in _VALID_STAGES if stage in selected)


class ResidualSpatialAdapter(nn.Module):
    """Identity-initialized 1x1 bottleneck adapter for a BCHW feature map."""

    def __init__(self, channels: int, bottleneck: int, dropout: float = 0.0):
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("adapter channels and bottleneck must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("adapter dropout must be in [0, 1)")
        self.norm = nn.GroupNorm(1, channels, eps=1e-6)
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.dropout = (
            nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(F.silu(self.down(self.norm(x)))))
        return x + residual


class ResidualPatchPredictor(nn.Module):
    """Identity-initialized spatial predictor for continuous DINO tokens.

    DINO-R50 exports spatial maps rather than ViT patch tokens.  The
    bottleneck depthwise convolution gives a masked location access to nearby
    context without introducing a large dense 3x3 convolution over the
    1024/2048-channel backbone maps.  The predictor is training-only; the
    generator metric consumes :class:`ResidualSpatialAdapter` outputs and never
    this module.
    """

    def __init__(
        self,
        channels: int,
        bottleneck: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("patch predictor channels and bottleneck must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("patch predictor dropout must be in [0, 1)")
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels, eps=1.0e-6)
        self.down = nn.Conv2d(self.channels, int(bottleneck), 1)
        self.context = nn.Conv2d(
            int(bottleneck),
            int(bottleneck),
            kernel_size=3,
            padding=1,
            groups=int(bottleneck),
        )
        self.dropout = (
            nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        self.up = nn.Conv2d(int(bottleneck), self.channels, 1)
        # The direct path is per-token LayerNorm, so the masked objective starts
        # as continuous DINO feature regression rather than a random metric.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"patch predictor expects [N,{self.channels},H,W], got {tuple(x.shape)}"
            )
        tokens = x.permute(0, 2, 3, 1)
        normalized = self.norm(tokens.float()).permute(0, 3, 1, 2)
        hidden = F.silu(self.down(normalized))
        hidden = F.silu(self.context(hidden))
        residual = self.up(self.dropout(hidden)).float()
        return normalized + residual


def make_dino_patch_masked_images(
    images: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    seed: int,
    fill_values: Sequence[float] = (-0.03, -0.088, -0.188),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask an exact number of raw-image patches without touching global RNG.

    ``images`` are the raw ImageNet tensors in ``[-1, 1]``.  The default fill
    is ``2 * ImageNetMean - 1``, which becomes exactly zero after the frozen
    DINO extractor's ImageNet normalization.  A private CPU generator makes
    masks deterministic from ``seed`` while leaving both CPU and CUDA global
    RNG streams unchanged.

    Returns the masked images and a boolean ``[N,grid_h,grid_w]`` patch mask.
    """
    if images.ndim != 4:
        raise ValueError(f"masked DINO images must be NCHW, got {tuple(images.shape)}")
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError("masked DINO patch_size must be positive")
    height, width = (int(images.shape[-2]), int(images.shape[-1]))
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image size {height}x{width} is not divisible by patch_size={patch_size}"
        )
    mask_ratio = float(mask_ratio)
    if not 0.0 < mask_ratio < 1.0 or not math.isfinite(mask_ratio):
        raise ValueError("masked DINO mask_ratio must be finite and in (0, 1)")
    if len(fill_values) != int(images.shape[1]):
        raise ValueError(
            f"masked DINO fill_values has {len(fill_values)} channels, "
            f"expected {images.shape[1]}"
        )

    grid_h, grid_w = height // patch_size, width // patch_size
    patch_count = grid_h * grid_w
    if patch_count < 2:
        raise ValueError("masked DINO needs at least two image patches")
    masked_count = min(
        patch_count - 1,
        max(1, int(mask_ratio * patch_count)),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    scores = torch.rand(
        int(images.shape[0]), patch_count, generator=generator, device="cpu"
    )
    selected = scores.topk(masked_count, dim=1, largest=False).indices
    patch_mask_cpu = torch.zeros(
        int(images.shape[0]), patch_count, dtype=torch.bool, device="cpu"
    )
    patch_mask_cpu.scatter_(1, selected, True)
    patch_mask = patch_mask_cpu.reshape(-1, grid_h, grid_w).to(images.device)
    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )
    fill = torch.as_tensor(
        tuple(float(value) for value in fill_values),
        device=images.device,
        dtype=images.dtype,
    ).view(1, -1, 1, 1)
    masked_images = torch.where(pixel_mask[:, None], fill, images)
    return masked_images, patch_mask


def project_patch_mask_to_feature_grid(
    patch_mask: torch.Tensor,
    *,
    output_size: Tuple[int, int],
    mask_ratio: float,
) -> torch.Tensor:
    """Project an image-patch mask to a coarser map with an exact budget.

    Max-pooling an 8x8 mask into a 4x4 map turns a 40% image mask into an
    approximately 87% feature mask (a coarse cell is selected when *any* of
    its four image patches is masked). Instead, rank coarse cells by masked
    area and retain exactly ``floor(mask_ratio * H * W)`` cells. Stable
    sorting makes coverage ties deterministic without drawing from any RNG.
    """
    if patch_mask.ndim != 3:
        raise ValueError(
            f"patch mask must be [N,H,W], got {tuple(patch_mask.shape)}"
        )
    out_h, out_w = (int(output_size[0]), int(output_size[1]))
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"feature mask output size must be positive, got {output_size}")
    mask_ratio = float(mask_ratio)
    if not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0:
        raise ValueError("feature mask ratio must be finite and in (0, 1)")

    coverage = F.adaptive_avg_pool2d(
        patch_mask[:, None].float(), output_size=(out_h, out_w)
    )[:, 0]
    token_count = out_h * out_w
    masked_count = max(1, int(mask_ratio * token_count))
    if token_count > 1:
        masked_count = min(token_count - 1, masked_count)
    order = torch.argsort(
        coverage.flatten(1), dim=1, descending=True, stable=True
    )
    selected = order[:, :masked_count]
    token_mask = torch.zeros_like(coverage.flatten(1), dtype=torch.bool)
    token_mask.scatter_(1, selected, True)
    return token_mask.reshape(-1, out_h, out_w)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Local-batch supervised contrastive loss with self-pairs removed."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be rank 2, got {embeddings.shape}")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be rank 1 and match embeddings")
    if embeddings.shape[0] < 2:
        raise ValueError("supervised contrastive loss needs at least two examples")
    if not float(temperature) > 0.0:
        raise ValueError("supervised contrastive temperature must be positive")

    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.transpose(0, 1)
    logits = logits / float(temperature)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, float("-inf"))
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    positive_count = positive.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every adapter example must have a same-label positive")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(
        log_prob.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.to(log_prob.dtype)
    ).mean()


def generated_to_real_multi_positive_info_nce(
    generated_embeddings: torch.Tensor,
    real_embeddings: torch.Tensor,
    generated_labels: torch.Tensor,
    real_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Generated-query/real-key multi-positive InfoNCE.

    For each generated query, every real key with the same class is a positive
    and every labeled real key participates in the denominator.  We average
    the log probability over positives, matching the supervised-contrastive
    form

      - 1/|P(q)| sum_{p in P(q)} log exp(sim(q,p)/tau)
                                      / sum_a exp(sim(q,a)/tau).

    Inputs to this helper are embeddings rather than images/features.  The
    caller is responsible for detaching the generated feature-encoder output;
    ``FeatureAdapterSystem`` does so unconditionally for this objective.
    """
    if generated_embeddings.ndim != 2 or real_embeddings.ndim != 2:
        raise ValueError("generated and real embeddings must both be rank 2")
    if generated_embeddings.shape[1] != real_embeddings.shape[1]:
        raise ValueError("generated and real embedding dimensions must match")
    if (
        generated_labels.ndim != 1
        or generated_labels.shape[0] != generated_embeddings.shape[0]
    ):
        raise ValueError("generated_labels must match generated embeddings")
    if real_labels.ndim != 1 or real_labels.shape[0] != real_embeddings.shape[0]:
        raise ValueError("real_labels must match real embeddings")
    if generated_embeddings.shape[0] == 0 or real_embeddings.shape[0] == 0:
        raise ValueError("multi-positive InfoNCE requires generated and real examples")
    if not float(temperature) > 0.0:
        raise ValueError("multi-positive InfoNCE temperature must be positive")

    queries = F.normalize(generated_embeddings.float(), dim=-1)
    keys = F.normalize(real_embeddings.float(), dim=-1)
    logits = queries @ keys.transpose(0, 1)
    logits = logits / float(temperature)
    positive = generated_labels[:, None].eq(real_labels[None, :])
    positive_count = positive.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every generated query must have a same-label real positive")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(
        log_prob.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.to(log_prob.dtype)
    ).mean()


def real_to_real_multi_positive_info_nce(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    positive_candidate_eligible: torch.Tensor,
    self_candidate_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Real-positive InfoNCE with exact self and positive-type masking.

    Every eligible non-self candidate with the query's class is a positive.
    Every candidate except the query itself participates in the denominator.
    The real-real adapter objective marks real candidates eligible and keeps
    detached generated candidates ineligible, so generated samples are
    denominator-only negatives. ``self_candidate_indices`` maps each rank-local
    query to its matching entry in the possibly all-gathered candidate bank.
    """
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("real query and candidate embeddings must both be rank 2")
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("real query and candidate embedding dimensions must match")
    if query_labels.ndim != 1 or query_labels.shape[0] != query_embeddings.shape[0]:
        raise ValueError("query_labels must match real query embeddings")
    if (
        candidate_labels.ndim != 1
        or candidate_labels.shape[0] != candidate_embeddings.shape[0]
    ):
        raise ValueError("candidate_labels must match candidate embeddings")
    if (
        positive_candidate_eligible.ndim != 1
        or positive_candidate_eligible.shape[0] != candidate_embeddings.shape[0]
    ):
        raise ValueError(
            "positive_candidate_eligible must match candidate embeddings"
        )
    if query_embeddings.shape[0] == 0 or candidate_embeddings.shape[0] < 2:
        raise ValueError("real-to-real InfoNCE requires at least two real examples")
    if not float(temperature) > 0.0:
        raise ValueError("real-to-real InfoNCE temperature must be positive")

    self_candidate_indices = self_candidate_indices.to(
        device=query_embeddings.device, dtype=torch.long
    )
    if (
        self_candidate_indices.ndim != 1
        or self_candidate_indices.shape[0] != query_embeddings.shape[0]
    ):
        raise ValueError("self_candidate_indices must provide one index per query")
    if bool(
        (
            (self_candidate_indices < 0)
            | (self_candidate_indices >= candidate_embeddings.shape[0])
        ).any()
    ):
        raise ValueError("self_candidate_indices contains an out-of-range key index")

    queries = F.normalize(query_embeddings.float(), dim=-1)
    candidates = F.normalize(candidate_embeddings.float(), dim=-1)
    logits = queries @ candidates.transpose(0, 1)
    logits = logits / float(temperature)
    row_indices = torch.arange(logits.shape[0], device=logits.device)
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[row_indices, self_candidate_indices] = True
    logits = logits.masked_fill(self_mask, float("-inf"))
    positive = (
        query_labels[:, None].eq(candidate_labels[None, :])
        & positive_candidate_eligible.to(
            device=query_embeddings.device, dtype=torch.bool
        )[None, :]
        & ~self_mask
    )
    positive_count = positive.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every real query must have a non-self same-label positive")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(
        log_prob.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.to(log_prob.dtype)
    ).mean()


def _centered_distance_correlation(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    epsilon: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Correlation of two one-dimensional distance vectors."""
    if weights is None:
        reference = reference - reference.mean()
        candidate = candidate - candidate.mean()
        reference_power = reference.square().sum()
        candidate_power = candidate.square().sum()
        covariance = (reference * candidate).sum()
    else:
        weights = weights.to(device=reference.device, dtype=reference.dtype)
        weight_sum = weights.sum().clamp_min(float(epsilon))
        reference = reference - (weights * reference).sum() / weight_sum
        candidate = candidate - (weights * candidate).sum() / weight_sum
        reference_power = (weights * reference.square()).sum()
        candidate_power = (weights * candidate.square()).sum()
        covariance = (weights * reference * candidate).sum()
    reference_norm = reference_power.sqrt()
    candidate_norm = candidate_power.sqrt()
    correlation = covariance / (reference_norm * candidate_norm).clamp_min(
        float(epsilon)
    )
    valid = (reference_norm > float(epsilon)) & (
        candidate_norm > float(epsilon)
    )
    return torch.where(
        valid,
        correlation.clamp(min=-1.0, max=1.0),
        correlation.new_zeros(()),
    )


def _pairwise_cosine_distances(
    embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return upper-triangle cosine distances and their pair indices."""
    # The surrounding adapter step runs under BF16 autocast. Explicitly
    # disable it here: centered variances need FP32, and the pair weights below
    # must share the distance tensor's dtype on both CPU and CUDA.
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        normalized = F.normalize(embeddings.float(), dim=-1)
        pair_index = torch.triu_indices(
            normalized.shape[0],
            normalized.shape[0],
            offset=1,
            device=normalized.device,
        )
        distances = (1.0 - normalized @ normalized.transpose(0, 1))[
            pair_index[0], pair_index[1]
        ]
    return distances, pair_index


def pairwise_distance_geometry_losses(
    base_embeddings: torch.Tensor,
    adapted_embeddings: torch.Tensor,
    *,
    real_count: int,
    epsilon: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return correlation, scale, and mean losses for pairwise geometry.

    Inputs contain the same real examples followed by the same generated
    examples.  The correlation term preserves the ordering of cosine
    distances but is intentionally invariant to affine rescaling.  The scale
    and mean terms close that loophole by matching the first two moments in
    each of the RR, RG, and GG pair groups independently.

    All three groups receive equal total weight so the smaller generated-
    generated group remains meaningful.  Moment calculations stay in FP32 and
    use ``sqrt(var + epsilon**2)`` to keep collapsed finite batches stable.
    The frozen reference is stopped unconditionally; gradients can update only
    the adapter representation.
    """
    if base_embeddings.ndim != 2 or adapted_embeddings.ndim != 2:
        raise ValueError("base and adapted embeddings must both be rank 2")
    if base_embeddings.shape != adapted_embeddings.shape:
        raise ValueError(
            "base and adapted embeddings must have identical shapes, got "
            f"{base_embeddings.shape} and {adapted_embeddings.shape}"
        )
    if base_embeddings.shape[0] < 6:
        raise ValueError("distance-relation alignment needs at least six examples")
    real_count = int(real_count)
    if real_count < 2 or base_embeddings.shape[0] - real_count < 2:
        raise ValueError(
            "distance-relation alignment needs at least two real and two "
            "generated examples"
        )
    epsilon = float(epsilon)
    if not epsilon > 0.0:
        raise ValueError("distance-geometry epsilon must be positive")

    base_distances, pair_index = _pairwise_cosine_distances(
        base_embeddings.detach()
    )
    adapted_distances, _ = _pairwise_cosine_distances(adapted_embeddings)
    left_real = pair_index[0] < real_count
    right_real = pair_index[1] < real_count
    masks = (
        left_real & right_real,
        left_real & ~right_real,
        ~left_real & ~right_real,
    )
    pair_weights = sum(
        (
            mask.to(base_distances.dtype)
            / mask.sum().to(base_distances.dtype)
            for mask in masks
        ),
        torch.zeros_like(base_distances),
    )
    correlation = _centered_distance_correlation(
        base_distances,
        adapted_distances,
        epsilon=epsilon,
        weights=pair_weights,
    )
    scale_losses = []
    mean_losses = []
    epsilon_tensor = base_distances.new_tensor(epsilon)
    for mask in masks:
        base_group = base_distances[mask]
        adapted_group = adapted_distances[mask]
        base_mean = base_group.mean()
        adapted_mean = adapted_group.mean()
        base_std = (
            (base_group - base_mean).square().mean() + epsilon_tensor.square()
        ).sqrt()
        adapted_std = (
            (adapted_group - adapted_mean).square().mean()
            + epsilon_tensor.square()
        ).sqrt()
        scale_losses.append((adapted_std / base_std).log().square())
        mean_losses.append(((adapted_mean - base_mean) / base_std).square())

    return (
        1.0 - correlation,
        torch.stack(scale_losses).mean(),
        torch.stack(mean_losses).mean(),
    )


def pairwise_distance_relation_loss(
    base_embeddings: torch.Tensor,
    adapted_embeddings: torch.Tensor,
    *,
    real_count: int,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Preserve pair-distance ordering; retained as a compatible wrapper."""
    relation, _, _ = pairwise_distance_geometry_losses(
        base_embeddings,
        adapted_embeddings,
        real_count=real_count,
        epsilon=epsilon,
    )
    return relation


def pairwise_cosine_relation_mse(
    base_embeddings: torch.Tensor,
    adapted_embeddings: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Preserve the frozen encoder's pairwise cosine geometry exactly.

    This is the DINO relation anchor used by the counterfactual-field
    objective.  It differs intentionally from
    :func:`pairwise_distance_relation_loss`: no centering, correlation, or
    real/generated pair reweighting is applied.  Only off-diagonal real-real
    pairs enter the mean, and the frozen representation is always stopped.
    """
    if base_embeddings.ndim != 2 or adapted_embeddings.ndim != 2:
        raise ValueError("base and adapted relation embeddings must be rank 2")
    if base_embeddings.shape != adapted_embeddings.shape:
        raise ValueError(
            "base and adapted relation embeddings must have identical shapes, "
            f"got {base_embeddings.shape} and {adapted_embeddings.shape}"
        )
    if base_embeddings.shape[0] < 2:
        raise ValueError("pairwise cosine relation loss needs at least two samples")
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("pairwise cosine relation epsilon must be positive")

    with torch.autocast(device_type=adapted_embeddings.device.type, enabled=False):
        base = F.normalize(base_embeddings.detach().float(), dim=-1, eps=epsilon)
        adapted = F.normalize(adapted_embeddings.float(), dim=-1, eps=epsilon)
        pair_index = torch.triu_indices(
            base.shape[0], base.shape[0], offset=1, device=base.device
        )
        base_cosine = (base @ base.transpose(0, 1))[
            pair_index[0], pair_index[1]
        ]
        adapted_cosine = (adapted @ adapted.transpose(0, 1))[
            pair_index[0], pair_index[1]
        ]
        return (adapted_cosine - base_cosine).square().mean()


def counterfactual_field_consistency_loss(
    first_field: torch.Tensor,
    second_field: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Moment-normalized agreement of two raw drift-field estimates.

    Both fields remain differentiable.  This implements

      ``1 - E<V1,V2> / (sqrt(E||V1||^2 E||V2||^2) + epsilon)``

    rather than averaging per-vector cosine similarities or stopping either
    view.  Distributed aggregation and multi-feature weighting are performed
    in :class:`FeatureAdapterSystem` from the same sufficient statistics.
    """
    if first_field.shape != second_field.shape:
        raise ValueError(
            "counterfactual raw fields must have identical shapes, got "
            f"{first_field.shape} and {second_field.shape}"
        )
    if first_field.numel() == 0:
        raise ValueError("counterfactual raw fields must be non-empty")
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("counterfactual field epsilon must be positive")
    first = first_field.float()
    second = second_field.float()
    dot = (first * second).mean()
    first_power = first.square().mean()
    second_power = second.square().mean()
    return 1.0 - dot / (first_power * second_power).sqrt().add(epsilon)


def rotating_partition_indices(
    total: int,
    sizes: Sequence[int],
    rotation: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Return deterministic, disjoint, cyclically rotating index groups."""
    total = int(total)
    normalized_sizes = tuple(int(size) for size in sizes)
    if total <= 0 or any(size <= 0 for size in normalized_sizes):
        raise ValueError("partition total and group sizes must be positive")
    if sum(normalized_sizes) > total:
        raise ValueError(
            f"partition sizes {normalized_sizes} exceed total={total}"
        )
    order = torch.arange(total, device=device)
    order = torch.roll(order, shifts=-(int(rotation) % total), dims=0)
    result = []
    offset = 0
    for size in normalized_sizes:
        result.append(order[offset : offset + size])
        offset += size
    return tuple(result)


@torch.no_grad()
def _pairwise_distance_relation_diagnostics(
    base_embeddings: torch.Tensor,
    adapted_embeddings: torch.Tensor,
    *,
    real_count: int,
    epsilon: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    """Return scale and RR/RG/GG relation checks for W&B logging."""
    base_distances, pair_index = _pairwise_cosine_distances(base_embeddings)
    adapted_distances, _ = _pairwise_cosine_distances(adapted_embeddings)
    left_real = pair_index[0] < int(real_count)
    right_real = pair_index[1] < int(real_count)
    masks = {
        "rr": left_real & right_real,
        "rg": left_real & ~right_real,
        "gg": ~left_real & ~right_real,
    }
    base_std = base_distances.std(correction=0)
    adapted_std = adapted_distances.std(correction=0)
    result = {
        "distance_relation_corr": _centered_distance_correlation(
            base_distances,
            adapted_distances,
            epsilon=float(epsilon),
            weights=sum(
                (
                    mask.to(base_distances.dtype)
                    / mask.sum().to(base_distances.dtype)
                    for mask in masks.values()
                ),
                torch.zeros_like(base_distances),
            ),
        ),
        "pre_distance_mean": base_distances.mean(),
        "pre_distance_std": base_std,
        "post_distance_mean": adapted_distances.mean(),
        "post_distance_std": adapted_std,
        "distance_std_scale": adapted_std / base_std.clamp_min(float(epsilon)),
        "relation_pair_count": base_distances.new_tensor(
            float(base_distances.numel())
        ),
    }
    for name, mask in masks.items():
        base_group = base_distances[mask]
        adapted_group = adapted_distances[mask]
        base_group_mean = base_group.mean()
        adapted_group_mean = adapted_group.mean()
        base_group_std = base_group.std(correction=0)
        adapted_group_std = adapted_group.std(correction=0)
        correlation = _centered_distance_correlation(
            base_group,
            adapted_group,
            epsilon=float(epsilon),
        )
        result[f"{name}_relation_corr"] = correlation
        result[f"{name}_pair_count"] = base_distances.new_tensor(float(mask.sum()))
        result[f"{name}_pre_distance_mean"] = base_group_mean
        result[f"{name}_post_distance_mean"] = adapted_group_mean
        result[f"{name}_pre_distance_std"] = base_group_std
        result[f"{name}_post_distance_std"] = adapted_group_std
        result[f"{name}_distance_std_scale"] = (
            adapted_group_std / base_group_std.clamp_min(float(epsilon))
        )
        result[f"{name}_standardized_mean_shift"] = (
            (adapted_group_mean - base_group_mean)
            / base_group_std.clamp_min(float(epsilon))
        )
    return result


def _gather_real_candidates(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Autograd-aware all-gather for the real-key InfoNCE denominator."""
    if (
        not bool(enabled)
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return embeddings, labels, 1

    world_size = dist.get_world_size()
    embedding_parts = dist_nn.all_gather(embeddings.contiguous())
    label_parts = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(label_parts, labels.contiguous())
    return (
        torch.cat(tuple(embedding_parts), dim=0),
        torch.cat(label_parts, dim=0),
        world_size,
    )


class FeatureAdapterSystem(nn.Module):
    """Per-stage residual adapters and optional legacy projection heads."""

    def __init__(
        self,
        stage_channels: Mapping[str, int],
        stages: Iterable[str],
        *,
        bottleneck: int = 64,
        projection_dim: int = 128,
        masked_predictor_bottleneck: int = 0,
        num_classes: int = 1000,
        dropout: float = 0.0,
        use_ce: bool = False,
    ):
        super().__init__()
        self.stages = canonical_adapter_stages(stages)
        if not self.stages:
            raise ValueError("feature adapter must select at least one stage")
        self.use_ce = bool(use_ce)
        self.adapters = nn.ModuleDict()
        projection_dim = int(projection_dim)
        if projection_dim < 0:
            raise ValueError("adapter projection_dim must be non-negative")
        if self.use_ce and projection_dim == 0:
            raise ValueError("adapter CE objective requires a projection head")
        masked_predictor_bottleneck = int(masked_predictor_bottleneck)
        if masked_predictor_bottleneck < 0:
            raise ValueError("masked predictor bottleneck must be non-negative")
        self.projectors = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        self.masked_predictors = nn.ModuleDict()
        for stage in self.stages:
            if stage not in stage_channels:
                raise ValueError(f"Missing channel width for {stage}")
            channels = int(stage_channels[stage])
            self.adapters[stage] = ResidualSpatialAdapter(
                channels, int(bottleneck), float(dropout)
            )
            if projection_dim > 0:
                self.projectors[stage] = nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Linear(channels, projection_dim),
                )
            if self.use_ce:
                self.classifiers[stage] = nn.Linear(
                    projection_dim, int(num_classes)
                )
            if masked_predictor_bottleneck > 0:
                self.masked_predictors[stage] = ResidualPatchPredictor(
                    channels,
                    masked_predictor_bottleneck,
                    float(dropout),
                )

    def forward(
        self,
        stage_features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        batch_size: int,
        positive_count: int,
        samples_per_class: int,
        temperature: float,
        supcon_weight: float,
        ce_weight: float,
        reg_weight: float,
        drift_align_weight: float = 0.0,
        distance_scale_weight: float = 0.0,
        distance_mean_weight: float = 0.0,
        objective: str = "supcon",
        generated_stage_features: Optional[Dict[str, torch.Tensor]] = None,
        generated_count: int = 0,
        generated_samples_per_class: Optional[int] = None,
        gather_distributed: bool = False,
        collect_diagnostics: bool = True,
        negative_count: int = 0,
        weight_neg: Optional[torch.Tensor] = None,
        target_positive_features: Optional[Dict[str, torch.Tensor]] = None,
        target_negative_features: Optional[Dict[str, torch.Tensor]] = None,
        target_generated_features: Optional[Dict[str, torch.Tensor]] = None,
        drift_options: Optional[Mapping[str, Any]] = None,
        masked_stage_features: Optional[Dict[str, torch.Tensor]] = None,
        masked_patch_mask: Optional[torch.Tensor] = None,
        mdino_options: Optional[Mapping[str, Any]] = None,
        view_swap: bool = False,
        split_index: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train the adapter with the configured contrastive objective."""
        objective = str(objective).lower().strip()
        if objective == "raw_drift_snr":
            required = (
                generated_stage_features,
                target_positive_features,
                target_generated_features,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "raw_drift_snr requires raw generated maps and EMA-target "
                    "positive/generated drift features"
                )
            return self._forward_raw_drift_snr_v2(
                stage_features,
                generated_stage_features,
                target_positive_features,
                target_generated_features,
                labels=labels,
                batch_size=batch_size,
                positive_count=positive_count,
                negative_count=negative_count,
                generated_count=generated_count,
                reg_weight=reg_weight,
                gather_distributed=gather_distributed,
                collect_diagnostics=collect_diagnostics,
                options={} if drift_options is None else drift_options,
                split_index=int(split_index),
            )
        if objective == "raw_drift_snr_consistency":
            required = (
                generated_stage_features,
                target_positive_features,
                target_negative_features,
                target_generated_features,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "raw_drift_snr_consistency requires raw generated maps and "
                    "EMA-target positive/negative/generated drift features"
                )
            return self._forward_raw_drift_snr_consistency(
                stage_features,
                generated_stage_features,
                target_positive_features,
                target_negative_features,
                target_generated_features,
                batch_size=batch_size,
                positive_count=positive_count,
                negative_count=negative_count,
                generated_count=generated_count,
                weight_neg=weight_neg,
                reg_weight=reg_weight,
                gather_distributed=gather_distributed,
                collect_diagnostics=collect_diagnostics,
                options={} if drift_options is None else drift_options,
                view_swap=bool(view_swap),
            )
        if objective == "dino_cf_drift_realreal_infonce":
            if generated_stage_features is None:
                raise ValueError(
                    "dino_cf_drift_realreal_infonce requires generated raw maps"
                )
            return self._forward_counterfactual_drift_consistency(
                stage_features,
                generated_stage_features,
                labels,
                batch_size=batch_size,
                positive_count=positive_count,
                negative_count=negative_count,
                generated_count=generated_count,
                real_samples_per_class=samples_per_class,
                generated_samples_per_class=(
                    generated_count
                    if generated_samples_per_class is None
                    else generated_samples_per_class
                ),
                temperature=temperature,
                infonce_weight=supcon_weight,
                reg_weight=reg_weight,
                gather_distributed=gather_distributed,
                collect_diagnostics=collect_diagnostics,
                options={} if drift_options is None else drift_options,
                split_index=int(split_index),
            )
        if objective == "gen_real_multipos_infonce":
            if generated_stage_features is None:
                raise ValueError(
                    "gen_real_multipos_infonce requires generated stage features"
                )
            return self._forward_generated_to_real(
                stage_features,
                generated_stage_features,
                labels,
                batch_size=batch_size,
                positive_count=positive_count,
                generated_count=generated_count,
                real_samples_per_class=samples_per_class,
                generated_samples_per_class=(
                    generated_count
                    if generated_samples_per_class is None
                    else generated_samples_per_class
                ),
                temperature=temperature,
                infonce_weight=supcon_weight,
                reg_weight=reg_weight,
                drift_align_weight=drift_align_weight,
                distance_scale_weight=distance_scale_weight,
                distance_mean_weight=distance_mean_weight,
                gather_distributed=gather_distributed,
                collect_diagnostics=collect_diagnostics,
            )
        if objective == "real_real_multipos_infonce":
            if generated_stage_features is None:
                raise ValueError(
                    "real_real_multipos_infonce requires generated denominator "
                    "features"
                )
            return self._forward_real_to_real(
                stage_features,
                generated_stage_features,
                labels,
                batch_size=batch_size,
                positive_count=positive_count,
                generated_count=generated_count,
                real_samples_per_class=samples_per_class,
                generated_samples_per_class=(
                    generated_count
                    if generated_samples_per_class is None
                    else generated_samples_per_class
                ),
                temperature=temperature,
                infonce_weight=supcon_weight,
                reg_weight=reg_weight,
                gather_distributed=gather_distributed,
                collect_diagnostics=collect_diagnostics,
            )
        if objective == "real_real_multipos_infonce_mdino":
            if generated_stage_features is None:
                raise ValueError(
                    "real_real_multipos_infonce_mdino requires generated "
                    "denominator features for its InfoNCE term"
                )
            if masked_stage_features is None or masked_patch_mask is None:
                raise ValueError(
                    "real_real_multipos_infonce_mdino requires masked DINO "
                    "stage features and their patch mask"
                )
            return self._forward_real_to_real_masked_dino(
                stage_features,
                masked_stage_features,
                generated_stage_features,
                masked_patch_mask,
                labels,
                batch_size=batch_size,
                positive_count=positive_count,
                generated_count=generated_count,
                real_samples_per_class=samples_per_class,
                generated_samples_per_class=(
                    generated_count
                    if generated_samples_per_class is None
                    else generated_samples_per_class
                ),
                temperature=temperature,
                infonce_weight=supcon_weight,
                reg_weight=reg_weight,
                gather_distributed=gather_distributed,
                collect_diagnostics=collect_diagnostics,
                options={} if mdino_options is None else mdino_options,
            )
        if objective not in {"supcon", "supcon_ce"}:
            raise ValueError(f"Unknown feature-adapter objective {objective!r}")

        # Legacy real-only supervised-contrastive objective.
        take = min(int(samples_per_class), int(positive_count))
        if take < 2:
            raise ValueError("feature adapter needs at least two positives per class")
        repeated_labels = labels[:, None].expand(batch_size, take).reshape(-1)
        total = torch.zeros((), device=labels.device, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}

        for stage in self.stages:
            layer = f"layer{stage[-1]}"
            if layer not in stage_features:
                raise KeyError(f"MAE did not emit required adapter feature {layer}")
            feature = stage_features[layer].detach()
            if feature.shape[0] < batch_size * positive_count:
                raise ValueError(
                    f"{layer} has {feature.shape[0]} examples, expected "
                    f"at least {batch_size * positive_count}"
                )
            feature = feature[: batch_size * positive_count]
            feature = feature.reshape(
                batch_size, positive_count, *feature.shape[1:]
            )[:, :take]
            feature = feature.reshape(-1, *feature.shape[2:])
            adapted = self.adapters[stage](feature)
            pooled = adapted.float().mean(dim=(2, 3))
            if stage not in self.projectors:
                raise RuntimeError(
                    "real-only SupCon requires feature_adapter_projection_dim > 0"
                )
            projected = self.projectors[stage](pooled)
            supcon = supervised_contrastive_loss(
                projected, repeated_labels, temperature
            )
            base_power = feature.float().square().mean().clamp_min(1e-6)
            residual_ratio = (
                (adapted.float() - feature.float()).square().mean() / base_power
            )
            stage_loss = float(supcon_weight) * supcon
            if self.use_ce:
                ce = F.cross_entropy(
                    self.classifiers[stage](projected.float()), repeated_labels
                )
                stage_loss = stage_loss + float(ce_weight) * ce
                metrics[f"adapter/{stage}_ce"] = ce.detach()
            stage_loss = stage_loss + float(reg_weight) * residual_ratio
            total = total + stage_loss
            metrics[f"adapter/{stage}_supcon"] = supcon.detach()
            metrics[f"adapter/{stage}_residual_ratio"] = residual_ratio.detach()

        total = total / float(len(self.stages))
        metrics["adapter/loss"] = total.detach()
        return total, metrics

    def _forward_counterfactual_drift_consistency(
        self,
        real_stage_features: Dict[str, torch.Tensor],
        generated_stage_features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        batch_size: int,
        positive_count: int,
        negative_count: int,
        generated_count: int,
        real_samples_per_class: int,
        generated_samples_per_class: int,
        temperature: float,
        infonce_weight: float,
        reg_weight: float,
        gather_distributed: bool,
        collect_diagnostics: bool,
        options: Mapping[str, Any],
        split_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Real-only InfoNCE plus direct counterfactual drift consistency.

        A single detached generated query group is evaluated against two
        disjoint same-class real/generated-support banks.  The two online raw
        fields are both differentiable.  A separate three-way real split
        estimates the population-zero p-vs-p field, while terminal DINO stage
        embeddings preserve the frozen encoder's pairwise cosine geometry.
        Encoder outputs are detached at entry and each raw map passes through
        its stage adapter exactly once.
        """
        from drifting_core.imagenet_loss import reverse_drift_raw_fields

        B = int(batch_size)
        P = min(int(real_samples_per_class), int(positive_count))
        N = int(negative_count)
        G = min(int(generated_samples_per_class), int(generated_count))
        if B <= 0:
            raise ValueError("counterfactual adapter batch_size must be positive")
        if P < 6:
            raise ValueError(
                "counterfactual adapter needs at least six same-class real samples"
            )
        if G < 3:
            raise ValueError(
                "counterfactual adapter needs at least three generated samples"
            )
        if N < 0:
            raise ValueError("counterfactual adapter negative_count cannot be negative")

        R_list = tuple(
            float(value) for value in options.get("R_list", (0.2, 0.05, 0.02))
        )
        if not R_list or any(value <= 0.0 for value in R_list):
            raise ValueError(
                "counterfactual adapter R_list must contain positive values"
            )
        patch_mean_size = tuple(
            int(v) for v in options.get("patch_mean_size", (2, 4))
        )
        patch_std_size = tuple(
            int(v) for v in options.get("patch_std_size", (2, 4))
        )
        use_mean = bool(options.get("use_mean", True))
        use_std = bool(options.get("use_std", True))
        feature_weights = {
            str(key): float(value)
            for key, value in dict(options.get("feature_loss_weights", {})).items()
        }
        temperature_multipliers = {
            str(key): float(value)
            for key, value in dict(options.get("temperature_multipliers", {})).items()
        }
        epsilon = float(options.get("cf_epsilon", 1.0e-8))
        rho = float(options.get("cf_rho", 1.0))
        cf_weight = float(options.get("cf_weight", 1.0))
        null_weight = float(options.get("null_weight", 1.0))
        relation_weight = float(options.get("relation_weight", 1.0))
        for name, value in (
            ("cf_weight", cf_weight),
            ("null_weight", null_weight),
            ("relation_weight", relation_weight),
            ("cf_rho", rho),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"counterfactual adapter {name} must be finite and non-negative"
                )
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("counterfactual adapter epsilon must be positive")
        global_scale_stats = bool(options.get("global_scale_stats", True))
        global_statistics = bool(options.get("global_statistics", True))

        common_force_options = {
            "global_scale_stats": global_scale_stats,
            "top_p": float(options.get("top_p", 1.0)),
            "top_p_min_keep": int(options.get("top_p_min_keep", 1)),
            "top_k_pos": int(options.get("top_k_pos", 0)),
            "top_k_neg": int(options.get("top_k_neg", 0)),
            "affinity_kernel": str(options.get("affinity_kernel", "exponential")),
            "kernel_shape": float(options.get("kernel_shape", 1.0)),
            "kernel_adaptive_k_pos": int(options.get("kernel_adaptive_k_pos", 0)),
            "kernel_adaptive_k_neg": int(options.get("kernel_adaptive_k_neg", 0)),
            "kernel_adaptive_margin": float(
                options.get("kernel_adaptive_margin", 1.05)
            ),
            "kernel_mix_weight": float(options.get("kernel_mix_weight", 0.5)),
            "kernel_temperature_mix": tuple(
                float(v) for v in options.get("kernel_temperature_mix", ())
            ),
            "kernel_temperature_mix_weights": tuple(
                float(v)
                for v in options.get("kernel_temperature_mix_weights", ())
            ),
        }

        device = labels.device
        p_first_size = P // 2
        p_second_size = P - p_first_size
        p_first, p_second = rotating_partition_indices(
            P,
            (p_first_size, p_second_size),
            split_index,
            device=device,
        )
        generated_query_size = G // 3
        remaining_generated = G - generated_query_size
        generated_support_first_size = remaining_generated // 2
        generated_support_second_size = (
            remaining_generated - generated_support_first_size
        )
        generated_query, generated_support_first, generated_support_second = (
            rotating_partition_indices(
                G,
                (
                    generated_query_size,
                    generated_support_first_size,
                    generated_support_second_size,
                ),
                split_index,
                device=device,
            )
        )
        null_support_size = P // 3
        null_query_size = P - 2 * null_support_size
        null_query, null_attractive, null_repulsive = rotating_partition_indices(
            P,
            (null_query_size, null_support_size, null_support_size),
            3 * split_index + 1,
            device=device,
        )

        real_labels = labels[:, None].expand(B, P).reshape(-1)
        generated_labels = labels[:, None].expand(B, G).reshape(-1)
        infonce_rows = []
        relation_rows = []
        residual_rows = []
        sufficient_rows = []
        term_weights = []
        derived_feature_count = 0
        raw_map_count = 0

        for map_name, real_raw_full in real_stage_features.items():
            stage = _feature_stage_for_name(map_name)
            if stage not in self.adapters:
                continue
            if map_name not in generated_stage_features:
                raise KeyError(f"missing generated pre-adapter map {map_name}")
            real_expected = B * int(positive_count + N)
            generated_expected = B * int(generated_count)
            if real_raw_full.shape[0] < real_expected:
                raise ValueError(
                    f"{map_name} has {real_raw_full.shape[0]} real maps, expected "
                    f"at least {real_expected}"
                )
            generated_raw_full = generated_stage_features[map_name]
            if generated_raw_full.shape[0] < generated_expected:
                raise ValueError(
                    f"{map_name} has {generated_raw_full.shape[0]} generated maps, "
                    f"expected at least {generated_expected}"
                )

            positive_raw_full = real_raw_full[: B * int(positive_count)].detach()
            positive_raw = positive_raw_full.reshape(
                B, int(positive_count), *positive_raw_full.shape[1:]
            )[:, :P].reshape(-1, *positive_raw_full.shape[1:])
            generated_raw_full = generated_raw_full[:generated_expected].detach()
            generated_raw = generated_raw_full.reshape(
                B, int(generated_count), *generated_raw_full.shape[1:]
            )[:, :G].reshape(-1, *generated_raw_full.shape[1:])
            raw_combined = torch.cat([positive_raw, generated_raw], dim=0)
            adapted_combined = self.adapters[stage](raw_combined)
            adapted_positive = adapted_combined[: B * P]
            adapted_generated = adapted_combined[B * P :]
            raw_map_count += 1

            base_power = raw_combined.float().square().mean().clamp_min(1.0e-6)
            residual_rows.append(
                (adapted_combined.float() - raw_combined.float()).square().mean()
                / base_power
            )

            terminal_name = f"layer{stage[-1]}"
            if map_name == terminal_name:
                base_real = positive_raw.float().mean(dim=(2, 3))
                base_generated = generated_raw.float().mean(dim=(2, 3))
                projected_real = adapted_positive.float().mean(dim=(2, 3))
                projected_generated = adapted_generated.float().mean(dim=(2, 3))
                local_adapted = torch.cat(
                    [projected_real, projected_generated], dim=0
                )
                local_base = torch.cat([base_real, base_generated], dim=0)
                local_labels = torch.cat(
                    [real_labels, torch.full_like(generated_labels, -1)], dim=0
                )
                packed_candidates, candidate_labels, candidate_ranks = (
                    _gather_real_candidates(
                        torch.cat([local_adapted, local_base], dim=1),
                        local_labels,
                        enabled=gather_distributed,
                    )
                )
                embedding_dim = local_adapted.shape[1]
                adapted_candidates = packed_candidates[:, :embedding_dim]
                base_candidates = packed_candidates[:, embedding_dim:]
                rank = (
                    dist.get_rank()
                    if candidate_ranks > 1
                    and dist.is_available()
                    and dist.is_initialized()
                    else 0
                )
                self_candidate_indices = (
                    torch.arange(projected_real.shape[0], device=device)
                    + rank * local_adapted.shape[0]
                )
                infonce = real_to_real_multi_positive_info_nce(
                    projected_real,
                    adapted_candidates,
                    real_labels,
                    candidate_labels,
                    candidate_labels.ge(0),
                    self_candidate_indices,
                    temperature,
                )
                real_candidate_mask = candidate_labels.ge(0)
                relation = pairwise_cosine_relation_mse(
                    base_candidates[real_candidate_mask],
                    adapted_candidates[real_candidate_mask],
                    epsilon=epsilon,
                )
                infonce_rows.append(infonce)
                relation_rows.append(relation)

            online_features = derive_ssl_map_features(
                map_name,
                adapted_combined,
                patch_mean_size=patch_mean_size,
                patch_std_size=patch_std_size,
                use_std=use_std,
                use_mean=use_mean,
            )
            for feature_name, online_all in online_features.items():
                feature_weight = float(feature_weights.get(feature_name, 1.0))
                if feature_weight <= 0.0:
                    continue
                tokens = online_all.shape[1]
                positive_online_bt = _reshape_feature_bank(
                    online_all[: B * P], batch_size=B, sample_count=P
                )
                generated_online_bt = _reshape_feature_bank(
                    online_all[B * P :], batch_size=B, sample_count=G
                )
                multiplier = float(
                    temperature_multipliers.get(
                        feature_name,
                        temperature_multipliers.get(
                            stage, temperature_multipliers.get("default", 1.0)
                        ),
                    )
                )
                feature_R_list = tuple(
                    round(value * multiplier, 12) for value in R_list
                )
                first_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    generated_online_bt[:, generated_query],
                    positive_online_bt[:, p_first],
                    generated_online_bt[:, generated_support_first],
                    R_list=feature_R_list,
                    include_query_targets=False,
                    repulsion_coefficient=rho,
                    **common_force_options,
                )
                second_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    generated_online_bt[:, generated_query],
                    positive_online_bt[:, p_second],
                    generated_online_bt[:, generated_support_second],
                    R_list=feature_R_list,
                    include_query_targets=False,
                    repulsion_coefficient=rho,
                    **common_force_options,
                )
                null_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    positive_online_bt[:, null_query],
                    positive_online_bt[:, null_attractive],
                    positive_online_bt[:, null_repulsive],
                    R_list=feature_R_list,
                    include_query_targets=False,
                    # Equal attraction/repulsion is required for a population-
                    # zero p-vs-p field, independent of the signal rho.
                    repulsion_coefficient=1.0,
                    **common_force_options,
                )
                for first_field, second_field, null_field in zip(
                    first_fields, second_fields, null_fields
                ):
                    sufficient_rows.append(
                        torch.stack(
                            [
                                (first_field.float() * second_field.float()).sum(),
                                first_field.float().square().sum(),
                                second_field.float().square().sum(),
                                null_field.float().square().sum(),
                                first_field.new_tensor(float(first_field.numel())),
                                null_field.new_tensor(float(null_field.numel())),
                            ]
                        )
                    )
                    term_weights.append(feature_weight)
                derived_feature_count += 1

        if not sufficient_rows:
            raise RuntimeError("counterfactual adapter found no active drift features")
        if not infonce_rows or not relation_rows:
            raise RuntimeError(
                "counterfactual adapter requires terminal layer maps for InfoNCE/relation"
            )

        local_statistics = torch.stack(sufficient_rows, dim=0)
        term_count = local_statistics.shape[0]
        gathered_statistics = _gather_with_gradient(
            local_statistics,
            bool(gather_distributed and global_statistics),
        )
        if gathered_statistics.shape[0] % term_count:
            raise RuntimeError("counterfactual field statistic gather shape mismatch")
        global_statistics_sum = gathered_statistics.reshape(
            -1, term_count, local_statistics.shape[1]
        ).sum(dim=0)
        field_count = global_statistics_sum[:, 4].clamp_min(1.0)
        null_count = global_statistics_sum[:, 5].clamp_min(1.0)
        mean_dot = global_statistics_sum[:, 0] / field_count
        mean_first_power = global_statistics_sum[:, 1] / field_count
        mean_second_power = global_statistics_sum[:, 2] / field_count
        null_energy = global_statistics_sum[:, 3] / null_count
        consistency = 1.0 - mean_dot / (
            (mean_first_power * mean_second_power).clamp_min(0.0).sqrt()
            + epsilon
        )
        weights = consistency.new_tensor(term_weights)
        weight_sum = weights.sum().clamp_min(1.0e-12)
        weighted_mean = lambda values: (values * weights).sum() / weight_sum
        mean_consistency = weighted_mean(consistency)
        mean_null = weighted_mean(null_energy)
        mean_infonce = torch.stack(infonce_rows).mean()
        mean_relation = torch.stack(relation_rows).mean()
        residual_ratio = torch.stack(residual_rows).mean()
        cf_objective = (
            mean_consistency
            + null_weight * mean_null
            + relation_weight * mean_relation
        )
        total = (
            float(infonce_weight) * mean_infonce
            + cf_weight * cf_objective
            + float(reg_weight) * residual_ratio
        )

        expected_feature_count = int(
            options.get("expected_feature_count", 0) or 0
        )
        if expected_feature_count and derived_feature_count != expected_feature_count:
            raise RuntimeError(
                "counterfactual adapter feature-scope mismatch: derived "
                f"{derived_feature_count}, expected {expected_feature_count}"
            )
        expected_raw_map_count = int(options.get("expected_raw_map_count", 0) or 0)
        if expected_raw_map_count and raw_map_count != expected_raw_map_count:
            raise RuntimeError(
                "counterfactual adapter raw-map scope mismatch: found "
                f"{raw_map_count}, expected {expected_raw_map_count}"
            )

        metrics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            metrics = {
                "adapter/loss": total.detach(),
                "adapter/infonce_loss": mean_infonce.detach(),
                "adapter/cf_objective": cf_objective.detach(),
                "adapter/cf_consistency": mean_consistency.detach(),
                "adapter/cf_null_energy": mean_null.detach(),
                "adapter/cf_relation": mean_relation.detach(),
                "adapter/residual_ratio": residual_ratio.detach(),
                "adapter/cf_rho": total.new_tensor(rho),
                "adapter/cf_weight": total.new_tensor(cf_weight),
                "adapter/cf_query_count": total.new_tensor(
                    float(generated_query_size)
                ),
                "adapter/cf_real_support_count": total.new_tensor(
                    float(min(p_first_size, p_second_size))
                ),
                "adapter/cf_generated_support_count": total.new_tensor(
                    float(
                        min(
                            generated_support_first_size,
                            generated_support_second_size,
                        )
                    )
                ),
                "adapter/cf_null_query_count": total.new_tensor(
                    float(null_query_size)
                ),
                "adapter/cf_null_support_count": total.new_tensor(
                    float(null_support_size)
                ),
                "adapter/drift_feature_count": total.new_tensor(
                    float(derived_feature_count)
                ),
                "adapter/drift_raw_map_count": total.new_tensor(float(raw_map_count)),
                "adapter/drift_temperature_terms": total.new_tensor(float(term_count)),
                "adapter/cf_split_index": total.new_tensor(float(split_index)),
            }
            for stage, infonce, relation in zip(
                self.stages, infonce_rows, relation_rows
            ):
                metrics[f"adapter/{stage}_real_real_infonce"] = infonce.detach()
                metrics[f"adapter/{stage}_relation"] = relation.detach()
        return total, metrics

    def _forward_real_to_real(
        self,
        real_stage_features: Dict[str, torch.Tensor],
        generated_stage_features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        batch_size: int,
        positive_count: int,
        generated_count: int,
        real_samples_per_class: int,
        generated_samples_per_class: int,
        temperature: float,
        infonce_weight: float,
        reg_weight: float,
        gather_distributed: bool,
        collect_diagnostics: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Use real anchors/positives with detached generated negatives."""
        real_take = min(int(real_samples_per_class), int(positive_count))
        generated_take = min(
            int(generated_samples_per_class), int(generated_count)
        )
        if real_take < 2:
            raise ValueError(
                "real/real adapter InfoNCE needs at least two real samples per class"
            )
        if generated_take < 1:
            raise ValueError(
                "real/real adapter InfoNCE needs at least one generated negative "
                "per class"
            )

        real_labels = labels[:, None].expand(batch_size, real_take).reshape(-1)
        generated_labels = labels[:, None].expand(
            batch_size, generated_take
        ).reshape(-1)
        total = torch.zeros((), device=labels.device, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}
        stage_infonce_losses = []

        for stage in self.stages:
            layer = f"layer{stage[-1]}"
            if layer not in real_stage_features:
                raise KeyError(f"feature encoder did not emit real adapter map {layer}")
            if layer not in generated_stage_features:
                raise KeyError(
                    f"feature encoder did not emit generated adapter map {layer}"
                )

            real = real_stage_features[layer].detach()
            generated = generated_stage_features[layer].detach()
            expected_real = int(batch_size) * int(positive_count)
            expected_generated = int(batch_size) * int(generated_count)
            if real.shape[0] < expected_real:
                raise ValueError(
                    f"{layer} has {real.shape[0]} real examples, expected at least "
                    f"{expected_real}"
                )
            if generated.shape[0] < expected_generated:
                raise ValueError(
                    f"{layer} has {generated.shape[0]} generated examples, expected "
                    f"at least {expected_generated}"
                )
            real = real[:expected_real].reshape(
                batch_size, positive_count, *real.shape[1:]
            )[:, :real_take]
            real = real.reshape(-1, *real.shape[2:])
            generated = generated[:expected_generated].reshape(
                batch_size, generated_count, *generated.shape[1:]
            )[:, :generated_take]
            generated = generated.reshape(-1, *generated.shape[2:])

            adapted_real = self.adapters[stage](real)
            adapted_generated = self.adapters[stage](generated)
            # Optimize the exact generator-visible representation rather than a
            # train-only projection head.
            projected_real = adapted_real.float().mean(dim=(2, 3))
            projected_generated = adapted_generated.float().mean(dim=(2, 3))
            local_candidates = torch.cat(
                [projected_real, projected_generated], dim=0
            )
            # Generated labels are deliberately replaced by a sentinel so they
            # can never enter the positive mask, including same-class generated
            # examples. They remain in the softmax denominator.
            local_candidate_labels = torch.cat(
                [real_labels, torch.full_like(generated_labels, -1)], dim=0
            )
            contrastive_candidates, contrastive_labels, candidate_ranks = (
                _gather_real_candidates(
                    local_candidates,
                    local_candidate_labels,
                    enabled=gather_distributed,
                )
            )
            rank = (
                dist.get_rank()
                if candidate_ranks > 1 and dist.is_available() and dist.is_initialized()
                else 0
            )
            self_candidate_indices = (
                torch.arange(projected_real.shape[0], device=labels.device)
                + rank * local_candidates.shape[0]
            )
            infonce = real_to_real_multi_positive_info_nce(
                projected_real,
                contrastive_candidates,
                real_labels,
                contrastive_labels,
                contrastive_labels.ge(0),
                self_candidate_indices,
                temperature,
            )

            residual_ratio = None
            if float(reg_weight) != 0.0:
                base_power = torch.cat(
                    [real.float(), generated.float()]
                ).square().mean().clamp_min(1.0e-6)
                residual_ratio = torch.cat(
                    [
                        adapted_real.float() - real.float(),
                        adapted_generated.float() - generated.float(),
                    ]
                ).square().mean() / base_power
                stage_loss = (
                    float(infonce_weight) * infonce
                    + float(reg_weight) * residual_ratio
                )
            else:
                stage_loss = float(infonce_weight) * infonce
                if collect_diagnostics:
                    with torch.no_grad():
                        base_power = torch.cat(
                            [real.float(), generated.float()]
                        ).square().mean().clamp_min(1.0e-6)
                        residual_ratio = torch.cat(
                            [
                                adapted_real.float() - real.float(),
                                adapted_generated.float() - generated.float(),
                            ]
                        ).square().mean() / base_power
            total = total + stage_loss
            stage_infonce_losses.append(infonce)

            if collect_diagnostics:
                assert residual_ratio is not None
                with torch.no_grad():
                    query = F.normalize(projected_real.float(), dim=-1)
                    key = F.normalize(contrastive_candidates.float(), dim=-1)
                    similarity = query @ key.transpose(0, 1)
                    row_indices = torch.arange(
                        similarity.shape[0], device=similarity.device
                    )
                    self_mask = torch.zeros_like(similarity, dtype=torch.bool)
                    self_mask[row_indices, self_candidate_indices] = True
                    positive = (
                        real_labels[:, None].eq(contrastive_labels[None, :])
                        & contrastive_labels.ge(0)[None, :]
                        & ~self_mask
                    )
                    negative = ~positive & ~self_mask
                    retrieval_similarity = similarity.masked_fill(
                        self_mask, float("-inf")
                    )
                    positive_similarity = similarity[positive].mean()
                    negative_similarity = (
                        similarity[negative].mean()
                        if bool(negative.any())
                        else torch.zeros((), device=similarity.device)
                    )
                    top1_accuracy = contrastive_labels[
                        retrieval_similarity.argmax(dim=1)
                    ].eq(real_labels).float().mean()

                metrics[f"adapter/{stage}_real_real_infonce"] = infonce.detach()
                metrics[f"adapter/{stage}_positive_similarity"] = positive_similarity
                metrics[f"adapter/{stage}_negative_similarity"] = negative_similarity
                metrics[f"adapter/{stage}_top1_accuracy"] = top1_accuracy
                metrics[f"adapter/{stage}_residual_ratio"] = residual_ratio.detach()
                metrics[f"adapter/{stage}_candidate_ranks"] = torch.tensor(
                    float(candidate_ranks), device=labels.device
                )
                metrics[f"adapter/{stage}_candidates"] = torch.tensor(
                    float(contrastive_candidates.shape[0] - 1), device=labels.device
                )
                metrics[f"adapter/{stage}_generated_negatives"] = torch.tensor(
                    float(generated_take * batch_size * candidate_ranks),
                    device=labels.device,
                )

        total = total / float(len(self.stages))
        if collect_diagnostics:
            metrics["adapter/loss"] = total.detach()
            metrics["adapter/infonce_loss"] = (
                torch.stack(stage_infonce_losses).mean().detach()
            )
        return total, metrics

    def _forward_real_to_real_masked_dino(
        self,
        clean_stage_features: Dict[str, torch.Tensor],
        masked_stage_features: Dict[str, torch.Tensor],
        generated_stage_features: Dict[str, torch.Tensor],
        masked_patch_mask: torch.Tensor,
        labels: torch.Tensor,
        *,
        batch_size: int,
        positive_count: int,
        generated_count: int,
        real_samples_per_class: int,
        generated_samples_per_class: int,
        temperature: float,
        infonce_weight: float,
        reg_weight: float,
        gather_distributed: bool,
        collect_diagnostics: bool,
        options: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Real-real InfoNCE plus masked frozen-DINO feature prediction.

        Clean and masked maps both come from the same frozen DINO backbone and
        are detached at entry.  Only the online spatial adapter and its small
        predictor receive gradients.  Generated maps participate exclusively
        as negatives in the existing InfoNCE term; they never enter the masked
        self-distillation term.
        """
        infonce_total, metrics = self._forward_real_to_real(
            clean_stage_features,
            generated_stage_features,
            labels,
            batch_size=batch_size,
            positive_count=positive_count,
            generated_count=generated_count,
            real_samples_per_class=real_samples_per_class,
            generated_samples_per_class=generated_samples_per_class,
            temperature=temperature,
            infonce_weight=infonce_weight,
            reg_weight=reg_weight,
            gather_distributed=gather_distributed,
            collect_diagnostics=collect_diagnostics,
        )

        mdino_weight = float(options.get("weight", 1.0))
        smooth_l1_weight = float(options.get("smooth_l1_weight", 1.0))
        cosine_weight = float(options.get("cosine_weight", 0.0))
        smooth_l1_beta = float(options.get("smooth_l1_beta", 1.0))
        mask_ratio = float(options.get("mask_ratio", float("nan")))
        masked_samples_per_class = min(
            int(options.get("samples_per_class", real_samples_per_class)),
            int(positive_count),
        )
        if not math.isfinite(mdino_weight) or mdino_weight < 0.0:
            raise ValueError("masked DINO weight must be finite and non-negative")
        if (
            not math.isfinite(smooth_l1_weight)
            or smooth_l1_weight < 0.0
            or not math.isfinite(cosine_weight)
            or cosine_weight < 0.0
            or smooth_l1_weight + cosine_weight <= 0.0
        ):
            raise ValueError(
                "masked DINO SmoothL1/cosine weights must be finite, "
                "non-negative, and not both zero"
            )
        if not math.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0.0:
            raise ValueError("masked DINO SmoothL1 beta must be positive")
        if not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0:
            raise ValueError("masked DINO mask_ratio must be finite and in (0, 1)")
        if masked_samples_per_class <= 0:
            raise ValueError("masked DINO samples_per_class must be positive")
        expected_masked = int(batch_size) * masked_samples_per_class
        if (
            masked_patch_mask.ndim != 3
            or masked_patch_mask.shape[0] != expected_masked
        ):
            raise ValueError(
                "masked DINO patch mask must be "
                f"[{expected_masked},grid_h,grid_w], got "
                f"{tuple(masked_patch_mask.shape)}"
            )

        stage_losses = []
        smooth_l1_losses = []
        cosine_losses = []
        for stage in self.stages:
            layer = f"layer{stage[-1]}"
            if layer not in clean_stage_features:
                raise KeyError(f"clean DINO teacher did not emit {layer}")
            if layer not in masked_stage_features:
                raise KeyError(f"masked DINO student did not emit {layer}")
            if stage not in self.masked_predictors:
                raise RuntimeError(
                    f"masked DINO objective requires a predictor for {stage}"
                )

            clean_full = clean_stage_features[layer].detach()
            expected_clean = int(batch_size) * int(positive_count)
            if clean_full.shape[0] < expected_clean:
                raise ValueError(
                    f"{layer} clean teacher has {clean_full.shape[0]} samples, "
                    f"expected at least {expected_clean}"
                )
            # The memory-bank layout is class-major [B,P,...].  Select within
            # each class before flattening so teacher and masked image order
            # stay exactly aligned.
            teacher = clean_full[:expected_clean].reshape(
                int(batch_size), int(positive_count), *clean_full.shape[1:]
            )[:, :masked_samples_per_class].reshape(
                expected_masked, *clean_full.shape[1:]
            )
            student = masked_stage_features[layer].detach()
            if tuple(student.shape) != tuple(teacher.shape):
                raise ValueError(
                    f"masked/clean {layer} maps must align, got "
                    f"{tuple(student.shape)} and {tuple(teacher.shape)}"
                )

            adapted_student = self.adapters[stage](student)
            predicted = self.masked_predictors[stage](adapted_student).float()
            teacher_tokens = F.layer_norm(
                teacher.detach().float().permute(0, 2, 3, 1),
                (int(teacher.shape[1]),),
                eps=1.0e-6,
            )
            predicted_tokens = predicted.permute(0, 2, 3, 1)
            token_mask = project_patch_mask_to_feature_grid(
                masked_patch_mask,
                output_size=tuple(int(v) for v in teacher.shape[-2:]),
                mask_ratio=mask_ratio,
            )
            if not bool(token_mask.any()):
                raise RuntimeError(f"masked DINO {layer} has no supervised tokens")
            selected_prediction = predicted_tokens[token_mask]
            selected_teacher = teacher_tokens[token_mask]
            smooth_l1 = F.smooth_l1_loss(
                selected_prediction,
                selected_teacher,
                beta=smooth_l1_beta,
            )
            cosine = 1.0 - F.cosine_similarity(
                selected_prediction,
                selected_teacher,
                dim=-1,
                eps=1.0e-6,
            ).mean()
            stage_loss = smooth_l1_weight * smooth_l1 + cosine_weight * cosine
            stage_losses.append(stage_loss)
            smooth_l1_losses.append(smooth_l1)
            cosine_losses.append(cosine)

            if collect_diagnostics:
                metrics[f"adapter/{stage}_mdino_smooth_l1"] = smooth_l1.detach()
                metrics[f"adapter/{stage}_mdino_cosine"] = cosine.detach()
                metrics[f"adapter/{stage}_mdino_mask_fraction"] = (
                    token_mask.float().mean().detach()
                )
                metrics[f"adapter/{stage}_mdino_masked_tokens"] = (
                    token_mask.sum().detach().float()
                )

        mdino_loss = torch.stack(stage_losses).mean()
        total = infonce_total + mdino_weight * mdino_loss
        if collect_diagnostics:
            metrics["adapter/loss"] = total.detach()
            metrics["adapter/mdino_loss"] = mdino_loss.detach()
            metrics["adapter/mdino_weighted_loss"] = (
                mdino_weight * mdino_loss.detach()
            )
            metrics["adapter/mdino_smooth_l1"] = (
                torch.stack(smooth_l1_losses).mean().detach()
            )
            metrics["adapter/mdino_cosine"] = (
                torch.stack(cosine_losses).mean().detach()
            )
            metrics["adapter/mdino_weight"] = total.new_tensor(mdino_weight)
            metrics["adapter/mdino_to_infonce_ratio"] = (
                mdino_weight
                * mdino_loss.detach()
                / metrics["adapter/infonce_loss"].clamp_min(1.0e-8)
            )
            metrics["adapter/mdino_samples_per_class"] = total.new_tensor(
                float(masked_samples_per_class)
            )
        return total, metrics

    def _forward_raw_drift_snr_v2(
        self,
        real_stage_features: Dict[str, torch.Tensor],
        generated_stage_features: Dict[str, torch.Tensor],
        target_positive_features: Dict[str, torch.Tensor],
        target_generated_features: Dict[str, torch.Tensor],
        *,
        labels: torch.Tensor,
        batch_size: int,
        positive_count: int,
        negative_count: int,
        generated_count: int,
        reg_weight: float,
        gather_distributed: bool,
        collect_diagnostics: bool,
        options: Mapping[str, Any],
        split_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Exact raw-drift-SNR Version 2 objective from the MAE experiment.

        Generator sampling remains P/N/G, but this alternating adapter update
        uses only same-class real P supports and current same-class generated Q
        supports.  Different-class real negatives never enter the adapter
        graph.  A private deterministic permutation makes Xq/Pa/Pb/Qa/Qb
        disjoint without consuming the training RNG stream.
        """
        from drifting_core.imagenet_loss import (
            reverse_drift_distance_scale,
            reverse_drift_raw_fields,
        )

        B = int(batch_size)
        P = int(positive_count)
        N = int(negative_count)
        G = int(generated_count)
        if B <= 0 or labels.numel() != B:
            raise ValueError("raw_drift_snr needs one class label per batch row")
        if P <= 0 or N < 0 or G <= 0:
            raise ValueError("raw_drift_snr P/G must be positive and N non-negative")

        real_pool_count = int(options.get("real_pool_count", 32))
        query_count = int(options.get("query_count", 8))
        bank_count = int(options.get("bank_count", 12))
        if min(real_pool_count, query_count, bank_count) <= 0:
            raise ValueError("Version-2 real pool, query, and bank counts must be positive")
        if real_pool_count != query_count + 2 * bank_count:
            raise ValueError(
                "Version-2 real pool must equal unused/query count + two banks: "
                f"got pool={real_pool_count}, query={query_count}, bank={bank_count}"
            )
        if P < real_pool_count:
            raise ValueError(
                f"Version-2 needs at least {real_pool_count} positive reals, got {P}"
            )
        if G != query_count + 2 * bank_count:
            raise ValueError(
                "Version-2 must use every generated sample as Xq+Qa+Qb: "
                f"got G={G}, query={query_count}, bank={bank_count}"
            )

        R_list = tuple(
            float(value) for value in options.get("R_list", (0.2, 0.05, 0.02))
        )
        if not R_list or any(not math.isfinite(value) or value <= 0.0 for value in R_list):
            raise ValueError("raw_drift_snr R_list must contain positive finite values")
        patch_mean_size = tuple(
            int(v) for v in options.get("patch_mean_size", (2, 4))
        )
        patch_std_size = tuple(
            int(v) for v in options.get("patch_std_size", (2, 4))
        )
        use_mean = bool(options.get("use_mean", True))
        use_std = bool(options.get("use_std", True))
        temperature_multipliers = {
            str(key): float(value)
            for key, value in dict(options.get("temperature_multipliers", {})).items()
        }
        snr_epsilon = float(options.get("snr_epsilon", 1.0e-6))
        consistency_epsilon = float(options.get("consistency_epsilon", 1.0e-8))
        snr_weight = float(options.get("snr_weight", 1.0))
        consistency_weight = float(options.get("consistency_weight", 0.25))
        for name, value in (
            ("snr_weight", snr_weight),
            ("consistency_weight", consistency_weight),
            ("reg_weight", float(reg_weight)),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Version-2 {name} must be finite and non-negative")
        if not math.isfinite(consistency_epsilon) or consistency_epsilon <= 0.0:
            raise ValueError("Version-2 consistency epsilon must be positive")
        if bool(options.get("stopgrad_variance", False)):
            raise ValueError("Version-2 does not detach D0 or pooled null variance")
        global_scale_stats = bool(options.get("global_scale_stats", True))
        global_statistics = bool(options.get("global_statistics", False))
        split_seed = int(options.get("split_seed", 43))

        common_force_options = {
            "global_scale_stats": global_scale_stats,
            "top_p": float(options.get("top_p", 1.0)),
            "top_p_min_keep": int(options.get("top_p_min_keep", 1)),
            "top_k_pos": int(options.get("top_k_pos", 0)),
            "top_k_neg": int(options.get("top_k_neg", 0)),
            "affinity_kernel": str(options.get("affinity_kernel", "exponential")),
            "kernel_shape": float(options.get("kernel_shape", 1.0)),
            "kernel_adaptive_k_pos": int(options.get("kernel_adaptive_k_pos", 0)),
            "kernel_adaptive_k_neg": int(options.get("kernel_adaptive_k_neg", 0)),
            "kernel_adaptive_margin": float(
                options.get("kernel_adaptive_margin", 1.05)
            ),
            "kernel_mix_weight": float(options.get("kernel_mix_weight", 0.5)),
            "kernel_temperature_mix": tuple(
                float(v) for v in options.get("kernel_temperature_mix", ())
            ),
            "kernel_temperature_mix_weights": tuple(
                float(v)
                for v in options.get("kernel_temperature_mix_weights", ())
            ),
            # Xq is an evaluation query, never part of the repulsion bank.
            "include_query_targets": False,
        }

        device = next(self.parameters()).device
        pa_indices, pb_indices, real_unused_indices = (
            deterministic_class_partition_indices(
                real_pool_count,
                (bank_count, bank_count, query_count),
                labels=labels,
                split_index=split_index,
                seed=split_seed,
                salt=11_830_291,
                device=device,
            )
        )
        xq_indices, qa_indices, qb_indices = deterministic_class_partition_indices(
            G,
            (query_count, bank_count, bank_count),
            labels=labels,
            split_index=split_index,
            seed=split_seed,
            salt=29_447_617,
            device=device,
        )

        statistics_rows: Dict[str, list[torch.Tensor]] = {
            name: []
            for name in ("D_a", "D_b", "Dpq", "Drr", "Dqq", "D0", "Var0", "J")
        }
        consistency_rows = []
        residual_rows = []
        derived_feature_count = 0
        raw_map_count = 0

        for map_name, real_raw_full in real_stage_features.items():
            stage = _feature_stage_for_name(map_name)
            if stage not in self.adapters:
                continue
            if map_name not in generated_stage_features:
                raise KeyError(f"missing generated pre-adapter map {map_name}")
            real_expected = B * (P + N)
            generated_expected = B * G
            if real_raw_full.shape[0] < real_expected:
                raise ValueError(
                    f"{map_name} has {real_raw_full.shape[0]} real maps, expected "
                    f"at least {real_expected}"
                )
            generated_raw_full = generated_stage_features[map_name]
            if generated_raw_full.shape[0] < generated_expected:
                raise ValueError(
                    f"{map_name} has {generated_raw_full.shape[0]} generated maps, "
                    f"expected at least {generated_expected}"
                )

            # Select before the adapter so the N different-class maps, the
            # extra P maps, and the eight unused real-pool maps cost no adapter
            # compute and have no gradient path.
            positive_raw = real_raw_full[: B * P].detach()
            generated_raw = generated_raw_full[:generated_expected].detach()
            selected_raw_parts = (
                _select_class_major_samples(
                    positive_raw, pa_indices, batch_size=B, sample_count=P
                ),
                _select_class_major_samples(
                    positive_raw, pb_indices, batch_size=B, sample_count=P
                ),
                _select_class_major_samples(
                    generated_raw, xq_indices, batch_size=B, sample_count=G
                ),
                _select_class_major_samples(
                    generated_raw, qa_indices, batch_size=B, sample_count=G
                ),
                _select_class_major_samples(
                    generated_raw, qb_indices, batch_size=B, sample_count=G
                ),
            )
            raw_combined = torch.cat(selected_raw_parts, dim=0)
            adapted_combined = self.adapters[stage](raw_combined)
            raw_map_count += 1
            base_power = raw_combined.float().square().mean().clamp_min(1.0e-6)
            residual_rows.append(
                (adapted_combined.float() - raw_combined.float()).square().mean()
                / base_power
            )
            online_features = derive_ssl_map_features(
                map_name,
                adapted_combined,
                patch_mean_size=patch_mean_size,
                patch_std_size=patch_std_size,
                use_std=use_std,
                use_mean=use_mean,
            )

            part_counts = (
                bank_count,
                bank_count,
                query_count,
                bank_count,
                bank_count,
            )
            part_offsets = []
            offset = 0
            for count in part_counts:
                part_offsets.append((offset, offset + B * count, count))
                offset += B * count

            for feature_name, online_all in online_features.items():
                if feature_name not in target_positive_features:
                    raise KeyError(f"EMA target positive features miss {feature_name}")
                if feature_name not in target_generated_features:
                    raise KeyError(f"EMA target generated features miss {feature_name}")

                online_banks = []
                for start, stop, count in part_offsets:
                    online_banks.append(
                        _reshape_feature_bank(
                            online_all[start:stop],
                            batch_size=B,
                            sample_count=count,
                        )
                    )
                pa_online, pb_online, xq_online, qa_online, qb_online = online_banks

                positive_target_bt = _reshape_feature_bank(
                    target_positive_features[feature_name].detach(),
                    batch_size=B,
                    sample_count=P,
                )
                generated_target_bt = _reshape_feature_bank(
                    target_generated_features[feature_name].detach(),
                    batch_size=B,
                    sample_count=G,
                )
                pa_target = _select_feature_bank(
                    positive_target_bt, pa_indices, batch_size=B
                )
                pb_target = _select_feature_bank(
                    positive_target_bt, pb_indices, batch_size=B
                )
                xq_target = _select_feature_bank(
                    generated_target_bt, xq_indices, batch_size=B
                )
                qa_target = _select_feature_bank(
                    generated_target_bt, qa_indices, batch_size=B
                )
                qb_target = _select_feature_bank(
                    generated_target_bt, qb_indices, batch_size=B
                )

                multiplier = float(
                    temperature_multipliers.get(
                        feature_name,
                        temperature_multipliers.get(
                            stage, temperature_multipliers.get("default", 1.0)
                        ),
                    )
                )
                feature_R_list = tuple(
                    round(value * multiplier, 12) for value in R_list
                )
                online_scale = _raw_fields_fp32(
                    reverse_drift_distance_scale,
                    xq_online,
                    torch.cat(
                        [pa_online, pb_online, qa_online, qb_online], dim=1
                    ),
                    global_scale_stats=global_scale_stats,
                )
                online_force_options = {
                    **common_force_options,
                    "distance_scale": online_scale,
                }
                fields_a = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    xq_online,
                    pa_online,
                    qa_online,
                    R_list=feature_R_list,
                    **online_force_options,
                )
                fields_b = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    xq_online,
                    pb_online,
                    qb_online,
                    R_list=feature_R_list,
                    **online_force_options,
                )
                fields_rr = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    xq_online,
                    pa_online,
                    pb_online,
                    R_list=feature_R_list,
                    **online_force_options,
                )
                fields_qq = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    xq_online,
                    qa_online,
                    qb_online,
                    R_list=feature_R_list,
                    **online_force_options,
                )
                # The EMA teacher computes its own detached distance scale in
                # its own feature space, exactly once for the independent B view.
                with torch.no_grad():
                    target_scale = _raw_fields_fp32(
                        reverse_drift_distance_scale,
                        xq_target,
                        torch.cat(
                            [pa_target, pb_target, qa_target, qb_target], dim=1
                        ),
                        global_scale_stats=global_scale_stats,
                    )
                    target_fields_b = _raw_fields_fp32(
                        reverse_drift_raw_fields,
                        xq_target,
                        pb_target,
                        qb_target,
                        R_list=feature_R_list,
                        **common_force_options,
                        distance_scale=target_scale,
                    )

                for field_a, field_b, field_rr, field_qq, target_field_b in zip(
                    fields_a,
                    fields_b,
                    fields_rr,
                    fields_qq,
                    target_fields_b,
                ):
                    statistic = drift_field_snr_v2_statistic(
                        field_a,
                        field_b,
                        field_rr,
                        field_qq,
                        epsilon=snr_epsilon,
                        global_statistics=bool(
                            gather_distributed and global_statistics
                        ),
                    )
                    for name, value in statistic.items():
                        statistics_rows[name].append(value)
                    consistency_rows.append(
                        drift_direction_consistency(
                            field_a,
                            target_field_b,
                            epsilon=consistency_epsilon,
                        )
                    )
                derived_feature_count += 1

        if not statistics_rows["J"]:
            raise RuntimeError("raw_drift_snr found no active stage features")
        statistics = {
            name: torch.stack(values) for name, values in statistics_rows.items()
        }
        mean_J = statistics["J"].mean()
        mean_consistency = torch.stack(consistency_rows).mean()
        residual_ratio = torch.stack(residual_rows).mean()
        total = (
            -snr_weight * mean_J
            + consistency_weight * mean_consistency
            + float(reg_weight) * residual_ratio
        )

        expected_feature_count = int(options.get("expected_feature_count", 0) or 0)
        if expected_feature_count and derived_feature_count != expected_feature_count:
            raise RuntimeError(
                "raw_drift_snr feature-scope mismatch: derived "
                f"{derived_feature_count}, expected {expected_feature_count}"
            )
        expected_raw_map_count = int(options.get("expected_raw_map_count", 0) or 0)
        if expected_raw_map_count and raw_map_count != expected_raw_map_count:
            raise RuntimeError(
                "raw_drift_snr raw-map scope mismatch: found "
                f"{raw_map_count}, expected {expected_raw_map_count}"
            )

        metrics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            metrics = {
                "adapter/loss": total.detach(),
                "adapter/drift_Da": statistics["D_a"].mean().detach(),
                "adapter/drift_Db": statistics["D_b"].mean().detach(),
                "adapter/drift_Dpq": statistics["Dpq"].mean().detach(),
                "adapter/drift_Drr": statistics["Drr"].mean().detach(),
                "adapter/drift_Dqq": statistics["Dqq"].mean().detach(),
                "adapter/drift_D0": statistics["D0"].mean().detach(),
                "adapter/drift_Var0": statistics["Var0"].mean().detach(),
                "adapter/drift_J": mean_J.detach(),
                "adapter/drift_consistency": mean_consistency.detach(),
                "adapter/residual_ratio": residual_ratio.detach(),
                "adapter/drift_feature_count": total.new_tensor(
                    float(derived_feature_count)
                ),
                "adapter/drift_raw_map_count": total.new_tensor(float(raw_map_count)),
                "adapter/drift_temperature_terms": total.new_tensor(
                    float(len(statistics_rows["J"]))
                ),
                "adapter/v2_real_source_count": total.new_tensor(float(P)),
                "adapter/v2_real_pool_count": total.new_tensor(
                    float(real_pool_count)
                ),
                "adapter/v2_real_unused_count": total.new_tensor(
                    float(real_unused_indices.shape[1])
                ),
                "adapter/v2_query_count": total.new_tensor(float(query_count)),
                "adapter/v2_bank_count": total.new_tensor(float(bank_count)),
                "adapter/v2_different_class_negative_count": total.new_tensor(0.0),
                "adapter/v2_split_index": total.new_tensor(float(split_index)),
                "adapter/drift_global_statistics": total.new_tensor(
                    float(global_statistics)
                ),
            }
        return total, metrics

    def _forward_raw_drift_snr_consistency(
        self,
        real_stage_features: Dict[str, torch.Tensor],
        generated_stage_features: Dict[str, torch.Tensor],
        target_positive_features: Dict[str, torch.Tensor],
        target_negative_features: Dict[str, torch.Tensor],
        target_generated_features: Dict[str, torch.Tensor],
        *,
        batch_size: int,
        positive_count: int,
        negative_count: int,
        generated_count: int,
        weight_neg: Optional[torch.Tensor],
        reg_weight: float,
        gather_distributed: bool,
        collect_diagnostics: bool,
        options: Mapping[str, Any],
        view_swap: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Maximize the exact pre-fnorm drift-field signal-to-null ratio.

        The frozen encoder maps are detached at entry.  Online adapters recreate
        every stage3/4 derived feature used by the generator metric, while the
        already-computed EMA-target features provide a stopped, independent
        half-bank direction target.  No extra images or encoder forward pass is
        needed.
        """
        from drifting_core.imagenet_loss import reverse_drift_raw_fields

        B = int(batch_size)
        P = int(positive_count)
        N = int(negative_count)
        G = int(generated_count)
        if B <= 0:
            raise ValueError("raw drift adapter batch_size must be positive")
        if P < 2 or P % 2:
            raise ValueError("raw drift adapter requires an even positive_count >= 2")
        if G < 2 or G % 2:
            raise ValueError("raw drift adapter requires an even generated_count >= 2")
        if N < 0 or (N > 0 and N % 2):
            raise ValueError("raw drift adapter negative_count must be even")
        if N > 0:
            if weight_neg is None or tuple(weight_neg.shape) != (B, N):
                raise ValueError(
                    f"raw drift adapter needs weight_neg [{B},{N}], got "
                    f"{None if weight_neg is None else tuple(weight_neg.shape)}"
                )
            class_weight_neg = weight_neg.detach().float()
        else:
            class_weight_neg = torch.empty(B, 0, device=next(self.parameters()).device)

        R_list = tuple(float(value) for value in options.get("R_list", (0.2, 0.05, 0.02)))
        if not R_list or any(value <= 0.0 for value in R_list):
            raise ValueError("raw drift adapter R_list must contain positive values")
        patch_mean_size = tuple(int(v) for v in options.get("patch_mean_size", (2, 4)))
        patch_std_size = tuple(int(v) for v in options.get("patch_std_size", (2, 4)))
        use_mean = bool(options.get("use_mean", True))
        use_std = bool(options.get("use_std", True))
        feature_weights = {
            str(key): float(value)
            for key, value in dict(options.get("feature_loss_weights", {})).items()
        }
        temperature_multipliers = {
            str(key): float(value)
            for key, value in dict(options.get("temperature_multipliers", {})).items()
        }
        epsilon = float(options.get("snr_epsilon", 1.0e-8))
        if epsilon <= 0.0:
            raise ValueError("feature_adapter_snr_epsilon must be positive")
        stopgrad_variance = bool(options.get("stopgrad_variance", False))
        snr_weight = float(options.get("snr_weight", 1.0))
        consistency_weight = float(options.get("consistency_weight", 0.1))
        global_scale_stats = bool(options.get("global_scale_stats", True))
        global_statistics = bool(options.get("global_statistics", True))

        common_force_options = {
            "global_scale_stats": global_scale_stats,
            "top_p": float(options.get("top_p", 1.0)),
            "top_p_min_keep": int(options.get("top_p_min_keep", 1)),
            "top_k_pos": int(options.get("top_k_pos", 0)),
            "top_k_neg": int(options.get("top_k_neg", 0)),
            "affinity_kernel": str(options.get("affinity_kernel", "exponential")),
            "kernel_shape": float(options.get("kernel_shape", 1.0)),
            "kernel_adaptive_k_pos": int(options.get("kernel_adaptive_k_pos", 0)),
            "kernel_adaptive_k_neg": int(options.get("kernel_adaptive_k_neg", 0)),
            "kernel_adaptive_margin": float(options.get("kernel_adaptive_margin", 1.05)),
            "kernel_mix_weight": float(options.get("kernel_mix_weight", 0.5)),
            "kernel_temperature_mix": tuple(
                float(v) for v in options.get("kernel_temperature_mix", ())
            ),
            "kernel_temperature_mix_weights": tuple(
                float(v) for v in options.get("kernel_temperature_mix_weights", ())
            ),
        }

        p_half = P // 2
        q_half = G // 2
        n_half = N // 2
        p_first, p_second = slice(0, p_half), slice(p_half, P)
        q_first, q_second = slice(0, q_half), slice(q_half, G)
        n_first, n_second = slice(0, n_half), slice(n_half, N)
        if view_swap:
            online_p, target_p = p_second, p_first
            online_q, other_q = q_second, q_first
            online_n, target_n = n_second, n_first
        else:
            online_p, target_p = p_first, p_second
            online_q, other_q = q_first, q_second
            online_n, target_n = n_first, n_second

        energy_pq_rows = []
        energy_pp_rows = []
        energy_qq_rows = []
        consistency_rows = []
        term_weights = []
        residual_rows = []
        derived_feature_count = 0
        raw_map_count = 0

        for map_name, real_raw_full in real_stage_features.items():
            stage = _feature_stage_for_name(map_name)
            if stage not in self.adapters:
                continue
            if map_name not in generated_stage_features:
                raise KeyError(f"missing generated pre-adapter map {map_name}")
            real_expected = B * (P + N)
            generated_expected = B * G
            if real_raw_full.shape[0] < real_expected:
                raise ValueError(
                    f"{map_name} has {real_raw_full.shape[0]} real maps, expected "
                    f"at least {real_expected}"
                )
            generated_raw_full = generated_stage_features[map_name]
            if generated_raw_full.shape[0] < generated_expected:
                raise ValueError(
                    f"{map_name} has {generated_raw_full.shape[0]} generated maps, "
                    f"expected at least {generated_expected}"
                )

            real_raw = real_raw_full[:real_expected].detach()
            positive_raw = real_raw[: B * P]
            negative_raw = real_raw[B * P : B * (P + N)]
            generated_raw = generated_raw_full[:generated_expected].detach()
            raw_combined = torch.cat(
                [positive_raw, negative_raw, generated_raw], dim=0
            )
            adapted_combined = self.adapters[stage](raw_combined)
            raw_map_count += 1

            base_power = raw_combined.float().square().mean().clamp_min(1.0e-6)
            residual_rows.append(
                (adapted_combined.float() - raw_combined.float()).square().mean()
                / base_power
            )
            online_features = derive_ssl_map_features(
                map_name,
                adapted_combined,
                patch_mean_size=patch_mean_size,
                patch_std_size=patch_std_size,
                use_std=use_std,
                use_mean=use_mean,
            )

            for feature_name, online_all in online_features.items():
                feature_weight = float(feature_weights.get(feature_name, 1.0))
                if feature_weight <= 0.0:
                    continue
                missing = [
                    source_name
                    for source_name, source in (
                        ("positive", target_positive_features),
                        ("negative", target_negative_features),
                        ("generated", target_generated_features),
                    )
                    if feature_name not in source
                ]
                if missing:
                    raise KeyError(
                        f"EMA target is missing {feature_name} in {','.join(missing)} features"
                    )

                tokens = online_all.shape[1]
                positive_online = online_all[: B * P]
                negative_online = online_all[B * P : B * (P + N)]
                generated_online = online_all[B * (P + N) :]
                positive_online_bt = _reshape_feature_bank(
                    positive_online, batch_size=B, sample_count=P
                )
                negative_online_bt = _reshape_feature_bank(
                    negative_online, batch_size=B, sample_count=N
                )
                generated_online_bt = _reshape_feature_bank(
                    generated_online, batch_size=B, sample_count=G
                )
                positive_target_bt = _reshape_feature_bank(
                    target_positive_features[feature_name].detach(),
                    batch_size=B,
                    sample_count=P,
                )
                negative_target_bt = _reshape_feature_bank(
                    target_negative_features[feature_name].detach(),
                    batch_size=B,
                    sample_count=N,
                )
                generated_target_bt = _reshape_feature_bank(
                    target_generated_features[feature_name].detach(),
                    batch_size=B,
                    sample_count=G,
                )

                multiplier = float(
                    temperature_multipliers.get(
                        feature_name,
                        temperature_multipliers.get(
                            stage, temperature_multipliers.get("default", 1.0)
                        ),
                    )
                )
                feature_R_list = tuple(
                    round(value * multiplier, 12) for value in R_list
                )
                weight_neg_bt = class_weight_neg.repeat_interleave(tokens, dim=0)

                signal_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    generated_online_bt,
                    positive_online_bt,
                    negative_online_bt if N > 0 else None,
                    weight_neg=weight_neg_bt if N > 0 else None,
                    R_list=feature_R_list,
                    **common_force_options,
                )
                real_null_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    positive_online_bt[:, online_p],
                    positive_online_bt[:, target_p],
                    R_list=feature_R_list,
                    **common_force_options,
                )
                generated_null_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    generated_online_bt[:, online_q],
                    generated_online_bt[:, other_q],
                    R_list=feature_R_list,
                    **common_force_options,
                )

                half_pos_weight = generated_online_bt.new_full(
                    (B * tokens, p_half), 2.0
                )
                if N > 0:
                    half_online_neg_weight = 2.0 * weight_neg_bt[:, online_n]
                    half_target_neg_weight = 2.0 * weight_neg_bt[:, target_n]
                else:
                    half_online_neg_weight = None
                    half_target_neg_weight = None
                online_consistency_fields = _raw_fields_fp32(
                    reverse_drift_raw_fields,
                    generated_online_bt,
                    positive_online_bt[:, online_p],
                    negative_online_bt[:, online_n] if N > 0 else None,
                    weight_pos=half_pos_weight,
                    weight_neg=half_online_neg_weight,
                    R_list=feature_R_list,
                    **common_force_options,
                )
                with torch.no_grad():
                    target_consistency_fields = _raw_fields_fp32(
                        reverse_drift_raw_fields,
                        generated_target_bt,
                        positive_target_bt[:, target_p],
                        negative_target_bt[:, target_n] if N > 0 else None,
                        weight_pos=half_pos_weight,
                        weight_neg=half_target_neg_weight,
                        R_list=feature_R_list,
                        **common_force_options,
                    )

                for signal, real_null, generated_null, online_dir, target_dir in zip(
                    signal_fields,
                    real_null_fields,
                    generated_null_fields,
                    online_consistency_fields,
                    target_consistency_fields,
                ):
                    energy_pq_rows.append(_class_field_energy(signal, B))
                    energy_pp_rows.append(_class_field_energy(real_null, B))
                    energy_qq_rows.append(_class_field_energy(generated_null, B))
                    consistency_rows.append(
                        drift_direction_consistency(
                            online_dir, target_dir, epsilon=epsilon
                        )
                    )
                    term_weights.append(feature_weight)
                derived_feature_count += 1

        if not energy_pq_rows:
            raise RuntimeError("raw drift adapter found no active stage features")
        term_count = len(energy_pq_rows)
        local_energies = torch.cat(
            [
                torch.stack(energy_pq_rows, dim=0),
                torch.stack(energy_pp_rows, dim=0),
                torch.stack(energy_qq_rows, dim=0),
            ],
            dim=0,
        ).transpose(0, 1)
        global_energies = _gather_with_gradient(
            local_energies, bool(gather_distributed and global_statistics)
        ).transpose(0, 1)
        energy_pq = global_energies[:term_count]
        energy_pp = global_energies[term_count : 2 * term_count]
        energy_qq = global_energies[2 * term_count :]
        Dpq = energy_pq.mean(dim=1)
        Dpp = energy_pp.mean(dim=1)
        Dqq = energy_qq.mean(dim=1)
        D0 = 0.5 * (Dpp + Dqq)
        correction = 1 if global_energies.shape[1] > 1 else 0
        Var0 = 0.5 * (
            energy_pp.var(dim=1, correction=correction)
            + energy_qq.var(dim=1, correction=correction)
        )
        denominator_variance = Var0.detach() if stopgrad_variance else Var0
        J = (Dpq - D0) / (denominator_variance + epsilon).sqrt()

        weights = J.new_tensor(term_weights)
        weight_sum = weights.sum().clamp_min(1.0e-12)
        weighted_mean = lambda values: (values * weights).sum() / weight_sum
        mean_J = weighted_mean(J)
        mean_consistency = weighted_mean(torch.stack(consistency_rows))
        residual_ratio = torch.stack(residual_rows).mean()
        total = (
            -snr_weight * mean_J
            + consistency_weight * mean_consistency
            + float(reg_weight) * residual_ratio
        )

        expected_feature_count = int(options.get("expected_feature_count", 0) or 0)
        if expected_feature_count and derived_feature_count != expected_feature_count:
            raise RuntimeError(
                "raw drift adapter feature-scope mismatch: derived "
                f"{derived_feature_count}, expected {expected_feature_count}"
            )
        expected_raw_map_count = int(options.get("expected_raw_map_count", 0) or 0)
        if expected_raw_map_count and raw_map_count != expected_raw_map_count:
            raise RuntimeError(
                "raw drift adapter raw-map scope mismatch: found "
                f"{raw_map_count}, expected {expected_raw_map_count}"
            )

        metrics: Dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            metrics = {
                "adapter/loss": total.detach(),
                "adapter/drift_Dpq": weighted_mean(Dpq).detach(),
                "adapter/drift_Dpp": weighted_mean(Dpp).detach(),
                "adapter/drift_Dqq": weighted_mean(Dqq).detach(),
                "adapter/drift_D0": weighted_mean(D0).detach(),
                "adapter/drift_Var0": weighted_mean(Var0).detach(),
                "adapter/drift_J": mean_J.detach(),
                "adapter/drift_consistency": mean_consistency.detach(),
                "adapter/residual_ratio": residual_ratio.detach(),
                "adapter/drift_feature_count": total.new_tensor(
                    float(derived_feature_count)
                ),
                "adapter/drift_raw_map_count": total.new_tensor(float(raw_map_count)),
                "adapter/drift_temperature_terms": total.new_tensor(float(term_count)),
                "adapter/drift_view_swap": total.new_tensor(float(view_swap)),
                "adapter/drift_stopgrad_variance": total.new_tensor(
                    float(stopgrad_variance)
                ),
            }
        return total, metrics

    def _forward_generated_to_real(
        self,
        real_stage_features: Dict[str, torch.Tensor],
        generated_stage_features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        batch_size: int,
        positive_count: int,
        generated_count: int,
        real_samples_per_class: int,
        generated_samples_per_class: int,
        temperature: float,
        infonce_weight: float,
        reg_weight: float,
        drift_align_weight: float,
        distance_scale_weight: float,
        distance_mean_weight: float,
        gather_distributed: bool,
        collect_diagnostics: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Update only this module from detached real/generated stage maps."""
        real_take = min(int(real_samples_per_class), int(positive_count))
        generated_take = min(
            int(generated_samples_per_class), int(generated_count)
        )
        if real_take < 2:
            raise ValueError(
                "generated/real adapter InfoNCE needs at least two real positives per class"
            )
        if generated_take < 1:
            raise ValueError(
                "generated/real adapter InfoNCE needs at least one generated query per class"
            )
        if not math.isfinite(float(drift_align_weight)) or float(
            drift_align_weight
        ) < 0.0:
            raise ValueError(
                "feature-adapter drift-align weight must be finite and non-negative"
            )
        for name, value in (
            ("distance-scale", distance_scale_weight),
            ("distance-mean", distance_mean_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"feature-adapter {name} weight must be finite and non-negative"
                )

        real_labels = labels[:, None].expand(batch_size, real_take).reshape(-1)
        generated_labels = labels[:, None].expand(
            batch_size, generated_take
        ).reshape(-1)
        total = torch.zeros((), device=labels.device, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}
        stage_infonce_losses = []
        stage_drift_align_losses = []
        stage_distance_scale_losses = []
        stage_distance_mean_losses = []

        for stage in self.stages:
            layer = f"layer{stage[-1]}"
            if layer not in real_stage_features:
                raise KeyError(f"feature encoder did not emit real adapter map {layer}")
            if layer not in generated_stage_features:
                raise KeyError(
                    f"feature encoder did not emit generated adapter map {layer}"
                )

            real = real_stage_features[layer].detach()
            generated = generated_stage_features[layer].detach()
            expected_real = int(batch_size) * int(positive_count)
            expected_generated = int(batch_size) * int(generated_count)
            if real.shape[0] < expected_real:
                raise ValueError(
                    f"{layer} has {real.shape[0]} real examples, expected at least "
                    f"{expected_real}"
                )
            if generated.shape[0] < expected_generated:
                raise ValueError(
                    f"{layer} has {generated.shape[0]} generated examples, expected at "
                    f"least {expected_generated}"
                )

            real = real[:expected_real].reshape(
                batch_size, positive_count, *real.shape[1:]
            )[:, :real_take]
            real = real.reshape(-1, *real.shape[2:])
            generated = generated[:expected_generated].reshape(
                batch_size, generated_count, *generated.shape[1:]
            )[:, :generated_take]
            generated = generated.reshape(-1, *generated.shape[2:])

            adapted_real = self.adapters[stage](real)
            adapted_generated = self.adapters[stage](generated)
            # Use the exact adapted representation consumed by the drift metric
            # (GAP only, no train-only projection head).  Otherwise the
            # projector can absorb the contrastive objective while the spatial
            # adapter seen by the generator remains effectively unchanged.
            projected_real = adapted_real.float().mean(dim=(2, 3))
            projected_generated = adapted_generated.float().mean(dim=(2, 3))
            contrastive_real, contrastive_real_labels, candidate_ranks = (
                _gather_real_candidates(
                    projected_real,
                    real_labels,
                    enabled=gather_distributed,
                )
            )
            infonce = generated_to_real_multi_positive_info_nce(
                projected_generated,
                contrastive_real,
                generated_labels,
                contrastive_real_labels,
                temperature,
            )

            drift_align = None
            distance_scale = None
            distance_mean = None
            relation_diagnostics = None
            if any(
                float(weight) != 0.0
                for weight in (
                    drift_align_weight,
                    distance_scale_weight,
                    distance_mean_weight,
                )
            ):
                # Reuse the already-computed stage maps.  This adds neither a
                # second MoCo forward nor another distributed all-gather.
                base_real = real.float().mean(dim=(2, 3))
                base_generated = generated.float().mean(dim=(2, 3))
                base_union = torch.cat([base_real, base_generated], dim=0)
                adapted_union = torch.cat(
                    [projected_real, projected_generated], dim=0
                )
                drift_align, distance_scale, distance_mean = (
                    pairwise_distance_geometry_losses(
                        base_union,
                        adapted_union,
                        real_count=base_real.shape[0],
                    )
                )
                if collect_diagnostics:
                    relation_diagnostics = _pairwise_distance_relation_diagnostics(
                        base_union,
                        adapted_union,
                        real_count=base_real.shape[0],
                    )

            residual_ratio = None
            if float(reg_weight) != 0.0:
                base_power = torch.cat(
                    [real.float(), generated.float()]
                ).square().mean().clamp_min(1.0e-6)
                residual_power = torch.cat(
                    [
                        adapted_real.float() - real.float(),
                        adapted_generated.float() - generated.float(),
                    ]
                ).square().mean()
                residual_ratio = residual_power / base_power
                stage_loss = (
                    float(infonce_weight) * infonce
                    + float(reg_weight) * residual_ratio
                )
            else:
                # Do not retain the large FP32 residual graph for a coefficient
                # that is exactly zero. Preserve its logged value only when the
                # caller will consume diagnostics.
                stage_loss = float(infonce_weight) * infonce
                if collect_diagnostics:
                    with torch.no_grad():
                        base_power = torch.cat(
                            [real.float(), generated.float()]
                        ).square().mean().clamp_min(1.0e-6)
                        residual_power = torch.cat(
                            [
                                adapted_real.float() - real.float(),
                                adapted_generated.float() - generated.float(),
                            ]
                        ).square().mean()
                        residual_ratio = residual_power / base_power
            if drift_align is not None:
                stage_loss = stage_loss + float(drift_align_weight) * drift_align
                assert distance_scale is not None and distance_mean is not None
                stage_loss = (
                    stage_loss
                    + float(distance_scale_weight) * distance_scale
                    + float(distance_mean_weight) * distance_mean
                )
            total = total + stage_loss
            stage_infonce_losses.append(infonce)
            if drift_align is not None:
                stage_drift_align_losses.append(drift_align)
                stage_distance_scale_losses.append(distance_scale)
                stage_distance_mean_losses.append(distance_mean)

            if collect_diagnostics:
                assert residual_ratio is not None
                with torch.no_grad():
                    query = F.normalize(projected_generated.float(), dim=-1)
                    key = F.normalize(contrastive_real.float(), dim=-1)
                    similarity = query @ key.transpose(0, 1)
                    positive = generated_labels[:, None].eq(
                        contrastive_real_labels[None, :]
                    )
                    negative = ~positive
                    positive_similarity = similarity[positive].mean()
                    negative_similarity = (
                        similarity[negative].mean()
                        if bool(negative.any())
                        else torch.zeros((), device=similarity.device)
                    )
                    top1_accuracy = contrastive_real_labels[
                        similarity.argmax(dim=1)
                    ].eq(generated_labels).float().mean()

                metrics[f"adapter/{stage}_gen_real_infonce"] = infonce.detach()
                metrics[f"adapter/{stage}_positive_similarity"] = positive_similarity
                metrics[f"adapter/{stage}_negative_similarity"] = negative_similarity
                metrics[f"adapter/{stage}_top1_accuracy"] = top1_accuracy
                metrics[f"adapter/{stage}_residual_ratio"] = residual_ratio.detach()
                metrics[f"adapter/{stage}_candidate_ranks"] = torch.tensor(
                    float(candidate_ranks), device=labels.device
                )
                if drift_align is not None:
                    assert relation_diagnostics is not None
                    assert distance_scale is not None and distance_mean is not None
                    metrics[f"adapter/{stage}_drift_align_loss"] = (
                        drift_align.detach()
                    )
                    metrics[f"adapter/{stage}_distance_scale_loss"] = (
                        distance_scale.detach()
                    )
                    metrics[f"adapter/{stage}_distance_mean_loss"] = (
                        distance_mean.detach()
                    )
                    for name, value in relation_diagnostics.items():
                        metrics[f"adapter/{stage}_{name}"] = value.detach()

        total = total / float(len(self.stages))
        if collect_diagnostics:
            metrics["adapter/loss"] = total.detach()
            mean_infonce = torch.stack(stage_infonce_losses).mean()
            metrics["adapter/infonce_loss"] = mean_infonce.detach()
            if stage_drift_align_losses:
                mean_drift_align = torch.stack(stage_drift_align_losses).mean()
                weighted_drift_align = float(
                    drift_align_weight
                ) * mean_drift_align
                metrics["adapter/drift_align_loss"] = mean_drift_align.detach()
                metrics["adapter/drift_align_weighted_loss"] = (
                    weighted_drift_align.detach()
                )
                metrics["adapter/drift_align_to_infonce_ratio"] = (
                    weighted_drift_align / mean_infonce.detach().clamp_min(1.0e-8)
                ).detach()
                metrics["adapter/distance_relation_corr"] = (
                    1.0 - mean_drift_align.detach()
                )
                metrics["adapter/drift_align_lambda"] = total.new_tensor(
                    float(drift_align_weight)
                )
                mean_distance_scale = torch.stack(
                    stage_distance_scale_losses
                ).mean()
                mean_distance_mean = torch.stack(stage_distance_mean_losses).mean()
                weighted_distance_scale = (
                    float(distance_scale_weight) * mean_distance_scale
                )
                weighted_distance_mean = (
                    float(distance_mean_weight) * mean_distance_mean
                )
                metrics["adapter/distance_scale_loss"] = (
                    mean_distance_scale.detach()
                )
                metrics["adapter/distance_scale_weighted_loss"] = (
                    weighted_distance_scale.detach()
                )
                metrics["adapter/distance_scale_to_infonce_ratio"] = (
                    weighted_distance_scale
                    / mean_infonce.detach().clamp_min(1.0e-8)
                ).detach()
                metrics["adapter/distance_scale_lambda"] = total.new_tensor(
                    float(distance_scale_weight)
                )
                metrics["adapter/distance_mean_loss"] = mean_distance_mean.detach()
                metrics["adapter/distance_mean_weighted_loss"] = (
                    weighted_distance_mean.detach()
                )
                metrics["adapter/distance_mean_to_infonce_ratio"] = (
                    weighted_distance_mean
                    / mean_infonce.detach().clamp_min(1.0e-8)
                ).detach()
                metrics["adapter/distance_mean_lambda"] = total.new_tensor(
                    float(distance_mean_weight)
                )
            metrics["adapter/real_samples_per_class"] = torch.tensor(
                float(real_take), device=labels.device
            )
            metrics["adapter/generated_samples_per_class"] = torch.tensor(
                float(generated_take), device=labels.device
            )
        return total, metrics


@torch.no_grad()
def update_adapter_ema(
    target: FeatureAdapterSystem,
    online: nn.Module,
    decay: float,
) -> None:
    """EMA-update a target adapter from a raw or DDP-wrapped online module."""
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("feature adapter EMA decay must be in [0, 1)")
    source = online.module if hasattr(online, "module") else online
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        if float(decay) == 0.0:
            target_param.copy_(source_param.detach())
        else:
            target_param.lerp_(source_param.detach(), 1.0 - float(decay))
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer.detach())
