"""PyTorch port of the official JAX drift_loss for ImageNet-scale training.

Input format: gen/fixed_pos/fixed_neg are [B, C, S]
  B  = batch (= batch_size * num_feature_tokens from get_activations)
  C  = number of samples (pos / neg count)
  S  = feature dimension

This matches the official JAX drift_loss interface exactly.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .indexed_reduce import indexed_weighted_sum


def _cdist_batched(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched pairwise L2 distance. [B,N,D] x [B,M,D] -> [B,N,M]"""
    xydot = torch.bmm(x, y.transpose(1, 2))
    xnorms = (x * x).sum(-1, keepdim=True)          # [B, N, 1]
    ynorms = (y * y).sum(-1).unsqueeze(1)            # [B, 1, M]
    sq = (xnorms + ynorms - 2.0 * xydot).clamp(min=eps)
    return sq.sqrt()


def _wpos_stats_from_matrix(W_pos: torch.Tensor) -> Dict[str, float]:
    """
    Compute W_pos diagnostic stats. W_pos: [B, C_gen, C_pos] (batched).

    Averages stats over the batch dimension B.

    Returned keys:
      wpos/*            - namespaced ImageNet logging keys
      pos_*             - legacy aliases (same values as CIFAR/MNIST repo)
    """
    W = W_pos.detach().float()
    B, C_g, C_p = W.shape

    row_sum = W.sum(dim=2, keepdim=True).clamp(min=1e-8)
    W_n = W / row_sum

    peak = W_n.max(dim=2).values.mean(dim=1)
    eps = 1e-8
    entropy = -(W_n * (W_n + eps).log()).sum(dim=2).mean(dim=1)

    stats: Dict[str, float] = {
        "wpos/peak": float(peak.mean().item()),
        "wpos/entropy": float(entropy.mean().item()),
    }

    winner_counts, max_loads, avg_loads = [], [], []
    for b in range(B):
        argmax_idx = W_n[b].argmax(dim=1)
        load = torch.bincount(argmax_idx, minlength=C_p).float()
        winners = load[load > 0]
        winner_counts.append(float(winners.numel()))
        max_loads.append(float(winners.max().item()) if winners.numel() > 0 else 0.0)
        avg_loads.append(float(winners.mean().item()) if winners.numel() > 0 else 0.0)

    wc = sum(winner_counts) / B
    ml = sum(max_loads) / B
    winner_ratio = wc / max(C_p, 1)
    dominance = ml / max(C_g, 1)
    avg_load = sum(avg_loads) / B

    stats["wpos/winner_count"] = wc
    stats["wpos/winner_ratio"] = winner_ratio
    stats["wpos/unique_ratio"] = winner_ratio
    stats["wpos/max_load"] = ml
    stats["wpos/dominance"] = dominance
    stats["wpos/avg_load"] = avg_load

    stats["pos_peak"] = stats["wpos/peak"]
    stats["pos_entropy"] = stats["wpos/entropy"]
    stats["pos_winner_count"] = wc
    stats["pos_winner_ratio"] = winner_ratio
    stats["pos_unique_ratio"] = winner_ratio
    stats["pos_max_load"] = ml
    stats["pos_dominance"] = dominance
    stats["pos_avg_load"] = avg_load
    return stats


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _global_mean_detached(values: torch.Tensor) -> torch.Tensor:
    """Global mean over all ranks for detached statistics tensors.

    Used to mimic the official JAX merged-batch normalization without
    all-gathering the full feature tensors in the PyTorch DDP port.
    """
    vals = values.detach()
    stats = torch.stack([
        vals.sum(dtype=torch.float64),
        vals.new_tensor(float(vals.numel()), dtype=torch.float64),
    ])
    if _dist_ready():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mean = stats[0] / stats[1].clamp(min=1.0)
    return mean.to(dtype=vals.dtype, device=vals.device)


def _local_mean_detached(values: torch.Tensor) -> torch.Tensor:
    """Process-local mean over a detached tensor."""
    vals = values.detach()
    return vals.mean().to(dtype=vals.dtype, device=vals.device)


def _global_ratio_of_means(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean(numerator) / mean(denominator) over the global DDP batch."""
    num = numerator.detach()
    den = denominator.detach()
    stats = torch.stack([
        num.sum(dtype=torch.float64),
        num.new_tensor(float(num.numel()), dtype=torch.float64),
        den.sum(dtype=torch.float64),
        den.new_tensor(float(den.numel()), dtype=torch.float64),
    ])
    if _dist_ready():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    num_mean = stats[0] / stats[1].clamp(min=1.0)
    den_mean = stats[2] / stats[3].clamp(min=1.0)
    ratio = num_mean / den_mean.clamp(min=eps)
    return ratio.to(dtype=num.dtype, device=num.device)


def _local_ratio_of_means(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Process-local ratio mean(numerator) / mean(denominator)."""
    num = numerator.detach()
    den = denominator.detach()
    num_mean = num.mean()
    den_mean = den.mean()
    return (num_mean / den_mean.clamp(min=eps)).to(dtype=num.dtype, device=num.device)


def _mean_detached(values: torch.Tensor, use_global_stats: bool) -> torch.Tensor:
    """Detached mean over either the global DDP batch or the local process batch."""
    if use_global_stats:
        return _global_mean_detached(values)
    return _local_mean_detached(values)


def _mean_square_detached(
    values: torch.Tensor,
    use_global_stats: bool,
    max_chunk_elements: int = 64 * 1024 * 1024,
) -> torch.Tensor:
    """Mean square without materializing a full-size ``values.square()``.

    MAE-640 raw feature forces can exceed 3 GiB per rank.  Squaring such a
    tensor eagerly creates another equally large temporary even though only a
    scalar is needed.  Chunking preserves the FP32-square/FP64-global-reduce
    calculation while bounding that temporary to at most 256 MiB.
    """
    vals = values.detach()
    flat = vals.reshape(-1)
    accum_dtype = torch.float64 if use_global_stats else vals.dtype
    sum_sq = torch.zeros((), dtype=accum_dtype, device=vals.device)
    for start in range(0, flat.numel(), max_chunk_elements):
        chunk = flat[start:start + max_chunk_elements]
        sum_sq.add_(chunk.square().sum(dtype=accum_dtype))

    stats = torch.stack([
        sum_sq,
        sum_sq.new_tensor(float(flat.numel())),
    ])
    if use_global_stats and _dist_ready():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mean = stats[0] / stats[1].clamp(min=1.0)
    return mean.to(dtype=vals.dtype, device=vals.device)


def _global_mean_squares_detached_fused(
    values: Tuple[torch.Tensor, ...],
    max_chunk_elements: int = 64 * 1024 * 1024,
) -> Tuple[torch.Tensor, ...]:
    """Compute exact global mean-squares with one packed collective.

    Reverse drift needs one global force norm for every temperature in
    ``R_list``. The streaming path performs one all-reduce per temperature to
    keep only one force tensor resident. The opt-in fused path retains those
    force tensors, packs each FP64 ``(sum_sq, count)`` pair into one vector, and
    issues a single all-reduce. Squaring remains FP32 (matching
    :func:`_mean_square_detached`) while accumulation and counts are FP64.
    """
    if not values:
        return ()

    device = values[0].device
    packed = torch.empty(len(values) * 2, dtype=torch.float64, device=device)
    for index, value in enumerate(values):
        vals = value.detach()
        if vals.device != device:
            raise ValueError("all fused force tensors must be on the same device")
        flat = vals.reshape(-1)
        sum_sq = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, flat.numel(), max_chunk_elements):
            chunk = flat[start:start + max_chunk_elements]
            sum_sq.add_(chunk.square().sum(dtype=torch.float64))
        packed[2 * index].copy_(sum_sq)
        packed[2 * index + 1] = float(flat.numel())

    if _dist_ready():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    means = packed[0::2] / packed[1::2].clamp(min=1.0)
    return tuple(
        means[index].to(dtype=value.dtype, device=value.device)
        for index, value in enumerate(values)
    )


def _ratio_of_means(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    use_global_stats: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Detached ratio-of-means over either the global DDP batch or the local process batch."""
    if use_global_stats:
        return _global_ratio_of_means(numerator, denominator, eps=eps)
    return _local_ratio_of_means(numerator, denominator, eps=eps)


def _validate_top_p(top_p: float, min_keep: int) -> Tuple[float, int]:
    """Validate nucleus-coupling parameters and return normalized values."""
    top_p_f = float(top_p)
    min_keep_i = int(min_keep)
    if not 0.0 < top_p_f <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p!r}")
    if min_keep_i < 1:
        raise ValueError(f"top_p_min_keep must be >= 1, got {min_keep!r}")
    return top_p_f, min_keep_i


def _top_p_preserve_mass(
    weights: torch.Tensor,
    dim: int,
    top_p: float = 1.0,
    min_keep: int = 1,
) -> torch.Tensor:
    """Apply deterministic nucleus truncation without changing total mass.

    The input is a non-negative force-weight group, such as all positive
    targets for one generated sample.  The smallest highest-weight support
    reaching ``top_p`` of that group's normalized mass is retained, then
    rescaled to the group's original mass.  Positive and negative force
    strengths therefore remain comparable to the original objective.

    ``top_p=1`` returns the exact input tensor without sorting or arithmetic.
    """
    top_p_f, min_keep_i = _validate_top_p(top_p, min_keep)
    if top_p_f >= 1.0:
        return weights

    dim = dim % weights.ndim
    group_mass = weights.sum(dim=dim, keepdim=True)
    probs = weights / group_mass.clamp_min(1e-12)
    sorted_probs, sorted_indices = probs.sort(dim=dim, descending=True)
    cumulative = sorted_probs.cumsum(dim=dim)

    # Include the element that crosses p, as in nucleus decoding.
    keep = (cumulative - sorted_probs) < top_p_f
    axis_size = sorted_probs.shape[dim]
    min_keep_i = min(min_keep_i, axis_size)
    positions = torch.arange(axis_size, device=weights.device)
    position_shape = [1] * weights.ndim
    position_shape[dim] = axis_size
    keep = keep | (positions.reshape(position_shape) < min_keep_i)

    truncated = sorted_probs * keep.to(dtype=sorted_probs.dtype)
    truncated = truncated / truncated.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    truncated = truncated * group_mass
    return torch.zeros_like(weights).scatter(dim, sorted_indices, truncated)


def _validate_top_k(top_k: int) -> int:
    """Validate a groupwise top-k value; zero disables truncation."""
    top_k_i = int(top_k)
    if top_k_i < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k!r}")
    return top_k_i


def _top_k_preserve_mass(
    weights: torch.Tensor,
    top_k: int = 0,
    dim: int = 2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return compact top-k weights and indices along ``dim``, preserving mass.

    ``top_k=0`` or a value at least as large as the target pool returns the
    original dense tensor and ``None``. Reverse drift uses ``dim=2`` and can
    consume the compact row-wise result directly. Forward drift uses ``dim=1``
    and restores a dense sparse matrix before its coupling calculation.
    """
    top_k_i = _validate_top_k(top_k)
    dim = dim % weights.ndim
    pool_size = weights.shape[dim]
    if top_k_i == 0 or top_k_i >= pool_size:
        return weights, None

    group_mass = weights.sum(dim=dim, keepdim=True)
    kept_weights, kept_indices = torch.topk(
        weights,
        k=top_k_i,
        dim=dim,
        largest=True,
        sorted=False,
    )
    kept_mass = kept_weights.sum(dim=dim, keepdim=True)
    kept_weights = kept_weights * (
        group_mass / kept_mass.clamp_min(1e-12)
    )
    return kept_weights, kept_indices


def _column_top_k_preserve_mass(
    weights: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Keep top generated rows per target column and restore a dense matrix.

    Forward drift normalizes each target column over generated samples. Its
    natural sparse support therefore selects along ``dim=1``. The dense sparse
    result is intentional: coupling needs row sums after column selection, and
    the following small-pool bmm is faster than irregular indexed gathers.
    """
    compact, indices = _top_k_preserve_mass(weights, top_k=top_k, dim=1)
    if indices is None:
        return compact, None
    return torch.zeros_like(weights).scatter(1, indices, compact), indices


def _truncate_force_group(
    weights: torch.Tensor,
    *,
    top_p: float,
    top_p_min_keep: int,
    top_k: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply either dense/top-p or compact top-k force truncation."""
    top_k_i = _validate_top_k(top_k)
    if top_k_i > 0 and float(top_p) < 1.0:
        raise ValueError("top-p and top-k drift truncation cannot be enabled together")
    if top_k_i > 0:
        return _top_k_preserve_mass(weights, top_k=top_k_i)
    return (
        _top_p_preserve_mass(
            weights,
            dim=2,
            top_p=top_p,
            min_keep=top_p_min_keep,
        ),
        None,
    )


def _dense_weights_for_stats(
    weights: torch.Tensor,
    indices: Optional[torch.Tensor],
    pool_size: int,
) -> torch.Tensor:
    """Reconstruct a small dense matrix only on infrequent diagnostic steps."""
    if indices is None:
        return weights
    return weights.new_zeros(
        weights.shape[0], weights.shape[1], pool_size
    ).scatter(2, indices, weights)


_REVERSE_AFFINITY_KERNELS = {
    "cauchy_wendland",
    "exponential",
    "generalized_exponential",
    "matern_32",
    "power_law",
    "student_t",
    "tapered_exponential",
    "wendland",
}


def _validate_reverse_affinity_kernel(
    kernel: str,
    shape: float,
    adaptive_k_pos: int,
    adaptive_k_neg: int,
    adaptive_margin: float,
) -> Tuple[str, float, int, int, float]:
    """Validate reverse-drift distance-to-affinity kernel parameters."""
    kernel_name = str(kernel).lower().strip().replace("-", "_")
    if kernel_name == "cauchy":
        kernel_name = "student_t"
    if kernel_name not in _REVERSE_AFFINITY_KERNELS:
        raise ValueError(
            f"Unknown reverse affinity kernel={kernel!r}; choose from "
            f"{sorted(_REVERSE_AFFINITY_KERNELS)} or 'cauchy'"
        )
    shape_f = float(shape)
    if not math.isfinite(shape_f) or shape_f <= 0.0:
        raise ValueError(f"reverse kernel shape must be finite and > 0, got {shape!r}")
    k_pos = int(adaptive_k_pos)
    k_neg = int(adaptive_k_neg)
    if k_pos < 0 or k_neg < 0:
        raise ValueError(
            "reverse adaptive kernel k values must be >= 0, got "
            f"{adaptive_k_pos!r} and {adaptive_k_neg!r}"
        )
    if (k_pos == 0) != (k_neg == 0):
        raise ValueError(
            "reverse adaptive kernel requires both k_pos and k_neg, or neither"
        )
    margin_f = float(adaptive_margin)
    if not math.isfinite(margin_f) or margin_f <= 0.0:
        raise ValueError(
            "reverse adaptive kernel margin must be finite and > 0, got "
            f"{adaptive_margin!r}"
        )
    return kernel_name, shape_f, k_pos, k_neg, margin_f


def _adaptive_reverse_bandwidth(
    distances: torch.Tensor,
    *,
    split_idx: int,
    self_mask: torch.Tensor,
    k_pos: int,
    k_neg: int,
    margin: float,
) -> torch.Tensor:
    """Choose a detached row-local scale that keeps both force groups alive.

    The positive and repulsive pools can have very different sizes. We find
    their k-th distances independently and use the larger radius, guaranteeing
    at least ``k_pos`` positive and ``k_neg`` non-self repulsive candidates per
    generated row for a compact-support kernel.
    """
    if k_pos <= 0 or k_neg <= 0:
        raise ValueError("adaptive reverse bandwidth requires positive k values")
    repulsive = distances[:, :, :split_idx].masked_fill(
        self_mask[:, :, :split_idx].bool(),
        float("inf"),
    )
    positive = distances[:, :, split_idx:]
    available_neg = max(1, repulsive.shape[2] - 1)
    available_pos = positive.shape[2]
    neg_radius = repulsive.kthvalue(min(k_neg, available_neg), dim=2).values
    pos_radius = positive.kthvalue(min(k_pos, available_pos), dim=2).values
    return (
        torch.maximum(neg_radius, pos_radius)
        .mul(float(margin))
        .clamp_min(1e-6)
        .unsqueeze(2)
        .detach()
    )


def _reverse_mutual_affinity(
    distances: torch.Tensor,
    *,
    bandwidth: float,
    kernel: str,
    shape: float,
    local_bandwidth: Optional[torch.Tensor],
    self_mask: torch.Tensor,
    mix_weight: float = 0.5,
    temperature_mix: Tuple[float, ...] = (),
    temperature_mix_weights: Tuple[float, ...] = (),
) -> torch.Tensor:
    """Map normalized distances to the reverse-drift mutual affinity grid.

    ``exponential`` without a local bandwidth deliberately preserves the
    original two-softmax operation bit for bit. Adaptive exponential and the
    other kernels use ordinary row/column sum normalization, followed by the
    same geometric mutual association used by the baseline.
    """
    bandwidth_f = float(bandwidth)
    if not math.isfinite(bandwidth_f) or bandwidth_f <= 0.0:
        raise ValueError(f"kernel bandwidth must be finite and > 0, got {bandwidth!r}")

    temperatures = tuple(float(value) for value in temperature_mix)
    mixture_weights = tuple(float(value) for value in temperature_mix_weights)
    if temperatures:
        if any(not math.isfinite(value) or value <= 0.0 for value in temperatures):
            raise ValueError(
                f"kernel temperature mixture values must be finite and > 0, got {temperatures}"
            )
        if not mixture_weights:
            mixture_weights = (1.0,) * len(temperatures)
        if len(mixture_weights) != len(temperatures):
            raise ValueError(
                "kernel temperature mixture weights must match temperatures, got "
                f"{len(mixture_weights)} and {len(temperatures)}"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in mixture_weights):
            raise ValueError(
                f"kernel temperature mixture weights must be finite and >= 0, got {mixture_weights}"
            )
        mixture_weight_sum = sum(mixture_weights)
        if mixture_weight_sum <= 0.0:
            raise ValueError("kernel temperature mixture weights must have positive mass")
        mixture_weights = tuple(value / mixture_weight_sum for value in mixture_weights)

    if kernel == "exponential" and local_bandwidth is None and not temperatures:
        logits = -distances / bandwidth_f
        aff_row = F.softmax(logits, dim=2)
        aff_col = F.softmax(logits, dim=1)
        return (aff_row * aff_col).clamp(min=1e-6).sqrt()

    scale = distances.new_tensor(bandwidth_f)
    if local_bandwidth is not None:
        scale = local_bandwidth * bandwidth_f
    base_scaled = distances / scale.clamp_min(1e-8)

    if temperatures:
        weights = torch.zeros_like(distances)
        for component_temperature, component_weight in zip(
            temperatures, mixture_weights
        ):
            weights.add_(
                _reverse_kernel_weights(
                    base_scaled / component_temperature,
                    kernel=kernel,
                    shape=shape,
                    mix_weight=mix_weight,
                ),
                alpha=component_weight,
            )
    else:
        weights = _reverse_kernel_weights(
            base_scaled,
            kernel=kernel,
            shape=shape,
            mix_weight=mix_weight,
        )

    # A heavy-tailed kernel does not make the +100 diagonal exactly zero.
    # Honor the reverse objective's self-exclusion explicitly.
    weights = weights.masked_fill(self_mask.bool(), 0.0)
    aff_row = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
    aff_col = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (aff_row * aff_col).clamp_min(0.0).sqrt()


def _reverse_kernel_weights(
    scaled: torch.Tensor,
    *,
    kernel: str,
    shape: float,
    mix_weight: float = 0.5,
) -> torch.Tensor:
    """Evaluate an unnormalized reverse-affinity kernel at scaled distance.

    This helper intentionally excludes self masking and row/column
    normalization. Besides keeping the training path compact, it lets the ESS
    diagnostic compare kernel geometry before mutual normalization.
    """
    if kernel == "exponential":
        return torch.exp(-scaled)
    if kernel == "generalized_exponential":
        return torch.exp(-scaled.pow(float(shape)))
    if kernel == "matern_32":
        z = scaled * math.sqrt(3.0)
        return (1.0 + z) * torch.exp(-z)
    if kernel == "tapered_exponential":
        taper = (1.0 - scaled).clamp_min(0.0)
        return torch.exp(-scaled) * taper.square()
    if kernel == "power_law":
        # A Gamma mixture of inverse temperatures. As shape -> infinity this
        # converges to exp(-distance / bandwidth); finite shape has a
        # polynomial tail representing many neighborhood scales at once.
        return (1.0 + scaled / shape).pow(-shape)
    if kernel == "student_t":
        return (1.0 + scaled.square() / shape).pow(
            -0.5 * (shape + 1.0)
        )
    if kernel == "wendland":
        radius = scaled
        one_minus = (1.0 - radius).clamp_min(0.0)
        return one_minus.pow(4) * (4.0 * radius + 1.0)
    if kernel == "cauchy_wendland":
        mix_weight_f = float(mix_weight)
        if not math.isfinite(mix_weight_f) or not 0.0 <= mix_weight_f <= 1.0:
            raise ValueError(
                "reverse kernel mix weight must be finite and in [0, 1], got "
                f"{mix_weight!r}"
            )
        cauchy = (1.0 + scaled.square()).reciprocal()
        radius = scaled
        one_minus = (1.0 - radius).clamp_min(0.0)
        wendland = one_minus.pow(4) * (4.0 * radius + 1.0)
        return mix_weight_f * cauchy + (1.0 - mix_weight_f) * wendland
    raise ValueError(f"Unsupported reverse affinity kernel={kernel!r}")


def _accumulate_weighted_targets(
    weights: torch.Tensor,
    indices: Optional[torch.Tensor],
    target0: torch.Tensor,
    target1: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Accumulate dense or compact weights against one logical target pool."""
    if indices is not None:
        return indexed_weighted_sum(
            weights,
            indices,
            target0,
            target1,
            out=out,
            alpha=alpha,
            beta=1.0,
        )

    n_target0 = target0.shape[1]
    if out is None:
        result = torch.bmm(weights[:, :, :n_target0], target0)
        if target1 is not None:
            result.baddbmm_(
                weights[:, :, n_target0:],
                target1,
                beta=1.0,
                alpha=1.0,
            )
        if float(alpha) != 1.0:
            result.mul_(float(alpha))
        return result

    out.baddbmm_(
        weights[:, :, :n_target0],
        target0,
        beta=1.0,
        alpha=float(alpha),
    )
    if target1 is not None:
        out.baddbmm_(
            weights[:, :, n_target0:],
            target1,
            beta=1.0,
            alpha=float(alpha),
        )
    return out


def _record_top_p_support(
    info: Dict[str, float],
    pos_weights: torch.Tensor,
    neg_weights: torch.Tensor,
) -> None:
    """Record actual per-generated-sample nucleus support diagnostics."""
    with torch.no_grad():
        for name, weights in (("pos", pos_weights), ("neg", neg_weights)):
            counts = weights.ne(0).sum(dim=2).float()
            row_mass = weights.sum(dim=2)
            info[f"top_p/{name}_kept_mean"] = float(counts.mean().item())
            info[f"top_p/{name}_pool_size"] = float(weights.shape[2])
            info[f"top_p/{name}_zero_row_fraction"] = float(
                row_mass.eq(0).float().mean().item()
            )


def _record_top_k_support(
    info: Dict[str, float],
    pos_weights: torch.Tensor,
    pos_indices: Optional[torch.Tensor],
    pos_pool_size: int,
    neg_weights: torch.Tensor,
    neg_indices: Optional[torch.Tensor],
    neg_pool_size: int,
) -> None:
    """Record the fixed support and zero-mass diagnostics for top-k."""
    with torch.no_grad():
        for name, weights, indices, pool_size in (
            ("pos", pos_weights, pos_indices, pos_pool_size),
            ("neg", neg_weights, neg_indices, neg_pool_size),
        ):
            kept = weights.shape[2] if indices is not None else pool_size
            row_mass = weights.sum(dim=2)
            info[f"top_k/{name}_kept"] = float(kept)
            info[f"top_k/{name}_pool_size"] = float(pool_size)
            info[f"top_k/{name}_zero_row_fraction"] = float(
                row_mass.eq(0).float().mean().item()
            )


def _record_column_top_k_support(
    info: Dict[str, float],
    pos_weights: torch.Tensor,
    pos_indices: Optional[torch.Tensor],
    neg_weights: torch.Tensor,
    neg_indices: Optional[torch.Tensor],
) -> None:
    """Record forward-drift support selected over generated rows per target."""
    with torch.no_grad():
        generated_pool_size = pos_weights.shape[1]
        for name, weights, indices in (
            ("pos", pos_weights, pos_indices),
            ("neg", neg_weights, neg_indices),
        ):
            kept = (
                indices.shape[1]
                if indices is not None
                else generated_pool_size
            )
            column_mass = weights.sum(dim=1)
            info[f"top_k/{name}_kept"] = float(kept)
            info[f"top_k/{name}_pool_size"] = float(generated_pool_size)
            info[f"top_k/{name}_zero_column_fraction"] = float(
                column_mass.eq(0).float().mean().item()
            )


def _accumulate_differentiable_weighted_targets(
    weights: torch.Tensor,
    indices: Optional[torch.Tensor],
    target0: torch.Tensor,
    target1: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Autograd-preserving variant used only by raw-field optimization.

    The optimized CUDA indexed-reduction kernel intentionally has no autograd
    wrapper because the generator loss detaches its force target.  Adapter
    training needs gradients through both compact weights and selected target
    features, so compact top-k uses a native PyTorch gather here.  Dense and
    top-p execution keeps the exact baseline bmm implementation.
    """
    if indices is None:
        return _accumulate_weighted_targets(
            weights,
            indices,
            target0,
            target1,
            out=out,
        )

    targets = target0 if target1 is None else torch.cat([target0, target1], dim=1)
    batch, n_gen, top_k = indices.shape
    n_features = targets.shape[2]
    selected = torch.gather(
        targets.unsqueeze(1).expand(-1, n_gen, -1, -1),
        dim=2,
        index=indices.unsqueeze(-1).expand(
            batch, n_gen, top_k, n_features
        ),
    )
    result = (weights.unsqueeze(-1) * selected).sum(dim=2)
    if out is not None:
        result = result + out
    return result


def reverse_drift_distance_scale(
    query: torch.Tensor,
    supports: torch.Tensor,
    *,
    global_scale_stats: bool = True,
) -> torch.Tensor:
    """Return the detached mean query/support distance used by raw fields.

    Version-2 adapter training shares one activation-wise scale across its
    Pa/Pb/Qa/Qb support partitions.  Keeping this helper beside the baseline
    implementation guarantees identical FP32 distance arithmetic and
    distributed ratio-of-means semantics.
    """
    if query.ndim != 3 or supports.ndim != 3:
        raise ValueError("raw drift query/support tensors must be [B,C,D]")
    if query.shape[0] != supports.shape[0] or query.shape[2] != supports.shape[2]:
        raise ValueError(
            "raw drift query/support batch and feature dimensions must match"
        )
    if query.shape[1] <= 0 or supports.shape[1] <= 0:
        raise ValueError("raw drift query/support sets must be non-empty")
    query_fp32 = query.float()
    supports_fp32 = supports.float()
    distances = _cdist_batched(query_fp32, supports_fp32)
    support_weights = distances.new_ones(
        distances.shape[0], distances.shape[2]
    )
    return _ratio_of_means(
        distances,
        support_weights,
        use_global_stats=bool(global_scale_stats),
    ).detach()


def reverse_drift_raw_fields(
    gen: torch.Tensor,                    # [B, C_g, S]
    fixed_pos: torch.Tensor,              # [B, C_p, S]
    fixed_neg: Optional[torch.Tensor] = None,  # [B, C_n, S]
    weight_gen: Optional[torch.Tensor] = None,  # [B, C_g]
    weight_pos: Optional[torch.Tensor] = None,  # [B, C_p]
    weight_neg: Optional[torch.Tensor] = None,  # [B, C_n]
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    active_mask_pos: Optional[torch.Tensor] = None,  # [B, C_p] 1=active
    active_mask_neg: Optional[torch.Tensor] = None,  # [B, C_n] 1=active
    global_scale_stats: bool = True,
    top_p: float = 1.0,
    top_p_min_keep: int = 1,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
    affinity_kernel: str = "exponential",
    kernel_shape: float = 1.0,
    kernel_adaptive_k_pos: int = 0,
    kernel_adaptive_k_neg: int = 0,
    kernel_adaptive_margin: float = 1.05,
    kernel_mix_weight: float = 0.5,
    kernel_temperature_mix: Tuple[float, ...] = (),
    kernel_temperature_mix_weights: Tuple[float, ...] = (),
    historical_gen: Optional[torch.Tensor] = None,  # [B, C_h, S]
    weight_history: Optional[torch.Tensor] = None,  # [B, C_h]
    include_query_targets: bool = True,
    repulsion_coefficient: float = 1.0,
    distance_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Return differentiable reverse-drift fields before RMS normalization.

    Each returned tensor is the corresponding ``total_force_R`` from
    :func:`drift_loss_imagenet` *after* division by the detached input scale
    and *before* the per-temperature ``f_norm`` division.  Consequently,
    ``field.square().mean()`` is numerically the same quantity recorded as
    ``loss_{R}`` by the baseline reverse loss.

    Unlike the generator training loss, this function does not detach the
    query or target features.  Gradients therefore flow through the complete
    field estimator into ``gen``, ``fixed_pos``, and ``fixed_neg``; this is
    required when learning a feature adapter from field energy.  Scale and
    adaptive-bandwidth statistics remain detached, matching the baseline's
    temperature calibration and preventing the adapter from changing the
    objective merely by differentiating through its normalization statistic.

    ``historical_gen`` is retained for parity with the reverse-loss target
    layout.  It is deliberately detached, just as it is in
    :func:`drift_loss_imagenet`.

    ``include_query_targets=False`` is the support-only estimator used by the
    counterfactual adapter objective.  It removes the query set from the
    repulsive target bank, so the same detached queries can be evaluated
    against two genuinely disjoint generated-support sets without a shared
    query-query force making the two fields agree trivially.  The default is
    unchanged and remains exactly equivalent to the generator's reverse
    drift implementation.

    ``repulsion_coefficient`` multiplies only the repulsive displacement term
    after affinities have been constructed.  It is therefore the explicit
    :math:`\rho` in ``attraction - rho * repulsion``.  This is deliberately
    different from ``weight_neg``, which changes target mass (and hence both
    affinity normalization and the detached input scale).  Its unit default
    preserves every existing caller bit for bit.

    ``distance_scale`` optionally supplies one already-detached common scale.
    The default computes the original per-call scale and is bit-for-bit
    unchanged; Version-2 adapter fields pass a shared Pa/Pb/Qa/Qb scale.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    (
        affinity_kernel,
        kernel_shape,
        kernel_adaptive_k_pos,
        kernel_adaptive_k_neg,
        kernel_adaptive_margin,
    ) = _validate_reverse_affinity_kernel(
        affinity_kernel,
        kernel_shape,
        kernel_adaptive_k_pos,
        kernel_adaptive_k_neg,
        kernel_adaptive_margin,
    )
    kernel_mix_weight = float(kernel_mix_weight)
    if not math.isfinite(kernel_mix_weight) or not 0.0 <= kernel_mix_weight <= 1.0:
        raise ValueError(
            "reverse kernel mix weight must be finite and in [0, 1], got "
            f"{kernel_mix_weight!r}"
        )

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]
    if historical_gen is None:
        historical_gen = gen.new_zeros(B, 0, S)
    if (
        historical_gen.ndim != 3
        or historical_gen.shape[0] != B
        or historical_gen.shape[2] != S
    ):
        raise ValueError(
            "historical_gen must have shape [B, C_h, S] matching gen; got "
            f"{tuple(historical_gen.shape)} for gen {tuple(gen.shape)}"
        )
    C_h = historical_gen.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)
    if weight_history is None:
        weight_history = gen.new_ones(B, C_h)

    if active_mask_pos is None:
        active_mask_pos = gen.new_ones(B, C_p)
    else:
        active_mask_pos = active_mask_pos.to(gen.device).float()
    if active_mask_neg is None:
        active_mask_neg = gen.new_ones(B, C_n)
    else:
        active_mask_neg = active_mask_neg.to(gen.device).float()
    include_query_targets = bool(include_query_targets)
    repulsion_coefficient = float(repulsion_coefficient)
    if (
        not math.isfinite(repulsion_coefficient)
        or repulsion_coefficient < 0.0
    ):
        raise ValueError(
            "raw-field repulsion_coefficient must be finite and non-negative, "
            f"got {repulsion_coefficient!r}"
        )
    active_mask_gen = gen.new_ones(B, C_g)
    active_mask_history = gen.new_ones(B, C_h)
    active_parts = [active_mask_neg, active_mask_history, active_mask_pos]
    if include_query_targets:
        active_parts.insert(0, active_mask_gen)
    targets_active = torch.cat(active_parts, dim=1)

    # Keep the baseline FP32 arithmetic, but intentionally retain autograd on
    # all current query/positive/negative feature tensors.
    query = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    historical_gen = historical_gen.detach().float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()
    weight_history = weight_history.float()

    target_weight_parts = [weight_neg, weight_history, weight_pos]
    if include_query_targets:
        target_weight_parts.insert(0, weight_gen)
    targets_w = torch.cat(target_weight_parts, dim=1) * targets_active

    # This deliberately mirrors drift_loss_imagenet.  In particular, replay
    # targets and non-unit current-gen force weights do not recalibrate scale.
    scale_weight_parts = [weight_neg, weight_pos]
    if include_query_targets:
        scale_weight_parts.insert(0, torch.ones_like(weight_gen))
    scale_targets_w = torch.cat(scale_weight_parts, dim=1)
    if C_h == 0:
        target_parts = [fixed_neg, fixed_pos]
        if include_query_targets:
            target_parts.insert(0, query)
        targets = torch.cat(target_parts, dim=1)
        dist = _cdist_batched(query, targets)
        weighted_dist = dist * scale_targets_w.unsqueeze(1)
    else:
        base_parts = [fixed_neg, fixed_pos]
        if include_query_targets:
            base_parts.insert(0, query)
        base_targets = torch.cat(base_parts, dim=1)
        dist_base = _cdist_batched(query, base_targets)
        base_split_idx = (C_g if include_query_targets else 0) + C_n
        dist_history = _cdist_batched(query, historical_gen)
        dist = torch.cat(
            [
                dist_base[:, :, :base_split_idx],
                dist_history,
                dist_base[:, :, base_split_idx:],
            ],
            dim=2,
        )
        weighted_dist = dist_base * scale_targets_w.unsqueeze(1)

    # _ratio_of_means detaches both inputs (and all-reduces when requested).
    if distance_scale is None:
        scale = _ratio_of_means(
            weighted_dist,
            scale_targets_w,
            use_global_stats=global_scale_stats,
        )
    else:
        scale = torch.as_tensor(
            distance_scale, device=query.device, dtype=torch.float32
        ).detach()
        if scale.numel() != 1:
            raise ValueError("raw drift distance_scale must be scalar")
    scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
    dist_normed_clean = dist / scale.clamp(min=1e-3)

    if include_query_targets:
        diag = torch.eye(C_g, dtype=torch.float32, device=query.device)
        block_mask = F.pad(diag, (0, C_n + C_h + C_p)).unsqueeze(0)
    else:
        block_mask = query.new_zeros(B, C_g, C_n + C_h + C_p)
    split_idx = (C_g if include_query_targets else 0) + C_n + C_h
    local_bandwidth = None
    if kernel_adaptive_k_pos > 0:
        if not include_query_targets:
            raise ValueError(
                "support-only raw fields do not yet support adaptive bandwidth"
            )
        if C_h > 0:
            bandwidth_dist = dist_base / scale.clamp(min=1e-3)
            bandwidth_mask = F.pad(diag, (0, C_n + C_p)).unsqueeze(0)
            bandwidth_split_idx = C_g + C_n
        else:
            bandwidth_dist = dist_normed_clean
            bandwidth_mask = block_mask
            bandwidth_split_idx = split_idx
        local_bandwidth = _adaptive_reverse_bandwidth(
            bandwidth_dist,
            split_idx=bandwidth_split_idx,
            self_mask=bandwidth_mask,
            k_pos=kernel_adaptive_k_pos,
            k_neg=kernel_adaptive_k_neg,
            margin=kernel_adaptive_margin,
        )
    dist_normed = dist_normed_clean + block_mask * 100.0

    fields = []
    for R in R_list:
        affinity = _reverse_mutual_affinity(
            dist_normed,
            bandwidth=R,
            kernel=affinity_kernel,
            shape=kernel_shape,
            local_bandwidth=local_bandwidth,
            self_mask=block_mask,
            mix_weight=kernel_mix_weight,
            temperature_mix=kernel_temperature_mix,
            temperature_mix_weights=kernel_temperature_mix_weights,
        )
        affinity = affinity * targets_w.unsqueeze(1)

        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        aff_pos, pos_indices = _truncate_force_group(
            aff_pos,
            top_p=top_p,
            top_p_min_keep=top_p_min_keep,
            top_k=top_k_pos,
        )
        aff_neg, neg_indices = _truncate_force_group(
            aff_neg,
            top_p=top_p,
            top_p_min_keep=top_p_min_keep,
            top_k=top_k_neg,
        )

        sum_pos = aff_pos.sum(dim=2, keepdim=True)
        r_coeff_neg = -aff_neg * sum_pos
        if repulsion_coefficient != 1.0:
            r_coeff_neg = r_coeff_neg * repulsion_coefficient
        sum_neg = aff_neg.sum(dim=2, keepdim=True)
        r_coeff_pos = aff_pos * sum_neg

        if C_h > 0:
            fixed_repulsive = (
                torch.cat([fixed_neg, historical_gen], dim=1)
                if C_n > 0
                else historical_gen
            )
        else:
            fixed_repulsive = fixed_neg if C_n > 0 else None
        if include_query_targets:
            total_force_R = _accumulate_differentiable_weighted_targets(
                r_coeff_neg,
                neg_indices,
                query,
                fixed_repulsive,
            )
        else:
            if fixed_repulsive is None:
                raise ValueError(
                    "support-only raw fields require a non-empty repulsive bank"
                )
            total_force_R = _accumulate_differentiable_weighted_targets(
                r_coeff_neg,
                neg_indices,
                fixed_repulsive,
                None,
            )
        total_force_R = _accumulate_differentiable_weighted_targets(
            r_coeff_pos,
            pos_indices,
            fixed_pos,
            out=total_force_R,
        )
        total_coeffs = r_coeff_neg.sum(dim=2) + r_coeff_pos.sum(dim=2)
        total_force_R.addcmul_(
            query,
            total_coeffs.unsqueeze(-1),
            value=-1.0,
        )
        total_force_R.div_(scale_inputs)
        fields.append(total_force_R)

    return tuple(fields)


def drift_loss_imagenet(
    gen: torch.Tensor,                    # [B, C_g, S]
    fixed_pos: torch.Tensor,              # [B, C_p, S]
    fixed_neg: Optional[torch.Tensor] = None,  # [B, C_n, S]
    weight_gen: Optional[torch.Tensor] = None,  # [B, C_g]
    weight_pos: Optional[torch.Tensor] = None,  # [B, C_p]
    weight_neg: Optional[torch.Tensor] = None,  # [B, C_n]
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    compute_wpos_stats: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,  # [B, C_p] 1=active, 0=excluded
    active_mask_neg: Optional[torch.Tensor] = None,  # [B, C_n] 1=active, 0=excluded
    global_scale_stats: bool = True,
    global_fnorm_stats: bool = True,
    top_p: float = 1.0,
    top_p_min_keep: int = 1,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
    force_multiplier: float = 1.0,
    affinity_kernel: str = "exponential",
    kernel_shape: float = 1.0,
    kernel_adaptive_k_pos: int = 0,
    kernel_adaptive_k_neg: int = 0,
    kernel_adaptive_margin: float = 1.05,
    kernel_mix_weight: float = 0.5,
    kernel_temperature_mix: Tuple[float, ...] = (),
    kernel_temperature_mix_weights: Tuple[float, ...] = (),
    historical_gen: Optional[torch.Tensor] = None,  # [B, C_h, S], detached replay
    weight_history: Optional[torch.Tensor] = None,  # [B, C_h]
    collect_diagnostics: bool = True,
    fuse_fnorm_across_R: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Official drift loss (ImageNet version) ported from JAX to PyTorch.

    Returns:
        loss:  Per-batch loss tensor [B] (same interface as official JAX code).
        info:  dict with 'scale' and 'loss_{R}' scalar logging stats when
               ``collect_diagnostics`` is enabled.

    ``fuse_fnorm_across_R`` trades VRAM for fewer distributed collectives: all
    per-temperature force tensors remain resident until their FP64 sum-square
    and count pairs can be reduced in one packed vector.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    (
        affinity_kernel,
        kernel_shape,
        kernel_adaptive_k_pos,
        kernel_adaptive_k_neg,
        kernel_adaptive_margin,
    ) = _validate_reverse_affinity_kernel(
        affinity_kernel,
        kernel_shape,
        kernel_adaptive_k_pos,
        kernel_adaptive_k_neg,
        kernel_adaptive_margin,
    )
    kernel_mix_weight = float(kernel_mix_weight)
    if not math.isfinite(kernel_mix_weight) or not 0.0 <= kernel_mix_weight <= 1.0:
        raise ValueError(
            "reverse kernel mix weight must be finite and in [0, 1], got "
            f"{kernel_mix_weight!r}"
        )

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]
    if historical_gen is None:
        historical_gen = gen.new_zeros(B, 0, S)
    if historical_gen.ndim != 3 or historical_gen.shape[0] != B or historical_gen.shape[2] != S:
        raise ValueError(
            "historical_gen must have shape [B, C_h, S] matching gen; got "
            f"{tuple(historical_gen.shape)} for gen {tuple(gen.shape)}"
        )
    C_h = historical_gen.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)
    if weight_history is None:
        weight_history = gen.new_ones(B, C_h)

    # active_mask: 1 for slots that participate, 0 for slots excluded (e.g., cfg-dependent
    # per-sample pos/neg count). Mirrors colwise version_b semantics: zero contribution
    # post-softmax, but inactive slots still appear in the softmax denominator.
    if active_mask_pos is None:
        active_mask_pos = gen.new_ones(B, C_p)
    else:
        active_mask_pos = active_mask_pos.to(gen.device).float()
    if active_mask_neg is None:
        active_mask_neg = gen.new_ones(B, C_n)
    else:
        active_mask_neg = active_mask_neg.to(gen.device).float()
    active_mask_gen = gen.new_ones(B, C_g)
    active_mask_history = gen.new_ones(B, C_h)
    targets_active = torch.cat(
        [active_mask_gen, active_mask_neg, active_mask_history, active_mask_pos],
        dim=1,
    )

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    historical_gen = historical_gen.detach().float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()
    weight_history = weight_history.float()

    old_gen = gen.detach()
    targets_w = torch.cat(
        [weight_gen, weight_neg, weight_history, weight_pos], dim=1
    ) * targets_active

    # Scale and adaptive bandwidth deliberately use only the original
    # [current gen | real negative | real positive] pool, with unit current-gen
    # mass. This keeps temperature calibration fixed when an ablation changes
    # current-gen force weight or adds replay candidates.
    scale_targets_w = torch.cat(
        [torch.ones_like(weight_gen), weight_neg, weight_pos], dim=1
    )
    if C_h == 0:
        targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        dist = _cdist_batched(old_gen, targets)
        weighted_dist = dist * scale_targets_w.unsqueeze(1)
    else:
        base_targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        dist_base = _cdist_batched(old_gen, base_targets)
        base_split_idx = C_g + C_n
        dist_history = _cdist_batched(old_gen, historical_gen)
        dist = torch.cat(
            [
                dist_base[:, :, :base_split_idx],
                dist_history,
                dist_base[:, :, base_split_idx:],
            ],
            dim=2,
        )
        # The original generated pool has unit mass for scale estimation even
        # when part of its force mass is reassigned to historical particles.
        weighted_dist = dist_base * scale_targets_w.unsqueeze(1)
    scale = _ratio_of_means(
        weighted_dist,
        scale_targets_w,
        use_global_stats=global_scale_stats,
    )

    scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
    old_gen_scaled = old_gen / scale_inputs

    dist_normed_clean = dist / scale.clamp(min=1e-3)

    # Diagonal self-mask applies only to the current generated block. Historical
    # particles are independent detached targets and must remain eligible.
    diag = torch.eye(C_g, dtype=torch.float32, device=gen.device)   # [C_g, C_g]
    block_mask = F.pad(diag, (0, C_n + C_h + C_p)).unsqueeze(0)
    local_bandwidth = None
    split_idx = C_g + C_n + C_h
    if kernel_adaptive_k_pos > 0:
        if C_h > 0:
            bandwidth_dist = dist_base / scale.clamp(min=1e-3)
            bandwidth_mask = F.pad(diag, (0, C_n + C_p)).unsqueeze(0)
            bandwidth_split_idx = C_g + C_n
        else:
            bandwidth_dist = dist_normed_clean
            bandwidth_mask = block_mask
            bandwidth_split_idx = split_idx
        local_bandwidth = _adaptive_reverse_bandwidth(
            bandwidth_dist,
            split_idx=bandwidth_split_idx,
            self_mask=bandwidth_mask,
            k_pos=kernel_adaptive_k_pos,
            k_neg=kernel_adaptive_k_neg,
            margin=kernel_adaptive_margin,
        )
    dist_normed = dist_normed_clean + block_mask * 100.0

    info: Dict[str, float] = {}
    if collect_diagnostics:
        info["scale"] = float(scale.item())
        if C_h > 0:
            info["history/count"] = float(C_h)
            info["history/current_mass"] = float(
                weight_gen.sum(dim=1).mean().item()
            )
            info["history/replay_mass"] = float(
                weight_history.sum(dim=1).mean().item()
            )
        if local_bandwidth is not None:
            info["kernel_bandwidth_mean"] = float(
                local_bandwidth.mean().item()
            )
    old_gen_scaled_goal = old_gen_scaled
    scale_inputs_goal = scale_inputs
    force_across_R = torch.zeros_like(old_gen_scaled_goal)
    fuse_global_fnorm = bool(fuse_fnorm_across_R) and global_fnorm_stats
    forces_pending_fnorm = []

    for R in R_list:
        affinity = _reverse_mutual_affinity(
            dist_normed,
            bandwidth=R,
            kernel=affinity_kernel,
            shape=kernel_shape,
            local_bandwidth=local_bandwidth,
            self_mask=block_mask,
            mix_weight=kernel_mix_weight,
            temperature_mix=kernel_temperature_mix,
            temperature_mix_weights=kernel_temperature_mix_weights,
        )
        affinity = affinity * targets_w.unsqueeze(1)         # weight by sample weights

        aff_neg = affinity[:, :, :split_idx]                 # [B, C_g, C_g+C_n]
        aff_pos = affinity[:, :, split_idx:]                 # [B, C_g, C_p]
        # Select the positive and [old_gen | real-negative] centroids
        # independently, retaining each group's pre-top-p total mass.
        aff_pos, pos_indices = _truncate_force_group(
            aff_pos,
            top_p=top_p,
            top_p_min_keep=top_p_min_keep,
            top_k=top_k_pos,
        )
        aff_neg, neg_indices = _truncate_force_group(
            aff_neg,
            top_p=top_p,
            top_p_min_keep=top_p_min_keep,
            top_k=top_k_neg,
        )

        sum_pos = aff_pos.sum(dim=2, keepdim=True)           # [B, C_g, 1]
        r_coeff_neg = -aff_neg * sum_pos                     # attract from neg: repel

        sum_neg = aff_neg.sum(dim=2, keepdim=True)           # [B, C_g, 1]
        r_coeff_pos = aff_pos * sum_neg                      # attract toward pos

        if collect_diagnostics and compute_wpos_stats and R == R_list[0]:
            with torch.no_grad():
                info.update(
                    _wpos_stats_from_matrix(
                        _dense_weights_for_stats(
                            r_coeff_pos, pos_indices, C_p
                        )
                    )
                )
            if int(top_k_pos) > 0 or int(top_k_neg) > 0:
                _record_top_k_support(
                    info,
                    aff_pos,
                    pos_indices,
                    C_p,
                    aff_neg,
                    neg_indices,
                    split_idx,
                )
            else:
                _record_top_p_support(info, aff_pos, aff_neg)

        # Compact top-k uses a fused indexed reduction. Dense/top-p retains the
        # original bmm path. Both are accumulated in raw feature units before
        # the common scale is applied.
        if C_h > 0:
            fixed_repulsive = (
                torch.cat([fixed_neg, historical_gen], dim=1)
                if C_n > 0
                else historical_gen
            )
        else:
            fixed_repulsive = fixed_neg if C_n > 0 else None
        total_force_R = _accumulate_weighted_targets(
            r_coeff_neg,
            neg_indices,
            old_gen,
            fixed_repulsive,
        )
        _accumulate_weighted_targets(
            r_coeff_pos,
            pos_indices,
            fixed_pos,
            out=total_force_R,
        )
        total_coeffs = r_coeff_neg.sum(dim=2) + r_coeff_pos.sum(dim=2)
        total_force_R.addcmul_(
            old_gen,
            total_coeffs.unsqueeze(-1),
            value=-1.0,
        )
        total_force_R.div_(scale_inputs)

        if fuse_global_fnorm:
            forces_pending_fnorm.append(total_force_R)
        else:
            f_norm_val = _mean_square_detached(
                total_force_R,
                use_global_stats=global_fnorm_stats,
            )
            if collect_diagnostics:
                info[f"loss_{R}"] = float(f_norm_val.item())

            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force_R / force_scale

    if forces_pending_fnorm:
        fused_fnorm_values = _global_mean_squares_detached_fused(
            tuple(forces_pending_fnorm)
        )
        for R, total_force_R, f_norm_val in zip(
            R_list,
            forces_pending_fnorm,
            fused_fnorm_values,
        ):
            if collect_diagnostics:
                info[f"loss_{R}"] = float(f_norm_val.item())
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force_R / force_scale

    force_multiplier_f = float(force_multiplier)
    if not math.isfinite(force_multiplier_f) or force_multiplier_f < 0.0:
        raise ValueError(
            f"force_multiplier must be finite and >= 0, got {force_multiplier!r}"
        )
    if force_multiplier_f != 1.0:
        force_across_R.mul_(force_multiplier_f)
    if collect_diagnostics:
        info["force_multiplier"] = force_multiplier_f
    goal_scaled = (old_gen_scaled_goal + force_across_R).detach()
    gen_scaled = gen / scale_inputs_goal
    diff = gen_scaled - goal_scaled
    # Match JAX interface: return per-batch loss, caller decides reduction.
    loss = (diff ** 2).mean(dim=(-1, -2))

    return loss, info


def drift_loss_imagenet_colwise(
    gen: torch.Tensor,                    # [B, C_g, S]
    fixed_pos: torch.Tensor,              # [B, C_p, S]
    fixed_neg: Optional[torch.Tensor] = None,  # [B, C_n, S]
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    coupling: bool = False,
    compute_wpos_stats: bool = False,
    per_sample_fnorm: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,  # [B, C_p] bool/float (1=active)
    active_mask_neg: Optional[torch.Tensor] = None,  # [B, C_n] bool/float (1=active)
    decouple_weight_from_coupling: bool = False,
    self_mask_on_raw: bool = False,
    global_scale_stats: bool = True,
    global_fnorm_stats: bool = True,
    top_p: float = 1.0,
    top_p_min_keep: int = 1,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Col-wise softmax drift loss. Rows = gen, cols = pos / neg, but softmax
    is taken along dim=1 (over gen) so each pos/neg gets budget=1.

    Version A  (coupling=False):
        A_pos = softmax(logit_pos, dim=1)   # [B, C_g, C_p]
        A_neg = softmax(logit_neg, dim=1)   # [B, C_g, C_gn]
        W_pos = A_pos
        W_neg = A_neg

    Version B  (coupling=True):
        same A_pos, A_neg as above, then
        W_pos = A_pos * A_neg.sum(dim=2, keepdim=True)   # scale by how much neg weight gen i gets
        W_neg = A_neg * A_pos.sum(dim=2, keepdim=True)   # scale by how much pos weight gen i gets

    Both preserve anti-symmetry (attraction - repulsion) while giving every
    positive a guaranteed training signal (column budget = 1 regardless of
    how many gen samples are nearby). Version B additionally preserves the
    pos/neg coupling of the baseline so the force scale is consistent.

    neg targets = [old_gen | fixed_neg], same as pos-centric baselines.
    Self-diagonal mask applied so gen[i] does not repel itself.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen        = gen.float()
    fixed_pos  = fixed_pos.float()
    fixed_neg  = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    # active_mask: 1 for slots that participate, 0 for slots that are excluded
    # (per-sample variable active count; weights themselves remain all 1s)
    if active_mask_pos is None:
        active_mask_pos = gen.new_ones(B, C_p)
    else:
        active_mask_pos = active_mask_pos.to(gen.device).float()
    if active_mask_neg is None:
        active_mask_neg = gen.new_ones(B, C_n)
    else:
        active_mask_neg = active_mask_neg.to(gen.device).float()
    # old_gen block (first C_g cols of neg_targets) is always active
    active_mask_gen = gen.new_ones(B, C_g)
    neg_targets_active = torch.cat([active_mask_gen, active_mask_neg], dim=1)  # [B, C_g+C_n]
    all_targets_active = torch.cat([active_mask_pos, neg_targets_active], dim=1)  # [B, C_p+C_g+C_n]

    old_gen = gen.detach()

    # colwise: pos/neg를 분리된 거리 행렬로 계산 — baseline은 [gen|neg|pos] 합친 단일 행렬
    neg_targets = torch.cat([old_gen, fixed_neg], dim=1)        # [B, C_g+C_n, S]
    C_gn = C_g + C_n

    dist_pos = _cdist_batched(old_gen, fixed_pos)               # [B, C_g, C_p]
    dist_neg = _cdist_batched(old_gen, neg_targets)             # [B, C_g, C_g+C_n]

    # Self-mask matrix: +100 on diagonal of old_gen block so gen[i] can't repel itself.
    diag = torch.eye(C_g, dtype=torch.float32, device=gen.device)
    neg_diag_mask = F.pad(diag, (0, C_n)) * 100.0                # [C_g, C_g+C_n]

    # Shared scale (baseline-style weighted average over all target columns).
    # self_mask_on_raw controls WHEN the +100 diagonal mask enters:
    #   False (default / current): +100 added to dist_neg_n AFTER scale normalization.
    #                              Scale is computed on clean raw dist.
    #   True  (commit 4f87011):    +100 added to raw dist_neg BEFORE scale computation.
    #                              Scale is "polluted" by the mask (inflated by ~2.08 for
    #                              uniform weights, C_g=16, total cols=48) — this
    #                              dilates scale_inputs and flattens softmax for
    #                              low-scale features (low-D / global-pool), acting as
    #                              an implicit temperature boost.
    if self_mask_on_raw:
        dist_neg_for_scale = dist_neg + neg_diag_mask.unsqueeze(0)
    else:
        dist_neg_for_scale = dist_neg

    neg_targets_w = torch.cat([weight_gen, weight_neg], dim=1) * neg_targets_active   # [B, C_g+C_n]
    eff_weight_pos = weight_pos * active_mask_pos                                      # [B, C_p]
    all_targets_w = torch.cat([eff_weight_pos, neg_targets_w], dim=1)                  # [B, C_p+C_g+C_n]
    weighted_dist_pos = dist_pos * eff_weight_pos.unsqueeze(1)        # [B, C_g, C_p]
    weighted_dist_neg = dist_neg_for_scale * neg_targets_w.unsqueeze(1)  # [B, C_g, C_g+C_n]
    weighted_dist_all = torch.cat([weighted_dist_pos, weighted_dist_neg], dim=2)
    scale = _ratio_of_means(weighted_dist_all, all_targets_w, use_global_stats=global_scale_stats)
    scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)

    dist_pos_n = dist_pos / scale.clamp(min=1e-3)
    if self_mask_on_raw:
        # +100 is already baked into dist_neg_for_scale; dividing by polluted scale
        # propagates it through.
        dist_neg_n = dist_neg_for_scale / scale.clamp(min=1e-3)
    else:
        # Current default: clean normalization, then add +100 at the normalized level.
        dist_neg_n = dist_neg / scale.clamp(min=1e-3)
        dist_neg_n = dist_neg_n + neg_diag_mask.unsqueeze(0)

    old_gen_sc = old_gen / scale_inputs                         # [B, C_g, S]

    info: Dict[str, float] = {"scale": float(scale.item())}
    force_across_R = torch.zeros_like(old_gen_sc)

    for R in R_list:
        logit_pos = -dist_pos_n / R                             # [B, C_g, C_p]
        logit_neg = -dist_neg_n / R                             # [B, C_g, C_g+C_n]

        # colwise: dim=1(gen 축)으로 softmax → 각 pos/neg가 gen들에게 budget=1 보장
        # baseline은 dim=2(target 축)로 softmax → gen이 budget=1 가짐 (pos가 많으면 gradient 묽어짐)
        A_pos_raw = F.softmax(logit_pos, dim=1)                 # [B, C_g, C_p]
        A_neg_raw = F.softmax(logit_neg, dim=1)                 # [B, C_g, C_g+C_n]
        forward_top_k = int(top_k_pos) > 0 or int(top_k_neg) > 0
        if forward_top_k and float(top_p) < 1.0:
            raise ValueError(
                "top-p and top-k drift truncation cannot be enabled together"
            )
        # Native forward-drift sparsity: every target column chooses generated
        # rows. Preserve each column's softmax budget before applying target
        # weights and before computing positive/negative coupling masses.
        A_pos_raw, pos_column_indices = _column_top_k_preserve_mass(
            A_pos_raw, top_k=top_k_pos
        )
        A_neg_raw, neg_column_indices = _column_top_k_preserve_mass(
            A_neg_raw, top_k=top_k_neg
        )
        # Apply per-column weights like baseline targets_w (CFG-derived weight_neg included).
        # active_mask excludes inactive slots (0 contribution). Weights themselves are all 1 by default.
        A_pos = A_pos_raw * eff_weight_pos.unsqueeze(1)
        A_neg = A_neg_raw * neg_targets_w.unsqueeze(1)

        if coupling:
            # version b: gen i가 받는 neg 총량으로 pos weight 스케일, 반대도 동일
            # → pos/neg force가 서로 연동되어 baseline의 coupling 특성 유지
            if decouple_weight_from_coupling:
                # 가중치(특히 CFG-기반 weight_neg)가 coupling mass 로 전파되는 것을 차단.
                # m 은 raw softmax 로만 계산 → W_pos 가 weight_neg 배율로 부풀지 않음.
                m_neg = A_neg_raw.sum(dim=2, keepdim=True)
                m_pos = A_pos_raw.sum(dim=2, keepdim=True)
            else:
                m_neg = A_neg.sum(dim=2, keepdim=True)          # gen i가 받는 neg 총 질량
                m_pos = A_pos.sum(dim=2, keepdim=True)          # gen i가 받는 pos 총 질량
            W_pos = A_pos * m_neg                               # [B, C_g, C_p]
            W_neg = A_neg * m_pos                               # [B, C_g, C_g+C_n]
        else:
            # version a: coupling 없이 A_pos/A_neg 그대로 사용
            W_pos = A_pos
            W_neg = A_neg

        if forward_top_k:
            # Column selection is already complete. Coupling needs dense row
            # sums, and dense bmm is faster for these 64x(128+96) grids.
            pos_indices = None
            neg_indices = None
        else:
            # Top-p remains the older row-wise centroid truncation experiment.
            W_pos, pos_indices = _truncate_force_group(
                W_pos,
                top_p=top_p,
                top_p_min_keep=top_p_min_keep,
                top_k=0,
            )
            W_neg, neg_indices = _truncate_force_group(
                W_neg,
                top_p=top_p,
                top_p_min_keep=top_p_min_keep,
                top_k=0,
            )

        if compute_wpos_stats and R == R_list[0]:
            with torch.no_grad():
                info.update(
                    _wpos_stats_from_matrix(
                        _dense_weights_for_stats(W_pos, pos_indices, C_p)
                    )
                )
            if forward_top_k:
                _record_column_top_k_support(
                    info,
                    A_pos,
                    pos_column_indices,
                    A_neg,
                    neg_column_indices,
                )
            else:
                _record_top_p_support(info, W_pos, W_neg)

        total_force = _accumulate_weighted_targets(
            W_pos,
            pos_indices,
            fixed_pos,
        )
        _accumulate_weighted_targets(
            W_neg,
            neg_indices,
            old_gen,
            fixed_neg if C_n > 0 else None,
            out=total_force,
            alpha=-1.0,
        )
        total_coeffs = W_pos.sum(dim=2) - W_neg.sum(dim=2)
        total_force.addcmul_(
            old_gen,
            total_coeffs.unsqueeze(-1),
            value=-1.0,
        )
        total_force.div_(scale_inputs)

        if per_sample_fnorm:
            # Per-sample f_norm: each sample normalized by its own force magnitude,
            # so high-cfg samples don't drown out low-cfg ones in batch normalization.
            f_norm_val = (total_force ** 2).mean(dim=(-1, -2), keepdim=True)  # [B, 1, 1]
            info[f"loss_{R}"] = float(_mean_detached(f_norm_val, use_global_stats=global_fnorm_stats).item())
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force / force_scale
        else:
            f_norm_val = _mean_square_detached(total_force, use_global_stats=global_fnorm_stats)
            info[f"loss_{R}"] = float(f_norm_val.item())
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force / force_scale

    goal_scaled = (old_gen_sc + force_across_R).detach()
    gen_scaled  = gen / scale_inputs
    loss = ((gen_scaled - goal_scaled) ** 2).mean(dim=(-1, -2))
    return loss, info


def _compute_force_baseline_grid(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    compute_wpos_stats: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,
    active_mask_neg: Optional[torch.Tensor] = None,
    dist_precomputed: Optional[torch.Tensor] = None,
    dist_normed_precomputed: Optional[torch.Tensor] = None,
    scale_precomputed: Optional[torch.Tensor] = None,
    global_scale_stats: bool = True,
    global_fnorm_stats: bool = True,
    collect_diagnostics: bool = True,
    top_p: float = 1.0,
    top_p_min_keep: int = 1,
    compute_top_p_stats: bool = False,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Helper: same math as drift_loss_imagenet but returns the unscaled drift
    force instead of forming the loss. Used by drift_loss_imagenet_mixed.

    Returns:
        old_gen:      [B, C_g, S]   detached
        V_unscaled:   [B, C_g, S]   detached, in data-space units
        scale_inputs: scalar tensor (detached)
        info:         dict
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]
    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)
    if active_mask_pos is None:
        active_mask_pos = gen.new_ones(B, C_p)
    else:
        active_mask_pos = active_mask_pos.to(gen.device).float()
    if active_mask_neg is None:
        active_mask_neg = gen.new_ones(B, C_n)
    else:
        active_mask_neg = active_mask_neg.to(gen.device).float()
    active_mask_gen = gen.new_ones(B, C_g)
    targets_active = torch.cat([active_mask_gen, active_mask_neg, active_mask_pos], dim=1)

    gen_f      = gen.float()
    fixed_pos  = fixed_pos.float()
    fixed_neg  = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen_f.detach()
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1) * targets_active

    using_shared_normed_dist = dist_normed_precomputed is not None
    if using_shared_normed_dist != (scale_precomputed is not None):
        raise ValueError(
            "dist_normed_precomputed and scale_precomputed must be provided together"
        )
    if using_shared_normed_dist and dist_precomputed is not None:
        raise ValueError(
            "dist_precomputed cannot be combined with dist_normed_precomputed"
        )

    if using_shared_normed_dist:
        expected_shape = (B, C_g, C_g + C_n + C_p)
        if tuple(dist_normed_precomputed.shape) != expected_shape:
            raise ValueError(
                "dist_normed_precomputed has shape "
                f"{tuple(dist_normed_precomputed.shape)}, expected {expected_shape}"
            )
        # The shared matrix is already scale-normalized and has the generated
        # self-diagonal masked. Keep it read-only so the forward branch can use
        # views of the same storage before this helper consumes the joint grid.
        dist_normed = dist_normed_precomputed
        scale = scale_precomputed
        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
    else:
        if dist_precomputed is not None:
            dist = dist_precomputed
        else:
            # Build only the comparatively small distance matrix. Concatenating
            # [old_gen | fixed_neg | fixed_pos] along the feature dimension can be
            # several GiB for high-resolution MAE activations.
            dist_parts = [_cdist_batched(old_gen, old_gen)]
            if C_n > 0:
                dist_parts.append(_cdist_batched(old_gen, fixed_neg))
            dist_parts.append(_cdist_batched(old_gen, fixed_pos))
            dist = torch.cat(dist_parts, dim=2)
        weighted_dist = dist * targets_w.unsqueeze(1)
        scale = _ratio_of_means(weighted_dist, targets_w, use_global_stats=global_scale_stats)
        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)

        dist_normed = dist / scale.clamp(min=1e-3)
        diag = torch.eye(C_g, dtype=torch.float32, device=gen.device)
        block_mask = F.pad(diag, (0, C_n + C_p)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * 100.0
        del dist, weighted_dist

    info: Dict[str, float] = {}
    if collect_diagnostics:
        info["scale"] = float(scale.item())
    force_across_R: Optional[torch.Tensor] = None

    for R in R_list:
        logits = -dist_normed / R
        aff_row = F.softmax(logits, dim=2)
        aff_col = F.softmax(logits, dim=1)
        affinity = (aff_row * aff_col).clamp(min=1e-6).sqrt()
        affinity = affinity * targets_w.unsqueeze(1)

        split_idx = C_g + C_n
        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        aff_pos, pos_indices = _truncate_force_group(
            aff_pos,
            top_p=top_p,
            top_p_min_keep=top_p_min_keep,
            top_k=top_k_pos,
        )
        aff_neg, neg_indices = _truncate_force_group(
            aff_neg,
            top_p=top_p,
            top_p_min_keep=top_p_min_keep,
            top_k=top_k_neg,
        )

        sum_pos = aff_pos.sum(dim=2, keepdim=True)
        r_coeff_neg = -aff_neg * sum_pos
        sum_neg = aff_neg.sum(dim=2, keepdim=True)
        r_coeff_pos = aff_pos * sum_neg

        if compute_wpos_stats and R == R_list[0]:
            with torch.no_grad():
                info.update(
                    _wpos_stats_from_matrix(
                        _dense_weights_for_stats(
                            r_coeff_pos, pos_indices, C_p
                        )
                    )
                )
        if compute_top_p_stats and R == R_list[0]:
            if int(top_k_pos) > 0 or int(top_k_neg) > 0:
                _record_top_k_support(
                    info,
                    aff_pos,
                    pos_indices,
                    C_p,
                    aff_neg,
                    neg_indices,
                    split_idx,
                )
            else:
                _record_top_p_support(info, aff_pos, aff_neg)

        # Algebraically identical to bmm(cat(coeffs), cat(targets) / scale),
        # without materializing either multi-GiB concatenated target tensor.
        total_force_R = _accumulate_weighted_targets(
            r_coeff_neg,
            neg_indices,
            old_gen,
            fixed_neg if C_n > 0 else None,
        )
        _accumulate_weighted_targets(
            r_coeff_pos,
            pos_indices,
            fixed_pos,
            out=total_force_R,
        )
        total_coeffs = r_coeff_neg.sum(dim=2) + r_coeff_pos.sum(dim=2)
        total_force_R.addcmul_(
            old_gen,
            total_coeffs.unsqueeze(-1),
            value=-1.0,
        )
        total_force_R.div_(scale_inputs)

        f_norm_val = _mean_square_detached(total_force_R, use_global_stats=global_fnorm_stats)
        if collect_diagnostics:
            info[f"loss_{R}"] = float(f_norm_val.item())
        force_scale = f_norm_val.clamp(min=1e-8).sqrt()
        if force_across_R is None:
            total_force_R.div_(force_scale)
            force_across_R = total_force_R
        else:
            force_across_R.addcdiv_(total_force_R, force_scale)

    assert force_across_R is not None
    force_across_R.mul_(scale_inputs)
    V_unscaled = force_across_R.detach()
    return old_gen, V_unscaled, scale_inputs.detach(), info


def _compute_force_colwise(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    coupling: bool = False,
    compute_wpos_stats: bool = False,
    per_sample_fnorm: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,
    active_mask_neg: Optional[torch.Tensor] = None,
    decouple_weight_from_coupling: bool = False,
    self_mask_on_raw: bool = False,
    dist_pos_precomputed: Optional[torch.Tensor] = None,
    dist_neg_precomputed: Optional[torch.Tensor] = None,
    dist_pos_normed_precomputed: Optional[torch.Tensor] = None,
    dist_neg_normed_precomputed: Optional[torch.Tensor] = None,
    scale_precomputed: Optional[torch.Tensor] = None,
    global_scale_stats: bool = True,
    global_fnorm_stats: bool = True,
    collect_diagnostics: bool = True,
    top_p: float = 1.0,
    top_p_min_keep: int = 1,
    compute_top_p_stats: bool = False,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Helper: same math as drift_loss_imagenet_colwise but returns the
    unscaled drift force instead of forming the loss.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None: weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None: weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None: weight_neg = gen.new_ones(B, C_n)

    gen_f      = gen.float()
    fixed_pos  = fixed_pos.float()
    fixed_neg  = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    if active_mask_pos is None:
        active_mask_pos = gen.new_ones(B, C_p)
    else:
        active_mask_pos = active_mask_pos.to(gen.device).float()
    if active_mask_neg is None:
        active_mask_neg = gen.new_ones(B, C_n)
    else:
        active_mask_neg = active_mask_neg.to(gen.device).float()
    active_mask_gen = gen.new_ones(B, C_g)
    neg_targets_active = torch.cat([active_mask_gen, active_mask_neg], dim=1)

    old_gen = gen_f.detach()

    neg_targets_w = torch.cat([weight_gen, weight_neg], dim=1) * neg_targets_active
    eff_weight_pos = weight_pos * active_mask_pos
    shared_args = (
        dist_pos_normed_precomputed,
        dist_neg_normed_precomputed,
        scale_precomputed,
    )
    using_shared_normed_dist = all(value is not None for value in shared_args)
    if not using_shared_normed_dist and any(value is not None for value in shared_args):
        raise ValueError(
            "dist_pos_normed_precomputed, dist_neg_normed_precomputed, and "
            "scale_precomputed must be provided together"
        )
    if using_shared_normed_dist and (
        dist_pos_precomputed is not None or dist_neg_precomputed is not None
    ):
        raise ValueError(
            "raw and normalized precomputed distances cannot be combined"
        )
    if using_shared_normed_dist and self_mask_on_raw:
        raise ValueError(
            "shared normalized distances require self_mask_on_raw=False"
        )

    if using_shared_normed_dist:
        expected_pos_shape = (B, C_g, C_p)
        expected_neg_shape = (B, C_g, C_g + C_n)
        if tuple(dist_pos_normed_precomputed.shape) != expected_pos_shape:
            raise ValueError(
                "dist_pos_normed_precomputed has shape "
                f"{tuple(dist_pos_normed_precomputed.shape)}, "
                f"expected {expected_pos_shape}"
            )
        if tuple(dist_neg_normed_precomputed.shape) != expected_neg_shape:
            raise ValueError(
                "dist_neg_normed_precomputed has shape "
                f"{tuple(dist_neg_normed_precomputed.shape)}, "
                f"expected {expected_neg_shape}"
            )
        # These are read-only views into the joint normalized matrix prepared
        # by drift_loss_imagenet_mixed. The negative view already includes the
        # generated self-diagonal mask.
        dist_pos_n = dist_pos_normed_precomputed
        dist_neg_n = dist_neg_normed_precomputed
        scale = scale_precomputed
        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
    else:
        dist_pos = dist_pos_precomputed if dist_pos_precomputed is not None else _cdist_batched(old_gen, fixed_pos)
        if dist_neg_precomputed is not None:
            dist_neg = dist_neg_precomputed
        else:
            dist_neg_parts = [_cdist_batched(old_gen, old_gen)]
            if C_n > 0:
                dist_neg_parts.append(_cdist_batched(old_gen, fixed_neg))
            dist_neg = torch.cat(dist_neg_parts, dim=2)

        diag = torch.eye(C_g, dtype=torch.float32, device=gen.device)
        neg_diag_mask = F.pad(diag, (0, C_n)) * 100.0

        if self_mask_on_raw:
            dist_neg_for_scale = dist_neg + neg_diag_mask.unsqueeze(0)
        else:
            dist_neg_for_scale = dist_neg

        all_targets_w = torch.cat([eff_weight_pos, neg_targets_w], dim=1)
        weighted_dist_pos = dist_pos * eff_weight_pos.unsqueeze(1)
        weighted_dist_neg = dist_neg_for_scale * neg_targets_w.unsqueeze(1)
        weighted_dist_all = torch.cat([weighted_dist_pos, weighted_dist_neg], dim=2)
        scale = _ratio_of_means(weighted_dist_all, all_targets_w, use_global_stats=global_scale_stats)
        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
        del weighted_dist_pos, weighted_dist_neg, weighted_dist_all, all_targets_w

        dist_pos_n = dist_pos / scale.clamp(min=1e-3)
        if self_mask_on_raw:
            dist_neg_n = dist_neg_for_scale / scale.clamp(min=1e-3)
        else:
            dist_neg_n = dist_neg / scale.clamp(min=1e-3)
            dist_neg_n = dist_neg_n + neg_diag_mask.unsqueeze(0)
        del dist_pos, dist_neg, dist_neg_for_scale

    info: Dict[str, float] = {}
    if collect_diagnostics:
        info["scale"] = float(scale.item())
    force_across_R: Optional[torch.Tensor] = None

    for R in R_list:
        logit_pos = -dist_pos_n / R
        logit_neg = -dist_neg_n / R

        A_pos_raw = F.softmax(logit_pos, dim=1)
        A_neg_raw = F.softmax(logit_neg, dim=1)
        forward_top_k = int(top_k_pos) > 0 or int(top_k_neg) > 0
        if forward_top_k and float(top_p) < 1.0:
            raise ValueError(
                "top-p and top-k drift truncation cannot be enabled together"
            )
        A_pos_raw, pos_column_indices = _column_top_k_preserve_mass(
            A_pos_raw, top_k=top_k_pos
        )
        A_neg_raw, neg_column_indices = _column_top_k_preserve_mass(
            A_neg_raw, top_k=top_k_neg
        )
        A_pos = A_pos_raw * eff_weight_pos.unsqueeze(1)
        A_neg = A_neg_raw * neg_targets_w.unsqueeze(1)

        if coupling:
            if decouple_weight_from_coupling:
                m_neg = A_neg_raw.sum(dim=2, keepdim=True)
                m_pos = A_pos_raw.sum(dim=2, keepdim=True)
            else:
                m_neg = A_neg.sum(dim=2, keepdim=True)
                m_pos = A_pos.sum(dim=2, keepdim=True)
            W_pos = A_pos * m_neg
            W_neg = A_neg * m_pos
        else:
            W_pos = A_pos
            W_neg = A_neg

        if forward_top_k:
            pos_indices = None
            neg_indices = None
        else:
            W_pos, pos_indices = _truncate_force_group(
                W_pos,
                top_p=top_p,
                top_p_min_keep=top_p_min_keep,
                top_k=0,
            )
            W_neg, neg_indices = _truncate_force_group(
                W_neg,
                top_p=top_p,
                top_p_min_keep=top_p_min_keep,
                top_k=0,
            )

        if compute_wpos_stats and R == R_list[0]:
            with torch.no_grad():
                info.update(
                    _wpos_stats_from_matrix(
                        _dense_weights_for_stats(W_pos, pos_indices, C_p)
                    )
                )
        if compute_top_p_stats and R == R_list[0]:
            if forward_top_k:
                _record_column_top_k_support(
                    info,
                    A_pos,
                    pos_column_indices,
                    A_neg,
                    neg_column_indices,
                )
            else:
                _record_top_p_support(info, W_pos, W_neg)

        # Algebraically identical to bmm(cat([W_pos, -W_neg]), targets / scale),
        # without materializing the multi-GiB concatenated/scaled target.
        total_force = _accumulate_weighted_targets(
            W_pos,
            pos_indices,
            fixed_pos,
        )
        _accumulate_weighted_targets(
            W_neg,
            neg_indices,
            old_gen,
            fixed_neg if C_n > 0 else None,
            out=total_force,
            alpha=-1.0,
        )
        total_coeffs = W_pos.sum(dim=2) - W_neg.sum(dim=2)
        total_force.addcmul_(
            old_gen,
            total_coeffs.unsqueeze(-1),
            value=-1.0,
        )
        total_force.div_(scale_inputs)

        if per_sample_fnorm:
            f_norm_val = (total_force ** 2).mean(dim=(-1, -2), keepdim=True)
            if collect_diagnostics:
                info[f"loss_{R}"] = float(
                    _mean_detached(
                        f_norm_val,
                        use_global_stats=global_fnorm_stats,
                    ).item()
                )
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
        else:
            f_norm_val = _mean_square_detached(total_force, use_global_stats=global_fnorm_stats)
            if collect_diagnostics:
                info[f"loss_{R}"] = float(f_norm_val.item())
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
        if force_across_R is None:
            total_force.div_(force_scale)
            force_across_R = total_force
        else:
            force_across_R.addcdiv_(total_force, force_scale)

    assert force_across_R is not None
    force_across_R.mul_(scale_inputs)
    V_unscaled = force_across_R.detach()
    return old_gen, V_unscaled, scale_inputs.detach(), info


def _prepare_shared_mixed_distances(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor],
    weight_neg: Optional[torch.Tensor],
    active_mask_pos: Optional[torch.Tensor],
    active_mask_neg: Optional[torch.Tensor],
    global_scale_stats: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one normalized, self-masked distance grid for both drift experts.

    The reverse expert consumes the complete ``[gen | neg | pos]`` grid. The
    forward expert consumes read-only views of its ``[gen | neg]`` and ``pos``
    blocks. Preallocating and filling the joint distance tensor avoids keeping
    the per-target cdist results alive alongside a ``torch.cat`` copy. The raw
    grid is normalized in place after the common scale has been computed, so
    sharing does not retain an additional full-size raw matrix during either
    expert's temperature loop.

    This common scale is valid when the generated self-mask is applied after
    scale estimation (``self_mask_on_raw=False``), which is the default and the
    mode used by the dual-drift training configs.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    gen_f = gen.float()
    fixed_pos_f = fixed_pos.float()
    fixed_neg_f = fixed_neg.float()
    old_gen = gen_f.detach()

    if weight_neg is None:
        weight_neg_f = old_gen.new_ones(B, C_n)
    else:
        weight_neg_f = weight_neg.to(device=old_gen.device, dtype=torch.float32)
    if active_mask_pos is None:
        active_mask_pos_f = old_gen.new_ones(B, C_p)
    else:
        active_mask_pos_f = active_mask_pos.to(
            device=old_gen.device, dtype=torch.float32
        )
    if active_mask_neg is None:
        active_mask_neg_f = old_gen.new_ones(B, C_n)
    else:
        active_mask_neg_f = active_mask_neg.to(
            device=old_gen.device, dtype=torch.float32
        )

    # Match the reverse-grid target order. The forward expert later receives
    # views into the same storage, so no distance data is copied or retained
    # twice across the two expert evaluations.
    split_idx = C_g + C_n
    total_targets = split_idx + C_p
    dist_joint = old_gen.new_empty(B, C_g, total_targets)
    with torch.no_grad():
        dist_joint[:, :, :C_g].copy_(_cdist_batched(old_gen, old_gen))
        if C_n > 0:
            dist_joint[:, :, C_g:split_idx].copy_(
                _cdist_batched(old_gen, fixed_neg_f.detach())
            )
        dist_joint[:, :, split_idx:].copy_(
            _cdist_batched(old_gen, fixed_pos_f.detach())
        )

    targets_w = torch.cat(
        [
            old_gen.new_ones(B, C_g),
            weight_neg_f * active_mask_neg_f,
            active_mask_pos_f,
        ],
        dim=1,
    )
    weighted_dist = dist_joint * targets_w.unsqueeze(1)
    scale = _ratio_of_means(
        weighted_dist,
        targets_w,
        use_global_stats=global_scale_stats,
    )
    del weighted_dist, targets_w

    dist_joint.div_(scale.clamp(min=1e-3))
    dist_joint[:, :, :C_g].diagonal(dim1=1, dim2=2).add_(100.0)

    return old_gen, fixed_pos_f, fixed_neg_f, dist_joint, scale.detach()


def drift_loss_imagenet_mixed(
    gen: torch.Tensor,                    # [B, C_g, S]
    fixed_pos: torch.Tensor,              # [B, C_p, S]
    fixed_neg: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    alpha: float = 0.0,                   # 0 → pure version_b, 1 → pure baseline
    R_list_baseline: Tuple[float, ...] = (0.2, 0.05, 0.02),
    R_list_versionb: Tuple[float, ...] = (0.4, 0.10, 0.04),
    compute_wpos_stats: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,
    active_mask_neg: Optional[torch.Tensor] = None,
    decouple_weight_from_coupling: bool = False,
    self_mask_on_raw: bool = False,
    per_sample_fnorm: bool = False,
    return_raw_winner_stats: bool = False,
    global_scale_stats: bool = True,
    global_fnorm_stats: bool = True,
    collect_diagnostics: bool = True,
    share_distances: bool = True,
    baseline_top_p: float = 1.0,
    versionb_top_p: float = 1.0,
    top_p_min_keep: int = 1,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mixed drift loss:

        loss = ||x - sg(x + alpha * V_baseline + (1 - alpha) * V_versionb)||^2

    where V_baseline is the gen-centric grid drift force (drift_loss_imagenet)
    and V_versionb is the col-wise + coupling drift force
    (drift_loss_imagenet_colwise, coupling=True). Both vector fields are
    computed from the same (gen, pos, neg) features and weights; only the
    matching scheme and softmax temperature R_list differ.

    The loss is normalized by an alpha-interpolated scale so it reduces
    exactly to the version_b loss at alpha=0 and the baseline loss at alpha=1.
    """
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    raw_winner_info: Dict[str, float] = {}
    # A self-mask included in scale estimation changes the forward expert's
    # scale but not the reverse expert's. Preserve that legacy/debug mode by
    # falling back to the sequential implementation; the normal post-scale
    # mask path can share both distances and scale exactly.
    use_shared_distances = bool(share_distances) and not self_mask_on_raw
    if use_shared_distances:
        (
            old_gen,
            fixed_pos_f,
            fixed_neg_f,
            dist_joint_n,
            shared_scale,
        ) = _prepare_shared_mixed_distances(
            gen=gen,
            fixed_pos=fixed_pos,
            fixed_neg=fixed_neg,
            weight_neg=weight_neg,
            active_mask_pos=active_mask_pos,
            active_mask_neg=active_mask_neg,
            global_scale_stats=global_scale_stats,
        )
        C_g = old_gen.shape[1]
        C_n = fixed_neg_f.shape[1]
        split_idx = C_g + C_n
        dist_neg_n = dist_joint_n[:, :, :split_idx]
        dist_pos_n = dist_joint_n[:, :, split_idx:]

        if return_raw_winner_stats:
            # Positive distances differ from the raw values only by one
            # positive scalar, so their argmin winner statistics are exact.
            ws = compute_raw_winner_stats(dist_pos_n)
            raw_winner_info = {f"raw/{k}": v for k, v in ws.items()}

        old_gen, V_v, scale_v, info_v = _compute_force_colwise(
            old_gen,
            fixed_pos_f,
            fixed_neg_f,
            weight_neg=weight_neg,
            R_list=R_list_versionb,
            coupling=True,
            compute_wpos_stats=compute_wpos_stats,
            per_sample_fnorm=per_sample_fnorm,
            active_mask_pos=active_mask_pos,
            active_mask_neg=active_mask_neg,
            decouple_weight_from_coupling=decouple_weight_from_coupling,
            self_mask_on_raw=False,
            dist_pos_normed_precomputed=dist_pos_n,
            dist_neg_normed_precomputed=dist_neg_n,
            scale_precomputed=shared_scale,
            global_scale_stats=global_scale_stats,
            global_fnorm_stats=global_fnorm_stats,
            collect_diagnostics=collect_diagnostics,
            top_p=versionb_top_p,
            top_p_min_keep=top_p_min_keep,
            compute_top_p_stats=compute_wpos_stats,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
        )
        _, V_b, scale_b, info_b = _compute_force_baseline_grid(
            old_gen,
            fixed_pos_f,
            fixed_neg_f,
            weight_neg=weight_neg,
            R_list=R_list_baseline,
            compute_wpos_stats=False,
            active_mask_pos=active_mask_pos,
            active_mask_neg=active_mask_neg,
            dist_normed_precomputed=dist_joint_n,
            scale_precomputed=shared_scale,
            global_scale_stats=global_scale_stats,
            global_fnorm_stats=global_fnorm_stats,
            collect_diagnostics=collect_diagnostics,
            top_p=baseline_top_p,
            top_p_min_keep=top_p_min_keep,
            compute_top_p_stats=compute_wpos_stats,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
        )
        del dist_pos_n, dist_neg_n, dist_joint_n
    else:
        if return_raw_winner_stats:
            # Cheap dedicated cdist for winner stats — only [B, C_g, C_p].
            with torch.no_grad():
                dist_pos_for_stats = _cdist_batched(
                    gen.detach().float(), fixed_pos.detach().float()
                )
                ws = compute_raw_winner_stats(dist_pos_for_stats)
            raw_winner_info = {f"raw/{k}": v for k, v in ws.items()}
            del dist_pos_for_stats

        old_gen, V_v, scale_v, info_v = _compute_force_colwise(
            gen, fixed_pos, fixed_neg,
            weight_neg=weight_neg,
            R_list=R_list_versionb,
            coupling=True,
            compute_wpos_stats=compute_wpos_stats,
            per_sample_fnorm=per_sample_fnorm,
            active_mask_pos=active_mask_pos,
            active_mask_neg=active_mask_neg,
            decouple_weight_from_coupling=decouple_weight_from_coupling,
            self_mask_on_raw=self_mask_on_raw,
            global_scale_stats=global_scale_stats,
            global_fnorm_stats=global_fnorm_stats,
            collect_diagnostics=collect_diagnostics,
            top_p=versionb_top_p,
            top_p_min_keep=top_p_min_keep,
            compute_top_p_stats=compute_wpos_stats,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
        )
        # Reuse the already materialized FP32 detached generator features.
        # Passing ``gen`` here would create a second BF16->FP32 copy.
        _, V_b, scale_b, info_b = _compute_force_baseline_grid(
            old_gen, fixed_pos, fixed_neg,
            weight_neg=weight_neg,
            R_list=R_list_baseline,
            compute_wpos_stats=False,
            active_mask_pos=active_mask_pos,
            active_mask_neg=active_mask_neg,
            global_scale_stats=global_scale_stats,
            global_fnorm_stats=global_fnorm_stats,
            collect_diagnostics=collect_diagnostics,
            top_p=baseline_top_p,
            top_p_min_keep=top_p_min_keep,
            compute_top_p_stats=compute_wpos_stats,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
        )

    cos_vb = F.cosine_similarity(V_v.flatten().unsqueeze(0), V_b.flatten().unsqueeze(0), dim=1).squeeze().clamp(-1.0, 1.0)
    cos_factor = (alpha * alpha + (1.0 - alpha) * (1.0 - alpha) + 2.0 * alpha * (1.0 - alpha) * cos_vb).sqrt()
    scale_inputs = (((1.0 - alpha) * scale_v + alpha * scale_b) * cos_factor).clamp(min=1e-3)

    # Both vector fields are detached scratch tensors, so reuse V_b for the
    # interpolation and then for the goal instead of allocating two more
    # full-size MAE feature tensors.
    V_b.mul_(alpha).add_(V_v, alpha=(1.0 - alpha))
    del V_v
    goal = V_b.add_(old_gen).detach()
    diff = gen.float() - goal
    diff.div_(scale_inputs)
    loss = (diff ** 2).mean(dim=(-1, -2))

    info: Dict[str, float] = dict(info_v)
    for k, v in info_b.items():
        info[f"b/{k}"] = v
    if collect_diagnostics:
        info["mix_alpha"] = float(alpha)
        info["mix_scale_v"] = float(scale_v.item())
        info["mix_scale_b"] = float(scale_b.item())
    info.update(raw_winner_info)
    return loss, info


def compute_raw_winner_stats(dist_pos: torch.Tensor) -> Dict[str, float]:
    """Cumulative-friendly raw L2 winner stats. dist_pos: [B, C_g, C_p].

    Returns batch-level numerator/denominator counts (not yet divided):
      pos_winner_count: Σ_b (# unique pos picked as #1 by some gen)  → numerator of β1
      pos_winner_total: B * C_p                                     → denominator of β1
      gen_winner_count: Σ_b (# unique gen picked as #1 by some pos)  → numerator of α1
      gen_winner_total: B * C_g                                     → denominator of α1

    Used by MixAlphaTracker for the adaptive mix-alpha schedule:
      V_versionb_coef  = α1 β1 / (α1 β1 + (1-α1)(1-β1))
      V_baseline_coef  = 1 - V_versionb_coef                          (= mix_alpha)

    Fully vectorized — only 2 CUDA→CPU syncs total (the two `.item()` calls).
    """
    with torch.no_grad():
        B, C_g, C_p = dist_pos.shape
        # gen→pos: each gen picks closest pos
        pos_idx = dist_pos.argmin(dim=2)                                       # [B, C_g]
        pos_grid = torch.zeros(B, C_p, dtype=torch.float32, device=dist_pos.device)
        pos_grid.scatter_(1, pos_idx, 1.0)                                     # 1 if any gen picked p
        pos_uniq = pos_grid.sum().item()                                       # ← single sync
        # pos→gen: each pos picks closest gen
        gen_idx = dist_pos.argmin(dim=1)                                       # [B, C_p]
        gen_grid = torch.zeros(B, C_g, dtype=torch.float32, device=dist_pos.device)
        gen_grid.scatter_(1, gen_idx, 1.0)
        gen_uniq = gen_grid.sum().item()                                       # ← single sync
    return {
        "pos_winner_count": float(pos_uniq),
        "pos_winner_total": float(B * C_p),
        "gen_winner_count": float(gen_uniq),
        "gen_winner_total": float(B * C_g),
    }


__all__ = [
    "drift_loss_imagenet",
    "drift_loss_imagenet_colwise",
    "drift_loss_imagenet_mixed",
    "compute_raw_winner_stats",
]
