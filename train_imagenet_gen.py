"""Generator training for ImageNet — PyTorch port of the official JAX train.py.

Requires a frozen MAE, DINO, or MoCo feature checkpoint.

Usage (single GPU):
    python train_imagenet_gen.py --config configs/gen/latent_sota_B.yaml \
        --workdir runs/gen_latent_B

Usage (multi-GPU, torchrun):
    torchrun --nproc_per_node=8 train_imagenet_gen.py \
        --config configs/gen/latent_sota_B.yaml --workdir runs/gen_latent_B
"""
from __future__ import annotations

import argparse
import copy
import datetime
import gc
import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.parallel import DistributedDataParallel as DDP


def _amp_ctx(use_bf16: bool):
    """torch.autocast(bfloat16) if use_bf16 else nullcontext().

    With autocast: model params stay fp32 (good for AdamW state), but linear/matmul
    ops execute in bf16 — matches jax `use_bf16: true` forward dtype.
    """
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _gen_use_bf16(generator: nn.Module) -> bool:
    raw = generator.module if hasattr(generator, "module") else generator
    return bool(getattr(raw, "use_bf16", True))


def _mae_use_bf16(feature_extractor: nn.Module) -> bool:
    raw = feature_extractor.module if hasattr(feature_extractor, "module") else feature_extractor
    return bool(getattr(raw, "use_bf16", False))


def _ddp_mean_scalar(value: torch.Tensor | float, device: torch.device) -> float:
    """Return the all-rank mean of a scalar for logging only."""
    if isinstance(value, torch.Tensor):
        scalar = value.detach().to(device=device, dtype=torch.float64).reshape(())
    else:
        scalar = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        scalar /= float(dist.get_world_size())
    return float(scalar.cpu().item())


def _ddp_mean_scalars(
    values: Dict[str, torch.Tensor | float], device: torch.device
) -> Dict[str, float]:
    """Reduce a mapping of logging scalars with one collective and host copy."""
    if not values:
        return {}
    names = sorted(values)
    packed = torch.stack(
        [
            (
                value.detach().to(device=device, dtype=torch.float64).reshape(())
                if isinstance(value, torch.Tensor)
                else torch.tensor(float(value), device=device, dtype=torch.float64)
            )
            for value in (values[name] for name in names)
        ]
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed /= float(dist.get_world_size())
    host_values = packed.cpu().tolist()
    return dict(zip(names, (float(value) for value in host_values)))


def _configure_level4_cuda_runtime(
    cfg: dict,
    throughput_opt_level: int,
    device: torch.device,
) -> None:
    """Enable opt-level-4 CUDA math/autotuning knobs without changing lower levels."""
    if int(throughput_opt_level) < 4 or device.type != "cuda":
        return

    allow_tf32 = bool(cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if allow_tf32:
        torch.set_float32_matmul_precision("high")

    cudnn_benchmark = bool(cfg.get("cudnn_benchmark", True))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if cudnn_benchmark and hasattr(torch.backends.cudnn, "benchmark_limit"):
        torch.backends.cudnn.benchmark_limit = int(
            cfg.get("cudnn_benchmark_limit", 10)
        )


def _collect_mix_alpha_stats_this_step(
    *,
    throughput_opt_level: int,
    collect_diagnostics: bool,
    drift_matching: str,
) -> bool:
    """Whether raw winner stats/tracker updates are needed on this step.

    Fixed reverse/forward objectives never consume MixAlphaTracker state. At
    level 4 their winner computation and tracker collective are diagnostics
    only, so they follow the diagnostic cadence. Dual drift keeps its prior
    every-step behavior, including the adaptive state transition.
    """
    fixed_matching = str(drift_matching).lower().strip() in (
        "rev-drift",
        "fwd-drift",
    )
    return not (
        int(throughput_opt_level) >= 4
        and fixed_matching
        and not collect_diagnostics
    )


def _collect_training_metrics_this_step(
    *,
    throughput_opt_level: int,
    step: int,
    log_every_k: int,
) -> bool:
    """Gate logging-only CUDA syncs/all-reduces on level-4 non-log steps."""
    return (
        int(throughput_opt_level) < 4
        or int(step) % max(1, int(log_every_k)) == 0
    )


def _zero_generator_grad(
    optimizer: torch.optim.Optimizer,
    throughput_opt_level: int,
) -> None:
    """Use gradient deallocation only in the level-4 runtime stack."""
    if int(throughput_opt_level) >= 4:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad()


def _stochastic_round(x):
    """Stochastic rounding: floor(x) w.p. (1-frac), ceil(x) w.p. frac, frac=x-floor(x).

    Why: deterministic round() biases the cfg→pos/neg count split when the
    fractional value (e.g. 14.2) is not an integer; stochastic rounding makes
    the expected count exactly equal to the fractional value.
    Tensor in → long tensor out (same shape). Scalar in → Python int out.
    Uses torch RNG so the training seed controls reproducibility.
    """
    if torch.is_tensor(x):
        floor = torch.floor(x)
        frac = x - floor
        bump = (torch.rand_like(x) < frac).to(x.dtype)
        return (floor + bump).long()
    floor_val = math.floor(float(x))
    frac = float(x) - floor_val
    return floor_val + int(torch.rand(()).item() < frac)


_STOCHASTIC_FEATURE_STAGES = ("stage1", "stage2", "stage3", "stage4")
_FEATURE_LOSS_GROUPS = ("global", "norm_x", *_STOCHASTIC_FEATURE_STAGES)


def _feature_stage_group(name: str) -> Optional[str]:
    """Map an MAE activation key to one of four stochastic loss groups.

    The encoder stem (``conv1*``) is grouped with stage 1. Global/raw-input
    features and unknown future feature keys return ``None`` and remain active
    on every step.
    """
    if name == "conv1" or name.startswith("conv1_"):
        return "stage1"
    for stage_index in range(1, 5):
        prefix = f"layer{stage_index}"
        if name == prefix or name.startswith(f"{prefix}_"):
            return f"stage{stage_index}"
    return None


def _feature_loss_group(name: str) -> Optional[str]:
    """Map an activation key to a configurable loss-importance group."""
    if name in ("global", "norm_x"):
        return name
    return _feature_stage_group(name)


def _resolve_drift_top_k_groups(cfg: dict) -> Optional[Tuple[str, ...]]:
    """Resolve feature groups on which fixed-support top-k is active.

    ``None`` means every feature, preserving the behavior of configs that
    predate group-scoped top-k.  A comma-separated string or sequence selects
    explicit groups, e.g. ``stage2,stage3,stage4`` keeps the inexpensive and
    critical ``norm_x`` objective dense while truncating MAE-stage forces.
    """
    raw = cfg.get("drift_top_k_groups", "all")
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped.lower() == "all":
            return None
        groups = tuple(part.strip() for part in stripped.split(",") if part.strip())
    elif isinstance(raw, (list, tuple, set)):
        groups = tuple(str(part).strip() for part in raw if str(part).strip())
    else:
        raise ValueError(
            "drift_top_k_groups must be 'all', a comma-separated string, "
            "or a sequence of feature-loss groups"
        )

    unknown = set(groups).difference(_FEATURE_LOSS_GROUPS)
    if unknown:
        raise ValueError(
            "Unknown drift top-k groups: "
            f"{sorted(unknown)}; expected {sorted(_FEATURE_LOSS_GROUPS)}"
        )
    if not groups:
        raise ValueError("drift_top_k_groups must select at least one group")
    # Preserve the canonical order and remove duplicates for stable logging.
    selected = set(groups)
    return tuple(group for group in _FEATURE_LOSS_GROUPS if group in selected)


def _top_k_for_feature(
    name: str,
    top_k_pos: int,
    top_k_neg: int,
    groups: Optional[Tuple[str, ...]],
) -> Tuple[int, int]:
    """Return fixed supports for one feature, or dense supports when excluded."""
    if groups is not None and _feature_loss_group(name) not in groups:
        return 0, 0
    return int(top_k_pos), int(top_k_neg)


def _resolve_feature_loss_group_weights(cfg: dict) -> Dict[str, float]:
    """Resolve a named feature-loss profile or a direct group mapping.

    Zero is valid and disables a group. Negative or non-finite coefficients
    are rejected. Unknown activation keys retain ``default`` (1.0 if omitted).
    """
    direct = cfg.get("feature_loss_group_weights")
    if direct is not None:
        if not isinstance(direct, dict):
            raise ValueError("feature_loss_group_weights must be a mapping")
        resolved = {str(k): float(v) for k, v in direct.items()}
    else:
        profile_name = str(cfg.get("feature_loss_profile", "all"))
        profiles = cfg.get("feature_loss_profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError("feature_loss_profiles must be a mapping")
        if not profiles:
            return {}
        if profile_name not in profiles:
            available = ", ".join(sorted(str(k) for k in profiles)) or "<none>"
            raise ValueError(
                f"Unknown feature_loss_profile={profile_name!r}; "
                f"available profiles: {available}"
            )
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise ValueError(
                f"feature_loss_profiles.{profile_name} must be a mapping"
            )
        resolved = {str(k): float(v) for k, v in profile.items()}

    allowed = set(_FEATURE_LOSS_GROUPS) | {"default"}
    unknown = set(resolved).difference(allowed)
    if unknown:
        raise ValueError(
            "Unknown feature-loss groups: "
            f"{sorted(unknown)}; expected {sorted(allowed)}"
        )
    invalid = {
        key: value
        for key, value in resolved.items()
        if not math.isfinite(value) or value < 0.0
    }
    if invalid:
        raise ValueError(
            "Feature-loss group weights must be finite and >= 0, got "
            f"{invalid}"
        )
    return resolved


def _feature_loss_weights_for_groups(
    feature_names: List[str] | Tuple[str, ...],
    group_weights: Dict[str, float],
    *,
    normalize: bool,
) -> Dict[str, float]:
    """Expand group coefficients to features, optionally preserving mass.

    Mass normalization keeps ``sum(feature weights) == feature count``. This
    prevents a leave-one-stage-out run from silently reducing the mean loss
    coefficient merely because that stage emits many derived objectives.
    """
    names = tuple(feature_names)
    default = float(group_weights.get("default", 1.0))
    weights: Dict[str, float] = {}
    for name in names:
        group = _feature_loss_group(name)
        weights[name] = float(group_weights.get(group, default))

    if normalize and weights:
        coefficient_mass = sum(weights.values())
        if coefficient_mass <= 0.0:
            raise ValueError(
                "feature_loss_group_normalize requires at least one active feature"
            )
        scale = len(weights) / coefficient_mass
        weights = {name: value * scale for name, value in weights.items()}
    return weights


def _feature_temperature_multiplier(
    name: str,
    multipliers: Optional[Dict[str, float]],
) -> float:
    """Return the static temperature multiplier for one MAE feature.

    Exact feature-name entries take precedence over the four encoder-stage
    entries.  Raw-input/global features do not belong to an encoder stage and
    therefore use ``default`` (1.0 when omitted).
    """
    if not multipliers:
        return 1.0
    if name in multipliers:
        return float(multipliers[name])
    stage = _feature_stage_group(name)
    if stage is not None and stage in multipliers:
        return float(multipliers[stage])
    return float(multipliers.get("default", 1.0))


def _resolve_layer_temperature_multipliers(cfg: dict) -> Dict[str, float]:
    """Resolve the selected named layer-temperature profile from a config."""
    direct = cfg.get("layer_temperature_multipliers")
    if direct is not None:
        if not isinstance(direct, dict):
            raise ValueError("layer_temperature_multipliers must be a mapping")
        resolved = {str(k): float(v) for k, v in direct.items()}
    else:
        profile_name = str(cfg.get("layer_temperature_profile", "uniform"))
        profiles = cfg.get("layer_temperature_profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError("layer_temperature_profiles must be a mapping")
        # Backward-compatible default for configs that predate layerwise
        # temperature control: no mapping means every feature keeps R as-is.
        if not profiles:
            return {}
        if profile_name not in profiles:
            available = ", ".join(sorted(str(k) for k in profiles)) or "<none>"
            raise ValueError(
                f"Unknown layer_temperature_profile={profile_name!r}; "
                f"available profiles: {available}"
            )
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise ValueError(
                f"layer_temperature_profiles.{profile_name} must be a mapping"
            )
        resolved = {str(k): float(v) for k, v in profile.items()}

    invalid = {k: v for k, v in resolved.items() if not math.isfinite(v) or v <= 0.0}
    if invalid:
        raise ValueError(
            "Layer temperature multipliers must be finite and > 0, got "
            f"{invalid}"
        )
    return resolved


def _feature_loss_weights_for_stages(
    feature_names: List[str] | Tuple[str, ...],
    selected_stages: Tuple[str, ...],
) -> Dict[str, float]:
    """Return Horvitz--Thompson weights for a uniform fixed-size stage draw."""
    selected = set(selected_stages)
    unknown = selected.difference(_STOCHASTIC_FEATURE_STAGES)
    if unknown:
        raise ValueError(f"Unknown stochastic feature stages: {sorted(unknown)}")
    stage_count = len(selected)
    if not 1 <= stage_count <= len(_STOCHASTIC_FEATURE_STAGES):
        raise ValueError(
            "selected_stages must contain between 1 and 4 unique stages"
        )

    # With k stages sampled uniformly without replacement from four stages,
    # every stage has inclusion probability k/4. Multiplying selected losses
    # by 4/k therefore preserves the original full-feature objective in
    # expectation. Always-on features retain weight 1.
    inverse_probability = len(_STOCHASTIC_FEATURE_STAGES) / stage_count
    weights: Dict[str, float] = {}
    for name in feature_names:
        group = _feature_stage_group(name)
        if group is None:
            weights[name] = 1.0
        elif group in selected:
            weights[name] = inverse_probability
        else:
            weights[name] = 0.0
    return weights


def _sample_stochastic_feature_stages(
    *,
    stage_count: int,
    seed: int,
    step: int,
) -> Tuple[str, ...]:
    """Draw a deterministic, resume-stable stage subset for one global step.

    Every DDP rank intentionally uses the same subset. The drift losses contain
    distributed collectives, so rank-specific subsets could mismatch collective
    call order and would also introduce step-time stragglers.
    """
    num_stages = len(_STOCHASTIC_FEATURE_STAGES)
    if not 1 <= stage_count <= num_stages:
        raise ValueError(
            f"stochastic_feature_stage_count must be in [1, {num_stages}], "
            f"got {stage_count}"
        )
    rng = torch.Generator(device="cpu")
    # Stateless sampling makes a resumed step select exactly the same stages.
    rng_seed = (int(seed) + 1_000_003 * int(step)) % (2**63 - 1)
    rng.manual_seed(rng_seed)
    indices = torch.randperm(num_stages, generator=rng)[:stage_count].tolist()
    return tuple(_STOCHASTIC_FEATURE_STAGES[i] for i in indices)
from tqdm import tqdm

# Local imports
from drifting_core.imagenet_loss import (
    _cdist_batched,
    compute_raw_winner_stats,
    drift_loss_imagenet,
    drift_loss_imagenet_colwise,
    drift_loss_imagenet_mixed,
)
from drifting_core.topk_diagnostics import diagnose_reverse_topk_heterogeneity
from memory_bank import ArrayMemoryBank, CompressedPixelMemoryBank
from models.adversarial_drift import AdversarialDriftSystem
from models.feature_adapter import (
    FeatureAdapterSystem,
    canonical_adapter_stages,
    make_dino_patch_masked_images,
    update_adapter_ema,
)
from models.feature_gan import (
    FrozenFeatureDiscriminator,
    canonical_feature_gan_stages,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from models.imagenet_generator import DitGen, build_ditgen_from_config
from models.mae_resnet import MAEResNet, build_mae_from_config
from models.ssl_resnet import build_ssl_resnet_from_config, canonical_ssl_backbone
from models.ssl_latent_bridge import SSLLatentBridgeFeatureExtractor
from train.train_data import (
    create_imagenet_split,
    create_raw_image_io_semaphore,
    infinite_sampler,
)
from utils import EMA


# ---------------------------------------------------------------------------
# Adaptive mix-alpha tracker (cumulative pos/gen winner ratios)
# ---------------------------------------------------------------------------

class MixAlphaTracker:
    """Per-step pos/gen winner ratio tracker for hedge mix-alpha schedule.

    Raw diagnostics:
      α1 = current step's gen winner ratio (unique-gen-as-#1-of-some-pos / C_g)
      β1 = current step's pos winner ratio (unique-pos-as-#1-of-some-gen / C_p)

    Hedge updates use capacity-normalized variants whose common denominator is
    min(C_g, C_p). This removes the structural bias caused by unequal set sizes:
    for example, with C_g=64 and C_p=128, raw β1 is capped at 0.5 while both
    capacity-normalized ratios can reach 1.0.

    Counts are overwritten each step (no accumulation). Across DDP ranks,
    counts are all-reduced sum'd before each update so all ranks see the
    same α/β at every step.
    """

    _PRIOR_EPS = 1.0e-8

    def __init__(
        self,
        gamma: float = 2.0,
        eta: float = 1.0e-4,
        decay: float = 1.0e-4,
        gamma_warmup_steps: int = 10000,
        mode: str = "hedge",
        initial_alpha: float = 0.5,
    ) -> None:
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.decay = float(decay)
        self.gamma_warmup_steps = int(gamma_warmup_steps)
        self.mode = str(mode).lower().strip()
        self.initial_alpha = float(initial_alpha)
        # Per-step (overwritten) winner counts.
        self.pos_win = 0.0
        self.pos_total = 0.0
        self.gen_win = 0.0
        self.gen_total = 0.0
        # Persistent hedge state. log_w_b/log_w_v accumulate log-weights for
        # the (rev-drift, fwd-drift) experts. Discount factor `decay` is
        # applied each step to bound them automatically (no clip).
        #
        # A finite epsilon is used for exact 0/1 requests. True zero weight
        # would be log(0)=-inf and could never recover under multiplicative
        # Hedge updates.
        prior = min(
            1.0 - self._PRIOR_EPS,
            max(self._PRIOR_EPS, self.initial_alpha),
        )
        self.log_w_b = math.log(prior)
        self.log_w_v = math.log1p(-prior)

    def update(
        self,
        info: Dict[str, float],
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Read 'raw/...' counts produced by compute_drift_loss_from_features
        and overwrite with this step's values (after DDP sum-reduce when
        world_size > 1)."""
        pw = float(info.get("raw/pos_winner_count", 0.0))
        pt = float(info.get("raw/pos_winner_total", 0.0))
        gw = float(info.get("raw/gen_winner_count", 0.0))
        gt = float(info.get("raw/gen_winner_total", 0.0))
        if world_size > 1 and device is not None and dist.is_available() and dist.is_initialized():
            t = torch.tensor([pw, pt, gw, gt], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            pw, pt, gw, gt = [float(x) for x in t.cpu().tolist()]
        self.pos_win = pw
        self.pos_total = pt
        self.gen_win = gw
        self.gen_total = gt

    @property
    def beta1(self) -> float:
        return self.pos_win / max(self.pos_total, 1.0)

    @property
    def alpha1(self) -> float:
        return self.gen_win / max(self.gen_total, 1.0)

    @property
    def matching_capacity(self) -> float:
        """Maximum possible unique matches on either side."""
        return min(self.pos_total, self.gen_total)

    @property
    def alpha1_capacity(self) -> float:
        """Gen winners normalized by the shared gen/pos matching capacity."""
        value = self.gen_win / max(self.matching_capacity, 1.0)
        return self._clamp_alpha(value)

    @property
    def beta1_capacity(self) -> float:
        """Positive winners normalized by the shared matching capacity."""
        value = self.pos_win / max(self.matching_capacity, 1.0)
        return self._clamp_alpha(value)

    def has_data(self) -> bool:
        return self.pos_total > 0.0 and self.gen_total > 0.0

    def _clamp_alpha(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def compute_closed_form_mix_alpha(self) -> float:
        """Data-only rev-drift coefficient from the latest winner ratios.

        Let a and b be the gen and positive winner counts divided by their
        shared matching capacity min(C_g, C_p). The fwd-drift coefficient is
        the odds-like closed form, and mix_alpha is the remaining rev coeff.
        This mode has no time-logit, hedge eta, gamma warmup, or discount term.
        """
        if not self.has_data():
            return self._clamp_alpha(self.initial_alpha)
        a = self.alpha1_capacity
        b = self.beta1_capacity
        fwd_score = a * b
        rev_score = (1.0 - a) * (1.0 - b)
        denom = fwd_score + rev_score
        if denom <= 1.0e-12:
            return self._clamp_alpha(self.initial_alpha)
        fwd_coef = fwd_score / denom
        return self._clamp_alpha(1.0 - fwd_coef)

    def compute_mix_alpha(self, step: int = 0, total_steps: int = 1) -> float:
        """Returns rev-drift coefficient (= `mix_alpha`).

        Modes:
          closed_form/ratio/data_only: coefficient from current winner ratios only.
          hedge_no_time/data_hedge: discounted hedge using winner ratios only.
          hedge or legacy names: discounted hedge + logit-form time bias.

        Hedge + logit-form time bias:
            t          = step / max(total_steps, 1)            ∈ [0, 1]
            γ_eff:
              step < W                  → 0
              W ≤ step ≤ total_steps    → γ · (step − W) / (T − W)   # smooth ramp
              (W = γ_warmup_steps)
            a      = gen_winners / min(gen_total, pos_total)
            b      = pos_winners / min(gen_total, pos_total)
            L_rev  = γ_eff · (1 - a)
            L_fwd  = γ_eff · (1 - b)
            ℓ_b ← (1-decay)·ℓ_b − η·L_rev
            ℓ_v ← (1-decay)·ℓ_v − η·L_fwd
            mix_alpha = sigmoid( (ℓ_b − ℓ_v) + log(t/(1-t)) )
                                 ──data_logit──   ──time_logit──
            t→0: mix_alpha→0 (pure fwd-drift at start)
            t→1: mix_alpha→1 (pure rev-drift at end)
        """
        if self.mode in ("closed_form", "ratio", "data_only", "winner_ratio"):
            return self.compute_closed_form_mix_alpha()

        a, b = self.alpha1_capacity, self.beta1_capacity
        T = max(int(total_steps), 1)
        t_raw = float(step) / T
        if self.mode in ("hedge_no_time", "data_hedge", "no_time", "winner_hedge"):
            gamma_eff = self.gamma * min(1.0, max(0.0, t_raw))
            L_b = gamma_eff * (1.0 - a)
            L_v = gamma_eff * (1.0 - b)
            d = 1.0 - self.decay
            self.log_w_b = d * self.log_w_b - self.eta * L_b
            self.log_w_v = d * self.log_w_v - self.eta * L_v
            logit = self.log_w_b - self.log_w_v
            if logit >= 0:
                z = math.exp(-logit)
                return 1.0 / (1.0 + z)
            z = math.exp(logit)
            return z / (1.0 + z)

        W = self.gamma_warmup_steps
        if W >= T or step < W:
            gamma_eff = 0.0
        else:
            progress = (step - W) / (T - W)
            gamma_eff = self.gamma * min(1.0, max(0.0, progress))
        L_b = gamma_eff * (1.0 - a)
        L_v = gamma_eff * (1.0 - b)
        d = 1.0 - self.decay
        self.log_w_b = d * self.log_w_b - self.eta * L_b
        self.log_w_v = d * self.log_w_v - self.eta * L_v
        eps = 1.0 / T
        t_safe = min(1.0 - eps, max(eps, t_raw))
        time_logit = math.log(t_safe / (1.0 - t_safe))
        data_logit = self.log_w_b - self.log_w_v
        logit = data_logit + time_logit
        if logit >= 0:
            z = math.exp(-logit)
            mix_alpha = 1.0 / (1.0 + z)
        else:
            z = math.exp(logit)
            mix_alpha = z / (1.0 + z)
        return float(max(0.0, min(1.0, mix_alpha)))

    def state_dict(self) -> Dict[str, float]:
        return {
            "pos_win": self.pos_win,
            "pos_total": self.pos_total,
            "gen_win": self.gen_win,
            "gen_total": self.gen_total,
            "log_w_b": self.log_w_b,
            "log_w_v": self.log_w_v,
        }

    def load_state_dict(self, sd: Dict[str, float]) -> None:
        # Backward-compat: older checkpoints used the cumulative `sum_*` keys.
        self.pos_win = float(sd.get("pos_win", sd.get("sum_pos_win", 0.0)))
        self.pos_total = float(sd.get("pos_total", sd.get("sum_pos_total", 0.0)))
        self.gen_win = float(sd.get("gen_win", sd.get("sum_gen_win", 0.0)))
        self.gen_total = float(sd.get("gen_total", sd.get("sum_gen_total", 0.0)))
        self.log_w_b = float(sd.get("log_w_b", 0.0))
        self.log_w_v = float(sd.get("log_w_v", 0.0))


# ---------------------------------------------------------------------------
# Shared helpers (same as train_imagenet_mae.py)
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg: dict = {}
    for section in ("logging", "env", "dataset", "model", "optimizer", "train", "feature"):
        cfg.update(raw.get(section, {}))
    cfg["_raw"] = raw
    return cfg


def setup_distributed() -> Tuple[int, int, torch.device]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr = float(cfg.get("lr", 4e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    b1 = float(cfg.get("adam_b1", 0.9))
    b2 = float(cfg.get("adam_b2", 0.95))
    fused = int(cfg.get("throughput_opt_level", 0)) >= 3
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(b1, b2),
        fused=fused,
    )


def get_lr(step: int, warmup_steps: int, base_lr: float, init_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return init_lr + (base_lr - init_lr) * frac
    return base_lr


def _split_evenly(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    base, rem = divmod(int(total), int(parts))
    return [base + (1 if idx < rem else 0) for idx in range(parts)]


def _steps_for_generated_epochs(
    *,
    dataset_size: int,
    generated_per_step: int,
    epochs: float,
) -> int:
    """Smallest step count that reaches a generated-sample epoch target."""
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    if generated_per_step <= 0:
        raise ValueError(
            f"generated_per_step must be positive, got {generated_per_step}"
        )
    if not math.isfinite(float(epochs)) or float(epochs) <= 0.0:
        raise ValueError(f"epochs must be finite and positive, got {epochs}")
    return int(math.ceil(float(dataset_size) * float(epochs) / generated_per_step))


def _crosses_generated_epoch_interval(
    *,
    completed_steps: int,
    generated_per_step: int,
    dataset_size: int,
    interval_epochs: float,
) -> bool:
    """Whether this completed step crosses a generated-epoch boundary."""
    if completed_steps <= 0:
        return False
    interval_samples = float(dataset_size) * float(interval_epochs)
    if interval_samples <= 0.0 or not math.isfinite(interval_samples):
        raise ValueError(
            "save_per_generated_epochs must define a finite positive interval"
        )
    previous_samples = (completed_steps - 1) * generated_per_step
    current_samples = completed_steps * generated_per_step
    return int(current_samples // interval_samples) > int(
        previous_samples // interval_samples
    )


def _split_bank_stream(
    samples: np.ndarray,
    labels: np.ndarray,
    n_banks: int,
    phase: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    samples_np = np.asarray(samples)
    labels_np = np.asarray(labels).astype(np.int64, copy=False)
    if samples_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"samples/labels length mismatch: {samples_np.shape[0]} vs {labels_np.shape[0]}"
        )
    if n_banks <= 1:
        return [(samples_np, labels_np)], 0
    if samples_np.shape[0] == 0:
        empty = [(samples_np[:0], labels_np[:0]) for _ in range(n_banks)]
        return empty, phase % n_banks

    bank_ids = (np.arange(samples_np.shape[0], dtype=np.int64) + phase) % n_banks
    split_batches: List[Tuple[np.ndarray, np.ndarray]] = []
    for bank_idx in range(n_banks):
        mask = bank_ids == bank_idx
        split_batches.append((samples_np[mask], labels_np[mask]))
    return split_batches, (phase + samples_np.shape[0]) % n_banks


def _step_choice_indices(n_items: int, n_select: int, seed: int, step: int, bank_idx: int = 0) -> np.ndarray:
    """Deterministic per-step choice for labels from the latest pushed batch.

    Official JAX folds the global step into the train RNG before choosing labels.
    Keep this local to label selection so it does not perturb NumPy's global RNG
    used by memory-bank sampling.
    """
    if n_select > n_items:
        raise ValueError(f"Cannot sample {n_select} labels from a batch of {n_items}.")
    entropy = [
        int(seed) & 0xFFFFFFFF,
        (int(seed) >> 32) & 0xFFFFFFFF,
        int(step) & 0xFFFFFFFF,
        (int(step) >> 32) & 0xFFFFFFFF,
        int(bank_idx) & 0xFFFFFFFF,
    ]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return rng.choice(n_items, size=n_select, replace=False)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def save_checkpoint(
    workdir: str,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    keep_last: int = 2,
    keep_every: int = 50000,
    mix_alpha_tracker: Optional[MixAlphaTracker] = None,
    feature_adapter: Optional[nn.Module] = None,
    feature_adapter_target: Optional[FeatureAdapterSystem] = None,
    feature_adapter_optimizer: Optional[torch.optim.Optimizer] = None,
    feature_discriminator: Optional[nn.Module] = None,
    feature_discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    adversarial_system: Optional[AdversarialDriftSystem] = None,
) -> None:
    ckpt_dir = Path(workdir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_step_{step:07d}.pt"
    raw = model.module if hasattr(model, "module") else model
    payload: Dict[str, Any] = {
        "step": step,
        "model": raw.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    if mix_alpha_tracker is not None:
        payload["mix_alpha_tracker"] = mix_alpha_tracker.state_dict()
    if feature_adapter is not None:
        raw_adapter = (
            feature_adapter.module
            if hasattr(feature_adapter, "module")
            else feature_adapter
        )
        payload["feature_adapter"] = raw_adapter.state_dict()
    if feature_adapter_target is not None:
        payload["feature_adapter_target"] = feature_adapter_target.state_dict()
    if feature_adapter_optimizer is not None:
        payload["feature_adapter_optimizer"] = feature_adapter_optimizer.state_dict()
    if feature_discriminator is not None:
        raw_discriminator = (
            feature_discriminator.module
            if hasattr(feature_discriminator, "module")
            else feature_discriminator
        )
        payload["feature_discriminator"] = raw_discriminator.state_dict()
    if feature_discriminator_optimizer is not None:
        payload["feature_discriminator_optimizer"] = (
            feature_discriminator_optimizer.state_dict()
        )
    if adversarial_system is not None:
        payload["adversarial_system"] = adversarial_system.state_dict()
    latest = ckpt_dir / "ckpt_latest.pt"
    if adversarial_system is not None and bool(cfg.get("checkpoint_latest_hardlink", False)):
        # Publish a new inode before replacing either name: a previous latest
        # can be hardlinked to a retained numbered checkpoint and must never
        # be truncated in place, including a repeated save of the same step.
        save_fd, save_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=ckpt_dir)
        os.close(save_fd)
        save_temp = Path(save_name)
        link_temp: Optional[Path] = None
        try:
            torch.save(payload, save_temp)
            os.replace(save_temp, path)
            link_fd, link_name = tempfile.mkstemp(prefix=".latest-", dir=ckpt_dir)
            os.close(link_fd)
            link_temp = Path(link_name)
            link_temp.unlink()
            os.link(path, link_temp)
            os.replace(link_temp, latest)
        finally:
            save_temp.unlink(missing_ok=True)
            if link_temp is not None:
                link_temp.unlink(missing_ok=True)
    else:
        torch.save(payload, path)
        torch.save(torch.load(path, map_location="cpu", weights_only=False), latest)

    checkpoints = sorted(ckpt_dir.glob("ckpt_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in checkpoints[:-keep_last]:
        old_step = int(old.stem.split("_")[-1])
        if old_step % keep_every != 0:
            old.unlink(missing_ok=True)


def load_checkpoint(
    workdir: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mix_alpha_tracker: Optional[MixAlphaTracker] = None,
    feature_adapter: Optional[nn.Module] = None,
    feature_adapter_target: Optional[FeatureAdapterSystem] = None,
    feature_adapter_optimizer: Optional[torch.optim.Optimizer] = None,
    feature_discriminator: Optional[nn.Module] = None,
    feature_discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    adversarial_system: Optional[AdversarialDriftSystem] = None,
) -> int:
    latest = Path(workdir) / "checkpoints" / "ckpt_latest.pt"
    if not latest.exists():
        return 0
    state = torch.load(latest, map_location=device, weights_only=False)
    if (adversarial_system is None) != ("adversarial_system" not in state):
        raise ValueError(
            "Checkpoint adversarial state does not match the requested experiment; "
            "resume requires the online discriminator, frozen target, and optimizer."
        )
    if adversarial_system is not None:
        adversarial_system.load_state_dict(state["adversarial_system"])
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    if mix_alpha_tracker is not None and "mix_alpha_tracker" in state:
        mix_alpha_tracker.load_state_dict(state["mix_alpha_tracker"])
    if feature_adapter is not None and "feature_adapter" in state:
        raw_adapter = (
            feature_adapter.module
            if hasattr(feature_adapter, "module")
            else feature_adapter
        )
        raw_adapter.load_state_dict(state["feature_adapter"])
    if feature_adapter_target is not None and "feature_adapter_target" in state:
        feature_adapter_target.load_state_dict(state["feature_adapter_target"])
    if (
        feature_adapter_optimizer is not None
        and "feature_adapter_optimizer" in state
    ):
        feature_adapter_optimizer.load_state_dict(state["feature_adapter_optimizer"])
    if feature_discriminator is not None and "feature_discriminator" in state:
        raw_discriminator = (
            feature_discriminator.module
            if hasattr(feature_discriminator, "module")
            else feature_discriminator
        )
        raw_discriminator.load_state_dict(state["feature_discriminator"])
    if (
        feature_discriminator_optimizer is not None
        and "feature_discriminator_optimizer" in state
    ):
        feature_discriminator_optimizer.load_state_dict(
            state["feature_discriminator_optimizer"]
        )
    return int(state.get("step", 0))


def build_adversarial_system(
    cfg: dict, device: torch.device
) -> Optional[AdversarialDriftSystem]:
    """Build the optional raw-image branch without consuming the generator RNG."""
    mode = str(cfg.get("adversarial_mode", "none")).strip().lower()
    if mode in {"", "none", "off"}:
        return None
    if mode not in {"raw_gan", "feature_drift", "mixed"}:
        raise ValueError(f"Unknown adversarial_mode={mode!r}")
    if bool(cfg.get("use_latent", True)) or int(cfg.get("in_channels", 3)) != 3:
        raise ValueError("The adversarial branch requires raw RGB generator outputs")
    if bool(cfg.get("feature_adapter", False)) or bool(cfg.get("feature_gan", False)):
        raise ValueError("The adversarial branch requires the unchanged frozen encoder")
    if resolve_feature_extractor_name(cfg) != "dino_resnet50":
        raise ValueError("The adversarial experiments require the frozen DINO baseline")
    samples_per_class = int(cfg.get("adversarial_samples_per_class", 8))
    if samples_per_class <= 0:
        raise ValueError("adversarial_samples_per_class must be positive")
    weight_defaults = {}
    if mode in {"raw_gan", "mixed"}:
        weight_defaults["adversarial_loss_weight"] = 0.1
    if mode in {"feature_drift", "mixed"}:
        weight_defaults["adversarial_drift_weight"] = 1.0
    for weight_key, default in weight_defaults.items():
        weight = float(cfg.get(weight_key, default))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"{weight_key} must be finite and positive")
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        return AdversarialDriftSystem.from_config(cfg, device)


def _require_finite_adversarial(
    value: torch.Tensor, name: str, *, require_nonzero: bool = False
) -> None:
    """Fail all ranks before an optimizer update if one rank has invalid values."""
    detached = value.detach()
    good = torch.isfinite(detached).all()
    if require_nonzero:
        good = good & detached.abs().sum().gt(0)
    flag = good.to(dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise FloatingPointError(f"Invalid adversarial {name} on at least one rank")


class Logger:
    def __init__(self, workdir: str, cfg: dict, rank: int):
        self.rank = rank
        self.log_file = Path(workdir) / "train_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.use_wandb = bool(cfg.get("use_wandb", False)) and rank == 0
        self.console_log = bool(cfg.get("console_log", True)) and rank == 0
        self.wandb_log_min_step = int(cfg.get("wandb_log_min_step", 0) or 0)
        if self.wandb_log_min_step < 0:
            raise ValueError("wandb_log_min_step must be non-negative")
        self._step = 0
        if self.use_wandb:
            try:
                import wandb
                # Persist run_id in workdir so resumed runs continue the same
                # wandb experiment instead of creating a new one each time.
                run_id_file = Path(workdir) / "wandb_run_id.txt"
                run_id = None
                rewind_on_resume = bool(cfg.get("wandb_rewind_on_resume", False))
                resume_mode = "never"
                resume_from = None
                if run_id_file.exists():
                    run_id = run_id_file.read_text().strip()
                    if rewind_on_resume:
                        rewind_step = cfg.get("wandb_resume_step")
                        if rewind_step is None:
                            ckpt = Path(workdir) / "checkpoints" / "ckpt_latest.pt"
                            if ckpt.exists():
                                try:
                                    state = torch.load(ckpt, map_location="cpu", weights_only=False)
                                    rewind_step = int(state.get("step", 0))
                                except Exception:
                                    rewind_step = None
                        if rewind_step is not None:
                            rewind_step = int(rewind_step)
                            resume_from = f"{run_id}?_step={rewind_step}"
                            print(f"[W&B] Rewinding run {run_id} to step {rewind_step}")
                        else:
                            resume_mode = "must"
                            print(f"[W&B] Resuming run {run_id}")
                    else:
                        resume_mode = "must"
                        print(f"[W&B] Resuming run {run_id}")
                init_kwargs = {
                    "project": cfg.get("project", "ReplayDrift"),
                    "entity": cfg.get("entity") or None,
                    "name": cfg.get("name", Path(workdir).name),
                    "config": cfg,
                    "settings": wandb.Settings(init_timeout=300),
                }
                if resume_from is not None:
                    init_kwargs["resume_from"] = resume_from
                else:
                    init_kwargs["id"] = run_id
                    init_kwargs["resume"] = resume_mode
                forked = False
                try:
                    wandb.init(**init_kwargs)
                except Exception as init_err:
                    err_text = str(init_err).lower()
                    if resume_from is not None and "rewind" in err_text:
                        print(f"[W&B] Rewind unavailable; forking from {resume_from} instead.")
                        try:
                            wandb.teardown()
                        except Exception:
                            pass
                        init_kwargs.pop("resume_from", None)
                        init_kwargs.pop("id", None)
                        init_kwargs.pop("resume", None)
                        init_kwargs["fork_from"] = resume_from
                        init_kwargs["settings"] = wandb.Settings(init_timeout=300)
                        wandb.init(**init_kwargs)
                        forked = True
                    elif run_id is not None and ("previously created and deleted" in err_text or "try a new run id" in err_text):
                        print(f"[W&B] stored run id {run_id} is invalid; starting a fresh run.")
                        try:
                            run_id_file.unlink()
                        except OSError:
                            pass
                        wandb.init(
                            project=cfg.get("project", "ReplayDrift"),
                            entity=cfg.get("entity") or None,
                            name=cfg.get("name", Path(workdir).name),
                            config=cfg,
                            resume="never",
                            settings=wandb.Settings(init_timeout=300),
                        )
                    else:
                        raise
                # Save run_id for future resumes (overwrite if we just forked into a new run).
                if forked or not run_id_file.exists():
                    run_id_file.write_text(wandb.run.id)
                self._wandb = wandb
                if self.wandb_log_min_step > 0:
                    print(
                        "[W&B] Holding cloud metric writes until step "
                        f"{self.wandb_log_min_step}; local JSONL remains enabled."
                    )
            except Exception as e:
                print(f"[W&B] init failed: {e}.")
                self.use_wandb = False

    @staticmethod
    def _fmt_console_value(value: Any) -> str:
        if isinstance(value, (float, int)):
            value_f = float(value)
            if abs(value_f) >= 1000.0:
                return f"{value_f:.1f}"
            if abs(value_f) >= 10.0:
                return f"{value_f:.3f}"
            return f"{value_f:.4f}"
        return str(value)

    def _emit_console_summary(self, step: int, metrics: dict) -> None:
        if not self.console_log:
            return
        ordered_keys = [
            "loss",
            "g_norm",
            "cfg_mean",
            "lr",
            "time/step",
            "time/per_step",
            "kimg",
            "mix_alpha_tracker/versionb_coef",
        ]
        parts = [f"step={step}"]
        for key in ordered_keys:
            if key in metrics:
                short_key = key.split("/")[-1].replace("time/", "")
                parts.append(f"{short_key}={self._fmt_console_value(metrics[key])}")
        if "drift_matching" in metrics:
            parts.append(f"mode={metrics['drift_matching']}")
        print("[train] " + " | ".join(parts), flush=True)

    def log(
        self,
        metrics: dict,
        step: Optional[int] = None,
        *,
        commit: bool = True,
    ) -> None:
        if self.rank != 0:
            return
        s = step if step is not None else self._step
        with open(self.log_file, "a") as f:
            f.write(json.dumps({"step": s, **metrics}) + "\n")
        self._emit_console_summary(s, metrics)
        if self.use_wandb and s >= self.wandb_log_min_step:
            # W&B defaults to commit=False when an explicit step is supplied.
            # Make commits explicit so a run that stops before the next logging
            # interval still publishes its latest metrics.
            self._wandb.log(metrics, step=s, commit=commit)

    def set_step(self, step: int) -> None:
        self._step = step

    def finish(self) -> None:
        if self.use_wandb and self.rank == 0:
            self._wandb.finish()



# ---------------------------------------------------------------------------
# MAE feature extraction
# ---------------------------------------------------------------------------

def _extract_mae_state_dict(state: Any) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
    if isinstance(state, dict):
        if "ema" in state and isinstance(state["ema"], dict):
            return state["ema"], "EMA weights"
        if "model" in state and isinstance(state["model"], dict):
            return state["model"], "model weights"
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"], "state_dict weights"
        if "module" in state and isinstance(state["module"], dict):
            return state["module"], "module weights"
        if any(k.startswith("encoder.") for k in state.keys()):
            return state, "weights"
    return None, "weights"


def _resolve_mae_cfg(cfg: dict, state: Any) -> dict:
    mae_cfg = {
        "num_classes": int(cfg.get("num_classes", 1000)),
        "in_channels": int(cfg.get("in_channels", 4)),
        "base_channels": 640,
        "patch_size": 2,
        "dropout_prob": 0.0,
        "layers": [3, 4, 6, 3],
        "use_bf16": bool(cfg.get("use_bf16", True)),
        "input_patch_size": 1 if bool(cfg.get("use_latent", True)) else 8,
        "use_remat": bool(cfg.get("mae_use_remat", False)),
        "fuse_stats": int(cfg.get("throughput_opt_level", 0)) >= 2,
    }

    candidates: List[Dict[str, Any]] = []
    if isinstance(state, dict):
        if isinstance(state.get("model_config"), dict):
            candidates.append(state["model_config"])
        if isinstance(state.get("hf_metadata"), dict):
            hf_meta_cfg = state["hf_metadata"].get("model_config")
            if isinstance(hf_meta_cfg, dict):
                candidates.append(hf_meta_cfg)
        if isinstance(state.get("config"), dict):
            nested_model_cfg = state["config"].get("model")
            if isinstance(nested_model_cfg, dict):
                candidates.append(nested_model_cfg)
            else:
                candidates.append(state["config"])

    for cand in candidates:
        for key in (
            "num_classes",
            "in_channels",
            "base_channels",
            "patch_size",
            "dropout_prob",
            "layers",
            "use_bf16",
            "input_patch_size",
        ):
            if key in cand and cand[key] is not None:
                mae_cfg[key] = cand[key]

    # MAE precision is determined by its checkpoint, not the generator config.
    # (Generator use_bf16 controls generator dtype only, not MAE.)

    sd, _ = _extract_mae_state_dict(state)
    if isinstance(sd, dict):
        conv1 = sd.get("encoder.conv1.weight")
        if isinstance(conv1, torch.Tensor) and conv1.ndim == 4:
            mae_cfg["base_channels"] = int(conv1.shape[0])
            projected_in = int(conv1.shape[1])
            in_ch = int(mae_cfg.get("in_channels", 4))
            if in_ch > 0 and projected_in % in_ch == 0:
                p2 = projected_in // in_ch
                p = int(round(p2 ** 0.5))
                if p * p == p2:
                    mae_cfg["input_patch_size"] = p

        fc_w = sd.get("fc.weight")
        if isinstance(fc_w, torch.Tensor) and fc_w.ndim == 2:
            mae_cfg["num_classes"] = int(fc_w.shape[0])

        inferred_layers: List[int] = []
        for stage_idx in range(8):
            prefix = f"encoder.stages.{stage_idx}."
            count = sum(
                1
                for key in sd.keys()
                if key.startswith(prefix) and key.endswith(".conv1.weight")
            )
            if count == 0:
                break
            inferred_layers.append(count)
        if inferred_layers:
            mae_cfg["layers"] = inferred_layers

    mae_cfg["layers"] = list(mae_cfg.get("layers", [3, 4, 6, 3]))
    mae_cfg["base_channels"] = int(mae_cfg.get("base_channels", 640))
    mae_cfg["in_channels"] = int(mae_cfg.get("in_channels", 4))
    mae_cfg["num_classes"] = int(mae_cfg.get("num_classes", 1000))
    mae_cfg["patch_size"] = int(mae_cfg.get("patch_size", 2))
    mae_cfg["input_patch_size"] = int(mae_cfg.get("input_patch_size", 1))
    mae_cfg["dropout_prob"] = float(mae_cfg.get("dropout_prob", 0.0))
    mae_cfg["use_bf16"] = bool(mae_cfg.get("use_bf16", True))
    # Runtime memory policy comes from the generator-training config, not the
    # checkpoint metadata. It does not alter MAE weights or forward values.
    mae_cfg["use_remat"] = bool(cfg.get("mae_use_remat", mae_cfg.get("use_remat", False)))
    mae_cfg["fuse_stats"] = int(cfg.get("throughput_opt_level", 0)) >= 2
    return mae_cfg


def load_mae(checkpoint_path: str, cfg: dict, device: torch.device) -> MAEResNet:
    """Load a pre-trained MAEResNet and freeze it."""
    state: Any = None
    if checkpoint_path:
        # This converted checkpoint contains both model and EMA trees.  Keep the
        # archive CPU-backed while selecting/loading one tree so each DDP rank
        # does not briefly materialize the full archive on its GPU.
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    mae_cfg = _resolve_mae_cfg(cfg, state)
    mae = build_mae_from_config(mae_cfg)
    print(f"[MAE] Build config: {mae_cfg}")

    if state is not None:
        sd, source_name = _extract_mae_state_dict(state)
        if sd is None:
            raise ValueError(f"Unsupported MAE checkpoint format: {checkpoint_path}")
        missing, unexpected = mae.load_state_dict(sd, strict=False)
        print(f"[MAE] Loaded {source_name} from {checkpoint_path}")
        if missing:
            print(f"[MAE] Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[MAE] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    mae = mae.to(device)

    mae.eval()
    for p in mae.parameters():
        p.requires_grad_(False)
    # Preserve fp32 MAE weights. bf16 compute is applied at callsites via autocast
    # when the checkpoint metadata says `use_bf16: true`, matching the JAX setup
    # more closely than converting stored weights to bf16.
    return mae


def resolve_feature_extractor_name(cfg: dict) -> str:
    """Resolve the configured frozen feature encoder to a canonical name."""
    raw_name = str(cfg.get("feature_extractor", "mae")).lower().strip()
    normalized = raw_name.replace("-", "_")
    if normalized in {"mae", "mae_resnet", "mae_resnet256"}:
        return "mae"
    bridge_aliases = {
        "dino_latent_bridge": "dino_latent_bridge",
        "dino_r50_latent_bridge": "dino_latent_bridge",
        "moco_latent_bridge": "moco_v2_latent_bridge",
        "moco_v2_latent_bridge": "moco_v2_latent_bridge",
        "moco_v2_r50_latent_bridge": "moco_v2_latent_bridge",
    }
    if normalized in bridge_aliases:
        return bridge_aliases[normalized]
    return canonical_ssl_backbone(normalized)


def load_feature_extractor(
    cfg: dict,
    device: torch.device,
) -> nn.Module:
    """Build the requested frozen feature encoder on ``device``."""
    extractor_name = resolve_feature_extractor_name(cfg)
    if extractor_name == "mae":
        checkpoint_path = (
            cfg.get("feature_checkpoint")
            or cfg.get("mae_checkpoint")
            or os.environ.get("MAE_CHECKPOINT", "")
        )
        if not checkpoint_path:
            raise ValueError("MAE feature extraction requires feature_checkpoint")
        return load_mae(str(checkpoint_path), cfg, device)

    if extractor_name in {"dino_latent_bridge", "moco_v2_latent_bridge"}:
        backbone_name = (
            "dino_resnet50"
            if extractor_name == "dino_latent_bridge"
            else "moco_v2_resnet50"
        )
        checkpoint_path = cfg.get("feature_checkpoint") or os.environ.get(
            "FEATURE_CHECKPOINT", ""
        )
        bridge_checkpoint = cfg.get("feature_bridge_checkpoint") or os.environ.get(
            "FEATURE_BRIDGE_CHECKPOINT", ""
        )
        if not checkpoint_path:
            raise ValueError(
                f"{extractor_name} requires the official feature_checkpoint"
            )
        if not bridge_checkpoint:
            raise ValueError(
                f"{extractor_name} requires feature_bridge_checkpoint"
            )
        model = SSLLatentBridgeFeatureExtractor(
            backbone_name,
            str(checkpoint_path),
            str(bridge_checkpoint),
            use_bf16=bool(cfg.get("feature_use_bf16", True)),
            use_remat=bool(cfg.get("feature_use_remat", False)),
            microbatch_size=int(cfg.get("feature_microbatch_size", 32)),
            spatial_pool=int(cfg.get("feature_spatial_pool", 2)),
            include_norm_x=bool(cfg.get("feature_include_norm_x", False)),
            verify_teacher_sha256=bool(
                cfg.get("feature_bridge_verify_teacher_sha256", True)
            ),
        ).to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    checkpoint_path = cfg.get("feature_checkpoint") or os.environ.get(
        "FEATURE_CHECKPOINT", ""
    )
    if not checkpoint_path:
        raise ValueError(
            f"{extractor_name} feature extraction requires feature_checkpoint"
        )
    return build_ssl_resnet_from_config(
        extractor_name,
        str(checkpoint_path),
        cfg,
        device,
    )


def build_feature_adapter_system(
    mae: MAEResNet,
    cfg: dict,
    device: torch.device,
    world_size: int,
) -> Tuple[
    Optional[nn.Module],
    Optional[FeatureAdapterSystem],
    Optional[torch.optim.Optimizer],
]:
    """Build the online adapter and its frozen generator-path target copy."""
    if not bool(cfg.get("feature_adapter", False)):
        return None, None, None

    objective = str(cfg.get("feature_adapter_objective", "supcon")).lower().strip()
    valid_objectives = {
        "supcon",
        "supcon_ce",
        "real_real_multipos_infonce",
        "real_real_multipos_infonce_mdino",
        "gen_real_multipos_infonce",
        "raw_drift_snr",
        "raw_drift_snr_consistency",
        "dino_cf_drift_realreal_infonce",
    }
    if objective not in valid_objectives:
        raise ValueError(
            "feature_adapter_objective must be one of "
            f"{sorted(valid_objectives)}, "
            f"got {objective!r}"
        )
    if objective in {
        "raw_drift_snr",
        "raw_drift_snr_consistency",
    } and resolve_feature_extractor_name(cfg) not in {
        "dino_resnet50",
        "moco_v2_resnet50",
    }:
        raise ValueError(
            f"{objective} currently requires a direct DINO/MoCo "
            "SSL ResNet feature extractor"
        )
    if objective == "raw_drift_snr":
        positive_count = int(cfg.get("pos_per_sample", 64))
        generated_count = int(cfg.get("gen_per_label", 32))
        real_pool_count = int(
            cfg.get("feature_adapter_real_pool_per_class", 32)
        )
        query_count = int(cfg.get("feature_adapter_query_count", 8))
        bank_count = int(cfg.get("feature_adapter_bank_count", 12))
        if min(real_pool_count, query_count, bank_count) <= 0:
            raise ValueError(
                "raw_drift_snr real-pool/query/bank counts must be positive"
            )
        if real_pool_count != query_count + 2 * bank_count:
            raise ValueError(
                "raw_drift_snr real pool must equal query/unused count plus "
                "two support banks"
            )
        if positive_count < real_pool_count:
            raise ValueError(
                "raw_drift_snr real pool exceeds pos_per_sample: "
                f"{real_pool_count}>{positive_count}"
            )
        if generated_count != query_count + 2 * bank_count:
            raise ValueError(
                "raw_drift_snr must partition every generated sample into "
                "Xq/Qa/Qb"
            )
        if bool(cfg.get("feature_adapter_snr_stopgrad_variance", False)):
            raise ValueError(
                "raw_drift_snr Version 2 requires differentiable null variance"
            )
    if (
        objective == "dino_cf_drift_realreal_infonce"
        and resolve_feature_extractor_name(cfg) != "dino_resnet50"
    ):
        raise ValueError(
            "dino_cf_drift_realreal_infonce requires the direct DINO ResNet50 "
            "feature extractor"
        )
    if (
        objective == "real_real_multipos_infonce_mdino"
        and resolve_feature_extractor_name(cfg) != "dino_resnet50"
    ):
        raise ValueError(
            "real_real_multipos_infonce_mdino requires the direct DINO "
            "ResNet50 feature extractor"
        )
    if objective == "real_real_multipos_infonce_mdino":
        resolution = int(cfg.get("resolution", 256))
        patch_size = int(cfg.get("feature_adapter_mdino_patch_size", 32))
        mask_ratio = float(cfg.get("feature_adapter_mdino_mask_ratio", 0.4))
        mdino_samples = int(
            cfg.get(
                "feature_adapter_mdino_samples_per_class",
                cfg.get("pos_per_sample", 64),
            )
        )
        positive_count = int(cfg.get("pos_per_sample", 64))
        mdino_weight = float(cfg.get("feature_adapter_mdino_weight", 1.0))
        smooth_weight = float(
            cfg.get("feature_adapter_mdino_smooth_l1_weight", 1.0)
        )
        cosine_weight = float(
            cfg.get("feature_adapter_mdino_cosine_weight", 0.0)
        )
        smooth_beta = float(cfg.get("feature_adapter_mdino_smooth_l1_beta", 1.0))
        predictor_bottleneck = int(
            cfg.get("feature_adapter_mdino_predictor_bottleneck", 128)
        )
        predictor_lr = float(
            cfg.get(
                "feature_adapter_mdino_predictor_lr",
                cfg.get("feature_adapter_lr", 1.0e-4),
            )
        )
        if patch_size <= 0 or resolution % patch_size:
            raise ValueError(
                "masked DINO patch size must be positive and divide the image "
                f"resolution, got patch={patch_size}, resolution={resolution}"
            )
        if not math.isfinite(mask_ratio) or not 0.0 < mask_ratio < 1.0:
            raise ValueError("masked DINO mask ratio must be finite and in (0, 1)")
        if mdino_samples <= 0 or mdino_samples > positive_count:
            raise ValueError(
                "masked DINO samples_per_class must be in "
                f"[1,{positive_count}], got {mdino_samples}"
            )
        if int(cfg.get("feature_adapter_mdino_teacher_last_k", 1)) != 1:
            raise ValueError("DINO-R50 masked distillation currently requires K=1")
        if predictor_bottleneck <= 0 or not math.isfinite(predictor_lr) or predictor_lr <= 0.0:
            raise ValueError("masked DINO predictor size and learning rate must be positive")
        if (
            not math.isfinite(mdino_weight)
            or mdino_weight <= 0.0
            or not math.isfinite(smooth_weight)
            or smooth_weight < 0.0
            or not math.isfinite(cosine_weight)
            or cosine_weight < 0.0
            or smooth_weight + cosine_weight <= 0.0
            or not math.isfinite(smooth_beta)
            or smooth_beta <= 0.0
        ):
            raise ValueError("masked DINO loss weights/beta are invalid")
        if len(
            tuple(
                cfg.get(
                    "feature_adapter_mdino_fill_values",
                    (-0.03, -0.088, -0.188),
                )
            )
        ) != 3:
            raise ValueError("masked DINO fill values must contain three RGB values")
        if float(cfg.get("feature_adapter_dropout", 0.0)) != 0.0:
            raise ValueError("masked DINO causal comparison requires adapter dropout=0")
    stages = canonical_adapter_stages(
        cfg.get("feature_adapter_keys", ["layer3", "layer4"])
    )
    base = int(mae.base_channels)
    stage_channels = {
        f"stage{index}": base * (2 ** (index - 1)) for index in range(1, 5)
    }
    online_raw = FeatureAdapterSystem(
        stage_channels,
        stages,
        bottleneck=int(cfg.get("feature_adapter_bottleneck", 64)),
        # The generated-to-real objective acts directly on GAP(adapter(map)).
        # Do not create an unused train-only projection head that could absorb
        # the contrastive loss without changing the metric seen by G.
        projection_dim=(
            0
            if objective in {
                "real_real_multipos_infonce",
                "real_real_multipos_infonce_mdino",
                "gen_real_multipos_infonce",
                "raw_drift_snr",
                "raw_drift_snr_consistency",
                "dino_cf_drift_realreal_infonce",
            }
            else int(cfg.get("feature_adapter_projection_dim", 128))
        ),
        masked_predictor_bottleneck=(
            int(cfg.get("feature_adapter_mdino_predictor_bottleneck", 128))
            if objective == "real_real_multipos_infonce_mdino"
            else 0
        ),
        num_classes=int(cfg.get("num_classes", 1000)),
        dropout=float(cfg.get("feature_adapter_dropout", 0.0)),
        use_ce=objective == "supcon_ce",
    ).to(device)
    if world_size > 1:
        online: nn.Module = DDP(
            online_raw,
            device_ids=[device.index],
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    else:
        online = online_raw

    # DDP construction broadcasts the online parameters from rank 0. Copy only
    # afterwards so every rank begins from an identical EMA target.
    target = copy.deepcopy(online_raw).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    adapter_lr = float(cfg.get("feature_adapter_lr", 1.0e-4))
    predictor_parameters = list(online_raw.masked_predictors.parameters())
    predictor_parameter_ids = {id(parameter) for parameter in predictor_parameters}
    base_parameters = [
        parameter
        for parameter in online_raw.parameters()
        if id(parameter) not in predictor_parameter_ids
    ]
    parameter_groups: list[dict[str, Any]] = [{"params": base_parameters}]
    if predictor_parameters:
        parameter_groups.append(
            {
                "params": predictor_parameters,
                "lr": float(cfg.get("feature_adapter_mdino_predictor_lr", adapter_lr)),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=adapter_lr,
        weight_decay=float(cfg.get("feature_adapter_weight_decay", 1.0e-4)),
        betas=(
            float(cfg.get("feature_adapter_adam_b1", 0.9)),
            float(cfg.get("feature_adapter_adam_b2", 0.999)),
        ),
    )
    return online, target, optimizer


def build_feature_discriminator_system(
    mae: MAEResNet,
    cfg: dict,
    device: torch.device,
    world_size: int,
) -> Tuple[Optional[nn.Module], Optional[torch.optim.Optimizer]]:
    """Build lightweight GAN heads over fixed terminal MAE stage maps."""
    if not bool(cfg.get("feature_gan", False)):
        return None, None

    stages = canonical_feature_gan_stages(
        cfg.get("feature_gan_keys", ["layer4"])
    )
    base = int(mae.base_channels)
    stage_channels = {
        f"stage{index}": base * (2 ** (index - 1)) for index in range(1, 5)
    }
    raw_discriminator = FrozenFeatureDiscriminator(
        stage_channels,
        stages,
        hidden_channels=int(cfg.get("feature_gan_hidden_channels", 128)),
        num_classes=int(cfg.get("num_classes", 1000)),
    ).to(device)
    if world_size > 1:
        discriminator: nn.Module = DDP(
            raw_discriminator,
            device_ids=[device.index],
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    else:
        discriminator = raw_discriminator
    optimizer = torch.optim.AdamW(
        raw_discriminator.parameters(),
        lr=float(cfg.get("feature_gan_lr", 2.0e-4)),
        weight_decay=float(cfg.get("feature_gan_weight_decay", 1.0e-4)),
        betas=(
            float(cfg.get("feature_gan_adam_b1", 0.0)),
            float(cfg.get("feature_gan_adam_b2", 0.9)),
        ),
    )
    return discriminator, optimizer


def _distributed_feature_gradient_norm(
    loss: torch.Tensor,
    features: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor:
    """L2 norm of a loss gradient at selected feature-map boundaries.

    Measuring at MAE stage outputs avoids an additional backward through the
    frozen backbone.  The all-rank norm gives every DDP worker the same GAN
    scale, while the ratio cancels the common world-size factor.
    """
    gradients = torch.autograd.grad(
        loss,
        features,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            norm_sq = norm_sq + gradient.detach().double().square().sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM)
    return norm_sq.clamp_min(0.0).sqrt().float()


# ---------------------------------------------------------------------------
# CFG scale sampling (matches JAX train.py)
# ---------------------------------------------------------------------------

def sample_cfg(
    bsz: int,
    cfg_min: float,
    cfg_max: float,
    neg_cfg_pw: float,
    no_cfg_frac: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample per-sample CFG scales; power-law distribution on log scale."""
    frac = torch.rand(bsz, device=device)
    pw = 1.0 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = torch.exp(math.log(cfg_min) + frac * (math.log(cfg_max) - math.log(cfg_min)))
    else:
        cfg = (cfg_min ** pw + frac * (cfg_max ** pw - cfg_min ** pw)) ** (1.0 / pw)

    if no_cfg_frac > 0.0:
        mask = torch.rand(bsz, device=device) < no_cfg_frac
        cfg = torch.where(mask, torch.ones_like(cfg), cfg)
    return cfg


# ---------------------------------------------------------------------------
# Drift loss computation (feature-by-feature)
# ---------------------------------------------------------------------------

def compute_drift_loss_from_features(
    gen_feats: Dict[str, torch.Tensor],          # {name: (B*G, T, D)}
    pos_feats: Dict[str, torch.Tensor],           # {name: (B*P, T, D)} after split
    neg_feats: Optional[Dict[str, torch.Tensor]], # {name: (B*N, T, D)} or None
    B: int,
    G: int,
    P: int,
    N: int,
    weight_neg: Optional[torch.Tensor],           # (B, N) or None
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    drift_matching: str = "rev-drift",
    compute_wpos_stats: bool = False,
    per_sample_fnorm: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,
    active_mask_neg: Optional[torch.Tensor] = None,
    decouple_weight_from_coupling: bool = False,
    self_mask_on_raw: bool = False,
    global_scale_stats: bool = True,
    global_fnorm_stats: bool = True,
    mix_alpha: float = 0.0,
    mix_R_list_baseline: Tuple[float, ...] = (0.2, 0.05, 0.02),
    compute_raw_winner_stats_flag: bool = False,
    collect_diagnostics: bool = True,
    fuse_fnorm_across_R: bool = False,
    feature_loss_weights: Optional[Dict[str, float]] = None,
    prune_zero_weight_features: bool = False,
    dual_drift_share_distances: bool = True,
    rev_drift_top_p: float = 1.0,
    fwd_drift_top_p: float = 1.0,
    drift_top_p_min_keep: int = 1,
    drift_top_k_pos: int = 0,
    drift_top_k_neg: int = 0,
    drift_top_k_groups: Optional[Tuple[str, ...]] = None,
    feature_temperature_multipliers: Optional[Dict[str, float]] = None,
    rev_drift_force_multiplier: float = 1.0,
    rev_drift_affinity_kernel: str = "exponential",
    rev_drift_kernel_shape: float = 1.0,
    rev_drift_kernel_adaptive_k_pos: int = 0,
    rev_drift_kernel_adaptive_k_neg: int = 0,
    rev_drift_kernel_adaptive_margin: float = 1.05,
    rev_drift_kernel_mix_weight: float = 0.5,
    rev_drift_kernel_temperature_mix: Tuple[float, ...] = (),
    rev_drift_kernel_temperature_mix_weights: Tuple[float, ...] = (),
    historical_feats: Optional[Dict[str, torch.Tensor]] = None,
    historical_count: int = 0,
    weight_gen: Optional[torch.Tensor] = None,
    weight_history: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total drift loss by summing over all feature maps.

    Each feature map has shape (B*X, T, D), reshaped to (B*T, X, D) before
    passing to drift_loss_imagenet — matching the official JAX convention where
    B and the spatial token axis are merged into the batch dimension.

    When `compute_raw_winner_stats_flag=True`, raw L2 winner counts are
    computed once at image level: mean-pool layer1..layer4 over tokens, concat
    into a single (D1+D2+D3+D4,) vector per image, then cdist. Emitted under
    keys "raw/pos_winner_count", "raw/pos_winner_total",
    "raw/gen_winner_count", "raw/gen_winner_total".

    ``feature_loss_weights`` can skip a feature with weight 0 or rescale its
    contribution. Raw Hedge winner statistics are deliberately computed before
    this filtering, so stochastic loss selection never changes their inputs.

    When ``prune_zero_weight_features`` is enabled, zero-weight tensors are
    removed from the input dictionaries after the full Hedge statistic has
    been collected. This intentionally mutates those dictionaries so skipped
    real features and unused generated autograd branches can be released before
    the selected drift losses and backward pass.
    """
    total_loss = torch.tensor(0.0, device=next(iter(gen_feats.values())).device)
    total_info: Dict[str, float] = {}
    _raw_collected = False

    mode = str(drift_matching).lower().strip()
    use_version_b   = mode in ("fwd-drift",)
    use_mixed_bv    = mode in ("dual-drift",)
    if not any([use_version_b, use_mixed_bv]) and mode not in ("rev-drift",):
        raise ValueError(
            f"Unknown drift_matching={drift_matching!r}. "
            "Use 'rev-drift', 'fwd-drift', or 'dual-drift'."
        )

    # Image-level winner stats: mean-pool layer1..layer4 over tokens, concat to
    # a single (D1+D2+D3+D4,) vector per image, then cdist. Replaces the older
    # first-key (norm_x) per-token approach.
    if compute_raw_winner_stats_flag:
        layer_keys = ["layer1", "layer2", "layer3", "layer4"]
        if all(k in gen_feats and k in pos_feats for k in layer_keys):
            with torch.no_grad():
                pool_gen_list = [gen_feats[k].detach().float().mean(dim=1) for k in layer_keys]
                pool_pos_list = [pos_feats[k].detach().float().mean(dim=1) for k in layer_keys]
                gen_pool = torch.cat(pool_gen_list, dim=-1).reshape(B, G, -1)  # (B, G, D_concat)
                pos_pool = torch.cat(pool_pos_list, dim=-1).reshape(B, P, -1)  # (B, P, D_concat)
                dist_pos_metric = _cdist_batched(gen_pool, pos_pool)
                _ws = compute_raw_winner_stats(dist_pos_metric)
            for _k, _v in _ws.items():
                total_info[f"raw/{_k}"] = _v
            _raw_collected = True
            del gen_pool, pos_pool, dist_pos_metric, pool_gen_list, pool_pos_list

    if prune_zero_weight_features and feature_loss_weights is not None:
        pruned_names = [
            name
            for name in tuple(gen_feats.keys())
            if float(feature_loss_weights.get(name, 1.0)) <= 0.0
        ]
        for name in pruned_names:
            gen_feats.pop(name, None)
            pos_feats.pop(name, None)
            if neg_feats is not None:
                neg_feats.pop(name, None)
            if historical_feats is not None:
                historical_feats.pop(name, None)
        total_info["stochastic_stage/pruned_feature_count"] = float(
            len(pruned_names)
        )

    for name in gen_feats:
        feature_loss_weight = (
            float(feature_loss_weights.get(name, 1.0))
            if feature_loss_weights is not None
            else 1.0
        )
        if feature_loss_weight <= 0.0:
            continue
        temperature_multiplier = _feature_temperature_multiplier(
            name,
            feature_temperature_multipliers,
        )
        feature_R_list = tuple(
            round(float(R) * temperature_multiplier, 12) for R in R_list
        )
        feature_mix_R_list_baseline = tuple(
            round(float(R) * temperature_multiplier, 12)
            for R in mix_R_list_baseline
        )
        feature_top_k_pos, feature_top_k_neg = _top_k_for_feature(
            name,
            drift_top_k_pos,
            drift_top_k_neg,
            drift_top_k_groups,
        )
        gen_f  = gen_feats[name]     # (B*G, T, D)
        pos_f  = pos_feats.get(name)
        if pos_f is None:
            continue
        T = gen_f.shape[1]
        D = gen_f.shape[2]

        # Reshape: (B*X, T, D) → (B, X, T, D) → (B*T, X, D)
        gen_bt = rearrange(gen_f,  "(b g) t d -> b g t d", b=B, g=G)  # (B, G, T, D)
        pos_bt = rearrange(pos_f,  "(b p) t d -> b p t d", b=B, p=P)  # (B, P, T, D)
        gen_bt = rearrange(gen_bt, "b g t d -> (b t) g d")            # (B*T, G, D)
        pos_bt = rearrange(pos_bt, "b p t d -> (b t) p d")            # (B*T, P, D)

        # Raw L2 winner stats — computed once on the first feature key only.
        # In mixed mode, the loss function reuses its own dist_pos slice
        # (return_raw_winner_stats=True below); for other modes we compute a
        # dedicated cdist here.
        emit_raw_here = compute_raw_winner_stats_flag and not _raw_collected
        if emit_raw_here and not use_mixed_bv:
            with torch.no_grad():
                dist_pos_metric = _cdist_batched(
                    gen_bt.detach().float(), pos_bt.detach().float()
                )
                _ws = compute_raw_winner_stats(dist_pos_metric)
            for _k, _v in _ws.items():
                total_info[f"raw/{_k}"] = _v
            _raw_collected = True

        neg_bt = None
        w_neg  = None
        if neg_feats is not None and name in neg_feats and N > 0:
            nf = neg_feats[name]   # (B*N, T, D)
            neg_bt = rearrange(rearrange(nf, "(b n) t d -> b n t d", b=B, n=N),
                               "b n t d -> (b t) n d")  # (B*T, N, D)
            if weight_neg is not None:
                # weight_neg: (B, N) → (B*T, N)
                w_neg = repeat(weight_neg, "b n -> (b t) n", t=T)

        historical_bt = None
        w_gen = repeat(weight_gen, "b g -> (b t) g", t=T) if weight_gen is not None else None
        w_history = None
        if historical_feats is not None and name in historical_feats:
            if historical_count <= 0:
                raise ValueError("historical_count must be positive when replay features are provided")
            hf = historical_feats[name]
            historical_bt = rearrange(
                rearrange(hf, "(b h) t d -> b h t d", b=B, h=historical_count),
                "b h t d -> (b t) h d",
            )
            if weight_history is not None:
                w_history = repeat(weight_history, "b h -> (b t) h", t=T)

        # Expand active masks from (B, *) to (B*T, *) to match the reshaped feats.
        am_pos = None
        am_neg = None
        if active_mask_pos is not None:
            am_pos = repeat(active_mask_pos, "b p -> (b t) p", t=T)
        if active_mask_neg is not None:
            am_neg = repeat(active_mask_neg, "b n -> (b t) n", t=T)

        if use_version_b:
            if historical_bt is not None:
                raise ValueError("historical generated replay currently supports rev-drift only")
            loss, info = drift_loss_imagenet_colwise(
                gen=gen_bt,
                fixed_pos=pos_bt,
                fixed_neg=neg_bt,
                weight_neg=w_neg,
                R_list=feature_R_list,
                coupling=True,
                compute_wpos_stats=compute_wpos_stats,
                per_sample_fnorm=per_sample_fnorm,
                active_mask_pos=am_pos,
                active_mask_neg=am_neg,
                decouple_weight_from_coupling=decouple_weight_from_coupling,
                self_mask_on_raw=self_mask_on_raw,
                global_scale_stats=global_scale_stats,
                global_fnorm_stats=global_fnorm_stats,
                top_p=fwd_drift_top_p,
                top_p_min_keep=drift_top_p_min_keep,
                top_k_pos=feature_top_k_pos,
                top_k_neg=feature_top_k_neg,
            )
        elif use_mixed_bv:
            if historical_bt is not None:
                raise ValueError("historical generated replay currently supports rev-drift only")
            loss, info = drift_loss_imagenet_mixed(
                gen=gen_bt,
                fixed_pos=pos_bt,
                fixed_neg=neg_bt,
                weight_neg=w_neg,
                alpha=mix_alpha,
                R_list_baseline=feature_mix_R_list_baseline,
                R_list_versionb=feature_R_list,
                compute_wpos_stats=compute_wpos_stats,
                per_sample_fnorm=per_sample_fnorm,
                active_mask_pos=am_pos,
                active_mask_neg=am_neg,
                decouple_weight_from_coupling=decouple_weight_from_coupling,
                self_mask_on_raw=self_mask_on_raw,
                return_raw_winner_stats=emit_raw_here,
                global_scale_stats=global_scale_stats,
                global_fnorm_stats=global_fnorm_stats,
                collect_diagnostics=collect_diagnostics,
                share_distances=dual_drift_share_distances,
                baseline_top_p=rev_drift_top_p,
                versionb_top_p=fwd_drift_top_p,
                top_p_min_keep=drift_top_p_min_keep,
                top_k_pos=feature_top_k_pos,
                top_k_neg=feature_top_k_neg,
            )
            if emit_raw_here:
                _raw_collected = True
        else:
            loss, info = drift_loss_imagenet(
                gen=gen_bt,
                fixed_pos=pos_bt,
                fixed_neg=neg_bt,
                weight_gen=w_gen,
                weight_neg=w_neg,
                R_list=feature_R_list,
                compute_wpos_stats=compute_wpos_stats,
                active_mask_pos=am_pos,
                active_mask_neg=am_neg,
                global_scale_stats=global_scale_stats,
                global_fnorm_stats=global_fnorm_stats,
                top_p=rev_drift_top_p,
                top_p_min_keep=drift_top_p_min_keep,
                top_k_pos=feature_top_k_pos,
                top_k_neg=feature_top_k_neg,
                force_multiplier=rev_drift_force_multiplier,
                affinity_kernel=rev_drift_affinity_kernel,
                kernel_shape=rev_drift_kernel_shape,
                kernel_adaptive_k_pos=rev_drift_kernel_adaptive_k_pos,
                kernel_adaptive_k_neg=rev_drift_kernel_adaptive_k_neg,
                kernel_adaptive_margin=rev_drift_kernel_adaptive_margin,
                kernel_mix_weight=rev_drift_kernel_mix_weight,
                kernel_temperature_mix=rev_drift_kernel_temperature_mix,
                kernel_temperature_mix_weights=rev_drift_kernel_temperature_mix_weights,
                historical_gen=historical_bt,
                weight_history=w_history,
                collect_diagnostics=collect_diagnostics,
                fuse_fnorm_across_R=fuse_fnorm_across_R,
            )
        total_loss = total_loss + feature_loss_weight * loss.mean()
        for k, v in info.items():
            v_f = float(v.mean().item() if isinstance(v, torch.Tensor) else v)
            # raw/* keys (winner stats for adaptive mix-alpha) stay at the top
            # level so MixAlphaTracker.update() can read them directly.
            if k.startswith("raw/"):
                total_info[k] = v_f
            else:
                total_info[f"{k}/{name}"] = v_f

    return total_loss, total_info



# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _historical_replay_ratio_for_step(cfg: dict, step: int, active: bool) -> float:
    """Return a static or linearly ramped historical-replay mass fraction."""
    if not active:
        return 0.0
    ratio_end = float(cfg.get("historical_gen_replay_ratio", 0.0))
    ratio_start_value = cfg.get("historical_gen_replay_ratio_start", None)
    ratio_start = (
        ratio_end if ratio_start_value is None else float(ratio_start_value)
    )
    for name, value in (("start", ratio_start), ("end", ratio_end)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"historical replay ratio {name} must be finite and in [0, 1], "
                f"got {value!r}"
            )
    ramp_start = int(cfg.get("historical_gen_replay_ratio_ramp_start_step", 0))
    ramp_end = int(
        cfg.get("historical_gen_replay_ratio_ramp_end_step", ramp_start)
    )
    if ramp_end < ramp_start:
        raise ValueError(
            "historical replay ratio ramp end step must be >= start step"
        )
    if ratio_start == ratio_end or ramp_end == ramp_start:
        return ratio_end
    if step <= ramp_start:
        return ratio_start
    if step >= ramp_end:
        return ratio_end
    fraction = (int(step) - ramp_start) / float(ramp_end - ramp_start)
    return ratio_start + fraction * (ratio_end - ratio_start)


def train_step(
    generator: nn.Module,
    feature_extractor: nn.Module,
    optimizer: torch.optim.Optimizer,
    labels: torch.Tensor,
    pos_samples: torch.Tensor,    # (B, P, C, H, W)
    neg_samples: torch.Tensor,    # (B, N, C, H, W)
    device: torch.device,
    step: int,
    cfg: dict,
    mix_alpha_tracker: Optional["MixAlphaTracker"] = None,
    feature_adapter: Optional[nn.Module] = None,
    feature_adapter_target: Optional[FeatureAdapterSystem] = None,
    feature_adapter_optimizer: Optional[torch.optim.Optimizer] = None,
    feature_adapter_bank_ready: bool = True,
    feature_adapter_real_samples: Optional[torch.Tensor] = None,
    feature_discriminator: Optional[nn.Module] = None,
    feature_discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    historical_samples: Optional[torch.Tensor] = None,
    fresh_historical_count: int = 0,
    adversarial_system: Optional[AdversarialDriftSystem] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """One generator optimization step.

    Args:
        generator:   DitGen wrapped in DDP (or raw).
        feature_extractor: Frozen MAE/DINO/MoCo feature encoder.
        optimizer:   AdamW.
        labels:      Class labels (B,).
        pos_samples: Real positive images (B, P, C, H, W).
        neg_samples: Real negative images (B, N, C, H, W).
        device:      Compute device.
        step:        Global training step.
        cfg:         Flat config dict.

    Returns:
        loss: scalar Tensor.
        metrics: dict of scalar values.
    """
    B = labels.shape[0]
    P = pos_samples.shape[1]
    N = neg_samples.shape[1]
    G = int(cfg.get("gen_per_label", 64))
    if adversarial_system is not None and (
        feature_adapter is not None or feature_discriminator is not None
    ):
        raise ValueError("Adversarial experiments cannot also change the DINO metric")
    adversarial_mode = adversarial_system.mode if adversarial_system is not None else None
    adversarial_teacher_real_features: Optional[Dict[str, torch.Tensor]] = None
    fresh_historical_count = int(fresh_historical_count)
    if fresh_historical_count < 0:
        raise ValueError("fresh_historical_count must be non-negative")
    if historical_samples is not None and fresh_historical_count > 0:
        raise ValueError(
            "frozen historical samples and fresh-current anchors are mutually exclusive"
        )
    H = (
        int(historical_samples.shape[1])
        if historical_samples is not None
        else fresh_historical_count
    )
    adapter_parts = (
        feature_adapter,
        feature_adapter_target,
        feature_adapter_optimizer,
    )
    if any(part is not None for part in adapter_parts) and not all(
        part is not None for part in adapter_parts
    ):
        raise ValueError(
            "feature adapter, EMA target, and optimizer must be provided together"
        )
    adapter_objective = str(
        cfg.get("feature_adapter_objective", "supcon")
    ).lower().strip()
    generated_real_adapter = (
        feature_adapter is not None
        and adapter_objective
        in {
            "real_real_multipos_infonce",
            "real_real_multipos_infonce_mdino",
            "gen_real_multipos_infonce",
            "raw_drift_snr",
            "raw_drift_snr_consistency",
            "dino_cf_drift_realreal_infonce",
        }
    )
    masked_dino_adapter = (
        feature_adapter is not None
        and adapter_objective == "real_real_multipos_infonce_mdino"
    )
    raw_drift_v2_adapter = (
        feature_adapter is not None
        and adapter_objective == "raw_drift_snr"
    )
    raw_drift_adapter = (
        feature_adapter is not None
        and adapter_objective
        in {"raw_drift_snr", "raw_drift_snr_consistency"}
    )
    counterfactual_drift_adapter = (
        feature_adapter is not None
        and adapter_objective == "dino_cf_drift_realreal_infonce"
    )
    raw_map_adapter = raw_drift_adapter or counterfactual_drift_adapter
    adapter_update_frequency = int(cfg.get("feature_adapter_update_freq", 1))
    adapter_start_step = int(cfg.get("feature_adapter_start_step", 0))
    adapter_freeze_step = int(
        cfg.get(
            "feature_adapter_freeze_step_resolved",
            cfg.get("feature_adapter_freeze_step", 0),
        )
        or 0
    )
    if feature_adapter is not None:
        if adapter_update_frequency <= 0:
            raise ValueError("feature_adapter_update_freq must be positive")
        if adapter_start_step < 0:
            raise ValueError("feature_adapter_start_step must be non-negative")
        if adapter_freeze_step < 0:
            raise ValueError("feature_adapter_freeze_step must be non-negative")
    adapter_update_due = (
        feature_adapter is not None
        and bool(feature_adapter_bank_ready)
        and step >= adapter_start_step
        and (adapter_freeze_step <= 0 or step < adapter_freeze_step)
        and (step - adapter_start_step) % adapter_update_frequency == 0
    )
    adapter_real_pool_count = int(
        cfg.get("feature_adapter_real_pool_per_class", 32)
    )
    if raw_drift_v2_adapter and adapter_update_due:
        if feature_adapter_real_samples is None:
            raise ValueError(
                "raw_drift_snr update requires a separate distinct-real pool"
            )
        if (
            feature_adapter_real_samples.ndim != pos_samples.ndim
            or feature_adapter_real_samples.shape[0] != B
            or feature_adapter_real_samples.shape[1] != adapter_real_pool_count
            or tuple(feature_adapter_real_samples.shape[2:])
            != tuple(pos_samples.shape[2:])
        ):
            raise ValueError(
                "raw_drift_snr distinct-real pool must be "
                f"[B,{adapter_real_pool_count},...], got "
                f"{tuple(feature_adapter_real_samples.shape)}"
            )
    elif feature_adapter_real_samples is not None:
        raise ValueError(
            "feature_adapter_real_samples is valid only on a raw_drift_snr update"
        )
    discriminator_parts = (
        feature_discriminator,
        feature_discriminator_optimizer,
    )
    if any(part is not None for part in discriminator_parts) and not all(
        part is not None for part in discriminator_parts
    ):
        raise ValueError(
            "feature discriminator and optimizer must be provided together"
        )

    cfg_min     = float(cfg.get("cfg_min", 1.0))
    cfg_max     = float(cfg.get("cfg_max", 4.0))
    neg_cfg_pw  = float(cfg.get("neg_cfg_pw", 5.0))
    no_cfg_frac = float(cfg.get("no_cfg_frac", 0.0))
    R_list         = tuple(cfg.get("R_list", [0.2, 0.05, 0.02]))
    drift_matching = str(cfg.get("drift_matching", "rev-drift")).lower().strip()
    compute_wpos_stats = bool(cfg.get("compute_wpos_stats", True))
    compute_wpos_stats_every_k = int(cfg.get("compute_wpos_stats_every_k", 1))
    if compute_wpos_stats:
        compute_wpos_stats = (step % max(1, compute_wpos_stats_every_k) == 0)
    per_sample_fnorm = bool(cfg.get("per_sample_fnorm", False))
    decouple_weight_from_coupling = bool(cfg.get("decouple_weight_from_coupling", False))
    self_mask_on_raw = bool(cfg.get("self_mask_on_raw", False))
    global_scale_stats = bool(cfg.get("global_scale_stats", True))
    global_fnorm_stats = bool(cfg.get("global_fnorm_stats", True))
    dual_drift_share_distances = bool(
        cfg.get("dual_drift_share_distances", True)
    )
    rev_drift_top_p = float(cfg.get("rev_drift_top_p", 1.0))
    fwd_drift_top_p = float(cfg.get("fwd_drift_top_p", 1.0))
    drift_top_p_min_keep = int(cfg.get("drift_top_p_min_keep", 1))
    drift_top_k_pos = int(cfg.get("drift_top_k_pos", 0))
    drift_top_k_neg = int(cfg.get("drift_top_k_neg", 0))
    drift_top_k_groups = _resolve_drift_top_k_groups(cfg)
    rev_drift_force_multiplier = float(
        cfg.get("rev_drift_force_multiplier", 1.0)
    )
    rev_drift_affinity_kernel = str(
        cfg.get("rev_drift_affinity_kernel", "exponential")
    )
    rev_drift_kernel_shape = float(cfg.get("rev_drift_kernel_shape", 1.0))
    rev_drift_kernel_adaptive_k_pos = int(
        cfg.get("rev_drift_kernel_adaptive_k_pos", 0)
    )
    rev_drift_kernel_adaptive_k_neg = int(
        cfg.get("rev_drift_kernel_adaptive_k_neg", 0)
    )
    rev_drift_kernel_adaptive_margin = float(
        cfg.get("rev_drift_kernel_adaptive_margin", 1.05)
    )
    rev_drift_kernel_mix_weight = float(
        cfg.get("rev_drift_kernel_mix_weight", 0.5)
    )
    rev_drift_kernel_temperature_mix = tuple(
        float(value) for value in cfg.get("rev_drift_kernel_temperature_mix", [])
    )
    rev_drift_kernel_temperature_mix_weights = tuple(
        float(value)
        for value in cfg.get("rev_drift_kernel_temperature_mix_weights", [])
    )
    feature_temperature_multipliers = _resolve_layer_temperature_multipliers(cfg)
    feature_loss_group_weights = _resolve_feature_loss_group_weights(cfg)
    feature_loss_group_normalize = bool(
        cfg.get("feature_loss_group_normalize", False)
    )
    topk_diagnostic_steps = int(cfg.get("topk_diagnostic_steps", 0))
    topk_diagnostic_pos = int(cfg.get("topk_diagnostic_pos", 16))
    topk_diagnostic_neg = int(cfg.get("topk_diagnostic_neg", 40))
    cfg_batch_slice = bool(cfg.get("cfg_batch_slice", False))

    # mix_alpha = coefficient on V_rev-drift (fwd-drift_coef = 1 - mix_alpha).
    # For non-mixed configs, the loss only computes one of the two forces, so we
    # log the coefficient that matches what's actually being optimized:
    #   rev-drift  → mix_alpha = 1.0 (fwd-drift_coef = 0)
    #   fwd-drift  → mix_alpha = 0.0 (fwd-drift_coef = 1)
    # For dual-drift, mix_alpha follows the adaptive tracker
    # (when mix_alpha_adaptive=True) or the linear schedule between
    # [mix_alpha_start_step, mix_alpha_end_step].
    mix_R_list_baseline = tuple(cfg.get("mix_baseline_R_list", [0.2, 0.05, 0.02]))
    # The adaptive-alpha ramp may intentionally finish before (or after) the
    # optimizer training run. Fall back to total_steps for older configs.
    total_steps_for_mix = int(
        cfg.get("mix_alpha_horizon_steps", cfg.get("total_steps", 1))
    )
    mix_alpha_adaptive = bool(cfg.get("mix_alpha_adaptive", False))
    if drift_matching in ("rev-drift",):
        mix_alpha = 1.0
    elif drift_matching in ("fwd-drift",):
        mix_alpha = 0.0
    elif mix_alpha_adaptive and mix_alpha_tracker is not None:
        mix_alpha = mix_alpha_tracker.compute_mix_alpha(step=step, total_steps=total_steps_for_mix)
    else:
        mix_alpha_start_step = int(cfg.get("mix_alpha_start_step", 0))
        mix_alpha_end_step = int(cfg.get("mix_alpha_end_step", total_steps_for_mix))
        if step <= mix_alpha_start_step:
            mix_alpha = 0.0
        elif step >= mix_alpha_end_step:
            mix_alpha = 1.0
        else:
            denom = max(1, mix_alpha_end_step - mix_alpha_start_step)
            mix_alpha = (step - mix_alpha_start_step) / denom
    act_kwargs  = cfg.get("activation_kwargs", {
        "patch_mean_size": [2, 4], "patch_std_size": [2, 4],
        "use_std": True, "use_mean": True, "with_global": True, "every_k_block": 2,
    })
    profile_train_step = bool(cfg.get("profile_train_step", False))
    throughput_opt_level = int(cfg.get("throughput_opt_level", 0))
    diagnostics_every_k = int(
        cfg.get("diagnostics_every_k", cfg.get("log_every_k", 20))
    )
    collect_diagnostics = (
        throughput_opt_level < 1
        or step % max(1, diagnostics_every_k) == 0
    )
    if throughput_opt_level >= 4 and not collect_diagnostics:
        # W_pos/support summaries are logging-only and contain several device
        # synchronizations; keep their configured cadence on diagnostic steps.
        compute_wpos_stats = False
    collect_mix_alpha_stats = _collect_mix_alpha_stats_this_step(
        throughput_opt_level=throughput_opt_level,
        collect_diagnostics=collect_diagnostics,
        drift_matching=drift_matching,
    )
    collect_training_metrics = _collect_training_metrics_this_step(
        throughput_opt_level=throughput_opt_level,
        step=step,
        log_every_k=int(cfg.get("log_every_k", 20)),
    )
    fuse_fnorm_across_R = throughput_opt_level >= 4
    stochastic_feature_stage_loss = bool(
        cfg.get("stochastic_feature_stage_loss", False)
    )
    stochastic_feature_stage_count = int(
        cfg.get("stochastic_feature_stage_count", 2)
    )
    stochastic_feature_stage_seed = int(
        cfg.get("stochastic_feature_stage_seed", cfg.get("seed", 0))
    )
    prune_skipped_feature_tensors = bool(
        cfg.get("prune_skipped_feature_tensors", False)
    )
    selected_feature_stages: Optional[Tuple[str, ...]] = None
    if stochastic_feature_stage_loss:
        selected_feature_stages = _sample_stochastic_feature_stages(
            stage_count=stochastic_feature_stage_count,
            seed=stochastic_feature_stage_seed,
            step=step,
        )
    prof_marks: List[Tuple[str, float]] = []

    def _prof_mark(nm: str) -> None:
        if not profile_train_step:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prof_marks.append((nm, time.perf_counter()))

    _prof_mark("step_start")

    # 1. Sample per-label CFG scales
    cfg_scales = sample_cfg(B, cfg_min, cfg_max, neg_cfg_pw, no_cfg_frac, device)

    raw_gen = generator.module if hasattr(generator, "module") else generator
    gen_use_bf16 = _gen_use_bf16(generator)

    # 1b. Option B — cfg_batch_slice: single cfg for whole batch + slice pos/neg.
    # Keeps distance matrix exactly 16×48 by loading max (pos=16, neg=max) then
    # slicing raw samples BEFORE MAE based on batch-uniform cfg.
    if cfg_batch_slice and N > 0:
        batch_cfg = cfg_scales[0].item()
        cfg_scales = cfg_scales[0:1].expand(B).contiguous()
        pos_base = int(cfg.get("cfg_ratio_pos_base", 16))
        pos_min  = int(cfg.get("cfg_ratio_pos_min", 4))
        total_budget = int(cfg.get("cfg_ratio_total_budget", pos_base + 16))
        t = max(0.0, min(1.0, (batch_cfg - cfg_min) / max(cfg_max - cfg_min, 1e-3)))
        # Stochastic rounding: e.g. 14.2 → 14 w.p. 0.8, 15 w.p. 0.2.
        # neg_k derived as total_budget - pos_k so pos+neg stays at total_budget.
        pos_k = _stochastic_round(pos_base - (pos_base - pos_min) * t)
        pos_k = max(1, min(P, int(pos_k)))
        neg_k = max(0, min(N, total_budget - pos_k))
        pos_samples = pos_samples[:, :pos_k].contiguous()
        neg_samples = neg_samples[:, :neg_k].contiguous()
        P = pos_k
        N = neg_k

    # 1c. cfg_uncond_split — partition the N "uncond" (mixed-class real) latents
    # between attraction (with pos_real) and repulsion (with old_gen) by cfg.
    # All weights = 1. Single cfg for the whole batch so x is scalar and we can
    # slice the (latent) samples BEFORE MAE — total MAE compute = baseline
    # (B*(P+N)), and the loss matrix C_p + C_g + C_n stays at P+G+N = 48.
    #   x = round(x_max * (1 - (cfg - cfg_min) / (cfg_max - cfg_min)))
    #   pos_samples_new = cat(pos_samples, neg_samples[:, :x])   (P_new = P + x)
    #   neg_samples_new = neg_samples[:, x:]                     (N_new = N - x)
    cfg_uncond_split = bool(cfg.get("cfg_uncond_split", False))
    uncond_split_x_max = cfg.get("cfg_uncond_split_x_max", None)
    if cfg_uncond_split and N > 0:
        x_max_val = int(uncond_split_x_max) if uncond_split_x_max is not None else N
        x_max_val = max(0, min(N, x_max_val))
        batch_cfg = cfg_scales[0].item()
        cfg_scales = cfg_scales[0:1].expand(B).contiguous()
        t_split = max(0.0, min(1.0, (batch_cfg - cfg_min) / max(cfg_max - cfg_min, 1e-3)))
        x = int(round(x_max_val * (1.0 - t_split)))
        x = max(0, min(N, x))

        new_pos_samples = torch.cat([pos_samples, neg_samples[:, :x]], dim=1).contiguous()
        new_neg_samples = neg_samples[:, x:].contiguous()
        pos_samples = new_pos_samples
        neg_samples = new_neg_samples
        P = P + x
        N = N - x

    # Matched-geometry causal control for historical replay. Generate H
    # independent targets from the current generator, but isolate their RNG so
    # the G query particles use the same random stream as the other controls.
    if fresh_historical_count > 0:
        expanded_history_labels = repeat(
            labels, "b -> (b h)", h=fresh_historical_count
        )
        expanded_history_cfg = repeat(
            cfg_scales, "b -> (b h)", h=fresh_historical_count
        )
        global_rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        anchor_seed = (
            int(cfg.get("seed", 42))
            + 1_000_003 * int(step)
            + 97_409 * int(global_rank)
            + 31_337
        ) % (2**63 - 1)
        fork_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(anchor_seed)
            with torch.no_grad(), _amp_ctx(gen_use_bf16):
                historical_samples = raw_gen(
                    expanded_history_labels,
                    cfg_scale=expanded_history_cfg,
                    train=True,
                )["samples"].detach()
        historical_samples = historical_samples.reshape(
            B, fresh_historical_count, *historical_samples.shape[1:]
        )
        _prof_mark("after_fresh_history_generator")

    # 2. Compute features of real samples (detached)
    #    Stack pos and neg along sample dim: (B*(P+N), C, H, W)
    real_parts = [
        pos_samples.reshape(B * P, *pos_samples.shape[2:]),
        neg_samples.reshape(B * N, *neg_samples.shape[2:]),
    ]
    if historical_samples is not None:
        real_parts.append(
            historical_samples.reshape(B * H, *historical_samples.shape[2:])
        )
    if feature_adapter_real_samples is not None:
        real_parts.append(
            feature_adapter_real_samples.reshape(
                B * adapter_real_pool_count,
                *feature_adapter_real_samples.shape[2:],
            )
        )
    all_real = torch.cat(real_parts, dim=0)

    mae_use_bf16 = _mae_use_bf16(feature_extractor)
    real_stage_features: Dict[str, torch.Tensor] = {}
    need_terminal_stage_features = (
        adapter_update_due or feature_discriminator is not None
        or (adversarial_mode in {"feature_drift", "mixed"}
            and adversarial_system.structure_weight > 0)
    )
    target_stage_adapters = (
        feature_adapter_target.adapters
        if feature_adapter_target is not None
        else None
    )
    with torch.no_grad(), _amp_ctx(mae_use_bf16):
        real_activation_result = feature_extractor.get_activations(
            all_real.to(device),
            **act_kwargs,
            stage_adapters=target_stage_adapters,
            return_stage_features=need_terminal_stage_features,
            **(
                {"return_pre_adapter_maps": True}
                if raw_map_adapter and adapter_update_due
                else {}
            ),
        )
        if need_terminal_stage_features:
            all_real_feats, real_stage_features = real_activation_result
        else:
            all_real_feats = real_activation_result
        # Each feat: (B*(P+N), T, D) — split into pos and neg parts
        pos_feats = {k: v[:B * P] for k, v in all_real_feats.items()}
        neg_feats = {k: v[B * P:] for k, v in all_real_feats.items()} if N > 0 else None
        if N > 0:
            neg_feats = {k: v[:B * N] for k, v in neg_feats.items()}
        history_offset = B * (P + N)
        historical_feats = (
            {k: v[history_offset : history_offset + B * H] for k, v in all_real_feats.items()}
            if H > 0
            else None
        )
        adapter_real_offset = B * (P + N + H)
        adapter_real_target_features = (
            {
                k: v[
                    adapter_real_offset : adapter_real_offset
                    + B * adapter_real_pool_count
                ]
                for k, v in all_real_feats.items()
            }
            if feature_adapter_real_samples is not None
            else None
        )
        # pos_feats/neg_feats views now own the required storage references.
        # Do not keep the original full dictionary alive: stochastic pruning
        # must be able to release a skipped feature once both views are popped.
        del all_real_feats

    if (adversarial_mode in {"feature_drift", "mixed"}
            and adversarial_system.structure_weight > 0):
        # Preserve the exact same real examples used by D below. These are
        # frozen terminal DINO maps from this step, not cached learned features.
        adversarial_real_count = min(P, int(cfg.get("adversarial_samples_per_class", 8)))
        adversarial_teacher_real_features = {}
        for layer in ("layer3", "layer4"):
            if layer not in real_stage_features:
                raise KeyError(f"DINO did not emit adversarial structure teacher {layer}")
            values = real_stage_features[layer][: B * P]
            adversarial_teacher_real_features[layer] = values.reshape(
                B, P, *values.shape[1:]
            )[:, :adversarial_real_count].flatten(0, 1).detach().clone()
        # The selected small teacher tensors own their storage; release all
        # unused negative/history terminal maps before the generator forward.
        del real_stage_features, real_activation_result, values

    adapter_real_stage_features = (
        {
            key: value[
                adapter_real_offset : adapter_real_offset
                + B * adapter_real_pool_count
            ]
            for key, value in real_stage_features.items()
        }
        if feature_adapter_real_samples is not None
        else None
    )
    del all_real, real_parts
    if raw_drift_v2_adapter and feature_adapter_real_samples is not None:
        del feature_adapter_real_samples

    _prof_mark("after_real_mae")

    masked_dino_stage_features: Optional[Dict[str, torch.Tensor]] = None
    masked_dino_patch_mask: Optional[torch.Tensor] = None
    if masked_dino_adapter and adapter_update_due:
        teacher_last_k = int(cfg.get("feature_adapter_mdino_teacher_last_k", 1))
        if teacher_last_k != 1:
            raise ValueError(
                "DINO-R50 masked feature distillation currently supports "
                "teacher_last_k=1 (the clean terminal map for each stage)"
            )
        mdino_samples_per_class = min(
            int(cfg.get("feature_adapter_mdino_samples_per_class", P)), P
        )
        if mdino_samples_per_class <= 0:
            raise ValueError(
                "feature_adapter_mdino_samples_per_class must be positive"
            )
        # Keep the class-major [B,P,...] ordering used by the clean teacher;
        # flattening before slicing would take samples from only the first
        # classes and silently misalign teacher/student pairs.
        selected_real = pos_samples.reshape(
            B, P, *pos_samples.shape[2:]
        )[:, :mdino_samples_per_class].reshape(
            B * mdino_samples_per_class, *pos_samples.shape[2:]
        )
        selected_real = selected_real.to(device, non_blocking=True).detach()
        rank_for_mask = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        mask_seed = (
            int(cfg.get("seed", 42))
            + 1_000_003 * int(step)
            + 97_409 * int(rank_for_mask)
            + 4_449_668_175
        ) % (2**63 - 1)
        selected_real_masked, masked_dino_patch_mask = (
            make_dino_patch_masked_images(
                selected_real,
                patch_size=int(cfg.get("feature_adapter_mdino_patch_size", 32)),
                mask_ratio=float(cfg.get("feature_adapter_mdino_mask_ratio", 0.4)),
                seed=mask_seed,
                fill_values=tuple(
                    cfg.get(
                        "feature_adapter_mdino_fill_values",
                        [-0.03, -0.088, -0.188],
                    )
                ),
            )
        )
        with torch.no_grad(), _amp_ctx(mae_use_bf16):
            _, masked_dino_stage_features = feature_extractor.get_activations(
                selected_real_masked,
                patch_mean_size=[],
                patch_std_size=[],
                use_std=False,
                use_mean=False,
                with_global=False,
                with_norm_x=False,
                every_k_block=float("inf"),
                exclude_terminal_block=False,
                active_stages=list(
                    canonical_adapter_stages(
                        cfg.get("feature_adapter_keys", ["layer3", "layer4"])
                    )
                ),
                stage_adapters=None,
                return_stage_features=True,
            )
        del selected_real, selected_real_masked
        _prof_mark("after_masked_dino_real")

    adapter_info: Dict[str, torch.Tensor] = {}
    adapter_updated = False
    if feature_adapter is not None and not generated_real_adapter:
        if adapter_update_due:
            feature_adapter.train()
            feature_adapter_optimizer.zero_grad(set_to_none=True)
            with _amp_ctx(mae_use_bf16):
                adapter_loss, adapter_info = feature_adapter(
                    real_stage_features,
                    labels,
                    batch_size=B,
                    positive_count=P,
                    samples_per_class=int(
                        cfg.get("feature_adapter_samples_per_class", 8)
                    ),
                    temperature=float(cfg.get("feature_adapter_temp", 0.1)),
                    supcon_weight=float(
                        cfg.get("feature_adapter_loss_weight", 1.0)
                    ),
                    ce_weight=float(cfg.get("feature_adapter_ce_weight", 0.1)),
                    reg_weight=float(cfg.get("feature_adapter_reg_lambda", 0.01)),
                    objective=adapter_objective,
                    gather_distributed=bool(
                        cfg.get("feature_adapter_global_contrastive", False)
                    ),
                    collect_diagnostics=(
                        adapter_objective != "real_real_multipos_infonce"
                        or step % max(1, int(cfg.get("log_every_k", 20))) == 0
                    ),
                )
            adapter_loss.backward()
            adapter_grad_norm = nn.utils.clip_grad_norm_(
                feature_adapter.parameters(),
                float(cfg.get("feature_adapter_max_grad_norm", 1.0)),
            )
            feature_adapter_optimizer.step()
            adapter_info["adapter/grad_norm"] = adapter_grad_norm.detach()
            adapter_updated = True
        if feature_discriminator is None and need_terminal_stage_features:
            del real_stage_features
    _prof_mark("after_adapter_update")

    # 3. Generate samples + compute their features (grad-enabled)
    expanded_labels = repeat(labels, "b -> (b g)", g=G)          # (B*G,)
    expanded_cfg    = repeat(cfg_scales, "b -> (b g)", g=G)      # (B*G,)

    # 4. Negative weighting: (cfg - 1) * (G - 1) / N (matches JAX).
    # Toggle with cfg.use_cfg_weight_neg (default True). When False, weight_neg
    # is fixed to the constant cfg.weight_neg_const (default 1.0) regardless of CFG.
    use_cfg_weight_neg = bool(cfg.get("use_cfg_weight_neg", True))
    weight_neg_const = float(cfg.get("weight_neg_const", 1.0))
    if N > 0:
        if use_cfg_weight_neg:
            uncond_w = (cfg_scales - 1.0) * (G - 1) / max(1, N)    # (B,)
            weight_neg = uncond_w.unsqueeze(1).expand(-1, N)        # (B, N)
        else:
            weight_neg = torch.full((B, N), weight_neg_const, device=device)
    else:
        weight_neg = None

    # 4b. CFG-dependent active pos/neg count per sample (weights all 1).
    # cfg=cfg_min → pos=P_base, neg=N_base; cfg=cfg_max → pos less, neg more
    # (pos_active + neg_active = P_base + N_base, i.e., target budget preserved).
    cfg_sample_ratio = bool(cfg.get("cfg_sample_ratio", False))
    active_mask_pos = None
    active_mask_neg = None
    if cfg_sample_ratio and N > 0:
        cfg_min_val = float(cfg.get("cfg_min", 1.0))
        cfg_max_val = float(cfg.get("cfg_max", 4.0))
        pos_base = int(cfg.get("cfg_ratio_pos_base", 16))
        neg_base = int(cfg.get("cfg_ratio_neg_base", 16))
        pos_min  = int(cfg.get("cfg_ratio_pos_min", 4))
        total_budget = pos_base + neg_base  # = 32 by default
        t = (cfg_scales - cfg_min_val) / max(cfg_max_val - cfg_min_val, 1e-3)
        t = t.clamp(0.0, 1.0)
        # pos linearly shrinks from pos_base → pos_min as cfg → cfg_max.
        # Stochastic rounding on the fractional value so E[pos_k] equals the
        # exact linear schedule (e.g. 14.2 → 14 w.p. 0.8, 15 w.p. 0.2).
        pos_k = _stochastic_round(pos_base - (pos_base - pos_min) * t)    # [B]
        pos_k = torch.clamp(pos_k, min=1, max=P)
        neg_k = torch.clamp(total_budget - pos_k, min=0, max=N).long()   # [B]
        arange_P = torch.arange(P, device=device).unsqueeze(0)           # [1, P]
        arange_N = torch.arange(N, device=device).unsqueeze(0)           # [1, N]
        active_mask_pos = (arange_P < pos_k.unsqueeze(1)).float()        # [B, P]
        active_mask_neg = (arange_N < neg_k.unsqueeze(1)).float()        # [B, N]

    # 4c. cfg_uncond_split: pixel-space partitioning was done in step 1c above.
    # Force weight_neg = ones so the (cfg-1)*(G-1)/N weighting does NOT apply.
    if cfg_uncond_split:
        if N > 0:
            weight_neg = torch.ones(B, N, device=device, dtype=torch.float32)
        else:
            weight_neg = None
        active_mask_pos = None
        active_mask_neg = None

    max_grad_norm = float(cfg.get("max_grad_norm", 2.0))
    _zero_generator_grad(optimizer, throughput_opt_level)
    def _forward_gen_and_feats() -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        # Match the JAX generator path: bf16 model compute when enabled, while
        # attention can still upcast internally via attn_fp32.
        with _amp_ctx(gen_use_bf16):
            out = generator(expanded_labels, cfg_scale=expanded_cfg, train=True)
        gen_samples_local = out["samples"]
        _prof_mark("after_generator")
        with _amp_ctx(mae_use_bf16):
            activation_result = feature_extractor.get_activations(
                gen_samples_local,
                **act_kwargs,
                stage_adapters=target_stage_adapters,
                return_stage_features=(
                    feature_discriminator is not None
                    or (generated_real_adapter and adapter_update_due)
                ),
                **(
                    {"return_pre_adapter_maps": True}
                    if raw_map_adapter and adapter_update_due
                    else {}
                ),
            )
        if (
            feature_discriminator is not None
            or (generated_real_adapter and adapter_update_due)
        ):
            gen_feats_local, gen_stage_features_local = activation_result
        else:
            gen_feats_local = activation_result
            gen_stage_features_local = {}
        _prof_mark("after_generated_mae")
        return gen_samples_local, gen_feats_local, gen_stage_features_local

    gen_samples, gen_feats, gen_stage_features = _forward_gen_and_feats()
    history_ratio = _historical_replay_ratio_for_step(cfg, step, active=H > 0)
    configured_current_weight = cfg.get("historical_gen_current_weight", None)
    if configured_current_weight is not None:
        configured_current_weight = float(configured_current_weight)
        if (
            not math.isfinite(configured_current_weight)
            or configured_current_weight < 0.0
        ):
            raise ValueError(
                "historical_gen_current_weight must be finite and non-negative, got "
                f"{configured_current_weight!r}"
            )
    weight_gen_replay = None
    weight_history = None
    if configured_current_weight is not None:
        weight_gen_replay = torch.full(
            (B, G), configured_current_weight, device=device, dtype=torch.float32
        )
    elif H > 0:
        weight_gen_replay = torch.full(
            (B, G), 1.0 - history_ratio, device=device, dtype=torch.float32
        )
    if H > 0:
        configured_history_weight = cfg.get("historical_gen_history_weight", None)
        per_history_weight = (
            float(configured_history_weight)
            if configured_history_weight is not None
            else history_ratio * G / H
        )
        if not math.isfinite(per_history_weight) or per_history_weight < 0.0:
            raise ValueError(
                "historical_gen_history_weight must be finite and non-negative, got "
                f"{per_history_weight!r}"
            )
        weight_history = torch.full(
            (B, H), per_history_weight, device=device, dtype=torch.float32
        )
    feature_loss_weights = None
    if feature_loss_group_weights:
        feature_loss_weights = _feature_loss_weights_for_groups(
            tuple(gen_feats.keys()),
            feature_loss_group_weights,
            normalize=feature_loss_group_normalize,
        )
    if selected_feature_stages is not None:
        stochastic_weights = _feature_loss_weights_for_stages(
            tuple(gen_feats.keys()),
            selected_feature_stages,
        )
        if feature_loss_weights is None:
            feature_loss_weights = stochastic_weights
        else:
            feature_loss_weights = {
                name: feature_loss_weights[name] * stochastic_weights[name]
                for name in gen_feats
            }
    topk_diagnostic_info: Dict[str, float] = {}
    if step < topk_diagnostic_steps:
        if drift_matching != "rev-drift":
            raise ValueError(
                "topk heterogeneity diagnostics currently require rev-drift"
            )
        topk_diagnostic_info = diagnose_reverse_topk_heterogeneity(
            gen_feats=gen_feats,
            pos_feats=pos_feats,
            neg_feats=neg_feats,
            batch_size=B,
            gen_count=G,
            pos_count=P,
            neg_count=N,
            weight_neg=weight_neg,
            R_list=R_list,
            top_k_pos=topk_diagnostic_pos,
            top_k_neg=topk_diagnostic_neg,
            active_mask_pos=active_mask_pos,
            active_mask_neg=active_mask_neg,
            global_scale_stats=global_scale_stats,
            feature_temperature_multipliers=feature_temperature_multipliers,
        )
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            pos_l14 = topk_diagnostic_info.get(
                "topk_diag/pos/layer1_to_stage4_overlap", float("nan")
            )
            neg_l14 = topk_diagnostic_info.get(
                "topk_diag/neg/layer1_to_stage4_overlap", float("nan")
            )
            pos_union = topk_diagnostic_info.get(
                "topk_diag/pos/token_query_union_stage1", float("nan")
            )
            neg_real_union = topk_diagnostic_info.get(
                "topk_diag/neg/token_real_neg_union_stage1", float("nan")
            )
            print(
                "[topk-diag] "
                f"step={step} layer1->stage4 overlap "
                f"pos={pos_l14:.4f} neg={neg_l14:.4f}; "
                f"actual stage1 candidate union "
                f"pos={pos_union:.4f} real_neg={neg_real_union:.4f}",
                flush=True,
            )
    loss, loss_info = compute_drift_loss_from_features(
        gen_feats=gen_feats,
        pos_feats=pos_feats,
        neg_feats=neg_feats,
        B=B, G=G, P=P, N=N,
        weight_neg=weight_neg,
        R_list=R_list,
        drift_matching=drift_matching,
        compute_wpos_stats=compute_wpos_stats,
        per_sample_fnorm=per_sample_fnorm,
        active_mask_pos=active_mask_pos,
        active_mask_neg=active_mask_neg,
        decouple_weight_from_coupling=decouple_weight_from_coupling,
        self_mask_on_raw=self_mask_on_raw,
        global_scale_stats=global_scale_stats,
        global_fnorm_stats=global_fnorm_stats,
        mix_alpha=mix_alpha,
        mix_R_list_baseline=mix_R_list_baseline,
        compute_raw_winner_stats_flag=collect_mix_alpha_stats,
        collect_diagnostics=collect_diagnostics,
        fuse_fnorm_across_R=fuse_fnorm_across_R,
        feature_loss_weights=feature_loss_weights,
        prune_zero_weight_features=prune_skipped_feature_tensors,
        dual_drift_share_distances=dual_drift_share_distances,
        rev_drift_top_p=rev_drift_top_p,
        fwd_drift_top_p=fwd_drift_top_p,
        drift_top_p_min_keep=drift_top_p_min_keep,
        drift_top_k_pos=drift_top_k_pos,
        drift_top_k_neg=drift_top_k_neg,
        drift_top_k_groups=drift_top_k_groups,
        feature_temperature_multipliers=feature_temperature_multipliers,
        rev_drift_force_multiplier=rev_drift_force_multiplier,
        rev_drift_affinity_kernel=rev_drift_affinity_kernel,
        rev_drift_kernel_shape=rev_drift_kernel_shape,
        rev_drift_kernel_adaptive_k_pos=rev_drift_kernel_adaptive_k_pos,
        rev_drift_kernel_adaptive_k_neg=rev_drift_kernel_adaptive_k_neg,
        rev_drift_kernel_adaptive_margin=rev_drift_kernel_adaptive_margin,
        rev_drift_kernel_mix_weight=rev_drift_kernel_mix_weight,
        rev_drift_kernel_temperature_mix=rev_drift_kernel_temperature_mix,
        rev_drift_kernel_temperature_mix_weights=rev_drift_kernel_temperature_mix_weights,
        historical_feats=historical_feats,
        historical_count=H,
        weight_gen=weight_gen_replay,
        weight_history=weight_history,
    )
    loss_info.update(topk_diagnostic_info)
    _prof_mark("after_drift_loss")

    feature_gan_info: Dict[str, torch.Tensor | float] = {}
    adversarial_info: Dict[str, torch.Tensor | float] = {}
    total_generator_loss = loss
    if adversarial_system is not None:
        adversarial_extra_loss = loss.new_zeros(())
        adversarial_components = {}
        if adversarial_mode == "mixed":
            adversarial_logits, adversarial_gen_feats = adversarial_system.target_logits_and_features(
                gen_samples, expanded_labels
            )
        if adversarial_mode in {"raw_gan", "mixed"}:
            if adversarial_mode == "raw_gan":
                adversarial_logits = adversarial_system.target_logits(gen_samples, expanded_labels)
            adversarial_gan_loss = F.softplus(-adversarial_logits).mean()
            adversarial_weight = float(cfg.get("adversarial_loss_weight", 0.1))
            adversarial_components["raw_gan"] = adversarial_weight * adversarial_gan_loss
            adversarial_extra_loss = adversarial_extra_loss + adversarial_components["raw_gan"]
            adversarial_info["adversarial/raw_gan_loss"] = adversarial_gan_loss.detach()
            adversarial_info["adversarial/raw_gan_weight"] = adversarial_weight
        if adversarial_mode in {"feature_drift", "mixed"}:
            # Re-encode every particle type with the one frozen pre-update
            # snapshot. The real memory banks continue to hold raw pixels.
            adversarial_apply_replay = bool(cfg.get("adversarial_apply_replay", True))
            with torch.no_grad():
                adversarial_pos_feats = adversarial_system.target_features(
                    pos_samples.flatten(0, 1)
                )
                adversarial_neg_feats = (
                    adversarial_system.target_features(neg_samples.flatten(0, 1))
                    if N > 0 else None
                )
                adversarial_history_feats = (
                    adversarial_system.target_features(historical_samples.flatten(0, 1))
                    if adversarial_apply_replay and historical_samples is not None else None
                )
            if adversarial_mode == "feature_drift":
                adversarial_gen_feats = adversarial_system.target_features(
                    gen_samples, expanded_labels
                )
            adversarial_feature_loss, adversarial_drift_info = compute_drift_loss_from_features(
                gen_feats=adversarial_gen_feats,
                pos_feats=adversarial_pos_feats,
                neg_feats=adversarial_neg_feats,
                B=B, G=G, P=P, N=N,
                weight_neg=weight_neg,
                R_list=R_list,
                drift_matching=drift_matching,
                compute_wpos_stats=compute_wpos_stats,
                per_sample_fnorm=per_sample_fnorm,
                active_mask_pos=active_mask_pos,
                active_mask_neg=active_mask_neg,
                decouple_weight_from_coupling=decouple_weight_from_coupling,
                self_mask_on_raw=self_mask_on_raw,
                global_scale_stats=global_scale_stats,
                global_fnorm_stats=global_fnorm_stats,
                mix_alpha=mix_alpha,
                mix_R_list_baseline=mix_R_list_baseline,
                collect_diagnostics=collect_diagnostics,
                fuse_fnorm_across_R=fuse_fnorm_across_R,
                dual_drift_share_distances=dual_drift_share_distances,
                rev_drift_top_p=rev_drift_top_p,
                fwd_drift_top_p=fwd_drift_top_p,
                drift_top_p_min_keep=drift_top_p_min_keep,
                drift_top_k_pos=drift_top_k_pos,
                drift_top_k_neg=drift_top_k_neg,
                rev_drift_force_multiplier=rev_drift_force_multiplier,
                rev_drift_affinity_kernel=rev_drift_affinity_kernel,
                rev_drift_kernel_shape=rev_drift_kernel_shape,
                rev_drift_kernel_adaptive_k_pos=rev_drift_kernel_adaptive_k_pos,
                rev_drift_kernel_adaptive_k_neg=rev_drift_kernel_adaptive_k_neg,
                rev_drift_kernel_adaptive_margin=rev_drift_kernel_adaptive_margin,
                rev_drift_kernel_mix_weight=rev_drift_kernel_mix_weight,
                rev_drift_kernel_temperature_mix=rev_drift_kernel_temperature_mix,
                rev_drift_kernel_temperature_mix_weights=rev_drift_kernel_temperature_mix_weights,
                historical_feats=adversarial_history_feats,
                historical_count=H if adversarial_apply_replay else 0,
                weight_gen=weight_gen_replay if adversarial_apply_replay else None,
                weight_history=weight_history if adversarial_apply_replay else None,
            )
            adversarial_info["adversarial/replay_enabled"] = float(adversarial_apply_replay)
            adversarial_info["adversarial/history_count"] = float(H if adversarial_apply_replay else 0)
            adversarial_weight = float(cfg.get("adversarial_drift_weight", 1.0))
            adversarial_components["feature_drift"] = adversarial_weight * adversarial_feature_loss
            adversarial_extra_loss = adversarial_extra_loss + adversarial_components["feature_drift"]
            adversarial_info["adversarial/feature_drift_loss"] = adversarial_feature_loss.detach()
            adversarial_info["adversarial/feature_drift_weight"] = adversarial_weight
            adversarial_info.update({
                f"adversarial/drift/{key}": value
                for key, value in adversarial_drift_info.items()
            })
        if not adversarial_components:
            raise ValueError(f"Unknown adversarial mode {adversarial_mode!r}")
        total_generator_loss = loss + adversarial_extra_loss
        _require_finite_adversarial(total_generator_loss, "generator loss")
        adversarial_info["adversarial/g_extra_loss"] = adversarial_extra_loss.detach()
        if adversarial_mode != "mixed":
            adversarial_info["adversarial/g_weight"] = adversarial_weight
        adversarial_info["adversarial/g_loss_finite"] = 1.0
        diagnostic_every = int(cfg.get("adversarial_diagnostics_every", 1000))
        if step == 0 or (diagnostic_every > 0 and step % diagnostic_every == 0):
            if adversarial_mode == "mixed":
                # Check each weighted branch so cancellation or an omitted
                # objective cannot hide behind a finite combined gradient.
                auxiliary_pixel_grad = None
                for component, component_loss in adversarial_components.items():
                    component_pixel_grad = torch.autograd.grad(
                        component_loss, gen_samples, retain_graph=True
                    )[0].detach()
                    component_pixel_norm = component_pixel_grad.float().norm()
                    _require_finite_adversarial(
                        component_pixel_norm, f"{component} pixel gradient", require_nonzero=True
                    )
                    adversarial_info[f"adversarial/{component}_pixel_grad_norm"] = component_pixel_norm
                    auxiliary_pixel_grad = (
                        component_pixel_grad if auxiliary_pixel_grad is None
                        else auxiliary_pixel_grad + component_pixel_grad
                    )
                del component_pixel_grad
            else:
                auxiliary_pixel_grad = torch.autograd.grad(
                    adversarial_extra_loss, gen_samples, retain_graph=True
                )[0]
            dino_pixel_grad = torch.autograd.grad(loss, gen_samples, retain_graph=True)[0]
            auxiliary_pixel_norm = auxiliary_pixel_grad.detach().float().norm()
            dino_pixel_norm = dino_pixel_grad.detach().float().norm()
            _require_finite_adversarial(
                auxiliary_pixel_norm, "auxiliary pixel gradient", require_nonzero=True
            )
            _require_finite_adversarial(dino_pixel_norm, "DINO pixel gradient")
            adversarial_info["adversarial/g_pixel_grad_norm"] = auxiliary_pixel_norm
            adversarial_info["adversarial/dino_pixel_grad_norm"] = dino_pixel_norm
            adversarial_info["adversarial/g_to_dino_pixel_grad_ratio"] = (
                auxiliary_pixel_norm / dino_pixel_norm.clamp_min(1.0e-12)
            )
            del auxiliary_pixel_grad, dino_pixel_grad
    if feature_discriminator is not None:
        raw_discriminator: FrozenFeatureDiscriminator = (
            feature_discriminator.module
            if hasattr(feature_discriminator, "module")
            else feature_discriminator
        )
        discriminator_stages = raw_discriminator.stages
        real_per_class = min(
            P,
            int(cfg.get("feature_gan_real_samples_per_class", G)),
        )
        if real_per_class <= 0:
            raise ValueError("feature_gan_real_samples_per_class must be positive")

        real_discriminator_features: Dict[str, torch.Tensor] = {}
        fake_discriminator_features: Dict[str, torch.Tensor] = {}
        for stage in discriminator_stages:
            layer = f"layer{stage[-1]}"
            if layer not in real_stage_features or layer not in gen_stage_features:
                raise KeyError(
                    f"MAE did not emit required feature-GAN stage {layer}"
                )
            real_feature = real_stage_features[layer][: B * P]
            real_feature = real_feature.reshape(
                B, P, *real_feature.shape[1:]
            )[:, :real_per_class]
            real_discriminator_features[stage] = real_feature.reshape(
                B * real_per_class, *real_feature.shape[2:]
            ).detach()
            fake_discriminator_features[stage] = gen_stage_features[layer]

        real_discriminator_labels = labels[:, None].expand(
            B, real_per_class
        ).reshape(-1)
        fake_discriminator_labels = expanded_labels
        discriminator_labels = torch.cat(
            [real_discriminator_labels, fake_discriminator_labels], dim=0
        )
        discriminator_features = {
            stage: torch.cat(
                [
                    real_discriminator_features[stage],
                    fake_discriminator_features[stage].detach(),
                ],
                dim=0,
            )
            for stage in discriminator_stages
        }

        feature_discriminator.train()
        feature_discriminator_optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(bool(cfg.get("feature_gan_use_bf16", True))):
            discriminator_scores = feature_discriminator(
                discriminator_features, discriminator_labels
            )
            real_count = real_discriminator_labels.shape[0]
            real_scores = discriminator_scores[:real_count]
            fake_scores_detached = discriminator_scores[real_count:]
            discriminator_loss = discriminator_hinge_loss(
                real_scores, fake_scores_detached
            )
        discriminator_loss.backward()
        discriminator_grad_norm = nn.utils.clip_grad_norm_(
            feature_discriminator.parameters(),
            float(cfg.get("feature_gan_discriminator_max_grad_norm", 5.0)),
        )
        feature_discriminator_optimizer.step()
        _prof_mark("after_feature_discriminator")

        # The generator sees the just-updated discriminator, but discriminator
        # weights are constants for this forward.  Gradients continue through
        # its operations into the generated MAE maps and then the generator.
        for parameter in raw_discriminator.parameters():
            parameter.requires_grad_(False)
        with _amp_ctx(bool(cfg.get("feature_gan_use_bf16", True))):
            generator_fake_scores = raw_discriminator(
                fake_discriminator_features, fake_discriminator_labels
            )
            adversarial_loss = generator_hinge_loss(generator_fake_scores)
        for parameter in raw_discriminator.parameters():
            parameter.requires_grad_(True)

        calibration_frequency = int(
            cfg.get("feature_gan_gradient_calibration_freq", 10)
        )
        if calibration_frequency <= 0:
            raise ValueError(
                "feature_gan_gradient_calibration_freq must be positive"
            )
        calibrate = (
            step % calibration_frequency == 0
            or int(raw_discriminator.gradient_ratio_updates.item()) == 0
        )
        if calibrate:
            calibration_features = tuple(
                fake_discriminator_features[stage]
                for stage in discriminator_stages
            )
            drift_gradient_norm = _distributed_feature_gradient_norm(
                loss, calibration_features, device
            )
            adversarial_gradient_norm = _distributed_feature_gradient_norm(
                adversarial_loss, calibration_features, device
            )
            raw_discriminator.update_gradient_calibration(
                drift_gradient_norm,
                adversarial_gradient_norm,
                ema_decay=float(cfg.get("feature_gan_gradient_ratio_ema", 0.0)),
            )
        else:
            drift_gradient_norm = raw_discriminator.drift_grad_ema.detach()
            adversarial_gradient_norm = (
                raw_discriminator.adversarial_grad_ema.detach()
            )

        target_ratio = float(cfg.get("feature_gan_gradient_ratio", 0.1))
        if target_ratio < 0.0 or not math.isfinite(target_ratio):
            raise ValueError(
                f"feature_gan_gradient_ratio must be finite and >= 0, got {target_ratio}"
            )
        gan_warmup_steps = int(cfg.get("feature_gan_warmup_steps", 1000))
        if gan_warmup_steps > 0:
            warmup_fraction = min(1.0, float(step + 1) / gan_warmup_steps)
        else:
            warmup_fraction = 1.0
        effective_target_ratio = target_ratio * warmup_fraction
        adversarial_scale = (
            raw_discriminator.gradient_unit_scale * effective_target_ratio
        ).clamp(
            min=float(cfg.get("feature_gan_scale_min", 0.0)),
            max=float(cfg.get("feature_gan_scale_max", 1000.0)),
        )
        total_generator_loss = loss + adversarial_scale.detach() * adversarial_loss
        estimated_gradient_ratio = (
            adversarial_scale.detach()
            * raw_discriminator.adversarial_grad_ema
            / raw_discriminator.drift_grad_ema.clamp_min(1.0e-12)
        )
        feature_gan_info = {
            "feature_gan/d_loss": discriminator_loss.detach(),
            "feature_gan/d_real_score": real_scores.detach().mean(),
            "feature_gan/d_fake_score": fake_scores_detached.detach().mean(),
            "feature_gan/d_grad_norm": discriminator_grad_norm.detach(),
            "feature_gan/g_loss": adversarial_loss.detach(),
            "feature_gan/g_scale": adversarial_scale.detach(),
            "feature_gan/target_gradient_ratio": target_ratio,
            "feature_gan/effective_target_gradient_ratio": effective_target_ratio,
            "feature_gan/estimated_gradient_ratio": estimated_gradient_ratio.detach(),
            "feature_gan/drift_feature_grad_norm": drift_gradient_norm.detach(),
            "feature_gan/adversarial_feature_grad_norm": (
                adversarial_gradient_norm.detach()
            ),
            "feature_gan/calibrated": float(calibrate),
        }
        if not (generated_real_adapter and adapter_update_due):
            del real_stage_features, gen_stage_features

    total_generator_loss.backward()
    _prof_mark("after_backward")

    g_norm = nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
    if adversarial_system is not None:
        _require_finite_adversarial(g_norm, "generator gradient norm")
        adversarial_info["adversarial/g_grad_finite"] = 1.0
    optimizer.step()

    if adversarial_system is not None:
        # The G graph has been released. Update D using detached particles
        # from that same G forward, then advance the frozen snapshot for the
        # next iteration. This cannot backpropagate into G or frozen DINO.
        discriminator_real_count = min(P, int(cfg.get("adversarial_samples_per_class", 8)))
        discriminator_fake_count = min(G, int(cfg.get("adversarial_samples_per_class", 8)))
        discriminator_reals = pos_samples[:, :discriminator_real_count].flatten(0, 1).detach()
        discriminator_fakes = gen_samples.detach().reshape(
            B, G, *gen_samples.shape[1:]
        )[:, :discriminator_fake_count].flatten(0, 1)
        adversarial_info.update(adversarial_system.discriminator_step(
            discriminator_reals,
            discriminator_fakes,
            labels[:, None].expand(B, discriminator_real_count).reshape(-1),
            labels[:, None].expand(B, discriminator_fake_count).reshape(-1),
            step,
            teacher_real_features=adversarial_teacher_real_features,
        ))
        del adversarial_teacher_real_features, discriminator_reals, discriminator_fakes
        _prof_mark("after_adversarial_update")

    # Strict alternating adapter update.  The generator step above used only
    # the frozen target copy. Reuse the already-computed selected pre-adapter
    # encoder maps here, but detach both branches so this backward can update
    # only the online adapter (never G or the frozen DINO/MoCo encoder).
    if generated_real_adapter and adapter_update_due:
        adapter_collect_diagnostics = (
            step % max(1, int(cfg.get("log_every_k", 20))) == 0
        )
        feature_adapter.train()
        feature_adapter_optimizer.zero_grad(set_to_none=True)
        drift_options = None
        if raw_map_adapter:
            drift_options = {
                "R_list": R_list,
                "patch_mean_size": act_kwargs.get("patch_mean_size", [2, 4]),
                "patch_std_size": act_kwargs.get("patch_std_size", [2, 4]),
                "use_mean": bool(act_kwargs.get("use_mean", True)),
                "use_std": bool(act_kwargs.get("use_std", True)),
                "feature_loss_weights": (
                    {} if feature_loss_weights is None else feature_loss_weights
                ),
                "temperature_multipliers": feature_temperature_multipliers,
                "global_scale_stats": global_scale_stats,
                "global_statistics": bool(
                    cfg.get("feature_adapter_global_statistics", True)
                ),
                "top_p": rev_drift_top_p,
                "top_p_min_keep": drift_top_p_min_keep,
                "top_k_pos": drift_top_k_pos,
                "top_k_neg": drift_top_k_neg,
                "affinity_kernel": rev_drift_affinity_kernel,
                "kernel_shape": rev_drift_kernel_shape,
                "kernel_adaptive_k_pos": rev_drift_kernel_adaptive_k_pos,
                "kernel_adaptive_k_neg": rev_drift_kernel_adaptive_k_neg,
                "kernel_adaptive_margin": rev_drift_kernel_adaptive_margin,
                "kernel_mix_weight": rev_drift_kernel_mix_weight,
                "kernel_temperature_mix": rev_drift_kernel_temperature_mix,
                "kernel_temperature_mix_weights": (
                    rev_drift_kernel_temperature_mix_weights
                ),
                "expected_feature_count": int(
                    cfg.get("feature_adapter_expected_feature_count", 42)
                ),
                "expected_raw_map_count": int(
                    cfg.get("feature_adapter_expected_raw_map_count", 6)
                ),
            }
            if raw_drift_adapter:
                drift_options.update(
                    {
                        "snr_epsilon": float(
                            cfg.get("feature_adapter_snr_epsilon", 1.0e-8)
                        ),
                        "stopgrad_variance": bool(
                            cfg.get(
                                "feature_adapter_snr_stopgrad_variance", False
                            )
                        ),
                        "snr_weight": float(
                            cfg.get("feature_adapter_drift_snr_weight", 1.0)
                        ),
                        "consistency_weight": float(
                            cfg.get("feature_adapter_consistency_weight", 0.1)
                        ),
                    }
                )
                if raw_drift_v2_adapter:
                    drift_options.update(
                        {
                            "real_pool_count": int(
                                cfg.get(
                                    "feature_adapter_real_pool_per_class", 32
                                )
                            ),
                            "query_count": int(
                                cfg.get("feature_adapter_query_count", 8)
                            ),
                            "bank_count": int(
                                cfg.get("feature_adapter_bank_count", 12)
                            ),
                            "consistency_epsilon": float(
                                cfg.get(
                                    "feature_adapter_consistency_epsilon",
                                    1.0e-8,
                                )
                            ),
                            "split_seed": int(
                                cfg.get(
                                    "feature_adapter_split_seed",
                                    cfg.get("seed", 43),
                                )
                            ),
                        }
                    )
            else:
                drift_options.update(
                    {
                        "cf_epsilon": float(
                            cfg.get("feature_adapter_cf_epsilon", 1.0e-8)
                        ),
                        "cf_rho": float(
                            cfg.get("feature_adapter_cf_rho", 1.0)
                        ),
                        "cf_weight": float(
                            cfg.get("feature_adapter_cf_weight", 1.0)
                        ),
                        "null_weight": float(
                            cfg.get("feature_adapter_null_weight", 1.0)
                        ),
                        "relation_weight": float(
                            cfg.get("feature_adapter_relation_weight", 1.0)
                        ),
                    }
                )
        adapter_update_index = (
            (step - adapter_start_step) // adapter_update_frequency
        )
        adapter_stage_input = (
            adapter_real_stage_features
            if raw_drift_v2_adapter
            else real_stage_features
        )
        adapter_positive_count = (
            adapter_real_pool_count if raw_drift_v2_adapter else P
        )
        adapter_negative_count = 0 if raw_drift_v2_adapter else N
        adapter_target_positive = (
            adapter_real_target_features
            if raw_drift_v2_adapter
            else pos_feats
        )
        with _amp_ctx(mae_use_bf16):
            adapter_loss, adapter_info = feature_adapter(
                adapter_stage_input,
                labels,
                batch_size=B,
                positive_count=adapter_positive_count,
                samples_per_class=(
                    adapter_real_pool_count
                    if raw_drift_v2_adapter
                    else int(cfg.get("feature_adapter_samples_per_class", 8))
                ),
                temperature=float(cfg.get("feature_adapter_temp", 0.1)),
                supcon_weight=float(
                    cfg.get("feature_adapter_loss_weight", 1.0)
                ),
                ce_weight=0.0,
                reg_weight=float(cfg.get("feature_adapter_reg_lambda", 0.01)),
                drift_align_weight=float(
                    cfg.get("feature_adapter_drift_align_lambda", 0.0)
                ),
                distance_scale_weight=float(
                    cfg.get("feature_adapter_distance_scale_lambda", 0.0)
                ),
                distance_mean_weight=float(
                    cfg.get("feature_adapter_distance_mean_lambda", 0.0)
                ),
                objective=adapter_objective,
                generated_stage_features=gen_stage_features,
                generated_count=G,
                generated_samples_per_class=int(
                    cfg.get("feature_adapter_generated_samples_per_class", G)
                ),
                gather_distributed=bool(
                    cfg.get(
                        "feature_adapter_global_statistics"
                        if raw_map_adapter
                        else "feature_adapter_global_contrastive",
                        raw_map_adapter,
                    )
                ),
                collect_diagnostics=adapter_collect_diagnostics,
                negative_count=adapter_negative_count,
                weight_neg=None if raw_drift_v2_adapter else weight_neg,
                target_positive_features=(
                    adapter_target_positive if raw_drift_adapter else None
                ),
                target_negative_features=(
                    neg_feats
                    if adapter_objective == "raw_drift_snr_consistency"
                    else None
                ),
                target_generated_features=gen_feats if raw_drift_adapter else None,
                drift_options=drift_options,
                masked_stage_features=masked_dino_stage_features,
                masked_patch_mask=masked_dino_patch_mask,
                mdino_options={
                    "weight": float(cfg.get("feature_adapter_mdino_weight", 1.0)),
                    "mask_ratio": float(
                        cfg.get("feature_adapter_mdino_mask_ratio", 0.4)
                    ),
                    "smooth_l1_weight": float(
                        cfg.get("feature_adapter_mdino_smooth_l1_weight", 1.0)
                    ),
                    "cosine_weight": float(
                        cfg.get("feature_adapter_mdino_cosine_weight", 0.0)
                    ),
                    "smooth_l1_beta": float(
                        cfg.get("feature_adapter_mdino_smooth_l1_beta", 1.0)
                    ),
                    "samples_per_class": int(
                        cfg.get("feature_adapter_mdino_samples_per_class", P)
                    ),
                },
                view_swap=bool(adapter_update_index % 2),
                split_index=adapter_update_index,
            )
        adapter_loss.backward()
        if adapter_collect_diagnostics and masked_dino_adapter:
            raw_online_adapter = (
                feature_adapter.module
                if isinstance(feature_adapter, DDP)
                else feature_adapter
            )

            def component_grad_norm(module: nn.Module) -> torch.Tensor:
                squared = [
                    parameter.grad.detach().float().square().sum()
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                if not squared:
                    return adapter_loss.detach().new_zeros(())
                return torch.stack(squared).sum().sqrt()

            adapter_info["adapter/spatial_grad_norm_unclipped"] = (
                component_grad_norm(raw_online_adapter.adapters).detach()
            )
            adapter_info["adapter/predictor_grad_norm_unclipped"] = (
                component_grad_norm(raw_online_adapter.masked_predictors).detach()
            )
        adapter_grad_norm = nn.utils.clip_grad_norm_(
            feature_adapter.parameters(),
            float(cfg.get("feature_adapter_max_grad_norm", 1.0)),
        )
        feature_adapter_optimizer.step()
        if adapter_collect_diagnostics:
            adapter_info["adapter/grad_norm"] = adapter_grad_norm.detach()
        adapter_updated = True
        _prof_mark("after_generated_real_adapter_update")
        del real_stage_features, gen_stage_features
        if masked_dino_stage_features is not None:
            del masked_dino_stage_features, masked_dino_patch_mask

    if adapter_updated:
        update_adapter_ema(
            feature_adapter_target,
            feature_adapter,
            float(cfg.get("feature_adapter_ema_decay", 0.999)),
        )
    _prof_mark("after_optimizer")

    metrics = {"drift_matching": drift_matching}
    if collect_training_metrics:
        # These reductions and scalar extractions exist only for logging. Skip
        # them on level-4 steps whose metrics will not be consumed by Logger.
        metrics.update(
            {
                "loss": _ddp_mean_scalar(total_generator_loss, device),
                "drift_loss": _ddp_mean_scalar(loss, device),
                "g_norm": (
                    g_norm.item()
                    if isinstance(g_norm, torch.Tensor)
                    else float(g_norm)
                ),
                "cfg_mean": cfg_scales.mean().item(),
            }
        )
        metrics.update(_ddp_mean_scalars(adapter_info, device))
        metrics.update(_ddp_mean_scalars(adversarial_info, device))
        if feature_adapter is not None:
            metrics["adapter/updated"] = float(adapter_updated)
            metrics["adapter/start_step"] = float(adapter_start_step)
            metrics["adapter/freeze_step"] = float(adapter_freeze_step)
            metrics["adapter/frozen"] = float(
                adapter_freeze_step > 0 and step >= adapter_freeze_step
            )
            metrics["adapter/bank_ready"] = float(feature_adapter_bank_ready)
        for name, value in feature_gan_info.items():
            metrics[name] = _ddp_mean_scalar(value, device)
        for stage_name in _STOCHASTIC_FEATURE_STAGES:
            metrics[f"drift_temperature/{stage_name}_multiplier"] = float(
                feature_temperature_multipliers.get(
                    stage_name,
                    feature_temperature_multipliers.get("default", 1.0),
                )
            )
        if feature_loss_weights is not None:
            for group_name in _FEATURE_LOSS_GROUPS:
                group_values = [
                    weight
                    for name, weight in feature_loss_weights.items()
                    if _feature_loss_group(name) == group_name
                ]
                if group_values:
                    metrics[f"feature_loss/{group_name}_weight"] = float(
                        sum(group_values) / len(group_values)
                    )
        if selected_feature_stages is not None:
            selected = set(selected_feature_stages)
            metrics["stochastic_stage/selected_count"] = float(len(selected))
            metrics["stochastic_stage/inverse_probability"] = float(
                len(_STOCHASTIC_FEATURE_STAGES) / len(selected)
            )
            for stage_name in _STOCHASTIC_FEATURE_STAGES:
                metrics[f"stochastic_stage/selected_{stage_name}"] = float(
                    stage_name in selected
                )
    # Log raw coverage and the capacity-normalized ratios used by Hedge,
    # regardless of drift_matching. Tracker is created unconditionally.
    if mix_alpha_tracker is not None:
        if collect_mix_alpha_stats:
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
            mix_alpha_tracker.update(
                loss_info,
                world_size=world_size,
                device=device,
            )
        if collect_training_metrics and collect_mix_alpha_stats:
            metrics["mix_alpha_tracker/alpha1"] = float(
                mix_alpha_tracker.alpha1
            )
            metrics["mix_alpha_tracker/beta1"] = float(
                mix_alpha_tracker.beta1
            )
            metrics["mix_alpha_tracker/alpha1_capacity"] = float(
                mix_alpha_tracker.alpha1_capacity
            )
            metrics["mix_alpha_tracker/beta1_capacity"] = float(
                mix_alpha_tracker.beta1_capacity
            )
            metrics["mix_alpha_tracker/versionb_coef"] = float(1.0 - mix_alpha)
        # Strip lower-level mix keys; their useful values are represented by
        # the tracker metrics above.
        for _k in ("raw/pos_winner_count", "raw/pos_winner_total",
                   "raw/gen_winner_count", "raw/gen_winner_total",
                   "mix_alpha", "mix_scale_v", "mix_scale_b"):
            loss_info.pop(_k, None)
    if collect_training_metrics and profile_train_step and len(prof_marks) >= 2:
        for i in range(1, len(prof_marks)):
            a, ta = prof_marks[i - 1]
            b, tb = prof_marks[i]
            metrics[f"prof_ms/{a}__{b}"] = (tb - ta) * 1000.0
    if collect_training_metrics:
        metrics.update({k: float(v) for k, v in loss_info.items()})
    extras = {
        "gen_samples_detached": gen_samples.detach(),
        "expanded_labels": expanded_labels.detach(),
    }
    return total_generator_loss, metrics, extras


# ---------------------------------------------------------------------------
# FID evaluation (lightweight: saves generated images + uses torchmetrics)
# ---------------------------------------------------------------------------

def _build_eval_metric(metric_cls, **kwargs):
    # In DDP, rank-0-only eval must not trigger torchmetrics all_gather sync.
    try:
        return metric_cls(sync_on_compute=False, **kwargs)
    except TypeError:
        return metric_cls(**kwargs)


def _to_chw_float01(images: np.ndarray) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims, got shape={images.shape}")
    t = torch.from_numpy(images)
    if images.shape[-1] in (1, 3):
        t = t.permute(0, 3, 1, 2).contiguous()
    elif images.shape[1] not in (1, 3):
        raise ValueError(f"Unsupported image batch shape={images.shape}")
    t = t.float()
    if np.issubdtype(images.dtype, np.integer) or t.max().item() > 1.0 or t.min().item() < 0.0:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def _load_labeled_image_npz(npz_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with np.load(npz_path) as data:
        keys = list(data.keys())
        image_key = next((k for k in ("arr_0", "images", "x", "samples") if k in keys), None)
        label_key = next((k for k in ("arr_1", "labels", "y") if k in keys), None)
        if image_key is None:
            image_key = next((k for k in keys if data[k].ndim == 4), None)
        if image_key is None:
            raise KeyError(f"Could not find image array in {npz_path}; keys={keys}")
        images = np.array(data[image_key])
        if label_key is None:
            label_key = next((k for k in keys if data[k].ndim == 1 and len(data[k]) == len(images)), None)
        if label_key is None:
            raise KeyError(f"Could not find label array in {npz_path}; keys={keys}")
        labels = np.array(data[label_key]).astype(np.int64, copy=False)
    return _to_chw_float01(images), torch.from_numpy(labels)


def _save_labeled_image_npz(npz_path: str, images_01: torch.Tensor, labels: torch.Tensor) -> None:
    path = Path(npz_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images_u8 = (
        images_01.clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    np.savez(path, images_u8, labels_np)


@torch.no_grad()
def _generate_from_labels(
    generator: nn.Module,
    postprocess_fn,
    labels: torch.Tensor,
    device: torch.device,
    cfg_scale: float,
    batch_size: int,
) -> torch.Tensor:
    raw_gen = generator.module if hasattr(generator, "module") else generator
    raw_gen.eval()
    use_bf16 = _gen_use_bf16(generator)
    gen_images: List[torch.Tensor] = []
    total = int(labels.shape[0])
    for start in range(0, total, max(1, batch_size)):
        labels_b = labels[start:start + max(1, batch_size)].to(device)
        with _amp_ctx(use_bf16):
            out = raw_gen(labels_b, cfg_scale=cfg_scale, train=False)
        gen_pixel = postprocess_fn(out["samples"])
        if gen_pixel.shape[1] != 3:
            raise ValueError(f"Expected generated RGB images, got shape={tuple(gen_pixel.shape)}")
        gen_images.append(gen_pixel.cpu().float())
    if not gen_images:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32)
    return torch.cat(gen_images, dim=0)


@torch.no_grad()
def _collect_eval_loader_pairs(
    generator: nn.Module,
    postprocess_fn,
    eval_loader,
    device: torch.device,
    cfg_scale: float,
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_gen = generator.module if hasattr(generator, "module") else generator
    raw_gen.eval()
    use_bf16 = _gen_use_bf16(generator)
    gen_images: List[torch.Tensor] = []
    real_images: List[torch.Tensor] = []
    n_collected = 0
    for batch in eval_loader:
        raw, labels = batch[0], batch[1]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(raw, np.ndarray):
            raw = torch.from_numpy(raw)
        bsz = min(labels.shape[0], n_samples - n_collected)
        labels = labels[:bsz].to(device)
        raw = raw[:bsz].to(device)

        if raw.shape[1] in (1, 3):
            real_01 = ((raw.float() + 1) / 2).clamp(0, 1)
        else:
            real_01 = postprocess_fn(raw)
        if real_01.shape[1] == 3:
            real_images.append(real_01.cpu().float())

        with _amp_ctx(use_bf16):
            out = raw_gen(labels, cfg_scale=cfg_scale, train=False)
        gen_pixel = postprocess_fn(out["samples"])
        if gen_pixel.shape[1] == 3:
            gen_images.append(gen_pixel.cpu().float())

        n_collected += bsz
        if n_collected >= n_samples:
            break

    if not gen_images or not real_images:
        return (
            torch.empty((0, 3, 0, 0), dtype=torch.float32),
            torch.empty((0, 3, 0, 0), dtype=torch.float32),
        )
    return (
        torch.cat(gen_images, dim=0)[:n_samples],
        torch.cat(real_images, dim=0)[:n_samples],
    )


@torch.no_grad()
def _compute_fid_is_from_tensors(
    all_gen: torch.Tensor,
    all_real: torch.Tensor,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    chunk = 64
    fid_val: Optional[float] = None
    is_mean_val: Optional[float] = None
    is_std_val: Optional[float] = None
    try:
        fid_metric = _build_eval_metric(FrechetInceptionDistance, normalize=True).to(device)
        for i in range(0, all_real.shape[0], chunk):
            fid_metric.update(all_real[i:i+chunk].to(device), real=True)
        for i in range(0, all_gen.shape[0], chunk):
            fid_metric.update(all_gen[i:i+chunk].to(device), real=False)
        fid_val = float(fid_metric.compute().item())
        del fid_metric
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[FID] compute failed: {e}")

    try:
        is_metric = _build_eval_metric(InceptionScore, normalize=True).to(device)
        for i in range(0, all_gen.shape[0], chunk):
            is_metric.update(all_gen[i:i+chunk].to(device))
        is_mean, is_std = is_metric.compute()
        is_mean_val = float(is_mean.item())
        is_std_val = float(is_std.item())
        del is_metric
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[IS] compute failed: {e}")

    return {"fid": fid_val, "is_mean": is_mean_val, "is_std": is_std_val}


@torch.no_grad()
def eval_fid_is(
    generator: nn.Module,
    postprocess_fn,
    eval_loader,
    device: torch.device,
    cfg_scale: float,
    n_samples: int,
    workdir: str,
    step: int,
    label: str = "CFG",
    eval_ref_npz: str = "",
    save_sample_npz: bool = False,
) -> Dict[str, Optional[float]]:
    """Generate n_samples images and compute FID/IS.

    Fast path: collect all generated and real images first (on CPU),
    then run Inception in one shot — same approach as inference_imagenet.py.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        print("[FID/IS] torchmetrics not available, skipping evaluation.")
        return {"fid": None, "is_mean": None, "is_std": None}
    all_gen: torch.Tensor
    all_real: torch.Tensor
    if eval_ref_npz:
        if not os.path.isfile(eval_ref_npz):
            print(f"[FID/IS] reference batch not found: {eval_ref_npz}")
            return {"fid": None, "is_mean": None, "is_std": None}
        all_real_ref, ref_labels = _load_labeled_image_npz(eval_ref_npz)
        n_target = min(int(n_samples), int(all_real_ref.shape[0]), int(ref_labels.shape[0]))
        if n_target <= 0:
            print("[FID/IS] Empty reference batch, skipping.")
            return {"fid": None, "is_mean": None, "is_std": None}
        eval_bsz = int(getattr(eval_loader, "batch_size", 0) or min(64, n_target))
        print(
            f"[FID/IS] Official ImageNet-256 ref batch: generating {n_target} images "
            f"(CFG={cfg_scale}) from labels in {eval_ref_npz}..."
        )
        all_real = all_real_ref[:n_target].cpu().float()
        all_gen = _generate_from_labels(
            generator,
            postprocess_fn,
            ref_labels[:n_target],
            device,
            cfg_scale,
            eval_bsz,
        )
        if save_sample_npz:
            sample_dir = Path(workdir) / "eval_npz"
            sample_path = sample_dir / f"step{step}_{label}.npz"
            _save_labeled_image_npz(str(sample_path), all_gen, ref_labels[:n_target])
            print(f"[FID/IS] Saved official-format sample batch: {sample_path}")
    else:
        if eval_loader is None:
            print("[FID/IS] eval_loader is None and no reference batch was provided.")
            return {"fid": None, "is_mean": None, "is_std": None}
        print(f"[FID/IS] Generating {n_samples} images (CFG={cfg_scale})...")
        all_gen, all_real = _collect_eval_loader_pairs(
            generator,
            postprocess_fn,
            eval_loader,
            device,
            cfg_scale,
            n_samples,
        )

    if all_gen.numel() == 0 or all_real.numel() == 0:
        print("[FID/IS] No usable images collected, skipping.")
        return {"fid": None, "is_mean": None, "is_std": None}

    print(f"[FID/IS] Collected {all_gen.shape[0]} gen + {all_real.shape[0]} real images. Computing metrics...")
    stats = _compute_fid_is_from_tensors(all_gen, all_real, device)
    del all_gen, all_real
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

_RAW_IMAGENET_EXPECTED = {
    "classes": 1_000,
    "train_files": 1_281_167,
    "val_files": 50_000,
    "train_archive_bytes": 147_897_477_120,
    "val_archive_bytes": 6_744_924_160,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw_temperature_calibration(cfg: dict) -> None:
    """Fail closed unless a raw DINO/MoCo tau artifact matches this run."""
    status = str(cfg.get("temperature_calibration_status", "pending")).strip().lower()
    if status == "calibration_capture":
        capture_invariants = {
            "use_wandb=false": not bool(cfg.get("use_wandb", False)),
            "eval_at_start=false": not bool(cfg.get("eval_at_start", False)),
            "total_generated_epochs=0": float(
                cfg.get("total_generated_epochs", -1.0)
            )
            == 0.0,
            "train_max_step_exclusive=1": int(
                cfg.get("train_max_step_exclusive", -1)
            )
            == 1,
        }
        failed = [name for name, valid in capture_invariants.items() if not valid]
        if failed:
            raise RuntimeError(
                "Unsafe temperature calibration capture config: " + ", ".join(failed)
            )
        return
    if status != "ready":
        raise RuntimeError(
            "Raw-pixel temperature calibration is pending; refusing to start "
            "production training with placeholder tau multipliers."
        )

    production_invariants = {
        "use_wandb=true": bool(cfg.get("use_wandb", False)),
        "wandb_project": cfg.get("project") == "Feature encoder - S4 model",
        "require_complete_raw_imagenet=true": bool(
            cfg.get("require_complete_raw_imagenet", False)
        ),
        "use_latent=false": not bool(cfg.get("use_latent", True)),
        "use_cache=false": not bool(cfg.get("use_cache", True)),
        "resolution=256": int(cfg.get("resolution", -1)) == 256,
        "pixel_uint8_bank": str(cfg.get("memory_bank_storage_mode", ""))
        == "pixel_uint8",
        "S4_hidden=384": int(cfg.get("hidden_size", -1)) == 384,
        "S4_depth=12": int(cfg.get("depth", -1)) == 12,
        "S4_heads=6": int(cfg.get("num_heads", -1)) == 6,
        "pixel_input=256x3_p32": int(cfg.get("input_size", -1)) == 256
        and int(cfg.get("in_channels", -1)) == 3
        and int(cfg.get("out_channels", -1)) == 3
        and int(cfg.get("patch_size", -1)) == 32,
        "R_list": [float(value) for value in cfg.get("R_list", ())]
        == [0.2, 0.05, 0.02],
        "B/G/P/N": (
            int(cfg.get("batch_size", -1)),
            int(cfg.get("gen_per_label", -1)),
            int(cfg.get("pos_per_sample", -1)),
            int(cfg.get("neg_per_sample", -1)),
        )
        == (4, 32, 64, 32),
    }
    failed_production = [
        name for name, valid in production_invariants.items() if not valid
    ]
    if failed_production:
        raise RuntimeError(
            "Raw DINO/MoCo production config changed after calibration: "
            + ", ".join(failed_production)
        )

    profile_name = str(cfg.get("layer_temperature_profile", ""))
    if profile_name != "raw_imagenet_dino_moco_symmetric_3seed_p32_v2":
        raise RuntimeError(
            f"Unexpected production temperature profile: {profile_name!r}"
        )
    role = str(cfg.get("temperature_calibration_role", "")).strip().lower()
    if role not in {"dino", "moco"}:
        raise RuntimeError(f"Invalid temperature_calibration_role={role!r}")
    artifact_path = Path(
        str(cfg.get("temperature_calibration_artifact", ""))
    ).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Temperature calibration artifact missing: {artifact_path}")
    expected_artifact_sha = str(
        cfg.get("temperature_calibration_artifact_sha256", "")
    ).strip().lower()
    actual_artifact_sha = _sha256_file(artifact_path)
    if len(expected_artifact_sha) != 64 or actual_artifact_sha != expected_artifact_sha:
        raise RuntimeError(
            "Temperature calibration artifact SHA256 mismatch: "
            f"{actual_artifact_sha} != {expected_artifact_sha or '<missing>'}"
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        int(artifact.get("schema_version", 0)) != 3
        or artifact.get("calibration_kind")
        != "symmetric_raw_imagenet_dino_moco"
        or artifact.get("centering") != "geometric_mean_one"
        or artifact.get("required_seeds") != [43, 44, 45]
        or float(artifact.get("min_distance", -1.0)) != 0.02
        or float(artifact.get("huber_tuning", -1.0)) != 1.5
        or float(artifact.get("trim_fraction", -1.0)) != 0.1
    ):
        raise RuntimeError("Temperature calibration artifact protocol is invalid")

    capture_protocol = artifact.get("capture_protocol") or {}
    capture_generator_geometry = capture_protocol.get("generator_geometry") or {}
    capture_batch = capture_protocol.get("batch") or {}
    capture_r_values = [
        float(value) for value in capture_protocol.get("R_list", ())
    ]
    feature_layout = capture_protocol.get("feature_layout") or {}
    feature_stage_counts = {
        stage: sum(
            1 for value in feature_layout.values() if value.get("stage") == stage
        )
        for stage in ("stage3", "stage4")
    }
    feature_layout_valid = (
        len(feature_layout) == 42
        and feature_stage_counts == {"stage3": 28, "stage4": 14}
        and all(
            int(value.get("dimension", -1))
            == (1024 if value.get("stage") == "stage3" else 2048)
            for value in feature_layout.values()
        )
    )
    fixed_input = artifact.get("fixed_input_feature_multipliers") or {}
    if (
        capture_generator_geometry
        != {
            "input_size": 256,
            "in_channels": 3,
            "out_channels": 3,
            "patch_size": 32,
            "image_tokens": 64,
            "class_tokens": 16,
        }
        or capture_batch != {"B": 4, "G": 32, "P": 64, "N": 32}
        or capture_r_values != [0.2, 0.05, 0.02]
        or int(capture_protocol.get("max_token_rows_per_feature", -1)) != 64
        or not feature_layout_valid
        or float(fixed_input.get("global", float("nan"))) != 1.0
        or float(fixed_input.get("norm_x", float("nan"))) != 1.0
    ):
        raise RuntimeError("Temperature calibration capture protocol is invalid")

    reciprocity = artifact.get("reciprocity_validation") or {}
    if reciprocity.get("verified") is not True:
        raise RuntimeError("Temperature calibration reciprocity was not verified")
    max_log_error = float(
        reciprocity.get("max_log_reciprocity_error", float("nan"))
    )
    if not math.isfinite(max_log_error) or max_log_error > 1e-10:
        raise RuntimeError("Temperature calibration reciprocity tolerance is invalid")
    reciprocity_stages = reciprocity.get("stages") or {}
    for stage in ("stage3", "stage4"):
        validation = reciprocity_stages.get(stage) or {}
        stage_log_error = float(validation.get("abs_log_product", float("nan")))
        stage_product = float(validation.get("product", float("nan")))
        if (
            not math.isfinite(stage_log_error)
            or stage_log_error > max_log_error
            or not math.isfinite(stage_product)
            or not math.isclose(
                stage_product, math.exp(stage_log_error), rel_tol=0.0, abs_tol=1e-10
            )
            and not math.isclose(
                stage_product, math.exp(-stage_log_error), rel_tol=0.0, abs_tol=1e-10
            )
        ):
            raise RuntimeError(f"{stage} temperature reciprocity evidence is invalid")

    raw_root = Path(str(cfg.get("imagenet_path", ""))).resolve()
    raw_manifest = raw_root / "raw_imagenet_manifest.json"
    if not raw_manifest.is_file():
        raise FileNotFoundError(f"Raw ImageNet manifest missing: {raw_manifest}")
    current_manifest_sha = _sha256_file(raw_manifest)
    configured_manifest_sha = str(
        cfg.get("temperature_calibration_raw_manifest_sha256", "")
    ).strip().lower()
    artifact_dataset = artifact.get("dataset_provenance") or {}
    artifact_manifest_sha = str(artifact_dataset.get("manifest_sha256", "")).lower()
    if not (
        len(configured_manifest_sha) == 64
        and current_manifest_sha == configured_manifest_sha == artifact_manifest_sha
    ):
        raise RuntimeError(
            "Raw ImageNet manifest is not the one used for tau calibration"
        )

    extractor_name = resolve_feature_extractor_name(cfg)
    expected_extractor = "dino_resnet50" if role == "dino" else "moco_v2_resnet50"
    if extractor_name != expected_extractor:
        raise RuntimeError(
            f"Calibration role {role} does not match extractor {extractor_name}"
        )
    checkpoint_value = cfg.get("feature_checkpoint")
    checkpoint = Path(str(checkpoint_value or "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Feature checkpoint missing: {checkpoint}")
    encoder_provenance = (artifact.get("encoder_provenance") or {}).get(role) or {}
    if (
        "temperature_calibration_inheritance_manifest" in cfg
        or "temperature_calibration_inheritance_manifest_sha256" in cfg
    ):
        from temperature_calibration_inheritance import validate_dino_temperature_inheritance

        validate_dino_temperature_inheritance(cfg, artifact, checkpoint)
        print(
            "[raw-tau] verified tuned-DINO weights-only ablation; "
            "inheriting baseline numeric temperatures without refitting",
            flush=True,
        )
    elif (
        encoder_provenance.get("feature_extractor") != expected_extractor
        or int(encoder_provenance.get("checkpoint_bytes", -1))
        != checkpoint.stat().st_size
        or str(encoder_provenance.get("checkpoint_sha256", "")).lower()
        != _sha256_file(checkpoint)
    ):
        raise RuntimeError(
            f"{role} feature checkpoint differs from the calibrated checkpoint"
        )

    profiles = artifact.get("symmetric_profiles") or {}
    if set(profiles) != {"dino", "moco"}:
        raise RuntimeError("Calibration artifact lacks the DINO/MoCo pair profiles")
    configured = _resolve_layer_temperature_multipliers(cfg)
    for stage in ("default", "stage3", "stage4"):
        expected = float(profiles[role][stage])
        actual = float(configured.get(stage, float("nan")))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-11):
            raise RuntimeError(
                f"Configured {role} {stage} tau multiplier {actual} != artifact {expected}"
            )
    for stage in ("stage3", "stage4"):
        dino_value = float(profiles["dino"][stage])
        moco_value = float(profiles["moco"][stage])
        ratio = float(
            artifact["candidates"]["moco"]["stages"][stage][
                "selected_multiplier"
            ]
        )
        if not math.isclose(dino_value * moco_value, 1.0, abs_tol=5e-11):
            raise RuntimeError(f"{stage} symmetric tau product is not one")
        if not math.isclose(
            moco_value / dino_value, ratio, rel_tol=0.0, abs_tol=5e-11
        ):
            raise RuntimeError(f"{stage} symmetric tau ratio is inconsistent")

    capture_validation = artifact.get("capture_validation") or {}
    for seed in ("43", "44", "45"):
        dino_hashes = (
            (capture_validation.get("dino") or {}).get(seed) or {}
        ).get("tensor_sha256")
        moco_hashes = (
            (capture_validation.get("moco") or {}).get(seed) or {}
        ).get("tensor_sha256")
        required_hashes = {
            "labels",
            "positive_samples",
            "negative_samples",
            "generated_samples",
            "cfg_scales",
        }
        if (
            not isinstance(dino_hashes, dict)
            or not isinstance(moco_hashes, dict)
            or not required_hashes.issubset(dino_hashes)
            or any(dino_hashes[key] != moco_hashes.get(key) for key in required_hashes)
        ):
            raise RuntimeError(f"Seed {seed} calibration input hashes do not match")

    activation_kwargs = cfg.get("activation_kwargs") or {}
    feature_profile = _resolve_feature_loss_group_weights(cfg)
    feature_default = float(feature_profile.get("default", 1.0))
    effective_feature_weight = {
        group: float(feature_profile.get(group, feature_default))
        for group in ("global", "norm_x", "stage1", "stage2", "stage3", "stage4")
    }
    if (
        list(activation_kwargs.get("active_stages", ())) != ["stage3", "stage4"]
        or activation_kwargs.get("with_global") is not True
        or activation_kwargs.get("with_norm_x") is not True
        or bool(activation_kwargs.get("exclude_terminal_block", False))
        or int(activation_kwargs.get("every_k_block", -1)) != 2
        or effective_feature_weight
        != {
            "global": 1.0,
            "norm_x": 2.0,
            "stage1": 0.0,
            "stage2": 0.0,
            "stage3": 1.0,
            "stage4": 1.0,
        }
        or not bool(cfg.get("feature_loss_group_normalize", False))
        or not bool(cfg.get("prune_skipped_feature_tensors", False))
        or not bool(cfg.get("feature_include_norm_x", False))
    ):
        raise RuntimeError(
            "Production feature set must keep global+norm_x+stage3+stage4 and "
            "remove only stage1+stage2"
        )


def _load_complete_raw_imagenet_manifest(imagenet_path: str) -> dict:
    """Load and validate the provenance manifest for raw ILSVRC2012.

    The downloader publishes this file only after checking every class/file in
    train and validation against the official devkit.  Keep the checks here
    deliberately strict so a partial transfer cannot silently become a run.
    """
    root = Path(imagenet_path).resolve()
    manifest_path = root / "raw_imagenet_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Required raw-ImageNet manifest is missing: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise RuntimeError(
            f"Raw ImageNet manifest is not complete: {manifest_path}"
        )
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError(
            f"Unsupported raw ImageNet manifest schema: {manifest.get('schema_version')!r}"
        )
    manifest_root = Path(str(manifest.get("output_root", ""))).resolve()
    if manifest_root != root:
        raise RuntimeError(
            f"Raw ImageNet manifest root mismatch: {manifest_root} != {root}"
        )

    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise RuntimeError(f"Raw ImageNet manifest lacks expected counts: {manifest_path}")
    for key in ("classes", "train_files", "val_files"):
        actual = int(expected.get(key, -1))
        canonical = _RAW_IMAGENET_EXPECTED[key]
        if actual != canonical:
            raise RuntimeError(
                f"Raw ImageNet manifest {key} mismatch: {actual:,} != {canonical:,}"
            )

    mapping = manifest.get("mapping")
    if not isinstance(mapping, dict) or mapping.get(
        "train_val_devkit_wnids_identical"
    ) is not True:
        raise RuntimeError("Raw ImageNet train/val/devkit mappings are not verified")
    class_to_idx = mapping.get("imagefolder_class_to_index")
    official_id_to_wnid = mapping.get("official_imagenet_id_to_wnid")
    if not isinstance(class_to_idx, dict) or not isinstance(
        official_id_to_wnid, dict
    ):
        raise RuntimeError("Raw ImageNet manifest lacks canonical class mappings")
    sorted_wnids = sorted(class_to_idx)
    if (
        len(sorted_wnids) != _RAW_IMAGENET_EXPECTED["classes"]
        or any(
            len(wnid) != 9
            or not wnid.startswith("n")
            or not wnid[1:].isdigit()
            for wnid in sorted_wnids
        )
        or class_to_idx != {wnid: index for index, wnid in enumerate(sorted_wnids)}
        or set(official_id_to_wnid.values()) != set(sorted_wnids)
    ):
        raise RuntimeError("Raw ImageNet manifest contains an invalid WNID mapping")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Raw ImageNet manifest lacks source archive provenance")
    for split, size_key in (
        ("train", "train_archive_bytes"),
        ("val", "val_archive_bytes"),
    ):
        remote = source.get(split)
        if not isinstance(remote, dict):
            raise RuntimeError(f"Raw ImageNet manifest lacks {split} source metadata")
        actual_size = int(remote.get("total_bytes", -1))
        canonical_size = _RAW_IMAGENET_EXPECTED[size_key]
        if actual_size != canonical_size:
            raise RuntimeError(
                f"Official {split} archive size mismatch: "
                f"{actual_size:,} != {canonical_size:,}"
            )

    for split, count_key in (("train", "train_files"), ("val", "val_files")):
        split_manifest = manifest.get(split)
        if not isinstance(split_manifest, dict) or split_manifest.get(
            "complete"
        ) is not True:
            raise RuntimeError(f"Raw ImageNet {split} split is not verified complete")
        if int(split_manifest.get("class_count", -1)) != _RAW_IMAGENET_EXPECTED[
            "classes"
        ]:
            raise RuntimeError(f"Raw ImageNet {split} class count is invalid")
        if int(split_manifest.get("file_count", -1)) != _RAW_IMAGENET_EXPECTED[
            count_key
        ]:
            raise RuntimeError(f"Raw ImageNet {split} file count is invalid")
        per_class = split_manifest.get("per_class_file_count")
        if (
            not isinstance(per_class, dict)
            or set(per_class) != set(sorted_wnids)
            or sum(int(value) for value in per_class.values())
            != _RAW_IMAGENET_EXPECTED[count_key]
        ):
            raise RuntimeError(
                f"Raw ImageNet {split} per-class counts are incomplete"
            )
    return manifest


def _validate_raw_imagefolder_dataset(dataset, split: str, manifest: dict) -> None:
    """Match torchvision's live ImageFolder scan to the verified manifest."""
    count_key = "train_files" if split == "train" else "val_files"
    expected_count = _RAW_IMAGENET_EXPECTED[count_key]
    if len(dataset) != expected_count:
        raise RuntimeError(
            f"Raw ImageNet {split} live file count mismatch: "
            f"{len(dataset):,} != {expected_count:,}"
        )
    expected_mapping = manifest["mapping"]["imagefolder_class_to_index"]
    if getattr(dataset, "class_to_idx", None) != expected_mapping:
        raise RuntimeError(
            f"Raw ImageNet {split} live ImageFolder mapping differs from the devkit"
        )
    targets = np.asarray(getattr(dataset, "targets", ()), dtype=np.int64)
    if targets.size != expected_count:
        raise RuntimeError(f"Raw ImageNet {split} ImageFolder targets are incomplete")
    live_counts = np.bincount(
        targets, minlength=_RAW_IMAGENET_EXPECTED["classes"]
    )
    expected_per_class = manifest[split]["per_class_file_count"]
    ordered_expected = np.asarray(
        [
            int(expected_per_class[wnid])
            for wnid, _ in sorted(expected_mapping.items(), key=lambda item: item[1])
        ],
        dtype=np.int64,
    )
    if not np.array_equal(live_counts, ordered_expected):
        mismatched = np.flatnonzero(live_counts != ordered_expected)[:10].tolist()
        raise RuntimeError(
            f"Raw ImageNet {split} live per-class counts differ; "
            f"mismatched class indices={mismatched}"
        )

def train_gen(cfg: dict, workdir: str, rank: int, world_size: int, device: torch.device) -> None:
    if bool(cfg.get("require_raw_temperature_calibration", False)):
        _validate_raw_temperature_calibration(cfg)
    process_seed = int(cfg.get("seed", 42)) + rank
    torch.manual_seed(process_seed)
    if bool(cfg.get("seed_host_rng", False)):
        # Opt-in for tightly matched causal ablations. Older experiments retain
        # their original host-RNG behavior unless this flag is explicitly set.
        np.random.seed(process_seed % (2**32))
    log_every = int(cfg.get("log_every_k", 20))

    # --- Build generator ---
    raw_cfg = cfg.get("_raw", {})
    throughput_opt_level = int(cfg.get("throughput_opt_level", 0))
    _configure_level4_cuda_runtime(cfg, throughput_opt_level, device)
    gen_raw = build_ditgen_from_config(raw_cfg.get("model", cfg), raw_cfg.get("dataset", cfg))
    gen_raw = gen_raw.to(device)
    if world_size > 1:
        if throughput_opt_level >= 3:
            generator = DDP(
                gen_raw,
                device_ids=[device.index],
                find_unused_parameters=False,
                static_graph=True,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )
        else:
            generator = DDP(gen_raw, device_ids=[device.index], find_unused_parameters=False)
    else:
        generator = gen_raw

    ema = EMA(
        gen_raw,
        decay=float(cfg.get("ema_decay", 0.999)),
        foreach=throughput_opt_level >= 3,
    )

    # --- Optimizer ---
    optimizer = build_optimizer(gen_raw, cfg)
    base_lr   = float(cfg.get("lr", 4e-4))
    warmup    = int(cfg.get("warmup_steps", 10000))
    warmup_init_lr = float(cfg.get("warmup_init_lr", 1e-6))

    # --- Load frozen feature encoder ---
    # Construction consumes different amounts of RNG for MAE and torchvision
    # ResNets. Preserve the post-generator RNG state so the three causal
    # comparisons draw the same labels, CFG values, and generator noise.
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        feature_extractor = load_feature_extractor(cfg, device)
    if is_main_process(rank):
        print(
            f"[feature] extractor={resolve_feature_extractor_name(cfg)} "
            f"checkpoint={cfg.get('feature_checkpoint') or cfg.get('mae_checkpoint')}",
            flush=True,
        )
    (
        feature_adapter,
        feature_adapter_target,
        feature_adapter_optimizer,
    ) = build_feature_adapter_system(feature_extractor, cfg, device, world_size)
    (
        feature_discriminator,
        feature_discriminator_optimizer,
    ) = build_feature_discriminator_system(feature_extractor, cfg, device, world_size)
    adversarial_system = build_adversarial_system(cfg, device)
    if adversarial_system is not None and is_main_process(rank):
        print(
            f"[adversarial] mode={adversarial_system.mode} "
            "frozen DINO drift unchanged; generator uses the previous target snapshot",
            flush=True,
        )

    # --- Data ---
    imagenet_path = cfg.get("imagenet_path") or os.environ.get("IMAGENET_PATH", "")
    cache_path    = cfg.get("cache_path") or os.environ.get("IMAGENET_CACHE_PATH", "")
    use_latent    = bool(cfg.get("use_latent", True))
    use_cache     = bool(cfg.get("use_cache", True))
    use_aug       = bool(cfg.get("use_aug", False))
    resolution    = int(cfg.get("resolution", 256))
    batch_size    = int(cfg.get("batch_size", 128))
    eval_bsz      = int(cfg.get("eval_batch_size", 256))
    memory_bank_storage_mode = str(
        cfg.get("memory_bank_storage_mode", "raw")
    ).strip().lower()
    raw_train_uint8 = bool(cfg.get("raw_train_uint8", False))
    if memory_bank_storage_mode == "pixel_uint8" and (use_latent or use_cache):
        raise ValueError(
            "memory_bank_storage_mode='pixel_uint8' requires a direct pixel "
            "dataset (use_latent=false, use_cache=false)."
        )
    if raw_train_uint8 and memory_bank_storage_mode != "pixel_uint8":
        raise ValueError(
            "raw_train_uint8=true requires "
            "memory_bank_storage_mode='pixel_uint8'."
        )

    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    eval_prefetch_factor = int(
        cfg.get("eval_prefetch_factor", prefetch_factor)
    )

    if bool(cfg.get("require_complete_pixel_cache", False)):
        if use_latent or use_cache:
            raise ValueError(
                "require_complete_pixel_cache is only valid for direct pixel "
                "ImageFolder datasets."
            )
        manifest_path = Path(imagenet_path) / "_pixel_cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Required pixel-cache manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as manifest_handle:
            pixel_manifest = json.load(manifest_handle)
        split_status = pixel_manifest.get("split_status", {})
        incomplete_splits = [
            split
            for split in ("train", "val")
            if not bool(split_status.get(split, {}).get("complete", False))
        ]
        if not bool(pixel_manifest.get("complete_requested", False)) or incomplete_splits:
            raise RuntimeError(
                "Pixel cache is incomplete; refusing to train from a partial "
                f"ImageFolder. incomplete_splits={incomplete_splits} "
                f"manifest={manifest_path}"
            )
        expected_quality = int(cfg.get("pixel_cache_jpeg_quality", 0) or 0)
        actual_quality = int(pixel_manifest.get("jpeg_quality", 0) or 0)
        if expected_quality and actual_quality != expected_quality:
            raise ValueError(
                "Pixel-cache JPEG quality does not match the experiment "
                f"config: expected={expected_quality}, actual={actual_quality}."
            )
        if is_main_process(rank):
            print(
                "[pixel-cache] verified complete "
                f"train={split_status['train']['output_count']} "
                f"val={split_status['val']['output_count']} "
                f"jpeg_quality={actual_quality} manifest={manifest_path}",
                flush=True,
            )

    raw_imagenet_manifest = None
    if bool(cfg.get("require_complete_raw_imagenet", False)):
        if use_latent or use_cache:
            raise ValueError(
                "require_complete_raw_imagenet requires direct pixels "
                "(use_latent=false, use_cache=false)."
            )
        raw_imagenet_manifest = _load_complete_raw_imagenet_manifest(imagenet_path)

    raw_image_io_concurrency = int(
        cfg.get("raw_image_io_concurrency_per_rank", 0)
    )
    if raw_image_io_concurrency < 0:
        raise ValueError(
            "raw_image_io_concurrency_per_rank must be non-negative"
        )
    if raw_image_io_concurrency > 0 and (use_latent or use_cache):
        raise ValueError(
            "raw_image_io_concurrency_per_rank is only valid for direct raw inputs"
        )
    # Rank 0's train and eval loaders share this gate so an evaluation cannot
    # temporarily double the rank's raw-image read/decode concurrency.
    raw_image_io_semaphore = create_raw_image_io_semaphore(
        raw_image_io_concurrency
    )

    train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
        imagenet_path=imagenet_path,
        resolution=resolution,
        batch_size=cfg["loader_batch_size"],
        split="train",
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        cache_path=cache_path,
        num_workers=int(cfg.get("num_workers", 8)),
        prefetch_factor=prefetch_factor,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)),
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
        latent_device=device,
        vae_model_id=str(cfg.get("feature_vae_model_id", "stabilityai/sd-vae-ft-mse")),
        vae_revision=(
            str(cfg["feature_vae_revision"])
            if cfg.get("feature_vae_revision")
            else None
        ),
        return_uint8=raw_train_uint8,
        raw_image_io_concurrency=raw_image_io_concurrency,
        raw_image_io_semaphore=raw_image_io_semaphore,
    )
    eval_loader = None
    eval_postprocess_fn = None
    if is_main_process(rank):
        # Eval is only run on rank 0 below. Build a non-sharded val loader so
        # intermediate FID/IS uses the full validation distribution.
        eval_loader, _, eval_postprocess_fn = create_imagenet_split(
            imagenet_path=imagenet_path,
            resolution=resolution,
            batch_size=eval_bsz,
            split="val",
            use_aug=False,
            use_latent=use_latent,
            use_cache=use_cache,
            cache_path=cache_path,
            num_workers=int(cfg.get("num_workers", 8)),
            prefetch_factor=eval_prefetch_factor,
            pin_memory=bool(cfg.get("pin_memory", True)),
            persistent_workers=bool(cfg.get("eval_persistent_workers", False)),
            distributed=False,
            rank=0,
            world_size=1,
            latent_device=device,
            vae_model_id=str(cfg.get("feature_vae_model_id", "stabilityai/sd-vae-ft-mse")),
            vae_revision=(
                str(cfg["feature_vae_revision"])
                if cfg.get("feature_vae_revision")
                else None
            ),
            raw_image_io_concurrency=raw_image_io_concurrency,
            raw_image_io_semaphore=raw_image_io_semaphore,
        )
        print(
            "[data-loader] "
            f"train_workers={int(cfg.get('num_workers', 8))} "
            f"train_prefetch={prefetch_factor} "
            f"train_persistent={bool(cfg.get('persistent_workers', True))} "
            f"raw_train_uint8={raw_train_uint8}; "
            f"raw_image_io_concurrency_per_rank={raw_image_io_concurrency}; "
            f"eval_prefetch={eval_prefetch_factor} "
            f"eval_persistent={bool(cfg.get('eval_persistent_workers', False))}",
            flush=True,
        )

    if raw_imagenet_manifest is not None:
        _validate_raw_imagefolder_dataset(
            train_loader.dataset, "train", raw_imagenet_manifest
        )
        if is_main_process(rank):
            if eval_loader is None:
                raise RuntimeError("Raw ImageNet validation loader was not constructed")
            _validate_raw_imagefolder_dataset(
                eval_loader.dataset, "val", raw_imagenet_manifest
            )
            print(
                "[raw-imagenet] verified live ImageFolder scan "
                "train=1,281,167 val=50,000 classes=1,000; "
                "input=official JPEG -> ADM center-crop 256 -> train-only hflip -> [-1,1]",
                flush=True,
            )

    # --- Memory banks ---
    positive_bank_size = int(cfg.get("positive_bank_size", 128))
    negative_bank_size = int(cfg.get("negative_bank_size", 1000))
    num_classes        = int(cfg.get("num_classes", 1000))
    banks_per_rank     = max(1, int(cfg.get("banks_per_rank", 1)))
    bank_batch_sizes   = _split_evenly(batch_size, banks_per_rank)
    positive_bank_backend = str(
        cfg.get("positive_memory_bank_backend", "dense")
    ).lower().strip()
    if positive_bank_backend not in {"dense", "zstd_delta"}:
        raise ValueError(
            "positive_memory_bank_backend must be 'dense' or 'zstd_delta', got "
            f"{positive_bank_backend!r}."
        )
    if (
        positive_bank_backend == "zstd_delta"
        and memory_bank_storage_mode != "pixel_uint8"
    ):
        raise ValueError(
            "positive_memory_bank_backend='zstd_delta' requires "
            "memory_bank_storage_mode='pixel_uint8'."
        )
    positive_bank_zstd_level = int(
        cfg.get("positive_memory_bank_zstd_level", 1)
    )
    positive_bank_codec_workers = int(
        cfg.get("positive_memory_bank_codec_workers", 1)
    )
    pos_banks = [
        (
            CompressedPixelMemoryBank(
                num_classes=num_classes,
                max_size=positive_bank_size,
                compression_level=positive_bank_zstd_level,
                codec_workers=positive_bank_codec_workers,
            )
            if positive_bank_backend == "zstd_delta"
            else ArrayMemoryBank(
                num_classes=num_classes,
                max_size=positive_bank_size,
                storage_mode=memory_bank_storage_mode,
            )
        )
        for _ in range(banks_per_rank)
    ]
    neg_banks = [
        ArrayMemoryBank(
            num_classes=1,
            max_size=negative_bank_size,
            storage_mode=memory_bank_storage_mode,
        )
        for _ in range(banks_per_rank)
    ]
    if is_main_process(rank):
        print(
            f"[banks] storage_mode={memory_bank_storage_mode} "
            f"positive_backend={positive_bank_backend} "
            f"positive={positive_bank_size}/class negative={negative_bank_size}",
            flush=True,
        )
    if is_main_process(rank) and banks_per_rank > 1:
        print(
            f"[banks] using {banks_per_rank} independent bank pairs per rank; "
            f"label quotas per bank={bank_batch_sizes}"
        )

    push_per_step  = int(cfg.get("push_per_step", 128))
    push_at_resume = int(cfg.get("push_at_resume", 3000))
    pos_per_sample = int(cfg.get("pos_per_sample", 64))
    neg_per_sample = int(cfg.get("neg_per_sample", 32))

    # --- Mix-alpha tracker ---
    # Always created so per-step α1/β1 are logged for every config (rev-drift,
    # fwd-drift, dual-drift linear, dual-drift adaptive). Whether mix_alpha is
    # *driven* by the tracker is still gated by `mix_alpha_adaptive`.
    mix_alpha_mode = str(cfg.get("mix_alpha_adaptive_mode", "hedge")).lower().strip()
    mix_alpha_initial = float(cfg.get("mix_alpha_adaptive_initial", 0.5))
    hedge_gamma = float(cfg.get("mix_alpha_adaptive_gamma", 2.0))
    hedge_eta = float(cfg.get("mix_alpha_adaptive_eta", 1.0e-4))
    hedge_decay = float(cfg.get("mix_alpha_adaptive_decay", 1.0e-4))
    hedge_gamma_warmup = int(cfg.get("mix_alpha_adaptive_gamma_warmup_steps", 10000))
    mix_alpha_tracker: MixAlphaTracker = MixAlphaTracker(
        gamma=hedge_gamma,
        eta=hedge_eta,
        decay=hedge_decay,
        gamma_warmup_steps=hedge_gamma_warmup,
        mode=mix_alpha_mode,
        initial_alpha=mix_alpha_initial,
    )
    if is_main_process(rank) and bool(cfg.get("mix_alpha_adaptive", False)):
        ratio_formula = (
            "a=gen_winners/min(gen_total,pos_total), "
            "b=pos_winners/min(gen_total,pos_total)"
        )
        if mix_alpha_mode in ("closed_form", "ratio", "data_only", "winner_ratio"):
            formula = (
                f"closed-form/data-only: {ratio_formula}; "
                "fwd_coef=a*b/(a*b + (1-a)*(1-b)); "
                "mix_alpha=1-fwd_coef; no time-logit/eta/decay/gamma term"
            )
        elif mix_alpha_mode in ("hedge_no_time", "data_hedge", "no_time", "winner_hedge"):
            formula = (
                f"data-hedge/no-time: {ratio_formula}; γ_eff=γ*step/T; "
                f"L_b=γ_eff(1-a), L_v=γ_eff(1-b); "
                f"ℓ←(1-{hedge_decay})·ℓ-{hedge_eta}·L; "
                f"initial_mix_alpha={mix_alpha_initial}; "
                "mix_alpha=sigmoid(ℓ_b-ℓ_v)"
            )
        else:
            formula = (
                f"hedge+logit-time-bias: {ratio_formula}; "
                f"L_b=γ_eff(1-a), L_v=γ_eff(1-b); "
                f"ℓ←(1-{hedge_decay})·ℓ-{hedge_eta}·L; "
                f"mix_alpha=sigmoid((ℓ_b-ℓ_v)+log(t/(1-t))) "
                f"(initial_mix_alpha={mix_alpha_initial}, γ={hedge_gamma}, "
                f"γ_warmup={hedge_gamma_warmup} steps)"
            )
        print(f"[mix_alpha] adaptive {mix_alpha_mode} schedule enabled; {formula}")

    if (
        is_main_process(rank)
        and str(cfg.get("drift_matching", "rev-drift")).lower().strip()
        == "dual-drift"
    ):
        share_requested = bool(cfg.get("dual_drift_share_distances", True))
        self_mask_raw = bool(cfg.get("self_mask_on_raw", False))
        share_active = share_requested and not self_mask_raw
        print(
            "[dual-drift] shared normalized distance grid "
            f"active={str(share_active).lower()} "
            f"(requested={str(share_requested).lower()}, "
            f"self_mask_on_raw={str(self_mask_raw).lower()})"
        )

    if is_main_process(rank):
        drift_mode = str(cfg.get("drift_matching", "rev-drift")).lower().strip()
        rev_top_p = float(cfg.get("rev_drift_top_p", 1.0))
        fwd_top_p = float(cfg.get("fwd_drift_top_p", 1.0))
        top_p_min_keep = int(cfg.get("drift_top_p_min_keep", 1))
        if drift_mode == "dual-drift":
            top_p_summary = f"rev={rev_top_p:g} fwd={fwd_top_p:g}"
        elif drift_mode == "fwd-drift":
            top_p_summary = f"fwd={fwd_top_p:g}"
        else:
            top_p_summary = f"rev={rev_top_p:g}"
        print(
            f"[drift-top-p] {top_p_summary} min_keep={top_p_min_keep} "
            "(1.0 disables nucleus truncation)"
        )
        top_k_pos = int(cfg.get("drift_top_k_pos", 0))
        top_k_neg = int(cfg.get("drift_top_k_neg", 0))
        top_k_groups = _resolve_drift_top_k_groups(cfg)
        top_k_group_summary = (
            "all" if top_k_groups is None else ",".join(top_k_groups)
        )
        print(
            f"[drift-top-k] pos={top_k_pos} neg={top_k_neg} "
            f"groups={top_k_group_summary} "
            "(0 disables fixed-support truncation)"
        )
        if drift_mode == "rev-drift":
            print(
                "[rev-affinity] "
                f"kernel={cfg.get('rev_drift_affinity_kernel', 'exponential')} "
                f"shape={float(cfg.get('rev_drift_kernel_shape', 1.0)):g} "
                f"adaptive_k_pos={int(cfg.get('rev_drift_kernel_adaptive_k_pos', 0))} "
                f"adaptive_k_neg={int(cfg.get('rev_drift_kernel_adaptive_k_neg', 0))} "
                f"adaptive_margin={float(cfg.get('rev_drift_kernel_adaptive_margin', 1.05)):g} "
                f"mix_weight={float(cfg.get('rev_drift_kernel_mix_weight', 0.5)):g} "
                f"temperature_mix={cfg.get('rev_drift_kernel_temperature_mix', [])} "
                f"temperature_mix_weights={cfg.get('rev_drift_kernel_temperature_mix_weights', [])} "
                f"force_multiplier={float(cfg.get('rev_drift_force_multiplier', 1.0)):g}"
            )
        layer_temperature_profile = str(
            cfg.get("layer_temperature_profile", "uniform")
        )
        layer_temperature_multipliers = (
            _resolve_layer_temperature_multipliers(cfg)
        )
        multiplier_summary = " ".join(
            f"{stage}={layer_temperature_multipliers.get(stage, layer_temperature_multipliers.get('default', 1.0)):g}"
            for stage in _STOCHASTIC_FEATURE_STAGES
        )
        print(
            "[drift-temperature] "
            f"profile={layer_temperature_profile} {multiplier_summary}; "
            "raw/global features=1"
        )

        feature_loss_profile = str(cfg.get("feature_loss_profile", "all"))
        feature_group_weights = _resolve_feature_loss_group_weights(cfg)
        feature_weight_summary = " ".join(
            f"{group}={feature_group_weights.get(group, feature_group_weights.get('default', 1.0)):g}"
            for group in _FEATURE_LOSS_GROUPS
        )
        print(
            "[feature-loss] "
            f"profile={feature_loss_profile} {feature_weight_summary}; "
            f"mass_normalize={str(bool(cfg.get('feature_loss_group_normalize', False))).lower()}"
        )
        if feature_adapter is None:
            print("[feature-adapter] disabled (frozen MAE metric)")
        else:
            raw_adapter = (
                feature_adapter.module
                if hasattr(feature_adapter, "module")
                else feature_adapter
            )
            trainable_parameters = sum(
                parameter.numel() for parameter in raw_adapter.parameters()
            )
            adapter_objective_name = str(
                cfg.get("feature_adapter_objective", "supcon")
            ).lower().strip()
            adapter_global_name = (
                "global_statistics"
                if adapter_objective_name
                in {
                    "raw_drift_snr",
                    "raw_drift_snr_consistency",
                    "dino_cf_drift_realreal_infonce",
                }
                else "global_contrastive"
            )
            adapter_global_enabled = bool(
                cfg.get(
                    "feature_adapter_global_statistics"
                    if adapter_objective_name
                    in {
                        "raw_drift_snr",
                        "raw_drift_snr_consistency",
                        "dino_cf_drift_realreal_infonce",
                    }
                    else "feature_adapter_global_contrastive",
                    adapter_objective_name
                    in {
                        "raw_drift_snr",
                        "raw_drift_snr_consistency",
                        "dino_cf_drift_realreal_infonce",
                    },
                )
            )
            print(
                "[feature-adapter] enabled "
                f"objective={adapter_objective_name} "
                f"stages={','.join(raw_adapter.stages)} "
                f"bottleneck={int(cfg.get('feature_adapter_bottleneck', 64))} "
                f"samples_per_class={int(cfg.get('feature_adapter_samples_per_class', 8))} "
                f"generated_samples_per_class={int(cfg.get('feature_adapter_generated_samples_per_class', cfg.get('gen_per_label', 64)))} "
                f"start_step={int(cfg.get('feature_adapter_start_step', 0))} "
                f"update_freq={int(cfg.get('feature_adapter_update_freq', 1))} "
                f"require_unique_reals={str(bool(cfg.get('feature_adapter_require_unique_reals', False))).lower()} "
                f"{adapter_global_name}={str(adapter_global_enabled).lower()} "
                f"lr={float(cfg.get('feature_adapter_lr', 1.0e-4)):g} "
                f"temp={float(cfg.get('feature_adapter_temp', 0.1)):g} "
                f"loss_weight={float(cfg.get('feature_adapter_loss_weight', 1.0)):g} "
                f"drift_align_lambda={float(cfg.get('feature_adapter_drift_align_lambda', 0.0)):g} "
                f"distance_scale_lambda={float(cfg.get('feature_adapter_distance_scale_lambda', 0.0)):g} "
                f"distance_mean_lambda={float(cfg.get('feature_adapter_distance_mean_lambda', 0.0)):g} "
                f"cf_weight={float(cfg.get('feature_adapter_cf_weight', 0.0)):g} "
                f"cf_rho={float(cfg.get('feature_adapter_cf_rho', 1.0)):g} "
                f"null_weight={float(cfg.get('feature_adapter_null_weight', 0.0)):g} "
                f"relation_weight={float(cfg.get('feature_adapter_relation_weight', 0.0)):g} "
                f"mdino_weight={float(cfg.get('feature_adapter_mdino_weight', 0.0)):g} "
                f"mdino_mask_ratio={float(cfg.get('feature_adapter_mdino_mask_ratio', 0.0)):g} "
                f"mdino_patch={int(cfg.get('feature_adapter_mdino_patch_size', 0))} "
                f"mdino_samples={int(cfg.get('feature_adapter_mdino_samples_per_class', 0))} "
                f"mdino_predictor_lr={float(cfg.get('feature_adapter_mdino_predictor_lr', cfg.get('feature_adapter_lr', 1.0e-4))):g} "
                f"ema={float(cfg.get('feature_adapter_ema_decay', 0.999)):g} "
                f"parameters={trainable_parameters}"
            )
            if adapter_objective_name == "raw_drift_snr":
                print(
                    "[feature-adapter-v2] "
                    f"real_pool={int(cfg.get('feature_adapter_real_pool_per_class', 32))} "
                    f"query={int(cfg.get('feature_adapter_query_count', 8))} "
                    f"bank_a={int(cfg.get('feature_adapter_bank_count', 12))} "
                    f"bank_b={int(cfg.get('feature_adapter_bank_count', 12))} "
                    "different_class_negatives=0 "
                    f"snr_epsilon={float(cfg.get('feature_adapter_snr_epsilon', 1.0e-6)):g} "
                    f"consistency_weight={float(cfg.get('feature_adapter_consistency_weight', 0.25)):g} "
                    f"consistency_epsilon={float(cfg.get('feature_adapter_consistency_epsilon', 1.0e-8)):g} "
                    "null_variance_detached=false"
                )
        if feature_discriminator is None:
            print("[feature-gan] disabled")
        else:
            raw_discriminator = (
                feature_discriminator.module
                if hasattr(feature_discriminator, "module")
                else feature_discriminator
            )
            discriminator_parameters = sum(
                parameter.numel() for parameter in raw_discriminator.parameters()
            )
            print(
                "[feature-gan] enabled "
                f"stages={','.join(raw_discriminator.stages)} "
                f"target_gradient_ratio={float(cfg.get('feature_gan_gradient_ratio', 0.1)):g} "
                f"warmup_steps={int(cfg.get('feature_gan_warmup_steps', 1000))} "
                f"calibration_freq={int(cfg.get('feature_gan_gradient_calibration_freq', 10))} "
                f"lr={float(cfg.get('feature_gan_lr', 2.0e-4)):g} "
                f"parameters={discriminator_parameters}"
            )

    generated_epoch_size = len(train_loader.dataset)
    generated_per_step = (
        batch_size * world_size * int(cfg.get("gen_per_label", 64))
    )
    adapter_freeze_epochs = float(
        cfg.get("feature_adapter_freeze_generated_epochs", 0.0) or 0.0
    )
    if adapter_freeze_epochs < 0.0 or not math.isfinite(adapter_freeze_epochs):
        raise ValueError(
            "feature_adapter_freeze_generated_epochs must be finite and non-negative"
        )
    if adapter_freeze_epochs > 0.0:
        adapter_freeze_step_resolved = _steps_for_generated_epochs(
            dataset_size=generated_epoch_size,
            generated_per_step=generated_per_step,
            epochs=adapter_freeze_epochs,
        )
        cfg["feature_adapter_freeze_step_resolved"] = adapter_freeze_step_resolved
        if is_main_process(rank):
            print(
                "[feature-adapter] late freeze at generated epoch "
                f"{adapter_freeze_epochs:g} (step {adapter_freeze_step_resolved})",
                flush=True,
            )
    historical_replay_enabled = bool(cfg.get("historical_gen_replay", False))
    historical_replay_ratio = float(
        cfg.get("historical_gen_replay_ratio", 0.0)
    )
    historical_replay_count = int(
        cfg.get("historical_gen_replay_count", 16)
    )
    historical_replay_bank_count = int(
        cfg.get("historical_gen_replay_bank_count", 0)
        or historical_replay_count
    )
    historical_replay_start_epochs = float(
        cfg.get("historical_gen_replay_start_generated_epochs", 10.0)
    )
    historical_replay_source = str(
        cfg.get("historical_gen_replay_source", "frozen_snapshot")
    ).lower().strip()
    historical_replay_bank: Optional[ArrayMemoryBank] = None
    historical_replay_start_step = 0
    historical_replay_path = (
        Path(workdir) / f"historical_gen_replay_rank{rank:02d}.npz"
    )
    if historical_replay_enabled:
        if str(cfg.get("drift_matching", "rev-drift")).lower().strip() != "rev-drift":
            raise ValueError("historical generated replay currently supports rev-drift only")
        if not math.isfinite(historical_replay_ratio) or not 0.0 < historical_replay_ratio <= 1.0:
            raise ValueError(
                "historical_gen_replay_ratio must be finite and in (0, 1] when enabled"
            )
        if historical_replay_count <= 0:
            raise ValueError("historical_gen_replay_count must be positive")
        if historical_replay_count > int(cfg.get("gen_per_label", 64)):
            raise ValueError(
                "historical_gen_replay_count cannot exceed gen_per_label"
            )
        if historical_replay_bank_count < historical_replay_count:
            raise ValueError(
                "historical_gen_replay_bank_count must be at least "
                "historical_gen_replay_count"
            )
        if historical_replay_bank_count > int(cfg.get("gen_per_label", 64)):
            raise ValueError(
                "historical_gen_replay_bank_count cannot exceed gen_per_label"
            )
        if historical_replay_source not in ("frozen_snapshot", "fresh_current"):
            raise ValueError(
                "historical_gen_replay_source must be frozen_snapshot or "
                f"fresh_current, got {historical_replay_source!r}"
            )
        if not math.isfinite(historical_replay_start_epochs) or historical_replay_start_epochs <= 0.0:
            raise ValueError(
                "historical_gen_replay_start_generated_epochs must be finite and positive"
            )
        storage_dtype_name = str(
            cfg.get("historical_gen_replay_storage_dtype", "float16")
        ).lower().strip()
        storage_dtypes = {"float16": np.float16, "float32": np.float32}
        if storage_dtype_name not in storage_dtypes:
            raise ValueError(
                "historical_gen_replay_storage_dtype must be float16 or float32"
            )
        historical_replay_start_step = _steps_for_generated_epochs(
            dataset_size=generated_epoch_size,
            generated_per_step=generated_per_step,
            epochs=historical_replay_start_epochs,
        )
        if historical_replay_source == "frozen_snapshot":
            historical_replay_bank = ArrayMemoryBank(
                num_classes=int(cfg.get("num_classes", 1000)),
                max_size=historical_replay_bank_count,
                dtype=storage_dtypes[storage_dtype_name],
            )
        if is_main_process(rank):
            print(
                "[historical-replay] enabled "
                f"source={historical_replay_source} "
                f"ratio={historical_replay_ratio:g} count={historical_replay_count} "
                f"bank_count={historical_replay_bank_count} "
                f"snapshot_epoch={historical_replay_start_epochs:g} "
                f"snapshot_step={historical_replay_start_step} "
                f"storage_dtype={storage_dtype_name}; "
                "current/replay generated mass is preserved",
                flush=True,
            )
    elif is_main_process(rank):
        print("[historical-replay] disabled", flush=True)
    total_generated_epochs = float(cfg.get("total_generated_epochs", 0.0))
    if total_generated_epochs > 0.0:
        cfg["total_steps"] = _steps_for_generated_epochs(
            dataset_size=generated_epoch_size,
            generated_per_step=generated_per_step,
            epochs=total_generated_epochs,
        )
    save_per_generated_epochs = float(
        cfg.get("save_per_generated_epochs", 0.0)
    )
    if save_per_generated_epochs < 0.0 or not math.isfinite(
        save_per_generated_epochs
    ):
        raise ValueError(
            "save_per_generated_epochs must be finite and >= 0, got "
            f"{save_per_generated_epochs}"
        )
    eval_per_generated_epochs = float(
        cfg.get("eval_per_generated_epochs", 0.0)
    )
    if eval_per_generated_epochs < 0.0 or not math.isfinite(
        eval_per_generated_epochs
    ):
        raise ValueError(
            "eval_per_generated_epochs must be finite and >= 0, got "
            f"{eval_per_generated_epochs}"
        )
    if is_main_process(rank):
        target_summary = (
            f" target_epochs={total_generated_epochs:g}"
            if total_generated_epochs > 0.0
            else ""
        )
        save_summary = (
            f" save_every_epochs={save_per_generated_epochs:g}"
            if save_per_generated_epochs > 0.0
            else ""
        )
        eval_summary = (
            f" eval_every_epochs={eval_per_generated_epochs:g}"
            if eval_per_generated_epochs > 0.0
            else ""
        )
        print(
            "[generated-epochs] "
            f"dataset_size={generated_epoch_size} "
            f"generated_per_step={generated_per_step}"
            f"{target_summary}{save_summary}{eval_summary} "
            f"total_steps={int(cfg.get('total_steps', 200000))}"
        )

    # --- Checkpoint / resume ---
    logger = Logger(workdir, cfg, rank)
    start_step = load_checkpoint(
        workdir,
        generator,
        ema,
        optimizer,
        device,
        mix_alpha_tracker=mix_alpha_tracker,
        feature_adapter=feature_adapter,
        feature_adapter_target=feature_adapter_target,
        feature_adapter_optimizer=feature_adapter_optimizer,
        feature_discriminator=feature_discriminator,
        feature_discriminator_optimizer=feature_discriminator_optimizer,
        adversarial_system=adversarial_system,
    )
    if is_main_process(rank):
        print(f"Resuming from step {start_step}")
    if historical_replay_bank is not None and start_step >= historical_replay_start_step:
        if not historical_replay_path.is_file():
            raise FileNotFoundError(
                "Cannot resume historical replay after its snapshot boundary; "
                f"missing {historical_replay_path}"
            )
        historical_replay_bank.load_npz(historical_replay_path)
        if not historical_replay_bank.is_ready(historical_replay_count):
            raise RuntimeError(
                f"Historical replay snapshot is incomplete: {historical_replay_path}"
            )
        if is_main_process(rank):
            print(
                f"[historical-replay] restored frozen snapshot from {historical_replay_path}",
                flush=True,
            )

    total_steps  = int(cfg.get("total_steps", 200000))
    save_per     = int(cfg.get("save_per_step", 2000))
    eval_per     = int(cfg.get("eval_per_step", 5000))
    eval_samples = int(cfg.get("eval_samples", 50000))
    keep_every   = int(cfg.get("keep_every", 50000))
    keep_last    = int(cfg.get("keep_last", 2))
    cfg_list     = list(cfg.get("cfg_list", [1.0, 1.4, 2.0]))

    # --- Fill memory bank if resuming ---
    initial_push = push_per_step
    if start_step > 0:
        initial_push = push_at_resume * push_per_step
        if is_main_process(rank):
            print(f"Filling memory bank at resume: {initial_push} samples")

    # Skip to the right epoch boundary without loading skipped batches.
    # Within-epoch skip would iterate (and load) thousands of batches just to discard them.
    _epoch_at_resume = start_step // max(len(train_loader), 1) if start_step > 0 else 0
    train_iter = infinite_sampler(train_loader, _epoch_at_resume * max(len(train_loader), 1))
    is_first_step = True
    bank_stream_phase = 0

    # --- Optional step-0 eval (baseline before any training) ---
    eval_at_start = bool(cfg.get("eval_at_start", False))
    if start_step == 0 and eval_at_start:
        if world_size > 1:
            dist.barrier()
        if is_main_process(rank):
            if eval_loader is None or eval_postprocess_fn is None:
                print("[eval] Step-0 eval skipped: eval_loader/eval_postprocess_fn is None.")
            else:
                print("[eval] Running step-0 baseline eval...")
                for bank in pos_banks:
                    if isinstance(bank, CompressedPixelMemoryBank):
                        bank.suspend_codec_workers()
                try:
                    # DDP reducers retain parameter/bucket references created
                    # at construction time. Moving a rank-0-only DDP module
                    # CPU -> CUDA invalidates those references and desynchronizes
                    # the next backward pass from the waiting rank(s). Keep DDP
                    # modules on their construction device; 48 GiB A6000 jobs
                    # have ample headroom for the temporary eval models.
                    if world_size == 1:
                        generator.to("cpu")
                    feature_extractor.to("cpu")
                    if feature_discriminator is not None and world_size == 1:
                        feature_discriminator.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    ema.shadow.to(device)
                    ema.shadow.eval()
                    # Keep an explicit generated-epoch axis beside the W&B
                    # step so baseline/10-epoch evaluations are unambiguous.
                    log_eval: Dict[str, float] = {"eval/generated_epochs": 0.0}
                    for eval_cfg_scale in cfg_list[:3]:
                        eval_stats = eval_fid_is(
                            ema.shadow, eval_postprocess_fn, eval_loader, device,
                            cfg_scale=eval_cfg_scale,
                            n_samples=eval_samples,
                            workdir=workdir,
                            step=0,
                            label=f"CFG{eval_cfg_scale}",
                        )
                        if eval_stats.get("fid") is not None:
                            fid = float(eval_stats["fid"])
                            log_eval[f"fid/cfg{eval_cfg_scale}"] = fid
                            print(f"[step 0] FID@CFG{eval_cfg_scale} = {fid:.4f}")
                        if eval_stats.get("is_mean") is not None:
                            is_mean = float(eval_stats["is_mean"])
                            log_eval[f"is/cfg{eval_cfg_scale}"] = is_mean
                            is_std = float(eval_stats.get("is_std", 0.0) or 0.0)
                            print(f"[step 0] IS@CFG{eval_cfg_scale} = {is_mean:.4f} +/- {is_std:.4f}")
                            if eval_stats.get("is_std") is not None:
                                log_eval[f"is_std/cfg{eval_cfg_scale}"] = float(eval_stats["is_std"])
                    if log_eval:
                        # Merge baseline metrics into the first training row;
                        # the step-0 train log below commits the combined row.
                        logger.log(log_eval, step=0, commit=False)
                except Exception as e:
                    print(f"[eval] Step-0 eval failed: {e}")
                    traceback.print_exc()
                finally:
                    # Mirror the step-N eval cleanup: gc.collect before reload so
                    # eval-time references (Inception V3 inside torchmetrics,
                    # generated/real image tensors) actually free GPU memory
                    # before train models are loaded back.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    ema.shadow.to(device)
                    feature_extractor.to(device)
                    if world_size == 1:
                        generator.to(device)
                    if feature_discriminator is not None and world_size == 1:
                        feature_discriminator.to(device)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                    for bank in pos_banks:
                        if isinstance(bank, CompressedPixelMemoryBank):
                            bank.resume_codec_workers()
        if world_size > 1:
            dist.barrier()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        generator.train()
    elif start_step == 0 and is_main_process(rank):
        print("[eval] Skipping step-0 eval (eval_at_start=false).")

    # Start progress and average-step timing only after the optional baseline
    # evaluation. Otherwise the first displayed iteration includes several
    # minutes of FID/IS work and falsely reports training as ~100x slower.
    pbar = (
        tqdm(range(start_step, total_steps), initial=start_step, total=total_steps)
        if is_main_process(rank)
        else range(start_step, total_steps)
    )
    start_time_all = time.time()

    adapter_requires_unique_reals = (
        feature_adapter is not None
        and str(cfg.get("feature_adapter_objective", "")).lower().strip()
        in {
            "real_real_multipos_infonce",
            "real_real_multipos_infonce_mdino",
            "gen_real_multipos_infonce",
            "raw_drift_snr",
            "raw_drift_snr_consistency",
            "dino_cf_drift_realreal_infonce",
        }
        and bool(cfg.get("feature_adapter_require_unique_reals", False))
    )
    adapter_unique_real_count = (
        int(cfg.get("feature_adapter_real_pool_per_class", 32))
        if str(cfg.get("feature_adapter_objective", "")).lower().strip()
        == "raw_drift_snr"
        else pos_per_sample
    )
    # Positive-bank counts are monotonic during a process lifetime. Keep this
    # latched once every rank has the objective's distinct-real requirement for
    # every class. On resume the latch resets with the empty in-memory banks.
    feature_adapter_bank_ready = not adapter_requires_unique_reals

    step_limit = int(cfg.get("train_max_step_exclusive", 0) or 0)
    for step in pbar:
        if step_limit > 0 and step >= step_limit:
            if is_main_process(rank):
                print(f"[train] stopping early: train_max_step_exclusive={step_limit}")
            break
        logger.set_step(step)
        iteration_t0 = time.time()

        # LR update
        lr = get_lr(step, warmup, base_lr, init_lr=warmup_init_lr)
        set_lr(optimizer, lr)

        # --- Push real images to memory bank ---
        push_goal = initial_push if is_first_step else push_per_step
        is_first_step = False

        n_pushed = 0
        last_labels_by_bank: List[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in range(banks_per_rank)
        ]
        while True:
            batch = next(train_iter)
            processed = preprocess_fn(batch)
            images = processed["images"]   # (B, C, H, W) or numpy
            labels_b = processed["labels"]

            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            else:
                images_np = np.array(images)
            if isinstance(labels_b, torch.Tensor):
                labels_np = labels_b.cpu().numpy()
            else:
                labels_np = np.array(labels_b)

            split_batches, bank_stream_phase = _split_bank_stream(
                images_np, labels_np, banks_per_rank, bank_stream_phase
            )
            last_labels_by_bank = [
                np.asarray(bank_labels, dtype=np.int64) for _, bank_labels in split_batches
            ]
            for bank_idx, (bank_images, bank_labels) in enumerate(split_batches):
                if bank_labels.size == 0:
                    continue
                pos_banks[bank_idx].add(bank_images, bank_labels)
                neg_banks[bank_idx].add(bank_images, np.zeros_like(bank_labels))
            n_pushed += images_np.shape[0]
            if n_pushed >= push_goal:
                break

        if adapter_requires_unique_reals and not feature_adapter_bank_ready:
            local_ready = all(
                bank.is_ready(adapter_unique_real_count) for bank in pos_banks
            )
            ready_tensor = torch.tensor(
                int(local_ready), device=device, dtype=torch.int32
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
            feature_adapter_bank_ready = bool(ready_tensor.item())
            if feature_adapter_bank_ready and is_main_process(rank):
                print(
                    "[feature-adapter] positive banks ready with "
                    f">={adapter_unique_real_count} distinct samples/class on every rank; "
                    "enabling adapter updates",
                    flush=True,
                )

        adapter_sampling_due = (
            feature_adapter is not None
            and feature_adapter_bank_ready
            and step >= int(cfg.get("feature_adapter_start_step", 0))
            and (
                int(cfg.get("feature_adapter_freeze_step_resolved", 0) or 0)
                <= 0
                or step
                < int(cfg.get("feature_adapter_freeze_step_resolved", 0) or 0)
            )
            and (
                step - int(cfg.get("feature_adapter_start_step", 0))
            )
            % int(cfg.get("feature_adapter_update_freq", 1))
            == 0
        )
        sample_distinct_adapter_reals = (
            adapter_sampling_due
            and str(cfg.get("feature_adapter_objective", "")).lower().strip()
            == "raw_drift_snr"
        )

        # --- Sample batch labels from the latest pushed batch, matching official JAX. ---
        labels_parts: List[np.ndarray] = []
        pos_parts: List[torch.Tensor] = []
        neg_parts: List[torch.Tensor] = []
        adapter_pos_parts: List[torch.Tensor] = []
        for bank_idx, bank_batch_size in enumerate(bank_batch_sizes):
            if bank_batch_size <= 0:
                continue
            labels_pool = last_labels_by_bank[bank_idx]
            if labels_pool.size == 0:
                continue
            rng_idx = _step_choice_indices(
                len(labels_pool),
                bank_batch_size,
                seed=int(cfg.get("seed", 42)),
                step=step,
                bank_idx=bank_idx,
            )
            labels_sel = labels_pool[rng_idx]
            labels_parts.append(labels_sel)
            pos_parts.append(
                pos_banks[bank_idx].sample(
                    labels_sel,
                    n_samples=pos_per_sample,
                    device=device,
                )
            )
            neg_parts.append(
                neg_banks[bank_idx].sample(
                    np.zeros(len(labels_sel), dtype=np.int64),
                    n_samples=neg_per_sample,
                    device=device,
                )
            )
            if sample_distinct_adapter_reals:
                adapter_pos_parts.append(
                    pos_banks[bank_idx].sample(
                        labels_sel,
                        n_samples=adapter_unique_real_count,
                        device=device,
                        rng=np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    int(cfg.get("feature_adapter_split_seed", 43)),
                                    int(rank),
                                    int(step) & 0xFFFFFFFF,
                                    (int(step) >> 32) & 0xFFFFFFFF,
                                    int(bank_idx),
                                    0x56325245,
                                ]
                            )
                        ),
                    )
                )

        if not labels_parts:
            if is_main_process(rank):
                print(f"[step {step}] Memory banks empty, skipping step...")
            continue

        # --- Sample from memory banks ---
        labels_sel = np.concatenate(labels_parts, axis=0)
        labels_t = torch.from_numpy(labels_sel).long().to(device)
        pos_smp = torch.cat(pos_parts, dim=0)
        neg_smp = torch.cat(neg_parts, dim=0)
        adapter_pos_smp = (
            torch.cat(adapter_pos_parts, dim=0)
            if adapter_pos_parts
            else None
        )
        historical_smp = None
        if historical_replay_bank is not None and step >= historical_replay_start_step:
            if not historical_replay_bank.is_ready(historical_replay_count):
                raise RuntimeError(
                    "Historical replay reached its activation step before all class "
                    "buffers were populated. Increase the snapshot warmup."
                )
            historical_smp = historical_replay_bank.sample(
                labels_sel,
                n_samples=historical_replay_count,
                device=device,
                rng=np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            int(cfg.get("seed", 42)),
                            int(rank),
                            int(step) & 0xFFFFFFFF,
                            (int(step) >> 32) & 0xFFFFFFFF,
                            0x48495354,
                        ]
                    )
                ),
            )
        fresh_historical_count = (
            historical_replay_count
            if historical_replay_enabled
            and historical_replay_source == "fresh_current"
            and step >= historical_replay_start_step
            else 0
        )

        # --- Forward + backward ---
        benchmark_profile = bool(cfg.get("profile_train_step", False))
        if benchmark_profile and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        generator.train()
        loss, metrics, step_extras = train_step(
            generator, feature_extractor, optimizer, labels_t,
            pos_smp, neg_smp, device, step, cfg,
            mix_alpha_tracker=mix_alpha_tracker,
            feature_adapter=feature_adapter,
            feature_adapter_target=feature_adapter_target,
            feature_adapter_optimizer=feature_adapter_optimizer,
            feature_adapter_bank_ready=feature_adapter_bank_ready,
            feature_adapter_real_samples=adapter_pos_smp,
            feature_discriminator=feature_discriminator,
            feature_discriminator_optimizer=feature_discriminator_optimizer,
            historical_samples=historical_smp,
            fresh_historical_count=fresh_historical_count,
            adversarial_system=adversarial_system,
        )
        ema.update(gen_raw)
        if historical_replay_bank is not None and step < historical_replay_start_step:
            local_label_count = int(labels_t.shape[0])
            generated = step_extras["gen_samples_detached"].reshape(
                local_label_count,
                int(cfg.get("gen_per_label", 64)),
                *step_extras["gen_samples_detached"].shape[1:],
            )
            # The final H candidates for each selected class replace that
            # class's snapshot, so the frozen bank represents the generator
            # immediately before the configured epoch boundary.
            snapshot_samples = generated[:, -historical_replay_bank_count:].reshape(
                local_label_count * historical_replay_bank_count,
                *generated.shape[2:],
            )
            snapshot_labels = labels_t[:, None].expand(
                -1, historical_replay_bank_count
            ).reshape(-1)
            historical_replay_bank.add(
                snapshot_samples.detach().to(device="cpu", dtype=torch.float32),
                snapshot_labels,
            )
            if step + 1 == historical_replay_start_step:
                if not historical_replay_bank.is_ready(historical_replay_bank_count):
                    missing_classes = np.flatnonzero(
                        historical_replay_bank.count < historical_replay_bank_count
                    )
                    raise RuntimeError(
                        "Historical replay snapshot warmup missed classes: "
                        f"{missing_classes[:20].tolist()}"
                    )
                historical_replay_bank.save_npz(historical_replay_path)
                if is_main_process(rank):
                    print(
                        "[historical-replay] froze epoch "
                        f"{historical_replay_start_epochs:g} snapshot at "
                        f"{historical_replay_path}",
                        flush=True,
                    )
        if benchmark_profile and device.type == "cuda":
            torch.cuda.synchronize(device)
            metrics["profile/peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
            metrics["profile/peak_reserved_gib"] = (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            )
        step_time = time.time() - t0

        # --- Logging ---
        if step % log_every == 0 and is_main_process(rank):
            compressed_pos_banks = [
                bank
                for bank in pos_banks
                if isinstance(bank, CompressedPixelMemoryBank)
            ]
            if compressed_pos_banks:
                compressed_bytes = sum(
                    bank.payload_bytes for bank in compressed_pos_banks
                )
                raw_bytes = sum(
                    bank.raw_equivalent_bytes for bank in compressed_pos_banks
                )
                metrics["bank/positive_payload_gib_rank0"] = (
                    compressed_bytes / (1024 ** 3)
                )
                metrics["bank/positive_raw_equivalent_gib_rank0"] = (
                    raw_bytes / (1024 ** 3)
                )
                metrics["bank/positive_compression_ratio_rank0"] = (
                    raw_bytes / max(1, compressed_bytes)
                )
                metrics["bank/positive_codec_add_seconds_rank0"] = sum(
                    bank.last_add_seconds for bank in compressed_pos_banks
                )
                metrics["bank/positive_codec_sample_seconds_rank0"] = sum(
                    bank.last_sample_seconds for bank in compressed_pos_banks
                )
            metrics["lr"]             = lr
            metrics["time/step"]      = step_time
            metrics["time/iteration"] = time.time() - iteration_t0
            metrics["time/per_step"]  = (time.time() - start_time_all) / (step - start_step + 1)
            metrics["kimg"]           = (step - start_step + 1) * batch_size * world_size / 1000.0
            metrics["generated_kimg"] = (step + 1) * generated_per_step / 1000.0
            metrics["generated_epochs"] = (
                (step + 1) * generated_per_step / generated_epoch_size
            )
            logger.log(metrics)

        # --- Checkpoint ---
        if save_per_generated_epochs > 0.0:
            save_due = _crosses_generated_epoch_interval(
                completed_steps=step + 1,
                generated_per_step=generated_per_step,
                dataset_size=generated_epoch_size,
                interval_epochs=save_per_generated_epochs,
            )
        else:
            save_due = (step + 1) % save_per == 0
        if save_due or (step + 1) == total_steps:
            if is_main_process(rank):
                save_checkpoint(
                    workdir,
                    step + 1,
                    generator,
                    ema,
                    optimizer,
                    cfg,
                    keep_last,
                    keep_every,
                    mix_alpha_tracker=mix_alpha_tracker,
                    feature_adapter=feature_adapter,
                    feature_adapter_target=feature_adapter_target,
                    feature_adapter_optimizer=feature_adapter_optimizer,
                    feature_discriminator=feature_discriminator,
                    feature_discriminator_optimizer=feature_discriminator_optimizer,
                    adversarial_system=adversarial_system,
                )

        # --- FID / IS evaluation ---
        if eval_per_generated_epochs > 0.0:
            eval_due = _crosses_generated_epoch_interval(
                completed_steps=step + 1,
                generated_per_step=generated_per_step,
                dataset_size=generated_epoch_size,
                interval_epochs=eval_per_generated_epochs,
            )
        else:
            eval_due = (step + 1) % eval_per == 0
        if eval_due or (step + 1) == total_steps:
            # Barrier: ensure all ranks finish the current training step before
            # rank 0 starts eval. Without this, other ranks proceed to the next
            # train_step (which triggers a DDP all_reduce) while rank 0 is still
            # in eval → deadlock.
            if world_size > 1:
                dist.barrier()
            if is_main_process(rank):
                if eval_loader is None or eval_postprocess_fn is None:
                    print("[eval] eval_loader/eval_postprocess_fn is None. Skipping eval.")
                else:
                    # Keep DDP modules on the devices where their reducers were
                    # constructed. Rank-0-only CPU round trips corrupt reducer
                    # bucket state and deadlock the following backward pass.
                    print(f"[eval] Preparing rank-0 eval at step {step+1}...")
                    for bank in pos_banks:
                        if isinstance(bank, CompressedPixelMemoryBank):
                            bank.suspend_codec_workers()
                    try:
                        if world_size == 1:
                            generator.to("cpu")
                        feature_extractor.to("cpu")
                        if feature_discriminator is not None and world_size == 1:
                            feature_discriminator.to("cpu")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        ema.shadow.to(device)
                        ema.shadow.eval()
                        log_eval: Dict[str, float] = {
                            "eval/generated_epochs": (
                                (step + 1) * generated_per_step
                                / generated_epoch_size
                            )
                        }
                        for eval_cfg_scale in cfg_list[:3]:  # limit to first 3 during training
                            eval_stats = eval_fid_is(
                                ema.shadow, eval_postprocess_fn, eval_loader, device,
                                cfg_scale=eval_cfg_scale,
                                n_samples=eval_samples,
                                workdir=workdir,
                                step=step + 1,
                                label=f"CFG{eval_cfg_scale}",
                            )
                            if eval_stats.get("fid") is not None:
                                fid = float(eval_stats["fid"])
                                log_eval[f"fid/cfg{eval_cfg_scale}"] = fid
                                print(f"[step {step+1}] FID@CFG{eval_cfg_scale} = {fid:.4f}")
                            if eval_stats.get("is_mean") is not None:
                                is_mean = float(eval_stats["is_mean"])
                                log_eval[f"is/cfg{eval_cfg_scale}"] = is_mean
                                is_std = float(eval_stats.get("is_std", 0.0) or 0.0)
                                print(f"[step {step+1}] IS@CFG{eval_cfg_scale} = {is_mean:.4f} +/- {is_std:.4f}")
                                if eval_stats.get("is_std") is not None:
                                    log_eval[f"is_std/cfg{eval_cfg_scale}"] = float(eval_stats["is_std"])
                        if log_eval:
                            logger.log(log_eval, step=step + 1)
                    except Exception as e:
                        print(f"[eval] Eval failed at step {step+1}: {e}")
                        traceback.print_exc()
                    finally:
                        # Reload training models back to GPU.
                        print(f"[eval] Reloading training models back to GPU...")
                        # Drop any eval-time references (Inception V3 inside torchmetrics,
                        # generated/real image tensors) before they collide with the
                        # reloaded train models. empty_cache alone won't free memory
                        # held by live Python references — we need gc.collect first.
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for bank in pos_banks:
                            if isinstance(bank, CompressedPixelMemoryBank):
                                bank.resume_codec_workers()
                        ema.shadow.to(device)
                        feature_extractor.to(device)
                        if world_size == 1:
                            generator.to(device)
                        if feature_discriminator is not None and world_size == 1:
                            feature_discriminator.to(device)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            # Barrier: make other ranks wait here until rank 0 finishes eval,
            # so the next train_step's DDP all_reduce can proceed safely.
            if world_size > 1:
                dist.barrier()
            generator.train()

    if is_main_process(rank):
        logger.finish()
        print("Generator training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet generator training (PyTorch)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--workdir", type=str, default="runs/gen", help="Working directory for checkpoints and logs.")
    parser.add_argument("--mae_checkpoint", type=str, default="", help="Override MAE checkpoint path.")
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="",
        help="Override frozen encoder: mae, dino_resnet50, or moco_v2_resnet50.",
    )
    parser.add_argument(
        "--feature_checkpoint",
        type=str,
        default="",
        help="Override the selected frozen encoder checkpoint path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Override the training seed; negative keeps the YAML setting.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If >0, stop before this global step index (debug / smoke). 0 = run full total_steps from config.",
    )
    parser.add_argument(
        "--skip_eval_at_start",
        action="store_true",
        help="Skip only the step-0 evaluation (useful for launch smoke tests).",
    )
    parser.add_argument(
        "--smoke_adapter_updates",
        action="store_true",
        help=(
            "Debug-only: allow adapter updates before each class has unique "
            "positive-bank coverage and disable W&B. Requires --max_steps."
        ),
    )
    parser.add_argument(
        "--throughput_opt_level",
        type=int,
        default=-1,
        choices=(0, 1, 2, 3, 4),
        help=(
            "Benchmark/runtime stack: 0=baseline, 1=diagnostic gating, "
            "2=+fused MAE stats, 3=+DDP/optimizer/EMA fast paths, "
            "4=+packed reverse f-norm reductions/runtime fast paths."
        ),
    )
    parser.add_argument(
        "--benchmark_throughput",
        action="store_true",
        help="Enable phase/peak profiling and log every benchmark step.",
    )
    parser.add_argument(
        "--stochastic_feature_stage_count",
        type=int,
        default=0,
        choices=(0, 1, 2, 3, 4),
        help=(
            "Override stochastic MAE stage-loss sampling: 1-4 enables the "
            "sampler with that many stages; 0 keeps the YAML settings."
        ),
    )
    parser.add_argument(
        "--rev_drift_top_p",
        type=float,
        default=None,
        help="Override reverse-drift nucleus mass in (0, 1]; 1 disables truncation.",
    )
    parser.add_argument(
        "--fwd_drift_top_p",
        type=float,
        default=None,
        help="Override forward-drift nucleus mass in (0, 1]; 1 disables truncation.",
    )
    parser.add_argument(
        "--drift_top_p_min_keep",
        type=int,
        default=None,
        help="Minimum targets retained in each positive/negative force group.",
    )
    parser.add_argument(
        "--drift_top_k_pos",
        type=int,
        default=None,
        help=(
            "Positive top-k support: targets per generated row for reverse "
            "drift, generated rows per target column for forward drift; 0 disables."
        ),
    )
    parser.add_argument(
        "--drift_top_k_neg",
        type=int,
        default=None,
        help=(
            "Generated/negative top-k support: targets per generated row for "
            "reverse drift, generated rows per target column for forward drift; "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--drift_top_k_groups",
        type=str,
        default="",
        help=(
            "Comma-separated feature-loss groups on which top-k is active "
            "(global,norm_x,stage1..stage4), or 'all'. Empty keeps YAML."
        ),
    )
    parser.add_argument(
        "--layer_temperature_profile",
        type=str,
        default="",
        help=(
            "Select a named entry from train.layer_temperature_profiles. "
            "Empty keeps train.layer_temperature_profile from YAML."
        ),
    )
    parser.add_argument(
        "--feature_loss_profile",
        type=str,
        default="",
        help=(
            "Select a named entry from train.feature_loss_profiles. "
            "Empty keeps train.feature_loss_profile from YAML."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Override per-rank conditioning-label batch size; 0 keeps YAML.",
    )
    parser.add_argument(
        "--pos_per_sample",
        type=int,
        default=0,
        help="Override positive targets per conditioning label; 0 keeps YAML.",
    )
    parser.add_argument(
        "--neg_per_sample",
        type=int,
        default=0,
        help="Override negative targets per conditioning label; 0 keeps YAML.",
    )
    parser.add_argument(
        "--total_generated_epochs",
        type=float,
        default=0.0,
        help=(
            "If >0, derive total optimizer steps from generated candidates "
            "and the training-set size."
        ),
    )
    parser.add_argument(
        "--save_per_generated_epochs",
        type=float,
        default=0.0,
        help=(
            "If >0, checkpoint whenever generated candidates cross this "
            "many training-set equivalents."
        ),
    )
    parser.add_argument(
        "--eval_per_step",
        type=int,
        default=0,
        help="Override periodic evaluation interval; 0 keeps YAML.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    wandb_name_override = os.environ.get("IDRIFT_WANDB_NAME", "").strip()
    if wandb_name_override:
        cfg["name"] = wandb_name_override
    if int(args.throughput_opt_level) >= 0:
        cfg["throughput_opt_level"] = int(args.throughput_opt_level)
    if int(args.seed) >= 0:
        cfg["seed"] = int(args.seed)
    if args.benchmark_throughput:
        cfg["profile_train_step"] = True
        cfg["log_every_k"] = 1
        # Keep detailed per-feature diagnostics on warm-up step 0 only so
        # optimized steady steps measure the intended gating behavior.
        cfg["diagnostics_every_k"] = 10
    if int(args.stochastic_feature_stage_count) > 0:
        cfg["stochastic_feature_stage_loss"] = True
        cfg["stochastic_feature_stage_count"] = int(
            args.stochastic_feature_stage_count
        )
    if args.rev_drift_top_p is not None:
        cfg["rev_drift_top_p"] = float(args.rev_drift_top_p)
    if args.fwd_drift_top_p is not None:
        cfg["fwd_drift_top_p"] = float(args.fwd_drift_top_p)
    if args.drift_top_p_min_keep is not None:
        cfg["drift_top_p_min_keep"] = int(args.drift_top_p_min_keep)
    if args.drift_top_k_pos is not None:
        cfg["drift_top_k_pos"] = int(args.drift_top_k_pos)
    if args.drift_top_k_neg is not None:
        cfg["drift_top_k_neg"] = int(args.drift_top_k_neg)
    if args.drift_top_k_groups:
        cfg["drift_top_k_groups"] = str(args.drift_top_k_groups)
    if args.layer_temperature_profile:
        cfg["layer_temperature_profile"] = str(
            args.layer_temperature_profile
        )
    if args.feature_loss_profile:
        cfg["feature_loss_profile"] = str(args.feature_loss_profile)
    if args.batch_size > 0:
        cfg["batch_size"] = int(args.batch_size)
    if args.pos_per_sample > 0:
        cfg["pos_per_sample"] = int(args.pos_per_sample)
    if args.neg_per_sample > 0:
        cfg["neg_per_sample"] = int(args.neg_per_sample)
    if args.total_generated_epochs > 0.0:
        cfg["total_generated_epochs"] = float(args.total_generated_epochs)
    if args.save_per_generated_epochs > 0.0:
        cfg["save_per_generated_epochs"] = float(
            args.save_per_generated_epochs
        )
    if args.eval_per_step > 0:
        cfg["eval_per_step"] = int(args.eval_per_step)
    if int(getattr(args, "max_steps", 0) or 0) > 0:
        cfg["train_max_step_exclusive"] = int(args.max_steps)
    if args.skip_eval_at_start:
        cfg["eval_at_start"] = False
    if args.smoke_adapter_updates:
        if int(getattr(args, "max_steps", 0) or 0) <= 0:
            parser.error("--smoke_adapter_updates requires --max_steps")
        cfg["feature_adapter_require_unique_reals"] = False
        cfg["eval_at_start"] = False

    for top_p_key in ("rev_drift_top_p", "fwd_drift_top_p"):
        top_p_value = float(cfg.get(top_p_key, 1.0))
        if not 0.0 < top_p_value <= 1.0:
            parser.error(f"{top_p_key} must be in (0, 1], got {top_p_value}")
    min_keep_value = int(cfg.get("drift_top_p_min_keep", 1))
    if min_keep_value < 1:
        parser.error(
            f"drift_top_p_min_keep must be >= 1, got {min_keep_value}"
        )
    top_k_pos_value = int(cfg.get("drift_top_k_pos", 0))
    top_k_neg_value = int(cfg.get("drift_top_k_neg", 0))
    if top_k_pos_value < 0 or top_k_neg_value < 0:
        parser.error(
            "drift_top_k_pos and drift_top_k_neg must both be >= 0, got "
            f"{top_k_pos_value} and {top_k_neg_value}"
        )
    if (top_k_pos_value > 0 or top_k_neg_value > 0) and any(
        float(cfg.get(key, 1.0)) < 1.0
        for key in ("rev_drift_top_p", "fwd_drift_top_p")
    ):
        parser.error("top-p and top-k drift truncation cannot be enabled together")
    try:
        _resolve_layer_temperature_multipliers(cfg)
        _resolve_feature_loss_group_weights(cfg)
        _resolve_drift_top_k_groups(cfg)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    if not cfg.get("imagenet_path"):
        cfg["imagenet_path"] = os.environ.get("IMAGENET_PATH", "")
    if not cfg.get("cache_path"):
        cfg["cache_path"] = os.environ.get("IMAGENET_CACHE_PATH", "")
    if args.mae_checkpoint:
        cfg["mae_checkpoint"] = args.mae_checkpoint
        cfg["feature_checkpoint"] = args.mae_checkpoint
    if args.feature_extractor:
        cfg["feature_extractor"] = args.feature_extractor
    if args.feature_checkpoint:
        cfg["feature_checkpoint"] = args.feature_checkpoint

    try:
        resolve_feature_extractor_name(cfg)
    except ValueError as exc:
        parser.error(str(exc))

    rank, world_size, device = setup_distributed()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    train_gen(cfg, args.workdir, rank, world_size, device)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
