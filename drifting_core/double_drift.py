"""Detached two-step corrections in generated-sample (latent) coordinates."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.distributed as dist


def validate_double_drift_coefficients(c0: float, c1: float) -> tuple[float, float]:
    c0, c1 = float(c0), float(c1)
    if not all(math.isfinite(c) and c >= 0 for c in (c0, c1)):
        raise ValueError("double drift coefficients must be finite and non-negative")
    if c0 + c1 <= 0:
        raise ValueError("at least one double drift coefficient must be positive")
    return c0, c1


def sample_double_drift_loss(
    samples: torch.Tensor,
    first_loss: torch.Tensor,
    loss_at_samples: Callable[[torch.Tensor], tuple[torch.Tensor, dict]],
    *,
    c0: float = 0.75,
    c1: float = 0.25,
    step_rms: float = 0.1,
    global_stats: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Two sample-space steps, with frozen feature-based drift at each stage.

    g_k = dL_drift(x_k)/dx_k, a = step_rms / RMS(g_0),
    x_1 = stopgrad(x_0 - c0*a*g_0), x_2 = x_1 - c1*a*g_1.

    Regress x_0 to detached x_2 using sum squared error / (2*a). This
    calibration gives dL/dx_0 = c0*g_0+c1*g_1: it preserves the baseline's
    gradient units and avoids adding a batch-size-dependent loss multiplier.
    The first scale a is reused for the second step. No Hessians or backward
    through the generator are used while constructing the target.
    """
    c0, c1 = validate_double_drift_coefficients(c0, c1)
    if not math.isfinite(step_rms) or step_rms <= 0:
        raise ValueError("double_drift_sample_step_rms must be finite and positive")
    if c1 == 0:
        return c0 * first_loss, {"double_drift/sample_field_evaluations": 1.0}

    # Stops at the generated tensor: consume the encoder graph, preserving
    # the generator graph for the final target-regression backward.
    grad0 = torch.autograd.grad(first_loss, samples, create_graph=False)[0].detach().float()
    stats = torch.stack((grad0.double().square().sum(), grad0.new_tensor(grad0.numel(), dtype=torch.float64)))
    if global_stats and dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
    grad_rms = (stats[0] / stats[1]).sqrt().float()
    step_scale = (step_rms / grad_rms.clamp_min(1e-12)).detach()
    probe = (samples.detach().float() - c0 * step_scale * grad0).requires_grad_(True)
    second_loss, second_info = loss_at_samples(probe)
    grad1 = torch.autograd.grad(second_loss, probe, create_graph=False)[0].detach().float()
    displacement = (-step_scale * (c0 * grad0 + c1 * grad1)).detach()

    # Algebraically samples - stopgrad(samples + displacement), written this
    # way to avoid cancellation of a small displacement against a large x.
    residual = (samples.float() - samples.detach().float()) - displacement
    loss = residual.square().sum() / (2 * step_scale)
    info = {f"double_drift/sample_second/{k}": float(v) for k, v in second_info.items()}
    info.update({
        "double_drift/sample_field_evaluations": 2.0,
        "double_drift/sample_first_loss": float(first_loss.detach().item()),
        "double_drift/sample_second_loss": float(second_loss.detach().item()),
        "double_drift/sample_gradient_rms": float(grad_rms.item()),
        "double_drift/sample_step_scale": float(step_scale.item()),
        "double_drift/sample_probe_rms": float((c0 * step_scale * grad0).square().mean().sqrt().item()),
        "double_drift/sample_displacement_rms": float(displacement.square().mean().sqrt().item()),
        "double_drift/sample_gradient_cosine": float(torch.nn.functional.cosine_similarity(
            grad0.flatten(), grad1.flatten(), dim=0
        ).item()),
    })
    return loss, info
