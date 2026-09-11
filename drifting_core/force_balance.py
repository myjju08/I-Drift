"""Attraction/repulsion schedules, separate from Double Drift step mixing."""
from __future__ import annotations

import math


def balance_coefficients(delta: float, step: int = 0,
                         anneal_steps: int = 0) -> tuple[float, float]:
    """Return (1+delta, 1-delta), optionally decaying delta to zero."""
    delta = float(delta)
    if not math.isfinite(delta) or not -1.0 < delta < 1.0:
        raise ValueError("force balance delta must be finite and strictly between -1 and 1")
    if int(step) != step or step < 0:
        raise ValueError("step must be a non-negative integer")
    if int(anneal_steps) != anneal_steps or anneal_steps < 0:
        raise ValueError("anneal_steps must be a non-negative integer")
    if anneal_steps:
        delta *= max(0.0, 1.0 - step / anneal_steps)
    return 1.0 + delta, 1.0 - delta
