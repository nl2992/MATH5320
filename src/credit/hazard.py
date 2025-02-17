"""
hazard.py
Reduced-form default modelling (formula-sheet §8).

Survival probability:
    s(t) = P(τ > t) = exp( − ∫₀ᵗ λ(u) du )

Default density:
    p(t) = −d/dt s(t) = λ(t) s(t)

Constant-hazard special case:
    s(t) = e^{−λ t}
    P(τ ≤ t)       = 1 − e^{−λ t}
    P(t₁ < τ ≤ t₂) = e^{−λ t₁} − e^{−λ t₂}

Risky zero-coupon bond + spread (§8):
    V(T) = e^{−rT} [ 1 − LGD · (1 − s(T)) ]
    S(T) = −(1/T) log( 1 − LGD · (1 − s(T)) )
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ── Constant hazard ───────────────────────────────────────────────────────────

def survival(t: float, lam: float) -> float:
    """s(t) = e^{−λ t} (constant hazard)."""
    if t < 0:
        raise ValueError(f"t must be non-negative (got {t}).")
    if lam < 0:
        raise ValueError(f"lambda must be non-negative (got {lam}).")
    return math.exp(-lam * t)


def default_density(t: float, lam: float) -> float:
    """p(t) = λ(t) s(t) under constant hazard."""
    return lam * survival(t, lam)


def interval_default_prob(t1: float, t2: float, lam: float) -> float:
    """P(t₁ < τ ≤ t₂) = e^{−λ t₁} − e^{−λ t₂}."""
    if t2 < t1:
        raise ValueError(f"t2 must be >= t1 (got {t1}, {t2}).")
    return math.exp(-lam * t1) - math.exp(-lam * t2)


def cumulative_default_prob(t: float, lam: float) -> float:
    """P(τ ≤ t) = 1 − e^{−λ t}."""
    return 1.0 - survival(t, lam)


# ── Piecewise-constant hazard ─────────────────────────────────────────────────

def survival_piecewise(
    t: float,
    grid: Sequence[float],
    hazards: Sequence[float],
) -> float:
    """
    Survival under a piecewise-constant hazard curve.

    Parameters
    ----------
    t : float
        Query time.
    grid : sequence[float]
        Strictly increasing knot times starting at 0 (e.g. [0, 1, 2, 5, 10]).
    hazards : sequence[float]
        Hazard rates active on each interval — ``len(hazards) == len(grid) − 1``.

    Returns
    -------
    s(t) = exp( − ∫₀ᵗ λ(u) du )
    """
    grid = np.asarray(grid, dtype=float)
    hazards = np.asarray(hazards, dtype=float)
    if grid[0] != 0.0:
        raise ValueError("grid must start at 0.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing.")
    if len(hazards) != len(grid) - 1:
        raise ValueError(
            f"len(hazards) must be len(grid) − 1 (got {len(hazards)} vs {len(grid) - 1})."
        )
    if np.any(hazards < 0):
        raise ValueError("hazards must be non-negative.")
    if t < 0:
        raise ValueError(f"t must be non-negative (got {t}).")

    # Integrate λ piecewise up to t.
    integral = 0.0
    for i in range(len(hazards)):
        lo, hi = grid[i], grid[i + 1]
        if t <= lo:
            break
        seg_hi = min(t, hi)
        integral += hazards[i] * (seg_hi - lo)
        if t <= hi:
            break
    else:
        # Past the last knot — extrapolate with the final hazard.
        if t > grid[-1]:
            integral += hazards[-1] * (t - grid[-1])

    return math.exp(-integral)


# ── Risky zero-coupon + spread ────────────────────────────────────────────────

def risky_zcb_price(r: float, T: float, LGD: float, s_T: float) -> float:
    """
    Price of a risky zero-coupon bond (face 1) with recovery paid at maturity.

        V(T) = e^{−rT} [ 1 − LGD · (1 − s(T)) ]
    """
    if not (0.0 <= LGD <= 1.0):
        raise ValueError(f"LGD must be in [0, 1] (got {LGD}).")
    if not (0.0 <= s_T <= 1.0):
        raise ValueError(f"s(T) must be in [0, 1] (got {s_T}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    return math.exp(-r * T) * (1.0 - LGD * (1.0 - s_T))


def credit_spread(T: float, LGD: float, s_T: float) -> float:
    """
    Implied credit spread for a risky zero-coupon bond:

        S(T) = −(1/T) log( 1 − LGD · (1 − s(T)) )
    """
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    inside = 1.0 - LGD * (1.0 - s_T)
    if inside <= 0.0:
        raise ValueError(
            "LGD · (1 − s(T)) must be < 1 for the spread to be finite."
        )
    return -math.log(inside) / T
