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
    """Survival probability under a constant hazard rate.

    s(t) = e^{−λ t}

    Args:
        t (float): Time horizon in years (must be ≥ 0).
        lam (float): Constant hazard rate λ (must be ≥ 0).

    Returns:
        float: Survival probability P(τ > t) ∈ [0, 1].

    Raises:
        ValueError: If ``t < 0`` or ``lam < 0``.
    """
    if t < 0:
        raise ValueError(f"t must be non-negative (got {t}).")
    if lam < 0:
        raise ValueError(f"lambda must be non-negative (got {lam}).")
    return math.exp(-lam * t)


def default_density(t: float, lam: float) -> float:
    """Instantaneous default density under constant hazard.

    p(t) = λ · s(t) = λ · e^{−λ t}

    Args:
        t (float): Time in years (must be ≥ 0).
        lam (float): Constant hazard rate λ (must be ≥ 0).

    Returns:
        float: Default probability density at time t.
    """
    return lam * survival(t, lam)


def interval_default_prob(t1: float, t2: float, lam: float) -> float:
    """Probability of defaulting within an interval under constant hazard.

    P(t₁ < τ ≤ t₂) = e^{−λ t₁} − e^{−λ t₂}

    Args:
        t1 (float): Start of interval in years (must be ≤ t2).
        t2 (float): End of interval in years.
        lam (float): Constant hazard rate λ (must be ≥ 0).

    Returns:
        float: Marginal default probability over (t1, t2].

    Raises:
        ValueError: If ``t2 < t1`` or ``lam < 0``.
    """
    if t2 < t1:
        raise ValueError(f"t2 must be >= t1 (got {t1}, {t2}).")
    if lam < 0:
        raise ValueError(f"lambda must be non-negative (got {lam}).")
    return math.exp(-lam * t1) - math.exp(-lam * t2)


def cumulative_default_prob(t: float, lam: float) -> float:
    """Cumulative default probability from time 0 to t under constant hazard.

    P(τ ≤ t) = 1 − e^{−λ t}

    Args:
        t (float): Time horizon in years (must be ≥ 0).
        lam (float): Constant hazard rate λ (must be ≥ 0).

    Returns:
        float: Cumulative default probability ∈ [0, 1].
    """
    return 1.0 - survival(t, lam)


# ── Piecewise-constant hazard ─────────────────────────────────────────────────

def survival_piecewise(
    t: float,
    grid: Sequence[float],
    hazards: Sequence[float],
) -> float:
    """Survival probability under a piecewise-constant hazard curve.

    s(t) = exp( − ∫₀ᵗ λ(u) du )

    Args:
        t (float): Query time in years (must be ≥ 0).
        grid (Sequence[float]): Knot times starting at 0, strictly increasing
            (e.g. ``[0, 1, 2, 5, 10]``). Must start at 0.
        hazards (Sequence[float]): Hazard rates active on each sub-interval;
            ``len(hazards)`` must equal ``len(grid) − 1``.  All must be ≥ 0.
            If ``t`` exceeds ``grid[-1]``, the final hazard is extrapolated.

    Returns:
        float: Survival probability s(t) ∈ (0, 1].

    Raises:
        ValueError: If grid does not start at 0, is not strictly increasing,
            ``len(hazards) != len(grid) - 1``, any hazard is negative, or
            ``t < 0``.
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
        # Past the last knot - extrapolate with the final hazard.
        if t > grid[-1]:
            integral += hazards[-1] * (t - grid[-1])

    return math.exp(-integral)


# ── Risky zero-coupon + spread ────────────────────────────────────────────────

def risky_zcb_price(r: float, T: float, LGD: float, s_T: float) -> float:
    """Price a risky zero-coupon bond with face value 1 (formula-sheet §8).

    V(T) = e^{−rT} [ 1 − LGD · (1 − s(T)) ]

    Args:
        r (float): Continuously-compounded risk-free rate.
        T (float): Maturity in years (must be > 0).
        LGD (float): Loss-given-default as a fraction of face value,
            in [0, 1].
        s_T (float): Survival probability at maturity, s(T) ∈ [0, 1].

    Returns:
        float: Present value of the risky bond (≤ risk-free discount).

    Raises:
        ValueError: If ``LGD`` or ``s_T`` are outside [0, 1] or ``T ≤ 0``.
    """
    if not (0.0 <= LGD <= 1.0):
        raise ValueError(f"LGD must be in [0, 1] (got {LGD}).")
    if not (0.0 <= s_T <= 1.0):
        raise ValueError(f"s(T) must be in [0, 1] (got {s_T}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    return math.exp(-r * T) * (1.0 - LGD * (1.0 - s_T))


def credit_spread(T: float, LGD: float, s_T: float) -> float:
    """Compute the implied credit spread for a risky zero-coupon bond (§8).

    S(T) = −(1/T) log( 1 − LGD · (1 − s(T)) )

    Args:
        T (float): Maturity in years (must be > 0).
        LGD (float): Loss-given-default fraction ∈ [0, 1].
        s_T (float): Survival probability at T ∈ [0, 1].

    Returns:
        float: Continuously-compounded credit spread in decimal
            (e.g. 0.018 = 180 bps).

    Raises:
        ValueError: If ``T ≤ 0`` or LGD · (1 − s_T) ≥ 1 (spread undefined).
    """
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    inside = 1.0 - LGD * (1.0 - s_T)
    if inside <= 0.0:
        raise ValueError(
            "LGD · (1 − s(T)) must be < 1 for the spread to be finite."
        )
    return -math.log(inside) / T


def hazard_at_piecewise(t: float, grid, hazards) -> float:
    """Return the hazard rate λ(t) active at time t under piecewise-constant hazard.

    Args:
        t (float): Query time in years (must be ≥ 0).
        grid: Knot times starting at 0 (see :func:`survival_piecewise`).
        hazards: Hazard rates per sub-interval (len = len(grid) − 1).

    Returns:
        float: Hazard rate active at t; extrapolates with the last rate
            if t > grid[-1].

    Raises:
        ValueError: Same constraints as :func:`survival_piecewise`.
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
    # Return last hazard for extrapolation past the final knot
    if t > grid[-1]:
        return float(hazards[-1])
    for i in range(len(hazards)):
        if t <= grid[i + 1]:
            return float(hazards[i])
    return float(hazards[-1])


def cumhazard_piecewise(t: float, grid, hazards) -> float:
    """Cumulative hazard Λ(t) = ∫₀ᵗ λ(u) du under piecewise-constant hazard.

    Args:
        t (float): Query time in years (must be ≥ 0).
        grid: Knot times starting at 0 (see :func:`survival_piecewise`).
        hazards: Hazard rates per sub-interval.

    Returns:
        float: Cumulative hazard Λ(t) = −log(s(t)) ≥ 0.
    """
    s = survival_piecewise(t, grid, hazards)
    return -math.log(s)


def density_piecewise(t: float, grid, hazards) -> float:
    """Instantaneous default density p(t) = λ(t) · s(t) under piecewise-constant hazard.

    Args:
        t (float): Query time in years (must be ≥ 0).
        grid: Knot times starting at 0 (see :func:`survival_piecewise`).
        hazards: Hazard rates per sub-interval.

    Returns:
        float: Default probability density at time t ≥ 0.
    """
    lam = hazard_at_piecewise(t, grid, hazards)
    s = survival_piecewise(t, grid, hazards)
    return lam * s


def interval_default_prob_piecewise(t1: float, t2: float, grid, hazards) -> float:
    """Marginal default probability over (t1, t2] under piecewise-constant hazard.

    P(t₁ < τ ≤ t₂) = s(t₁) − s(t₂)

    Args:
        t1 (float): Start of interval in years.
        t2 (float): End of interval in years.
        grid: Knot times starting at 0 (see :func:`survival_piecewise`).
        hazards: Hazard rates per sub-interval.

    Returns:
        float: Marginal default probability ∈ [0, 1].
    """
    s1 = survival_piecewise(t1, grid, hazards)
    s2 = survival_piecewise(t2, grid, hazards)
    return s1 - s2
