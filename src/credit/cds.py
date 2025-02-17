"""
cds.py
CDS par spread pricing (formula-sheet §10).

Continuous-time par spread:
    D(t) = e^{-r t}
    C(T) = [ ∫₀ᵀ (1 − R) D(u) p(u) du ]
           / [ Σ_i a(t_i) D(t_i) s(t_i) + ∫₀ᵀ a*(u) D(u) p(u) du ]

Constant-hazard approximation (sheet §10):
    C ≈ (1 − R) λ = LGD · λ

Sanity-check landmark (§14):
    λ = 3%, R = 40%   →   approx spread ≈ 180 bps.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.credit.hazard import survival_piecewise


def cds_par_spread_constant_hazard(lam: float, R: float) -> float:
    """
    Constant-hazard approximation to the CDS par spread.

        C ≈ (1 − R) λ

    Returned as a decimal (e.g. 0.018 = 180 bps).
    """
    if lam < 0:
        raise ValueError(f"lambda must be non-negative (got {lam}).")
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")
    return (1.0 - R) * lam


def cds_par_spread(
    payment_times: Sequence[float],
    hazards: Sequence[float],
    r: float,
    R: float,
    accrual: bool = True,
    n_sub: int = 20,
) -> float:
    """
    Discrete approximation to the §10 par-spread formula with piecewise-constant hazard.

    Parameters
    ----------
    payment_times : sequence[float]
        Premium payment dates t_1 < t_2 < … < t_n (years). Final t_n = T.
    hazards : sequence[float]
        Piecewise-constant hazard levels — ``len(hazards) == len(payment_times)``.
        hazards[i] applies to the interval (t_{i-1}, t_i] with t_0 = 0.
    r : float
        Flat continuously-compounded discount rate.
    R : float
        Recovery rate.
    accrual : bool
        If True, include the accrued-premium-at-default term in the denominator
        (mid-period approximation a*(u) ≈ Δt_i / 2).
    n_sub : int
        Sub-steps per premium interval used to integrate the protection leg
        and the accrual term numerically (simple midpoint rule).

    Returns
    -------
    Par spread as a decimal (e.g. 0.018 = 180 bps).
    """
    payment_times = np.asarray(payment_times, dtype=float)
    hazards = np.asarray(hazards, dtype=float)
    if payment_times[0] <= 0 or np.any(np.diff(payment_times) <= 0):
        raise ValueError("payment_times must be strictly positive and increasing.")
    if len(hazards) != len(payment_times):
        raise ValueError(
            f"len(hazards) must match len(payment_times) "
            f"(got {len(hazards)} vs {len(payment_times)})."
        )
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")

    grid = np.concatenate(([0.0], payment_times))

    def s(t: float) -> float:
        return survival_piecewise(t, grid, hazards)

    # Numerator — protection leg: (1−R) ∫₀ᵀ D(u) p(u) du, p(u) = -s'(u).
    # Premium-leg denominator: Σ a(t_i) D(t_i) s(t_i)
    #                          + ∫₀ᵀ a*(u) D(u) p(u) du   (if accrual).
    numerator = 0.0
    premium_leg = 0.0
    accrual_leg = 0.0

    for i in range(len(payment_times)):
        t_prev = grid[i]
        t_i = grid[i + 1]
        dt = t_i - t_prev

        # Premium payment at t_i.
        premium_leg += dt * math.exp(-r * t_i) * s(t_i)

        # Numerical integrals over (t_{i-1}, t_i]: midpoint rule with n_sub steps.
        h = dt / n_sub
        for k in range(n_sub):
            u_lo = t_prev + k * h
            u_mid = u_lo + 0.5 * h
            u_hi = u_lo + h
            # p(u) ≈ (s(u_lo) - s(u_hi)) / h gives the marginal-density over the sub-interval.
            # Integral over the sub-interval ≈ D(u_mid) * (s(u_lo) - s(u_hi)).
            D = math.exp(-r * u_mid)
            mass = s(u_lo) - s(u_hi)
            numerator += (1.0 - R) * D * mass
            if accrual:
                accrual_leg += (u_mid - t_prev) * D * mass

    denominator = premium_leg + accrual_leg
    if denominator <= 0:
        raise ValueError("CDS premium-leg denominator is non-positive.")
    return numerator / denominator


def cds_spread_curve(
    tenors: Sequence[float],
    lam: float,
    r: float,
    R: float,
    premium_freq: float = 1.0,
    accrual: bool = True,
) -> list[tuple[float, float]]:
    """
    Build a par-spread curve under a flat hazard ``lam``.

    Each point uses quarterly/annual premium payments based on ``premium_freq``
    (payments per year).

    Returns a list of ``(tenor, spread)`` pairs, spread as decimal.
    """
    out: list[tuple[float, float]] = []
    for T in tenors:
        n = max(1, int(round(T * premium_freq)))
        payment_times = list(np.linspace(1.0 / premium_freq, T, n))
        hazards = [lam] * len(payment_times)
        spread = cds_par_spread(payment_times, hazards, r=r, R=R, accrual=accrual)
        out.append((float(T), spread))
    return out
