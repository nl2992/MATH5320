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
    """Constant-hazard approximation to the CDS par spread (formula-sheet §10).

    C ≈ (1 − R) λ = LGD · λ

    This is the landmark approximation from §10: at λ = 3%, R = 40% the
    result is ≈ 180 bps.

    Args:
        lam (float): Constant hazard rate λ (must be ≥ 0).
        R (float): Recovery rate ∈ [0, 1].

    Returns:
        float: Approximate CDS par spread as a decimal
            (e.g. 0.018 = 180 bps).

    Raises:
        ValueError: If ``lam < 0`` or ``R`` outside [0, 1].
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
    """Compute CDS par spread via numerical integration with piecewise-constant hazard.

    Implements the §10 formula numerically using a midpoint-rule quadrature.

    Args:
        payment_times (Sequence[float]): Premium payment dates t₁ < t₂ < … < tₙ
            in years; the final entry tₙ = T is the CDS maturity.  All values
            must be strictly positive and increasing.
        hazards (Sequence[float]): Piecewise-constant hazard rates, one per
            payment interval.  ``len(hazards)`` must equal
            ``len(payment_times)``; hazards[i] applies to (t_{i-1}, t_i]
            with t_0 = 0.
        r (float): Flat continuously-compounded risk-free discount rate.
        R (float): Recovery rate ∈ [0, 1].
        accrual (bool): If True, include the accrued-premium-at-default term
            in the premium-leg denominator (mid-period approximation).
        n_sub (int): Midpoint-rule sub-steps per premium interval for
            numerically integrating the protection leg and accrual term.

    Returns:
        float: CDS par spread as a decimal (e.g. 0.018 = 180 bps).

    Raises:
        ValueError: If ``payment_times`` are not strictly positive/increasing,
            ``len(hazards) != len(payment_times)``, ``R`` outside [0, 1], or
            the premium-leg denominator is non-positive.
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


def cds_par_spread_constant_full_closed_form(
    T: float,
    freq: float,
    r: float,
    lam: float,
    R: float,
    accrual: bool = True,
) -> float:
    """Exact CDS par spread under constant hazard with discrete premium payments.

    Closed-form expressions for each leg:

    *Protection leg* (exact integral):
        (1−R) · lam / (r+lam) · (1 − exp(−(r+lam)·T))

    *Premium leg* (exact sum at each t_i = i/freq):
        dt · Σ_{i=1}^{n} exp(−(r+lam)·t_i)

    *Accrual term* (exact per-period integral):
        Σ_i  lam · exp(−q·t_{i-1}) · [(1−exp(−q·dt))/q² − dt·exp(−q·dt)/q]

    Note: the simple approximation :func:`cds_par_spread_constant_hazard` gives
    ≈ 180 bps; this exact formula gives ≈ 184.55 bps for λ=3%, R=40%, T=5,
    freq=1. The two are not intended to match.

    Args:
        T (float): CDS maturity in years (must be > 0).
        freq (float): Premium payments per year, e.g. 1 = annual, 4 = quarterly.
        r (float): Flat continuously-compounded risk-free rate.
        lam (float): Constant hazard rate λ (must be ≥ 0).
        R (float): Recovery rate ∈ [0, 1].
        accrual (bool): Include the accrued-premium-at-default term.

    Returns:
        float: CDS par spread as a decimal (e.g. 0.018 = 180 bps).

    Raises:
        ValueError: If ``lam < 0``, ``R`` outside [0, 1], ``T ≤ 0``,
            ``freq ≤ 0``, or the denominator is non-positive.
    """
    if lam < 0:
        raise ValueError(f"lambda must be non-negative (got {lam}).")
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    if freq <= 0:
        raise ValueError(f"freq must be positive (got {freq}).")

    n = int(round(T * freq))
    dt = 1.0 / freq
    q = r + lam

    # Protection leg
    if q == 0.0:
        protection = (1.0 - R) * lam * T
    else:
        protection = (1.0 - R) * lam / q * (1.0 - math.exp(-q * T))

    # Premium leg
    premium = 0.0
    for i in range(1, n + 1):
        t_i = i * dt
        premium += dt * math.exp(-q * t_i)

    # Accrual term
    accrual_val = 0.0
    if accrual:
        for i in range(1, n + 1):
            t_prev = (i - 1) * dt
            t_i = i * dt
            seg_dt = t_i - t_prev
            if q == 0.0:
                # integral of lam*(u - t_prev)*du from t_prev to t_i = lam * seg_dt^2/2
                accrual_val += lam * seg_dt ** 2 / 2.0
            else:
                exp_q_prev = math.exp(-q * t_prev)
                exp_q_dt = math.exp(-q * seg_dt)
                accrual_val += lam * exp_q_prev * (
                    (1.0 - exp_q_dt) / (q ** 2) - seg_dt * exp_q_dt / q
                )

    denominator = premium + accrual_val
    if denominator <= 0:
        raise ValueError("CDS premium-leg denominator is non-positive.")
    return protection / denominator


def cds_spread_curve(
    tenors: Sequence[float],
    lam: float,
    r: float,
    R: float,
    premium_freq: float = 1.0,
    accrual: bool = True,
) -> list[tuple[float, float]]:
    """Build a CDS par-spread curve under a flat hazard rate.

    Evaluates :func:`cds_par_spread` at each requested tenor using a uniform
    payment schedule with ``premium_freq`` payments per year.

    Args:
        tenors (Sequence[float]): CDS maturities in years at which to
            evaluate the par spread (e.g. ``[1, 2, 3, 5, 7, 10]``).
        lam (float): Flat constant hazard rate λ (must be ≥ 0).
        r (float): Flat continuously-compounded risk-free rate.
        R (float): Recovery rate ∈ [0, 1].
        premium_freq (float): Premium payments per year (default 1 = annual).
        accrual (bool): Include accrued-premium-at-default term.

    Returns:
        list[tuple[float, float]]: List of ``(tenor, spread)`` pairs where
            spread is in decimal (e.g. 0.018 = 180 bps).
    """
    out: list[tuple[float, float]] = []
    for T in tenors:
        n = max(1, int(round(T * premium_freq)))
        payment_times = list(np.linspace(1.0 / premium_freq, T, n))
        hazards = [lam] * len(payment_times)
        spread = cds_par_spread(payment_times, hazards, r=r, R=R, accrual=accrual)
        out.append((float(T), spread))
    return out
