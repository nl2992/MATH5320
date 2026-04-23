"""
cva.py
Counterparty valuation adjustment (formula-sheet §11).

    CVA = (1 − R) ∫₀ᵀ S(t) p(t) dt
        ≈ (1 − R) Σ_i S̄(t_i) p̄(t_i)

where S(t) is the (positive) exposure profile and p(t) the marginal default
density.

Risky coupon-bond price (sheet §11):
    V = Σ_i (C/f) D(t_i) s(t_i) + N D(t_n) s(t_n) + ∫₀^{t_n} N · R · D(t) p(t) dt
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def cva_discrete(
    exposures: Sequence[float],
    marginal_default_probs: Sequence[float],
    R: float,
) -> float:
    """Discrete counterparty valuation adjustment (formula-sheet §11).

    CVA = (1 − R) Σ_i E_i · p_i

    where E_i is the expected-positive exposure at time bucket i and p_i is
    the marginal default probability for that interval.

    Args:
        exposures (Sequence[float]): Expected-positive exposure at each grid
            point.  Typically E_i = max(MtM_i, 0).  Must be non-negative;
            length must match ``marginal_default_probs``.
        marginal_default_probs (Sequence[float]): Per-interval marginal
            default probability P(t_{i-1} < τ ≤ t_i).  Must be non-negative
            and sum to ≤ 1.
        R (float): Recovery rate ∈ [0, 1].

    Returns:
        float: CVA in the same currency units as ``exposures``.

    Raises:
        ValueError: If array lengths differ, any exposure is negative, the
            default probs are invalid, or ``R`` outside [0, 1].
    """
    exp_arr = np.asarray(exposures, dtype=float)
    pd_arr = np.asarray(marginal_default_probs, dtype=float)
    if len(exp_arr) != len(pd_arr):
        raise ValueError(
            f"len(exposures) must match len(marginal_default_probs) "
            f"(got {len(exp_arr)} vs {len(pd_arr)})."
        )
    if np.any(exp_arr < 0):
        raise ValueError("exposures must be non-negative.")
    if np.any(pd_arr < 0) or pd_arr.sum() > 1.0 + 1e-9:
        raise ValueError(
            "marginal_default_probs must be non-negative and sum to ≤ 1."
        )
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")

    return float((1.0 - R) * (exp_arr * pd_arr).sum())


def risky_bond_price(
    coupon: float,
    freq: int,
    times: Sequence[float],
    r: float,
    survival_of_t: Sequence[float],
    R: float,
    notional: float = 100.0,
) -> float:
    """Discrete approximation to the §11 risky coupon-bond price.

    V = Σ_i (C/f) D(t_i) s(t_i) + N D(t_n) s(t_n) + recovery integral

    where the recovery integral is approximated by a midpoint sum over each
    coupon interval.

    Args:
        coupon (float): Annual coupon rate as a decimal (e.g. 0.05 for 5%).
        freq (int): Coupon payments per year (e.g. 2 = semi-annual).
            Must be > 0.
        times (Sequence[float]): Coupon payment dates in years, strictly
            increasing; the final entry is the bond maturity.
        r (float): Flat continuously-compounded risk-free discount rate.
        survival_of_t (Sequence[float]): Survival probability at each entry
            in ``times``.  Must have the same length and all values in [0, 1].
        R (float): Recovery rate on notional, paid at default mid-interval.
            Must be in [0, 1].
        notional (float): Bond face value (default 100.0).

    Returns:
        float: Present value of the risky coupon bond.

    Raises:
        ValueError: If ``times`` and ``survival_of_t`` differ in length,
            ``times`` is not strictly increasing, survival values are outside
            [0, 1], ``R`` outside [0, 1], or ``freq ≤ 0``.
    """
    t = np.asarray(times, dtype=float)
    s = np.asarray(survival_of_t, dtype=float)
    if len(t) != len(s):
        raise ValueError("times and survival_of_t must have the same length.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing.")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("survival_of_t must be in [0, 1].")
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")
    if freq <= 0:
        raise ValueError(f"freq must be positive (got {freq}).")

    # Coupon payments.
    c = notional * coupon / freq
    pv = 0.0
    for ti, si in zip(t, s):
        pv += c * math.exp(-r * ti) * si

    # Notional at maturity.
    pv += notional * math.exp(-r * t[-1]) * s[-1]

    # Recovery integral - midpoint approximation over each coupon interval.
    prev_t, prev_s = 0.0, 1.0
    for ti, si in zip(t, s):
        mid = 0.5 * (prev_t + ti)
        D = math.exp(-r * mid)
        marginal_pd = max(prev_s - si, 0.0)
        pv += notional * R * D * marginal_pd
        prev_t, prev_s = ti, si

    return pv


# ── Helpers: build a toy EPE (expected positive exposure) profile ─────────────

def epe_profile_from_mc(
    V_paths: np.ndarray,
    V0: float,
) -> np.ndarray:
    """Build an expected-positive-exposure (EPE) profile from simulated portfolio values.

    EPE_i = E[max(V_{T_i} − V0, 0)]

    Args:
        V_paths (np.ndarray): Simulated future portfolio values.  Shape
            ``(n_paths,)`` for a single horizon, or ``(n_paths, n_horizons)``
            for a term structure of EPE.
        V0 (float): Current portfolio mark-to-market value used as the
            "at-the-money" reference.

    Returns:
        np.ndarray: EPE at each horizon - shape ``(n_horizons,)`` or a
            length-1 array for 1-D input.
    """
    arr = np.asarray(V_paths, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    exposure = np.maximum(arr - V0, 0.0)
    return exposure.mean(axis=0)


def positive_exposure(values: np.ndarray) -> np.ndarray:
    """Clip portfolio values to their positive part element-wise.

    Args:
        values (np.ndarray): Array of mark-to-market values.

    Returns:
        np.ndarray: max(V, 0) for each element; same shape as ``values``.
    """
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def epe(exposure_paths: np.ndarray) -> np.ndarray:
    """Compute expected positive exposure from a matrix of scenario exposures.

    Args:
        exposure_paths (np.ndarray): Scenario exposures.  Shape
            ``(n_scenarios, n_times)`` for a term structure, or
            ``(n_scenarios,)`` for a single horizon.

    Returns:
        np.ndarray | float: mean(max(exposure, 0)) along the scenario axis.
            Shape ``(n_times,)`` for 2-D input, or a scalar float for 1-D.
    """
    arr = np.asarray(exposure_paths, dtype=float)
    if arr.ndim == 1:
        return float(np.mean(np.maximum(arr, 0.0)))
    return np.mean(np.maximum(arr, 0.0), axis=0)


def cva_discounted(
    exposures: Sequence[float],
    marginal_default_probs: Sequence[float],
    discount_factors: Sequence[float],
    R: float,
) -> float:
    """Discounted discrete CVA (§11 with risk-free discounting).

    CVA = (1 − R) Σ_i D_i · E_i · p_i

    Args:
        exposures (Sequence[float]): EPE at each time bucket (must be ≥ 0).
        marginal_default_probs (Sequence[float]): Per-interval marginal
            default probability (must be ≥ 0, sum ≤ 1).
        discount_factors (Sequence[float]): Risk-free discount factor
            D(t_i) = exp(−r·t_i) ∈ (0, 1].  Same length as ``exposures``.
        R (float): Recovery rate ∈ [0, 1].

    Returns:
        float: Discounted CVA in the same currency as ``exposures``.

    Raises:
        ValueError: If array lengths differ, any exposure is negative, default
            probs sum > 1, discount factors are outside (0, 1], or ``R``
            outside [0, 1].
    """
    exp_arr = np.asarray(exposures, dtype=float)
    pd_arr  = np.asarray(marginal_default_probs, dtype=float)
    df_arr  = np.asarray(discount_factors, dtype=float)
    if not (len(exp_arr) == len(pd_arr) == len(df_arr)):
        raise ValueError("exposures, marginal_default_probs, discount_factors must have the same length.")
    if np.any(exp_arr < 0):
        raise ValueError("exposures must be non-negative.")
    if np.any(pd_arr < 0) or pd_arr.sum() > 1.0 + 1e-9:
        raise ValueError("marginal_default_probs must be non-negative and sum to ≤ 1.")
    if np.any(df_arr <= 0) or np.any(df_arr > 1.0):
        raise ValueError("discount_factors must be in (0, 1].")
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")
    return float((1.0 - R) * (df_arr * exp_arr * pd_arr).sum())


def cva_continuous_constant_exposure(
    K: float,
    lam: float,
    T: float,
    R: float,
    r: float = 0.0,
) -> float:
    """Closed-form CVA for constant exposure K under constant hazard rate.

    r = 0:
        CVA = (1−R) · K · (1 − exp(−lam·T))
    r > 0:
        CVA = (1−R) · K · lam / (r+lam) · (1 − exp(−(r+lam)·T))

    Args:
        K (float): Constant positive exposure (must be ≥ 0).
        lam (float): Constant hazard rate λ (must be ≥ 0).
        T (float): CVA horizon in years (must be > 0).
        R (float): Recovery rate ∈ [0, 1].
        r (float): Continuously-compounded risk-free rate for discounting
            (default 0.0 = no discounting).

    Returns:
        float: CVA in the same currency units as ``K``.

    Raises:
        ValueError: If ``K < 0``, ``lam < 0``, ``T ≤ 0``, ``R`` outside
            [0, 1], or ``r < 0``.
    """
    if K < 0:
        raise ValueError(f"K must be non-negative (got {K}).")
    if lam < 0:
        raise ValueError(f"lam must be non-negative (got {lam}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    if not (0.0 <= R <= 1.0):
        raise ValueError(f"R must be in [0, 1] (got {R}).")
    if r < 0:
        raise ValueError(f"r must be non-negative (got {r}).")
    lgd = 1.0 - R
    if r == 0.0:
        return float(lgd * K * (1.0 - math.exp(-lam * T)))
    q = r + lam
    return float(lgd * K * lam / q * (1.0 - math.exp(-q * T)))
