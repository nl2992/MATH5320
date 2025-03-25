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
    """
    Discrete CVA.

        CVA = (1 − R) Σ_i E_i p_i

    Parameters
    ----------
    exposures : sequence[float]
        Positive exposure at each grid point. Must be non-negative; typically
        ``max(0, mark-to-market)`` at each future time.
    marginal_default_probs : sequence[float]
        Per-interval marginal default probability. Must be non-negative and
        consistent with some survival curve (sum ≤ 1).
    R : float
        Recovery rate in [0, 1].

    Returns
    -------
    CVA in the same units as ``exposures``.
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
    """
    Discrete approximation to the §11 risky coupon-bond price.

    Parameters
    ----------
    coupon : float
        Annual coupon rate (e.g. 0.05 for 5%).
    freq : int
        Coupon payments per year (e.g. 2 for semi-annual).
    times : sequence[float]
        Coupon payment dates in years, strictly increasing; the final entry is
        the bond's maturity.
    r : float
        Flat continuously-compounded discount rate.
    survival_of_t : sequence[float]
        Survival probability at each time in ``times``. Same length.
    R : float
        Recovery rate (paid on notional at default, approximated midpoint on
        each coupon interval).
    notional : float
        Face amount.

    Returns
    -------
    Present value of the risky bond.
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

    # Recovery integral — midpoint approximation over each coupon interval.
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
    """
    Build an expected-positive-exposure profile from simulated portfolio values.

    Parameters
    ----------
    V_paths : np.ndarray
        Array of simulated future portfolio values at one or more horizons —
        shape ``(n_paths,)`` for a single horizon, or ``(n_paths, n_horizons)``
        for a time profile.
    V0 : float
        Current portfolio value.

    Returns
    -------
    np.ndarray
        EPE at each horizon: ``E[max(V_T − V0, 0)]``, length ``n_horizons``.
        For a 1-D input, returns a length-1 array.
    """
    arr = np.asarray(V_paths, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    exposure = np.maximum(arr - V0, 0.0)
    return exposure.mean(axis=0)


def positive_exposure(values: np.ndarray) -> np.ndarray:
    """max(V, 0) element-wise."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def epe(exposure_paths: np.ndarray) -> np.ndarray:
    """
    Expected positive exposure.
    exposure_paths shape: (n_scenarios, n_times) or (n_scenarios,).
    Returns mean(max(exposure, 0)) along axis 0 → shape (n_times,) or scalar.
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
    """
    Discounted discrete CVA.

        CVA = (1 − R) Σ_i D_i · E_i · p_i

    Parameters
    ----------
    exposures : EPE at each time bucket (non-negative).
    marginal_default_probs : per-interval marginal PD.
    discount_factors : risk-free discount factor D(t_i) = exp(−r·t_i).
    R : recovery rate in [0, 1].
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
    """
    Closed-form CVA for constant exposure K under constant hazard lam.

    r = 0:
        CVA = (1−R) · K · (1 − exp(−lam·T))
    r > 0:
        CVA = (1−R) · K · lam/(r+lam) · (1 − exp(−(r+lam)·T))
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
