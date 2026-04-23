"""
credit_service.py
Orchestration for credit-risk calculations (formula-sheet §8 - §11).

Streamlit panels call only this service; it never touches the UI layer.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.credit.cds import (
    cds_par_spread,
    cds_par_spread_constant_hazard,
    cds_spread_curve,
)
from src.credit.cva import cva_discrete, epe_profile_from_mc
from src.credit.hazard import (
    credit_spread,
    cumulative_default_prob,
    default_density,
    interval_default_prob,
    risky_zcb_price,
    survival,
)
from src.credit.merton import (
    merton_d1_d2,
    merton_debt,
    merton_equity,
    merton_implied_B,
    merton_pd,
)


# ── Reduced-form ──────────────────────────────────────────────────────────────

def reduced_form_summary(
    lam: float,
    horizons: Sequence[float],
    R: float,
    r: float = 0.0,
) -> dict:
    """Build a per-horizon reduced-form credit table under constant hazard.

    Args:
        lam (float): Constant hazard rate λ (must be ≥ 0).
        horizons (Sequence[float]): Time horizons in years at which to
            evaluate survival, default density, and spread (e.g. [1,2,3,5]).
        R (float): Recovery rate ∈ [0, 1].
        r (float): Risk-free rate for risky ZCB pricing (default 0.0).

    Returns:
        dict: Summary with keys:
            - ``"LGD"`` (float): 1 − R.
            - ``"approx_cds"`` (float): §10 landmark (1−R)·λ.
            - ``"rows"`` (list[dict]): One dict per horizon with keys
              ``t``, ``survival``, ``cum_default``, ``density``,
              ``risky_zcb``, ``spread``.  ``risky_zcb`` and ``spread``
              are NaN when t = 0.
    """
    LGD = 1.0 - R
    rows: list[dict] = []
    for t in horizons:
        s_t = survival(float(t), lam)
        rows.append(
            {
                "t": float(t),
                "survival": s_t,
                "cum_default": cumulative_default_prob(float(t), lam),
                "density": default_density(float(t), lam),
                "risky_zcb": risky_zcb_price(r, float(t), LGD, s_t) if t > 0 else float("nan"),
                "spread": credit_spread(float(t), LGD, s_t) if t > 0 else float("nan"),
            }
        )
    return {
        "LGD": LGD,
        "approx_cds": cds_par_spread_constant_hazard(lam, R),
        "rows": rows,
    }


def interval_default_table(
    edges: Sequence[float],
    lam: float,
) -> list[dict]:
    """Build a per-bucket marginal-default table using the constant-hazard formula.

    Args:
        edges (Sequence[float]): Knot times starting at 0, e.g. [0,1,2,3,5].
            Consecutive pairs define each interval.
        lam (float): Constant hazard rate λ (must be ≥ 0).

    Returns:
        list[dict]: One dict per interval with keys ``"t1"``, ``"t2"``,
            and ``"marginal_default"`` = P(t1 < τ ≤ t2).
    """
    edges = list(edges)
    out: list[dict] = []
    for i in range(len(edges) - 1):
        t1, t2 = float(edges[i]), float(edges[i + 1])
        out.append(
            {
                "t1": t1,
                "t2": t2,
                "marginal_default": interval_default_prob(t1, t2, lam),
            }
        )
    return out


# ── Merton ────────────────────────────────────────────────────────────────────

def merton_summary(
    V0: float,
    B: float,
    r: float,
    mu: float,
    sigma: float,
    T: float,
) -> dict:
    """Full Merton model snapshot under both Q and P measures in one call.

    Computes d₁, d₂, and default probability under both risk-neutral (Q,
    ν=r) and real-world (P, ν=μ) measures, plus equity and debt values.

    Args:
        V0 (float): Current firm asset value (must be > 0).
        B (float): Debt face value / default barrier (must be > 0).
        r (float): Risk-free rate (continuously compounded).
        mu (float): Real-world drift of firm assets.
        sigma (float): Firm asset volatility (must be > 0).
        T (float): Time to maturity in years (must be > 0).

    Returns:
        dict: Merton snapshot with keys:
            - ``"V0"``, ``"B"``, ``"T"`` (float): Input parameters echoed.
            - ``"Q"`` (dict): Q-measure results - ``{d1, d2, PD}``.
            - ``"P"`` (dict): P-measure results - ``{d1, d2, PD}``.
            - ``"E0"`` (float): Equity value (call on assets under Q).
            - ``"D0"`` (float): Risky debt value (V0 − E0).
    """
    d1_Q, d2_Q = merton_d1_d2(V0, B, r, sigma, T)
    d1_P, d2_P = merton_d1_d2(V0, B, mu, sigma, T)
    E0 = merton_equity(V0, B, r, sigma, T)
    D0 = merton_debt(V0, B, r, sigma, T)
    return {
        "V0": V0,
        "B": B,
        "T": T,
        "Q": {
            "d1": d1_Q,
            "d2": d2_Q,
            "PD": merton_pd(V0, B, r, sigma, T),
        },
        "P": {
            "d1": d1_P,
            "d2": d2_P,
            "PD": merton_pd(V0, B, mu, sigma, T),
        },
        "E0": E0,
        "D0": D0,
    }


def merton_implied_B_for_survival(
    V0: float,
    target_survival: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """Return the default barrier B* implied by a target Q-survival probability.

    Thin service-layer wrapper around :func:`src.credit.merton.merton_implied_B`.

    Args:
        V0 (float): Current firm asset value (must be > 0).
        target_survival (float): Desired Q-survival probability ∈ (0, 1).
        r (float): Risk-free rate (continuously compounded).
        sigma (float): Firm asset volatility (must be > 0).
        T (float): Time to maturity in years (must be > 0).

    Returns:
        float: Implied barrier B* such that Q-survival(T) = target_survival.

    Raises:
        ValueError: If any positivity constraint is violated or
            ``target_survival`` is outside (0, 1).
    """
    return merton_implied_B(V0, target_survival, r, sigma, T)


# ── CDS ───────────────────────────────────────────────────────────────────────

def cds_summary(
    lam: float,
    R: float,
    tenors: Sequence[float],
    r: float = 0.03,
    premium_freq: float = 1.0,
) -> dict:
    """Build a CDS summary: constant-hazard approximation and par-spread curve.

    Args:
        lam (float): Constant hazard rate λ (must be ≥ 0).
        R (float): Recovery rate ∈ [0, 1].
        tenors (Sequence[float]): CDS maturities in years for the spread
            curve (e.g. [1, 2, 3, 5, 7, 10]).
        r (float): Flat risk-free rate for premium-leg discounting
            (default 0.03).
        premium_freq (float): Premium payments per year (default 1 = annual).

    Returns:
        dict: CDS summary with keys:
            - ``"approx_spread"`` (float): §10 landmark (1−R)·λ.
            - ``"curve"`` (list[tuple[float, float]]): Par-spread term
              structure as ``[(tenor, spread)]`` pairs (spread in decimal).
    """
    approx = cds_par_spread_constant_hazard(lam, R)
    curve = cds_spread_curve(list(tenors), lam=lam, r=r, R=R, premium_freq=premium_freq)
    return {
        "approx_spread": approx,
        "curve": curve,  # list[(tenor, spread)]
    }


def cds_spread_for_schedule(
    payment_times: Sequence[float],
    hazards: Sequence[float],
    r: float,
    R: float,
    accrual: bool = True,
) -> float:
    """Compute the CDS par spread for a bespoke payment schedule and hazard curve.

    Thin wrapper around :func:`src.credit.cds.cds_par_spread` for service-layer
    consumers who supply a custom payment schedule.

    Args:
        payment_times (Sequence[float]): Premium payment dates t₁ < … < tₙ
            in years.
        hazards (Sequence[float]): Piecewise-constant hazard rates, one per
            payment interval.  Length must match ``payment_times``.
        r (float): Flat continuously-compounded risk-free rate.
        R (float): Recovery rate ∈ [0, 1].
        accrual (bool): Include accrued-premium-at-default term (default True).

    Returns:
        float: CDS par spread as a decimal (e.g. 0.018 = 180 bps).
    """
    return cds_par_spread(list(payment_times), list(hazards), r=r, R=R, accrual=accrual)


# ── CVA ───────────────────────────────────────────────────────────────────────

def cva_summary(
    exposure_profile: Sequence[float],
    marginal_default_probs: Sequence[float],
    R: float,
    V0: float | None = None,
) -> dict:
    """Compute CVA from a discrete EPE and marginal-default profile.

    Args:
        exposure_profile (Sequence[float]): Expected-positive exposure at
            each grid point (must be non-negative).
        marginal_default_probs (Sequence[float]): Marginal default probability
            on each interval (same length; must sum to ≤ 1).
        R (float): Recovery rate ∈ [0, 1].
        V0 (float | None): Current portfolio mark-to-market value; if
            provided, also returns CVA as a percentage of V0.

    Returns:
        dict: CVA summary with keys:
            - ``"cva"`` (float): CVA in dollars.
            - ``"R"`` (float): Recovery rate echoed.
            - ``"cva_pct"`` (float): cva / V0 (only present when V0 > 0).
    """
    cva = cva_discrete(exposure_profile, marginal_default_probs, R)
    out = {"cva": cva, "R": R}
    if V0 is not None and V0 > 0:
        out["cva_pct"] = cva / V0
    return out


def epe_from_portfolio_mc(
    V_paths: np.ndarray,
    V0: float,
) -> np.ndarray:
    """Build an EPE profile from Monte Carlo simulated portfolio values.

    Thin service-layer wrapper around :func:`src.credit.cva.epe_profile_from_mc`.

    Args:
        V_paths (np.ndarray): Simulated future portfolio values - shape
            ``(n_paths,)`` for a single horizon or ``(n_paths, n_horizons)``
            for a term structure.
        V0 (float): Current portfolio value used as the exposure reference.

    Returns:
        np.ndarray: EPE at each horizon: E[max(V − V0, 0)], shape
            ``(n_horizons,)`` or length-1 for a 1-D input.
    """
    return epe_profile_from_mc(V_paths, V0)
