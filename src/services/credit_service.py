"""
credit_service.py
Orchestration for credit-risk calculations (formula-sheet §8–§11).

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
    """
    Build a per-horizon table under a constant hazard ``lam``.

    Returns
    -------
    dict
        ``{"LGD": float, "approx_cds": float, "rows": list[dict]}`` where
        each row has ``{t, survival, cum_default, density, risky_zcb, spread}``.
        ``approx_cds`` is the §10 landmark ``(1-R)·λ``.
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
    """
    Per-bucket marginal-default table using the constant-hazard formula.
    ``edges`` are knot times starting at 0.
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
    """
    Full Merton snapshot under **both** measures in a single call.

    Returns
    -------
    dict
        ``{"Q": {d1,d2,PD}, "P": {d1,d2,PD}, "E0": float, "D0": float,
           "V0": float, "B": float, "T": float}``
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
    """Thin wrapper over :func:`merton_implied_B` for service-layer consumers."""
    return merton_implied_B(V0, target_survival, r, sigma, T)


# ── CDS ───────────────────────────────────────────────────────────────────────

def cds_summary(
    lam: float,
    R: float,
    tenors: Sequence[float],
    r: float = 0.03,
    premium_freq: float = 1.0,
) -> dict:
    """
    CDS summary: constant-hazard approx + full-formula par-spread curve.
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
    """Par spread for a bespoke schedule + piecewise-constant hazard curve."""
    return cds_par_spread(list(payment_times), list(hazards), r=r, R=R, accrual=accrual)


# ── CVA ───────────────────────────────────────────────────────────────────────

def cva_summary(
    exposure_profile: Sequence[float],
    marginal_default_probs: Sequence[float],
    R: float,
    V0: float | None = None,
) -> dict:
    """
    CVA from a discrete exposure + marginal-default profile.

    Parameters
    ----------
    exposure_profile : sequence[float]
        Expected-positive-exposure at each grid point.
    marginal_default_probs : sequence[float]
        Marginal default probability on each interval (same length).
    R : float
    V0 : float, optional
        Current portfolio value; if provided the result includes CVA as %.
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
    """
    Thin wrapper: expected-positive-exposure profile from simulated path values.
    """
    return epe_profile_from_mc(V_paths, V0)
