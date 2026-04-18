"""
lognormal.py
Exact long- and short-position VaR / ES under GBM / lognormal pricing.
Formula-sheet §4 (long) and §7 (short).

Notation follows the sheet:
    m_h = (μ − ½σ²) h
    s_h = σ √h
    z_p = N⁻¹(p)

All four functions return **positive dollar losses** (ES ≥ VaR ≥ 0 for
reasonable parameters).
"""
from __future__ import annotations

import math

from scipy.stats import norm


# ── Long position ──────────────────────────────────────────────────────────────

def var_long_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """
    Exact long-position VaR under GBM (§4).

        VaR_long(p, h) = V₀ [ 1 − exp( m_h + s_h · z_{1−p} ) ]

    where ``z_{1−p}`` is negative for p > 0.5 (tail quantile).
    """
    _validate(V0, sigma, h, p)
    m_h = (mu - 0.5 * sigma ** 2) * h
    s_h = sigma * math.sqrt(h)
    z = norm.ppf(1.0 - p)
    return V0 * (1.0 - math.exp(m_h + s_h * z))


def es_long_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """
    Exact long-position ES under GBM (§4).

        ES_long(p, h) = V₀ [ 1 − exp(m_h + ½ s_h²) · N(z_{1−p} − s_h) / (1 − p) ]
    """
    _validate(V0, sigma, h, p)
    m_h = (mu - 0.5 * sigma ** 2) * h
    s_h = sigma * math.sqrt(h)
    z = norm.ppf(1.0 - p)
    alpha = 1.0 - p
    return V0 * (1.0 - math.exp(m_h + 0.5 * s_h ** 2) * norm.cdf(z - s_h) / alpha)


# ── Short position ─────────────────────────────────────────────────────────────

def var_short_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """
    Exact short-position VaR under GBM (§7).

        VaR_short(p, h) = V₀ [ exp( μ h + z_p σ √h ) − 1 ]
    """
    _validate(V0, sigma, h, p)
    s_h = sigma * math.sqrt(h)
    z_p = norm.ppf(p)
    return V0 * (math.exp(mu * h + z_p * s_h) - 1.0)


def es_short_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """
    Exact short-position ES under GBM (§7).

        ES_short(p, h) = V₀ [ exp(μ h + ½σ²h) · N( σ√h − z_p ) / (1 − p) − 1 ]
    """
    _validate(V0, sigma, h, p)
    s_h = sigma * math.sqrt(h)
    z_p = norm.ppf(p)
    alpha = 1.0 - p
    return V0 * (
        math.exp(mu * h + 0.5 * sigma ** 2 * h) * norm.cdf(s_h - z_p) / alpha - 1.0
    )


# ── Internal ──────────────────────────────────────────────────────────────────

def _validate(V0: float, sigma: float, h: float, p: float) -> None:
    if V0 <= 0:
        raise ValueError(f"V0 must be positive (got {V0}).")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma}).")
    if h <= 0:
        raise ValueError(f"h must be positive (got {h}).")
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1) (got {p}).")
