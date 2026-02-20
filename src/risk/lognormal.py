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
    """Exact long-position VaR under GBM (formula-sheet §4).

    VaR_long(p, h) = V₀ × [ 1 − exp( m_h + s_h × z_{1−p} ) ]

    where  m_h = (μ − ½σ²) h  and  s_h = σ √h.
    z_{1−p} = Φ⁻¹(1−p) is negative for p > 0.5, making the loss positive.

    Args:
        V0 (float): Current portfolio value in dollars (must be > 0).
        mu (float): Annualised arithmetic drift of the log-normal process.
        sigma (float): Annualised volatility (must be > 0).
        h (float): Horizon in years (must be > 0; use h = days/252 for trading days).
        p (float): VaR confidence level, e.g. 0.99 (must be in (0, 1)).

    Returns:
        float: VaR in dollars (positive = loss). Returns a negative value only when
            the drift is so large that the loss quantile is a gain.

    Raises:
        ValueError: If V0 ≤ 0, sigma ≤ 0, h ≤ 0, or p not in (0, 1).

    Example:
        >>> round(var_long_lognormal(V0=100_000, mu=0.08, sigma=0.20, h=5/252, p=0.99), 2)
        5893.36
    """
    _validate(V0, sigma, h, p)
    m_h = (mu - 0.5 * sigma ** 2) * h
    s_h = sigma * math.sqrt(h)
    z = norm.ppf(1.0 - p)
    return V0 * (1.0 - math.exp(m_h + s_h * z))


def es_long_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """Exact long-position ES under GBM (formula-sheet §4).

    ES_long(p, h) = V₀ × [ 1 − exp(m_h + ½ s_h²) × N(z_{1−p} − s_h) / (1 − p) ]

    where  m_h = (μ − ½σ²) h,  s_h = σ √h,  z_{1−p} = Φ⁻¹(1−p).

    Args:
        V0 (float): Current portfolio value in dollars (must be > 0).
        mu (float): Annualised arithmetic drift.
        sigma (float): Annualised volatility (must be > 0).
        h (float): Horizon in years (must be > 0).
        p (float): ES averaging confidence level, e.g. 0.975 (must be in (0, 1)).

    Returns:
        float: ES in dollars (positive = loss). Always ≥ the VaR at the same p.

    Raises:
        ValueError: If V0 ≤ 0, sigma ≤ 0, h ≤ 0, or p not in (0, 1).
    """
    _validate(V0, sigma, h, p)
    m_h = (mu - 0.5 * sigma ** 2) * h
    s_h = sigma * math.sqrt(h)
    z = norm.ppf(1.0 - p)
    alpha = 1.0 - p
    return V0 * (1.0 - math.exp(m_h + 0.5 * s_h ** 2) * norm.cdf(z - s_h) / alpha)


# ── Short position ─────────────────────────────────────────────────────────────

def var_short_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """Exact short-position VaR under GBM (formula-sheet §7).

    For a short position of value V₀ in a GBM asset, losses come from upward moves.
    The VaR is the loss at the p-th upper quantile of the return distribution:

    VaR_short(p, h) = V₀ × [ exp( m_h + z_p × σ √h ) − 1 ]

    where  m_h = (μ − ½σ²) h  and  z_p = Φ⁻¹(p).

    Args:
        V0 (float): Absolute value of the short position in dollars (must be > 0).
        mu (float): Annualised arithmetic drift of the underlying.
        sigma (float): Annualised volatility (must be > 0).
        h (float): Horizon in years (must be > 0).
        p (float): VaR confidence level, e.g. 0.99 (must be in (0, 1)).

    Returns:
        float: VaR in dollars (positive = loss for the short holder).

    Raises:
        ValueError: If V0 ≤ 0, sigma ≤ 0, h ≤ 0, or p not in (0, 1).
    """
    _validate(V0, sigma, h, p)
    m_h = (mu - 0.5 * sigma ** 2) * h
    s_h = sigma * math.sqrt(h)
    z_p = norm.ppf(p)
    return V0 * (math.exp(m_h + z_p * s_h) - 1.0)


def es_short_lognormal(V0: float, mu: float, sigma: float, h: float, p: float) -> float:
    """Exact short-position ES under GBM (formula-sheet §7).

    ES_short(p, h) = V₀ × [ exp(m_h + ½ s_h²) × N(s_h − z_p) / (1 − p) − 1 ]

    Note: m_h + ½ s_h² = (μ − ½σ²)h + ½σ²h = μ h, so the exponential term
    simplifies to exp(μ h) — the expected growth of the underlying.

    Args:
        V0 (float): Absolute value of the short position in dollars (must be > 0).
        mu (float): Annualised arithmetic drift of the underlying.
        sigma (float): Annualised volatility (must be > 0).
        h (float): Horizon in years (must be > 0).
        p (float): ES confidence level, e.g. 0.975 (must be in (0, 1)).

    Returns:
        float: ES in dollars (positive = loss). Always ≥ VaR_short at the same p.

    Raises:
        ValueError: If V0 ≤ 0, sigma ≤ 0, h ≤ 0, or p not in (0, 1).
    """
    _validate(V0, sigma, h, p)
    m_h = (mu - 0.5 * sigma ** 2) * h
    s_h = sigma * math.sqrt(h)
    z_p = norm.ppf(p)
    alpha = 1.0 - p
    return V0 * (
        math.exp(m_h + 0.5 * s_h ** 2) * norm.cdf(s_h - z_p) / alpha - 1.0
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
