"""
normal.py
Standalone normal (delta-normal) VaR and ES formulae (formula-sheet §4, HW IV).

    VaR = -m + s * Φ^{-1}(p)
    ES  = -m + s * φ(z) / (1-p)

where m = portfolio mean P&L, s = portfolio std dev P&L.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def normal_var(mean_pnl: float, std_pnl: float, confidence: float) -> float:
    """Closed-form normal VaR for a portfolio with known mean and std dev.

    VaR = −m + s × Φ⁻¹(confidence)

    Args:
        mean_pnl (float): Expected P&L over the horizon (positive = expected gain).
        std_pnl (float): Standard deviation of P&L over the horizon (must be ≥ 0).
        confidence (float): VaR confidence level, e.g. 0.99.

    Returns:
        float: VaR in the same units as mean_pnl / std_pnl (positive = loss).

    Example:
        >>> round(normal_var(0.0, 1000.0, 0.99), 2)
        2326.35
    """
    return float(-mean_pnl + std_pnl * norm.ppf(confidence))


def normal_es(mean_pnl: float, std_pnl: float, confidence: float) -> float:
    """Closed-form normal ES for a portfolio with known mean and std dev.

    ES = −m + s × φ(z) / (1 − confidence)

    where z = Φ⁻¹(confidence) and φ is the standard normal PDF.

    Args:
        mean_pnl (float): Expected P&L over the horizon.
        std_pnl (float): Standard deviation of P&L over the horizon.
        confidence (float): ES confidence level, e.g. 0.975.

    Returns:
        float: ES in the same units as mean_pnl / std_pnl (positive = loss).
            Always ≥ normal_var at the same confidence.
    """
    z = norm.ppf(confidence)
    return float(-mean_pnl + std_pnl * norm.pdf(z) / (1.0 - confidence))


def portfolio_delta_normal_mean_var(
    exposures: np.ndarray,
    mu_h: np.ndarray,
    cov_h: np.ndarray,
) -> tuple[float, float]:
    """Portfolio mean P&L and standard deviation via the delta-normal formula.

    Given dollar-delta exposure vector x and horizon distribution N(μ_h, Σ_h):
        m = x' μ_h
        s = sqrt(x' Σ_h x)

    Args:
        exposures (np.ndarray): Dollar-delta exposure vector x, shape (k,).
        mu_h (np.ndarray): Horizon mean log-return vector, shape (k,).
        cov_h (np.ndarray): Horizon covariance matrix, shape (k, k).

    Returns:
        tuple[float, float]: (m, s) - portfolio mean P&L and portfolio std dev.
            s is clamped to 0 if the quadratic form is slightly negative (numerical noise).
    """
    x = np.asarray(exposures, dtype=float)
    mu = np.asarray(mu_h, dtype=float)
    cov = np.asarray(cov_h, dtype=float)
    m = float(x @ mu)
    variance = float(x @ cov @ x)
    s = float(np.sqrt(max(variance, 0.0)))
    return (m, s)
