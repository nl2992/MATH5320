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
    """VaR = -mean_pnl + std_pnl * Φ^{-1}(confidence)."""
    return float(-mean_pnl + std_pnl * norm.ppf(confidence))


def normal_es(mean_pnl: float, std_pnl: float, confidence: float) -> float:
    """ES = -mean_pnl + std_pnl * φ(z) / (1-confidence)."""
    z = norm.ppf(confidence)
    return float(-mean_pnl + std_pnl * norm.pdf(z) / (1.0 - confidence))


def portfolio_delta_normal_mean_var(
    exposures: np.ndarray,
    mu_h: np.ndarray,
    cov_h: np.ndarray,
) -> tuple[float, float]:
    """
    Given dollar-delta exposure vector x, horizon mean μ_h, horizon covariance Σ_h:
        m = x' μ_h
        s = sqrt(x' Σ_h x)
    Returns (m, s).
    """
    x = np.asarray(exposures, dtype=float)
    mu = np.asarray(mu_h, dtype=float)
    cov = np.asarray(cov_h, dtype=float)
    m = float(x @ mu)
    variance = float(x @ cov @ x)
    s = float(np.sqrt(max(variance, 0.0)))
    return (m, s)
