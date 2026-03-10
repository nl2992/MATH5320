"""
parametric.py
Delta-Normal (Parametric) VaR and ES.

Spec §12:
    Exposure vector  : x_i = Δ_i S_i
    Horizon scaling  : μ_h = μ × h,   Σ_h = Σ × h
    Portfolio mean   : m = x' μ_h
    Portfolio vol    : s = sqrt(x' Σ_h x)
    VaR              : VaR = -m + s Φ^{-1}(confidence)
    ES               : ES  = -m + s φ(z) / α     where α = 1 - confidence
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.portfolio.portfolio import portfolio_exposure
from src.risk.estimators import get_mean_cov
from src.risk.returns import compute_log_returns
from src.schemas import Portfolio


def parametric_var_es(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    pricing_date: date,
    lookback_days: int,
    horizon_days: int,
    var_confidence: float,
    es_confidence: float,
    estimator: str = "window",
    ewma_N: int = 60,
) -> dict:
    """
    Compute Parametric (Delta-Normal) VaR and ES.

    Returns
    -------
    dict with keys:
        var         : float — VaR in dollars
        es          : float — ES in dollars
        portfolio_mean : float — m
        portfolio_vol  : float — s
        exposure    : pd.Series — dollar-delta exposure per underlying
    """
    # Current spots
    spots_0 = prices.iloc[-1]

    # Exposure vector x (dollar-delta per underlying)
    exposure = portfolio_exposure(portfolio, spots_0, pricing_date)

    # Restrict to underlyings that have price data
    underlyings = [u for u in exposure.index if u in prices.columns]
    exposure = exposure[underlyings]

    # Estimate daily mean and covariance from log returns
    log_ret = compute_log_returns(prices[underlyings])
    mu_daily, cov_daily = get_mean_cov(log_ret, lookback_days, estimator, ewma_N)

    # Horizon scaling
    mu_h = mu_daily * horizon_days
    cov_h = cov_daily * horizon_days

    x = exposure.values  # shape (k,)
    mu_h_arr = mu_h.values
    cov_h_arr = cov_h.values

    # Portfolio-level mean and volatility
    m = float(x @ mu_h_arr)
    variance = float(x @ cov_h_arr @ x)
    s = float(np.sqrt(max(variance, 0.0)))

    # VaR
    z_var = norm.ppf(var_confidence)
    var = float(-m + s * z_var)

    # ES
    alpha_var = 1.0 - var_confidence
    phi_z = norm.pdf(z_var)
    es_var = float(-m + s * phi_z / alpha_var)

    return {
        "var": var,
        "es": es_var,
        "portfolio_mean": m,
        "portfolio_vol": s,
        "exposure": exposure,
    }
