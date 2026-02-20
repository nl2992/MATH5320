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
from src.risk.estimators import get_mean_cov, manual_mean_cov
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
    calibration_mode: str = "historical",
    manual_market_params: dict | None = None,
) -> dict:
    """Parametric (Delta-Normal) VaR and ES.

    The delta-normal method approximates the portfolio P&L as a linear function
    of underlying log returns. Given the exposure vector x (dollar-delta per
    underlying) and scaled distribution N(μ_h, Σ_h):
        m = x' μ_h,   s = sqrt(x' Σ_h x)
        VaR = -m + s × Φ⁻¹(var_confidence)
        ES  = -m + s × φ(z_es) / (1 − es_confidence)

    Args:
        portfolio (Portfolio): Stock and option positions. Options contribute
            via their Black-Scholes delta-dollar exposure only (no repricing).
        prices (pd.DataFrame): Aligned price history (DatetimeIndex × tickers).
        pricing_date (date): Valuation date for computing option deltas.
        lookback_days (int): Trailing observations for mean/cov estimation.
        horizon_days (int): Risk horizon h; μ_h = μ×h, Σ_h = Σ×h.
        var_confidence (float): VaR confidence level, e.g. 0.99.
        es_confidence (float): ES averaging confidence, e.g. 0.975.
        estimator (str): ``"window"`` (simple rolling) or ``"ewma"`` (exponentially
            weighted). Default ``"window"``.
        ewma_N (int): EWMA half-life parameter N; λ = (N-1)/(N+1). Only used
            when estimator=``"ewma"``.
        calibration_mode (str): ``"historical"`` (estimate from prices) or
            ``"manual"`` (supply μ and Σ directly via manual_market_params).
        manual_market_params (dict | None): Required when calibration_mode=``"manual"``.
            Must contain ``"mu_daily"`` (pd.Series) and ``"cov_daily"`` (pd.DataFrame)
            keyed by underlying ticker.

    Returns:
        dict: Result dictionary with keys:
            - ``"var"`` (float): Parametric VaR in dollars.
            - ``"es"`` (float): Parametric ES in dollars.
            - ``"var_confidence"`` (float): var_confidence passed in.
            - ``"es_confidence"`` (float): es_confidence passed in.
            - ``"portfolio_mean"`` (float): m = x' μ_h.
            - ``"portfolio_vol"`` (float): s = sqrt(x' Σ_h x).
            - ``"exposure"`` (pd.Series): Dollar-delta exposure vector x.
            - ``"calibration_mode"`` (str): Mode used.
    """
    # Current spots
    spots_0 = prices.iloc[-1]

    # Exposure vector x (dollar-delta per underlying)
    exposure = portfolio_exposure(portfolio, spots_0, pricing_date)

    # Restrict to underlyings that have price data
    underlyings = [u for u in exposure.index if u in prices.columns]
    exposure = exposure[underlyings]

    # Estimate daily mean and covariance from log returns
    if calibration_mode == "manual":
        mu_daily, cov_daily = manual_mean_cov(manual_market_params or {}, underlyings)
    else:
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

    # ES — use es_confidence separately
    z_es = norm.ppf(es_confidence)
    alpha_es = 1.0 - es_confidence
    es = float(-m + s * norm.pdf(z_es) / alpha_es)

    return {
        "var": var,
        "es": es,
        "var_confidence": var_confidence,
        "es_confidence": es_confidence,
        "portfolio_mean": m,
        "portfolio_vol": s,
        "exposure": exposure,
        "calibration_mode": calibration_mode,
    }
