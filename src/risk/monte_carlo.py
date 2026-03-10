"""
monte_carlo.py
Monte Carlo VaR and ES via full portfolio repricing.

Spec §13:
    Simulate  R_sim ~ N(μ_h, Σ_h)
    For each simulation:
        S_sim = S0 × exp(R_sim)
        Reprice portfolio.
        Compute losses.
    VaR and ES computed empirically.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.portfolio.portfolio import reprice_portfolio
from src.risk.estimators import get_mean_cov
from src.risk.returns import compute_log_returns
from src.schemas import Portfolio


def monte_carlo_var_es(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    pricing_date: date,
    lookback_days: int,
    horizon_days: int,
    var_confidence: float,
    es_confidence: float,
    n_simulations: int = 10_000,
    estimator: str = "window",
    ewma_N: int = 60,
    random_seed: int | None = 42,
) -> dict:
    """
    Compute Monte Carlo VaR and ES.

    Returns
    -------
    dict with keys:
        var        : float
        es         : float
        losses     : np.ndarray — simulated loss distribution
        n_simulations : int
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # Current spots
    spots_0 = prices.iloc[-1]

    # Identify underlyings with price data
    underlyings = _portfolio_underlyings(portfolio)
    underlyings = [u for u in underlyings if u in prices.columns]

    # Estimate daily mean and covariance
    log_ret = compute_log_returns(prices[underlyings])
    mu_daily, cov_daily = get_mean_cov(log_ret, lookback_days, estimator, ewma_N)

    # Horizon scaling
    mu_h = mu_daily.values * horizon_days          # shape (k,)
    cov_h = cov_daily.values * horizon_days        # shape (k, k)

    # Simulate R_sim ~ N(μ_h, Σ_h)
    R_sim = rng.multivariate_normal(mu_h, cov_h, size=n_simulations)  # (n_sim, k)

    # Current portfolio value
    from src.portfolio.portfolio import portfolio_value
    V0 = portfolio_value(portfolio, spots_0, pricing_date)

    # Compute losses
    losses = np.empty(n_simulations)
    spots_0_arr = np.array([float(spots_0[u]) for u in underlyings])

    for i in range(n_simulations):
        shocked_arr = spots_0_arr * np.exp(R_sim[i])
        shocked = pd.Series(shocked_arr, index=underlyings)
        # Merge with full spots_0 (in case portfolio has more tickers)
        full_shocked = spots_0.copy()
        for u, v in shocked.items():
            full_shocked[u] = v
        V_sim = reprice_portfolio(portfolio, full_shocked, pricing_date)
        losses[i] = V0 - V_sim

    var = float(np.quantile(losses, var_confidence))
    tail_losses = losses[losses >= var]
    es = float(tail_losses.mean()) if len(tail_losses) > 0 else var

    return {
        "var": var,
        "es": es,
        "losses": losses,
        "n_simulations": n_simulations,
    }


def _portfolio_underlyings(portfolio: Portfolio) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for pos in portfolio.stocks:
        if pos.ticker not in seen:
            seen.add(pos.ticker)
            result.append(pos.ticker)
    for pos in portfolio.options:
        if pos.underlying_ticker not in seen:
            seen.add(pos.underlying_ticker)
            result.append(pos.underlying_ticker)
    return result
