"""
historical.py
Historical simulation VaR and ES via full portfolio repricing.

Algorithm (from spec §11):
    1. Compute daily log returns.
    2. Use last lookback_days observations.
    3. Build overlapping h-day returns.
    4. Compute current portfolio value V0.
    5. For each scenario:
           S_shocked = S0 × exp(R)
           Reprice entire portfolio.
           loss = V0 − V_scenario
    6. VaR = empirical quantile(loss, confidence)
    7. ES  = mean(loss | loss ≥ VaR)
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.portfolio.portfolio import portfolio_value, reprice_portfolio
from src.risk.returns import build_overlapping_horizon_log_returns, compute_log_returns
from src.schemas import Portfolio


def historical_var_es(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    pricing_date: date,
    lookback_days: int,
    horizon_days: int,
    var_confidence: float,
    es_confidence: float,
) -> dict:
    """
    Compute Historical VaR and ES.

    Parameters
    ----------
    portfolio : Portfolio
    prices : pd.DataFrame
        Full price history (index: DatetimeIndex, columns: tickers).
    pricing_date : date
        The "today" date for option time-to-maturity.
    lookback_days : int
        Number of daily observations to include in the lookback window.
    horizon_days : int
        Risk horizon h in trading days.
    var_confidence : float
        VaR confidence level, e.g. 0.99.
    es_confidence : float
        ES confidence level, e.g. 0.975.

    Returns
    -------
    dict with keys:
        var        : float  — VaR in dollars (positive = loss)
        es         : float  — ES in dollars  (positive = loss)
        losses     : np.ndarray — full loss distribution (dollars)
        n_scenarios: int    — number of scenarios used
    """
    # Current spot prices (last available row)
    spots_0 = prices.iloc[-1]

    # V0
    V0 = portfolio_value(portfolio, spots_0, pricing_date)

    # Log returns → overlapping h-day returns
    log_ret = compute_log_returns(prices)
    horizon_ret = build_overlapping_horizon_log_returns(log_ret, horizon_days)

    # Restrict to lookback window
    scenario_ret = horizon_ret.tail(lookback_days)

    # Identify the underlyings we actually need
    underlyings = _portfolio_underlyings(portfolio)
    # Only keep columns that appear in the portfolio
    available = [u for u in underlyings if u in scenario_ret.columns]
    scenario_ret = scenario_ret[available]

    losses = _compute_losses(portfolio, spots_0, scenario_ret, pricing_date, V0)

    var = float(np.quantile(losses, var_confidence))
    tail_losses = losses[losses >= var]
    es = float(tail_losses.mean()) if len(tail_losses) > 0 else var

    return {
        "var": var,
        "es": es,
        "losses": losses,
        "n_scenarios": len(losses),
    }


def _compute_losses(
    portfolio: Portfolio,
    spots_0: pd.Series,
    scenario_returns: pd.DataFrame,
    pricing_date: date,
    V0: float,
) -> np.ndarray:
    """Vectorised loss computation over all scenarios."""
    losses = np.empty(len(scenario_returns))

    for i, (_, row) in enumerate(scenario_returns.iterrows()):
        # Shocked spots: S_shocked = S0 × exp(R)
        shocked = spots_0.copy()
        for ticker in row.index:
            if ticker in shocked.index:
                shocked[ticker] = float(spots_0[ticker]) * np.exp(float(row[ticker]))

        V_scenario = reprice_portfolio(portfolio, shocked, pricing_date)
        losses[i] = V0 - V_scenario  # loss = V0 - V_T

    return losses


def _portfolio_underlyings(portfolio: Portfolio) -> list[str]:
    """Return all unique underlying tickers referenced by the portfolio."""
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
