"""
portfolio.py
Portfolio-level valuation and exposure computation.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.portfolio.positions import (
    option_delta_exposure,
    option_value,
    stock_value,
)
from src.schemas import Portfolio


# ── Valuation ──────────────────────────────────────────────────────────────────

def portfolio_value(
    portfolio: Portfolio,
    spots: pd.Series,
    pricing_date: date,
) -> float:
    """
    Compute total portfolio market value.

    V = Σ stock_values + Σ option_values

    Parameters
    ----------
    portfolio : Portfolio
    spots : pd.Series
        Current spot prices indexed by ticker symbol.
    pricing_date : date
        Date used for time-to-maturity calculation.

    Returns
    -------
    float
        Total portfolio value in dollars.
    """
    total = 0.0

    for pos in portfolio.stocks:
        total += stock_value(pos, float(spots[pos.ticker]))

    for pos in portfolio.options:
        total += option_value(pos, float(spots[pos.underlying_ticker]), pricing_date)

    return total


def reprice_portfolio(
    portfolio: Portfolio,
    shocked_spots: pd.Series,
    pricing_date: date,
) -> float:
    """
    Re-value the portfolio under a shocked spot price vector.
    Identical to portfolio_value but named separately for clarity in risk loops.
    """
    return portfolio_value(portfolio, shocked_spots, pricing_date)


# ── Exposure ───────────────────────────────────────────────────────────────────

def portfolio_exposure(
    portfolio: Portfolio,
    spots: pd.Series,
    pricing_date: date,
) -> pd.Series:
    """
    Compute the net delta-dollar exposure vector x_i for each underlying.

    For each underlying i:
        Δ_i = stock_quantity_i + Σ (option_deltas for underlying i)
        x_i = Δ_i × S_i

    Returns
    -------
    pd.Series
        Dollar-delta exposure indexed by underlying ticker.
    """
    underlyings = _all_underlyings(portfolio)
    exposure = pd.Series(0.0, index=underlyings)

    # Stock contributions: delta = quantity (one share = delta 1)
    for pos in portfolio.stocks:
        exposure[pos.ticker] += pos.quantity * float(spots[pos.ticker])

    # Option contributions: delta-dollar = quantity × multiplier × BS_delta × S
    for pos in portfolio.options:
        u = pos.underlying_ticker
        exposure[u] += option_delta_exposure(pos, float(spots[u]), pricing_date)

    return exposure


def _all_underlyings(portfolio: Portfolio) -> list[str]:
    """Return a deduplicated list of all underlying tickers in the portfolio."""
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
