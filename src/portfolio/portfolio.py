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
    shocked_option_volatility,
    option_value,
    stock_value,
)
from src.schemas import Portfolio


# ── Valuation ──────────────────────────────────────────────────────────────────

def portfolio_value(
    portfolio: Portfolio,
    spots: pd.Series,
    pricing_date: date,
    underlying_returns: pd.Series | dict[str, float] | None = None,
    option_vol_shock_mode: str = "fixed",
    option_vol_shock_beta: float = 1.0,
    option_vol_shock_floor: float = 0.05,
) -> float:
    """Total mark-to-market value of the portfolio.

    V = Σ_i (quantity_i × spot_i)          [stocks]
      + Σ_j (qty_j × mult_j × BS_price_j)  [options, Black-Scholes]

    Args:
        portfolio (Portfolio): Collection of stock and option positions.
        spots (pd.Series): Current spot prices indexed by ticker symbol (dollars).
        pricing_date (date): Valuation date, used to compute option time-to-maturity.
        underlying_returns (pd.Series | dict | None): Per-underlying log returns for
            this scenario. Required only when option_vol_shock_mode != ``"fixed"``
            to compute the shocked implied vol. Pass None to use fixed vols.
        option_vol_shock_mode (str): ``"fixed"`` (default) or ``"underlying_beta"``.
            See ``shocked_option_volatility`` for details.
        option_vol_shock_beta (float): Beta coefficient for ``"underlying_beta"`` mode.
        option_vol_shock_floor (float): Minimum shocked volatility for option positions.

    Returns:
        float: Total portfolio value in dollars. Negative is possible for net short books.

    Raises:
        KeyError: If spots is missing a ticker required by the portfolio.
    """
    total = 0.0

    for pos in portfolio.stocks:
        total += stock_value(pos, float(spots[pos.ticker]))

    for pos in portfolio.options:
        u = pos.underlying_ticker
        underlying_return = None
        if underlying_returns is not None and u in underlying_returns:
            underlying_return = float(underlying_returns[u])
        vol_override = shocked_option_volatility(
            pos,
            underlying_return=underlying_return,
            mode=option_vol_shock_mode,
            beta=option_vol_shock_beta,
            floor=option_vol_shock_floor,
        )
        total += option_value(
            pos,
            float(spots[u]),
            pricing_date,
            volatility_override=vol_override,
        )

    return total


def reprice_portfolio(
    portfolio: Portfolio,
    shocked_spots: pd.Series,
    pricing_date: date,
    underlying_returns: pd.Series | dict[str, float] | None = None,
    option_vol_shock_mode: str = "fixed",
    option_vol_shock_beta: float = 1.0,
    option_vol_shock_floor: float = 0.05,
) -> float:
    """Re-value the portfolio under a shocked spot price vector.

    Alias for ``portfolio_value`` with shocked_spots. Named separately so
    risk-loop code reads clearly: ``V0 = portfolio_value(...)``,
    ``V_scenario = reprice_portfolio(..., shocked_spots)``.

    Args:
        portfolio (Portfolio): Collection of stock and option positions.
        shocked_spots (pd.Series): Spot prices after applying a market scenario shock
            (dollars), indexed by ticker symbol.
        pricing_date (date): Valuation date for option time-to-maturity.
        underlying_returns (pd.Series | dict | None): Per-underlying log returns
            of this scenario (used for vol shocks when mode != ``"fixed"``).
        option_vol_shock_mode (str): ``"fixed"`` or ``"underlying_beta"``.
        option_vol_shock_beta (float): Beta for ``"underlying_beta"`` mode.
        option_vol_shock_floor (float): Minimum shocked vol.

    Returns:
        float: Portfolio value at the shocked spot prices (dollars).
    """
    return portfolio_value(
        portfolio,
        shocked_spots,
        pricing_date,
        underlying_returns=underlying_returns,
        option_vol_shock_mode=option_vol_shock_mode,
        option_vol_shock_beta=option_vol_shock_beta,
        option_vol_shock_floor=option_vol_shock_floor,
    )


# ── Exposure ───────────────────────────────────────────────────────────────────

def portfolio_exposure(
    portfolio: Portfolio,
    spots: pd.Series,
    pricing_date: date,
) -> pd.Series:
    """Net dollar-delta exposure vector across all underlyings.

    For each underlying i in the portfolio:
        x_i = (stock_quantity_i × spot_i) + Σ_j (option_delta_exposure_j)

    This vector is used by the parametric (delta-normal) VaR engine as the
    exposure multiplier in the quadratic form  x' Σ_h x.

    Args:
        portfolio (Portfolio): Collection of stock and option positions.
        spots (pd.Series): Current spot prices indexed by ticker (dollars).
        pricing_date (date): Valuation date, used for option delta computation.

    Returns:
        pd.Series: Dollar-delta exposure indexed by underlying ticker symbol.
            Stock contribution: quantity × spot.
            Option contribution: quantity × multiplier × BS_delta × spot.
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
