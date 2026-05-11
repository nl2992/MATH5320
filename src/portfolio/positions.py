"""
positions.py
Per-position valuation and delta helpers.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.pricing.black_scholes import bs_price, bs_delta
from src.schemas import OptionPosition, StockPosition


def stock_value(pos: StockPosition, spot: float) -> float:
    """
    Compute the market value of a stock position.

    value = quantity × spot
    """
    return pos.quantity * spot


def option_value(
    pos: OptionPosition,
    spot: float,
    pricing_date: date,
    volatility_override: float | None = None,
) -> float:
    """
    Compute the market value of an option position using Black-Scholes.

    value = quantity × multiplier × BS_price
    """
    T = _time_to_maturity(pricing_date, pos.maturity_date)
    if T <= 0.0:
        # Expired option — intrinsic value only
        if pos.option_type.lower() == "call":
            intrinsic = max(spot - pos.strike, 0.0)
        else:
            intrinsic = max(pos.strike - spot, 0.0)
        return pos.quantity * pos.contract_multiplier * intrinsic

    price = bs_price(
        S=spot,
        K=pos.strike,
        T=T,
        r=pos.risk_free_rate,
        q=pos.dividend_yield,
        sigma=pos.volatility if volatility_override is None else volatility_override,
        option_type=pos.option_type,
    )
    return pos.quantity * pos.contract_multiplier * price


def shocked_option_volatility(
    pos: OptionPosition,
    underlying_return: float | None,
    mode: str = "fixed",
    beta: float = 1.0,
    floor: float = 0.05,
) -> float:
    """
    Compute a scenario-specific option volatility.

    Modes
    -----
    fixed
        Keep the option's input implied volatility unchanged.
    underlying_beta
        Apply a simple leverage-style shock:

            sigma' = max(floor, sigma0 * (1 - beta * R))

        so negative returns increase volatility when beta > 0.
    """
    if underlying_return is None or mode == "fixed":
        return pos.volatility

    if mode == "underlying_beta":
        return max(floor, pos.volatility * (1.0 - beta * underlying_return))

    raise ValueError(f"Unknown option volatility shock mode: '{mode}'")


def option_delta_exposure(
    pos: OptionPosition, spot: float, pricing_date: date
) -> float:
    """
    Compute the dollar-delta exposure of an option position.

    Δ_exposure = quantity × multiplier × BS_delta × spot

    This is the sensitivity of the option position value to a unit
    proportional move in the underlying (i.e., the delta-dollar exposure).
    """
    T = _time_to_maturity(pricing_date, pos.maturity_date)
    if T <= 0.0:
        return 0.0

    delta = bs_delta(
        S=spot,
        K=pos.strike,
        T=T,
        r=pos.risk_free_rate,
        q=pos.dividend_yield,
        sigma=pos.volatility,
        option_type=pos.option_type,
    )
    return pos.quantity * pos.contract_multiplier * delta * spot


def _time_to_maturity(pricing_date: date, maturity_date: date) -> float:
    """Return time to maturity in years (act/365)."""
    days = (maturity_date - pricing_date).days
    return max(days / 365.0, 0.0)
