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
    """Market value of a stock position.

    value = quantity × spot

    Args:
        pos (StockPosition): Stock position with a ticker and quantity.
            Positive quantity = long; negative = short.
        spot (float): Current spot price of the stock in dollars.

    Returns:
        float: Dollar market value. Negative if the position is short.

    Example:
        >>> from src.schemas import StockPosition
        >>> stock_value(StockPosition("AAPL", 100), 150.0)
        15000.0
    """
    return pos.quantity * spot


def option_value(
    pos: OptionPosition,
    spot: float,
    pricing_date: date,
    volatility_override: float | None = None,
) -> float:
    """Market value of an option position using Black-Scholes.

    value = quantity × contract_multiplier × BS_price(spot, ...)

    For expired options (T ≤ 0) the position is valued at intrinsic value:
        call intrinsic = max(spot − strike, 0)
        put  intrinsic = max(strike − spot, 0)

    Args:
        pos (OptionPosition): Option position dataclass with all pricing params.
        spot (float): Current spot price of the underlying in dollars.
        pricing_date (date): Evaluation date used to compute time to maturity.
        volatility_override (float | None): If given, replaces pos.volatility for
            this pricing call. Useful for scenario-specific vol shocks.

    Returns:
        float: Dollar value of the full position (quantity × multiplier × per-share price).
            Positive if long and in-the-money; can be negative if the position is short.

    Raises:
        ValueError: Propagated from ``bs_price`` if spot ≤ 0 or vol ≤ 0.
    """
    T = _time_to_maturity(pricing_date, pos.maturity_date)
    if T <= 0.0:
        # Expired option - intrinsic value only
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
    """Compute a scenario-specific implied volatility for an option position.

    Two modes are supported:
        ``"fixed"`` - returns pos.volatility unchanged regardless of the return.
        ``"underlying_beta"`` - applies a leverage-style shock so that a
            negative underlying return increases the implied vol:
                sigma' = max(floor, sigma0 × (1 − beta × R))

    Args:
        pos (OptionPosition): Option position (provides the base volatility).
        underlying_return (float | None): Log return of the underlying for this
            scenario. If None, the position's own volatility is returned regardless
            of mode.
        mode (str): ``"fixed"`` (default) or ``"underlying_beta"``.
        beta (float): Sensitivity coefficient for ``"underlying_beta"`` mode.
            Default 1.0 (one-for-one leverage adjustment).
        floor (float): Minimum allowed shocked volatility. Default 0.05 (5%).

    Returns:
        float: Scenario implied volatility to use when repricing the option.

    Raises:
        ValueError: If mode is not ``"fixed"`` or ``"underlying_beta"``.
    """
    if underlying_return is None or mode == "fixed":
        return pos.volatility

    if mode == "underlying_beta":
        return max(floor, pos.volatility * (1.0 - beta * underlying_return))

    raise ValueError(f"Unknown option volatility shock mode: '{mode}'")


def option_delta_exposure(
    pos: OptionPosition, spot: float, pricing_date: date
) -> float:
    """Dollar-delta exposure of an option position.

    Δ_exposure = quantity × contract_multiplier × BS_delta × spot

    This gives the first-order sensitivity of the position value to a 1%
    proportional move in the underlying (scaled by 100). For expired options
    (T ≤ 0) the delta-dollar exposure is zero.

    Args:
        pos (OptionPosition): Option position with pricing parameters.
        spot (float): Current spot price of the underlying in dollars.
        pricing_date (date): Evaluation date used to compute time to maturity.

    Returns:
        float: Dollar-delta exposure. Positive for long calls, negative for long puts.
            Zero if the option has expired.
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
