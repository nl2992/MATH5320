"""
validation.py
Input validation helpers for market data and portfolio definitions.
"""
from __future__ import annotations

import pandas as pd

from src.schemas import Portfolio


def validate_price_dataframe(prices: pd.DataFrame) -> list[str]:
    """
    Return a list of validation error messages for a price DataFrame.
    Empty list means the data is valid.
    """
    errors: list[str] = []

    if prices is None or prices.empty:
        errors.append("Price DataFrame is empty.")
        return errors

    if not isinstance(prices.index, pd.DatetimeIndex):
        errors.append("Price DataFrame index must be a DatetimeIndex.")

    if prices.isnull().all().any():
        bad = prices.columns[prices.isnull().all()].tolist()
        errors.append(f"Columns with all NaN values: {bad}")

    if (prices <= 0).any().any():
        errors.append("Price DataFrame contains non-positive prices.")

    return errors


def validate_portfolio_tickers(portfolio: Portfolio, price_columns: list[str]) -> list[str]:
    """
    Ensure every underlying ticker in the portfolio exists in the price data.
    Returns a list of error messages.
    """
    errors: list[str] = []
    available = set(price_columns)

    for pos in portfolio.stocks:
        if pos.ticker not in available:
            errors.append(f"Stock ticker '{pos.ticker}' not found in price data.")

    for pos in portfolio.options:
        if pos.underlying_ticker not in available:
            errors.append(
                f"Option '{pos.ticker}' underlying '{pos.underlying_ticker}' "
                "not found in price data."
            )

    return errors
