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
    else:
        if prices.index.has_duplicates:
            dupes = prices.index[prices.index.duplicated()].unique()
            errors.append(
                "Price DataFrame contains duplicate dates: "
                + ", ".join(str(ts.date()) if hasattr(ts, "date") else str(ts) for ts in dupes[:5])
                + (" ..." if len(dupes) > 5 else "")
            )
        if not prices.index.is_monotonic_increasing:
            errors.append("Price DataFrame index must be sorted in ascending date order.")

    if prices.isnull().all().any():
        bad = prices.columns[prices.isnull().all()].tolist()
        errors.append(f"Columns with all NaN values: {bad}")

    partial_nan_cols = prices.columns[prices.isnull().any() & ~prices.isnull().all()].tolist()
    if partial_nan_cols:
        errors.append(
            "Price DataFrame contains missing values in active columns: "
            + ", ".join(partial_nan_cols)
        )

    if (prices <= 0).any().any():
        errors.append("Price DataFrame contains non-positive prices.")

    return errors


def warn_price_dataframe(prices: pd.DataFrame, stale_run_limit: int = 10) -> list[str]:
    """
    Return non-fatal data-quality warnings for a price DataFrame.

    Current warnings focus on unusually long runs of unchanged prices, which can
    indicate stale data even when the series remains positive and complete.
    """
    warnings: list[str] = []
    if prices is None or prices.empty:
        return warnings
    if not isinstance(prices.index, pd.DatetimeIndex):
        return warnings

    stale_cols: list[str] = []
    for col in prices.columns:
        series = pd.Series(prices[col], dtype=float)
        longest = _longest_constant_run(series)
        if longest >= stale_run_limit:
            stale_cols.append(f"{col} ({longest} days)")

    if stale_cols:
        warnings.append(
            "Potential stale-price runs detected: " + ", ".join(stale_cols)
        )

    return warnings


def validate_history_requirements(
    prices: pd.DataFrame,
    lookback_days: int,
    horizon_days: int,
    *,
    for_backtest: bool = False,
) -> list[str]:
    """
    Validate that a price history is long enough for the requested calculation.
    """
    errors: list[str] = []
    if prices is None or prices.empty:
        return ["Price DataFrame is empty."]

    required_rows = (
        lookback_days + horizon_days + 2 if for_backtest else lookback_days + horizon_days
    )
    if len(prices) < required_rows:
        label = "backtest" if for_backtest else "risk run"
        errors.append(
            f"Insufficient history for the requested {label}: need at least "
            f"{required_rows} price rows for lookback_days={lookback_days} and "
            f"horizon_days={horizon_days}, got {len(prices)}."
        )
    return errors


def _longest_constant_run(series: pd.Series) -> int:
    """
    Longest consecutive run of unchanged finite values.
    """
    if series.empty:
        return 0

    values = series.to_numpy(dtype=float)
    longest = 1
    current = 1
    for i in range(1, len(values)):
        if pd.isna(values[i]) or pd.isna(values[i - 1]):
            current = 1
            continue
        if values[i] == values[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


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
