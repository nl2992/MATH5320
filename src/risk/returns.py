"""
returns.py
Log-return computation and overlapping horizon return construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    r_t = log(S_t / S_{t-1})

    Args:
        prices (pd.DataFrame): Price history; DatetimeIndex × ticker columns.
            Each column should contain positive price levels.

    Returns:
        pd.DataFrame: Daily log returns with the same column layout as
            ``prices``. The first row (NaN from the shift) is dropped, so
            the returned DataFrame has one fewer row than the input.
    """
    return np.log(prices / prices.shift(1)).dropna()


def build_overlapping_horizon_log_returns(
    log_returns: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Build overlapping h-day log returns by rolling summation.

    R_t^(h) = r_t + r_{t-1} + ... + r_{t-h+1}

    Args:
        log_returns (pd.DataFrame): Daily log returns (output of
            :func:`compute_log_returns`).
        horizon_days (int): Horizon h in trading days (must be ≥ 1).

    Returns:
        pd.DataFrame: Overlapping h-day log returns.  NaN rows from the
            initial rolling window are dropped; shape is
            ``(len(log_returns) − horizon_days + 1) × n_tickers``.

    Raises:
        ValueError: If ``horizon_days < 1``.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    return log_returns.rolling(horizon_days).sum().dropna()


def build_overlapping_horizon_absolute_returns(
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Build overlapping h-day dollar changes.

    dollar_change_t^(h) = S_t − S_{t-h}

    Indexed by the ending date.

    Args:
        prices (pd.DataFrame): Price levels (not returns); DatetimeIndex
            × ticker columns.
        horizon (int): Horizon h in trading days (must be ≥ 1).

    Returns:
        pd.DataFrame: Overlapping h-day dollar changes. The first
            ``horizon`` rows (NaN from the shift) are dropped.

    Raises:
        ValueError: If ``horizon < 1``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    return (prices - prices.shift(horizon)).dropna()
