"""
returns.py
Log-return computation and overlapping horizon return construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    r_t = log(S_t / S_{t-1})

    Returns
    -------
    pd.DataFrame
        Log returns with the first row dropped (NaN from shift).
    """
    return np.log(prices / prices.shift(1)).dropna()


def build_overlapping_horizon_log_returns(
    log_returns: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Build overlapping h-day log returns by rolling summation.

    R_t^(h) = r_t + r_{t-1} + ... + r_{t-h+1}

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    horizon_days : int
        Horizon h in trading days.

    Returns
    -------
    pd.DataFrame
        Overlapping h-day log returns (NaN rows from the initial window dropped).
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    return log_returns.rolling(horizon_days).sum().dropna()
