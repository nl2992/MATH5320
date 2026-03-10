"""
estimators.py
Mean and covariance estimators: rolling window and EWMA.

EWMA lambda formula (from spec):
    λ = (N - 1) / (N + 1)

EWMA weights:
    w_i = (1 - λ) λ^(n-i)   for i = 1 ... n   (oldest to newest)
    Weights are then normalised to sum to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_window_mean_cov(
    returns: pd.DataFrame,
    lookback_days: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Estimate mean and covariance using a simple rolling window.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (all history available).
    lookback_days : int
        Number of most-recent observations to use.

    Returns
    -------
    mu : pd.Series
        Sample mean of daily log returns.
    cov : pd.DataFrame
        Sample covariance matrix of daily log returns.
    """
    window = returns.tail(lookback_days)
    mu = window.mean()
    cov = window.cov()
    return mu, cov


def _ewma_lambda(N: int) -> float:
    """Compute EWMA decay parameter from half-life parameter N."""
    return (N - 1) / (N + 1)


def estimate_ewma_mean_cov(
    returns: pd.DataFrame,
    lookback_days: int,
    N: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Estimate mean and covariance using exponentially weighted moving averages.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (all history available).
    lookback_days : int
        Number of most-recent observations to use as the EWMA window.
    N : int
        EWMA half-life parameter; λ = (N-1)/(N+1).

    Returns
    -------
    mu : pd.Series
        EWMA-weighted mean of daily log returns.
    cov : pd.DataFrame
        EWMA-weighted covariance matrix.
    """
    window = returns.tail(lookback_days).values  # shape (n, k)
    n, k = window.shape
    lam = _ewma_lambda(N)

    # Build raw weights w_i = (1-λ) λ^(n-i) for i=1..n (oldest first)
    exponents = np.arange(n - 1, -1, -1, dtype=float)  # n-1, n-2, ..., 0
    raw_weights = (1.0 - lam) * (lam ** exponents)
    weights = raw_weights / raw_weights.sum()  # normalise

    # Weighted mean
    mu_arr = weights @ window  # shape (k,)

    # Weighted covariance
    demeaned = window - mu_arr  # (n, k)
    cov_arr = (demeaned * weights[:, None]).T @ demeaned  # (k, k)

    cols = returns.columns
    mu = pd.Series(mu_arr, index=cols)
    cov = pd.DataFrame(cov_arr, index=cols, columns=cols)
    return mu, cov


def get_mean_cov(
    returns: pd.DataFrame,
    lookback_days: int,
    estimator: str = "window",
    ewma_N: int = 60,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Dispatcher: choose window or EWMA estimator.

    Parameters
    ----------
    estimator : str
        "window" or "ewma".
    ewma_N : int
        EWMA parameter N (only used when estimator="ewma").
    """
    if estimator == "ewma":
        return estimate_ewma_mean_cov(returns, lookback_days, ewma_N)
    return estimate_window_mean_cov(returns, lookback_days)
