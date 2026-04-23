"""
estimators.py
Mean and covariance estimators: rolling window and EWMA.

Two EWMA lambda conventions are implemented:

  Spec convention (active, used throughout):
    λ = (N - 1) / (N + 1)          → _ewma_lambda(N)
    Effective exponential memory: (N+1)/2 observations.
    For default N=60: λ ≈ 0.9672.

  Course / textbook convention (reference, not used in production):
    λ = 1 - 1/N                     → _ewma_lambda_course(N)
    Effective exponential memory: N observations.
    For default N=60: λ ≈ 0.9833.

All production paths call _ewma_lambda (spec convention).
_ewma_lambda_course is provided as a standalone reference and is not
wired into get_mean_cov or estimate_ewma_mean_cov.

EWMA weights:
    w_i = (1 - λ) λ^(n-i)   for i = 1 ... n   (oldest to newest)
    Weights are then normalised to sum to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def manual_mean_cov(
    manual_market_params: dict,
    underlyings: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Extract and validate manually supplied daily mean and covariance.

    Allows the user to bypass historical estimation and supply μ and Σ directly
    (e.g. from an external calibration or course homework inputs).

    Args:
        manual_market_params (dict): Must contain two keys:
            - ``"mu_daily"``: daily mean log-return vector, indexable by ticker
              (pd.Series or dict or array-like that pd.Series accepts).
            - ``"cov_daily"``: daily covariance matrix, indexable by ticker
              (pd.DataFrame or dict of dicts).
        underlyings (list[str]): Ordered list of ticker symbols required. The
            returned μ and Σ are aligned to this order.

    Returns:
        tuple[pd.Series, pd.DataFrame]: (mu, cov) - daily mean vector and
            daily covariance matrix, both aligned to ``underlyings``.

    Raises:
        ValueError: If manual_market_params is not a dict, is missing required
            keys, is missing any ticker in ``underlyings``, contains non-finite
            values, is asymmetric, has negative diagonal entries, or fails the
            positive-semidefinite check.
    """
    if not isinstance(manual_market_params, dict):
        raise ValueError("manual_market_params must be a dict.")
    if "mu_daily" not in manual_market_params or "cov_daily" not in manual_market_params:
        raise ValueError("manual_market_params must contain 'mu_daily' and 'cov_daily'.")

    mu = pd.Series(manual_market_params["mu_daily"], dtype=float)
    cov = pd.DataFrame(manual_market_params["cov_daily"], dtype=float)

    missing = [u for u in underlyings if u not in mu.index or u not in cov.index or u not in cov.columns]
    if missing:
        raise ValueError(
            "manual_market_params missing one or more portfolio underlyings: "
            + ", ".join(missing)
        )

    mu = mu.loc[underlyings]
    cov = cov.loc[underlyings, underlyings]

    if not np.isfinite(mu.values).all():
        raise ValueError("manual mean vector contains non-finite values.")
    if not np.isfinite(cov.values).all():
        raise ValueError("manual covariance matrix contains non-finite values.")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("manual covariance matrix must be square.")
    if not np.allclose(cov.values, cov.values.T, atol=1e-10):
        raise ValueError("manual covariance matrix must be symmetric.")
    if (np.diag(cov.values) < 0).any():
        raise ValueError("manual covariance matrix has negative diagonal entries.")

    eigvals = np.linalg.eigvalsh(cov.values)
    if np.min(eigvals) < -1e-10:
        raise ValueError("manual covariance matrix must be positive semidefinite.")

    return mu, cov


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
    """Compute EWMA decay parameter using the project specification convention.

    Formula (from spec): λ = (N - 1) / (N + 1)

    Effective exponential memory is (N+1)/2 observations.
    For N=60: λ ≈ 0.9672.

    This is the convention used throughout the production codebase.
    See also _ewma_lambda_course for the textbook alternative.
    """
    return (N - 1) / (N + 1)


def _ewma_lambda_course(N: int) -> float:
    """Compute EWMA decay parameter using the course / textbook convention.

    Formula: λ = 1 - 1/N

    Under this convention, N equals the effective exponential memory
    1/(1−λ), i.e. the approximate number of observations with material
    weight.  For N=60: λ ≈ 0.9833.

    This function is provided as a reference implementation of the textbook
    form discussed in the course notes.  It is NOT wired into
    estimate_ewma_mean_cov or get_mean_cov; all production paths use
    _ewma_lambda (the spec convention).  Call this function directly when
    you need the course-convention decay factor for comparison or
    stand-alone calculations.
    """
    return 1.0 - 1.0 / N


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
    """Dispatcher: estimate daily mean and covariance using the chosen method.

    Args:
        returns (pd.DataFrame): Full daily log-return history (all rows available).
            Only the last ``lookback_days`` rows are used.
        lookback_days (int): Number of trailing observations to include.
        estimator (str): ``"window"`` for simple rolling sample statistics,
            ``"ewma"`` for exponentially weighted estimates. Default ``"window"``.
        ewma_N (int): EWMA half-life parameter N; λ = (N−1)/(N+1).
            Only used when estimator=``"ewma"``. Default 60.

    Returns:
        tuple[pd.Series, pd.DataFrame]: (mu, cov) - daily mean log-return vector
            and daily covariance matrix, indexed by ticker.

    Raises:
        ValueError: If estimator is not ``"window"`` or ``"ewma"`` (raised by
            the downstream estimator functions).
    """
    if estimator == "ewma":
        return estimate_ewma_mean_cov(returns, lookback_days, ewma_N)
    return estimate_window_mean_cov(returns, lookback_days)
