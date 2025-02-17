"""
backtest.py
Walk-forward VaR backtesting and Kupiec unconditional coverage test.

Spec §14–15:
    Walk-forward algorithm:
        For each time t:
            Fit model using data up to t
            Forecast VaR
            Compute realized loss from t to t+h
            exception = 1 if loss > VaR

    Kupiec test:
        α = 1 − confidence
        p̂ = exceptions / observations
        LR_uc = -2 [log L0 − log L1]
        L0 = (1-α)^(N-x) α^x
        L1 = (1-p̂)^(N-x) p̂^x
        Test statistic ~ χ²(1)
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.portfolio.portfolio import portfolio_value
from src.risk.estimators import get_mean_cov
from src.risk.historical import _compute_losses, _portfolio_underlyings
from src.risk.parametric import parametric_var_es
from src.risk.returns import (
    build_overlapping_horizon_log_returns,
    compute_log_returns,
)
from src.schemas import Portfolio


# ── Walk-forward backtesting ───────────────────────────────────────────────────

def run_backtest(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    pricing_date: date,
    lookback_days: int,
    horizon_days: int,
    var_confidence: float,
    model: str = "historical",
    estimator: str = "window",
    ewma_N: int = 60,
    n_simulations: int = 2_000,
) -> pd.DataFrame:
    """
    Walk-forward VaR backtest.

    For each date t in the test window:
        - Fit the model on prices up to and including t.
        - Forecast 1-step (horizon_days) VaR.
        - Realised loss = V(t) − V(t + horizon_days).
        - Exception = 1 if realised loss > VaR.

    The test window starts at index (lookback_days + horizon_days) so that
    both the estimation window and the realised return window are fully available.

    Parameters
    ----------
    model : str
        "historical" | "parametric" | "monte_carlo"

    Returns
    -------
    pd.DataFrame with columns:
        date, var_forecast, realized_loss, exception
    """
    log_ret = compute_log_returns(prices)
    dates = log_ret.index  # dates for which we have a return

    underlyings = _portfolio_underlyings(portfolio)
    underlyings = [u for u in underlyings if u in prices.columns]

    records = []
    # We need at least lookback_days of history before t,
    # and horizon_days of future returns after t.
    start_idx = lookback_days
    end_idx = len(dates) - horizon_days

    if end_idx <= start_idx:
        empty = pd.DataFrame(columns=["date", "var_forecast", "realized_loss", "exception"])
        empty.attrs["reason"] = (
            f"Backtest window empty: need at least "
            f"{lookback_days + horizon_days + 1} trading days of history, "
            f"got {len(dates) + 1}. Reduce lookback_days or horizon_days, "
            f"or load more history."
        )
        return empty

    for i in range(start_idx, end_idx):
        t_date = dates[i]

        # Prices available up to and including t
        prices_up_to_t = prices.loc[prices.index <= t_date]

        # Forecast VaR
        try:
            var_forecast = _forecast_var(
                portfolio=portfolio,
                prices=prices_up_to_t,
                pricing_date=t_date.date() if hasattr(t_date, "date") else pricing_date,
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                var_confidence=var_confidence,
                model=model,
                estimator=estimator,
                ewma_N=ewma_N,
                n_simulations=n_simulations,
            )
        except Exception:
            continue

        # Realised loss: portfolio value at t vs t+horizon
        spots_t = prices_up_to_t.iloc[-1]
        t_plus_h_date = dates[i + horizon_days]
        spots_t_h = prices.loc[t_plus_h_date]

        V_t = portfolio_value(
            portfolio,
            spots_t,
            t_date.date() if hasattr(t_date, "date") else pricing_date,
        )
        V_t_h = portfolio_value(
            portfolio,
            spots_t_h,
            t_plus_h_date.date() if hasattr(t_plus_h_date, "date") else pricing_date,
        )
        realized_loss = V_t - V_t_h  # loss = V0 - V_T

        exception = int(realized_loss > var_forecast)

        records.append(
            {
                "date": t_date,
                "var_forecast": var_forecast,
                "realized_loss": realized_loss,
                "exception": exception,
            }
        )

    return pd.DataFrame(records)


def _forecast_var(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    pricing_date: date,
    lookback_days: int,
    horizon_days: int,
    var_confidence: float,
    model: str,
    estimator: str,
    ewma_N: int,
    n_simulations: int,
) -> float:
    """Compute a single VaR forecast using the chosen model."""
    if model == "historical":
        from src.risk.historical import historical_var_es
        result = historical_var_es(
            portfolio=portfolio,
            prices=prices,
            pricing_date=pricing_date,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            var_confidence=var_confidence,
            es_confidence=var_confidence,
        )
        return result["var"]

    elif model == "parametric":
        result = parametric_var_es(
            portfolio=portfolio,
            prices=prices,
            pricing_date=pricing_date,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            var_confidence=var_confidence,
            es_confidence=var_confidence,
            estimator=estimator,
            ewma_N=ewma_N,
        )
        return result["var"]

    elif model == "monte_carlo":
        from src.risk.monte_carlo import monte_carlo_var_es
        result = monte_carlo_var_es(
            portfolio=portfolio,
            prices=prices,
            pricing_date=pricing_date,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            var_confidence=var_confidence,
            es_confidence=var_confidence,
            n_simulations=n_simulations,
            estimator=estimator,
            ewma_N=ewma_N,
            random_seed=None,
        )
        return result["var"]

    else:
        raise ValueError(f"Unknown backtest model: '{model}'")


# ── Kupiec Unconditional Coverage Test ────────────────────────────────────────

def kupiec_test(
    n_observations: int,
    n_exceptions: int,
    var_confidence: float,
) -> dict:
    """
    Kupiec proportions-of-failures (POF) test.

    Parameters
    ----------
    n_observations : int
        Total number of VaR forecasts (N).
    n_exceptions : int
        Number of exceptions (x).
    var_confidence : float
        VaR confidence level, e.g. 0.99.

    Returns
    -------
    dict with keys:
        alpha            : float — expected exception rate
        p_hat            : float — observed exception rate
        lr_stat          : float — likelihood-ratio test statistic
        p_value          : float — p-value under χ²(1)
        reject_h0        : bool  — True if H0 rejected at 5% level
        n_observations   : int
        n_exceptions     : int
    """
    alpha = 1.0 - var_confidence  # expected exception rate
    N = n_observations
    x = n_exceptions

    if N == 0:
        return {
            "alpha": alpha,
            "p_hat": np.nan,
            "lr_stat": np.nan,
            "p_value": np.nan,
            "reject_h0": False,
            "n_observations": N,
            "n_exceptions": x,
        }

    p_hat = x / N if N > 0 else 0.0

    # Avoid log(0) edge cases
    eps = 1e-10

    def log_likelihood(p: float) -> float:
        p = np.clip(p, eps, 1 - eps)
        return (N - x) * np.log(1 - p) + x * np.log(p)

    L0 = log_likelihood(alpha)
    L1 = log_likelihood(p_hat)

    lr_stat = -2.0 * (L0 - L1)
    lr_stat = max(lr_stat, 0.0)  # numerical safety

    p_value = float(chi2.sf(lr_stat, df=1))
    reject_h0 = p_value < 0.05

    return {
        "alpha": alpha,
        "p_hat": p_hat,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_h0": reject_h0,
        "n_observations": N,
        "n_exceptions": x,
    }
