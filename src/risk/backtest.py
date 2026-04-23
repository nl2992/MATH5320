"""
backtest.py
Walk-forward VaR backtesting and Kupiec unconditional coverage test.

Spec §14 - 15:
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
    calibration_mode: str = "historical",
    manual_market_params: dict | None = None,
    option_vol_shock_mode: str = "fixed",
    option_vol_shock_beta: float = 1.0,
    option_vol_shock_floor: float = 0.05,
) -> pd.DataFrame:
    """Walk-forward VaR backtest with exception flagging.

    For each forecast date t in the test window:
        1. Fit the model on all prices up to and including t.
        2. Forecast VaR for a horizon of horizon_days trading days.
        3. Compute realized loss = V(t) − V(t + horizon_days).
        4. Flag exception = 1 if realized_loss > var_forecast.

    The test window starts at index lookback_days (so the first window has a
    full lookback) and ends at len(dates) − horizon_days (so the last realized
    return window is complete).

    Args:
        portfolio (Portfolio): Stock and option positions to backtest.
        prices (pd.DataFrame): Full aligned price history (DatetimeIndex × tickers).
            Must cover at least lookback_days + horizon_days + 1 rows.
        pricing_date (date): Fallback valuation date for option time-to-maturity
            (used when a per-row date cannot be inferred from the index).
        lookback_days (int): Estimation window size in trading days.
        horizon_days (int): VaR forecast horizon in trading days (≥ 1).
        var_confidence (float): VaR tail probability, e.g. 0.99 for 99% VaR.
        model (str): Risk model to use. One of ``"historical"``, ``"parametric"``,
            or ``"monte_carlo"``.
        estimator (str): ``"window"`` or ``"ewma"`` - mean/cov estimator for
            parametric and Monte Carlo models.
        ewma_N (int): EWMA half-life parameter N; only used when estimator=``"ewma"``.
        n_simulations (int): MC paths per forecast; only used when model=``"monte_carlo"``.
        calibration_mode (str): ``"historical"`` or ``"manual"``.
        manual_market_params (dict | None): Required when calibration_mode=``"manual"``.
        option_vol_shock_mode (str): ``"fixed"`` or ``"underlying_beta"``.
        option_vol_shock_beta (float): Beta for vol shock mode.
        option_vol_shock_floor (float): Minimum shocked vol.

    Returns:
        pd.DataFrame: One row per completed forecast date with columns:
            - ``"date"`` (pd.Timestamp): Forecast date.
            - ``"var_forecast"`` (float): Predicted VaR in dollars.
            - ``"realized_loss"`` (float): Actual loss V(t) − V(t+h) in dollars.
            - ``"exception"`` (int): 1 if realized_loss > var_forecast, else 0.

        If the backtest window is empty (not enough data), returns an empty
        DataFrame with ``df.attrs["reason"]`` explaining why.
        Skipped forecast dates (individual failures) are recorded in
        ``df.attrs["skipped_forecasts"]``, ``df.attrs["n_skipped_forecasts"]``,
        and ``df.attrs["skipped_forecast_dates"]``.
    """
    log_ret = compute_log_returns(prices)
    dates = log_ret.index  # dates for which we have a return

    underlyings = _portfolio_underlyings(portfolio)
    underlyings = [u for u in underlyings if u in prices.columns]

    records = []
    skipped_forecasts: list[dict[str, object]] = []
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
                calibration_mode=calibration_mode,
                manual_market_params=manual_market_params,
                option_vol_shock_mode=option_vol_shock_mode,
                option_vol_shock_beta=option_vol_shock_beta,
                option_vol_shock_floor=option_vol_shock_floor,
            )
        except Exception as exc:
            skipped_forecasts.append(
                {
                    "date": t_date,
                    "error": str(exc),
                    "model": model,
                }
            )
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
        realized_returns = {}
        for u in underlyings:
            s0 = float(spots_t[u])
            s1 = float(spots_t_h[u])
            realized_returns[u] = float(np.log(s1 / s0))
        V_t_h = portfolio_value(
            portfolio,
            spots_t_h,
            t_plus_h_date.date() if hasattr(t_plus_h_date, "date") else pricing_date,
            underlying_returns=realized_returns,
            option_vol_shock_mode=option_vol_shock_mode,
            option_vol_shock_beta=option_vol_shock_beta,
            option_vol_shock_floor=option_vol_shock_floor,
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

    backtest_df = pd.DataFrame(records)
    backtest_df.attrs["skipped_forecasts"] = skipped_forecasts
    backtest_df.attrs["n_skipped_forecasts"] = len(skipped_forecasts)
    backtest_df.attrs["skipped_forecast_dates"] = [
        item["date"] for item in skipped_forecasts
    ]
    if backtest_df.empty and skipped_forecasts:
        backtest_df.attrs["reason"] = (
            "Backtest produced no completed forecasts because every forecast date "
            "failed. Inspect skipped_forecasts for the underlying errors."
        )
    return backtest_df


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
    calibration_mode: str,
    manual_market_params: dict | None,
    option_vol_shock_mode: str,
    option_vol_shock_beta: float,
    option_vol_shock_floor: float,
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
            option_vol_shock_mode=option_vol_shock_mode,
            option_vol_shock_beta=option_vol_shock_beta,
            option_vol_shock_floor=option_vol_shock_floor,
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
            calibration_mode=calibration_mode,
            manual_market_params=manual_market_params,
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
            calibration_mode=calibration_mode,
            manual_market_params=manual_market_params,
            option_vol_shock_mode=option_vol_shock_mode,
            option_vol_shock_beta=option_vol_shock_beta,
            option_vol_shock_floor=option_vol_shock_floor,
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
    """Kupiec proportions-of-failures (POF) unconditional coverage test.

    Tests H0: the true exception rate equals the nominal rate (1 − var_confidence).
    The likelihood-ratio statistic LR_uc ~ χ²(1) under H0.

    Args:
        n_observations (int): Total number of VaR forecasts in the backtest window (N).
        n_exceptions (int): Observed number of exceptions (x), where exception means
            realized_loss > VaR_forecast.
        var_confidence (float): VaR confidence level, e.g. 0.99 for 99% VaR.
            The expected exception rate is α = 1 − var_confidence.

    Returns:
        dict: Test results with keys:
            - ``"alpha"`` (float): Expected exception rate (1 − var_confidence).
            - ``"p_hat"`` (float): Observed exception rate x / N.
            - ``"lr_stat"`` (float): Kupiec LR test statistic (≥ 0).
            - ``"p_value"`` (float): p-value under χ²(1); small = reject.
            - ``"reject_h0"`` (bool): True if p_value < 0.05.
            - ``"n_observations"`` (int): N passed in.
            - ``"n_exceptions"`` (int): x passed in.

        If N = 0, all numeric results are NaN and reject_h0 is False.
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

# ── Christoffersen Independence + Conditional Coverage Tests ──────────────────

def christoffersen_test(
    exceptions: "np.ndarray | list[int]",
) -> dict:
    """Christoffersen (1998) independence test for VaR exception clustering.

    Tests H0: consecutive exception indicators are i.i.d. Bernoulli (no
    clustering). The LR_ind statistic ~ χ²(1) under H0.

    Args:
        exceptions (array-like of int): Sequence of 0/1 exception indicators
            ordered by date, as produced by the ``"exception"`` column of
            ``run_backtest``.

    Returns:
        dict: Test results with keys:
            - ``"n00"``, ``"n01"``, ``"n10"``, ``"n11"`` (int): Transition counts.
            - ``"pi_01"`` (float): P(exception | previous non-exception).
            - ``"pi_11"`` (float): P(exception | previous exception).
            - ``"pi_hat"`` (float): Unconditional exception rate.
            - ``"lr_ind"`` (float): Independence LR statistic (≥ 0).
            - ``"p_value_ind"`` (float): p-value under χ²(1).
            - ``"reject_independence"`` (bool): True if p_value_ind < 0.05.

        Returns NaN statistics if fewer than 2 observations are provided.
    """
    exc = np.asarray(exceptions, dtype=int)
    n = len(exc)
    if n < 2:
        return {
            "n00": 0, "n01": 0, "n10": 0, "n11": 0,
            "pi_01": np.nan, "pi_11": np.nan, "pi_hat": np.nan,
            "lr_ind": np.nan,
            "p_value_ind": np.nan,
            "reject_independence": False,
        }

    # Count transitions: I_{t-1}=i  →  I_t=j
    n00 = int(((exc[:-1] == 0) & (exc[1:] == 0)).sum())
    n01 = int(((exc[:-1] == 0) & (exc[1:] == 1)).sum())
    n10 = int(((exc[:-1] == 1) & (exc[1:] == 0)).sum())
    n11 = int(((exc[:-1] == 1) & (exc[1:] == 1)).sum())

    eps = 1e-10

    # Conditional exception rates
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe_log(x):
        return np.log(np.clip(x, eps, 1 - eps))

    # L_A: alternative (heterogeneous transition matrix)
    log_la = (
        n00 * _safe_log(1 - pi_01)
        + n01 * _safe_log(pi_01)
        + n10 * _safe_log(1 - pi_11)
        + n11 * _safe_log(pi_11)
    )
    # L_0: null (independent, single-parameter pi_hat)
    log_l0 = (
        (n00 + n10) * _safe_log(1 - pi_hat)
        + (n01 + n11) * _safe_log(pi_hat)
    )

    lr_ind = float(max(-2.0 * (log_l0 - log_la), 0.0))
    p_value_ind = float(chi2.sf(lr_ind, df=1))

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": pi_01,
        "pi_11": pi_11,
        "pi_hat": pi_hat,
        "lr_ind": lr_ind,
        "p_value_ind": p_value_ind,
        "reject_independence": p_value_ind < 0.05,
    }


def conditional_coverage_test(
    n_observations: int,
    n_exceptions: int,
    var_confidence: float,
    exceptions: "np.ndarray | list[int]",
) -> dict:
    """Christoffersen (1998) conditional-coverage test (joint UC + independence).

    Combines the Kupiec unconditional-coverage LR_uc ~ χ²(1) and the
    Christoffersen independence LR_ind ~ χ²(1) into:
        LR_cc = LR_uc + LR_ind ~ χ²(2) under H0.

    Args:
        n_observations (int): Total forecast count (N).
        n_exceptions (int): Total exception count (x).
        var_confidence (float): VaR confidence level.
        exceptions (array-like of int): Ordered 0/1 exception sequence.

    Returns:
        dict: Merged result of ``kupiec_test`` and ``christoffersen_test``, plus:
            - ``"lr_cc"`` (float): Combined conditional-coverage LR statistic.
            - ``"p_value_cc"`` (float): p-value under χ²(2).
            - ``"reject_cc"`` (bool): True if p_value_cc < 0.05.
    """
    kupiec = kupiec_test(n_observations, n_exceptions, var_confidence)
    christo = christoffersen_test(exceptions)

    lr_uc = kupiec.get("lr_stat", np.nan)
    lr_ind = christo.get("lr_ind", np.nan)

    if np.isnan(lr_uc) or np.isnan(lr_ind):
        lr_cc = np.nan
        p_value_cc = np.nan
        reject_cc = False
    else:
        lr_cc = float(lr_uc + lr_ind)
        p_value_cc = float(chi2.sf(lr_cc, df=2))
        reject_cc = p_value_cc < 0.05

    return {
        **kupiec,
        **{k: v for k, v in christo.items() if k not in kupiec},
        "lr_cc": lr_cc,
        "p_value_cc": p_value_cc,
        "reject_cc": reject_cc,
    }


# ── Basel Traffic-Light Classification ────────────────────────────────────────

def basel_traffic_light(n_exceptions: int) -> dict:
    """Basel II/III traffic-light zone and capital multiplier for a VaR model.

    Classifies model quality based on exception count in a 250-day window:
        0 - 4  → GREEN  (model acceptable, multiplier 3.00)
        5 - 9  → YELLOW (model under scrutiny, multiplier 3.40 - 3.85)
        10+  → RED    (model rejected, multiplier 4.00)

    Args:
        n_exceptions (int): Number of VaR exceptions observed in the 250-day
            backtesting window (must be ≥ 0).

    Returns:
        dict: Classification with keys:
            - ``"n_exceptions"`` (int): Input exception count.
            - ``"zone"`` (str): ``"GREEN"``, ``"YELLOW"``, or ``"RED"``.
            - ``"capital_multiplier"`` (float): Basel capital add-on multiplier.
            - ``"description"`` (str): Plain-English interpretation.

    Raises:
        ValueError: If n_exceptions < 0.
    """
    _yellow_multipliers = {
        5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85,
    }

    if n_exceptions < 0:
        raise ValueError(f"n_exceptions must be non-negative (got {n_exceptions}).")

    if n_exceptions <= 4:
        zone = "GREEN"
        multiplier = 3.00
        description = "Model acceptable; no capital add-on."
    elif n_exceptions <= 9:
        zone = "YELLOW"
        multiplier = _yellow_multipliers[n_exceptions]
        description = "Model under scrutiny; capital multiplier increased."
    else:
        zone = "RED"
        multiplier = 4.00
        description = "Model rejected; internal model approval revoked."

    return {
        "n_exceptions": n_exceptions,
        "zone": zone,
        "capital_multiplier": multiplier,
        "description": description,
    }


# ── Exception Severity Diagnostics ───────────────────────────────────────────

def exception_severity(backtest_df: pd.DataFrame) -> dict:
    """Summary statistics describing how bad the VaR exceptions were.

    Supplements the exception count with gap and loss severity metrics, which
    are useful for understanding whether exceptions were marginal or extreme.

    Args:
        backtest_df (pd.DataFrame): Output of ``run_backtest`` containing at
            minimum the columns ``"var_forecast"``, ``"realized_loss"``,
            and ``"exception"`` (0/1 integer).

    Returns:
        dict: Severity statistics with keys:
            - ``"n_observations"`` (int): Total forecast rows.
            - ``"n_exceptions"`` (int): Number of exception days.
            - ``"exception_rate"`` (float): n_exceptions / n_observations.
            - ``"exception_gap"`` (float): Mean of (realized_loss − var_forecast)
              on exception days - how far losses exceeded the VaR forecast.
            - ``"average_exception_loss"`` (float): Mean realized_loss on
              exception days.
            - ``"max_exception_loss"`` (float): Worst realized_loss on any
              exception day.
            - ``"mean_loss_given_exception"`` (float): Alias for
              average_exception_loss.

        Returns zeros / NaN for all numeric fields if backtest_df is empty.
    """
    if backtest_df.empty:
        return {
            "n_observations": 0,
            "n_exceptions": 0,
            "exception_rate": np.nan,
            "exception_gap": np.nan,
            "average_exception_loss": np.nan,
            "max_exception_loss": np.nan,
            "mean_loss_given_exception": np.nan,
        }

    n_obs = len(backtest_df)
    exc_rows = backtest_df[backtest_df["exception"] == 1]
    n_exc = len(exc_rows)

    exception_rate = n_exc / n_obs if n_obs > 0 else np.nan

    if n_exc == 0:
        return {
            "n_observations": n_obs,
            "n_exceptions": 0,
            "exception_rate": 0.0,
            "exception_gap": np.nan,
            "average_exception_loss": np.nan,
            "max_exception_loss": np.nan,
            "mean_loss_given_exception": np.nan,
        }

    excess_losses = exc_rows["realized_loss"] - exc_rows["var_forecast"]
    exception_gap = float(excess_losses.mean())
    avg_exc_loss = float(exc_rows["realized_loss"].mean())
    max_exc_loss = float(exc_rows["realized_loss"].max())

    return {
        "n_observations": n_obs,
        "n_exceptions": n_exc,
        "exception_rate": exception_rate,
        "exception_gap": exception_gap,
        "average_exception_loss": avg_exc_loss,
        "max_exception_loss": max_exc_loss,
        "mean_loss_given_exception": avg_exc_loss,
    }
