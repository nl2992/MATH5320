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
    calibration_mode: str = "historical",
    manual_market_params: dict | None = None,
    option_vol_shock_mode: str = "fixed",
    option_vol_shock_beta: float = 1.0,
    option_vol_shock_floor: float = 0.05,
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
                calibration_mode=calibration_mode,
                manual_market_params=manual_market_params,
                option_vol_shock_mode=option_vol_shock_mode,
                option_vol_shock_beta=option_vol_shock_beta,
                option_vol_shock_floor=option_vol_shock_floor,
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

# ── Christoffersen Independence + Conditional Coverage Tests ──────────────────

def christoffersen_test(
    exceptions: "np.ndarray | list[int]",
) -> dict:
    """
    Christoffersen (1998) independence test for VaR exception clustering.

    Tests whether consecutive exceptions are i.i.d. Bernoulli, i.e. that
    exceptions are not clustered in time.

    The test statistic LR_ind ~ χ²(1) under H0 (independent exceptions).
    The combined conditional-coverage statistic LR_cc = LR_uc + LR_ind ~ χ²(2).

    Parameters
    ----------
    exceptions : array-like of 0/1
        Sequence of exception indicators from the backtest.

    Returns
    -------
    dict with keys:
        n00, n01, n10, n11  : int  — transition counts
        pi_01, pi_11        : float — conditional exception rates
        pi_hat              : float — unconditional exception rate
        lr_ind              : float — independence LR statistic ~ χ²(1)
        p_value_ind         : float — p-value for independence test
        reject_independence : bool  — True if H0 rejected at 5%
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
    """
    Christoffersen (1998) conditional-coverage test.

    Combines the Kupiec unconditional-coverage test (LR_uc ~ χ²(1)) and the
    independence test (LR_ind ~ χ²(1)) into a joint test:
        LR_cc = LR_uc + LR_ind ~ χ²(2)

    Returns a dict merging the Kupiec and Christoffersen results, plus:
        lr_cc        : float — combined statistic
        p_value_cc   : float — p-value under χ²(2)
        reject_cc    : bool  — True if H0 rejected at 5%
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
    """
    Basel II/III traffic-light classification of VaR model quality.

    Based on the number of exceptions in a 250-day window:
        0–4  : GREEN  — model likely adequate
        5–9  : YELLOW — model under scrutiny; capital multiplier raised
        10+  : RED    — model rejected; internal model approval revoked

    The capital multiplier m_c is given by the Basel table:
        GREEN  : m_c = 3.00
        YELLOW : m_c ranges from 3.40 (5 exceptions) to 3.85 (9 exceptions)
        RED    : m_c = 4.00

    Parameters
    ----------
    n_exceptions : int
        Number of VaR exceptions in the 250-day observation window.

    Returns
    -------
    dict with keys:
        n_exceptions      : int
        zone              : str  — "GREEN", "YELLOW", or "RED"
        capital_multiplier: float
        description       : str
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
    """
    Compute summary statistics for VaR exceptions.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``run_backtest`` with columns:
        ``var_forecast``, ``realized_loss``, ``exception``.

    Returns
    -------
    dict with keys:
        n_observations          : int
        n_exceptions            : int
        exception_rate          : float  — observed / expected ratio
        expected_exceptions     : float  — always = n_obs * (1 - conf) but
                                           conf is inferred from mean VaR
                                           sign pattern; here raw count
        exception_gap           : float  — mean (loss − VaR) across exceptions
        average_exception_loss  : float  — mean realized_loss on exception days
        max_exception_loss      : float  — worst realized loss on exception days
        mean_loss_given_exception: float — average_exception_loss (alias)
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
