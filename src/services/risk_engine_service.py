"""
risk_engine_service.py
Orchestration layer: coordinates all risk computations.
Streamlit calls only this service; it never calls risk modules directly.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.validation import validate_history_requirements, validate_price_dataframe
from src.portfolio.portfolio import portfolio_value
from src.risk.backtest import (
    basel_traffic_light,
    conditional_coverage_test,
    exception_severity,
    kupiec_test,
    run_backtest,
)
from src.risk.historical import historical_var_es
from src.risk.monte_carlo import monte_carlo_var_es
from src.risk.parametric import parametric_var_es
from src.schemas import Portfolio


class RiskEngineService:
    """Stateless service that orchestrates all VaR/ES and backtest calculations.

    Streamlit UI panels call only this class; they never import risk modules
    directly.  All three VaR models (historical simulation, parametric
    delta-normal, Monte Carlo full repricing) share a single set of
    parameters captured at construction time.

    Args:
        portfolio (Portfolio): Stock and option positions to be risk-managed.
        prices (pd.DataFrame): Price history; DatetimeIndex × ticker columns.
        pricing_date (date): Valuation date for Black-Scholes option pricing.
        lookback_days (int): Trailing price rows to use for return estimation.
        horizon_days (int): Risk horizon h in trading days.
        var_confidence (float): VaR confidence level, e.g. 0.99.
        es_confidence (float): ES averaging threshold, e.g. 0.975.
        estimator (str): ``"window"`` or ``"ewma"`` for mean/cov estimation.
        ewma_N (int): EWMA half-life parameter; only used when
            estimator=``"ewma"`` (default 60).
        n_simulations (int): Monte Carlo path count (default 10 000).
        calibration_mode (str): ``"historical"`` or ``"manual"``; see
            ``manual_market_params``.
        manual_market_params (dict | None): Required when
            calibration_mode=``"manual"``; see :func:`manual_mean_cov`.
        option_vol_shock_mode (str): ``"fixed"`` or ``"underlying_beta"``.
        option_vol_shock_beta (float): Beta for ``"underlying_beta"`` mode.
        option_vol_shock_floor (float): Minimum post-shock option volatility.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        prices: pd.DataFrame,
        pricing_date: date,
        lookback_days: int,
        horizon_days: int,
        var_confidence: float,
        es_confidence: float,
        estimator: str = "window",
        ewma_N: int = 60,
        n_simulations: int = 10_000,
        calibration_mode: str = "historical",
        manual_market_params: dict | None = None,
        option_vol_shock_mode: str = "fixed",
        option_vol_shock_beta: float = 1.0,
        option_vol_shock_floor: float = 0.05,
    ):
        self.portfolio = portfolio
        self.prices = prices
        self.pricing_date = pricing_date
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.var_confidence = var_confidence
        self.es_confidence = es_confidence
        self.estimator = estimator
        self.ewma_N = ewma_N
        self.n_simulations = n_simulations
        self.calibration_mode = calibration_mode
        self.manual_market_params = manual_market_params
        self.option_vol_shock_mode = option_vol_shock_mode
        self.option_vol_shock_beta = option_vol_shock_beta
        self.option_vol_shock_floor = option_vol_shock_floor

    # ── Current portfolio value ────────────────────────────────────────────────

    def portfolio_value(self) -> float:
        """Return the current mark-to-market portfolio value.

        Uses the last row of ``self.prices`` as the spot-price vector.

        Returns:
            float: Portfolio value in dollars; may be negative for a net-short
                book.
        """
        spots = self.prices.iloc[-1]
        return portfolio_value(self.portfolio, spots, self.pricing_date)

    # ── Main risk run ──────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        """Run all three VaR/ES models and return a unified results dict.

        Validates price history before delegating to :func:`historical_var_es`,
        :func:`parametric_var_es`, and :func:`monte_carlo_var_es`.

        Returns:
            dict: Results with keys ``"historical"``, ``"parametric"``, and
                ``"monte_carlo"``.  Each sub-dict contains at minimum:
                    - ``"var"`` (float): VaR in dollars.
                    - ``"es"`` (float): ES in dollars.

        Raises:
            ValueError: If the price DataFrame fails validation, or the
                history is too short for the requested lookback + horizon.
        """
        price_errors = validate_price_dataframe(self.prices)
        if price_errors:
            raise ValueError("Invalid price history: " + " | ".join(price_errors))
        history_errors = validate_history_requirements(
            self.prices,
            lookback_days=self.lookback_days,
            horizon_days=self.horizon_days,
            for_backtest=False,
        )
        if history_errors:
            raise ValueError("Invalid risk run request: " + " | ".join(history_errors))

        hist = historical_var_es(
            portfolio=self.portfolio,
            prices=self.prices,
            pricing_date=self.pricing_date,
            lookback_days=self.lookback_days,
            horizon_days=self.horizon_days,
            var_confidence=self.var_confidence,
            es_confidence=self.es_confidence,
            option_vol_shock_mode=self.option_vol_shock_mode,
            option_vol_shock_beta=self.option_vol_shock_beta,
            option_vol_shock_floor=self.option_vol_shock_floor,
        )

        param = parametric_var_es(
            portfolio=self.portfolio,
            prices=self.prices,
            pricing_date=self.pricing_date,
            lookback_days=self.lookback_days,
            horizon_days=self.horizon_days,
            var_confidence=self.var_confidence,
            es_confidence=self.es_confidence,
            estimator=self.estimator,
            ewma_N=self.ewma_N,
            calibration_mode=self.calibration_mode,
            manual_market_params=self.manual_market_params,
        )

        mc = monte_carlo_var_es(
            portfolio=self.portfolio,
            prices=self.prices,
            pricing_date=self.pricing_date,
            lookback_days=self.lookback_days,
            horizon_days=self.horizon_days,
            var_confidence=self.var_confidence,
            es_confidence=self.es_confidence,
            n_simulations=self.n_simulations,
            estimator=self.estimator,
            ewma_N=self.ewma_N,
            calibration_mode=self.calibration_mode,
            manual_market_params=self.manual_market_params,
            option_vol_shock_mode=self.option_vol_shock_mode,
            option_vol_shock_beta=self.option_vol_shock_beta,
            option_vol_shock_floor=self.option_vol_shock_floor,
        )

        return {
            "historical": hist,
            "parametric": param,
            "monte_carlo": mc,
        }

    # ── Backtesting ────────────────────────────────────────────────────────────

    def run_backtest(self, model: str = "historical") -> dict:
        """Run walk-forward VaR backtest, Kupiec POF test, and Basel traffic light.

        Args:
            model (str): VaR model to backtest — ``"historical"``,
                ``"parametric"``, or ``"monte_carlo"``.

        Returns:
            dict: Backtest output with keys:
                - ``"backtest_df"`` (pd.DataFrame): Per-date realized losses,
                  VaR forecasts, and exception flags.
                - ``"kupiec"`` (dict): Kupiec unconditional coverage test.
                - ``"conditional_coverage"`` (dict): Christoffersen CC test.
                - ``"basel"`` (dict | None): Basel traffic-light zone, or
                  None if the backtest produced no observations.
                - ``"severity"`` (dict): Exception severity statistics.
                - ``"model"`` (str): Model name echoed back.
                - ``"reason"`` (str | None): Reason string if backtest_df
                  is empty.
                - ``"skipped_forecasts"`` (list[dict]): Per-date failures.
                - ``"n_skipped_forecasts"`` (int): Count of skipped days.

        Raises:
            ValueError: If the price DataFrame fails validation.
        """
        price_errors = validate_price_dataframe(self.prices)
        if price_errors:
            raise ValueError("Invalid price history: " + " | ".join(price_errors))

        bt_df = run_backtest(
            portfolio=self.portfolio,
            prices=self.prices,
            pricing_date=self.pricing_date,
            lookback_days=self.lookback_days,
            horizon_days=self.horizon_days,
            var_confidence=self.var_confidence,
            model=model,
            estimator=self.estimator,
            ewma_N=self.ewma_N,
            n_simulations=min(self.n_simulations, 2_000),  # faster for backtest
            calibration_mode=self.calibration_mode,
            manual_market_params=self.manual_market_params,
            option_vol_shock_mode=self.option_vol_shock_mode,
            option_vol_shock_beta=self.option_vol_shock_beta,
            option_vol_shock_floor=self.option_vol_shock_floor,
        )

        if bt_df.empty:
            kupiec = {
                "alpha": 1 - self.var_confidence,
                "p_hat": float("nan"),
                "lr_stat": float("nan"),
                "p_value": float("nan"),
                "reject_h0": False,
                "n_observations": 0,
                "n_exceptions": 0,
            }
            conditional_coverage = {
                **kupiec,
                "n00": 0,
                "n01": 0,
                "n10": 0,
                "n11": 0,
                "pi_01": float("nan"),
                "pi_11": float("nan"),
                "pi_hat": float("nan"),
                "lr_ind": float("nan"),
                "p_value_ind": float("nan"),
                "reject_independence": False,
                "lr_cc": float("nan"),
                "p_value_cc": float("nan"),
                "reject_cc": False,
            }
            basel = None
        else:
            kupiec = kupiec_test(
                n_observations=len(bt_df),
                n_exceptions=int(bt_df["exception"].sum()),
                var_confidence=self.var_confidence,
            )
            conditional_coverage = conditional_coverage_test(
                n_observations=len(bt_df),
                n_exceptions=int(bt_df["exception"].sum()),
                var_confidence=self.var_confidence,
                exceptions=bt_df["exception"].to_numpy(),
            )
            basel = basel_traffic_light(int(bt_df["exception"].sum()))

        severity = exception_severity(bt_df)

        return {
            "backtest_df": bt_df,
            "kupiec": kupiec,
            "conditional_coverage": conditional_coverage,
            "basel": basel,
            "severity": severity,
            "model": model,
            "reason": bt_df.attrs.get("reason") if bt_df.empty else None,
            "skipped_forecasts": bt_df.attrs.get("skipped_forecasts", []),
            "n_skipped_forecasts": int(bt_df.attrs.get("n_skipped_forecasts", 0)),
        }
