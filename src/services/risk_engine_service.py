"""
risk_engine_service.py
Orchestration layer: coordinates all risk computations.
Streamlit calls only this service; it never calls risk modules directly.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.portfolio.portfolio import portfolio_value
from src.risk.backtest import kupiec_test, run_backtest
from src.risk.historical import historical_var_es
from src.risk.monte_carlo import monte_carlo_var_es
from src.risk.parametric import parametric_var_es
from src.schemas import Portfolio


class RiskEngineService:
    """
    Stateless service that runs risk calculations for a given portfolio
    and market data snapshot.
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

    # ── Current portfolio value ────────────────────────────────────────────────

    def portfolio_value(self) -> float:
        """Return current mark-to-market portfolio value."""
        spots = self.prices.iloc[-1]
        return portfolio_value(self.portfolio, spots, self.pricing_date)

    # ── Main risk run ──────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        """
        Run all three VaR/ES models and return a unified results dict.

        Returns
        -------
        dict with keys "historical", "parametric", "monte_carlo".
        Each sub-dict contains at minimum:
            var  : float
            es   : float
        """
        hist = historical_var_es(
            portfolio=self.portfolio,
            prices=self.prices,
            pricing_date=self.pricing_date,
            lookback_days=self.lookback_days,
            horizon_days=self.horizon_days,
            var_confidence=self.var_confidence,
            es_confidence=self.es_confidence,
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
        )

        return {
            "historical": hist,
            "parametric": param,
            "monte_carlo": mc,
        }

    # ── Backtesting ────────────────────────────────────────────────────────────

    def run_backtest(self, model: str = "historical") -> dict:
        """
        Run walk-forward VaR backtest and Kupiec test.

        Parameters
        ----------
        model : str
            "historical" | "parametric" | "monte_carlo"

        Returns
        -------
        dict with keys:
            backtest_df  : pd.DataFrame — per-date results
            kupiec       : dict         — Kupiec test results
            model        : str
        """
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
        else:
            kupiec = kupiec_test(
                n_observations=len(bt_df),
                n_exceptions=int(bt_df["exception"].sum()),
                var_confidence=self.var_confidence,
            )

        return {
            "backtest_df": bt_df,
            "kupiec": kupiec,
            "model": model,
            "reason": bt_df.attrs.get("reason") if bt_df.empty else None,
        }
