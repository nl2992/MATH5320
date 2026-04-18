"""
test_config_and_validation.py
Smoke tests for src/config.py and src/data/validation.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pandas as pd
import pytest

import src.config as cfg
from src.data.validation import validate_portfolio_tickers, validate_price_dataframe
from src.schemas import OptionPosition, Portfolio, StockPosition


# ── config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults_loaded(self):
        assert cfg.DEFAULT_LOOKBACK_DAYS == 252
        assert cfg.DEFAULT_HORIZON_DAYS == 1
        assert cfg.DEFAULT_VAR_CONFIDENCE == 0.99
        assert cfg.DEFAULT_ES_CONFIDENCE == 0.975
        assert cfg.DEFAULT_ESTIMATOR == "window"
        assert cfg.DEFAULT_EWMA_N == 60
        assert cfg.DEFAULT_MC_SIMULATIONS == 10_000
        assert cfg.DEFAULT_YFINANCE_PERIOD == "5y"
        assert cfg.TRADING_DAYS_PER_YEAR == 252
        assert cfg.DEFAULT_BACKTEST_MODEL == "historical"
        assert cfg.MIN_BACKTEST_OBSERVATIONS == 30


# ── validate_price_dataframe ──────────────────────────────────────────────────

class TestValidatePrices:
    def test_empty_returns_error(self):
        errors = validate_price_dataframe(pd.DataFrame())
        assert any("empty" in e.lower() for e in errors)

    def test_none_returns_error(self):
        errors = validate_price_dataframe(None)
        assert any("empty" in e.lower() for e in errors)

    def test_non_datetime_index(self):
        df = pd.DataFrame({"AAPL": [100, 101]}, index=[0, 1])
        errors = validate_price_dataframe(df)
        assert any("DatetimeIndex" in e for e in errors)

    def test_column_all_nan(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        df = pd.DataFrame({"AAPL": [100.0, 101.0], "GOOG": [float("nan"), float("nan")]}, index=idx)
        errors = validate_price_dataframe(df)
        assert any("all NaN" in e for e in errors)

    def test_non_positive_prices(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        df = pd.DataFrame({"AAPL": [100.0, -1.0]}, index=idx)
        errors = validate_price_dataframe(df)
        assert any("non-positive" in e for e in errors)

    def test_good_frame_no_errors(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=idx)
        assert validate_price_dataframe(df) == []


# ── validate_portfolio_tickers ────────────────────────────────────────────────

class TestValidatePortfolioTickers:
    def test_all_present(self):
        pf = Portfolio(
            stocks=[StockPosition("AAPL", 10)],
            options=[OptionPosition(
                ticker="o1", underlying_ticker="MSFT", option_type="call",
                quantity=1, strike=400, maturity_date=date.today() + timedelta(days=30),
                volatility=0.25, risk_free_rate=0.04,
            )],
        )
        errors = validate_portfolio_tickers(pf, ["AAPL", "MSFT"])
        assert errors == []

    def test_missing_stock_ticker(self):
        pf = Portfolio(stocks=[StockPosition("ZZZ", 1)], options=[])
        errors = validate_portfolio_tickers(pf, ["AAPL"])
        assert any("ZZZ" in e for e in errors)

    def test_missing_option_underlying(self):
        pf = Portfolio(
            stocks=[],
            options=[OptionPosition(
                ticker="o1", underlying_ticker="ZZZ", option_type="call",
                quantity=1, strike=400, maturity_date=date.today() + timedelta(days=30),
                volatility=0.25, risk_free_rate=0.04,
            )],
        )
        errors = validate_portfolio_tickers(pf, ["AAPL"])
        assert any("ZZZ" in e and "underlying" in e for e in errors)
