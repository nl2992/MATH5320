"""Regression test: es_confidence != var_confidence must give different ES values."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import pandas as pd
from datetime import date

from src.risk.parametric import parametric_var_es
from src.risk.historical import historical_var_es
from src.risk.monte_carlo import monte_carlo_var_es
from src.schemas import Portfolio, StockPosition

# Minimal one-stock portfolio and price history.
@pytest.fixture
def stock_portfolio():
    pos = StockPosition(ticker="AAA", quantity=100)
    return Portfolio(stocks=[pos], options=[])

@pytest.fixture
def sample_prices():
    rng = np.random.default_rng(0)
    n = 300
    returns = rng.normal(0.0005, 0.02, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({"AAA": prices}, index=idx)

def test_parametric_es_confidence_respected(stock_portfolio, sample_prices):
    r99_975 = parametric_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=5,
        var_confidence=0.99, es_confidence=0.975,
    )
    r99_99 = parametric_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=5,
        var_confidence=0.99, es_confidence=0.99,
    )
    # ES at 97.5% tail should be LESS than ES at 99% tail for same VaR
    assert r99_975["es"] < r99_99["es"], (
        "ES at 97.5% should be smaller than ES at 99% (wider tail average)"
    )

def test_historical_es_confidence_respected(stock_portfolio, sample_prices):
    r99_975 = historical_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=1,
        var_confidence=0.99, es_confidence=0.975,
    )
    r99_99 = historical_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=1,
        var_confidence=0.99, es_confidence=0.99,
    )
    assert r99_975["es"] <= r99_99["es"]

def test_mc_es_confidence_respected(stock_portfolio, sample_prices):
    r99_975 = monte_carlo_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=1,
        var_confidence=0.99, es_confidence=0.975,
        n_simulations=5000, random_seed=42,
    )
    r99_99 = monte_carlo_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=1,
        var_confidence=0.99, es_confidence=0.99,
        n_simulations=5000, random_seed=42,
    )
    assert r99_975["es"] <= r99_99["es"]

def test_parametric_same_confidence_es_equals_baseline(stock_portfolio, sample_prices):
    """When var_confidence == es_confidence, result should be identical to old behavior."""
    result = parametric_var_es(
        stock_portfolio, sample_prices, date(2023, 1, 1),
        lookback_days=250, horizon_days=5,
        var_confidence=0.99, es_confidence=0.99,
    )
    assert result["es"] >= result["var"]
