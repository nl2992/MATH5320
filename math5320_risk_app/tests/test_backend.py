"""
test_backend.py
End-to-end smoke tests for the MATH5320 risk backend.
Run from the project root: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from src.schemas import StockPosition, OptionPosition, Portfolio
from src.pricing.black_scholes import bs_price, bs_delta
from src.portfolio.portfolio import portfolio_value, portfolio_exposure
from src.risk.returns import compute_log_returns, build_overlapping_horizon_log_returns
from src.risk.estimators import estimate_window_mean_cov, estimate_ewma_mean_cov
from src.risk.historical import historical_var_es
from src.risk.parametric import parametric_var_es
from src.risk.monte_carlo import monte_carlo_var_es
from src.risk.backtest import kupiec_test, run_backtest
from src.services.risk_engine_service import RiskEngineService


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """Synthetic price history: 2 stocks, 500 days."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    aapl = 150 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    msft = 300 * np.exp(np.cumsum(np.random.normal(0.0004, 0.018, n)))
    return pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)


@pytest.fixture
def simple_portfolio():
    return Portfolio(
        stocks=[
            StockPosition(ticker="AAPL", quantity=100),
            StockPosition(ticker="MSFT", quantity=50),
        ],
        options=[],
    )


@pytest.fixture
def option_portfolio():
    maturity = date.today() + timedelta(days=90)
    return Portfolio(
        stocks=[StockPosition(ticker="AAPL", quantity=100)],
        options=[
            OptionPosition(
                ticker="AAPL_CALL",
                underlying_ticker="AAPL",
                option_type="call",
                quantity=10,
                strike=160.0,
                maturity_date=maturity,
                volatility=0.25,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                contract_multiplier=100,
            )
        ],
    )


# ── Black-Scholes tests ────────────────────────────────────────────────────────

def test_bs_call_price_positive():
    price = bs_price(S=100, K=100, T=1, r=0.05, q=0, sigma=0.2, option_type="call")
    assert price > 0

def test_bs_put_call_parity():
    S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.0, 0.2
    call = bs_price(S, K, T, r, q, sigma, "call")
    put  = bs_price(S, K, T, r, q, sigma, "put")
    # C - P = S*e^{-qT} - K*e^{-rT}
    import math
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8

def test_bs_call_delta_between_0_and_1():
    delta = bs_delta(S=100, K=100, T=1, r=0.05, q=0, sigma=0.2, option_type="call")
    assert 0 < delta < 1

def test_bs_put_delta_between_minus1_and_0():
    delta = bs_delta(S=100, K=100, T=1, r=0.05, q=0, sigma=0.2, option_type="put")
    assert -1 < delta < 0


# ── Portfolio valuation tests ──────────────────────────────────────────────────

def test_portfolio_value_stocks_only(sample_prices, simple_portfolio):
    spots = sample_prices.iloc[-1]
    pv = portfolio_value(simple_portfolio, spots, date.today())
    expected = 100 * spots["AAPL"] + 50 * spots["MSFT"]
    assert abs(pv - expected) < 1e-6

def test_portfolio_exposure_stocks_only(sample_prices, simple_portfolio):
    spots = sample_prices.iloc[-1]
    exp = portfolio_exposure(simple_portfolio, spots, date.today())
    assert abs(exp["AAPL"] - 100 * spots["AAPL"]) < 1e-6
    assert abs(exp["MSFT"] - 50 * spots["MSFT"]) < 1e-6


# ── Returns tests ──────────────────────────────────────────────────────────────

def test_log_returns_shape(sample_prices):
    lr = compute_log_returns(sample_prices)
    assert len(lr) == len(sample_prices) - 1

def test_overlapping_horizon_returns(sample_prices):
    lr = compute_log_returns(sample_prices)
    h5 = build_overlapping_horizon_log_returns(lr, 5)
    assert len(h5) == len(lr) - 4  # 5-1 = 4 rows dropped


# ── Estimator tests ────────────────────────────────────────────────────────────

def test_window_estimator_shape(sample_prices):
    lr = compute_log_returns(sample_prices)
    mu, cov = estimate_window_mean_cov(lr, 252)
    assert len(mu) == 2
    assert cov.shape == (2, 2)

def test_ewma_estimator_shape(sample_prices):
    lr = compute_log_returns(sample_prices)
    mu, cov = estimate_ewma_mean_cov(lr, 252, N=60)
    assert len(mu) == 2
    assert cov.shape == (2, 2)

def test_ewma_cov_positive_definite(sample_prices):
    lr = compute_log_returns(sample_prices)
    _, cov = estimate_ewma_mean_cov(lr, 252, N=60)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert np.all(eigenvalues > -1e-10)


# ── Historical VaR/ES tests ────────────────────────────────────────────────────

def test_historical_var_es_positive(sample_prices, simple_portfolio):
    result = historical_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
    )
    assert result["var"] > 0
    assert result["es"] >= result["var"]

def test_historical_var_es_with_option(sample_prices, option_portfolio):
    result = historical_var_es(
        portfolio=option_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
    )
    assert result["var"] > 0


# ── Parametric VaR/ES tests ────────────────────────────────────────────────────

def test_parametric_var_es_positive(sample_prices, simple_portfolio):
    result = parametric_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
    )
    assert result["var"] > 0
    assert result["es"] >= result["var"]

def test_parametric_es_greater_than_var(sample_prices, simple_portfolio):
    result = parametric_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
    )
    assert result["es"] > result["var"]


# ── Monte Carlo VaR/ES tests ───────────────────────────────────────────────────

def test_mc_var_es_positive(sample_prices, simple_portfolio):
    result = monte_carlo_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        n_simulations=2000,
        random_seed=42,
    )
    assert result["var"] > 0
    assert result["es"] >= result["var"]


# ── Kupiec test ────────────────────────────────────────────────────────────────

def test_kupiec_perfect_model():
    """If exception rate exactly matches alpha, LR stat should be ~0."""
    result = kupiec_test(n_observations=1000, n_exceptions=10, var_confidence=0.99)
    assert result["lr_stat"] >= 0
    assert 0 <= result["p_value"] <= 1

def test_kupiec_bad_model():
    """If exception rate is way too high, should reject H0."""
    result = kupiec_test(n_observations=250, n_exceptions=50, var_confidence=0.99)
    assert result["reject_h0"] is True


# ── Service layer test ─────────────────────────────────────────────────────────

def test_service_run_all(sample_prices, simple_portfolio):
    service = RiskEngineService(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        n_simulations=1000,
    )
    results = service.run_all()
    assert "historical" in results
    assert "parametric" in results
    assert "monte_carlo" in results
    for model in results.values():
        assert model["var"] > 0
        assert model["es"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
