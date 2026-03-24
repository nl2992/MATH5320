"""
test_numerical_precision.py
Floating-point failure mode tests for the MATH5320 risk engine.

Covers IEEE 754 edge cases: overflow/underflow in Black-Scholes,
catastrophic cancellation in log returns, near-singular Cholesky,
EWMA long-series stability, and extreme-confidence parametric VaR/ES.

Run from the project root:
    python -m pytest tests/test_numerical_precision.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.pricing.black_scholes import bs_price
from src.risk.returns import compute_log_returns
from src.risk.estimators import estimate_ewma_mean_cov
from src.risk.parametric import parametric_var_es
from src.schemas import StockPosition, Portfolio


# ── NP_01: BS underflow at extreme low volatility ─────────────────────────────

def test_NP_01_bs_underflow_low_vol():
    """BS call at sigma=1e-10: should return value close to intrinsic, not NaN/inf."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 1e-10, 1.0
    price = bs_price(S=S, K=K, T=T, r=r, q=0.0, sigma=sigma, option_type="call")
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    assert math.isfinite(price), "BS price is not finite at sigma=1e-10"
    assert price >= 0.0, "BS price is negative"
    assert abs(price - intrinsic) < 0.01, (
        f"BS call price {price:.6f} deviates from intrinsic {intrinsic:.6f} "
        f"by more than 0.01 at sigma=1e-10"
    )


# ── NP_02: BS overflow guard at extreme high volatility ───────────────────────

def test_NP_02_bs_overflow_high_vol():
    """BS call at sigma=50: should return a finite positive number not exceeding S."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 50.0, 1.0
    price = bs_price(S=S, K=K, T=T, r=r, q=0.0, sigma=sigma, option_type="call")
    assert math.isfinite(price), "BS price is not finite at sigma=50"
    assert price > 0.0, "BS price is non-positive at sigma=50"
    # As sigma -> inf, call -> S (spot); allow small margin
    assert price <= S * 1.01, (
        f"BS call {price:.4f} exceeds S={S} * 1.01 at sigma=50"
    )


# ── NP_03: BS with near-zero time to expiry ───────────────────────────────────

def test_NP_03_bs_near_zero_T():
    """BS call at T=1e-8 (deep ITM): should be close to intrinsic ~10, finite."""
    S, K, r, sigma, T = 110.0, 100.0, 0.05, 0.2, 1e-8
    price = bs_price(S=S, K=K, T=T, r=r, q=0.0, sigma=sigma, option_type="call")
    assert math.isfinite(price), "BS price is not finite at T=1e-8"
    # Intrinsic for ITM call with near-zero T = S - K ~ 10
    assert abs(price - 10.0) < 0.01, (
        f"BS call {price:.6f} deviates from intrinsic 10.0 by more than 0.01"
    )


# ── NP_04: Log-return catastrophic cancellation ───────────────────────────────

def test_NP_04_log_return_near_zero_increment():
    """Log returns for prices very close together should be finite and nonzero."""
    epsilon = 1e-10
    prices = pd.DataFrame(
        {"AAPL": [100.0, 100.0 + epsilon, 100.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="B"),
    )
    log_ret = compute_log_returns(prices)
    vals = log_ret["AAPL"].values
    assert np.all(np.isfinite(vals)), "Log returns contain NaN or inf"
    # The first return ln(100+1e-10) - ln(100) should be ~1e-12, nonzero
    first_ret = vals[0]
    assert first_ret != 0.0, "First log return is exactly zero (should be ~1e-12)"
    # Absolute value should be roughly epsilon/100 by first-order approximation
    assert abs(first_ret) < 1e-6, (
        f"First log return {first_ret:.2e} unexpectedly large"
    )


# ── NP_05: Cholesky on near-singular covariance ───────────────────────────────

def test_NP_05_near_singular_covariance_var():
    """Near-singular covariance (cond ~ 1e12) should not crash; VaR should be finite."""
    # Build a 3x3 covariance with two nearly identical assets
    # sigma1 = 0.02, sigma2 = 0.02 + 1e-6, sigma3 = 0.03; high correlation
    s1, s2, s3 = 0.02, 0.02 + 1e-6, 0.03
    rho = 0.9999  # near-singular between asset 1 and 2
    cov_vals = np.array([
        [s1**2,          rho * s1 * s2,  0.3 * s1 * s3],
        [rho * s1 * s2,  s2**2,          0.3 * s2 * s3],
        [0.3 * s1 * s3,  0.3 * s2 * s3,  s3**2        ],
    ])
    tickers = ["A", "B", "C"]

    # Generate synthetic price history consistent with this covariance
    np.random.seed(99)
    n = 300
    # Use Cholesky of the cov; if singular, add small ridge
    try:
        L = np.linalg.cholesky(cov_vals)
    except np.linalg.LinAlgError:
        cov_vals += 1e-8 * np.eye(3)
        L = np.linalg.cholesky(cov_vals)

    raw_returns = np.random.randn(n, 3) @ L.T
    prices_arr = 100.0 * np.exp(np.cumsum(raw_returns, axis=0))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    prices_df = pd.DataFrame(prices_arr, index=dates, columns=tickers)

    portfolio = Portfolio(
        stocks=[
            StockPosition(ticker="A", quantity=100),
            StockPosition(ticker="B", quantity=100),
            StockPosition(ticker="C", quantity=50),
        ],
        options=[],
    )

    result = parametric_var_es(
        portfolio=portfolio,
        prices=prices_df,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
    )
    assert math.isfinite(result["var"]), "VaR is not finite on near-singular covariance"
    assert result["var"] > 0, "VaR is non-positive on near-singular covariance"


# ── NP_06: EWMA stability over 2000-day series ────────────────────────────────

def test_NP_06_ewma_long_series_stability():
    """EWMA over 2000-day series should produce finite, symmetric, PSD covariance."""
    np.random.seed(7)
    n = 2000
    dates = pd.date_range("2016-01-01", periods=n, freq="B")
    prices_arr = np.column_stack([
        150.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n))),
        300.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.018, n))),
    ])
    prices_df = pd.DataFrame(prices_arr, index=dates, columns=["X", "Y"])
    log_ret = compute_log_returns(prices_df)

    mu, cov = estimate_ewma_mean_cov(log_ret, lookback_days=2000, N=60)

    assert np.all(np.isfinite(mu.values)), "EWMA mean contains non-finite values"
    assert np.all(np.isfinite(cov.values)), "EWMA covariance contains non-finite values"

    # Symmetry
    assert np.allclose(cov.values, cov.values.T, atol=1e-12), (
        "EWMA covariance is not symmetric"
    )

    # PSD: all eigenvalues >= 0 (allowing tiny numerical negatives)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert np.all(eigvals >= -1e-10), (
        f"EWMA covariance has negative eigenvalue: {eigvals.min():.2e}"
    )


# ── NP_07: VaR/ES at extreme confidence (99.99%) ─────────────────────────────

def test_NP_07_parametric_var_extreme_confidence():
    """Parametric VaR/ES at alpha=0.9999 should be finite, positive, ES >= VaR."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    aapl = 150.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    msft = 300.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.018, n)))
    prices_df = pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)

    portfolio = Portfolio(
        stocks=[
            StockPosition(ticker="AAPL", quantity=100),
            StockPosition(ticker="MSFT", quantity=50),
        ],
        options=[],
    )

    result = parametric_var_es(
        portfolio=portfolio,
        prices=prices_df,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.9999,
        es_confidence=0.9999,
    )
    assert math.isfinite(result["var"]), "VaR at alpha=0.9999 is not finite"
    assert math.isfinite(result["es"]), "ES at alpha=0.9999 is not finite"
    assert result["var"] > 0, "VaR at alpha=0.9999 is non-positive"
    assert result["es"] > 0, "ES at alpha=0.9999 is non-positive"
    assert result["es"] >= result["var"], (
        f"ES {result['es']:.4f} < VaR {result['var']:.4f} at alpha=0.9999"
    )
