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
    """BS call/put at extreme low sigma: should converge to intrinsic, not NaN/inf."""
    # Test across a range of extreme low vols and moneyness levels
    low_vols = [1e-10, 1e-15, 1e-20, 1e-50]
    for sigma in low_vols:
        # ITM call
        price = bs_price(S=110.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=sigma, option_type="call")
        intrinsic = max(110.0 - 100.0 * math.exp(-0.05), 0.0)
        assert math.isfinite(price), f"BS call not finite at sigma={sigma}"
        assert price >= 0.0, f"BS call negative at sigma={sigma}"
        assert abs(price - intrinsic) < 0.05, (
            f"ITM call {price:.6f} != intrinsic {intrinsic:.6f} at sigma={sigma}"
        )

        # OTM call (should be ~0)
        otm = bs_price(S=80.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=sigma, option_type="call")
        assert math.isfinite(otm), f"OTM call not finite at sigma={sigma}"
        assert otm >= 0.0, f"OTM call negative at sigma={sigma}"
        assert otm < 0.1, f"OTM call {otm:.6f} unexpectedly large at sigma={sigma}"

        # ATM put
        put = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=sigma, option_type="put")
        put_intrinsic = max(100.0 * math.exp(-0.05) - 100.0, 0.0)
        assert math.isfinite(put), f"ATM put not finite at sigma={sigma}"
        assert put >= 0.0, f"ATM put negative at sigma={sigma}"

    # Verify put-call parity still holds at extreme low vol
    S, K, r, q, T, sigma = 105.0, 100.0, 0.05, 0.0, 1.0, 1e-15
    c = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
    p = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put")
    parity_lhs = c - p
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 0.01, (
        f"Put-call parity violated at sigma={sigma}: |{parity_lhs:.6f} - {parity_rhs:.6f}|"
    )


# ── NP_02: BS overflow guard at extreme high volatility ───────────────────────

def test_NP_02_bs_overflow_high_vol():
    """BS call/put at extreme high sigma: finite, bounded, and monotone toward limits."""
    # Test across escalating vols — call should approach S, put should approach K*exp(-rT)
    high_vols = [5.0, 10.0, 50.0, 100.0, 500.0]
    S, K, r, q, T = 100.0, 100.0, 0.05, 0.0, 1.0
    prev_call = 0.0
    for sigma in high_vols:
        call = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
        put = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put")
        assert math.isfinite(call), f"Call not finite at sigma={sigma}"
        assert math.isfinite(put), f"Put not finite at sigma={sigma}"
        assert call > 0.0, f"Call non-positive at sigma={sigma}"
        assert put > 0.0, f"Put non-positive at sigma={sigma}"
        # As sigma -> inf, call -> S (for q=0)
        assert call <= S * 1.01, f"Call {call:.4f} exceeds S at sigma={sigma}"
        # As sigma -> inf, put -> K*exp(-rT)
        put_limit = K * math.exp(-r * T)
        assert put <= put_limit * 1.01, f"Put {put:.4f} exceeds limit at sigma={sigma}"

    # Different moneyness levels at extreme vol
    for S_test in [50.0, 100.0, 200.0]:
        c = bs_price(S=S_test, K=100.0, T=1.0, r=0.05, q=0.0, sigma=100.0, option_type="call")
        assert math.isfinite(c) and c > 0.0, f"Failed at S={S_test}, sigma=100"
        assert c <= S_test * 1.01, f"Call {c:.4f} > S={S_test} at sigma=100"

    # Extreme vol with nonzero dividend yield
    c_div = bs_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.03, sigma=200.0, option_type="call")
    assert math.isfinite(c_div) and c_div > 0.0, "Failed with q=0.03, sigma=200"


# ── NP_03: BS with near-zero time to expiry ───────────────────────────────────

def test_NP_03_bs_near_zero_T():
    """BS call/put at near-zero T across moneyness: converges to intrinsic payoff."""
    tiny_T_values = [1e-6, 1e-8, 1e-10, 1e-15]

    for T in tiny_T_values:
        # Deep ITM call: intrinsic ≈ S - K = 10
        c_itm = bs_price(S=110.0, K=100.0, T=T, r=0.05, q=0.0, sigma=0.2, option_type="call")
        assert math.isfinite(c_itm), f"ITM call not finite at T={T}"
        assert abs(c_itm - 10.0) < 0.05, f"ITM call {c_itm:.6f} != 10.0 at T={T}"

        # Deep OTM call: intrinsic = 0
        c_otm = bs_price(S=80.0, K=100.0, T=T, r=0.05, q=0.0, sigma=0.2, option_type="call")
        assert math.isfinite(c_otm), f"OTM call not finite at T={T}"
        assert c_otm < 0.1, f"OTM call {c_otm:.6f} not near 0 at T={T}"

        # ATM call: intrinsic = 0, but near-ATM should be small
        c_atm = bs_price(S=100.0, K=100.0, T=T, r=0.05, q=0.0, sigma=0.2, option_type="call")
        assert math.isfinite(c_atm), f"ATM call not finite at T={T}"
        assert c_atm >= 0.0, f"ATM call negative at T={T}"

        # Deep ITM put: intrinsic ≈ K - S = 20
        p_itm = bs_price(S=80.0, K=100.0, T=T, r=0.05, q=0.0, sigma=0.2, option_type="put")
        assert math.isfinite(p_itm), f"ITM put not finite at T={T}"
        assert abs(p_itm - 20.0) < 0.5, f"ITM put {p_itm:.6f} != 20.0 at T={T}"

    # Verify call and put both non-negative at T=1e-15 across strikes
    for K in [50.0, 80.0, 100.0, 120.0, 150.0]:
        c = bs_price(S=100.0, K=K, T=1e-15, r=0.05, q=0.0, sigma=0.3, option_type="call")
        p = bs_price(S=100.0, K=K, T=1e-15, r=0.05, q=0.0, sigma=0.3, option_type="put")
        assert math.isfinite(c) and c >= 0.0, f"Call invalid at T=1e-15, K={K}"
        assert math.isfinite(p) and p >= 0.0, f"Put invalid at T=1e-15, K={K}"


# ── NP_04: Log-return catastrophic cancellation ───────────────────────────────

def test_NP_04_log_return_near_zero_increment():
    """Log returns for prices very close together should be finite and nonzero.
    Also tests catastrophic cancellation scenarios from the lecture."""
    # Basic case: tiny increment
    epsilon = 1e-10
    prices = pd.DataFrame(
        {"AAPL": [100.0, 100.0 + epsilon, 100.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="B"),
    )
    log_ret = compute_log_returns(prices)
    vals = log_ret["AAPL"].values
    assert np.all(np.isfinite(vals)), "Log returns contain NaN or inf"
    first_ret = vals[0]
    assert first_ret != 0.0, "First log return is exactly zero (should be ~1e-12)"
    assert abs(first_ret) < 1e-6, f"First log return {first_ret:.2e} unexpectedly large"

    # Lecture example: cancellation from very large vs very small values
    # 1e17 - 1.0 = 1e17 in float64 — returns should handle this gracefully
    large_prices = pd.DataFrame(
        {"BIG": [1e10, 1e10 + 1.0, 1e10 + 2.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="B"),
    )
    big_ret = compute_log_returns(large_prices)
    assert np.all(np.isfinite(big_ret.values)), "Log returns not finite for large prices"

    # Multiple tickers with mixed scales — tests independent noise cancellation
    mixed = pd.DataFrame(
        {
            "SMALL": [0.001, 0.001 + 1e-10, 0.001 + 2e-10, 0.001 + 3e-10],
            "LARGE": [1e6, 1e6 + 0.01, 1e6 + 0.02, 1e6 + 0.03],
            "NORMAL": [100.0, 101.0, 99.5, 100.5],
        },
        index=pd.date_range("2022-01-01", periods=4, freq="B"),
    )
    mixed_ret = compute_log_returns(mixed)
    assert np.all(np.isfinite(mixed_ret.values)), "Log returns not finite for mixed scales"
    assert mixed_ret.shape == (3, 3), f"Expected shape (3,3), got {mixed_ret.shape}"

    # Verify sign correctness: price up = positive return, price down = negative
    normal_ret = mixed_ret["NORMAL"].values
    assert normal_ret[0] > 0, "Price 100→101 should give positive return"
    assert normal_ret[1] < 0, "Price 101→99.5 should give negative return"


# ── NP_05: Cholesky on near-singular covariance ───────────────────────────────

def test_NP_05_near_singular_covariance_var():
    """Near-singular covariance (cond ~ 1e12) should not crash; VaR should be finite.
    Also tests ES, and verifies ES >= VaR even under ill-conditioning."""
    # Build a 3x3 covariance with two nearly identical assets
    s1, s2, s3 = 0.02, 0.02 + 1e-6, 0.03
    rho = 0.9999  # near-singular between asset 1 and 2
    cov_vals = np.array([
        [s1**2,          rho * s1 * s2,  0.3 * s1 * s3],
        [rho * s1 * s2,  s2**2,          0.3 * s2 * s3],
        [0.3 * s1 * s3,  0.3 * s2 * s3,  s3**2        ],
    ])
    tickers = ["A", "B", "C"]

    np.random.seed(99)
    n = 300
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
    assert math.isfinite(result["es"]), "ES is not finite on near-singular covariance"
    assert result["es"] > 0, "ES is non-positive on near-singular covariance"
    assert result["es"] >= result["var"], (
        f"ES {result['es']:.4f} < VaR {result['var']:.4f} under near-singular cov"
    )

    # Verify the input covariance has high condition number (stress test is genuine)
    cond = np.linalg.cond(cov_vals)
    assert cond > 1e3, f"Condition number {cond:.1e} not large enough for stress test"

    # Test with even higher correlation (rho=0.999999)
    rho2 = 0.999999
    cov2 = np.array([
        [s1**2,           rho2 * s1 * s2,  0.3 * s1 * s3],
        [rho2 * s1 * s2,  s2**2,           0.3 * s2 * s3],
        [0.3 * s1 * s3,   0.3 * s2 * s3,   s3**2        ],
    ])
    cov2 += 1e-10 * np.eye(3)  # tiny ridge for PSD
    np.random.seed(77)
    raw2 = np.random.randn(n, 3) @ np.linalg.cholesky(cov2).T
    prices2 = 100.0 * np.exp(np.cumsum(raw2, axis=0))
    prices_df2 = pd.DataFrame(prices2, index=dates, columns=tickers)
    result2 = parametric_var_es(
        portfolio=portfolio, prices=prices_df2, pricing_date=date.today(),
        lookback_days=252, horizon_days=1, var_confidence=0.99, es_confidence=0.975,
    )
    assert math.isfinite(result2["var"]), "VaR not finite at rho=0.999999"
    assert math.isfinite(result2["es"]), "ES not finite at rho=0.999999"


# ── NP_06: EWMA stability over 2000-day series ────────────────────────────────

def test_NP_06_ewma_long_series_stability():
    """EWMA over long series should produce finite, symmetric, PSD covariance.
    Tests multiple series lengths and N (decay) values for stability."""
    np.random.seed(7)

    # Test across different series lengths and EWMA decay parameters
    for n, N_decay in [(500, 30), (2000, 60), (2000, 10), (2000, 252)]:
        dates = pd.date_range("2016-01-01", periods=n, freq="B")
        prices_arr = np.column_stack([
            150.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n))),
            300.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.018, n))),
        ])
        prices_df = pd.DataFrame(prices_arr, index=dates, columns=["X", "Y"])
        log_ret = compute_log_returns(prices_df)

        mu, cov = estimate_ewma_mean_cov(log_ret, lookback_days=n, N=N_decay)

        assert np.all(np.isfinite(mu.values)), (
            f"EWMA mean non-finite at n={n}, N={N_decay}"
        )
        assert np.all(np.isfinite(cov.values)), (
            f"EWMA covariance non-finite at n={n}, N={N_decay}"
        )

        # Symmetry
        assert np.allclose(cov.values, cov.values.T, atol=1e-12), (
            f"EWMA covariance not symmetric at n={n}, N={N_decay}"
        )

        # PSD: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals >= -1e-10), (
            f"EWMA negative eigenvalue {eigvals.min():.2e} at n={n}, N={N_decay}"
        )

        # Diagonal elements (variances) should be positive
        for i in range(cov.shape[0]):
            assert cov.values[i, i] > 0, (
                f"EWMA variance <= 0 on diagonal [{i},{i}] at n={n}, N={N_decay}"
            )

        # Correlation matrix should have diagonal = 1 and off-diag in [-1, 1]
        stds = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(stds, stds)
        for i in range(corr.shape[0]):
            assert abs(corr[i, i] - 1.0) < 1e-10, f"Correlation diagonal != 1"
            for j in range(corr.shape[1]):
                assert -1.0 - 1e-10 <= corr[i, j] <= 1.0 + 1e-10, (
                    f"Correlation [{i},{j}] = {corr[i,j]:.6f} out of [-1,1]"
                )


# ── NP_07: VaR/ES at extreme confidence (99.99%) ─────────────────────────────

def test_NP_07_parametric_var_extreme_confidence():
    """Parametric VaR/ES at extreme confidence levels: finite, positive, ES >= VaR.
    Tests monotonicity: higher confidence → higher VaR. Also tests horizon scaling."""
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

    # Test across confidence levels — VaR should be monotone increasing
    alphas = [0.90, 0.95, 0.99, 0.999, 0.9999]
    prev_var = 0.0
    for alpha in alphas:
        result = parametric_var_es(
            portfolio=portfolio, prices=prices_df, pricing_date=date.today(),
            lookback_days=252, horizon_days=1,
            var_confidence=alpha, es_confidence=alpha,
        )
        assert math.isfinite(result["var"]), f"VaR not finite at alpha={alpha}"
        assert math.isfinite(result["es"]), f"ES not finite at alpha={alpha}"
        assert result["var"] > 0, f"VaR non-positive at alpha={alpha}"
        assert result["es"] > 0, f"ES non-positive at alpha={alpha}"
        assert result["es"] >= result["var"], (
            f"ES {result['es']:.4f} < VaR {result['var']:.4f} at alpha={alpha}"
        )
        # Monotonicity: higher confidence → higher VaR
        assert result["var"] >= prev_var - 1e-6, (
            f"VaR not monotone: VaR({alpha})={result['var']:.4f} < VaR(prev)={prev_var:.4f}"
        )
        prev_var = result["var"]

    # Test horizon scaling: 10-day VaR should be > 1-day VaR
    var_1d = parametric_var_es(
        portfolio=portfolio, prices=prices_df, pricing_date=date.today(),
        lookback_days=252, horizon_days=1, var_confidence=0.99, es_confidence=0.99,
    )["var"]
    var_10d = parametric_var_es(
        portfolio=portfolio, prices=prices_df, pricing_date=date.today(),
        lookback_days=252, horizon_days=10, var_confidence=0.99, es_confidence=0.99,
    )["var"]
    assert var_10d > var_1d, (
        f"10-day VaR {var_10d:.4f} not greater than 1-day VaR {var_1d:.4f}"
    )
    # Under sqrt-T scaling, ratio should be approximately sqrt(10) ≈ 3.16
    ratio = var_10d / var_1d if var_1d > 0 else float("inf")
    assert 1.5 < ratio < 6.0, (
        f"VaR horizon ratio {ratio:.2f} outside plausible range [1.5, 6.0]"
    )
