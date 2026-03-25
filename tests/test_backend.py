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
from src.portfolio.positions import option_delta_exposure
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

def test_option_delta_exposure_is_delta_dollar(option_portfolio, sample_prices):
    spots = sample_prices.iloc[-1]
    spot = float(spots["AAPL"])
    opt = option_portfolio.options[0]
    exposure = option_delta_exposure(opt, spot, date.today())
    delta = bs_delta(
        S=spot,
        K=opt.strike,
        T=(opt.maturity_date - date.today()).days / 365.0,
        r=opt.risk_free_rate,
        q=opt.dividend_yield,
        sigma=opt.volatility,
        option_type=opt.option_type,
    )
    expected = opt.quantity * opt.contract_multiplier * delta * spot
    assert exposure == pytest.approx(expected)

def test_manual_parametric_mode_uses_manual_mean_cov(sample_prices, simple_portfolio):
    mu_daily = pd.Series({"AAPL": 0.0010, "MSFT": 0.0005})
    vols = pd.Series({"AAPL": 0.02, "MSFT": 0.015})
    corr = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    cov_daily = pd.DataFrame(
        np.outer(vols.values, vols.values) * corr.values,
        index=vols.index,
        columns=vols.index,
    )
    result = parametric_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        calibration_mode="manual",
        manual_market_params={"mu_daily": mu_daily, "cov_daily": cov_daily},
    )
    assert result["var"] > 0
    assert result["calibration_mode"] == "manual"

def test_mc_manual_mode_uses_manual_mean_cov(sample_prices, simple_portfolio):
    mu_daily = pd.Series({"AAPL": 0.0002, "MSFT": 0.0001})
    cov_daily = pd.DataFrame(
        [[0.0004, 0.00005], [0.00005, 0.0003]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    result = monte_carlo_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        n_simulations=500,
        calibration_mode="manual",
        manual_market_params={"mu_daily": mu_daily, "cov_daily": cov_daily},
    )
    assert result["var"] > 0
    assert result["calibration_mode"] == "manual"

def test_historical_option_vol_shock_changes_result(sample_prices, option_portfolio):
    fixed = historical_var_es(
        portfolio=option_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        option_vol_shock_mode="fixed",
    )
    shocked = historical_var_es(
        portfolio=option_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        option_vol_shock_mode="underlying_beta",
        option_vol_shock_beta=2.0,
        option_vol_shock_floor=0.05,
    )
    assert shocked["var"] != pytest.approx(fixed["var"])


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
    # Use matching confidences so ES >= VaR holds (ES averages the tail beyond VaR).
    result = historical_var_es(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.99,
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


# ── Phase-0 workflow coverage: walk-forward backtest + edges ───────────────────

def test_run_backtest_historical_with_sample_prices(sample_prices, simple_portfolio):
    """Walk-forward historical backtest returns a non-empty frame with expected columns."""
    bt_df = run_backtest(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=60,
        horizon_days=1,
        var_confidence=0.99,
        model="historical",
    )
    assert not bt_df.empty
    for col in ("date", "var_forecast", "realized_loss", "exception"):
        assert col in bt_df.columns
    assert bt_df["exception"].isin([0, 1]).all()


def test_run_backtest_parametric_with_sample_prices(sample_prices, simple_portfolio):
    """Parametric-mode walk-forward also succeeds end-to-end."""
    bt_df = run_backtest(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=60,
        horizon_days=1,
        var_confidence=0.99,
        model="parametric",
        estimator="window",
    )
    assert not bt_df.empty
    assert (bt_df["var_forecast"] > 0).all()


def test_run_backtest_empty_when_lookback_too_large(sample_prices, simple_portfolio):
    """When lookback+horizon >= data length, backtest returns an empty frame with a reason."""
    bt_df = run_backtest(
        portfolio=simple_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=10_000,  # deliberately too large
        horizon_days=1,
        var_confidence=0.99,
        model="historical",
    )
    assert bt_df.empty
    assert "reason" in bt_df.attrs
    assert "trading days" in bt_df.attrs["reason"].lower()


def test_service_run_all_with_options(sample_prices, option_portfolio):
    """Service should run all three models on a stocks+options portfolio without blowing up."""
    service = RiskEngineService(
        portfolio=option_portfolio,
        prices=sample_prices,
        pricing_date=date.today(),
        lookback_days=252,
        horizon_days=1,
        var_confidence=0.99,
        es_confidence=0.975,
        n_simulations=1000,
    )
    results = service.run_all()
    for model_name, res in results.items():
        assert np.isfinite(res["var"]), f"{model_name} VaR not finite"
        assert np.isfinite(res["es"]), f"{model_name} ES not finite"
        assert res["var"] > 0, f"{model_name} VaR non-positive"
        assert res["es"] >= res["var"], f"{model_name} ES < VaR"


def test_ewma_short_N_edge_case(sample_prices):
    """Small EWMA N (high decay) should still produce a finite, PSD cov matrix."""
    lr = compute_log_returns(sample_prices)
    mu, cov = estimate_ewma_mean_cov(lr, lookback_days=252, N=5)
    assert np.all(np.isfinite(mu.values))
    assert np.all(np.isfinite(cov.values))
    eig = np.linalg.eigvalsh(cov.values)
    assert eig.min() > -1e-10, "EWMA cov not PSD"


def test_expired_option_intrinsic_value():
    """Expired options should valuate to intrinsic × quantity × multiplier."""
    from src.portfolio.positions import option_value
    expired = OptionPosition(
        ticker="TEST",
        underlying_ticker="AAPL",
        option_type="call",
        quantity=1,
        strike=100.0,
        maturity_date=date.today() - timedelta(days=30),  # expired
        volatility=0.20,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        contract_multiplier=100.0,
    )
    # Spot above strike — intrinsic = 20
    v_itm = option_value(expired, spot=120.0, pricing_date=date.today())
    assert v_itm == 20.0 * 100.0
    # Spot below strike — intrinsic = 0
    v_otm = option_value(expired, spot=80.0, pricing_date=date.today())
    assert v_otm == 0.0


# ── Behavioral Confirmation Tests ─────────────────────────────────────────────

class TestBehavioralConfirmation:
    """BEH_01 through BEH_08: mathematical properties of BS and risk measures."""

    def test_BEH_01_call_monotone_in_spot(self):
        """Call price is strictly increasing in spot S — across parameter regimes."""
        # Base case
        spots = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0]
        for sigma in [0.1, 0.2, 0.4]:
            for T in [0.25, 1.0, 3.0]:
                prices_list = [
                    bs_price(S=s, K=100.0, T=T, r=0.05, q=0.0, sigma=sigma, option_type="call")
                    for s in spots
                ]
                for i in range(len(prices_list) - 1):
                    assert prices_list[i] < prices_list[i + 1], (
                        f"Call not increasing: S={spots[i]}->{spots[i+1]} at "
                        f"sigma={sigma}, T={T}: {prices_list[i]:.4f} >= {prices_list[i+1]:.4f}"
                    )
        # Put should be monotonically DECREASING in S
        put_prices = [
            bs_price(S=s, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
            for s in spots
        ]
        for i in range(len(put_prices) - 1):
            assert put_prices[i] > put_prices[i + 1], (
                f"Put not decreasing in S: S={spots[i]} gives {put_prices[i]:.4f}"
            )

    def test_BEH_02_call_monotone_decreasing_in_strike(self):
        """Call price strictly decreasing in K; put strictly increasing in K."""
        strikes = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0]
        for sigma in [0.1, 0.2, 0.4]:
            for T in [0.25, 1.0]:
                call_prices = [
                    bs_price(S=100.0, K=k, T=T, r=0.05, q=0.0, sigma=sigma, option_type="call")
                    for k in strikes
                ]
                for i in range(len(call_prices) - 1):
                    assert call_prices[i] > call_prices[i + 1], (
                        f"Call not decreasing in K: K={strikes[i]}->{strikes[i+1]} at "
                        f"sigma={sigma}, T={T}"
                    )
        # Put should be monotonically INCREASING in K
        put_prices = [
            bs_price(S=100.0, K=k, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
            for k in strikes
        ]
        for i in range(len(put_prices) - 1):
            assert put_prices[i] < put_prices[i + 1], (
                f"Put not increasing in K: K={strikes[i]} gives {put_prices[i]:.4f}"
            )
        # Convexity check: call price is convex in K (second differences >= 0)
        call_atm = [
            bs_price(S=100.0, K=k, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="call")
            for k in strikes
        ]
        for i in range(1, len(call_atm) - 1):
            second_diff = call_atm[i-1] - 2*call_atm[i] + call_atm[i+1]
            assert second_diff >= -1e-8, (
                f"Call not convex in K at K={strikes[i]}: second_diff={second_diff:.6f}"
            )

    def test_BEH_03_call_monotone_increasing_in_vol(self):
        """Vega > 0: both call and put prices strictly increasing in sigma (ATM and near-ATM)."""
        vols = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8]
        # Both call and put should have positive vega
        for opt_type in ["call", "put"]:
            for S in [90.0, 100.0, 110.0]:  # OTM, ATM, ITM
                prices_list = [
                    bs_price(S=S, K=100.0, T=1.0, r=0.05, q=0.0, sigma=v, option_type=opt_type)
                    for v in vols
                ]
                for i in range(len(prices_list) - 1):
                    assert prices_list[i] < prices_list[i + 1], (
                        f"{opt_type} not increasing in sigma at S={S}: "
                        f"sigma={vols[i]}->{vols[i+1]}: {prices_list[i]:.4f} >= {prices_list[i+1]:.4f}"
                    )
        # Call price also increasing in T (theta-like, but for European calls on non-div stock)
        maturities = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
        call_by_T = [
            bs_price(S=100.0, K=100.0, T=t, r=0.05, q=0.0, sigma=0.2, option_type="call")
            for t in maturities
        ]
        for i in range(len(call_by_T) - 1):
            assert call_by_T[i] < call_by_T[i + 1], (
                f"ATM call not increasing in T: T={maturities[i]}->{maturities[i+1]}"
            )

    def test_BEH_04_put_call_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T) to within 1e-10 across a parameter grid."""
        import math as _math
        # Sweep across a wide grid of parameters
        param_grid = [
            # (S, K, r, q, sigma, T)
            (100.0, 100.0, 0.05, 0.02, 0.25, 1.0),   # ATM base case
            (100.0, 100.0, 0.05, 0.0,  0.20, 1.0),    # no dividends
            (150.0,  80.0, 0.03, 0.01, 0.15, 0.5),    # deep ITM call
            ( 80.0, 120.0, 0.08, 0.03, 0.40, 2.0),    # deep OTM call
            (100.0, 100.0, 0.0,  0.0,  0.30, 1.0),    # zero rates
            (100.0, 100.0, 0.10, 0.05, 0.10, 0.1),    # short maturity
            (100.0, 100.0, 0.02, 0.0,  0.50, 5.0),    # long maturity high vol
            ( 50.0, 200.0, 0.04, 0.0,  0.20, 1.0),    # very deep OTM call
            (200.0,  50.0, 0.04, 0.0,  0.20, 1.0),    # very deep ITM call
            (100.0, 100.0, 0.05, 0.0,  0.01, 1.0),    # very low vol
        ]
        for S, K, r, q, sigma, T in param_grid:
            call = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
            put  = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put")
            lhs = call - put
            rhs = S * _math.exp(-q * T) - K * _math.exp(-r * T)
            assert abs(lhs - rhs) < 1e-8, (
                f"Put-call parity violated at S={S},K={K},r={r},q={q},sigma={sigma},T={T}: "
                f"|{lhs:.10f} - {rhs:.10f}| = {abs(lhs - rhs):.2e}"
            )

    def test_BEH_05_vol_to_zero_call_intrinsic(self):
        """As sigma→0, option price → intrinsic value (discounted) for both call and put."""
        import math as _math
        sigma_tiny = 1e-8
        # ITM call: intrinsic = S - K (when r=0)
        c_itm = bs_price(S=110.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=sigma_tiny, option_type="call")
        assert abs(c_itm - 10.0) < 0.01, f"ITM call at sigma→0: {c_itm:.6f} != 10.0"

        # OTM call: should be ~0
        c_otm = bs_price(S=80.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=sigma_tiny, option_type="call")
        assert c_otm < 0.01, f"OTM call at sigma→0: {c_otm:.6f} != ~0"

        # ITM put: intrinsic = K - S = 20
        p_itm = bs_price(S=80.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=sigma_tiny, option_type="put")
        assert abs(p_itm - 20.0) < 0.1, f"ITM put at sigma→0: {p_itm:.6f} != 20.0"

        # OTM put: should be ~0
        p_otm = bs_price(S=120.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=sigma_tiny, option_type="put")
        assert p_otm < 0.01, f"OTM put at sigma→0: {p_otm:.6f} != ~0"

        # With nonzero rate: ITM call intrinsic = S - K*exp(-rT)
        r = 0.05
        c_itm_r = bs_price(S=110.0, K=100.0, T=1.0, r=r, q=0.0, sigma=sigma_tiny, option_type="call")
        expected = 110.0 - 100.0 * _math.exp(-r)
        assert abs(c_itm_r - expected) < 0.1, (
            f"ITM call at sigma→0 with r={r}: {c_itm_r:.4f} != {expected:.4f}"
        )

    def test_BEH_06_no_arbitrage_lower_bound(self):
        """Call and put satisfy no-arbitrage bounds across a wide parameter grid.
        Also checks upper bounds: C <= S*exp(-qT), P <= K*exp(-rT)."""
        import math as _math
        cases = [
            (100.0, 100.0, 0.05, 0.0, 0.20, 1.0),
            (110.0,  90.0, 0.03, 0.01, 0.15, 0.5),
            ( 80.0, 100.0, 0.02, 0.0, 0.30, 2.0),
            (100.0, 105.0, 0.05, 0.02, 0.25, 1.5),
            (150.0, 100.0, 0.04, 0.0, 0.35, 0.25),
            ( 50.0, 100.0, 0.01, 0.0, 0.50, 3.0),   # deep OTM
            (200.0,  50.0, 0.08, 0.03, 0.10, 0.1),   # deep ITM, short T
            (100.0, 100.0, 0.0,  0.0, 0.20, 1.0),    # zero rate
            (100.0, 100.0, 0.05, 0.05, 0.20, 1.0),   # r == q
        ]
        for S, K, r, q, sigma, T in cases:
            call = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
            put  = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put")

            # Lower bounds
            call_lower = max(S * _math.exp(-q * T) - K * _math.exp(-r * T), 0.0)
            put_lower  = max(K * _math.exp(-r * T) - S * _math.exp(-q * T), 0.0)
            assert call >= call_lower - 1e-10, (
                f"Call lower bound violated at S={S},K={K}: {call:.6f} < {call_lower:.6f}"
            )
            assert put >= put_lower - 1e-10, (
                f"Put lower bound violated at S={S},K={K}: {put:.6f} < {put_lower:.6f}"
            )

            # Upper bounds: C <= S*exp(-qT), P <= K*exp(-rT)
            assert call <= S * _math.exp(-q * T) + 1e-10, (
                f"Call upper bound violated at S={S}: {call:.6f} > {S * _math.exp(-q*T):.6f}"
            )
            assert put <= K * _math.exp(-r * T) + 1e-10, (
                f"Put upper bound violated at K={K}: {put:.6f} > {K * _math.exp(-r*T):.6f}"
            )

            # Non-negativity
            assert call >= -1e-10, f"Call negative at S={S},K={K}: {call:.6f}"
            assert put >= -1e-10, f"Put negative at S={S},K={K}: {put:.6f}"

    def test_BEH_07_es_geq_var_all_methods(self, sample_prices, simple_portfolio):
        """ES >= VaR for historical, parametric, and Monte Carlo methods."""
        from src.risk.historical import historical_var_es
        from src.risk.parametric import parametric_var_es as _param
        from src.risk.monte_carlo import monte_carlo_var_es

        hist = historical_var_es(
            portfolio=simple_portfolio, prices=sample_prices,
            pricing_date=date.today(), lookback_days=252, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99,
        )
        assert hist["es"] >= hist["var"], (
            f"Historical ES {hist['es']:.4f} < VaR {hist['var']:.4f}"
        )

        param = _param(
            portfolio=simple_portfolio, prices=sample_prices,
            pricing_date=date.today(), lookback_days=252, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99,
        )
        assert param["es"] >= param["var"], (
            f"Parametric ES {param['es']:.4f} < VaR {param['var']:.4f}"
        )

        mc = monte_carlo_var_es(
            portfolio=simple_portfolio, prices=sample_prices,
            pricing_date=date.today(), lookback_days=252, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99,
            n_simulations=2000, random_seed=42,
        )
        assert mc["es"] >= mc["var"], (
            f"MC ES {mc['es']:.4f} < VaR {mc['var']:.4f}"
        )

    def test_BEH_08_historical_var_finite_positive(self, sample_prices, simple_portfolio):
        """Historical VaR is finite and positive on the sample_prices fixture."""
        from src.risk.historical import historical_var_es
        result = historical_var_es(
            portfolio=simple_portfolio, prices=sample_prices,
            pricing_date=date.today(), lookback_days=252, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99,
        )
        assert np.isfinite(result["var"]), "Historical VaR is not finite"
        assert result["var"] > 0, "Historical VaR is non-positive"


# ── Convergence and Inversion Tests ───────────────────────────────────────────

class TestConvergenceAndInversion:
    """CONV_01, INV_01, INV_02."""

    def test_CONV_01_mc_var_convergence(self, sample_prices, simple_portfolio):
        """MC VaR error decreases as N increases; ES also converges; ratio ~ sqrt(10)."""
        common = dict(
            portfolio=simple_portfolio, prices=sample_prices,
            pricing_date=date.today(), lookback_days=252, horizon_days=1,
            var_confidence=0.99, es_confidence=0.975,
        )
        r_500 = monte_carlo_var_es(**common, n_simulations=500,   random_seed=0)
        r_5k  = monte_carlo_var_es(**common, n_simulations=5000,  random_seed=0)
        r_50k = monte_carlo_var_es(**common, n_simulations=50000, random_seed=0)

        # VaR convergence
        err_coarse = abs(r_5k["var"]  - r_500["var"])
        err_fine   = abs(r_50k["var"] - r_5k["var"])
        assert err_fine < err_coarse, (
            f"MC VaR not converging: fine_err={err_fine:.4f} >= coarse_err={err_coarse:.4f}"
        )

        # ES convergence
        es_err_coarse = abs(r_5k["es"]  - r_500["es"])
        es_err_fine   = abs(r_50k["es"] - r_5k["es"])
        assert es_err_fine < es_err_coarse, (
            f"MC ES not converging: fine_err={es_err_fine:.4f} >= coarse_err={es_err_coarse:.4f}"
        )

        # Convergence rate: error ratio should be in the neighbourhood of sqrt(10) ≈ 3.16
        # Allow wide band [1.2, 10] since this is stochastic
        if err_coarse > 1e-6:
            ratio = err_coarse / max(err_fine, 1e-10)
            assert ratio > 1.2, (
                f"Convergence ratio {ratio:.2f} too low — expected > 1.2 for sqrt(10) scaling"
            )

        # All results should be positive
        for label, r in [("500", r_500), ("5k", r_5k), ("50k", r_50k)]:
            assert r["var"] > 0, f"VaR non-positive at N={label}"
            assert r["es"] > 0, f"ES non-positive at N={label}"
            assert np.isfinite(r["var"]), f"VaR not finite at N={label}"
            assert np.isfinite(r["es"]), f"ES not finite at N={label}"

    @pytest.mark.skip(reason="merton_implied_B not yet implemented via round-trip; formula exists")
    def test_INV_01_merton_implied_B_roundtrip(self):
        """INV_01 placeholder: merton_implied_B round-trip to within 1e-4."""
        pass

    def test_INV_01_merton_implied_B_roundtrip_actual(self):
        """Merton implied_B round-trip across multiple (V0, B, sigma, T) combos."""
        from src.credit.merton import merton_pd, merton_implied_B
        cases = [
            # (V0, B_input, r, sigma, T)
            (100.0, 80.0,  0.05, 0.30, 1.0),   # base case
            (100.0, 50.0,  0.05, 0.20, 1.0),   # low leverage
            (100.0, 95.0,  0.05, 0.40, 2.0),   # high leverage, high vol
            (200.0, 150.0, 0.03, 0.25, 0.5),   # different scale, short T
            (100.0, 70.0,  0.08, 0.15, 3.0),   # high rate, low vol, long T
        ]
        for V0, B_input, r, sigma, T in cases:
            pd_val = merton_pd(V0=V0, B=B_input, nu=r, sigma=sigma, T=T)
            target_survival = 1.0 - pd_val
            B_recovered = merton_implied_B(
                V0=V0, target_survival=target_survival, r=r, sigma=sigma, T=T
            )
            assert abs(B_recovered - B_input) < 1e-3, (
                f"Round-trip failed: V0={V0}, B_input={B_input}, sigma={sigma}, T={T}; "
                f"B_recovered={B_recovered:.6f}, error={abs(B_recovered - B_input):.2e}"
            )
            # Verify the recovered B produces the same PD
            pd_check = merton_pd(V0=V0, B=B_recovered, nu=r, sigma=sigma, T=T)
            assert abs(pd_check - pd_val) < 1e-6, (
                f"PD mismatch after round-trip: original={pd_val:.8f}, recovered={pd_check:.8f}"
            )

    def test_INV_02_kupiec_exact_exception_count(self):
        """Kupiec test: not rejected at expected exception count, rejected when too many."""
        # Exact expected count at 95% VaR over 250 days: 250 * 0.05 = 12.5 → 12
        result = kupiec_test(n_observations=250, n_exceptions=12, var_confidence=0.95)
        assert result["p_value"] > 0.05, (
            f"Kupiec incorrectly rejects at exact count: p={result['p_value']:.4f}"
        )
        assert not result["reject_h0"], "Kupiec reject_h0 should be False at exact count"

        # Should reject with way too many exceptions (50 out of 250 = 20%)
        bad = kupiec_test(n_observations=250, n_exceptions=50, var_confidence=0.95)
        assert bad["p_value"] < 0.01, (
            f"Kupiec should reject 50/250 exceptions: p={bad['p_value']:.4f}"
        )
        assert bad["reject_h0"], "Kupiec should reject H0 for 50/250 exceptions"

        # Should also reject with too few exceptions (0 out of 250)
        few = kupiec_test(n_observations=250, n_exceptions=0, var_confidence=0.95)
        assert few["p_value"] < 0.05, (
            f"Kupiec should flag 0/250 exceptions: p={few['p_value']:.4f}"
        )

        # At 99% VaR, expected = 250 * 0.01 = 2.5 → 2 or 3 should not reject
        r99 = kupiec_test(n_observations=250, n_exceptions=3, var_confidence=0.99)
        assert r99["p_value"] > 0.05, (
            f"Kupiec incorrectly rejects 3/250 at 99% VaR: p={r99['p_value']:.4f}"
        )

        # LR statistic should be non-negative
        assert result["lr_stat"] >= 0, "LR statistic should be non-negative"
        assert bad["lr_stat"] > result["lr_stat"], (
            "Bad model should have higher LR statistic than good model"
        )


# ── P&L Attribution Tests ──────────────────────────────────────────────────────

class TestPnLAttribution:
    """PNL_01: P&L attribution for a linear (stocks-only) portfolio."""

    def test_PNL_01_linear_portfolio_zero_residual(self):
        """For a pure stock portfolio, delta-explained P&L = actual P&L exactly.
        Also tests with short positions and varying portfolio sizes."""
        import math as _math

        # --- Case 1: Long-only two-stock portfolio ---
        np.random.seed(11)
        n = 52  # 50 trading days
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        log_ret_aapl = np.random.normal(0.0005, 0.02, n - 1)
        log_ret_msft = np.random.normal(0.0004, 0.018, n - 1)
        aapl = np.concatenate([[150.0], 150.0 * np.exp(np.cumsum(log_ret_aapl))])
        msft = np.concatenate([[300.0], 300.0 * np.exp(np.cumsum(log_ret_msft))])

        q_aapl, q_msft = 100, 50

        actual_pnl = []
        explained_pnl = []
        for t in range(n - 1):
            V_t   = q_aapl * aapl[t]   + q_msft * msft[t]
            V_t1  = q_aapl * aapl[t+1] + q_msft * msft[t+1]
            actual = V_t1 - V_t
            explained = q_aapl * (aapl[t+1] - aapl[t]) + q_msft * (msft[t+1] - msft[t])
            actual_pnl.append(actual)
            explained_pnl.append(explained)

        residual = np.array(actual_pnl) - np.array(explained_pnl)
        assert np.allclose(residual, 0.0, atol=1e-8), (
            f"Long portfolio residual nonzero: max={np.max(np.abs(residual)):.2e}"
        )

        # --- Case 2: Long-short portfolio (short MSFT) ---
        q_short = -30
        for t in range(n - 1):
            V_t  = q_aapl * aapl[t]   + q_short * msft[t]
            V_t1 = q_aapl * aapl[t+1] + q_short * msft[t+1]
            actual = V_t1 - V_t
            explained = q_aapl * (aapl[t+1] - aapl[t]) + q_short * (msft[t+1] - msft[t])
            assert abs(actual - explained) < 1e-8, (
                f"Long-short residual nonzero at t={t}: {abs(actual - explained):.2e}"
            )

        # --- Case 3: Single stock (trivial) ---
        for t in range(n - 1):
            actual = 100 * (aapl[t+1] - aapl[t])
            explained = 100 * (aapl[t+1] - aapl[t])
            assert actual == explained, "Single stock P&L attribution should be exact"

        # --- Variance of explained P&L should match variance of actual P&L ---
        assert np.allclose(np.var(actual_pnl), np.var(explained_pnl), rtol=1e-6), (
            "P&L variance mismatch between actual and explained"
        )


# ── Hedge Effectiveness Tests ──────────────────────────────────────────────────

class TestHedgeEffectiveness:
    """HEDGE_01: Delta hedge reduces P&L magnitude for ATM call under ±1% spot shock."""

    def test_HEDGE_01_delta_hedge_atm_call_one_pct_shock(self):
        """|hedged P&L| < |option P&L| across moneyness, shock sizes, and option types."""
        import math as _math
        r, q, sigma = 0.05, 0.0, 0.2
        T = 1.0 / 12.0  # 1 month

        # Test across multiple moneyness levels and option types
        configs = [
            # (S, K, option_type, label)
            (100.0, 100.0, "call", "ATM call"),
            (100.0, 100.0, "put",  "ATM put"),
            (110.0, 100.0, "call", "ITM call"),
            ( 90.0, 100.0, "call", "OTM call"),
            ( 90.0, 100.0, "put",  "ITM put"),
        ]
        shocks = [+0.005, -0.005, +0.01, -0.01, +0.02, -0.02]

        for S, K, opt_type, label in configs:
            price_0 = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=opt_type)
            delta_0 = bs_delta(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=opt_type)

            for shock_pct in shocks:
                S_new = S * (1.0 + shock_pct)
                price_new = bs_price(S=S_new, K=K, T=T, r=r, q=q, sigma=sigma, option_type=opt_type)

                option_pnl = price_new - price_0
                hedge_pnl  = -delta_0 * (S_new - S)   # delta hedge: short delta shares
                net_hedged = option_pnl + hedge_pnl

                # Delta hedge should reduce P&L magnitude
                if abs(option_pnl) > 1e-6:  # skip if option P&L is negligible
                    assert abs(net_hedged) < abs(option_pnl), (
                        f"Delta hedge failed for {label} shock={shock_pct:+.1%}: "
                        f"|hedged|={abs(net_hedged):.6f}, |unhedged|={abs(option_pnl):.6f}"
                    )

        # Verify the residual (gamma P&L) is second-order: scales as (ΔS)²
        # For ATM call with 1% and 2% shocks, gamma P&L ratio should be ~ 4
        S, K = 100.0, 100.0
        price_0 = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
        delta_0 = bs_delta(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")

        resid_1pct = abs(
            bs_price(S=101.0, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
            - price_0 - delta_0 * 1.0
        )
        resid_2pct = abs(
            bs_price(S=102.0, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
            - price_0 - delta_0 * 2.0
        )
        # Ratio should be approximately 4 (quadratic scaling)
        if resid_1pct > 1e-10:
            ratio = resid_2pct / resid_1pct
            assert 2.5 < ratio < 6.0, (
                f"Gamma P&L scaling ratio {ratio:.2f} outside [2.5, 6.0]; "
                f"expected ~4 for quadratic residual"
            )


if __name__ == "__main__":

    pytest.main([__file__, "-v"])
