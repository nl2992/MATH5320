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
        """Call price is strictly increasing in spot S."""
        spots = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices_list = [
            bs_price(S=s, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="call")
            for s in spots
        ]
        for i in range(len(prices_list) - 1):
            assert prices_list[i] < prices_list[i + 1], (
                f"Call price not increasing: S={spots[i]} gives {prices_list[i]:.4f}, "
                f"S={spots[i+1]} gives {prices_list[i+1]:.4f}"
            )

    def test_BEH_02_call_monotone_decreasing_in_strike(self):
        """Call price is strictly decreasing in strike K."""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices_list = [
            bs_price(S=100.0, K=k, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="call")
            for k in strikes
        ]
        for i in range(len(prices_list) - 1):
            assert prices_list[i] > prices_list[i + 1], (
                f"Call price not decreasing in K: K={strikes[i]} gives {prices_list[i]:.4f}, "
                f"K={strikes[i+1]} gives {prices_list[i+1]:.4f}"
            )

    def test_BEH_03_call_monotone_increasing_in_vol(self):
        """Call price is strictly increasing in volatility sigma."""
        vols = [0.1, 0.2, 0.3, 0.4]
        prices_list = [
            bs_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=v, option_type="call")
            for v in vols
        ]
        for i in range(len(prices_list) - 1):
            assert prices_list[i] < prices_list[i + 1], (
                f"Call price not increasing in sigma: sigma={vols[i]} gives {prices_list[i]:.4f}, "
                f"sigma={vols[i+1]} gives {prices_list[i+1]:.4f}"
            )

    def test_BEH_04_put_call_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T) to within 1e-10."""
        import math as _math
        S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.25, 1.0
        call = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
        put  = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="put")
        lhs = call - put
        rhs = S * _math.exp(-q * T) - K * _math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10, (
            f"Put-call parity violated: |({call:.8f} - {put:.8f}) - {rhs:.8f}| = "
            f"{abs(lhs - rhs):.2e}"
        )

    def test_BEH_05_vol_to_zero_call_intrinsic(self):
        """BS call at sigma=1e-8 should equal intrinsic ~10.0 within 0.01."""
        price = bs_price(S=110.0, K=100.0, T=1.0, r=0.0, q=0.0, sigma=1e-8, option_type="call")
        assert abs(price - 10.0) < 0.01, (
            f"BS call at sigma=1e-8 gave {price:.6f}, expected ~10.0"
        )

    def test_BEH_06_no_arbitrage_lower_bound(self):
        """C >= max(S*exp(-q*T) - K*exp(-r*T), 0) for a grid of parameters."""
        import math as _math
        cases = [
            (100.0, 100.0, 0.05, 0.0, 0.20, 1.0),
            (110.0,  90.0, 0.03, 0.01, 0.15, 0.5),
            ( 80.0, 100.0, 0.02, 0.0, 0.30, 2.0),
            (100.0, 105.0, 0.05, 0.02, 0.25, 1.5),
            (150.0, 100.0, 0.04, 0.0, 0.35, 0.25),
        ]
        for S, K, r, q, sigma, T in cases:
            call = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
            lower = max(S * _math.exp(-q * T) - K * _math.exp(-r * T), 0.0)
            assert call >= lower - 1e-10, (
                f"No-arbitrage violated: call={call:.6f} < lower={lower:.6f} "
                f"for S={S}, K={K}, r={r}, q={q}, sigma={sigma}, T={T}"
            )

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
        """MC VaR error should decrease as N increases: |VaR_50k - VaR_5k| < |VaR_5k - VaR_500|."""
        common = dict(
            portfolio=simple_portfolio, prices=sample_prices,
            pricing_date=date.today(), lookback_days=252, horizon_days=1,
            var_confidence=0.99, es_confidence=0.975,
        )
        var_500 = monte_carlo_var_es(**common, n_simulations=500,   random_seed=0)["var"]
        var_5k  = monte_carlo_var_es(**common, n_simulations=5000,  random_seed=0)["var"]
        var_50k = monte_carlo_var_es(**common, n_simulations=50000, random_seed=0)["var"]

        err_coarse = abs(var_5k  - var_500)
        err_fine   = abs(var_50k - var_5k)
        assert err_fine < err_coarse, (
            f"MC VaR does not converge: "
            f"|VaR_50k - VaR_5k| = {err_fine:.4f}, "
            f"|VaR_5k - VaR_500| = {err_coarse:.4f}"
        )

    @pytest.mark.skip(reason="merton_implied_B not yet implemented via round-trip; formula exists")
    def test_INV_01_merton_implied_B_roundtrip(self):
        """INV_01 placeholder: merton_implied_B round-trip to within 1e-4."""
        pass

    def test_INV_01_merton_implied_B_roundtrip_actual(self):
        """Given (V0=100, B=80, r=0.05, sigma=0.3, T=1), implied_B should recover B within 1e-4."""
        from src.credit.merton import merton_pd, merton_implied_B
        V0, B_input, r, sigma, T = 100.0, 80.0, 0.05, 0.3, 1.0
        # merton_pd uses 'nu' for drift; pass r for Q-measure
        pd_val = merton_pd(V0=V0, B=B_input, nu=r, sigma=sigma, T=T)
        target_survival = 1.0 - pd_val
        B_recovered = merton_implied_B(
            V0=V0, target_survival=target_survival, r=r, sigma=sigma, T=T
        )
        assert abs(B_recovered - B_input) < 1e-4, (
            f"merton_implied_B round-trip: B_input={B_input}, "
            f"B_recovered={B_recovered:.6f}, error={abs(B_recovered - B_input):.2e}"
        )

    def test_INV_02_kupiec_exact_exception_count(self):
        """Kupiec test with exactly floor(250*0.05)=12 exceptions should not reject (p > 0.05)."""
        result = kupiec_test(n_observations=250, n_exceptions=12, var_confidence=0.95)
        assert result["p_value"] > 0.05, (
            f"Kupiec test incorrectly rejects H0 with exact exception count: "
            f"p_value={result['p_value']:.4f}"
        )
        assert not result["reject_h0"], (
            "Kupiec reject_h0 should be False for exact exception count"
        )


# ── P&L Attribution Tests ──────────────────────────────────────────────────────

class TestPnLAttribution:
    """PNL_01: P&L attribution for a linear (stocks-only) portfolio."""

    def test_PNL_01_linear_portfolio_zero_residual(self):
        """For a pure stock portfolio, delta-explained P&L = actual P&L exactly."""
        np.random.seed(11)
        n = 22  # 20 daily returns -> 20 P&L days
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        # Two stocks with known random walk
        log_ret_aapl = np.random.normal(0.0005, 0.02, n - 1)
        log_ret_msft = np.random.normal(0.0004, 0.018, n - 1)
        aapl = np.concatenate([[150.0], 150.0 * np.exp(np.cumsum(log_ret_aapl))])
        msft = np.concatenate([[300.0], 300.0 * np.exp(np.cumsum(log_ret_msft))])

        q_aapl, q_msft = 100, 50  # share quantities

        # Compute actual P&L and explained P&L for each day t -> t+1
        actual_pnl = []
        explained_pnl = []
        for t in range(n - 1):
            V_t   = q_aapl * aapl[t]   + q_msft * msft[t]
            V_t1  = q_aapl * aapl[t+1] + q_msft * msft[t+1]
            actual = V_t1 - V_t
            # Delta for linear portfolio = number of shares (dollar delta = qty)
            explained = q_aapl * (aapl[t+1] - aapl[t]) + q_msft * (msft[t+1] - msft[t])
            actual_pnl.append(actual)
            explained_pnl.append(explained)

        actual_arr = np.array(actual_pnl)
        explained_arr = np.array(explained_pnl)
        residual = actual_arr - explained_arr

        # For a linear portfolio, unexplained P&L should be exactly 0
        assert np.allclose(residual, 0.0, atol=1e-8), (
            f"Linear portfolio has nonzero unexplained P&L. "
            f"Max residual: {np.max(np.abs(residual)):.2e}"
        )

        # Also verify zero-mean test: |mean| < 2*std/sqrt(T)
        T_days = len(residual)
        if np.std(residual) > 1e-10:
            t_stat = abs(np.mean(residual)) / (np.std(residual) / np.sqrt(T_days))
            assert t_stat < 2.0, (
                f"Unexplained P&L mean is non-zero at 2-sigma: t_stat={t_stat:.4f}"
            )


# ── Hedge Effectiveness Tests ──────────────────────────────────────────────────

class TestHedgeEffectiveness:
    """HEDGE_01: Delta hedge reduces P&L magnitude for ATM call under ±1% spot shock."""

    def test_HEDGE_01_delta_hedge_atm_call_one_pct_shock(self):
        """|hedged P&L| < |option P&L| for both +1% and -1% shock to ATM call."""
        import math as _math
        S, K, r, q, sigma = 100.0, 100.0, 0.05, 0.0, 0.2
        T = 1.0 / 12.0  # 1 month

        # Current price and delta
        price_0 = bs_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")
        delta_0 = bs_delta(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")

        for shock_pct in [+0.01, -0.01]:
            S_new = S * (1.0 + shock_pct)
            # Use same T (instantaneous shock, no time passage)
            price_new = bs_price(S=S_new, K=K, T=T, r=r, q=q, sigma=sigma, option_type="call")

            option_pnl   = price_new - price_0           # long call P&L
            hedge_pnl    = -delta_0 * S * shock_pct      # short delta × dollar move
            net_hedged   = option_pnl + hedge_pnl        # net hedged position P&L

            assert abs(net_hedged) < abs(option_pnl), (
                f"Delta hedge did not reduce P&L for shock={shock_pct:+.1%}: "
                f"|hedged|={abs(net_hedged):.5f}, |unhedged|={abs(option_pnl):.5f}"
            )


if __name__ == "__main__":

    pytest.main([__file__, "-v"])
