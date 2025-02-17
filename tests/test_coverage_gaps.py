"""
test_coverage_gaps.py
Targeted tests for branches not hit by test_backend.py or the other test files.
Each test names the file + line it was added for.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.pricing.black_scholes import bs_delta, bs_price
from src.portfolio.portfolio import portfolio_exposure
from src.portfolio.positions import option_delta_exposure, option_value
from src.risk.backtest import _forecast_var, kupiec_test, run_backtest
from src.risk.estimators import get_mean_cov
from src.risk.monte_carlo import monte_carlo_var_es
from src.risk.returns import build_overlapping_horizon_log_returns
from src.schemas import OptionPosition, Portfolio, StockPosition
from src.services.risk_engine_service import RiskEngineService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    np.random.seed(7)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    aapl = 150 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 300)))
    msft = 300 * np.exp(np.cumsum(np.random.normal(0.0004, 0.018, 300)))
    return pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=dates)


@pytest.fixture
def pf_stocks():
    return Portfolio(
        stocks=[StockPosition("AAPL", 100), StockPosition("MSFT", 50)],
        options=[],
    )


@pytest.fixture
def pf_with_option_same_underlying():
    """Option on a stock that is already held — exercises the dedup branch."""
    mat = date.today() + timedelta(days=90)
    return Portfolio(
        stocks=[StockPosition("AAPL", 100)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1, strike=160.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )


# ── black_scholes.py ──────────────────────────────────────────────────────────

class TestBSValidation:
    def test_T_zero_raises(self):
        with pytest.raises(ValueError, match="Time to maturity"):
            bs_price(S=100, K=100, T=0, r=0.05, q=0, sigma=0.2, option_type="call")

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="Volatility sigma"):
            bs_price(S=100, K=100, T=1, r=0.05, q=0, sigma=0, option_type="call")

    def test_S_zero_raises(self):
        with pytest.raises(ValueError, match="Spot price"):
            bs_price(S=0, K=100, T=1, r=0.05, q=0, sigma=0.2, option_type="call")

    def test_bs_price_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown option_type"):
            bs_price(S=100, K=100, T=1, r=0.05, q=0, sigma=0.2, option_type="banana")

    def test_bs_delta_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown option_type"):
            bs_delta(S=100, K=100, T=1, r=0.05, q=0, sigma=0.2, option_type="banana")


# ── portfolio.py — option underlying dedup ────────────────────────────────────

class TestPortfolioDedup:
    def test_portfolio_exposure_option_already_seen(self, pf_with_option_same_underlying):
        spots = pd.Series({"AAPL": 180.0})
        exp = portfolio_exposure(
            pf_with_option_same_underlying, spots, date.today(),
        )
        assert "AAPL" in exp.index
        # Underlying should appear only once (dedup path hit).
        assert len(exp) == 1


# ── positions.py — expired put branch + T<=0 delta exposure ───────────────────

class TestPositionsEdgeCases:
    def test_expired_put_intrinsic(self):
        pos = OptionPosition(
            ticker="P", underlying_ticker="AAPL", option_type="put",
            quantity=1, strike=200.0, maturity_date=date.today() - timedelta(days=1),
            volatility=0.25, risk_free_rate=0.04,
        )
        v = option_value(pos, spot=180.0, pricing_date=date.today())
        # intrinsic put = max(200-180, 0) = 20, times qty=1, multiplier=100 → 2000
        assert v == pytest.approx(2000.0)

    def test_expired_option_delta_exposure_is_zero(self):
        pos = OptionPosition(
            ticker="C", underlying_ticker="AAPL", option_type="call",
            quantity=1, strike=150.0, maturity_date=date.today() - timedelta(days=1),
            volatility=0.25, risk_free_rate=0.04,
        )
        delta = option_delta_exposure(pos, spot=200.0, pricing_date=date.today())
        assert delta == 0.0


# ── returns.py — horizon_days < 1 raise ───────────────────────────────────────

class TestReturns:
    def test_overlapping_horizon_bad_horizon_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"AAPL": np.log([100, 101, 102, 103, 104]) -
                                    np.log([99, 100, 101, 102, 103])}, index=dates)
        with pytest.raises(ValueError, match="horizon_days must be >= 1"):
            build_overlapping_horizon_log_returns(df, 0)


# ── estimators.py — dispatcher routes to ewma ─────────────────────────────────

class TestEstimatorsDispatch:
    def test_get_mean_cov_ewma_path(self, sample_prices):
        ret = np.log(sample_prices / sample_prices.shift(1)).dropna()
        mu, cov = get_mean_cov(ret, lookback_days=120, estimator="ewma", ewma_N=30)
        assert mu.shape == (2,)
        assert cov.shape == (2, 2)

    def test_get_mean_cov_window_default(self, sample_prices):
        ret = np.log(sample_prices / sample_prices.shift(1)).dropna()
        mu, cov = get_mean_cov(ret, lookback_days=120)  # default estimator
        assert cov.shape == (2, 2)


# ── backtest.py — exception path, monte_carlo path, unknown model, Kupiec N=0 ─

class TestBacktestGaps:
    def test_forecast_var_unknown_model_raises(self, sample_prices, pf_stocks):
        with pytest.raises(ValueError, match="Unknown backtest model"):
            _forecast_var(
                portfolio=pf_stocks, prices=sample_prices,
                pricing_date=date.today(), lookback_days=60, horizon_days=1,
                var_confidence=0.99, model="nope", estimator="window",
                ewma_N=60, n_simulations=100,
            )

    def test_run_backtest_monte_carlo(self, sample_prices, pf_stocks):
        bt = run_backtest(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=60, horizon_days=1, var_confidence=0.95,
            model="monte_carlo", n_simulations=200,
        )
        assert not bt.empty or bt.attrs.get("reason") is not None

    def test_run_backtest_swallows_forecast_errors(self, monkeypatch, sample_prices, pf_stocks):
        """If _forecast_var raises, the date is skipped — line 118-119 branch."""
        from src.risk import backtest as bt_mod
        original = bt_mod._forecast_var
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom on first call")
            return original(*a, **kw)

        monkeypatch.setattr(bt_mod, "_forecast_var", flaky)
        df = bt_mod.run_backtest(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=60, horizon_days=1, var_confidence=0.95, model="historical",
        )
        assert not df.empty  # other dates still ran

    def test_kupiec_zero_observations(self):
        out = kupiec_test(n_observations=0, n_exceptions=0, var_confidence=0.99)
        assert out["n_observations"] == 0
        assert out["p_hat"] != out["p_hat"]  # NaN
        assert out["reject_h0"] is False


# ── monte_carlo.py — random_seed=None + option dedup ──────────────────────────

class TestMonteCarloGaps:
    def test_mc_without_seed(self, sample_prices, pf_stocks):
        res = monte_carlo_var_es(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99, n_simulations=500,
            random_seed=None,
        )
        assert res["var"] > 0

    def test_mc_with_option_dedup(self, sample_prices, pf_with_option_same_underlying):
        res = monte_carlo_var_es(
            portfolio=pf_with_option_same_underlying, prices=sample_prices,
            pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99, n_simulations=300,
        )
        assert res["var"] > 0


# ── historical.py — option underlying dedup ───────────────────────────────────

class TestHistoricalGaps:
    def test_historical_with_duplicate_underlying(self, sample_prices, pf_with_option_same_underlying):
        from src.risk.historical import historical_var_es
        res = historical_var_es(
            portfolio=pf_with_option_same_underlying, prices=sample_prices,
            pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.99, es_confidence=0.975,
        )
        assert res["var"] > 0


# ── Option on a *new* underlying — exercises the append branch in `_all_underlyings`
# and `_portfolio_underlyings` helpers inside portfolio.py, historical.py, monte_carlo.py.

class TestOptionOnNewUnderlying:
    @pytest.fixture
    def pf_option_new_underlying(self):
        mat = date.today() + timedelta(days=90)
        return Portfolio(
            stocks=[StockPosition("AAPL", 100)],
            options=[OptionPosition(
                ticker="MSFT_C", underlying_ticker="MSFT", option_type="call",
                quantity=1, strike=400.0, maturity_date=mat,
                volatility=0.25, risk_free_rate=0.04,
            )],
        )

    def test_portfolio_exposure(self, pf_option_new_underlying):
        spots = pd.Series({"AAPL": 180.0, "MSFT": 400.0})
        exp = portfolio_exposure(pf_option_new_underlying, spots, date.today())
        # Both AAPL and MSFT should appear (MSFT introduced by the option).
        assert set(exp.index) == {"AAPL", "MSFT"}

    def test_historical_on_new_underlying(self, sample_prices, pf_option_new_underlying):
        from src.risk.historical import historical_var_es
        res = historical_var_es(
            portfolio=pf_option_new_underlying, prices=sample_prices,
            pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.99, es_confidence=0.975,
        )
        assert res["var"] > 0

    def test_mc_on_new_underlying(self, sample_prices, pf_option_new_underlying):
        res = monte_carlo_var_es(
            portfolio=pf_option_new_underlying, prices=sample_prices,
            pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.99, es_confidence=0.99, n_simulations=300,
        )
        assert res["var"] > 0


# ── market_data.py line 265: ^TNX column missing after a successful fetch ─────

class TestFetchRiskFreeRateColumnMissing:
    def test_missing_column_triggers_fallback(self, tmp_path, monkeypatch):
        """download succeeds but returns a frame without ^TNX column → fallback."""
        from src.data import market_data

        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        wrong_col = pd.DataFrame({"Other": [4.0, 4.1, 4.2]}, index=dates)
        monkeypatch.setattr(
            market_data, "download_adjusted_close_cached",
            lambda *a, **kw: wrong_col,
        )
        r = market_data.fetch_risk_free_rate(
            date(2024, 1, 5), fallback=0.06, cache_dir=str(tmp_path),
        )
        assert r == 0.06


# ── services/risk_engine_service.py — portfolio_value + run_backtest empty ────

class TestRiskEngineServiceGaps:
    def test_service_portfolio_value(self, sample_prices, pf_stocks):
        svc = RiskEngineService(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.99, es_confidence=0.975,
        )
        pv = svc.portfolio_value()
        # Manual: 100*last(AAPL) + 50*last(MSFT)
        expected = (
            100 * float(sample_prices["AAPL"].iloc[-1])
            + 50 * float(sample_prices["MSFT"].iloc[-1])
        )
        assert pv == pytest.approx(expected)

    def test_service_run_backtest_empty_window(self, sample_prices, pf_stocks):
        """Lookback > history → backtest_df is empty; Kupiec fallback branch hit."""
        svc = RiskEngineService(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=10_000, horizon_days=1,  # way too large
            var_confidence=0.99, es_confidence=0.975,
        )
        res = svc.run_backtest(model="historical")
        assert res["backtest_df"].empty
        assert res["kupiec"]["n_observations"] == 0
        assert res["reason"] is not None

    def test_service_run_backtest_has_kupiec(self, sample_prices, pf_stocks):
        svc = RiskEngineService(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.95, es_confidence=0.975,
        )
        res = svc.run_backtest(model="historical")
        assert res["model"] == "historical"
        assert res["kupiec"]["n_observations"] > 0
