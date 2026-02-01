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

from src.credit.cds import cds_par_spread, cds_par_spread_constant_full_closed_form
from src.credit.cva import cva_continuous_constant_exposure, cva_discounted
from src.credit.hazard import (
    cumhazard_piecewise, density_piecewise, hazard_at_piecewise,
    interval_default_prob_piecewise,
)
from src.credit.mitigation import default_waterfall_loss_allocation, mitigated_cva
from src.pricing.black_scholes import bs_delta, bs_price
from src.portfolio.portfolio import portfolio_exposure
from src.portfolio.positions import option_delta_exposure, option_value
from src.risk.backtest import _forecast_var, kupiec_test, run_backtest
from src.risk.estimators import get_mean_cov, manual_mean_cov
from src.risk.monte_carlo import monte_carlo_var_es
from src.risk.normal import portfolio_delta_normal_mean_var
from src.risk.returns import (
    build_overlapping_horizon_absolute_returns,
    build_overlapping_horizon_log_returns,
)
from src.schemas import OptionPosition, Portfolio, StockPosition
from src.services.regulatory_service import run_dfast_capital_path
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
                calibration_mode="historical", manual_market_params=None,
                option_vol_shock_mode="fixed", option_vol_shock_beta=1.0,
                option_vol_shock_floor=0.05,
            )

    def test_run_backtest_monte_carlo(self, sample_prices, pf_stocks):
        bt = run_backtest(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=60, horizon_days=1, var_confidence=0.95,
            model="monte_carlo", n_simulations=200,
        )
        assert not bt.empty or bt.attrs.get("reason") is not None

    def test_run_backtest_records_forecast_errors(self, monkeypatch, sample_prices, pf_stocks):
        """If _forecast_var raises, the date is skipped and the error is retained."""
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
        assert df.attrs["n_skipped_forecasts"] == 1
        assert len(df.attrs["skipped_forecasts"]) == 1
        assert "boom on first call" in df.attrs["skipped_forecasts"][0]["error"]

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


class TestManualMeanCovValidation:
    def test_manual_mean_cov_missing_underlying_raises(self):
        with pytest.raises(ValueError, match="missing one or more portfolio underlyings"):
            manual_mean_cov(
                {
                    "mu_daily": pd.Series({"AAPL": 0.0}),
                    "cov_daily": pd.DataFrame([[0.0004]], index=["AAPL"], columns=["AAPL"]),
                },
                ["AAPL", "MSFT"],
            )

    def test_manual_mean_cov_non_psd_raises(self):
        with pytest.raises(ValueError, match="positive semidefinite"):
            manual_mean_cov(
                {
                    "mu_daily": pd.Series({"AAPL": 0.0, "MSFT": 0.0}),
                    "cov_daily": pd.DataFrame(
                        [[0.0004, 0.0010], [0.0010, 0.0003]],
                        index=["AAPL", "MSFT"],
                        columns=["AAPL", "MSFT"],
                    ),
                },
                ["AAPL", "MSFT"],
            )

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
        assert "n_skipped_forecasts" in res


# ── normal.py — portfolio_delta_normal_mean_var ──────────────────────────────

class TestNormalPortfolioMeanVar:
    def test_basic_two_asset(self):
        x = np.array([100.0, 200.0])
        mu = np.array([0.001, 0.002])
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        m, s = portfolio_delta_normal_mean_var(x, mu, cov)
        assert abs(m - float(x @ mu)) < 1e-10
        assert s > 0

    def test_single_asset(self):
        m, s = portfolio_delta_normal_mean_var([50.0], [0.0005], [[0.0004]])
        assert m == pytest.approx(0.025)
        assert s == pytest.approx(np.sqrt(0.0004) * 50.0)

    def test_zero_variance_clamps_to_zero(self):
        m, s = portfolio_delta_normal_mean_var([100.0], [0.001], [[0.0]])
        assert s == 0.0


# ── returns.py — absolute-return horizon builder ─────────────────────────────

class TestAbsoluteReturns:
    def test_build_absolute_returns(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"AAPL": np.linspace(100.0, 110.0, 10)}, index=dates)
        result = build_overlapping_horizon_absolute_returns(prices, horizon=2)
        assert result.shape[1] == 1
        assert len(result) == 8  # 10 - 2

    def test_bad_horizon_raises(self):
        prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            build_overlapping_horizon_absolute_returns(prices, horizon=0)


# ── estimators.py — validation branches in manual_mean_cov ───────────────────

class TestManualMeanCovAllBranches:
    def _valid_params(self):
        return {
            "mu_daily": pd.Series({"A": 0.001, "B": 0.002}),
            "cov_daily": pd.DataFrame(
                [[0.0004, 0.0001], [0.0001, 0.0009]],
                index=["A", "B"], columns=["A", "B"],
            ),
        }

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            manual_mean_cov([0.001, 0.002], ["A", "B"])

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="must contain"):
            manual_mean_cov({"mu_daily": pd.Series({"A": 0.001})}, ["A"])

    def test_non_finite_mu_raises(self):
        p = self._valid_params()
        p["mu_daily"] = pd.Series({"A": float("nan"), "B": 0.002})
        with pytest.raises(ValueError, match="non-finite"):
            manual_mean_cov(p, ["A", "B"])

    def test_non_finite_cov_raises(self):
        p = self._valid_params()
        p["cov_daily"] = pd.DataFrame(
            [[float("inf"), 0.0], [0.0, 0.0009]],
            index=["A", "B"], columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="non-finite"):
            manual_mean_cov(p, ["A", "B"])

    def test_asymmetric_cov_raises(self):
        p = self._valid_params()
        p["cov_daily"] = pd.DataFrame(
            [[0.0004, 0.0005], [0.0001, 0.0009]],
            index=["A", "B"], columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="symmetric"):
            manual_mean_cov(p, ["A", "B"])

    def test_negative_diagonal_raises(self):
        p = self._valid_params()
        p["cov_daily"] = pd.DataFrame(
            [[-0.0001, 0.0], [0.0, 0.0009]],
            index=["A", "B"], columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="negative diagonal"):
            manual_mean_cov(p, ["A", "B"])


# ── risk/historical.py — absolute shock path ─────────────────────────────────

class TestHistoricalAbsoluteShock:
    def test_absolute_shock_mode(self, sample_prices, pf_stocks):
        from src.risk.historical import historical_var_es
        res = historical_var_es(
            portfolio=pf_stocks, prices=sample_prices, pricing_date=date.today(),
            lookback_days=100, horizon_days=1,
            var_confidence=0.95, es_confidence=0.95,
            shock_type="absolute",
        )
        assert res["var"] > 0
        assert res["es"] >= res["var"]


# ── credit/cds.py — input validation + q==0 edge case ────────────────────────

class TestCDSParSpreadClosedForm:
    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda must be non-negative"):
            cds_par_spread_constant_full_closed_form(T=5, freq=1, r=0.02, lam=-0.01, R=0.4)

    def test_R_out_of_range_raises(self):
        with pytest.raises(ValueError, match="R must be in"):
            cds_par_spread_constant_full_closed_form(T=5, freq=1, r=0.02, lam=0.03, R=1.5)

    def test_T_nonpositive_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            cds_par_spread_constant_full_closed_form(T=0, freq=1, r=0.02, lam=0.03, R=0.4)

    def test_freq_nonpositive_raises(self):
        with pytest.raises(ValueError, match="freq must be positive"):
            cds_par_spread_constant_full_closed_form(T=5, freq=0, r=0.02, lam=0.03, R=0.4)

    def test_q_zero_branch(self):
        # r=0, lam=0 → q=0 exercises the q==0.0 branch; spread = 0 (no hazard)
        spread = cds_par_spread_constant_full_closed_form(T=5, freq=1, r=0.0, lam=0.0, R=0.4)
        assert spread == 0.0

    def test_accrual_false(self):
        spread = cds_par_spread_constant_full_closed_form(T=5, freq=1, r=0.02, lam=0.03, R=0.4, accrual=False)
        assert spread > 0


# ── credit/hazard.py — piecewise functions ───────────────────────────────────

class TestHazardPiecewise:
    _GRID = [0.0, 1.0, 3.0, 5.0]
    _LAMS = [0.02, 0.03, 0.025]

    def test_hazard_at_within_grid(self):
        h = hazard_at_piecewise(0.5, self._GRID, self._LAMS)
        assert h == pytest.approx(0.02)

    def test_hazard_at_extrapolates_past_end(self):
        h = hazard_at_piecewise(10.0, self._GRID, self._LAMS)
        assert h == pytest.approx(0.025)

    def test_cumhazard_piecewise(self):
        ch = cumhazard_piecewise(2.0, self._GRID, self._LAMS)
        assert ch > 0

    def test_density_piecewise(self):
        d = density_piecewise(1.5, self._GRID, self._LAMS)
        assert d > 0

    def test_interval_default_prob_piecewise(self):
        p = interval_default_prob_piecewise(1.0, 2.0, self._GRID, self._LAMS)
        assert 0 < p < 1


# ── credit/cva.py — validation branches ──────────────────────────────────────

class TestCVAValidation:
    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            cva_discounted([-1.0], [0.01], [0.99], R=0.4)

    def test_bad_default_probs_raises(self):
        with pytest.raises(ValueError, match="marginal_default_probs"):
            cva_discounted([1.0], [-0.01], [0.99], R=0.4)

    def test_bad_discount_factor_raises(self):
        with pytest.raises(ValueError, match="discount_factors"):
            cva_discounted([1.0], [0.01], [1.5], R=0.4)

    def test_bad_R_raises(self):
        with pytest.raises(ValueError, match="R must be in"):
            cva_discounted([1.0], [0.01], [0.99], R=1.5)

    def test_cva_continuous_negative_lam_raises(self):
        with pytest.raises(ValueError, match="lam must be non-negative"):
            cva_continuous_constant_exposure(K=1.0, lam=-0.01, T=1, R=0.4)

    def test_cva_continuous_bad_R_raises(self):
        with pytest.raises(ValueError, match="R must be in"):
            cva_continuous_constant_exposure(K=1.0, lam=0.03, T=1, R=1.5)

    def test_cva_continuous_bad_r_raises(self):
        with pytest.raises(ValueError, match="r must be non-negative"):
            cva_continuous_constant_exposure(K=1.0, lam=0.03, T=1, R=0.4, r=-0.01)


# ── credit/mitigation.py — ccp waterfall validation + csa branch ─────────────

class TestMitigationCoverage:
    def test_ccp_negative_loss_raises(self):
        with pytest.raises(ValueError, match="loss must be non-negative"):
            default_waterfall_loss_allocation(loss=-1.0, defaulter_margin=0.5, default_fund=0.3, ccp_capital=0.1)

    def test_ccp_negative_margin_raises(self):
        with pytest.raises(ValueError, match="defaulter_margin must be non-negative"):
            default_waterfall_loss_allocation(loss=1.0, defaulter_margin=-0.1, default_fund=0.3, ccp_capital=0.1)

    def test_mitigated_cva_csa_branch(self):
        # threshold > 0 → exercises the csa_residual_exposure_after_margin_call branch
        result = mitigated_cva(
            mtm_paths=[[2.0, 1.0], [3.0]],
            marginal_default_probs=[0.01, 0.01],
            discount_factors=[0.99, 0.98],
            R=0.4,
            collateral=[0.5, 0.5],
            threshold=0.5,
            mta=0.1,
        )
        assert result >= 0


# ── services/regulatory_service.py — run_dfast_capital_path ──────────────────

class TestRegulatoryServiceDFAST:
    def test_run_dfast_capital_path_returns_all_scenarios(self):
        result = run_dfast_capital_path(
            tier1_capital=10_000_000,
            rwa=80_000_000,
            assets=100_000_000,
        )
        assert isinstance(result, dict)
        assert len(result) >= 1
        for name, val in result.items():
            assert "path" in val
            assert "passes" in val
            assert "min_ratio" in val
            assert len(val["path"]) == 9
