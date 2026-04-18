"""
test_regulatory.py
Unit tests for src/risk/regulatory.py + src/services/regulatory_service.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pandas as pd
import pytest

from src.risk.regulatory import (
    CAPITAL_RATIO_FLOOR,
    DFAST_SCENARIOS,
    apply_stress_scenario,
    build_equity_shock_map,
    capital_ratio,
    risk_weighted_assets,
)
from src.schemas import OptionPosition, Portfolio, StockPosition
from src.services.regulatory_service import (
    compute_rwa_and_ratio,
    run_custom_stress,
    run_dfast,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pf_stock_only():
    return Portfolio(
        stocks=[StockPosition("AAPL", 100), StockPosition("MSFT", 50)],
        options=[],
    )


@pytest.fixture
def pf_with_option():
    mat = date.today() + timedelta(days=60)
    return Portfolio(
        stocks=[StockPosition("AAPL", 100)],
        options=[
            OptionPosition(
                ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
                quantity=1, strike=210.0, maturity_date=mat,
                volatility=0.25, risk_free_rate=0.04,
            ),
        ],
    )


@pytest.fixture
def spots():
    return pd.Series({"AAPL": 200.0, "MSFT": 400.0})


# ── risk_weighted_assets + capital_ratio ──────────────────────────────────────

class TestRWA:
    def test_rwa_basic(self):
        assert risk_weighted_assets([100, 50], [1.0, 0.5]) == pytest.approx(125.0)

    def test_rwa_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="asset_values and risk_weights"):
            risk_weighted_assets([100, 50], [1.0])

    def test_rwa_negative_weight_raises(self):
        with pytest.raises(ValueError, match="risk_weights must be non-negative"):
            risk_weighted_assets([100, 50], [1.0, -0.5])

    def test_ratio_pass(self):
        out = capital_ratio(equity=20, rwa=200)
        assert out["ratio"] == pytest.approx(0.1)
        assert out["pass"] is True
        assert out["floor"] == CAPITAL_RATIO_FLOOR

    def test_ratio_fail(self):
        out = capital_ratio(equity=5, rwa=200)
        assert out["pass"] is False

    def test_ratio_zero_rwa_raises(self):
        with pytest.raises(ValueError, match="rwa must be positive"):
            capital_ratio(equity=100, rwa=0)


# ── apply_stress_scenario + DFAST ─────────────────────────────────────────────

class TestStress:
    def test_apply_stress_stock_only(self, pf_stock_only, spots):
        out = apply_stress_scenario(
            pf_stock_only, spots, {"AAPL": -0.10}, date.today(),
        )
        # V_pre = 100*200 + 50*400 = 40000
        # V_post = 100*180 + 50*400 = 38000 → pnl -2000
        assert out["V_pre"] == pytest.approx(40_000.0)
        assert out["V_post"] == pytest.approx(38_000.0)
        assert out["pnl"] == pytest.approx(-2_000.0)
        assert out["pnl_pct"] == pytest.approx(-0.05)

    def test_apply_stress_missing_ticker_ignored(self, pf_stock_only, spots):
        out = apply_stress_scenario(
            pf_stock_only, spots, {"GOOG": -0.50}, date.today(),
        )
        assert out["pnl"] == 0.0

    def test_apply_stress_zero_V_pre_nan_pct(self):
        # Zero-quantity portfolio → V_pre = 0
        pf = Portfolio(stocks=[StockPosition("AAPL", 0)], options=[])
        spots = pd.Series({"AAPL": 100.0})
        out = apply_stress_scenario(pf, spots, {"AAPL": -0.5}, date.today())
        assert out["V_pre"] == 0.0
        import math
        assert math.isnan(out["pnl_pct"])

    def test_build_equity_shock_map(self, pf_with_option):
        m = build_equity_shock_map(pf_with_option, equity_shock=-0.2)
        assert m == {"AAPL": -0.2}

    def test_dfast_structure(self):
        assert set(DFAST_SCENARIOS.keys()) == {"baseline", "adverse", "severely_adverse"}
        for name, params in DFAST_SCENARIOS.items():
            assert "equity" in params and "rates_bp" in params


# ── services/regulatory_service.py ────────────────────────────────────────────

class TestRegulatoryService:
    def test_compute_rwa_and_ratio_default_weight(self, pf_stock_only, spots):
        out = compute_rwa_and_ratio(
            portfolio=pf_stock_only,
            prices=spots,
            risk_weights={"AAPL": 0.5},  # MSFT missing → default 1.0
            equity=5_000.0,
            pricing_date=date.today(),
        )
        # AAPL exposure = 100*200 = 20000 (weight 0.5 → 10000)
        # MSFT exposure = 50*400 = 20000 (weight 1.0 → 20000)
        # RWA = 30000
        assert out["rwa"] == pytest.approx(30_000.0)
        assert out["V"] == pytest.approx(40_000.0)
        assert out["ratio"] == pytest.approx(5_000.0 / 30_000.0)
        assert out["exposures"] == {"AAPL": 20_000.0, "MSFT": 20_000.0}
        assert out["weights"]["MSFT"] == 1.0  # default

    def test_compute_rwa_and_ratio_zero_rwa(self):
        # Zero-quantity portfolio → no exposure → ratio = inf, pass
        pf = Portfolio(
            stocks=[StockPosition("AAPL", 0), StockPosition("MSFT", 0)],
            options=[],
        )
        spots = pd.Series({"AAPL": 200.0, "MSFT": 400.0})
        out = compute_rwa_and_ratio(
            portfolio=pf, prices=spots, risk_weights={},
            equity=1_000.0, pricing_date=date.today(),
        )
        assert out["rwa"] == 0.0
        assert out["ratio"] == float("inf")
        assert out["pass"] is True

    def test_run_dfast_contains_all_scenarios(self, pf_stock_only, spots):
        res = run_dfast(pf_stock_only, spots, date.today())
        assert set(res.keys()) == {"baseline", "adverse", "severely_adverse"}
        # Severely adverse must be worse than baseline
        assert res["severely_adverse"]["pnl"] < res["baseline"]["pnl"]
        for r in res.values():
            assert "equity_shock" in r
            assert "rates_bp" in r

    def test_run_custom_stress(self, pf_stock_only, spots):
        res = run_custom_stress(
            pf_stock_only, spots, {"AAPL": -0.20, "MSFT": -0.10}, date.today()
        )
        # V_post = 100*160 + 50*360 = 34000
        assert res["V_post"] == pytest.approx(34_000.0)
        assert res["pnl"] == pytest.approx(-6_000.0)
