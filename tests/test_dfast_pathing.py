"""Tests for DFAST capital-path simulator in src/risk/regulatory.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from src.risk.regulatory import (
    CapitalState, StressQuarter,
    project_capital_one_quarter, project_capital_path,
    min_capital_ratio, passes_stress,
    apply_global_market_shock, apply_counterparty_default_component,
    run_dfast_scenarios,
)

def _shock(**kwargs):
    defaults = dict(quarter=1, credit_loss=0, trading_loss=0,
                    counterparty_loss=0, pre_provision_net_revenue=0,
                    provisions=0, dividends=0, buybacks=0, rwa_change=0)
    defaults.update(kwargs)
    return StressQuarter(**defaults)

class TestOneQuarter:
    def test_capital_math(self):
        state = CapitalState(tier1_capital=100.0, rwa=1000.0)
        shock = _shock(quarter=1, credit_loss=5, trading_loss=2,
                       counterparty_loss=1, pre_provision_net_revenue=4,
                       provisions=1, dividends=1, rwa_change=10)
        new = project_capital_one_quarter(state, shock)
        assert new.tier1_capital == pytest.approx(94.0)   # 100+4-5-2-1-1-1=94
        assert new.rwa == pytest.approx(1010.0)

class TestNineQuarterPath:
    def _nine_shocks(self, **kw):
        return [_shock(quarter=q+1, **kw) for q in range(9)]

    def test_failing_path(self):
        initial = CapitalState(100.0, 1000.0)
        shocks = self._nine_shocks(credit_loss=5, trading_loss=2,
                                   counterparty_loss=1, pre_provision_net_revenue=4,
                                   provisions=1, dividends=1, rwa_change=10)
        path = project_capital_path(initial, shocks)
        last = path[-1]
        assert last["tier1_capital"] == pytest.approx(100 - 9*6)   # 46
        assert last["rwa"] == pytest.approx(1090.0)
        assert last["capital_ratio"] == pytest.approx(46/1090)
        assert not passes_stress(path)

    def test_passing_path(self):
        initial = CapitalState(150.0, 1000.0)
        shocks = self._nine_shocks(credit_loss=3, trading_loss=1,
                                   counterparty_loss=0.5, pre_provision_net_revenue=5,
                                   provisions=0.5, dividends=0.5, rwa_change=5)
        path = project_capital_path(initial, shocks)
        last = path[-1]
        assert last["tier1_capital"] == pytest.approx(145.5)
        assert last["rwa"] == pytest.approx(1045.0)
        assert last["capital_ratio"] == pytest.approx(145.5/1045)
        assert passes_stress(path)

class TestMinRatio:
    def test_min_ratio(self):
        initial = CapitalState(100.0, 1000.0)
        shocks = [_shock(quarter=q+1, credit_loss=10, rwa_change=0) for q in range(9)]
        path = project_capital_path(initial, shocks)
        assert min_capital_ratio(path) == pytest.approx(10/1000)

class TestGlobalShocks:
    def test_global_market_shock(self):
        state = CapitalState(100.0, 1000.0)
        new = apply_global_market_shock(state, 20.0)
        assert new.tier1_capital == pytest.approx(80.0)

    def test_counterparty_default(self):
        state = CapitalState(100.0, 1000.0)
        new = apply_counterparty_default_component(state, 15.0)
        assert new.tier1_capital == pytest.approx(85.0)

class TestDFASTScenarios:
    def test_three_scenarios_run(self):
        initial = CapitalState(100.0, 1000.0)
        results = run_dfast_scenarios(initial)
        names = {r["scenario"] for r in results}
        assert "baseline" in names
        assert "adverse" in names
        assert "severely_adverse" in names

    def test_severely_adverse_worse_than_baseline(self):
        initial = CapitalState(100.0, 1000.0)
        results = run_dfast_scenarios(initial)
        by_name = {r["scenario"]: r for r in results}
        assert (by_name["severely_adverse"]["ending_capital_ratio"]
                < by_name["baseline"]["ending_capital_ratio"])

class TestValidation:
    def test_negative_rwa_raises(self):
        with pytest.raises(ValueError):
            CapitalState(100.0, -1.0)

    def test_zero_rwa_raises(self):
        with pytest.raises(ValueError):
            CapitalState(100.0, 0.0)

    def test_rwa_goes_nonpositive_raises(self):
        state = CapitalState(100.0, 1000.0)
        shock = _shock(quarter=1, rwa_change=-2000.0)
        with pytest.raises(ValueError):
            project_capital_one_quarter(state, shock)
