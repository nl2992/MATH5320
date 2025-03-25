"""Tests for src/credit/mitigation.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from src.credit.mitigation import (
    gross_positive_exposure, netted_exposure, netting_benefit,
    netted_exposure_by_counterparty,
    simple_collateralized_exposure,
    csa_call_amount, csa_residual_exposure_after_margin_call,
    ccp_cleared_exposure, default_waterfall_loss_allocation,
    mitigated_cva,
)

class TestNetting:
    def test_gross_positive(self):
        assert gross_positive_exposure([10.0, -6.0, 4.0]) == pytest.approx(14.0)

    def test_netted(self):
        assert netted_exposure([10.0, -6.0, 4.0]) == pytest.approx(8.0)

    def test_benefit(self):
        assert netting_benefit([10.0, -6.0, 4.0]) == pytest.approx(6.0)

    def test_all_negative_gross_zero(self):
        assert gross_positive_exposure([-5.0, -4.0]) == pytest.approx(0.0)
        assert netted_exposure([-5.0, -4.0]) == pytest.approx(0.0)
        assert netting_benefit([-5.0, -4.0]) == pytest.approx(0.0)

    def test_large_benefit(self):
        assert gross_positive_exposure([10, -12, 3]) == pytest.approx(13.0)
        assert netted_exposure([10, -12, 3]) == pytest.approx(1.0)
        assert netting_benefit([10, -12, 3]) == pytest.approx(12.0)

    def test_netting_never_increases_exposure(self):
        mtms = [10.0, -3.0, 7.0, -2.0]
        assert netted_exposure(mtms) <= gross_positive_exposure(mtms)

class TestNettingByCounterparty:
    def test_multi_counterparty(self):
        mtms = [10.0, -6.0, 7.0, -2.0, -8.0]
        cids = ["A", "A", "B", "B", "C"]
        result = netted_exposure_by_counterparty(mtms, cids)
        assert result["A"] == pytest.approx(4.0)
        assert result["B"] == pytest.approx(5.0)
        assert result["C"] == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            netted_exposure_by_counterparty([1.0, 2.0], ["A"])

    def test_nan_raises(self):
        import math
        with pytest.raises(ValueError):
            netted_exposure_by_counterparty([float("nan"), 2.0], ["A", "B"])

class TestCollateral:
    def test_simple(self):
        assert simple_collateralized_exposure(10.0, 7.0) == pytest.approx(3.0)

    def test_overcollateralized(self):
        assert simple_collateralized_exposure(10.0, 12.0) == pytest.approx(0.0)

    def test_csa_call_triggered(self):
        # exposure - collateral - threshold = 100 - 80 - 5 = 15 > mta=2
        assert csa_call_amount(100.0, 80.0, threshold=5.0, mta=2.0) == pytest.approx(15.0)

    def test_csa_call_not_triggered(self):
        # exposure - collateral - threshold = 100 - 94 - 5 = 1 <= mta=2
        assert csa_call_amount(100.0, 94.0, threshold=5.0, mta=2.0) == pytest.approx(0.0)

    def test_csa_residual_when_call_triggered(self):
        assert csa_residual_exposure_after_margin_call(100.0, 80.0, 5.0, 2.0) == pytest.approx(5.0)

    def test_csa_residual_when_no_call(self):
        assert csa_residual_exposure_after_margin_call(100.0, 94.0, 5.0, 2.0) == pytest.approx(6.0)

    def test_gap_risk_residual(self):
        # After gap move, old collateral still in place
        assert simple_collateralized_exposure(112.0, 95.0) == pytest.approx(17.0)

class TestCCP:
    def test_ccp_cleared(self):
        assert ccp_cleared_exposure([10.0, -3.0], initial_margin=3.0, variation_margin=4.0) == pytest.approx(0.0)

    def test_default_waterfall(self):
        result = default_waterfall_loss_allocation(100.0, 40.0, 35.0, 15.0)
        assert result["covered_by_margin"] == pytest.approx(40.0)
        assert result["covered_by_default_fund"] == pytest.approx(35.0)
        assert result["covered_by_ccp_capital"] == pytest.approx(15.0)
        assert result["unfunded_loss"] == pytest.approx(10.0)

    def test_fully_covered_waterfall(self):
        result = default_waterfall_loss_allocation(50.0, 60.0, 0.0, 0.0)
        assert result["covered_by_margin"] == pytest.approx(50.0)
        assert result["unfunded_loss"] == pytest.approx(0.0)

class TestMitigatedCVA:
    def test_mitigated_less_than_unmitigated(self):
        mtm_paths = [[10.0, -4.0], [15.0, -8.0], [-3.0, 12.0]]
        pds = [0.01, 0.015, 0.02]
        dfs = [0.99, 0.97, 0.95]
        R = 0.40
        collateral = [2.0, 4.0, 5.0]
        mit = mitigated_cva(mtm_paths, pds, dfs, R, collateral=collateral)
        unmit = mitigated_cva(mtm_paths, pds, dfs, R, collateral=None)
        assert mit <= unmit

    def test_acceptance(self):
        mtm_paths = [[10.0, -4.0], [15.0, -8.0], [-3.0, 12.0]]
        pds = [0.01, 0.015, 0.02]
        dfs = [0.99, 0.97, 0.95]
        R = 0.40
        collateral = [2.0, 4.0, 5.0]
        # netted: [6,7,9], residual after collateral: [4,3,4]
        expected = 0.60 * (4*0.01*0.99 + 3*0.015*0.97 + 4*0.02*0.95)
        result = mitigated_cva(mtm_paths, pds, dfs, R, collateral=collateral)
        assert result == pytest.approx(expected, rel=1e-6)
