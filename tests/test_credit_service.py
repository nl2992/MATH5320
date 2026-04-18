"""
test_credit_service.py
Unit tests for src/services/credit_service.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.services.credit_service import (
    cds_spread_for_schedule,
    cds_summary,
    cva_summary,
    epe_from_portfolio_mc,
    interval_default_table,
    merton_implied_B_for_survival,
    merton_summary,
    reduced_form_summary,
)


class TestReducedForm:
    def test_reduced_form_summary_basic(self):
        out = reduced_form_summary(lam=0.03, horizons=[1, 2, 5], R=0.4, r=0.03)
        assert out["LGD"] == pytest.approx(0.6)
        assert out["approx_cds"] == pytest.approx(0.018)
        assert len(out["rows"]) == 3
        for row in out["rows"]:
            assert 0 < row["survival"] < 1
            assert 0 < row["cum_default"] < 1

    def test_reduced_form_summary_zero_horizon(self):
        # t=0 path hits the `nan` branch for risky_zcb/spread
        out = reduced_form_summary(lam=0.03, horizons=[0, 1], R=0.4, r=0.03)
        row0 = out["rows"][0]
        assert row0["t"] == 0.0
        assert row0["survival"] == 1.0
        # The nan branch sets risky_zcb and spread to nan.
        import math
        assert math.isnan(row0["risky_zcb"])
        assert math.isnan(row0["spread"])

    def test_interval_default_table(self):
        tbl = interval_default_table(edges=[0, 1, 2, 5], lam=0.03)
        assert len(tbl) == 3
        for row in tbl:
            assert row["marginal_default"] > 0


class TestMertonService:
    def test_merton_summary_contents(self):
        snap = merton_summary(V0=100, B=80, r=0.05, mu=0.02, sigma=0.25, T=1.0)
        assert set(snap.keys()) == {"V0", "B", "T", "Q", "P", "E0", "D0"}
        assert set(snap["Q"].keys()) == {"d1", "d2", "PD"}
        assert set(snap["P"].keys()) == {"d1", "d2", "PD"}
        assert 0 < snap["Q"]["PD"] < 1
        # μ < r ⇒ P-PD > Q-PD
        assert snap["P"]["PD"] > snap["Q"]["PD"]
        assert snap["E0"] + snap["D0"] == pytest.approx(100.0)

    def test_merton_implied_B_wrapper(self):
        B = merton_implied_B_for_survival(V0=100, target_survival=0.95, r=0.04,
                                          sigma=0.25, T=1.0)
        assert B > 0


class TestCDSService:
    def test_cds_summary(self):
        out = cds_summary(lam=0.03, R=0.4, tenors=[1, 3, 5])
        assert out["approx_spread"] == pytest.approx(0.018)
        assert len(out["curve"]) == 3
        for tenor, spread in out["curve"]:
            assert tenor > 0
            assert spread > 0

    def test_cds_spread_for_schedule(self):
        s = cds_spread_for_schedule(
            payment_times=[1, 2, 3], hazards=[0.02, 0.03, 0.04], r=0.03, R=0.4,
        )
        assert s > 0


class TestCVAService:
    def test_cva_summary_basic(self):
        out = cva_summary(
            exposure_profile=[10, 20, 30],
            marginal_default_probs=[0.01, 0.02, 0.03],
            R=0.4,
        )
        assert out["cva"] > 0
        assert out["R"] == 0.4
        assert "cva_pct" not in out  # V0 not supplied

    def test_cva_summary_with_V0(self):
        out = cva_summary(
            exposure_profile=[10, 20, 30],
            marginal_default_probs=[0.01, 0.02, 0.03],
            R=0.4,
            V0=1_000.0,
        )
        assert out["cva_pct"] == pytest.approx(out["cva"] / 1_000.0)

    def test_cva_summary_ignores_nonpositive_V0(self):
        out = cva_summary([10, 20, 30], [0.01, 0.02, 0.03], R=0.4, V0=0.0)
        assert "cva_pct" not in out

    def test_epe_from_portfolio_mc(self):
        paths = np.array([100.0, 110.0, 120.0])
        epe = epe_from_portfolio_mc(paths, V0=105.0)
        assert len(epe) == 1
        assert epe[0] == pytest.approx((0 + 5 + 15) / 3)
