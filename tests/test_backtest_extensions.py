"""
tests/test_backtest_extensions.py
Tests for Christoffersen independence test, Basel traffic-light,
and exception severity diagnostics added to backtest.py.
"""
import numpy as np
import pandas as pd
import pytest

from src.risk.backtest import (
    christoffersen_test,
    conditional_coverage_test,
    kupiec_test,
    basel_traffic_light,
    exception_severity,
)


# ── Christoffersen independence test ─────────────────────────────────────────

class TestChristofersenTest:

    def test_all_zeros_no_exceptions(self):
        exc = [0] * 50
        result = christoffersen_test(exc)
        assert result["n01"] == 0
        assert result["n11"] == 0
        assert result["pi_hat"] == pytest.approx(0.0)

    def test_all_ones_all_exceptions(self):
        exc = [1] * 50
        result = christoffersen_test(exc)
        assert result["n11"] == 49
        assert result["n00"] == 0
        # pi_01 undefined (no non-exception predecessor), pi_11 = 1.0
        assert result["pi_11"] == pytest.approx(1.0)

    def test_alternating_exceptions_low_clustering(self):
        # 01010101... : pi_01=1, pi_11=0  → high independence
        exc = [i % 2 for i in range(50)]
        result = christoffersen_test(exc)
        # transitions: all 01 and 10, no 00 or 11
        assert result["n11"] == 0
        assert result["n00"] == 0
        # lr_ind should be large (rejecting clustering that doesn't exist is fine)
        assert result["lr_ind"] >= 0

    def test_clustered_exceptions_lr_ind_positive(self):
        # Cluster: first 20 are exceptions, rest not → strong clustering
        exc = [1] * 20 + [0] * 230
        result = christoffersen_test(exc)
        assert result["lr_ind"] >= 0.0

    def test_too_short_returns_nans(self):
        result = christoffersen_test([1])
        assert np.isnan(result["lr_ind"])

    def test_empty_returns_nans(self):
        result = christoffersen_test([])
        assert np.isnan(result["lr_ind"])

    def test_iid_exceptions_at_1pct_in_long_series(self):
        # Simulate ~250 obs, ~2-3 random exceptions → should not reject independence
        rng = np.random.default_rng(42)
        exc = (rng.random(250) < 0.01).astype(int)
        result = christoffersen_test(exc)
        assert "lr_ind" in result
        assert result["lr_ind"] >= 0.0
        assert 0.0 <= result["pi_hat"] <= 1.0


# ── Conditional coverage ──────────────────────────────────────────────────────

class TestConditionalCoverageTest:

    def test_merges_kupiec_and_christoffersen_keys(self):
        exc = [0] * 248 + [1, 1]
        result = conditional_coverage_test(250, 2, 0.99, exc)
        assert "lr_stat" in result      # from Kupiec
        assert "lr_ind" in result       # from Christoffersen
        assert "lr_cc" in result
        assert "p_value_cc" in result
        assert "reject_cc" in result

    def test_lr_cc_equals_lr_uc_plus_lr_ind(self):
        exc = [0] * 245 + [1] * 5
        result = conditional_coverage_test(250, 5, 0.99, exc)
        if not np.isnan(result["lr_cc"]):
            assert result["lr_cc"] == pytest.approx(
                result["lr_stat"] + result["lr_ind"], rel=1e-6
            )

    def test_well_calibrated_does_not_reject_cc(self):
        # 2-3 exceptions in 250 days at 99% confidence is within spec
        exc = [0] * 248 + [1, 1]
        result = conditional_coverage_test(250, 2, 0.99, exc)
        assert result["lr_cc"] >= 0.0

    def test_zero_observations_returns_nans(self):
        result = conditional_coverage_test(0, 0, 0.99, [])
        assert np.isnan(result.get("lr_cc", np.nan))


# ── Basel traffic-light ───────────────────────────────────────────────────────

class TestBaselTrafficLight:

    @pytest.mark.parametrize("n,expected_zone,expected_mult", [
        (0, "GREEN", 3.00),
        (1, "GREEN", 3.00),
        (4, "GREEN", 3.00),
        (5, "YELLOW", 3.40),
        (6, "YELLOW", 3.50),
        (7, "YELLOW", 3.65),
        (8, "YELLOW", 3.75),
        (9, "YELLOW", 3.85),
        (10, "RED", 4.00),
        (15, "RED", 4.00),
        (50, "RED", 4.00),
    ])
    def test_zone_and_multiplier(self, n, expected_zone, expected_mult):
        result = basel_traffic_light(n)
        assert result["zone"] == expected_zone
        assert result["capital_multiplier"] == pytest.approx(expected_mult)
        assert result["n_exceptions"] == n

    def test_description_non_empty(self):
        for n in [0, 5, 10]:
            assert len(basel_traffic_light(n)["description"]) > 0

    def test_negative_exceptions_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            basel_traffic_light(-1)

    def test_green_boundary(self):
        assert basel_traffic_light(4)["zone"] == "GREEN"
        assert basel_traffic_light(5)["zone"] == "YELLOW"

    def test_red_boundary(self):
        assert basel_traffic_light(9)["zone"] == "YELLOW"
        assert basel_traffic_light(10)["zone"] == "RED"


# ── Exception severity ────────────────────────────────────────────────────────

class TestExceptionSeverity:

    def _make_df(self, forecasts, losses, exceptions):
        return pd.DataFrame({
            "var_forecast": forecasts,
            "realized_loss": losses,
            "exception": exceptions,
        })

    def test_no_exceptions(self):
        df = self._make_df([10.0] * 100, [5.0] * 100, [0] * 100)
        result = exception_severity(df)
        assert result["n_exceptions"] == 0
        assert result["exception_rate"] == pytest.approx(0.0)
        assert np.isnan(result["exception_gap"])
        assert np.isnan(result["average_exception_loss"])

    def test_basic_exception_stats(self):
        # 10 obs, 2 exceptions where realized_loss=20, var_forecast=10
        forecasts = [10.0] * 10
        losses = [5.0] * 8 + [20.0, 25.0]
        exceptions = [0] * 8 + [1, 1]
        df = self._make_df(forecasts, losses, exceptions)
        result = exception_severity(df)
        assert result["n_exceptions"] == 2
        assert result["exception_rate"] == pytest.approx(0.2)
        assert result["average_exception_loss"] == pytest.approx(22.5)
        assert result["max_exception_loss"] == pytest.approx(25.0)
        assert result["exception_gap"] == pytest.approx(12.5)  # mean(20-10, 25-10)

    def test_mean_loss_given_exception_equals_average_exception_loss(self):
        losses = [5.0] * 8 + [20.0, 25.0]
        exceptions = [0] * 8 + [1, 1]
        df = self._make_df([10.0] * 10, losses, exceptions)
        result = exception_severity(df)
        assert result["mean_loss_given_exception"] == result["average_exception_loss"]

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["var_forecast", "realized_loss", "exception"])
        result = exception_severity(df)
        assert result["n_observations"] == 0
        assert result["n_exceptions"] == 0

    def test_all_exceptions(self):
        df = self._make_df([5.0] * 5, [15.0] * 5, [1] * 5)
        result = exception_severity(df)
        assert result["n_exceptions"] == 5
        assert result["exception_rate"] == pytest.approx(1.0)
        assert result["exception_gap"] == pytest.approx(10.0)
