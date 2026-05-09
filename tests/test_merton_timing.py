"""
tests/test_merton_timing.py
Unit tests for merton_survival_step and merton_interval_default_prob.

These encode the Merton timing defect: under the basic Merton model, default
can only occur at maturity T.  Between-period default probability is zero for
any interval that does not straddle T.
"""
import pytest
from src.credit.merton import merton_survival_step, merton_interval_default_prob


# ── merton_survival_step ──────────────────────────────────────────────────────

class TestMertonSurvivalStep:
    """s(u) = 1 for u < T; s(u) = 1 − pd_T for u >= T."""

    def test_before_maturity_is_one(self):
        assert merton_survival_step(u=1.0, T=5.0, pd_T=0.10) == 1.0

    def test_at_maturity_returns_one_minus_pd(self):
        assert merton_survival_step(u=5.0, T=5.0, pd_T=0.10) == pytest.approx(0.90)

    def test_after_maturity_returns_one_minus_pd(self):
        assert merton_survival_step(u=6.0, T=5.0, pd_T=0.20) == pytest.approx(0.80)

    def test_zero_pd_always_one(self):
        for u in [0.0, 4.99, 5.0, 5.01]:
            result = merton_survival_step(u=u, T=5.0, pd_T=0.0)
            assert result == pytest.approx(1.0), f"failed at u={u}"

    def test_pd_one_returns_zero_at_or_after_T(self):
        assert merton_survival_step(u=5.0, T=5.0, pd_T=1.0) == pytest.approx(0.0)
        assert merton_survival_step(u=10.0, T=5.0, pd_T=1.0) == pytest.approx(0.0)

    def test_just_before_maturity_is_one(self):
        assert merton_survival_step(u=4.9999, T=5.0, pd_T=0.30) == 1.0

    def test_u_zero_is_one(self):
        assert merton_survival_step(u=0.0, T=5.0, pd_T=0.15) == 1.0

    # ── validation ────────────────────────────────────────────────────────────

    def test_negative_u_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            merton_survival_step(u=-0.1, T=5.0, pd_T=0.10)

    def test_nonpositive_T_raises(self):
        with pytest.raises(ValueError, match="positive"):
            merton_survival_step(u=1.0, T=0.0, pd_T=0.10)
        with pytest.raises(ValueError, match="positive"):
            merton_survival_step(u=1.0, T=-1.0, pd_T=0.10)

    def test_pd_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            merton_survival_step(u=1.0, T=5.0, pd_T=-0.01)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            merton_survival_step(u=1.0, T=5.0, pd_T=1.01)


# ── merton_interval_default_prob ──────────────────────────────────────────────

class TestMertonIntervalDefaultProb:
    """
    P(t1 < τ ≤ t2) = pd_T if t1 < T ≤ t2, else 0.
    This is the Merton timing defect.
    """

    def test_interval_straddles_T_returns_pd(self):
        # t1 < T <= t2  →  pd_T
        assert merton_interval_default_prob(3.0, 5.0, T=5.0, pd_T=0.10) == pytest.approx(0.10)

    def test_interval_includes_T_exactly_at_t2(self):
        # t2 == T, t1 < T  →  returns pd_T
        assert merton_interval_default_prob(4.0, 5.0, T=5.0, pd_T=0.25) == pytest.approx(0.25)

    def test_interval_before_T_returns_zero(self):
        # t2 < T  →  0
        assert merton_interval_default_prob(1.0, 4.0, T=5.0, pd_T=0.20) == pytest.approx(0.0)

    def test_interval_after_T_returns_zero(self):
        # t1 > T  →  0  (default already occurred at T if it happened)
        assert merton_interval_default_prob(5.1, 7.0, T=5.0, pd_T=0.20) == pytest.approx(0.0)

    def test_t1_equals_T_returns_zero(self):
        # t1 == T means T is not in (t1, t2]  →  0
        assert merton_interval_default_prob(5.0, 6.0, T=5.0, pd_T=0.20) == pytest.approx(0.0)

    def test_full_span_returns_pd(self):
        # 0 < T <= large t2
        assert merton_interval_default_prob(0.0, 10.0, T=5.0, pd_T=0.15) == pytest.approx(0.15)

    def test_zero_pd_always_zero(self):
        assert merton_interval_default_prob(0.0, 5.0, T=5.0, pd_T=0.0) == pytest.approx(0.0)
        assert merton_interval_default_prob(5.0, 6.0, T=5.0, pd_T=0.0) == pytest.approx(0.0)

    def test_consecutive_intervals_sum_to_pd(self):
        """P(0,5] + P(5,10] should equal P(0,10] = pd_T (one default event at T=5)."""
        pd = 0.12
        p1 = merton_interval_default_prob(0.0, 5.0, T=5.0, pd_T=pd)
        p2 = merton_interval_default_prob(5.0, 10.0, T=5.0, pd_T=pd)
        # T is at endpoint of p1 interval, not in p2
        assert p1 + p2 == pytest.approx(pd)

    def test_period_before_T_plus_straddling_period_equals_pd(self):
        """P(0,3] + P(3,5] should equal pd_T when T=5."""
        pd = 0.18
        p1 = merton_interval_default_prob(0.0, 3.0, T=5.0, pd_T=pd)   # 0
        p2 = merton_interval_default_prob(3.0, 5.0, T=5.0, pd_T=pd)   # pd
        assert p1 + p2 == pytest.approx(pd)

    # ── validation ────────────────────────────────────────────────────────────

    def test_t2_less_than_t1_raises(self):
        with pytest.raises(ValueError, match="t2 must be"):
            merton_interval_default_prob(5.0, 3.0, T=5.0, pd_T=0.10)

    def test_nonpositive_T_raises(self):
        with pytest.raises(ValueError, match="positive"):
            merton_interval_default_prob(1.0, 6.0, T=0.0, pd_T=0.10)

    def test_pd_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            merton_interval_default_prob(1.0, 6.0, T=5.0, pd_T=1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            merton_interval_default_prob(1.0, 6.0, T=5.0, pd_T=-0.01)
