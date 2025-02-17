"""
test_credit.py
Unit tests for src/credit/ — hazard, Merton, CDS, CVA.
Targets 100% branch coverage of each module.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np
import pytest

from src.credit.cds import (
    cds_par_spread,
    cds_par_spread_constant_hazard,
    cds_spread_curve,
)
from src.credit.cva import (
    cva_discrete,
    epe_profile_from_mc,
    risky_bond_price,
)
from src.credit.hazard import (
    credit_spread,
    cumulative_default_prob,
    default_density,
    interval_default_prob,
    risky_zcb_price,
    survival,
    survival_piecewise,
)
from src.credit.merton import (
    merton_credit_spread,
    merton_d1_d2,
    merton_debt,
    merton_equity,
    merton_implied_B,
    merton_pd,
)


# ── hazard.py ─────────────────────────────────────────────────────────────────

class TestHazard:
    def test_survival_zero_hazard_is_one(self):
        assert survival(5.0, 0.0) == 1.0

    def test_survival_decays(self):
        assert survival(1.0, 0.05) < 1.0
        assert survival(10.0, 0.05) < survival(1.0, 0.05)

    def test_survival_negative_t_raises(self):
        with pytest.raises(ValueError, match="t must be non-negative"):
            survival(-0.1, 0.05)

    def test_survival_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda must be non-negative"):
            survival(1.0, -0.01)

    def test_default_density_matches_lambda_s(self):
        lam = 0.07
        t = 2.0
        assert default_density(t, lam) == pytest.approx(lam * survival(t, lam))

    def test_interval_default_prob(self):
        lam = 0.05
        p = interval_default_prob(1.0, 3.0, lam)
        assert p == pytest.approx(math.exp(-lam) - math.exp(-3 * lam))

    def test_interval_default_prob_reverse_raises(self):
        with pytest.raises(ValueError, match="t2 must be >= t1"):
            interval_default_prob(3.0, 1.0, 0.05)

    def test_cumulative_default_prob(self):
        lam = 0.03
        assert cumulative_default_prob(5.0, lam) == pytest.approx(1 - math.exp(-5 * lam))

    # Piecewise
    def test_survival_piecewise_interior(self):
        s = survival_piecewise(1.5, grid=[0, 1, 2, 5], hazards=[0.01, 0.02, 0.03])
        # Integral to 1.5: 0.01*1 + 0.02*0.5
        assert s == pytest.approx(math.exp(-(0.01 + 0.01)))

    def test_survival_piecewise_at_knot(self):
        s = survival_piecewise(2.0, grid=[0, 1, 2, 5], hazards=[0.01, 0.02, 0.03])
        assert s == pytest.approx(math.exp(-(0.01 + 0.02)))

    def test_survival_piecewise_beyond_last_knot_extrapolates(self):
        s = survival_piecewise(6.0, grid=[0, 1, 2, 5], hazards=[0.01, 0.02, 0.03])
        # 0.01 + 0.02 + 0.03*3 + 0.03*1
        expected = math.exp(-(0.01 + 0.02 + 0.03 * 3 + 0.03 * 1))
        assert s == pytest.approx(expected)

    def test_survival_piecewise_t_zero(self):
        s = survival_piecewise(0.0, grid=[0, 1, 2, 5], hazards=[0.01, 0.02, 0.03])
        assert s == 1.0

    def test_survival_piecewise_bad_grid_start_raises(self):
        with pytest.raises(ValueError, match="grid must start at 0"):
            survival_piecewise(1.0, grid=[0.1, 1, 2], hazards=[0.01, 0.02])

    def test_survival_piecewise_non_monotonic_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            survival_piecewise(1.0, grid=[0, 2, 1], hazards=[0.01, 0.02])

    def test_survival_piecewise_bad_hazard_length_raises(self):
        with pytest.raises(ValueError, match="len\\(hazards\\) must be len\\(grid\\) − 1"):
            survival_piecewise(1.0, grid=[0, 1, 2], hazards=[0.01])

    def test_survival_piecewise_negative_hazard_raises(self):
        with pytest.raises(ValueError, match="hazards must be non-negative"):
            survival_piecewise(1.0, grid=[0, 1, 2], hazards=[0.01, -0.01])

    def test_survival_piecewise_negative_t_raises(self):
        with pytest.raises(ValueError, match="t must be non-negative"):
            survival_piecewise(-1.0, grid=[0, 1, 2], hazards=[0.01, 0.02])

    # Risky ZCB + spread
    def test_risky_zcb_price(self):
        p = risky_zcb_price(r=0.05, T=1.0, LGD=0.6, s_T=0.97)
        expected = math.exp(-0.05) * (1 - 0.6 * 0.03)
        assert p == pytest.approx(expected)

    def test_risky_zcb_bad_LGD(self):
        with pytest.raises(ValueError, match="LGD must be in"):
            risky_zcb_price(r=0.05, T=1.0, LGD=-0.1, s_T=0.97)

    def test_risky_zcb_bad_s_T(self):
        with pytest.raises(ValueError, match="s\\(T\\) must be in"):
            risky_zcb_price(r=0.05, T=1.0, LGD=0.6, s_T=1.1)

    def test_risky_zcb_bad_T(self):
        with pytest.raises(ValueError, match="T must be positive"):
            risky_zcb_price(r=0.05, T=0.0, LGD=0.6, s_T=0.9)

    def test_credit_spread(self):
        s = credit_spread(T=1.0, LGD=0.6, s_T=0.97)
        expected = -math.log(1 - 0.6 * 0.03)
        assert s == pytest.approx(expected)

    def test_credit_spread_bad_T(self):
        with pytest.raises(ValueError, match="T must be positive"):
            credit_spread(T=0.0, LGD=0.6, s_T=0.97)

    def test_credit_spread_bad_inside(self):
        with pytest.raises(ValueError, match="must be < 1"):
            credit_spread(T=1.0, LGD=1.0, s_T=0.0)


# ── merton.py ─────────────────────────────────────────────────────────────────

class TestMerton:
    def test_d1_d2_relationship(self):
        d1, d2 = merton_d1_d2(V0=100, B=80, nu=0.05, sigma=0.25, T=1.0)
        assert d1 == pytest.approx(d2 + 0.25)

    def test_pd_between_zero_and_one(self):
        pd_ = merton_pd(V0=100, B=80, nu=0.05, sigma=0.25, T=1.0)
        assert 0.0 < pd_ < 1.0

    def test_equity_debt_sum_to_V0(self):
        V0 = 120.0
        E = merton_equity(V0=V0, B=80, r=0.05, sigma=0.25, T=1.0)
        D = merton_debt(V0=V0, B=80, r=0.05, sigma=0.25, T=1.0)
        assert E + D == pytest.approx(V0)

    def test_credit_spread_is_positive(self):
        cs = merton_credit_spread(V0=100, B=80, r=0.05, sigma=0.25, T=1.0)
        assert cs > 0

    def test_credit_spread_non_positive_debt_raises(self):
        # Contrive D0 <= 0 by making equity exceed V0: V0 >> B with very long T and low rate.
        # Easier: negative face B0 case is blocked; instead use a B so small that D0<=0 never happens naturally.
        # Force with sigma huge so equity blows up? That's not reliable. Use monkey-patch path:
        from src.credit import merton as m
        orig = m.merton_debt
        try:
            m.merton_debt = lambda *a, **kw: -1.0
            with pytest.raises(ValueError, match="non-positive"):
                m.merton_credit_spread(V0=100, B=80, r=0.05, sigma=0.25, T=1.0)
        finally:
            m.merton_debt = orig

    def test_implied_B_matches_target_survival(self):
        # Invert, then forward-compute PD: 1 - PD should equal target.
        target = 0.90
        B = merton_implied_B(V0=100, target_survival=target, r=0.04, sigma=0.3, T=2.0)
        pd_ = merton_pd(V0=100, B=B, nu=0.04, sigma=0.3, T=2.0)
        assert 1 - pd_ == pytest.approx(target, abs=1e-6)

    def test_implied_B_bad_target_raises(self):
        with pytest.raises(ValueError, match="target_survival"):
            merton_implied_B(V0=100, target_survival=0.0, r=0.04, sigma=0.3, T=1.0)
        with pytest.raises(ValueError, match="target_survival"):
            merton_implied_B(V0=100, target_survival=1.0, r=0.04, sigma=0.3, T=1.0)

    def test_validate_V0(self):
        with pytest.raises(ValueError, match="V0 must be positive"):
            merton_d1_d2(V0=0.0, B=80, nu=0.05, sigma=0.25, T=1.0)

    def test_validate_B(self):
        with pytest.raises(ValueError, match="B must be positive"):
            merton_d1_d2(V0=100, B=0.0, nu=0.05, sigma=0.25, T=1.0)

    def test_validate_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            merton_d1_d2(V0=100, B=80, nu=0.05, sigma=0.0, T=1.0)

    def test_validate_T(self):
        with pytest.raises(ValueError, match="T must be positive"):
            merton_d1_d2(V0=100, B=80, nu=0.05, sigma=0.25, T=0.0)


# ── cds.py ────────────────────────────────────────────────────────────────────

class TestCDS:
    def test_const_hazard_approx_landmark(self):
        # §14 sheet landmark
        s = cds_par_spread_constant_hazard(0.03, 0.40)
        assert abs(s * 1e4 - 180.0) < 1e-6

    def test_const_hazard_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda must be non-negative"):
            cds_par_spread_constant_hazard(-0.01, 0.4)

    def test_const_hazard_bad_R(self):
        with pytest.raises(ValueError, match="R must be in"):
            cds_par_spread_constant_hazard(0.03, 1.5)

    def test_cds_par_spread_positive(self):
        s = cds_par_spread(
            payment_times=[1, 2, 3], hazards=[0.03] * 3, r=0.03, R=0.40,
        )
        assert s > 0

    def test_cds_par_spread_no_accrual_is_smaller(self):
        s_a = cds_par_spread([1, 2, 3], [0.03] * 3, r=0.03, R=0.4, accrual=True)
        s_no = cds_par_spread([1, 2, 3], [0.03] * 3, r=0.03, R=0.4, accrual=False)
        # With accrual the denominator is larger → spread is smaller.
        assert s_no > s_a

    def test_cds_par_spread_bad_schedule_raises(self):
        with pytest.raises(ValueError, match="payment_times"):
            cds_par_spread([0.0, 1, 2], [0.03] * 3, r=0.03, R=0.4)
        with pytest.raises(ValueError, match="payment_times"):
            cds_par_spread([1, 0.5, 2], [0.03] * 3, r=0.03, R=0.4)

    def test_cds_par_spread_bad_hazards_length(self):
        with pytest.raises(ValueError, match="len\\(hazards\\)"):
            cds_par_spread([1, 2, 3], [0.03, 0.03], r=0.03, R=0.4)

    def test_cds_par_spread_bad_R(self):
        with pytest.raises(ValueError, match="R must be in"):
            cds_par_spread([1, 2], [0.03, 0.03], r=0.03, R=-0.1)

    def test_cds_par_spread_zero_hazard_raises(self):
        # λ=0 → no defaults → denominator still > 0, numerator 0 → spread = 0.
        s = cds_par_spread([1, 2, 3], [0.0] * 3, r=0.03, R=0.4)
        assert s == 0.0

    def test_cds_par_spread_bad_denominator_raises(self):
        # Contrive by zeroing n_sub to 0? Spec says n_sub > 0. Actually denominator
        # only fails if accrual=False AND all premium_leg terms vanish. Force with r huge + T tiny.
        # Easier: denominator check is exercised when grid_yields no contribution.
        # Use n_sub=1, payment_times tiny, r enormous → premium_leg > 0 still.
        # Instead validate by monkey-patching the numerical loop.
        from src.credit import cds as c
        with pytest.raises(ValueError, match="non-positive"):
            # Construct a degenerate case: r extremely high collapses premium_leg to ~0,
            # but it will still be > 0 in float. Use accrual=False and a truly empty spectrum:
            # simplest reliable path is to monkey-patch math.exp to return 0.
            import math as _m
            orig_exp = _m.exp
            try:
                _m.exp = lambda x: 0.0
                c.cds_par_spread([1.0], [0.03], r=0.03, R=0.4, accrual=False)
            finally:
                _m.exp = orig_exp

    def test_cds_spread_curve_shape(self):
        curve = cds_spread_curve([1, 2, 5], lam=0.03, r=0.03, R=0.40)
        assert len(curve) == 3
        for tenor, spread in curve:
            assert tenor > 0
            assert spread > 0

    def test_cds_spread_curve_quarterly(self):
        curve = cds_spread_curve([1.0], lam=0.03, r=0.03, R=0.40, premium_freq=4.0)
        assert len(curve) == 1


# ── cva.py ────────────────────────────────────────────────────────────────────

class TestCVA:
    def test_cva_discrete_basic(self):
        cva = cva_discrete(exposures=[10, 20, 30], marginal_default_probs=[0.01, 0.02, 0.03], R=0.4)
        expected = 0.6 * (10 * 0.01 + 20 * 0.02 + 30 * 0.03)
        assert cva == pytest.approx(expected)

    def test_cva_discrete_length_mismatch(self):
        with pytest.raises(ValueError, match="len\\(exposures\\)"):
            cva_discrete([10, 20], [0.01], R=0.4)

    def test_cva_discrete_negative_exposure(self):
        with pytest.raises(ValueError, match="exposures must be non-negative"):
            cva_discrete([10, -5, 30], [0.01, 0.02, 0.03], R=0.4)

    def test_cva_discrete_bad_pd_negative(self):
        with pytest.raises(ValueError, match="marginal_default_probs"):
            cva_discrete([10, 20, 30], [0.01, -0.02, 0.03], R=0.4)

    def test_cva_discrete_bad_pd_sum(self):
        with pytest.raises(ValueError, match="marginal_default_probs"):
            cva_discrete([10, 20, 30], [0.5, 0.5, 0.5], R=0.4)

    def test_cva_discrete_bad_R(self):
        with pytest.raises(ValueError, match="R must be in"):
            cva_discrete([10, 20, 30], [0.01, 0.02, 0.03], R=1.5)

    def test_risky_bond_price_reduces_with_hazard(self):
        times = [1, 2, 3]
        s_high = [0.99, 0.98, 0.97]
        s_low = [0.90, 0.80, 0.70]
        v_high = risky_bond_price(coupon=0.05, freq=1, times=times, r=0.03,
                                  survival_of_t=s_high, R=0.4)
        v_low = risky_bond_price(coupon=0.05, freq=1, times=times, r=0.03,
                                 survival_of_t=s_low, R=0.4)
        assert v_high > v_low

    def test_risky_bond_price_bad_length(self):
        with pytest.raises(ValueError, match="times and survival_of_t"):
            risky_bond_price(coupon=0.05, freq=1, times=[1, 2, 3],
                             r=0.03, survival_of_t=[0.99, 0.98], R=0.4)

    def test_risky_bond_price_non_monotonic(self):
        with pytest.raises(ValueError, match="times must be strictly increasing"):
            risky_bond_price(coupon=0.05, freq=1, times=[1, 3, 2],
                             r=0.03, survival_of_t=[0.99, 0.98, 0.97], R=0.4)

    def test_risky_bond_price_bad_s(self):
        with pytest.raises(ValueError, match="survival_of_t must be in"):
            risky_bond_price(coupon=0.05, freq=1, times=[1, 2],
                             r=0.03, survival_of_t=[0.99, 1.5], R=0.4)

    def test_risky_bond_price_bad_R(self):
        with pytest.raises(ValueError, match="R must be in"):
            risky_bond_price(coupon=0.05, freq=1, times=[1, 2],
                             r=0.03, survival_of_t=[0.99, 0.98], R=1.5)

    def test_risky_bond_price_bad_freq(self):
        with pytest.raises(ValueError, match="freq must be positive"):
            risky_bond_price(coupon=0.05, freq=0, times=[1, 2],
                             r=0.03, survival_of_t=[0.99, 0.98], R=0.4)

    def test_epe_profile_1d(self):
        paths = np.array([100.0, 110.0, 120.0])
        epe = epe_profile_from_mc(paths, V0=105.0)
        # max(paths - 105, 0) = [0, 5, 15], mean = 20/3
        assert len(epe) == 1
        assert epe[0] == pytest.approx((0 + 5 + 15) / 3)

    def test_epe_profile_2d(self):
        paths = np.array([[100, 120], [110, 90], [120, 100]], dtype=float)
        epe = epe_profile_from_mc(paths, V0=105.0)
        # col 0: max(.-105, 0) = [0,5,15] mean=20/3
        # col 1: max(.-105, 0) = [15,0,0] mean=5
        assert len(epe) == 2
        assert epe[0] == pytest.approx(20 / 3)
        assert epe[1] == pytest.approx(5.0)
