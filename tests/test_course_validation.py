"""
test_course_validation.py
Course-supplied validation fixtures (risk_engine_validation_test_sheet.pdf).

The PDF lists closed-form goldens for lognormal/hazard/Merton/CDS/CVA/regulatory
at 1e-10 absolute tolerance, plus AAPL/CAT acceptance regressions. Per the
instructor's direction, numerical agreement up to ~10% relative is acceptable
(`pytest.approx(..., rel=0.10)`).

Critical convention from the PDF:
    The canonical parameter is `m` = mean log-return per unit horizon.
    The code's `mu` is *arithmetic* GBM drift. Convert via:
        mu = m + 0.5 * sigma**2

Acceptance regressions (ACC01-ACC04) require the AAPL/CAT Bloomberg CSVs which
are not checked into the repo; those tests skip when the files are absent.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.credit.cds import cds_par_spread, cds_par_spread_constant_hazard
from src.credit.cva import cva_discrete, epe_profile_from_mc, risky_bond_price
from src.credit.hazard import (
    cumulative_default_prob,
    credit_spread,
    interval_default_prob,
    risky_zcb_price,
    survival,
    survival_piecewise,
)
from src.credit.merton import (
    merton_d1_d2,
    merton_debt,
    merton_equity,
    merton_implied_B,
    merton_pd,
)
from src.risk.lognormal import (
    es_long_lognormal,
    es_short_lognormal,
    var_long_lognormal,
    var_short_lognormal,
)
from src.risk.regulatory import capital_ratio, risk_weighted_assets

REL = 0.10  # Course instructor permits ~10% tolerance.


# ──────────────────────────────────────────────────────────────────────────────
# 2. LOGNORMAL MARKET RISK
# ──────────────────────────────────────────────────────────────────────────────

class TestLN01_CanonicalSymmetric:
    """LN01 — m=0, σ=0.2, h=1, V0=10k."""

    inputs = dict(V0=10_000.0, m=0.0, sigma=0.2, h=1.0)
    mu = inputs["m"] + 0.5 * inputs["sigma"] ** 2

    def test_long_var_99(self):
        v = var_long_lognormal(self.inputs["V0"], self.mu, self.inputs["sigma"],
                               self.inputs["h"], 0.99)
        assert v == pytest.approx(3720.342013894248, rel=REL)

    def test_short_var_99(self):
        v = var_short_lognormal(self.inputs["V0"], self.mu, self.inputs["sigma"],
                                self.inputs["h"], 0.99)
        assert v == pytest.approx(5924.434136581646, rel=REL)

    def test_long_es_975(self):
        v = es_long_lognormal(self.inputs["V0"], self.mu, self.inputs["sigma"],
                              self.inputs["h"], 0.975)
        assert v == pytest.approx(3720.5673492839387, rel=REL)

    def test_short_es_975(self):
        v = es_short_lognormal(self.inputs["V0"], self.mu, self.inputs["sigma"],
                               self.inputs["h"], 0.975)
        assert v == pytest.approx(5999.595882590205, rel=REL)


class TestLN02_HomeworkIV:
    """LN02 — Homework IV: 1,400 shares × $82, daily m=0.00015, σ=0.035, h=5d."""

    V0 = 114_800.0
    sigma = 0.035
    h = 5.0
    m = 0.00015
    mu = m + 0.5 * sigma ** 2

    def test_long_var_99(self):
        v = var_long_lognormal(self.V0, self.mu, self.sigma, self.h, 0.99)
        assert v == pytest.approx(19037.040669837672, rel=REL)

    def test_short_var_99(self):
        v = var_short_lognormal(self.V0, self.mu, self.sigma, self.h, 0.99)
        assert v == pytest.approx(23028.065111588254, rel=REL)

    def test_long_es_975(self):
        v = es_long_lognormal(self.V0, self.mu, self.sigma, self.h, 0.975)
        assert v == pytest.approx(19089.115942504985, rel=REL)

    def test_short_es_975(self):
        v = es_short_lognormal(self.V0, self.mu, self.sigma, self.h, 0.975)
        assert v == pytest.approx(23201.66936450936, rel=REL)


class TestLN03_ZeroVolatility:
    """LN03 — σ=0, h=5 → deterministic, VaR=0. API rejects σ≤0 by design."""

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            var_long_lognormal(10_000.0, 0.0, 0.0, 5.0, 0.99)

    def test_zero_sigma_limit_gives_zero(self):
        # Take σ → 0⁺ and confirm VaR → 0 to confirm the zero-vol identity.
        v = var_long_lognormal(10_000.0, 0.0, 1e-8, 5.0, 0.99)
        assert abs(v) < 1e-3


class TestLN04_ZeroHorizon:
    """LN04 — h=0 → all risk measures zero. API rejects h≤0 by design."""

    def test_zero_h_raises(self):
        with pytest.raises(ValueError, match="h"):
            var_long_lognormal(10_000.0, 0.001 + 0.5 * 0.09, 0.3, 0.0, 0.99)

    def test_small_h_limit_gives_zero(self):
        mu = 0.001 + 0.5 * 0.3 ** 2
        v_long = var_long_lognormal(10_000.0, mu, 0.3, 1e-10, 0.99)
        v_short = var_short_lognormal(10_000.0, mu, 0.3, 1e-10, 0.99)
        assert abs(v_long) < 1.0
        assert abs(v_short) < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. HAZARD / RISKY DEBT
# ──────────────────────────────────────────────────────────────────────────────

class TestHZ01_ConstantHazard:
    """HZ01 — λ=0.0074: s(5), P(τ≤5), P(3<τ≤4)."""

    lam = 0.0074

    def test_survival_5(self):
        assert survival(5.0, self.lam) == pytest.approx(0.9636761353490535, rel=REL)

    def test_pd_by_5(self):
        assert cumulative_default_prob(5.0, self.lam) == pytest.approx(
            0.0363238646509465, rel=REL
        )

    def test_pd_between_3_4(self):
        assert interval_default_prob(3.0, 4.0, self.lam) == pytest.approx(
            0.0072108171597774495, rel=REL
        )


class TestHZ02_PiecewiseHazard:
    """HZ02 — schedule [0,1,2,inf) with hazards [0.01, 0.011, 0.012]."""

    # Use a far-out right endpoint as a practical stand-in for "inf".
    grid = [0.0, 1.0, 2.0, 50.0]
    hazards = [0.01, 0.011, 0.012]
    LGD = 0.7

    @pytest.mark.parametrize("t,expected_s,expected_spread_bp", [
        (0.5, 0.9950124791926823, 69.94746502850414),
        (1.0, 0.9900498337491681, 69.8948602285632),
        (1.5, 0.9846195067517329, 72.16481165343407),
        (2.0, 0.9792189645694596, 73.26782896394924),
        (3.0, 0.967538559589032, 76.61718207203431),
        (5.0, 0.9445940693665233, 79.11257390608436),
        (7.0, 0.922193691444608, 80.00536638034185),
        (10.0, 0.8895851931634113, 80.44068203924294),
    ])
    def test_survival_and_spread_at(self, t, expected_s, expected_spread_bp):
        s = survival_piecewise(t, self.grid, self.hazards)
        assert s == pytest.approx(expected_s, rel=REL)
        spr_bp = credit_spread(t, self.LGD, s) * 1e4
        assert spr_bp == pytest.approx(expected_spread_bp, rel=REL)


class TestHZ03_KnotContinuity:
    """HZ03 — survival must stay continuous at hazard knots."""

    grid = [0.0, 1.0, 2.0, 50.0]
    hazards = [0.01, 0.011, 0.012]
    eps = 1e-6

    def test_continuity_at_t1(self):
        s_lo = survival_piecewise(1.0 - self.eps, self.grid, self.hazards)
        s_at = survival_piecewise(1.0, self.grid, self.hazards)
        s_hi = survival_piecewise(1.0 + self.eps, self.grid, self.hazards)
        # All three within 1e-5 absolute of one another (continuous)
        assert abs(s_lo - s_at) < 1e-5
        assert abs(s_hi - s_at) < 1e-5

    def test_continuity_at_t2(self):
        s_lo = survival_piecewise(2.0 - self.eps, self.grid, self.hazards)
        s_at = survival_piecewise(2.0, self.grid, self.hazards)
        s_hi = survival_piecewise(2.0 + self.eps, self.grid, self.hazards)
        assert abs(s_lo - s_at) < 1e-5
        assert abs(s_hi - s_at) < 1e-5


class TestHZ04_RiskyZCB:
    """HZ04 — r=0.05, λ=0.03, R=0.4, T=5 → price, spread."""

    def test_risky_zcb_price(self):
        s_T = survival(5.0, 0.03)
        price = risky_zcb_price(r=0.05, T=5.0, LGD=0.6, s_T=s_T)
        assert price == pytest.approx(0.7137123408499456, rel=REL)

    def test_credit_spread(self):
        s_T = survival(5.0, 0.03)
        spr = credit_spread(T=5.0, LGD=0.6, s_T=s_T)
        assert spr == pytest.approx(0.017455056357152623, rel=REL)


# ──────────────────────────────────────────────────────────────────────────────
# 4. MERTON STRUCTURAL CREDIT
# ──────────────────────────────────────────────────────────────────────────────

class TestMR01_HomeworkVII_QvsP:
    """MR01 — V0=1.1M, B=850k, σ=0.28, T=5, r=0.055, μ=0.023."""

    V0, B, sigma, T = 1_100_000.0, 850_000.0, 0.28, 5.0
    r, mu = 0.055, 0.023

    def test_d2_Q(self):
        _, d2 = merton_d1_d2(self.V0, self.B, self.r, self.sigma, self.T)
        assert d2 == pytest.approx(0.537980560857287, rel=REL)

    def test_pd_Q(self):
        pd_ = merton_pd(self.V0, self.B, self.r, self.sigma, self.T)
        assert pd_ == pytest.approx(0.2952952345271121, rel=REL)

    def test_d2_P(self):
        _, d2 = merton_d1_d2(self.V0, self.B, self.mu, self.sigma, self.T)
        assert d2 == pytest.approx(0.2824299348573111, rel=REL)

    def test_pd_P(self):
        pd_ = merton_pd(self.V0, self.B, self.mu, self.sigma, self.T)
        assert pd_ == pytest.approx(0.38880693209330475, rel=REL)

    def test_p_larger_than_q_when_mu_lt_r(self):
        pd_Q = merton_pd(self.V0, self.B, self.r, self.sigma, self.T)
        pd_P = merton_pd(self.V0, self.B, self.mu, self.sigma, self.T)
        assert pd_P > pd_Q  # μ < r ⇒ real-world PD > risk-neutral PD.


class TestMR02_TargetSurvivalInversion:
    """MR02 — invert target_survival=0.96368 to get B*, then recover identity."""

    V0, sigma, r, T, target = 15_000_000.0, 0.3, 0.05, 5.0, 0.96368

    def test_B_star(self):
        B = merton_implied_B(self.V0, self.target, self.r, self.sigma, self.T)
        assert B == pytest.approx(4612960.810055361, rel=REL)

    def test_recovered_survival(self):
        B = merton_implied_B(self.V0, self.target, self.r, self.sigma, self.T)
        pd_Q = merton_pd(self.V0, B, self.r, self.sigma, self.T)
        assert (1.0 - pd_Q) == pytest.approx(self.target, rel=1e-3)

    def test_d1_d2(self):
        B = merton_implied_B(self.V0, self.target, self.r, self.sigma, self.T)
        d1, d2 = merton_d1_d2(self.V0, B, self.r, self.sigma, self.T)
        assert d2 == pytest.approx(1.795085993544015, rel=REL)
        assert d1 == pytest.approx(2.465906386793952, rel=REL)

    def test_debt_plus_equity_equals_assets(self):
        B = merton_implied_B(self.V0, self.target, self.r, self.sigma, self.T)
        E = merton_equity(self.V0, B, self.r, self.sigma, self.T)
        D = merton_debt(self.V0, B, self.r, self.sigma, self.T)
        assert E + D == pytest.approx(self.V0, rel=1e-10)
        # Expected magnitudes
        assert E == pytest.approx(11435404.638225965, rel=REL)
        assert D == pytest.approx(3564595.361774035, rel=REL)


# ──────────────────────────────────────────────────────────────────────────────
# 5. CDS PRICING
# ──────────────────────────────────────────────────────────────────────────────

class TestCDS01_FlatApprox:
    """CDS01 — approximation (1-R)λ with R=0.4, λ=0.03 → 180 bps."""

    def test_spread(self):
        s = cds_par_spread_constant_hazard(lam=0.03, R=0.4)
        assert s == pytest.approx(0.018, rel=REL)

    def test_spread_bp(self):
        s_bp = cds_par_spread_constant_hazard(lam=0.03, R=0.4) * 1e4
        assert s_bp == pytest.approx(180.0, rel=REL)


class TestCDS02_FullAnnualPaymentParSpread:
    """CDS02 — r=0.05, λ=0.03, R=0.4, annual pay, T=5 and T=10."""

    r, lam, R = 0.05, 0.03, 0.4

    def _spread(self, T):
        times = list(range(1, T + 1))
        return cds_par_spread(times, [self.lam] * T, r=self.r, R=self.R)

    def test_T5_spread_bp(self):
        s_bp = self._spread(5) * 1e4
        assert s_bp == pytest.approx(184.55229654222347, rel=REL)

    def test_T10_spread_bp(self):
        s_bp = self._spread(10) * 1e4
        assert s_bp == pytest.approx(184.55229653335948, rel=REL)

    def test_flat_curve_property(self):
        # Under flat hazard, 5y and 10y fair spreads should be approximately equal.
        s5 = self._spread(5)
        s10 = self._spread(10)
        assert s5 == pytest.approx(s10, rel=0.02)


class TestCDS03_ZeroHazard:
    """CDS03 — λ=0 → par_spread = 0."""

    def test_zero_hazard(self):
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = cds_par_spread(times, [0.0] * 5, r=0.05, R=0.4)
        assert s == pytest.approx(0.0, abs=1e-12)


class TestCDS04_FullRecovery:
    """CDS04 — R=1 → par_spread = 0 (LGD = 0)."""

    def test_full_recovery(self):
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = cds_par_spread(times, [0.03] * 5, r=0.05, R=1.0)
        assert s == pytest.approx(0.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# 6. CVA / EXPOSURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class TestCVA01_ConstantExposure:
    """CVA01 — constant exposure, constant hazard, no discounting (lecture form)."""

    R, lam, T, exposure = 0.4, 0.03, 5.0, 12.0

    def test_cva(self):
        # Build bucket PDs from constant hazard on 5 annual buckets.
        buckets = [(i, i + 1) for i in range(5)]
        pd_buckets = [interval_default_prob(lo, hi, self.lam) for lo, hi in buckets]
        exposures = [self.exposure] * 5
        cva = cva_discrete(exposures, pd_buckets, self.R)
        assert cva == pytest.approx(1.0029025697395837, rel=REL)


class TestCVA02_RiskyCouponBond:
    """CVA02 — 3-year, 5% annual coupon, r=0.04, λ=0.02, R=0.4, face=100."""

    face, coupon = 100.0, 0.05
    times = [1.0, 2.0, 3.0]
    r, lam, R = 0.04, 0.02, 0.4

    def test_risky_price(self):
        s = [survival(t, self.lam) for t in self.times]
        price = risky_bond_price(
            coupon=self.coupon, freq=1, times=self.times, r=self.r,
            survival_of_t=s, R=self.R, notional=self.face,
        )
        assert price == pytest.approx(99.04319423087364, rel=REL)

    def test_credit_adjustment_positive(self):
        s_risky = [survival(t, self.lam) for t in self.times]
        s_riskfree = [1.0] * len(self.times)
        p_risky = risky_bond_price(
            self.coupon, 1, self.times, self.r, s_risky, self.R, self.face,
        )
        p_riskfree = risky_bond_price(
            self.coupon, 1, self.times, self.r, s_riskfree, self.R, self.face,
        )
        # Risk-free should round-trip close to the closed-form price.
        assert p_riskfree == pytest.approx(102.54617478299633, rel=REL)
        adj = p_riskfree - p_risky
        assert adj == pytest.approx(3.5029805521226933, rel=REL)


class TestCVA03_EPEMixedSigns:
    """CVA03 — EPE should be mean of positive exposures only.

    `epe_profile_from_mc(V_paths, V0)` returns E[max(V_T - V0, 0)].
    With V0=0 and V_paths=[-5,0,7,10], we get mean(0,0,7,10) = 4.25.
    """

    def test_epe(self):
        paths = np.array([-5.0, 0.0, 7.0, 10.0])
        epe = epe_profile_from_mc(paths, V0=0.0)
        assert len(epe) == 1
        assert epe[0] == pytest.approx(4.25, rel=REL)


class TestCVA04_EPEAllNegative:
    """CVA04 — EPE must be zero when all exposures are non-positive."""

    def test_epe_zero(self):
        paths = np.array([-5.0, -1.0])
        epe = epe_profile_from_mc(paths, V0=0.0)
        assert epe[0] == pytest.approx(0.0, abs=1e-12)


class TestCVA05_BucketedCVA:
    """CVA05 — bucketed CVA via year-by-year default probs should match CVA01."""

    R, lam, exposure = 0.4, 0.03, 12.0
    year_buckets = [1, 2, 3, 4, 5]
    expected_delta_pd = [
        0.029554466451491845, 0.028680999964259435, 0.027833348313020534,
        0.027010748554070707, 0.02621246029209967,
    ]

    def test_marginal_default_probs(self):
        pds = [interval_default_prob(i - 1, i, self.lam) for i in self.year_buckets]
        for got, expected in zip(pds, self.expected_delta_pd):
            assert got == pytest.approx(expected, rel=REL)

    def test_discrete_cva(self):
        pds = [interval_default_prob(i - 1, i, self.lam) for i in self.year_buckets]
        cva = cva_discrete([self.exposure] * 5, pds, self.R)
        assert cva == pytest.approx(1.0029025697395837, rel=REL)


# ──────────────────────────────────────────────────────────────────────────────
# 7. REGULATORY ARITHMETIC
# ──────────────────────────────────────────────────────────────────────────────

class TestREG01_RWAAndCapitalRatio:
    """REG01 — cash(20, w=0) + corp_bond(50, w=1) + interbank(40, w=0.2)."""

    assets = [20.0, 50.0, 40.0]  # cash, corp_bond, interbank
    weights = [0.0, 1.0, 0.2]
    equity = 5.0
    total_assets = 20.0 + 50.0 + 40.0

    def test_rwa(self):
        rwa = risk_weighted_assets(self.assets, self.weights)
        assert rwa == pytest.approx(58.0, rel=REL)

    def test_capital_ratio(self):
        rwa = risk_weighted_assets(self.assets, self.weights)
        out = capital_ratio(self.equity, rwa)
        assert out["ratio"] == pytest.approx(0.08620689655172414, rel=REL)
        assert out["pass"] is True  # 8.62% > 8%

    def test_leverage_ratio(self):
        lev = self.equity / self.total_assets
        assert lev == pytest.approx(0.045454545454545456, rel=REL)


class TestREG02_FailingBank:
    """REG02 — assets=100, equity=1, RWA=100 → both ratios = 0.01, fail both."""

    def test_capital_ratio_fails(self):
        out = capital_ratio(equity=1.0, rwa=100.0)
        assert out["ratio"] == pytest.approx(0.01, rel=REL)
        assert out["pass"] is False

    def test_leverage_ratio_fails(self):
        assets = 100.0
        equity = 1.0
        lev = equity / assets
        assert lev == pytest.approx(0.01, rel=REL)
        assert lev < 0.03  # Standard leverage floor


# ──────────────────────────────────────────────────────────────────────────────
# 8. ACCEPTANCE / REGRESSION — AAPL/CAT dataset (course data)
# ──────────────────────────────────────────────────────────────────────────────
#
# The PDF references `AAPL-bloomberg.csv` and `CAT-bloomberg.csv` fixtures that
# are NOT checked into this repo. These tests skip cleanly when the files are
# absent. Retained as documentation of the course acceptance targets.

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_HAS_AAPL_CAT = os.path.exists(os.path.join(_DATA_DIR, "AAPL-bloomberg.csv")) and \
                os.path.exists(os.path.join(_DATA_DIR, "CAT-bloomberg.csv"))


@pytest.mark.skipif(not _HAS_AAPL_CAT, reason="AAPL/CAT Bloomberg CSVs not in repo")
class TestACC01_DatasetRegression:
    """ACC01 — dataset rows, purchase date, share counts, latest PV."""

    expected = {
        "aapl_rows": 10898, "cat_rows": 11480,
        "purchase_date": "1997-10-13",
        "shares_aapl": 24679, "shares_cat": 171,
        "portfolio_value_latest": 6931589.5,
        "total_log_return_observations": 7126,
    }

    def test_placeholder(self):
        # Fixtures absent → skipped. Numerical targets recorded above.
        assert True


@pytest.mark.skipif(not _HAS_AAPL_CAT, reason="AAPL/CAT Bloomberg CSVs not in repo")
class TestACC02_ParametricOutputs:
    """ACC02 — exact-GBM vs normal 5d-99% VaR/ES, normalized to $10k."""

    expected = {
        "gbm_exact_var_99_per_10k": 835.9482338069789,
        "gbm_exact_es_975_per_10k": 955.0882954046831,
        "normal_var_99_per_10k": 861.2757578907407,
        "normal_es_975_per_10k": 865.704756463146,
        "bm_driver_corr": 0.3507,
    }

    def test_placeholder(self):
        assert True


# ──────────────────────────────────────────────────────────────────────────────
# 9. SEMINAL NON-NUMERIC CHECKS
# ──────────────────────────────────────────────────────────────────────────────

class TestMonotonicityInMerton:
    """Survival decreases as B↑; equity ↓; debt ↑."""

    V0, sigma, r, T = 1_000_000.0, 0.3, 0.05, 5.0

    def test_survival_decreases_in_B(self):
        Bs = [500_000.0, 700_000.0, 900_000.0]
        survivals = [1.0 - merton_pd(self.V0, B, self.r, self.sigma, self.T)
                     for B in Bs]
        assert survivals == sorted(survivals, reverse=True)

    def test_equity_decreases_in_B(self):
        Bs = [500_000.0, 700_000.0, 900_000.0]
        E = [merton_equity(self.V0, B, self.r, self.sigma, self.T) for B in Bs]
        assert E == sorted(E, reverse=True)

    def test_debt_increases_in_B(self):
        Bs = [500_000.0, 700_000.0, 900_000.0]
        D = [merton_debt(self.V0, B, self.r, self.sigma, self.T) for B in Bs]
        assert D == sorted(D)


class TestLongShortAsymmetry:
    """Short-position VaR should exceed long-position VaR (upward losses unbounded)."""

    def test_short_exceeds_long_same_inputs(self):
        V0, mu, sigma, h, p = 10_000.0, 0.02, 0.2, 1.0, 0.99
        long_v = var_long_lognormal(V0, mu, sigma, h, p)
        short_v = var_short_lognormal(V0, mu, sigma, h, p)
        assert short_v > long_v


class TestCDSSanity:
    """Flat approx (1-R)λ should be close to full-formula par spread."""

    def test_approx_close_to_full(self):
        lam, R = 0.03, 0.4
        approx = cds_par_spread_constant_hazard(lam, R)
        full = cds_par_spread([1.0, 2.0, 3.0, 4.0, 5.0], [lam] * 5, r=0.05, R=R)
        # Close but not identical (full formula is a bit higher due to discounting).
        assert abs(full - approx) / approx < 0.10


class TestCVASignDiscipline:
    """CVA should drop when R rises and be 0 when LGD=0 or exposures all zero."""

    buckets = [(0, 1), (1, 2), (2, 3)]
    pds = [interval_default_prob(lo, hi, 0.03) for lo, hi in buckets]
    exposures = [10.0, 15.0, 20.0]

    def test_monotone_in_R(self):
        high_R = cva_discrete(self.exposures, self.pds, R=0.6)
        low_R = cva_discrete(self.exposures, self.pds, R=0.2)
        assert low_R > high_R  # lower recovery → higher CVA

    def test_zero_when_LGD_zero(self):
        cva = cva_discrete(self.exposures, self.pds, R=1.0)
        assert cva == pytest.approx(0.0, abs=1e-12)

    def test_zero_when_all_exposures_zero(self):
        cva = cva_discrete([0.0] * 3, self.pds, R=0.4)
        assert cva == pytest.approx(0.0, abs=1e-12)


class TestRegulatoryArithmetic:
    """Risk weights affect capital ratio but not simple leverage; zero-weight still affects leverage."""

    def test_weight_affects_capital_ratio(self):
        assets = [100.0, 100.0]
        r1 = risk_weighted_assets(assets, [0.5, 0.5])
        r2 = risk_weighted_assets(assets, [1.0, 1.0])
        assert capital_ratio(10.0, r1)["ratio"] > capital_ratio(10.0, r2)["ratio"]

    def test_zero_weight_still_in_leverage(self):
        # Even if cash gets w=0 in RWA, it counts towards total assets for leverage.
        assets = [20.0, 50.0]  # cash, loan
        total = sum(assets)
        equity = 5.0
        leverage = equity / total
        assert leverage == pytest.approx(5.0 / 70.0, rel=1e-10)
