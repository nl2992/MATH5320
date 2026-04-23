"""
test_homework_cases.py
Assessment-testable homework cases from:
  MATH5320_assessment_testable_homework_cases.docx

Covers READY_NOW and TESTABLE_IF_HELPER_EXISTS cases that are NOT already
present in test_course_validation.py (LN01-04, HZ01-04, MR01-02, CDS01-04,
CVA01-05, REG01-02 are covered there).

New cases added here
--------------------
READY_NOW
  HW3_GBM_FORMULA_VAR   - GBM EV / SD / 98% VaR at 1d, 5d, 1y
  HW4_SINGLE_STOCK      - sigma_annual, mu_annual, exponent, 5-day 99% VaR
  HW9_DISCRETE_CVA      - one-period risk-neutral CVA (HW9 structure)
  HW10_RWA_CAPITAL      - balance sheet: cash + mortgages + corporate bonds
  LN_MONOTONICITY       - VaR monotone in alpha and sigma
  HZ_EDGE_ZERO_HAZARD   - lambda=0 gives certain survival, zero spread
  HZ_NEGATIVE_HAZARD    - lambda<0 raises ValueError
  MR_ATM_SMOKE          - V0=B=100 Merton structural smoke test
  MR_INVALID_SURVIVAL   - target_survival >= 1 raises ValueError
  CVA_FULL_RECOVERY     - R=1 makes CVA zero
  CVA_LINEARITY         - CVA scales linearly with deterministic exposure

TESTABLE_IF_HELPER_EXISTS (all helpers exist in this repo)
  HW3_SCENARIO_VAR_ES   - 10-scenario empirical VaR / ES (pure arithmetic)
  HW4_TWO_STOCK_VAR     - 2-stock sum-of-lognormal moments parametric VaR
  HW5_LAMBDA_20PCT      - EWMA lambda: lambda^N = 0.20 (20% weight heuristic)
  HW5_LAMBDA_L2         - EWMA lambda: half-life = N/2 (L2-optimal)

OPTIONAL_EXTENSION (helpers exist in this repo)
  HW5_BS_DELTA_FD       - Black-Scholes call price and delta at ATM
  HW3_INTEL_BSM         - Intel delta-hedge call price, delta, shocked P&L
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.stats import norm

from src.credit.cds import cds_par_spread_constant_hazard
from src.credit.cva import cva_discrete, epe_profile_from_mc
from src.credit.hazard import (
    credit_spread,
    cumulative_default_prob,
    default_density,
    interval_default_prob,
    risky_zcb_price,
    survival,
)
from src.credit.merton import (
    merton_d1_d2,
    merton_debt,
    merton_equity,
    merton_implied_B,
    merton_pd,
)
from src.pricing.black_scholes import bs_delta, bs_price
from src.risk.lognormal import (
    es_long_lognormal,
    es_short_lognormal,
    var_long_lognormal,
    var_short_lognormal,
)
from src.risk.regulatory import capital_ratio, risk_weighted_assets

REL = 0.01   # 1% relative tolerance
ABS_MONEY = 1.0   # monetary tolerance for source-rounded homework outputs


# ──────────────────────────────────────────────────────────────────────────────
# READY_NOW
# ──────────────────────────────────────────────────────────────────────────────

class TestHW3_GBM_FormulaVaR:
    """
    HW3_GBM_FORMULA_VAR - GBM expected value, std-dev, and 98% VaR.
    Source: Homework 3. V0=29000, mu=0.04 (arithmetic drift), sigma=0.35.
    Horizons: 1d = 1/252, 5d = 5/252, 1y = 1.0.
    """
    V0, mu, sigma = 29000.0, 0.04, 0.35

    @staticmethod
    def _ev(V0, mu, h):
        return V0 * math.exp(mu * h)

    @staticmethod
    def _sd(V0, mu, sigma, h):
        return V0 * math.exp(mu * h) * math.sqrt(math.exp(sigma ** 2 * h) - 1)

    @pytest.mark.parametrize("label,h,exp_ev,exp_sd,exp_var98", [
        ("1d",  1/252,   29004.60354,   639.56912,  1286.19686),
        ("5d",  5/252,   29023.02501,  1431.72431,  2803.57329),
        ("1y",  1.0,     30183.51245, 10896.1694,  15164.55592),
    ])
    def test_ev_sd_var(self, label, h, exp_ev, exp_sd, exp_var98):
        ev = self._ev(self.V0, self.mu, h)
        sd = self._sd(self.V0, self.mu, self.sigma, h)
        var = var_long_lognormal(self.V0, self.mu, self.sigma, h, 0.98)
        assert ev  == pytest.approx(exp_ev,  rel=REL), f"{label} EV mismatch"
        assert sd  == pytest.approx(exp_sd,  rel=REL), f"{label} SD mismatch"
        assert var == pytest.approx(exp_var98, rel=REL), f"{label} VaR98 mismatch"

    def test_var_grows_with_horizon(self):
        v1 = var_long_lognormal(self.V0, self.mu, self.sigma, 1/252, 0.98)
        v5 = var_long_lognormal(self.V0, self.mu, self.sigma, 5/252, 0.98)
        vy = var_long_lognormal(self.V0, self.mu, self.sigma, 1.0,   0.98)
        assert v1 < v5 < vy


class TestHW4_SingleStockParamVaR:
    """
    HW4_SINGLE_STOCK_PARAM_VAR - calibration and 5-day 99% VaR.
    Source: Homework 4, Problem 1.
    1,400 shares at $82; daily m=0.00015, daily sigma=0.035; horizon 5 days.
    API converts: mu_arithmetic = m + 0.5*sigma^2.
    """
    m_daily   = 0.00015
    sig_daily = 0.035
    S0, shares, h = 82.0, 1400, 5.0  # h in days (matching sigma_daily units)

    @property
    def V0(self):
        return self.S0 * self.shares

    @property
    def mu_daily(self):
        return self.m_daily + 0.5 * self.sig_daily ** 2

    def test_V0(self):
        assert self.V0 == pytest.approx(114_800.0, abs=0.01)

    def test_sigma_annual(self):
        sig_ann = self.sig_daily * math.sqrt(252)
        assert sig_ann == pytest.approx(0.55561, rel=REL)

    def test_mu_annual(self):
        sig_ann = self.sig_daily * math.sqrt(252)
        mu_ann = self.m_daily * 252 + 0.5 * sig_ann ** 2
        assert mu_ann == pytest.approx(0.19215, rel=REL)

    def test_exponent(self):
        # exponent = m*h + sigma*sqrt(h)*z_{0.01}   (m = mean log return)
        m_h = self.m_daily * self.h
        s_h = self.sig_daily * math.sqrt(self.h)
        z = norm.ppf(0.01)
        exponent = m_h + s_h * z
        assert exponent == pytest.approx(-0.18131552, rel=REL)

    def test_var_99_5day(self):
        var = var_long_lognormal(self.V0, self.mu_daily, self.sig_daily, self.h, 0.99)
        assert var == pytest.approx(19037.0, abs=ABS_MONEY)


class TestHW9_DiscreteCVA:
    """
    HW9_DISCRETE_CVA - one-period risk-neutral CVA.
    Source: Homework 9.  Short-forward setup (deliver stock, receive bond).
    Counterparty defaults with risk-neutral prob 0.25.
    """
    S0, ST_up, ST_down, BT = 100.0, 165.0, 45.0, 100.0
    R, pd_Q = 0.3, 0.25

    @property
    def p_up(self):
        return (self.S0 - self.ST_down) / (self.ST_up - self.ST_down)

    @property
    def p_down(self):
        return 1.0 - self.p_up

    @property
    def exposure_down(self):
        # Short forward: we receive stock price, pay bond; exposure = max(BT - ST_down, 0)
        return max(self.BT - self.ST_down, 0.0)

    def test_risk_neutral_up_prob(self):
        assert self.p_up == pytest.approx(0.4583333333, rel=REL)

    def test_risk_neutral_down_prob(self):
        assert self.p_down == pytest.approx(0.5416666667, rel=REL)

    def test_risk_free_contract_value(self):
        # Fair forward: value should be zero under Q
        v = self.p_up * (self.ST_up - self.BT) + self.p_down * (self.ST_down - self.BT)
        assert abs(v) == pytest.approx(0.0, abs=1e-10)

    def test_positive_exposure_down(self):
        assert self.exposure_down == pytest.approx(55.0, abs=1e-10)

    def test_lgd_down_state(self):
        lgd = (1.0 - self.R) * self.exposure_down
        assert lgd == pytest.approx(38.5, abs=1e-10)

    def test_cva_via_discrete(self):
        # EPE = p_down * exposure_down  (p_up contribution is 0)
        epe = self.p_down * self.exposure_down
        cva = cva_discrete([epe], [self.pd_Q], R=self.R)
        # Source rounds to 5.21; both 5.21 and 5.208 are within ABS=0.1
        assert cva == pytest.approx(5.21, abs=0.1)

    def test_cva_positive(self):
        epe = self.p_down * self.exposure_down
        cva = cva_discrete([epe], [self.pd_Q], R=self.R)
        assert cva > 0

    def test_risky_contract_value_negative(self):
        # Risky value = risk-free value - CVA (negative because CVA > 0)
        epe = self.p_down * self.exposure_down
        cva = cva_discrete([epe], [self.pd_Q], R=self.R)
        risky = 0.0 - cva
        assert risky < 0


class TestHW10_RWA_Capital:
    """
    HW10_RWA_CAPITAL - bank balance sheet: cash + residential mortgages + corporate bonds.
    Source: Homework 10. Cash=69k (w=0), mortgages=73k (w=0.45), corp=47k (w=1.0).
    Deposits=182k (liabilities). Capital = assets - liabilities.
    """
    cash, mortgages, corp = 69_000.0, 73_000.0, 47_000.0
    deposits = 182_000.0
    weights = {"cash": 0.0, "mortgages": 0.45, "corp": 1.0}

    @property
    def total_assets(self):
        return self.cash + self.mortgages + self.corp

    @property
    def capital(self):
        return self.total_assets - self.deposits

    def test_total_assets(self):
        assert self.total_assets == pytest.approx(189_000.0, abs=0.01)

    def test_capital(self):
        assert self.capital == pytest.approx(7_000.0, abs=0.01)

    def test_rwa(self):
        rwa = risk_weighted_assets(
            [self.cash, self.mortgages, self.corp],
            [self.weights["cash"], self.weights["mortgages"], self.weights["corp"]],
        )
        assert rwa == pytest.approx(79_850.0, rel=REL)

    def test_capital_ratio(self):
        rwa = risk_weighted_assets(
            [self.cash, self.mortgages, self.corp],
            [self.weights["cash"], self.weights["mortgages"], self.weights["corp"]],
        )
        out = capital_ratio(self.capital, rwa)
        assert out["ratio"] == pytest.approx(0.0876643707, rel=REL)
        assert out["ratio"] * 100 == pytest.approx(8.77, abs=0.02)
        assert out["pass"] is True   # 8.77% > 8%

    def test_leverage_ratio(self):
        lev = self.capital / self.total_assets
        assert lev == pytest.approx(0.037037037, rel=REL)
        assert lev * 100 == pytest.approx(3.7, abs=0.05)


class TestLN_Monotonicity:
    """
    LN_MONOTONICITY_SMOKE - VaR increases with confidence level and volatility.
    Source: seminal edge-case from assessment doc.
    m=0 (log-return mean); mu = m + 0.5*sigma^2 (arithmetic drift for API).
    """
    V0, m, h = 10_000.0, 0.0, 1.0

    def _var(self, sigma, alpha):
        mu = self.m + 0.5 * sigma ** 2
        return var_long_lognormal(self.V0, mu, sigma, self.h, alpha)

    def test_var_increases_with_confidence(self):
        for sigma in [0.1, 0.2]:
            v95 = self._var(sigma, 0.95)
            v99 = self._var(sigma, 0.99)
            assert v99 > v95, f"sigma={sigma}: VaR99 should exceed VaR95"

    def test_var_increases_with_volatility(self):
        for alpha in [0.95, 0.99]:
            v_low  = self._var(0.1, alpha)
            v_high = self._var(0.2, alpha)
            assert v_high > v_low, f"alpha={alpha}: higher sigma should give higher VaR"

    def test_known_var99_sigma20(self):
        # Matches LN01 exactly (m=0, sigma=0.2, h=1, V0=10000, p=0.99)
        v = self._var(0.2, 0.99)
        assert v == pytest.approx(3720.342013894248, rel=REL)

    @pytest.mark.parametrize("sigma,alpha,approx_val", [
        (0.1, 0.95, 1516.4),
        (0.1, 0.99, 2075.1),
        (0.2, 0.95, 2802.4),
        (0.2, 0.99, 3720.3),
    ])
    def test_specific_values(self, sigma, alpha, approx_val):
        v = self._var(sigma, alpha)
        assert v == pytest.approx(approx_val, abs=5.0)


class TestHZ_EdgeZeroHazard:
    """
    HZ_EDGE_ZERO_HAZARD - zero hazard: certain survival, zero default, zero credit spread.
    Source: assessment doc edge cases.
    """
    lam = 0.0
    T, LGD = 5.0, 0.6

    def test_survival_is_one(self):
        assert survival(self.T, self.lam) == pytest.approx(1.0, abs=1e-12)

    def test_cumulative_pd_is_zero(self):
        assert cumulative_default_prob(self.T, self.lam) == pytest.approx(0.0, abs=1e-12)

    def test_density_is_zero(self):
        assert default_density(self.T, self.lam) == pytest.approx(0.0, abs=1e-12)

    def test_risky_zcb_equals_riskfree(self):
        # When s(T)=1, LGD*(1-s(T))=0, so risky ZCB = e^{-rT} (risk-free)
        r = 0.05
        s_T = survival(self.T, self.lam)
        zcb = risky_zcb_price(r=r, T=self.T, LGD=self.LGD, s_T=s_T)
        risk_free = math.exp(-r * self.T)
        assert zcb == pytest.approx(risk_free, rel=1e-10)

    def test_credit_spread_is_zero(self):
        s_T = survival(self.T, self.lam)
        spr = credit_spread(T=self.T, LGD=self.LGD, s_T=s_T)
        assert spr == pytest.approx(0.0, abs=1e-12)


class TestHZ_NegativeHazardReject:
    """
    HZ_INPUT_NEGATIVE_HAZARD_REJECT - negative lambda must raise ValueError.
    Source: assessment doc. Negative intensities imply invalid survival.
    """
    def test_survival_rejects_negative_lambda(self):
        with pytest.raises(ValueError):
            survival(5.0, -0.01)

    def test_default_density_rejects_negative_lambda(self):
        with pytest.raises(ValueError):
            default_density(5.0, -0.01)

    def test_interval_default_prob_rejects_negative_lambda(self):
        with pytest.raises(ValueError):
            interval_default_prob(1.0, 2.0, -0.01)


class TestMR_AtTheMoney:
    """
    MR_AT_THE_MONEY_STRUCTURAL_SMOKE - Merton model when V0 = B (at-the-money debt).
    Source: assessment doc.  V0=B=100, r=0.05, sigma=0.2, T=1.
    d2 = (r - 0.5*sigma^2)*T / (sigma*sqrt(T)) = 0.15
    d1 = d2 + sigma*sqrt(T)    = 0.35
    """
    V0, B, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    def test_d2(self):
        _, d2 = merton_d1_d2(self.V0, self.B, self.r, self.sigma, self.T)
        assert d2 == pytest.approx(0.15, abs=1e-8)

    def test_d1(self):
        d1, _ = merton_d1_d2(self.V0, self.B, self.r, self.sigma, self.T)
        assert d1 == pytest.approx(0.35, abs=1e-8)

    def test_survival_Q(self):
        pd = merton_pd(self.V0, self.B, self.r, self.sigma, self.T)
        assert (1.0 - pd) == pytest.approx(0.5596176924, rel=REL)

    def test_default_Q(self):
        pd = merton_pd(self.V0, self.B, self.r, self.sigma, self.T)
        assert pd == pytest.approx(0.4403823076, rel=REL)

    def test_equity(self):
        E = merton_equity(self.V0, self.B, self.r, self.sigma, self.T)
        assert E == pytest.approx(10.4505835722, rel=REL)

    def test_debt(self):
        D = merton_debt(self.V0, self.B, self.r, self.sigma, self.T)
        assert D == pytest.approx(89.5494164278, rel=REL)

    def test_balance_sheet_identity(self):
        E = merton_equity(self.V0, self.B, self.r, self.sigma, self.T)
        D = merton_debt(self.V0, self.B, self.r, self.sigma, self.T)
        assert E + D == pytest.approx(self.V0, rel=1e-10)


class TestMR_InvalidSurvivalReject:
    """
    MR_TARGET_SURVIVAL_INVALID_REJECT - target_survival outside (0,1) raises ValueError.
    Source: assessment doc.
    """
    def test_survival_above_one_raises(self):
        with pytest.raises(ValueError):
            merton_implied_B(15_000_000.0, target_survival=1.2, r=0.05, sigma=0.3, T=5.0)

    def test_survival_zero_raises(self):
        with pytest.raises(ValueError):
            merton_implied_B(15_000_000.0, target_survival=0.0, r=0.05, sigma=0.3, T=5.0)

    def test_survival_one_raises(self):
        with pytest.raises(ValueError):
            merton_implied_B(15_000_000.0, target_survival=1.0, r=0.05, sigma=0.3, T=5.0)


class TestCVA_FullRecovery:
    """
    CVA_EDGE_FULL_RECOVERY - R=1 makes CVA exactly zero even with exposure and default risk.
    Source: assessment doc.  Tests LGD multiplier discipline.
    """
    def test_cva_zero_when_R_one(self):
        lam, T, exposure = 0.03, 5.0, 12.0
        pds = [interval_default_prob(i - 1, i, lam) for i in range(1, 6)]
        cva = cva_discrete([exposure] * 5, pds, R=1.0)
        assert cva == pytest.approx(0.0, abs=1e-12)

    def test_cva_approx_zero_when_R_one(self):
        spread = cds_par_spread_constant_hazard(lam=0.03, R=1.0)
        assert spread == pytest.approx(0.0, abs=1e-12)


class TestCVA_LinearityExposure:
    """
    CVA_LINEARITY_EXPOSURE - CVA scales linearly with deterministic exposure.
    Source: assessment doc.  Catches missing notional / exposure scaling.
    """
    R, lam = 0.4, 0.03
    pds = [interval_default_prob(i - 1, i, 0.03) for i in range(1, 6)]

    def test_doubling_exposure_doubles_cva(self):
        cva12 = cva_discrete([12.0] * 5, self.pds, R=self.R)
        cva24 = cva_discrete([24.0] * 5, self.pds, R=self.R)
        assert cva12 == pytest.approx(1.0029025697, rel=REL)
        assert cva24 == pytest.approx(2.0058051395, rel=REL)
        assert cva24 / cva12 == pytest.approx(2.0, rel=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# TESTABLE_IF_HELPER_EXISTS  (helpers are in this repo)
# ──────────────────────────────────────────────────────────────────────────────

class TestHW3_ScenarioVaR_ES:
    """
    HW3_SCENARIO_VAR_ES - 10-scenario empirical VaR and ES.
    Source: Homework 3. Apple + IBM portfolio.
    """
    initial = {"Apple": 228.15, "IBM": 205.23}
    shares  = {"Apple": 100,    "IBM": 120}
    scenarios = [
        (1,  170.42, 255.80),
        (2,  191.79, 220.14),
        (3,  210.15, 227.69),
        (4,  221.77, 206.39),
        (5,  228.97, 220.69),
        (6,  233.33, 201.93),
        (7,  225.94, 191.95),
        (8,  237.05, 173.43),
        (9,  250.29, 166.49),
        (10, 235.65, 166.22),
    ]

    @property
    def V0(self):
        return (self.initial["Apple"] * self.shares["Apple"]
                + self.initial["IBM"] * self.shares["IBM"])

    def _losses(self):
        losses = {}
        for sid, p_apple, p_ibm in self.scenarios:
            V_scenario = (p_apple * self.shares["Apple"]
                          + p_ibm  * self.shares["IBM"])
            losses[sid] = self.V0 - V_scenario
        return losses

    def test_initial_value(self):
        assert self.V0 == pytest.approx(47_442.6, abs=0.01)

    def test_loss_scenario_10_is_largest(self):
        losses = self._losses()
        assert losses[10] == pytest.approx(3931.2, abs=0.1)

    def test_loss_scenario_5_is_gain(self):
        # Scenario 5: loss is negative (portfolio gained)
        assert self._losses()[5] == pytest.approx(-1937.2, abs=0.1)

    def test_var_90(self):
        losses = self._losses()
        sorted_losses = sorted(losses.values(), reverse=True)
        # 90% VaR = worst 10% = 1 out of 10 → the top-ranked loss
        var_90 = sorted_losses[0]
        assert var_90 == pytest.approx(3931.2, abs=0.1)

    def test_es_80(self):
        losses = self._losses()
        sorted_losses = sorted(losses.values(), reverse=True)
        # 80% ES = average of top 20% = 2 scenarios out of 10
        es_80 = (sorted_losses[0] + sorted_losses[1]) / 2
        assert es_80 == pytest.approx(3428.6, abs=0.1)


class TestHW4_TwoStockNormalVaR:
    """
    HW4_TWO_STOCK_NORMAL_VAR - two-stock sum-of-lognormal moments VaR.
    Source: Homework 4.
    Stock 1: 400 shares × $102, mu=0.035, sigma=0.33
    Stock 2: 600 shares × $81,  mu=0.023, sigma=0.22
    rho=0.31, T=10/252 (10 trading days), alpha=0.99
    """
    n1, S1, mu1, sig1 = 400, 102.0, 0.035, 0.33
    n2, S2, mu2, sig2 = 600,  81.0, 0.023, 0.22
    rho = 0.31
    T = 10.0 / 252
    alpha = 0.99

    @property
    def V0(self):
        return self.n1 * self.S1 + self.n2 * self.S2

    def _moments(self):
        """Compute EV and EV2 from sum-of-lognormals."""
        T = self.T
        # E[V_T]
        EV1 = self.n1 * self.S1 * math.exp(self.mu1 * T)
        EV2_ = self.n2 * self.S2 * math.exp(self.mu2 * T)
        EV = EV1 + EV2_

        # E[V_T^2] = n1^2*E[S1^2] + 2*n1*n2*E[S1*S2] + n2^2*E[S2^2]
        ES1sq  = self.S1**2 * math.exp(2*self.mu1*T + self.sig1**2*T)
        ES2sq  = self.S2**2 * math.exp(2*self.mu2*T + self.sig2**2*T)
        ES1S2  = (self.S1 * self.S2
                  * math.exp((self.mu1 + self.mu2)*T)
                  * math.exp(self.rho * self.sig1 * self.sig2 * T))

        EV2 = (self.n1**2 * ES1sq
               + 2 * self.n1 * self.n2 * ES1S2
               + self.n2**2 * ES2sq)
        return EV, EV2

    def test_initial_value(self):
        assert self.V0 == pytest.approx(89_400.0, abs=0.01)

    def test_expected_value(self):
        EV, _ = self._moments()
        assert EV == pytest.approx(89_501.08, abs=ABS_MONEY)

    def test_second_moment(self):
        _, EV2 = self._moments()
        assert EV2 == pytest.approx(8_025_773_843.49, rel=REL)

    def test_variance_and_stdev(self):
        EV, EV2 = self._moments()
        var = EV2 - EV**2
        stdev = math.sqrt(var)
        assert var   == pytest.approx(15_329_908.69, rel=REL)
        assert stdev == pytest.approx(3_915.34, rel=REL)

    def test_var_99(self):
        EV, EV2 = self._moments()
        stdev = math.sqrt(EV2 - EV**2)
        # Normal approx VaR: -(drift) + |z_{1-alpha}| * stdev
        z = norm.ppf(self.alpha)      # = +2.3263 at 99%
        var = z * stdev - (EV - self.V0)
        assert var == pytest.approx(9_007.37, abs=ABS_MONEY)


class TestHW5_Lambda20PctHeuristic:
    """
    HW5_LAMBDA_20PCT_HEURISTIC - EWMA lambda such that lambda^N = 0.2.
    Source: Homework 5.  "The last observation gets 20% of total weight."
    """
    @pytest.mark.parametrize("years,expected_lam", [
        (2,  0.9968117641),
        (5,  0.9987234838),
        (10, 0.9993615381),
    ])
    def test_lambda(self, years, expected_lam):
        N = years * 252
        lam = 0.2 ** (1.0 / N)
        assert lam == pytest.approx(expected_lam, rel=1e-8)

    @pytest.mark.parametrize("years", [2, 5, 10])
    def test_check_property(self, years):
        N = years * 252
        lam = 0.2 ** (1.0 / N)
        assert lam ** N == pytest.approx(0.2, rel=1e-8)


class TestHW5_LambdaL2Equivalent:
    """
    HW5_LAMBDA_L2_EQUIVALENT - EWMA lambda from half-life = N/2 (L2-optimal).
    Source: Homework 5.  lambda = 0.5^(1/half_life), half_life = N/2.
    """
    @pytest.mark.parametrize("years,expected_lam,expected_hl", [
        (2,   0.9972531953, 252),
        (5,   0.9989003714, 630),
        (10,  0.9994500345, 1260),
    ])
    def test_lambda_and_halflife(self, years, expected_lam, expected_hl):
        N = years * 252
        half_life = N // 2
        lam = 0.5 ** (1.0 / half_life)
        assert half_life == expected_hl
        assert lam == pytest.approx(expected_lam, rel=1e-8)

    @pytest.mark.parametrize("years", [2, 5, 10])
    def test_halflife_property(self, years):
        N = years * 252
        half_life = N // 2
        lam = 0.5 ** (1.0 / half_life)
        assert lam ** half_life == pytest.approx(0.5, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL_EXTENSION_TESTABLE
# ──────────────────────────────────────────────────────────────────────────────

class TestHW5_BS_DeltaFiniteDiff:
    """
    HW5_BS_DELTA_FINITE_DIFF - ATM Black-Scholes call price and delta.
    Source: Homework 5.  S=K=85, r=0.045, sigma=0.3, T=2 (no dividend).
    d1 = (0.045 + 0.5*0.09)*2 / (0.3*sqrt(2)) = 0.42426...
    d2 = d1 - 0.3*sqrt(2) = 0.0
    """
    S, K, r, sigma, T = 85.0, 85.0, 0.045, 0.3, 2.0

    def test_d1_d2(self):
        s_root_T = self.sigma * math.sqrt(self.T)
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / s_root_T
        d2 = d1 - s_root_T
        assert d1 == pytest.approx(0.4242640687, rel=REL)
        assert d2 == pytest.approx(0.0,          abs=1e-8)

    def test_call_price(self):
        price = bs_price(self.S, self.K, self.T, self.r, 0.0, self.sigma, "call")
        assert price == pytest.approx(17.624561902985718, rel=REL)

    def test_delta(self):
        delta = bs_delta(self.S, self.K, self.T, self.r, 0.0, self.sigma, "call")
        assert delta == pytest.approx(0.6643133797295637, rel=REL)

    def test_delta_exceeds_half_for_atm(self):
        # For ATM call with positive r and T>0, d1>0 so delta>0.5
        delta = bs_delta(self.S, self.K, self.T, self.r, 0.0, self.sigma, "call")
        assert delta > 0.5


class TestHW3_IntelBSM_DeltaHedge:
    """
    HW3_INTEL_BSM_DELTA_HEDGE - Intel BSM call price, delta hedge, shocked P&L.
    Source: Homework 3.
    S0=24.65, K=25, r=0.047, sigma=0.40, T=1.5
    1,200 shares of stock. Hedge by writing calls.
    """
    S0, K, r, sigma, T = 24.65, 25.0, 0.047, 0.40, 1.5
    shares = 1200

    @property
    def call_price(self):
        return bs_price(self.S0, self.K, self.T, self.r, 0.0, self.sigma, "call")

    @property
    def call_delta(self):
        return bs_delta(self.S0, self.K, self.T, self.r, 0.0, self.sigma, "call")

    def test_stock_value(self):
        assert self.shares * self.S0 == pytest.approx(29_580.0, abs=0.01)

    def test_call_price(self):
        assert self.call_price == pytest.approx(5.3450794945, rel=REL)

    def test_call_delta(self):
        assert self.call_delta == pytest.approx(0.6406052942, rel=REL)

    def test_hedge_calls_to_write(self):
        # Write calls so that stock delta = calls written × call_delta
        # N_calls = shares / delta  (so delta exposure nets to zero)
        N_calls = int(round(self.shares / self.call_delta))
        assert N_calls == pytest.approx(1873, abs=1)

    def test_shocked_call_price_up(self):
        S_up = self.S0 * (1 + 0.15)
        price_up = bs_price(S_up, self.K, self.T, self.r, 0.0, self.sigma, "call")
        assert price_up == pytest.approx(7.9074017684, rel=REL)

    def test_shocked_call_price_down(self):
        S_down = self.S0 * (1 - 0.15)
        price_down = bs_price(S_down, self.K, self.T, self.r, 0.0, self.sigma, "call")
        assert price_down == pytest.approx(3.2064489719, rel=REL)

    def test_hedged_portfolio_down_loss_less_than_unhedged(self):
        # Unhedged: just the stock position loses 15%
        unhedged_loss = self.shares * self.S0 * 0.15
        # Down scenario loss of hedged position
        S_down = self.S0 * (1 - 0.15)
        N_calls = int(round(self.shares / self.call_delta))
        cash_received = N_calls * self.call_price
        V_stock_down = self.shares * S_down
        price_down = bs_price(S_down, self.K, self.T, self.r, 0.0, self.sigma, "call")
        V_calls_down = -N_calls * price_down   # short calls
        V_hedged = V_stock_down + V_calls_down + cash_received
        hedged_loss = self.shares * self.S0 - V_hedged
        assert hedged_loss < unhedged_loss
