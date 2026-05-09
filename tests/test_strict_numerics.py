"""
test_strict_numerics.py
Deterministic closed-form functions must match to machine precision (rel=1e-10).
"""
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
from scipy.stats import norm

from src.risk.lognormal import var_long_lognormal, es_long_lognormal, var_short_lognormal, es_short_lognormal
from src.credit.hazard import survival, default_density, interval_default_prob, cumulative_default_prob
from src.credit.merton import merton_d1_d2, merton_pd
from src.credit.cds import cds_par_spread_constant_hazard
from src.credit.cva import cva_discrete, cva_discounted, cva_continuous_constant_exposure
from src.risk.regulatory import risk_weighted_assets, capital_ratio
from src.risk.normal import normal_var, normal_es

STRICT = dict(rel=1e-10)

class TestStrictLognormal:
    """LN01 canonical: m=0, sigma=0.2, h=1, V0=10000, mu=0.02."""
    V0, mu, sigma, h = 10_000.0, 0.02, 0.2, 1.0

    def test_long_var_99(self):
        # m_h = (0.02 - 0.5*0.04)*1 = 0; z_{0.01} = -2.326348...
        # VaR = 10000 * (1 - exp(0 + 0.2 * (-2.326348))) = 10000*(1-exp(-0.465270))
        expected = 10_000.0 * (1 - math.exp(0.2 * norm.ppf(0.01)))
        assert var_long_lognormal(self.V0, self.mu, self.sigma, self.h, 0.99) == pytest.approx(expected, **STRICT)

    def test_long_es_975(self):
        m_h, s_h = 0.0, 0.2
        alpha = 0.025
        z = norm.ppf(0.025)
        expected = self.V0 * (1 - math.exp(m_h + 0.5*s_h**2) * norm.cdf(z - s_h) / alpha)
        assert es_long_lognormal(self.V0, self.mu, self.sigma, self.h, 0.975) == pytest.approx(expected, **STRICT)

    def test_short_var_99(self):
        m_h, s_h = 0.0, 0.2
        expected = self.V0 * (math.exp(m_h + s_h * norm.ppf(0.99)) - 1)
        assert var_short_lognormal(self.V0, self.mu, self.sigma, self.h, 0.99) == pytest.approx(expected, **STRICT)

class TestStrictHazard:
    def test_survival_exact(self):
        assert survival(1.0, 0.03) == pytest.approx(math.exp(-0.03), **STRICT)

    def test_default_density_exact(self):
        assert default_density(1.0, 0.03) == pytest.approx(0.03 * math.exp(-0.03), **STRICT)

    def test_interval_default_prob_exact(self):
        assert interval_default_prob(1.0, 2.0, 0.03) == pytest.approx(
            math.exp(-0.03) - math.exp(-0.06), **STRICT)

    def test_cumulative_default_prob_exact(self):
        assert cumulative_default_prob(5.0, 0.03) == pytest.approx(1 - math.exp(-0.15), **STRICT)

class TestStrictMerton:
    def test_d1_d2_exact(self):
        V0, B, r, sigma, T = 100.0, 80.0, 0.05, 0.25, 1.0
        d1_expected = (math.log(V0/B) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2_expected = d1_expected - sigma*math.sqrt(T)
        d1, d2 = merton_d1_d2(V0, B, r, sigma, T)
        assert d1 == pytest.approx(d1_expected, **STRICT)
        assert d2 == pytest.approx(d2_expected, **STRICT)

    def test_pd_exact(self):
        V0, B, r, sigma, T = 100.0, 80.0, 0.05, 0.25, 1.0
        _, d2 = merton_d1_d2(V0, B, r, sigma, T)
        assert merton_pd(V0, B, r, sigma, T) == pytest.approx(norm.cdf(-d2), **STRICT)

class TestStrictCDS:
    def test_constant_hazard_approx(self):
        assert cds_par_spread_constant_hazard(0.03, 0.40) == pytest.approx(0.018, **STRICT)

class TestStrictCVA:
    def test_cva_discrete_exact(self):
        # (1-0.4) * (10*0.01 + 12*0.015 + 14*0.02) = 0.6*(0.1+0.18+0.28) = 0.6*0.56 = 0.336
        assert cva_discrete([10, 12, 14], [0.01, 0.015, 0.02], 0.40) == pytest.approx(0.336, **STRICT)

    def test_cva_discounted_exact(self):
        expected = 0.60 * (10.0*0.01*0.99 + 12.0*0.015*0.97 + 14.0*0.02*0.95)
        assert cva_discounted([10,12,14],[0.01,0.015,0.02],[0.99,0.97,0.95],0.40) == pytest.approx(expected, **STRICT)

    def test_cva_continuous_undiscounted(self):
        K, lam, T, R = 12.0, 0.03, 5.0, 0.40
        expected = 0.60 * 12.0 * (1 - math.exp(-0.15))
        assert cva_continuous_constant_exposure(K, lam, T, R, r=0.0) == pytest.approx(expected, **STRICT)

    def test_cva_continuous_discounted(self):
        K, lam, T, R, r = 12.0, 0.03, 5.0, 0.40, 0.05
        expected = 0.60 * 12.0 * 0.03/0.08 * (1 - math.exp(-0.08*5.0))
        assert cva_continuous_constant_exposure(K, lam, T, R, r=r) == pytest.approx(expected, **STRICT)

class TestStrictRWA:
    def test_rwa_exact(self):
        # 100*0.0 + 200*0.5 + 300*1.0 = 0 + 100 + 300 = 400
        assert risk_weighted_assets([100, 200, 300], [0.0, 0.5, 1.0]) == pytest.approx(400.0, **STRICT)

    def test_capital_ratio_exact(self):
        result = capital_ratio(20.0, 250.0)
        assert result["ratio"] == pytest.approx(0.08, **STRICT)

class TestStrictNormal:
    def test_normal_var_exact(self):
        m, s, p = 100.0, 500.0, 0.99
        expected = -m + s * norm.ppf(p)
        assert normal_var(m, s, p) == pytest.approx(expected, **STRICT)

    def test_normal_es_exact(self):
        m, s, p = 100.0, 500.0, 0.99
        z = norm.ppf(p)
        expected = -m + s * norm.pdf(z) / (1 - p)
        assert normal_es(m, s, p) == pytest.approx(expected, **STRICT)
