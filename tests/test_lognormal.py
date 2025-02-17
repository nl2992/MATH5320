"""
test_lognormal.py
Unit tests for src/risk/lognormal.py — §4 (long) + §7 (short) exact GBM VaR/ES.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.risk.lognormal import (
    es_long_lognormal,
    es_short_lognormal,
    var_long_lognormal,
    var_short_lognormal,
)


# ── Landmark values ───────────────────────────────────────────────────────────

class TestLognormalValues:
    def test_var_long_positive(self):
        v = var_long_lognormal(V0=100, mu=0.08, sigma=0.20, h=1 / 252, p=0.99)
        assert v > 0

    def test_var_short_positive(self):
        v = var_short_lognormal(V0=100, mu=0.08, sigma=0.20, h=1 / 252, p=0.99)
        assert v > 0

    def test_es_ge_var_long(self):
        kw = dict(V0=100, mu=0.08, sigma=0.20, h=10 / 252, p=0.99)
        assert es_long_lognormal(**kw) >= var_long_lognormal(**kw)

    def test_es_ge_var_short(self):
        kw = dict(V0=100, mu=0.08, sigma=0.20, h=10 / 252, p=0.99)
        assert es_short_lognormal(**kw) >= var_short_lognormal(**kw)

    def test_short_var_greater_than_long_var(self):
        # Structural: short VaR > long VaR for same drift/vol (§14 landmark).
        kw = dict(V0=100, mu=0.08, sigma=0.25, h=5 / 252, p=0.99)
        assert var_short_lognormal(**kw) > var_long_lognormal(**kw)

    def test_higher_vol_increases_var(self):
        base = dict(V0=100, mu=0.05, h=5 / 252, p=0.99)
        assert var_long_lognormal(sigma=0.40, **base) > var_long_lognormal(sigma=0.15, **base)


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:
    @pytest.mark.parametrize("fn", [var_long_lognormal, es_long_lognormal,
                                    var_short_lognormal, es_short_lognormal])
    def test_bad_V0(self, fn):
        with pytest.raises(ValueError, match="V0 must be positive"):
            fn(V0=0.0, mu=0.05, sigma=0.2, h=0.01, p=0.99)

    @pytest.mark.parametrize("fn", [var_long_lognormal, es_long_lognormal,
                                    var_short_lognormal, es_short_lognormal])
    def test_bad_sigma(self, fn):
        with pytest.raises(ValueError, match="sigma must be positive"):
            fn(V0=100, mu=0.05, sigma=0.0, h=0.01, p=0.99)

    @pytest.mark.parametrize("fn", [var_long_lognormal, es_long_lognormal,
                                    var_short_lognormal, es_short_lognormal])
    def test_bad_h(self, fn):
        with pytest.raises(ValueError, match="h must be positive"):
            fn(V0=100, mu=0.05, sigma=0.2, h=0.0, p=0.99)

    @pytest.mark.parametrize("fn", [var_long_lognormal, es_long_lognormal,
                                    var_short_lognormal, es_short_lognormal])
    @pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.5])
    def test_bad_p(self, fn, bad_p):
        with pytest.raises(ValueError, match="p must be in"):
            fn(V0=100, mu=0.05, sigma=0.2, h=0.01, p=bad_p)
