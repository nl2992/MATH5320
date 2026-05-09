"""Tests for extended CVA functions in src/credit/cva.py."""
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from src.credit.cva import (
    positive_exposure, epe, cva_discounted,
    cva_continuous_constant_exposure, cva_discrete,
)

class TestPositiveExposure:
    def test_positive_only(self):
        np.testing.assert_array_equal(positive_exposure([1, 2, 3]), [1, 2, 3])

    def test_all_negative(self):
        np.testing.assert_array_equal(positive_exposure([-1, -2, -3]), [0, 0, 0])

    def test_mixed(self):
        np.testing.assert_array_equal(positive_exposure([5, -3, 0, 2]), [5, 0, 0, 2])

class TestEPE:
    def test_all_positive(self):
        assert epe([4.0, 5.0, 6.0]) == pytest.approx(5.0)

    def test_mixed_signs(self):
        # max([10, -5, 0, 5]) = [10,0,0,5], mean = 15/4 = 3.75
        assert epe([10.0, -5.0, 0.0, 5.0]) == pytest.approx(3.75)

    def test_2d(self):
        paths = np.array([[4, -2], [6, 8]])
        result = epe(paths)
        np.testing.assert_allclose(result, [5.0, 4.0])  # mean(max([4,6],0))=5, mean(max([-2,8],0))=mean(0,8)=4

class TestCVADiscounted:
    def test_golden(self):
        expected = 0.60 * (10*0.01*0.99 + 12*0.015*0.97 + 14*0.02*0.95)
        assert cva_discounted([10,12,14],[0.01,0.015,0.02],[0.99,0.97,0.95],0.40) == pytest.approx(expected, rel=1e-10)

    def test_linearity_in_exposure(self):
        e = [10.0, 12.0]; pd_ = [0.01, 0.02]; df = [0.99, 0.97]
        c1 = cva_discounted(e, pd_, df, 0.4)
        c2 = cva_discounted([2*x for x in e], pd_, df, 0.4)
        assert c2 == pytest.approx(2 * c1, rel=1e-10)

    def test_zero_recovery_greater_than_high_recovery(self):
        e = [10.0]; pd_ = [0.05]; df = [0.95]
        assert cva_discounted(e, pd_, df, 0.0) > cva_discounted(e, pd_, df, 0.8)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            cva_discounted([10], [0.01, 0.02], [0.99], 0.4)

class TestCVAContinuousConstantExposure:
    def test_undiscounted(self):
        K, lam, T, R = 12.0, 0.03, 5.0, 0.40
        expected = 0.60 * 12.0 * (1 - math.exp(-0.15))
        assert cva_continuous_constant_exposure(K, lam, T, R, r=0.0) == pytest.approx(expected, rel=1e-10)

    def test_discounted(self):
        K, lam, T, R, r = 12.0, 0.03, 5.0, 0.40, 0.05
        expected = 0.60 * 12.0 * 0.03/0.08 * (1 - math.exp(-0.08*5.0))
        assert cva_continuous_constant_exposure(K, lam, T, R, r=r) == pytest.approx(expected, rel=1e-10)

    def test_zero_hazard_is_zero(self):
        assert cva_continuous_constant_exposure(100.0, 0.0, 5.0, 0.4) == pytest.approx(0.0, abs=1e-15)

    def test_negative_K_raises(self):
        with pytest.raises(ValueError):
            cva_continuous_constant_exposure(-1.0, 0.03, 5.0, 0.4)
