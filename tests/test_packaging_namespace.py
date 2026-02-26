"""
Tests for the public PyPI-facing package namespace.
"""
from __future__ import annotations


def test_public_namespace_imports():
    import math5320_portfolio_risk_system as pkg

    assert pkg.__version__ == "0.1.0"
    assert pkg.risk.__name__ == "src.risk"
    assert pkg.pricing.__name__ == "src.pricing"
    assert pkg.schemas.__name__ == "src.schemas"


def test_public_namespace_submodule_import():
    from math5320_portfolio_risk_system.risk import historical
    from math5320_portfolio_risk_system.pricing import black_scholes

    assert historical.__name__ == "src.risk.historical"
    assert hasattr(black_scholes, "bs_price")
