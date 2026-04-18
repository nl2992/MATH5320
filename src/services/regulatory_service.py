"""
regulatory_service.py
Orchestration for regulatory-capital + stress testing (formula-sheet §12).
"""
from __future__ import annotations

from datetime import date
from typing import Mapping

import pandas as pd

from src.portfolio.portfolio import portfolio_value, reprice_portfolio
from src.risk.regulatory import (
    DFAST_SCENARIOS,
    apply_stress_scenario,
    build_equity_shock_map,
    capital_ratio,
    risk_weighted_assets,
)
from src.schemas import Portfolio


# ── RWA + capital ratio ───────────────────────────────────────────────────────

def compute_rwa_and_ratio(
    portfolio: Portfolio,
    prices: pd.Series,
    risk_weights: Mapping[str, float],
    equity: float,
    pricing_date: date,
) -> dict:
    """
    Compute RWA and capital ratio for the given portfolio.

    Per-ticker dollar exposure is computed via :func:`portfolio_exposure`
    (delta-dollar for equities + BS-delta for options, grouped by underlying).
    The user supplies ``risk_weights`` per ticker; missing tickers default to 1.0.

    Returns
    -------
    dict
        ``{"exposures": {ticker: dollars}, "weights": {ticker: w},
           "rwa": float, "V": float, "equity": equity, "ratio": float,
           "pass": bool, "floor": 0.08}``
    """
    from src.portfolio.portfolio import portfolio_exposure

    exposure = portfolio_exposure(portfolio, prices, pricing_date)
    exposures: dict[str, float] = {t: float(v) for t, v in exposure.items()}

    weights: dict[str, float] = {}
    for t in exposures:
        weights[t] = float(risk_weights.get(t, 1.0))

    rwa = risk_weighted_assets(
        asset_values=[abs(exposures[t]) for t in exposures],
        risk_weights=[weights[t] for t in exposures],
    )
    V = portfolio_value(portfolio, prices, pricing_date)

    cap = capital_ratio(equity=equity, rwa=rwa) if rwa > 0 else {
        "ratio": float("inf"),
        "pass": True,
        "floor": 0.08,
    }

    return {
        "exposures": exposures,
        "weights": weights,
        "rwa": float(rwa),
        "V": float(V),
        "equity": float(equity),
        "ratio": cap["ratio"],
        "pass": cap["pass"],
        "floor": cap["floor"],
    }


# ── DFAST ─────────────────────────────────────────────────────────────────────

def run_dfast(
    portfolio: Portfolio,
    prices: pd.Series,
    pricing_date: date,
) -> dict:
    """
    Run the three textbook DFAST-style scenarios against the portfolio.

    For each scenario we apply the ``equity`` multiplicative shock uniformly
    to every underlying in the portfolio (rate shocks are recorded but not
    applied — the portfolio priced here contains no explicit rate instruments).

    Returns
    -------
    dict
        ``{scenario_name: {V_pre, V_post, pnl, pnl_pct, equity_shock, rates_bp}}``
    """
    results: dict[str, dict] = {}
    for name, params in DFAST_SCENARIOS.items():
        shock_map = build_equity_shock_map(portfolio, params["equity"])
        res = apply_stress_scenario(portfolio, prices, shock_map, pricing_date)
        res["equity_shock"] = params["equity"]
        res["rates_bp"] = params["rates_bp"]
        results[name] = res
    return results


def run_custom_stress(
    portfolio: Portfolio,
    prices: pd.Series,
    shock_map: Mapping[str, float],
    pricing_date: date,
) -> dict:
    """User-defined per-ticker multiplicative shock scenario."""
    return apply_stress_scenario(portfolio, prices, dict(shock_map), pricing_date)
