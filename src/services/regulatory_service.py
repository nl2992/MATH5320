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
    DFAST_ILLUSTRATIVE_SCENARIOS,
    DFAST_SCENARIOS,
    CapitalState,
    StressQuarter,
    apply_stress_scenario,
    build_equity_shock_map,
    capital_ratio,
    min_capital_ratio,
    passes_stress,
    project_capital_path,
    risk_weighted_assets,
    run_dfast_scenarios,
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


# ── DFAST capital path projection ─────────────────────────────────────────────

def run_dfast_capital_path(
    tier1_capital: float,
    rwa: float,
    assets: float = 0.0,
    hurdle: float = 0.08,
) -> dict:
    """
    Run all three DFAST illustrative scenarios via the quarter-by-quarter
    capital-path model (CapitalState / StressQuarter).

    This is distinct from ``run_dfast``, which applies multiplicative equity
    price shocks to the current portfolio.  Here we project regulatory capital
    quarter-by-quarter through a 9-quarter horizon.

    Parameters
    ----------
    tier1_capital : float
        Starting Tier-1 capital ($).
    rwa : float
        Starting risk-weighted assets ($).  Must be positive.
    assets : float
        Total assets (used for leverage ratio; can be 0 if unavailable).
    hurdle : float
        Minimum capital ratio threshold (default 0.08 for Basel Tier-1).

    Returns
    -------
    dict
        ``{scenario_name: {"path": [...], "passes": bool, "min_ratio": float}}``
        where each ``path`` entry is
        ``{"quarter": int, "tier1_capital": float, "rwa": float, "capital_ratio": float}``.
    """
    initial = CapitalState(
        tier1_capital=tier1_capital,
        rwa=rwa,
        assets=assets,
    )

    _N_QUARTERS = 9
    result: dict[str, dict] = {}
    for name, params in DFAST_ILLUSTRATIVE_SCENARIOS.items():
        # Build 9 identical quarterly shocks from the scenario params dict.
        # DFAST_ILLUSTRATIVE_SCENARIOS values are per-quarter averages.
        shocks = [
            StressQuarter(
                quarter=q + 1,
                pre_provision_net_revenue=params["pre_provision_net_revenue"],
                credit_loss=params["credit_loss"],
                trading_loss=params["trading_loss"],
                counterparty_loss=params["counterparty_loss"],
                provisions=params["provisions"],
                dividends=0.0,
                buybacks=0.0,
                rwa_change=params["rwa_change"],
            )
            for q in range(_N_QUARTERS)
        ]
        path = project_capital_path(initial, shocks)
        min_rat = min_capital_ratio(path)
        ok = passes_stress(path, hurdle)
        last = path[-1]
        result[name] = {
            "path": path,
            "passes": ok,
            "min_ratio": min_rat,
            "ending_capital": last["tier1_capital"],
            "ending_rwa": last["rwa"],
            "ending_ratio": last["capital_ratio"],
        }

    return result
