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
    """Compute risk-weighted assets (RWA) and the Basel capital ratio.

    Dollar exposures are computed via :func:`portfolio_exposure` (delta-dollar
    for stocks + BS-delta-dollar for options, grouped by underlying ticker).
    Missing tickers in ``risk_weights`` default to 1.0.

    Args:
        portfolio (Portfolio): Stock and option positions.
        prices (pd.Series): Current spot prices indexed by ticker.
        risk_weights (Mapping[str, float]): Per-ticker Basel risk weights;
            missing tickers default to 1.0.
        equity (float): Tier-1 capital or book equity in dollars.
        pricing_date (date): Option pricing date for Black-Scholes delta.

    Returns:
        dict: Regulatory capital result with keys:
            - ``"exposures"`` (dict[str, float]): Dollar exposure per ticker.
            - ``"weights"`` (dict[str, float]): Risk weight per ticker.
            - ``"rwa"`` (float): Total risk-weighted assets.
            - ``"V"`` (float): Current portfolio mark-to-market value.
            - ``"equity"`` (float): Equity capital echoed.
            - ``"ratio"`` (float): equity / rwa.
            - ``"pass"`` (bool): True iff ratio > 0.08.
            - ``"floor"`` (float): The Basel minimum used (0.08).
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
    """Run the three textbook DFAST-style equity stress scenarios.

    Applies a uniform multiplicative equity shock to every underlying in the
    portfolio for each of the three scenarios (baseline / adverse /
    severely_adverse).  Rate shocks from ``DFAST_SCENARIOS`` are recorded in
    the output but not applied - the portfolio contains no explicit rate
    instruments.

    Args:
        portfolio (Portfolio): Stock and option positions to stress-test.
        prices (pd.Series): Current spot prices indexed by ticker.
        pricing_date (date): Option pricing date for Black-Scholes repricing.

    Returns:
        dict: Mapping ``{scenario_name: result_dict}`` where each result
            contains:
            - ``"V_pre"`` (float): Pre-shock portfolio value.
            - ``"V_post"`` (float): Post-shock portfolio value.
            - ``"pnl"`` (float): V_post − V_pre (negative = loss).
            - ``"pnl_pct"`` (float): pnl / V_pre.
            - ``"equity_shock"`` (float): Multiplicative shock applied.
            - ``"rates_bp"`` (float): Rate shock in bps (informational).
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
    """Apply a user-defined per-ticker multiplicative stress scenario.

    Args:
        portfolio (Portfolio): Stock and option positions to be stressed.
        prices (pd.Series): Current spot prices indexed by ticker.
        shock_map (Mapping[str, float]): Per-ticker multiplicative return
            shocks (e.g. ``{"AAPL": -0.20}`` for a 20% drop).  Tickers
            absent from the map are left unshocked.
        pricing_date (date): Option pricing date.

    Returns:
        dict: Stress result with keys ``"V_pre"``, ``"V_post"``, ``"pnl"``,
            ``"pnl_pct"`` - see :func:`apply_stress_scenario`.
    """
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
