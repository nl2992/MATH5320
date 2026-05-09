"""
regulatory.py
Regulatory capital and stress testing (formula-sheet §12).

Risk-weighted assets:
    RWA = Σ_i w_i · A_i

Capital adequacy:
    ratio = equity / RWA
    PASS iff ratio > 0.08  (Basel minimum Tier-1 guideline used in lecture)

Stress scenario:
    V_post = Σ reprice(portfolio, shocked_spots)
    PnL    = V_post − V_pre
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.portfolio.portfolio import reprice_portfolio
from src.schemas import Portfolio


# ── RWA + capital ratio ───────────────────────────────────────────────────────

def risk_weighted_assets(
    asset_values: Sequence[float],
    risk_weights: Sequence[float],
) -> float:
    """
    RWA = Σ_i w_i · A_i

    Parameters
    ----------
    asset_values : sequence[float]
        Dollar exposures to each asset.
    risk_weights : sequence[float]
        Corresponding risk weights (typically in [0, 1.5]).
    """
    a = np.asarray(asset_values, dtype=float)
    w = np.asarray(risk_weights, dtype=float)
    if len(a) != len(w):
        raise ValueError(
            f"asset_values and risk_weights must align "
            f"(got {len(a)} vs {len(w)})."
        )
    if np.any(w < 0):
        raise ValueError("risk_weights must be non-negative.")
    return float((a * w).sum())


CAPITAL_RATIO_FLOOR = 0.08  # Basel Tier-1-style minimum used in §12.


def capital_ratio(equity: float, rwa: float) -> dict:
    """
    Capital adequacy ratio.

        ratio = equity / RWA
        PASS iff ratio > 0.08

    Returns
    -------
    dict
        ``{"ratio": float, "pass": bool, "floor": 0.08}``
    """
    if rwa <= 0:
        raise ValueError(f"rwa must be positive (got {rwa}).")
    ratio = equity / rwa
    return {
        "ratio": float(ratio),
        "pass": bool(ratio > CAPITAL_RATIO_FLOOR),
        "floor": CAPITAL_RATIO_FLOOR,
    }


# ── Stress scenarios ──────────────────────────────────────────────────────────

def apply_stress_scenario(
    portfolio: Portfolio,
    current_prices: pd.Series,
    shock_map: Mapping[str, float],
    pricing_date: date,
) -> dict:
    """
    Apply a multiplicative price shock to the portfolio and report PnL.

    Parameters
    ----------
    portfolio : Portfolio
    current_prices : pd.Series
        Current spot prices indexed by ticker.
    shock_map : mapping[ticker -> multiplicative shock]
        e.g. ``{"AAPL": -0.30}`` for a 30% drop. Missing tickers are unshocked.
    pricing_date : date

    Returns
    -------
    dict
        ``{"V_pre": float, "V_post": float, "pnl": float, "pnl_pct": float}``
    """
    V_pre = reprice_portfolio(portfolio, current_prices, pricing_date)

    shocked = current_prices.copy()
    for ticker, shock in shock_map.items():
        if ticker in shocked.index:
            shocked[ticker] = shocked[ticker] * (1.0 + shock)

    V_post = reprice_portfolio(portfolio, shocked, pricing_date)
    pnl = V_post - V_pre
    pnl_pct = pnl / V_pre if V_pre != 0 else float("nan")
    return {
        "V_pre": float(V_pre),
        "V_post": float(V_post),
        "pnl": float(pnl),
        "pnl_pct": float(pnl_pct),
    }


# ── DFAST textbook scenarios ──────────────────────────────────────────────────
#
# These are illustrative defaults of the shape used in the §12 lecture examples
# (broad equity shocks with a flight-to-quality rate move). They are NOT the
# official Federal Reserve DFAST numbers — real DFAST scenarios are multi-asset,
# path-dependent, and published annually. Treat these as teaching values.

DFAST_SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": {
        "equity": 0.05,     # +5%
        "rates_bp": 25.0,   # +25 bp
    },
    "adverse": {
        "equity": -0.15,    # -15%
        "rates_bp": -50.0,
    },
    "severely_adverse": {
        "equity": -0.35,    # -35%
        "rates_bp": -150.0,
    },
}


def build_equity_shock_map(
    portfolio: Portfolio,
    equity_shock: float,
) -> dict[str, float]:
    """
    Build a per-ticker multiplicative shock map that applies ``equity_shock``
    to every underlying ticker referenced by the portfolio.
    """
    tickers: set[str] = set()
    for pos in portfolio.stocks:
        tickers.add(pos.ticker)
    for pos in portfolio.options:
        tickers.add(pos.underlying_ticker)
    return {t: equity_shock for t in tickers}


# ── DFAST capital-path simulator ──────────────────────────────────────────────
#
# This is an illustrative 9-quarter capital-path engine, NOT an official
# Federal Reserve DFAST model. Real DFAST uses 28 economic indicators,
# path-dependent losses, and official supervisory scenario definitions.


@dataclass
class CapitalState:
    """Snapshot of bank capital and risk-weighted assets at one point in time."""
    tier1_capital: float
    rwa: float
    assets: float = 0.0

    def __post_init__(self) -> None:
        if self.rwa <= 0:
            raise ValueError(f"rwa must be positive (got {self.rwa}).")
        if self.tier1_capital < 0:
            raise ValueError(f"tier1_capital must be non-negative (got {self.tier1_capital}).")


@dataclass
class StressQuarter:
    """P&L and balance-sheet changes for one stress quarter."""
    quarter: int
    pre_provision_net_revenue: float = 0.0
    credit_loss: float = 0.0
    trading_loss: float = 0.0
    counterparty_loss: float = 0.0
    provisions: float = 0.0
    dividends: float = 0.0
    buybacks: float = 0.0
    rwa_change: float = 0.0


def project_capital_one_quarter(state: CapitalState, shock: StressQuarter) -> CapitalState:
    """
    Advance capital by one stress quarter.

        capital_next = capital
                       + pre_provision_net_revenue
                       − credit_loss
                       − trading_loss
                       − counterparty_loss
                       − provisions
                       − dividends
                       − buybacks

        rwa_next = rwa + rwa_change
    """
    capital_next = (
        state.tier1_capital
        + shock.pre_provision_net_revenue
        - shock.credit_loss
        - shock.trading_loss
        - shock.counterparty_loss
        - shock.provisions
        - shock.dividends
        - shock.buybacks
    )
    rwa_next = state.rwa + shock.rwa_change
    if rwa_next <= 0:
        raise ValueError(f"rwa became non-positive after quarter {shock.quarter}.")
    return CapitalState(tier1_capital=capital_next, rwa=rwa_next)


def project_capital_path(
    initial_state: CapitalState,
    shocks: list[StressQuarter],
) -> list[dict]:
    """
    Project capital over multiple stress quarters.

    Returns a list of dicts with keys:
        quarter, tier1_capital, rwa, capital_ratio
    """
    path: list[dict] = []
    state = initial_state
    for shock in shocks:
        state = project_capital_one_quarter(state, shock)
        ratio = state.tier1_capital / state.rwa
        path.append({
            "quarter": shock.quarter,
            "tier1_capital": state.tier1_capital,
            "rwa": state.rwa,
            "capital_ratio": ratio,
        })
    return path


def min_capital_ratio(path: list[dict]) -> float:
    """Minimum capital ratio across a projected path."""
    if not path:
        raise ValueError("path is empty.")
    return float(min(row["capital_ratio"] for row in path))


def passes_stress(path: list[dict], hurdle: float = 0.08) -> bool:
    """True if the minimum capital ratio across the path meets the hurdle."""
    return bool(min_capital_ratio(path) >= hurdle)


def apply_global_market_shock(state: CapitalState, trading_loss: float) -> CapitalState:
    """Apply a one-time global-market-shock trading loss."""
    return CapitalState(
        tier1_capital=state.tier1_capital - trading_loss,
        rwa=state.rwa,
    )


def apply_counterparty_default_component(
    state: CapitalState, counterparty_loss: float
) -> CapitalState:
    """Apply a one-time counterparty-default loss."""
    return CapitalState(
        tier1_capital=state.tier1_capital - counterparty_loss,
        rwa=state.rwa,
    )


DFAST_ILLUSTRATIVE_SCENARIOS: dict[str, dict] = {
    "baseline": {
        "credit_loss": 1.0,
        "trading_loss": 0.2,
        "counterparty_loss": 0.0,
        "pre_provision_net_revenue": 4.0,
        "provisions": 0.5,
        "rwa_change": 2.0,
    },
    "adverse": {
        "credit_loss": 4.0,
        "trading_loss": 1.0,
        "counterparty_loss": 0.5,
        "pre_provision_net_revenue": 3.0,
        "provisions": 1.0,
        "rwa_change": 6.0,
    },
    "severely_adverse": {
        "credit_loss": 7.0,
        "trading_loss": 2.5,
        "counterparty_loss": 1.5,
        "pre_provision_net_revenue": 2.0,
        "provisions": 1.5,
        "rwa_change": 10.0,
    },
}


def run_dfast_scenarios(
    initial_state: CapitalState,
    scenarios: dict[str, list[StressQuarter]] | None = None,
    hurdle: float = 0.08,
) -> list[dict]:
    """
    Run three DFAST-style scenarios and return summary rows.

    Parameters
    ----------
    initial_state : CapitalState
    scenarios : dict mapping scenario name → list of StressQuarter.
        Defaults to 9 identical quarters built from DFAST_ILLUSTRATIVE_SCENARIOS.
    hurdle : capital-ratio pass threshold.

    Returns
    -------
    list of dicts: scenario, ending_capital, ending_rwa,
                   ending_capital_ratio, min_capital_ratio, passes.
    """
    if scenarios is None:
        scenarios = {}
        for name, params in DFAST_ILLUSTRATIVE_SCENARIOS.items():
            quarters = [
                StressQuarter(
                    quarter=q + 1,
                    credit_loss=params["credit_loss"],
                    trading_loss=params["trading_loss"],
                    counterparty_loss=params["counterparty_loss"],
                    pre_provision_net_revenue=params["pre_provision_net_revenue"],
                    provisions=params["provisions"],
                    rwa_change=params["rwa_change"],
                )
                for q in range(9)
            ]
            scenarios[name] = quarters

    results = []
    for scenario_name, quarter_shocks in scenarios.items():
        path = project_capital_path(initial_state, quarter_shocks)
        last = path[-1]
        results.append({
            "scenario": scenario_name,
            "ending_capital": last["tier1_capital"],
            "ending_rwa": last["rwa"],
            "ending_capital_ratio": last["capital_ratio"],
            "min_capital_ratio": min_capital_ratio(path),
            "passes": passes_stress(path, hurdle),
        })
    return results
