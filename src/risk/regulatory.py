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
