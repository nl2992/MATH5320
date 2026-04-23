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
    """Compute total risk-weighted assets.

    RWA = Σ_i w_i · A_i

    Args:
        asset_values (Sequence[float]): Dollar exposures to each asset.
            Lengths must match ``risk_weights``.
        risk_weights (Sequence[float]): Corresponding Basel-style risk
            weights (typically in [0, 1.5]). Must be non-negative.

    Returns:
        float: Total risk-weighted assets in dollars.

    Raises:
        ValueError: If ``asset_values`` and ``risk_weights`` have different
            lengths, or if any weight is negative.
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
    """Compute the capital adequacy ratio and Basel pass/fail flag.

        ratio = equity / RWA
        PASS iff ratio > 0.08

    Args:
        equity (float): Tier-1 capital in dollars.
        rwa (float): Risk-weighted assets in dollars (must be > 0).

    Returns:
        dict: Capital result with keys:
            - ``"ratio"`` (float): equity / rwa.
            - ``"pass"`` (bool): True iff ratio > 0.08.
            - ``"floor"`` (float): The Basel minimum used (0.08).

    Raises:
        ValueError: If ``rwa <= 0``.
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
    """Apply a multiplicative price shock to the portfolio and report PnL.

    Args:
        portfolio (Portfolio): Stock and option positions to be re-priced.
        current_prices (pd.Series): Current spot prices indexed by ticker
            symbol.
        shock_map (Mapping[str, float]): Per-ticker multiplicative return
            shock, e.g. ``{"AAPL": -0.30}`` for a 30% price drop.
            Tickers absent from the map are left unshocked.
        pricing_date (date): Option pricing date (used by Black-Scholes).

    Returns:
        dict: Stress-scenario summary with keys:
            - ``"V_pre"`` (float): Portfolio value before shock.
            - ``"V_post"`` (float): Portfolio value after shock.
            - ``"pnl"`` (float): V_post − V_pre (negative = loss).
            - ``"pnl_pct"`` (float): pnl / V_pre, or NaN if V_pre == 0.
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
# official Federal Reserve DFAST numbers - real DFAST scenarios are multi-asset,
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
    """Build a per-ticker multiplicative shock map from a single equity shock.

    Applies ``equity_shock`` uniformly to every underlying ticker referenced
    by the portfolio (stocks and option underlyings).

    Args:
        portfolio (Portfolio): Stock and option positions whose underlying
            tickers will be enumerated.
        equity_shock (float): Multiplicative return shock to apply, e.g.
            ``-0.30`` for a 30% price decline.

    Returns:
        dict[str, float]: Mapping ``{ticker: equity_shock}`` covering all
            unique underlying tickers in the portfolio.
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
    """Advance the bank's capital position by one stress quarter.

        capital_next = capital
                       + pre_provision_net_revenue
                       − credit_loss − trading_loss − counterparty_loss
                       − provisions − dividends − buybacks
        rwa_next = rwa + rwa_change

    Args:
        state (CapitalState): Capital snapshot at the start of the quarter.
        shock (StressQuarter): P&L and balance-sheet changes for this quarter.

    Returns:
        CapitalState: Updated capital snapshot after applying all charges.

    Raises:
        ValueError: If the resulting RWA is non-positive.
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
    """Project capital over multiple stress quarters.

    Args:
        initial_state (CapitalState): Starting capital and RWA.
        shocks (list[StressQuarter]): Ordered list of quarterly shocks; each
            element advances the state by one quarter.

    Returns:
        list[dict]: One dict per quarter with keys ``"quarter"``,
            ``"tier1_capital"``, ``"rwa"``, ``"capital_ratio"``.
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
    """Return the minimum capital ratio across a projected capital path.

    Args:
        path (list[dict]): Output of :func:`project_capital_path` - each
            dict must contain a ``"capital_ratio"`` key.

    Returns:
        float: Minimum capital_ratio value across all quarters.

    Raises:
        ValueError: If ``path`` is empty.
    """
    if not path:
        raise ValueError("path is empty.")
    return float(min(row["capital_ratio"] for row in path))


def passes_stress(path: list[dict], hurdle: float = 0.08) -> bool:
    """Return True if the minimum capital ratio across the path meets the hurdle.

    Args:
        path (list[dict]): Output of :func:`project_capital_path`.
        hurdle (float): Minimum acceptable capital ratio threshold
            (default 0.08 for Basel Tier-1).

    Returns:
        bool: True iff ``min_capital_ratio(path) >= hurdle``.
    """
    return bool(min_capital_ratio(path) >= hurdle)


def apply_global_market_shock(state: CapitalState, trading_loss: float) -> CapitalState:
    """Apply a one-time global-market-shock trading loss to capital.

    Args:
        state (CapitalState): Current capital snapshot.
        trading_loss (float): Dollar trading loss to deduct from Tier-1 capital.

    Returns:
        CapitalState: Updated snapshot with capital reduced by ``trading_loss``.
    """
    return CapitalState(
        tier1_capital=state.tier1_capital - trading_loss,
        rwa=state.rwa,
    )


def apply_counterparty_default_component(
    state: CapitalState, counterparty_loss: float
) -> CapitalState:
    """Apply a one-time counterparty-default loss to capital.

    Args:
        state (CapitalState): Current capital snapshot.
        counterparty_loss (float): Dollar loss from counterparty default.

    Returns:
        CapitalState: Updated snapshot with capital reduced by
            ``counterparty_loss``.
    """
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
    """Run DFAST-style stress scenarios and return per-scenario summary rows.

    Args:
        initial_state (CapitalState): Starting Tier-1 capital and RWA.
        scenarios (dict[str, list[StressQuarter]] | None): Mapping of
            scenario name to a list of quarterly shocks.  Defaults to 9
            identical quarters built from ``DFAST_ILLUSTRATIVE_SCENARIOS``
            (baseline / adverse / severely_adverse).
        hurdle (float): Minimum capital ratio pass threshold (default 0.08).

    Returns:
        list[dict]: One dict per scenario with keys:
            - ``"scenario"`` (str): Scenario name.
            - ``"ending_capital"`` (float): Tier-1 capital at quarter 9.
            - ``"ending_rwa"`` (float): RWA at quarter 9.
            - ``"ending_capital_ratio"`` (float): Ratio at quarter 9.
            - ``"min_capital_ratio"`` (float): Worst ratio across all quarters.
            - ``"passes"`` (bool): True iff min ratio ≥ hurdle.
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


# ── Balance-sheet helpers (§12 / HW XII) ─────────────────────────────────────

def balance_sheet_equity(assets: float, liabilities: float) -> float:
    """Compute book equity using the balance-sheet identity.

    Equity = Assets − Liabilities

    Args:
        assets (float): Total book-value assets (must be ≥ 0).
        liabilities (float): Total book-value liabilities (must be ≥ 0).

    Returns:
        float: Book equity in the same currency units; may be negative for
            an insolvent entity.

    Raises:
        ValueError: If ``assets < 0`` or ``liabilities < 0``.
    """
    if assets < 0:
        raise ValueError(f"assets must be non-negative (got {assets}).")
    if liabilities < 0:
        raise ValueError(f"liabilities must be non-negative (got {liabilities}).")
    return assets - liabilities


def balance_sheet_after_asset_loss(
    assets: float, liabilities: float, loss: float
) -> dict:
    """Compute the post-stress balance sheet after an asset write-down.

    Assets absorb the loss; liabilities are unchanged (market losses do not
    affect the liability side).

    Args:
        assets (float): Pre-stress total book-value assets (must be ≥ 0).
        liabilities (float): Total book-value liabilities (must be ≥ 0).
        loss (float): Dollar write-down to deduct from assets (must be ≥ 0).

    Returns:
        dict: Post-stress balance sheet with keys:
            - ``"assets_post"`` (float): assets − loss.
            - ``"liabilities"`` (float): unchanged liabilities.
            - ``"equity_post"`` (float): assets_post − liabilities.
            - ``"solvent"`` (bool): True iff equity_post ≥ 0.

    Raises:
        ValueError: If ``assets < 0``, ``liabilities < 0``, or ``loss < 0``.
    """
    if assets < 0:
        raise ValueError(f"assets must be non-negative (got {assets}).")
    if liabilities < 0:
        raise ValueError(f"liabilities must be non-negative (got {liabilities}).")
    if loss < 0:
        raise ValueError(f"loss must be non-negative (got {loss}).")
    assets_post = assets - loss
    equity_post = assets_post - liabilities
    return {
        "assets_post": assets_post,
        "liabilities": liabilities,
        "equity_post": equity_post,
        "solvent": equity_post >= 0.0,
    }


def leverage_ratio(equity: float, assets: float) -> float:
    """Compute the simple leverage ratio (equity / assets).

    Basel III uses Tier-1 capital over a broader exposure measure; this
    function uses the textbook approximation equity / total assets.

    Args:
        equity (float): Tier-1 capital or book equity in dollars.
        assets (float): Total assets in dollars (must be > 0).

    Returns:
        float: Leverage ratio (dimensionless; e.g. 0.08 = 8%).

    Raises:
        ValueError: If ``assets <= 0``.
    """
    if assets <= 0:
        raise ValueError(f"assets must be positive (got {assets}).")
    return equity / assets
