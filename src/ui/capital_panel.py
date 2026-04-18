"""
capital_panel.py
Streamlit UI for regulatory capital + stress testing (§12).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.schemas import Portfolio
from src.services.regulatory_service import (
    compute_rwa_and_ratio,
    run_custom_stress,
    run_dfast,
)


# Reasonable default Basel-ish weights for equity (1.0) — users can override.
_DEFAULT_EQUITY_WEIGHT = 1.0


def render_capital_panel(
    portfolio: Portfolio,
    prices: pd.DataFrame | None,
) -> None:
    """Render RWA / capital-ratio / DFAST stress sections."""
    st.subheader("Capital & Stress")
    st.caption(
        "Risk-weighted assets, Basel-style capital ratio (PASS iff > 8%), "
        "and DFAST-style stress scenarios."
    )

    if prices is None:
        st.warning("Load market data in the **Market Data** tab first.")
        return
    if len(portfolio.stocks) + len(portfolio.options) == 0:
        st.warning("Add at least one position in the **Portfolio Input** tab.")
        return

    pricing_date = date.today()
    current_prices = prices.iloc[-1]

    # ── Risk weights table (prefilled from portfolio) ─────────────────────────
    st.markdown("### A · Risk weights & capital ratio")
    tickers = sorted(
        {p.ticker for p in portfolio.stocks}
        | {p.underlying_ticker for p in portfolio.options}
    )
    default_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Risk weight": [_DEFAULT_EQUITY_WEIGHT] * len(tickers),
        }
    )
    edited = st.data_editor(
        default_df,
        num_rows="fixed",
        use_container_width=True,
        key="cap_risk_weights",
    )

    # Current portfolio value for default equity.
    try:
        from src.portfolio.portfolio import portfolio_value

        V_now = float(portfolio_value(portfolio, current_prices, pricing_date))
    except Exception as exc:
        st.error(f"Cannot price portfolio: {exc}")
        return

    equity_default = max(0.08 * V_now, 1.0)
    equity = st.number_input(
        "Equity capital ($)",
        min_value=0.0,
        value=float(equity_default),
        step=max(equity_default * 0.1, 1.0),
        format="%.2f",
        key="cap_equity",
        help=f"Current portfolio V = {V_now:,.2f}. Defaults to 8% × V.",
    )

    try:
        weights_map = {
            str(row["Ticker"]): float(row["Risk weight"])
            for _, row in edited.iterrows()
            if pd.notna(row["Ticker"]) and pd.notna(row["Risk weight"])
        }
        summary = compute_rwa_and_ratio(
            portfolio=portfolio,
            prices=current_prices,
            risk_weights=weights_map,
            equity=equity,
            pricing_date=pricing_date,
        )
    except Exception as exc:
        st.error(f"Capital calc failed: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RWA", f"{summary['rwa']:,.2f}")
    c2.metric("Equity", f"{summary['equity']:,.2f}")
    c3.metric("Capital ratio", f"{summary['ratio']:.4%}")
    pass_str = "PASS ✅" if summary["pass"] else "FAIL ❌"
    c4.metric(f"vs floor {summary['floor']:.0%}", pass_str)

    exp_df = pd.DataFrame(
        [
            {
                "Ticker": t,
                "Dollar exposure": summary["exposures"][t],
                "Risk weight": summary["weights"][t],
                "RWA contribution": abs(summary["exposures"][t]) * summary["weights"][t],
            }
            for t in summary["exposures"]
        ]
    )
    st.dataframe(
        exp_df.style.format(
            {
                "Dollar exposure": "{:,.2f}",
                "Risk weight": "{:.3f}",
                "RWA contribution": "{:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ── DFAST scenarios ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### B · DFAST-style stress")
    st.caption(
        "Uniform equity shock applied across all portfolio underlyings. "
        "Rates shock recorded for reference (no rate-sensitive instruments in "
        "this portfolio)."
    )

    if st.button("Run DFAST scenarios", key="cap_dfast"):
        try:
            results = run_dfast(portfolio, current_prices, pricing_date)
            rows = []
            for name, r in results.items():
                rows.append(
                    {
                        "Scenario": name,
                        "Equity shock": f"{r['equity_shock']:+.0%}",
                        "Rates (bp)": f"{r['rates_bp']:+.0f}",
                        "V_pre": r["V_pre"],
                        "V_post": r["V_post"],
                        "PnL": r["pnl"],
                        "PnL %": r["pnl_pct"],
                    }
                )
            dfast_df = pd.DataFrame(rows)
            st.dataframe(
                dfast_df.style.format(
                    {
                        "V_pre": "{:,.2f}",
                        "V_post": "{:,.2f}",
                        "PnL": "{:,.2f}",
                        "PnL %": "{:+.2%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"DFAST failed: {exc}")

    # ── Custom stress ─────────────────────────────────────────────────────────
    with st.expander("Custom stress scenario"):
        shock_df = pd.DataFrame(
            {"Ticker": tickers, "Shock (e.g. -0.30 = −30%)": [0.0] * len(tickers)}
        )
        edited_shock = st.data_editor(
            shock_df, num_rows="fixed", use_container_width=True, key="cap_custom_shock"
        )
        if st.button("Run custom shock", key="cap_custom_run"):
            try:
                shock_map = {
                    str(row["Ticker"]): float(row["Shock (e.g. -0.30 = −30%)"])
                    for _, row in edited_shock.iterrows()
                    if pd.notna(row["Ticker"])
                }
                res = run_custom_stress(
                    portfolio, current_prices, shock_map, pricing_date
                )
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("V pre", f"{res['V_pre']:,.2f}")
                cc2.metric("V post", f"{res['V_post']:,.2f}")
                cc3.metric("PnL", f"{res['pnl']:,.2f}", f"{res['pnl_pct']:+.2%}")
            except Exception as exc:
                st.error(f"Custom stress failed: {exc}")
