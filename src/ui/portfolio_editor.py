"""
portfolio_editor.py
Streamlit UI component for building and editing the portfolio.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.schemas import OptionPosition, Portfolio, StockPosition


def render_portfolio_editor() -> Portfolio:
    """
    Render the portfolio editor UI.

    Returns
    -------
    Portfolio
        The portfolio constructed from the user's inputs.
    """
    st.subheader("Stock Positions")
    stocks = _render_stock_table()

    st.subheader("Option Positions")
    options = _render_option_table()

    return Portfolio(stocks=stocks, options=options)


# ── Stock table ────────────────────────────────────────────────────────────────

_DEFAULT_STOCKS = pd.DataFrame(
    {
        "Ticker": ["AAPL", "MSFT"],
        "Quantity": [100.0, 50.0],
    }
)


def _render_stock_table() -> list[StockPosition]:
    """Editable stock position table."""
    edited = st.data_editor(
        st.session_state.get("stock_df", _DEFAULT_STOCKS),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance symbol"),
            "Quantity": st.column_config.NumberColumn(
                "Quantity", help="Positive = long, negative = short", format="%.0f"
            ),
        },
        key="stock_editor",
    )
    st.session_state["stock_df"] = edited

    positions: list[StockPosition] = []
    for _, row in edited.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        qty = float(row["Quantity"]) if pd.notna(row["Quantity"]) else 0.0
        if ticker:
            positions.append(StockPosition(ticker=ticker, quantity=qty))
    return positions


# ── Option table ───────────────────────────────────────────────────────────────

_DEFAULT_OPTIONS = pd.DataFrame(
    {
        "Label": ["AAPL_CALL"],
        "Underlying": ["AAPL"],
        "Type": ["call"],
        "Quantity": [10.0],
        "Strike": [200.0],
        "Maturity": [str(date.today() + timedelta(days=90))],
        "Volatility": [0.25],
        "Risk-Free Rate": [0.05],
        "Dividend Yield": [0.0],
        "Multiplier": [100.0],
    }
)


def _render_option_table() -> list[OptionPosition]:
    """Editable option position table."""
    edited = st.data_editor(
        st.session_state.get("option_df", _DEFAULT_OPTIONS),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Label": st.column_config.TextColumn("Label"),
            "Underlying": st.column_config.TextColumn("Underlying Ticker"),
            "Type": st.column_config.SelectboxColumn("Type", options=["call", "put"]),
            "Quantity": st.column_config.NumberColumn("Qty (contracts)", format="%.0f"),
            "Strike": st.column_config.NumberColumn("Strike ($)", format="%.2f"),
            "Maturity": st.column_config.TextColumn("Maturity (YYYY-MM-DD)"),
            "Volatility": st.column_config.NumberColumn("Vol (e.g. 0.25)", format="%.4f"),
            "Risk-Free Rate": st.column_config.NumberColumn("r (e.g. 0.05)", format="%.4f"),
            "Dividend Yield": st.column_config.NumberColumn("q (e.g. 0.00)", format="%.4f"),
            "Multiplier": st.column_config.NumberColumn("Multiplier", format="%.0f"),
        },
        key="option_editor",
    )
    st.session_state["option_df"] = edited

    positions: list[OptionPosition] = []
    for _, row in edited.iterrows():
        try:
            label = str(row["Label"]).strip()
            underlying = str(row["Underlying"]).strip().upper()
            if not label or not underlying:
                continue
            maturity = date.fromisoformat(str(row["Maturity"]).strip())
            positions.append(
                OptionPosition(
                    ticker=label,
                    underlying_ticker=underlying,
                    option_type=str(row["Type"]).lower(),
                    quantity=float(row["Quantity"]),
                    strike=float(row["Strike"]),
                    maturity_date=maturity,
                    volatility=float(row["Volatility"]),
                    risk_free_rate=float(row["Risk-Free Rate"]),
                    dividend_yield=float(row["Dividend Yield"]),
                    contract_multiplier=float(row["Multiplier"]),
                )
            )
        except Exception:
            continue
    return positions
