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
    """Editable option position table with per-row validation."""
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
    row_errors: list[str] = []

    for idx, row in edited.iterrows():
        row_err = _validate_option_row(idx, row)
        if row_err is None:
            # Row is entirely blank — skip silently.
            continue
        if row_err:
            row_errors.append(row_err)
            continue

        # All fields validated individually in _validate_option_row.
        positions.append(
            OptionPosition(
                ticker=str(row["Label"]).strip(),
                underlying_ticker=str(row["Underlying"]).strip().upper(),
                option_type=str(row["Type"]).lower(),
                quantity=float(row["Quantity"]),
                strike=float(row["Strike"]),
                maturity_date=date.fromisoformat(str(row["Maturity"]).strip()),
                volatility=float(row["Volatility"]),
                risk_free_rate=float(row["Risk-Free Rate"]),
                dividend_yield=float(row["Dividend Yield"]),
                contract_multiplier=float(row["Multiplier"]),
            )
        )

    for err in row_errors:
        st.error(err)

    return positions


def _validate_option_row(idx: int, row) -> str | None:
    """
    Validate one option row.

    Returns
    -------
    None   -- row is entirely blank, caller should skip silently.
    ""     -- row is valid.
    <msg>  -- human-readable error describing the first failing field.
    """
    # A row is "entirely blank" if Label AND Underlying are both empty.
    label_raw = row.get("Label")
    underlying_raw = row.get("Underlying")
    label_blank = pd.isna(label_raw) or not str(label_raw).strip()
    under_blank = pd.isna(underlying_raw) or not str(underlying_raw).strip()
    if label_blank and under_blank:
        return None  # skip silently

    prefix = f"Option row {idx + 1}:"

    # Label + underlying presence.
    if label_blank:
        return f"{prefix} Label is required."
    if under_blank:
        return f"{prefix} Underlying is required."

    # Option type.
    opt_type = str(row.get("Type", "")).strip().lower()
    if opt_type not in ("call", "put"):
        return f"{prefix} Type must be 'call' or 'put' (got {row.get('Type')!r})."

    # Numeric fields.
    for field, positive in (
        ("Quantity", False),
        ("Strike", True),
        ("Volatility", True),
        ("Risk-Free Rate", False),
        ("Dividend Yield", False),
        ("Multiplier", True),
    ):
        val = row.get(field)
        if pd.isna(val):
            return f"{prefix} {field} is required."
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return f"{prefix} {field} must be a number (got {val!r})."
        if positive and fval <= 0:
            return f"{prefix} {field} must be strictly positive (got {fval})."

    # Maturity parse + future check.
    maturity_raw = row.get("Maturity")
    if pd.isna(maturity_raw) or not str(maturity_raw).strip():
        return f"{prefix} Maturity is required (YYYY-MM-DD)."
    try:
        maturity = date.fromisoformat(str(maturity_raw).strip())
    except ValueError:
        return f"{prefix} Maturity must be YYYY-MM-DD (got {maturity_raw!r})."
    if maturity <= date.today():
        return (
            f"{prefix} Maturity {maturity} must be in the future "
            "(expired options have no time value)."
        )

    return ""  # valid
