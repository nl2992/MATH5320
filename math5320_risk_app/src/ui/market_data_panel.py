"""
market_data_panel.py
Streamlit UI component for loading market data (CSV or yfinance).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.data.market_data import download_adjusted_close, load_price_history_csv
from src.data.validation import validate_price_dataframe


def render_market_data_panel(portfolio_tickers: list[str]) -> pd.DataFrame | None:
    """
    Render the market data panel.

    Parameters
    ----------
    portfolio_tickers : list[str]
        Tickers required by the current portfolio (for yfinance pre-fill).

    Returns
    -------
    pd.DataFrame or None
        Loaded price DataFrame, or None if not yet loaded.
    """
    st.subheader("Market Data Source")

    source = st.radio(
        "Data source",
        options=["Yahoo Finance (download)", "Upload CSV"],
        horizontal=True,
        key="data_source",
    )

    prices: pd.DataFrame | None = None

    # Deduplicate tickers for display
    unique_tickers = list(dict.fromkeys(portfolio_tickers))

    if source == "Yahoo Finance (download)":
        prices = _render_yfinance_panel(unique_tickers)
    else:
        prices = _render_csv_panel()

    if prices is not None:
        errors = validate_price_dataframe(prices)
        if errors:
            for err in errors:
                st.error(err)
            return None

        st.success(
            f"Loaded {len(prices)} rows × {len(prices.columns)} tickers "
            f"({prices.index[0].date()} → {prices.index[-1].date()})"
        )
        with st.expander("Preview price data"):
            st.dataframe(prices.tail(10), use_container_width=True)

    return prices


# ── yfinance panel ─────────────────────────────────────────────────────────────

def _render_yfinance_panel(portfolio_tickers: list[str]) -> pd.DataFrame | None:
    default_tickers = " ".join(portfolio_tickers) if portfolio_tickers else "AAPL MSFT"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tickers_input = st.text_input(
            "Tickers (space-separated)",
            value=default_tickers,
            key="yf_tickers",
        )
    with col2:
        start_date = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=5 * 365),
            key="yf_start",
        )
    with col3:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            key="yf_end",
        )

    if st.button("Download from Yahoo Finance", key="yf_download"):
        tickers = list(dict.fromkeys(
            t.strip().upper() for t in tickers_input.split() if t.strip()
        ))
        if not tickers:
            st.error("Please enter at least one ticker.")
            return None
        with st.spinner("Downloading price data…"):
            try:
                prices = download_adjusted_close(
                    tickers=tickers,
                    start=str(start_date),
                    end=str(end_date),
                )
                st.session_state["prices"] = prices
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                return None

    return st.session_state.get("prices")


# ── CSV upload panel ───────────────────────────────────────────────────────────

def _render_csv_panel() -> pd.DataFrame | None:
    st.markdown(
        "Upload a wide-format CSV: `Date` column as index, one column per ticker."
    )
    st.code("Date,AAPL,MSFT,SPY\n2020-01-02,300.1,150.2,320.5\n...", language="text")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")

    if uploaded is not None:
        try:
            prices = load_price_history_csv(uploaded)
            st.session_state["prices"] = prices
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")
            return None

    return st.session_state.get("prices")
