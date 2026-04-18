"""
app.py
MATH5320 Portfolio Risk System — Streamlit entry point.

All risk logic lives in src/. This file handles only UI orchestration.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(__file__))

import json
from datetime import date

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_BACKTEST_MODEL,
    DEFAULT_ES_CONFIDENCE,
    DEFAULT_ESTIMATOR,
    DEFAULT_EWMA_N,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MC_SIMULATIONS,
    DEFAULT_VAR_CONFIDENCE,
)
from src.data.validation import validate_portfolio_tickers
from src.schemas import Portfolio
from src.services.risk_engine_service import RiskEngineService
from src.ui.capital_panel import render_capital_panel
from src.ui.cds_cva_panel import render_cds_cva_panel
from src.ui.charts import backtest_chart
from src.ui.credit_panel import render_credit_panel
from src.ui.market_data_panel import render_market_data_panel
from src.ui.portfolio_editor import render_portfolio_editor
from src.ui.results_panel import render_results_panel
from src.ui.risk_settings import render_risk_settings

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MATH5320 Risk System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session-state defaults (run once per session) ─────────────────────────────
# Initialize before any tab code references these keys so the app is stable
# regardless of which tab the user visits first.
st.session_state.setdefault(
    "risk_params",
    {
        "lookback_days": DEFAULT_LOOKBACK_DAYS,
        "horizon_days": DEFAULT_HORIZON_DAYS,
        "var_confidence": DEFAULT_VAR_CONFIDENCE,
        "es_confidence": DEFAULT_ES_CONFIDENCE,
        "estimator": DEFAULT_ESTIMATOR,
        "ewma_N": DEFAULT_EWMA_N,
        "n_simulations": DEFAULT_MC_SIMULATIONS,
    },
)
st.session_state.setdefault("portfolio", Portfolio())
st.session_state.setdefault("prices", None)


# ── Backtest results renderer ──────────────────────────────────────────────────
# Defined at module scope BEFORE the tabs so Streamlit's top-to-bottom
# reruns can always find it.
def _render_backtest_results(bt_result: dict, params: dict) -> None:
    """Display backtest results inline."""
    bt_df: pd.DataFrame = bt_result["backtest_df"]
    kupiec: dict = bt_result["kupiec"]
    model: str = bt_result["model"]

    if bt_df.empty:
        st.warning(
            bt_result.get(
                "reason",
                "Not enough data for backtesting. Try a shorter lookback window.",
            )
        )
        return

    # ── Realised loss vs VaR chart ─────────────────────────────────────────────
    st.plotly_chart(
        backtest_chart(bt_df, params["var_confidence"]),
        use_container_width=True,
    )

    # ── Exception summary ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", kupiec["n_observations"])
    col2.metric("Exceptions", kupiec["n_exceptions"])
    col3.metric(
        "Observed Exception Rate",
        f"{kupiec['p_hat']:.2%}" if kupiec["p_hat"] == kupiec["p_hat"] else "N/A",
    )
    col4.metric(
        "Expected Exception Rate",
        f"{kupiec['alpha']:.2%}",
    )

    # ── Kupiec test results ────────────────────────────────────────────────────
    st.subheader("Kupiec Unconditional Coverage Test")
    if kupiec["n_observations"] == 0:
        st.info("Backtest produced no observations — cannot run Kupiec test.")
    else:
        kupiec_df = pd.DataFrame(
            {
                "Metric": [
                    "LR Statistic",
                    "p-value",
                    "Reject H₀ at 5%?",
                    "Interpretation",
                ],
                "Value": [
                    f"{kupiec['lr_stat']:.4f}" if kupiec["lr_stat"] == kupiec["lr_stat"] else "N/A",
                    f"{kupiec['p_value']:.4f}" if kupiec["p_value"] == kupiec["p_value"] else "N/A",
                    "Yes" if kupiec["reject_h0"] else "No",
                    (
                        "Model FAILS: exception rate is statistically different from expected."
                        if kupiec["reject_h0"]
                        else "Model PASSES: exception rate is consistent with the VaR confidence level."
                    ),
                ],
            }
        )
        st.table(kupiec_df)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.subheader("Downloads")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        csv_bytes = bt_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Backtest CSV",
            data=csv_bytes,
            file_name=f"backtest_{model}.csv",
            mime="text/csv",
        )

    with col_d2:
        kupiec_json = json.dumps(
            {k: (v if v == v else None) for k, v in kupiec.items()},
            indent=2,
        ).encode()
        st.download_button(
            label="Download Kupiec Results JSON",
            data=kupiec_json,
            file_name="kupiec_test.json",
            mime="application/json",
        )


# ── Title ──────────────────────────────────────────────────────────────────────
st.title("📊 MATH5320 Portfolio Risk System")
st.caption(
    "Market Risk (Historical · Parametric · Monte Carlo) · "
    "Credit Risk (Reduced-form · Merton) · CDS / CVA · Capital & Stress"
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
(
    tab_portfolio,
    tab_data,
    tab_settings,
    tab_results,
    tab_backtest,
    tab_credit,
    tab_cds_cva,
    tab_capital,
) = st.tabs(
    [
        "1 · Portfolio Input",
        "2 · Market Data",
        "3 · Risk Settings",
        "4 · Run Analysis",
        "5 · Backtesting",
        "6 · Credit Risk",
        "7 · CDS / CVA",
        "8 · Capital & Stress",
    ]
)

# ── Tab 1: Portfolio Input ─────────────────────────────────────────────────────
with tab_portfolio:
    portfolio = render_portfolio_editor()
    st.session_state["portfolio"] = portfolio

    # Show summary
    n_stocks = len(portfolio.stocks)
    n_options = len(portfolio.options)
    st.info(f"Portfolio: **{n_stocks}** stock position(s), **{n_options}** option position(s).")

# ── Tab 2: Market Data ─────────────────────────────────────────────────────────
with tab_data:
    portfolio_for_data = st.session_state.get("portfolio", portfolio)
    all_tickers = [p.ticker for p in portfolio_for_data.stocks] + [
        p.underlying_ticker for p in portfolio_for_data.options
    ]
    prices = render_market_data_panel(all_tickers)
    if prices is not None:
        st.session_state["prices"] = prices

# ── Tab 3: Risk Settings ───────────────────────────────────────────────────────
with tab_settings:
    risk_params = render_risk_settings()
    st.session_state["risk_params"] = risk_params

    # Show EWMA lambda
    if risk_params["estimator"] == "ewma":
        N = risk_params["ewma_N"]
        lam = (N - 1) / (N + 1)
        st.info(f"EWMA λ = (N-1)/(N+1) = ({N}-1)/({N}+1) = **{lam:.4f}**")

# ── Tab 4: Run Analysis ────────────────────────────────────────────────────────
with tab_results:
    st.subheader("Run Risk Analysis")

    prices_ready = st.session_state.get("prices") is not None
    portfolio_ready = (
        len(st.session_state.get("portfolio", portfolio).stocks) > 0
        or len(st.session_state.get("portfolio", portfolio).options) > 0
    )

    if not prices_ready:
        st.warning("Please load market data in the **Market Data** tab first.")
    elif not portfolio_ready:
        st.warning("Please add at least one position in the **Portfolio Input** tab.")
    else:
        current_portfolio = st.session_state.get("portfolio", portfolio)
        current_prices: pd.DataFrame = st.session_state["prices"]
        current_params: dict = st.session_state["risk_params"]

        # Validate tickers
        ticker_errors = validate_portfolio_tickers(
            current_portfolio, list(current_prices.columns)
        )
        if ticker_errors:
            for err in ticker_errors:
                st.error(err)
        else:
            if st.button("Run Risk Analysis", type="primary", key="run_analysis"):
                with st.spinner("Computing VaR and ES across all models…"):
                    try:
                        service = RiskEngineService(
                            portfolio=current_portfolio,
                            prices=current_prices,
                            pricing_date=date.today(),
                            **current_params,
                        )
                        results = service.run_all()
                        pv = service.portfolio_value()
                        st.session_state["results"] = results
                        st.session_state["portfolio_value"] = pv
                        st.session_state["service"] = service
                        st.success("Analysis complete.")
                    except Exception as exc:
                        st.error(f"Risk engine error: {exc}")
                        st.exception(exc)

            if "results" in st.session_state:
                render_results_panel(
                    results=st.session_state["results"],
                    portfolio_value=st.session_state["portfolio_value"],
                    prices=st.session_state["prices"],
                    lookback_days=st.session_state["risk_params"]["lookback_days"],
                    var_confidence=st.session_state["risk_params"]["var_confidence"],
                )

# ── Tab 5: Backtesting ─────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("VaR Backtesting")

    if st.session_state.get("prices") is None:
        st.warning("Please load market data first.")
    else:
        current_portfolio = st.session_state.get("portfolio", portfolio)
        current_prices = st.session_state["prices"]
        current_params = st.session_state["risk_params"]

        col_bt1, col_bt2 = st.columns([1, 2])
        with col_bt1:
            bt_model = st.selectbox(
                "Backtest model",
                options=["historical", "parametric", "monte_carlo"],
                index=0,
                key="bt_model",
                help="Model used for each walk-forward VaR forecast.",
            )
            st.caption(
                f"Walk-forward window: {current_params['lookback_days']} days  |  "
                f"Horizon: {current_params['horizon_days']} day(s)  |  "
                f"Confidence: {current_params['var_confidence']:.1%}"
            )

        ticker_errors = validate_portfolio_tickers(
            current_portfolio, list(current_prices.columns)
        )
        if ticker_errors:
            for err in ticker_errors:
                st.error(err)
        else:
            if st.button("Run Backtest", type="primary", key="run_backtest"):
                with st.spinner(
                    "Running walk-forward backtest (this may take a moment)…"
                ):
                    try:
                        service = RiskEngineService(
                            portfolio=current_portfolio,
                            prices=current_prices,
                            pricing_date=date.today(),
                            **current_params,
                        )
                        bt_result = service.run_backtest(model=bt_model)
                        st.session_state["bt_result"] = bt_result
                        st.success("Backtest complete.")
                    except Exception as exc:
                        st.error(f"Backtest error: {exc}")
                        st.exception(exc)

            if "bt_result" in st.session_state:
                _render_backtest_results(st.session_state["bt_result"], current_params)

# ── Tab 6: Credit Risk ─────────────────────────────────────────────────────────
with tab_credit:
    render_credit_panel(
        portfolio=st.session_state.get("portfolio", portfolio),
        prices=st.session_state.get("prices"),
    )

# ── Tab 7: CDS / CVA ───────────────────────────────────────────────────────────
with tab_cds_cva:
    render_cds_cva_panel(
        portfolio=st.session_state.get("portfolio", portfolio),
        prices=st.session_state.get("prices"),
        risk_params=st.session_state["risk_params"],
    )

# ── Tab 8: Capital & Stress ────────────────────────────────────────────────────
with tab_capital:
    render_capital_panel(
        portfolio=st.session_state.get("portfolio", portfolio),
        prices=st.session_state.get("prices"),
    )
