"""
results_panel.py
Streamlit UI component for displaying risk analysis results and downloads.
"""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from src.risk.returns import compute_log_returns
from src.ui.charts import (
    correlation_heatmap,
    loss_histogram,
    var_comparison_bar,
    price_history_chart,
)


def render_results_panel(
    results: dict,
    portfolio_value: float,
    prices: pd.DataFrame,
    lookback_days: int,
    var_confidence: float,
    risk_params: dict | None = None,
) -> None:
    """
    Render the risk analysis results panel.

    Parameters
    ----------
    results : dict
        Output of RiskEngineService.run_all().
    portfolio_value : float
        Current mark-to-market portfolio value.
    prices : pd.DataFrame
        Full price history.
    lookback_days : int
        Lookback window used.
    var_confidence : float
        VaR confidence level.
    """
    # ── Portfolio value ────────────────────────────────────────────────────────
    st.subheader("Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    col2.metric(
        f"Historical VaR ({var_confidence:.1%})",
        f"${results['historical']['var']:,.2f}",
    )
    col3.metric(
        f"Historical ES",
        f"${results['historical']['es']:,.2f}",
    )

    pv_ok = portfolio_value is not None and portfolio_value > 0
    if not pv_ok:
        st.warning(
            "Portfolio value is non-positive — likely all weights cancelled, "
            "or market data missing. VaR/Portfolio ratios suppressed."
        )

    # ── VaR comparison table ───────────────────────────────────────────────────
    st.subheader("VaR / ES Comparison")
    comparison_cols = {
        "Model": ["Historical", "Parametric (Delta-Normal)", "Monte Carlo"],
        "VaR ($)": [
            results["historical"]["var"],
            results["parametric"]["var"],
            results["monte_carlo"]["var"],
        ],
        "ES ($)": [
            results["historical"]["es"],
            results["parametric"]["es"],
            results["monte_carlo"]["es"],
        ],
    }
    if pv_ok:
        comparison_cols["VaR / Portfolio (%)"] = [
            results["historical"]["var"] / portfolio_value * 100,
            results["parametric"]["var"] / portfolio_value * 100,
            results["monte_carlo"]["var"] / portfolio_value * 100,
        ]
    comparison_df = pd.DataFrame(comparison_cols)
    fmt = {"VaR ($)": "${:,.2f}", "ES ($)": "${:,.2f}"}
    if pv_ok:
        fmt["VaR / Portfolio (%)"] = "{:.2f}%"
    st.dataframe(
        comparison_df.style.format(fmt),
        use_container_width=True,
    )

    # ── VaR comparison bar chart ───────────────────────────────────────────────
    st.plotly_chart(var_comparison_bar(results), use_container_width=True)

    # ── Loss histograms ────────────────────────────────────────────────────────
    st.subheader("Loss Distributions")
    tab_hist, tab_mc = st.tabs(["Historical Simulation", "Monte Carlo"])

    with tab_hist:
        if "losses" in results["historical"]:
            st.plotly_chart(
                loss_histogram(
                    losses=results["historical"]["losses"],
                    var=results["historical"]["var"],
                    es=results["historical"]["es"],
                    title="Historical Simulation Loss Distribution",
                    var_confidence=var_confidence,
                ),
                use_container_width=True,
            )

    with tab_mc:
        if "losses" in results["monte_carlo"]:
            st.plotly_chart(
                loss_histogram(
                    losses=results["monte_carlo"]["losses"],
                    var=results["monte_carlo"]["var"],
                    es=results["monte_carlo"]["es"],
                    title="Monte Carlo Loss Distribution",
                    var_confidence=var_confidence,
                ),
                use_container_width=True,
            )

    # ── Correlation heatmap ────────────────────────────────────────────────────
    st.subheader("Return Correlations")
    log_ret = compute_log_returns(prices)
    st.plotly_chart(
        correlation_heatmap(log_ret, lookback_days),
        use_container_width=True,
    )

    # ── Price history ──────────────────────────────────────────────────────────
    st.subheader("Price History")
    st.plotly_chart(
        price_history_chart(prices, lookback_days),
        use_container_width=True,
    )

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("Downloads")
    _render_downloads(results, portfolio_value, prices, lookback_days, var_confidence, risk_params or {})


def _render_downloads(
    results: dict,
    portfolio_value: float,
    prices: pd.DataFrame,
    lookback_days: int,
    var_confidence: float,
    risk_params: dict,
) -> None:
    """Render download buttons for JSON summary and losses CSV."""
    col1, col2 = st.columns(2)

    # JSON summary
    summary = {
        "portfolio_value": portfolio_value,
        "pricing_date": (
            prices.index[-1].date().isoformat() if hasattr(prices.index[-1], "date") else None
        ),
        "var_confidence": var_confidence,
        "es_confidence": risk_params.get("es_confidence"),
        "lookback_days": lookback_days,
        "horizon_days": risk_params.get("horizon_days"),
        "estimator": risk_params.get("estimator"),
        "ewma_N": risk_params.get("ewma_N"),
        "n_simulations": risk_params.get("n_simulations"),
        "calibration_mode": risk_params.get("calibration_mode", "historical"),
        "manual_market_params_supplied": risk_params.get("manual_market_params") is not None,
        "option_vol_shock_mode": risk_params.get("option_vol_shock_mode", "fixed"),
        "option_vol_shock_beta": risk_params.get("option_vol_shock_beta"),
        "option_vol_shock_floor": risk_params.get("option_vol_shock_floor"),
        "historical": {
            "var": results["historical"]["var"],
            "es": results["historical"]["es"],
            "n_scenarios": results["historical"].get("n_scenarios"),
        },
        "parametric": {
            "var": results["parametric"]["var"],
            "es": results["parametric"]["es"],
            "portfolio_mean": results["parametric"].get("portfolio_mean"),
            "portfolio_vol": results["parametric"].get("portfolio_vol"),
        },
        "monte_carlo": {
            "var": results["monte_carlo"]["var"],
            "es": results["monte_carlo"]["es"],
            "n_simulations": results["monte_carlo"].get("n_simulations"),
        },
    }
    json_bytes = json.dumps(summary, indent=2).encode()

    with col1:
        st.download_button(
            label="Download JSON Summary",
            data=json_bytes,
            file_name="risk_summary.json",
            mime="application/json",
        )

    # Losses CSV
    loss_rows: list[dict[str, object]] = []
    if "losses" in results["historical"]:
        for idx, loss in enumerate(results["historical"]["losses"], start=1):
            loss_rows.append(
                {
                    "method": "historical",
                    "scenario_id": idx,
                    "loss": float(loss),
                }
            )
    if "losses" in results["monte_carlo"]:
        for idx, loss in enumerate(results["monte_carlo"]["losses"], start=1):
            loss_rows.append(
                {
                    "method": "monte_carlo",
                    "scenario_id": idx,
                    "loss": float(loss),
                }
            )

    if loss_rows:
        losses_df = pd.DataFrame(loss_rows)
        csv_bytes = losses_df.to_csv(index=False).encode()

        with col2:
            st.download_button(
                label="Download Losses CSV",
                data=csv_bytes,
                file_name="losses.csv",
                mime="text/csv",
            )
