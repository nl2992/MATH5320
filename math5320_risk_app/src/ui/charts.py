"""
charts.py
Plotly chart helpers for the Streamlit UI.
All functions return plotly.graph_objects.Figure objects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Loss distribution histogram ────────────────────────────────────────────────

def loss_histogram(
    losses: np.ndarray,
    var: float,
    es: float,
    title: str = "Loss Distribution",
    var_confidence: float = 0.99,
) -> go.Figure:
    """Histogram of simulated/historical losses with VaR and ES lines."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=losses,
            nbinsx=60,
            name="Losses",
            marker_color="#4C78A8",
            opacity=0.75,
        )
    )
    fig.add_vline(
        x=var,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"VaR ({var_confidence:.0%}): ${var:,.0f}",
        annotation_position="top right",
        annotation_font_color="#E45756",
    )
    fig.add_vline(
        x=es,
        line_dash="dot",
        line_color="#F58518",
        annotation_text=f"ES: ${es:,.0f}",
        annotation_position="top left",
        annotation_font_color="#F58518",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Loss ($)",
        yaxis_title="Frequency",
        legend_title="",
        template="plotly_white",
        height=400,
    )
    return fig


# ── Correlation heatmap ────────────────────────────────────────────────────────

def correlation_heatmap(log_returns: pd.DataFrame, lookback_days: int) -> go.Figure:
    """Correlation matrix heatmap of the lookback window."""
    window = log_returns.tail(lookback_days)
    corr = window.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Return Correlation Matrix (last {lookback_days} days)",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


# ── Backtest chart ─────────────────────────────────────────────────────────────

def backtest_chart(bt_df: pd.DataFrame, var_confidence: float) -> go.Figure:
    """Realised loss vs VaR forecast with exception markers."""
    fig = go.Figure()

    # VaR forecast line
    fig.add_trace(
        go.Scatter(
            x=bt_df["date"],
            y=bt_df["var_forecast"],
            mode="lines",
            name=f"VaR Forecast ({var_confidence:.0%})",
            line=dict(color="#E45756", width=1.5),
        )
    )

    # Realised loss line
    fig.add_trace(
        go.Scatter(
            x=bt_df["date"],
            y=bt_df["realized_loss"],
            mode="lines",
            name="Realised Loss",
            line=dict(color="#4C78A8", width=1),
            opacity=0.8,
        )
    )

    # Exception markers
    exceptions = bt_df[bt_df["exception"] == 1]
    if not exceptions.empty:
        fig.add_trace(
            go.Scatter(
                x=exceptions["date"],
                y=exceptions["realized_loss"],
                mode="markers",
                name="Exception",
                marker=dict(color="#F58518", size=8, symbol="x"),
            )
        )

    fig.update_layout(
        title="Walk-Forward VaR Backtest",
        xaxis_title="Date",
        yaxis_title="Loss ($)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── VaR comparison bar chart ───────────────────────────────────────────────────

def var_comparison_bar(results: dict) -> go.Figure:
    """Side-by-side bar chart comparing VaR and ES across models."""
    models = ["Historical", "Parametric", "Monte Carlo"]
    keys = ["historical", "parametric", "monte_carlo"]

    var_vals = [results[k]["var"] for k in keys]
    es_vals = [results[k]["es"] for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="VaR", x=models, y=var_vals, marker_color="#4C78A8")
    )
    fig.add_trace(
        go.Bar(name="ES", x=models, y=es_vals, marker_color="#F58518")
    )
    fig.update_layout(
        barmode="group",
        title="VaR and ES Comparison by Model",
        yaxis_title="Dollar Loss ($)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Price history line chart ───────────────────────────────────────────────────

def price_history_chart(prices: pd.DataFrame, lookback_days: int) -> go.Figure:
    """Normalised price history for the lookback window."""
    window = prices.tail(lookback_days + 1)
    normalised = window / window.iloc[0] * 100

    fig = go.Figure()
    for col in normalised.columns:
        fig.add_trace(
            go.Scatter(x=normalised.index, y=normalised[col], mode="lines", name=col)
        )
    fig.update_layout(
        title=f"Normalised Price History (last {lookback_days} days, base=100)",
        xaxis_title="Date",
        yaxis_title="Indexed Price",
        template="plotly_white",
        height=380,
    )
    return fig
