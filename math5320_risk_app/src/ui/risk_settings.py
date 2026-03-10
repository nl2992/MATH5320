"""
risk_settings.py
Streamlit UI component for risk parameter configuration.
"""
from __future__ import annotations

import streamlit as st

from src.config import (
    DEFAULT_ES_CONFIDENCE,
    DEFAULT_ESTIMATOR,
    DEFAULT_EWMA_N,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MC_SIMULATIONS,
    DEFAULT_VAR_CONFIDENCE,
)


def render_risk_settings() -> dict:
    """
    Render the risk settings panel.

    Returns
    -------
    dict with keys:
        lookback_days, horizon_days, var_confidence, es_confidence,
        estimator, ewma_N, n_simulations
    """
    st.subheader("Risk Parameters")

    col1, col2 = st.columns(2)

    with col1:
        lookback_days = st.number_input(
            "Lookback window (trading days)",
            min_value=30,
            max_value=2520,
            value=DEFAULT_LOOKBACK_DAYS,
            step=10,
            help="Number of historical observations used for estimation.",
            key="lookback_days",
        )
        horizon_days = st.number_input(
            "Risk horizon (trading days)",
            min_value=1,
            max_value=60,
            value=DEFAULT_HORIZON_DAYS,
            step=1,
            help="Forecast horizon h.",
            key="horizon_days",
        )
        var_confidence = st.slider(
            "VaR confidence level",
            min_value=0.90,
            max_value=0.999,
            value=DEFAULT_VAR_CONFIDENCE,
            step=0.001,
            format="%.3f",
            key="var_confidence",
        )

    with col2:
        es_confidence = st.slider(
            "ES confidence level",
            min_value=0.90,
            max_value=0.999,
            value=DEFAULT_ES_CONFIDENCE,
            step=0.001,
            format="%.3f",
            key="es_confidence",
        )
        estimator = st.selectbox(
            "Estimator type",
            options=["window", "ewma"],
            index=0 if DEFAULT_ESTIMATOR == "window" else 1,
            help="'window' = equal-weight rolling; 'ewma' = exponentially weighted.",
            key="estimator",
        )
        ewma_N = st.number_input(
            "EWMA N parameter  (λ = (N-1)/(N+1))",
            min_value=5,
            max_value=500,
            value=DEFAULT_EWMA_N,
            step=5,
            help="Only used when estimator = 'ewma'.",
            key="ewma_N",
            disabled=(estimator == "window"),
        )

    n_simulations = st.number_input(
        "Monte Carlo simulations",
        min_value=1_000,
        max_value=100_000,
        value=DEFAULT_MC_SIMULATIONS,
        step=1_000,
        key="n_simulations",
    )

    return {
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "var_confidence": float(var_confidence),
        "es_confidence": float(es_confidence),
        "estimator": estimator,
        "ewma_N": int(ewma_N),
        "n_simulations": int(n_simulations),
    }
