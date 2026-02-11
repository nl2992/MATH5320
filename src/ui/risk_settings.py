"""
risk_settings.py
Streamlit UI component for risk parameter configuration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_CALIBRATION_MODE,
    DEFAULT_ES_CONFIDENCE,
    DEFAULT_ESTIMATOR,
    DEFAULT_EWMA_N,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MC_SIMULATIONS,
    DEFAULT_OPTION_VOL_SHOCK_BETA,
    DEFAULT_OPTION_VOL_SHOCK_FLOOR,
    DEFAULT_OPTION_VOL_SHOCK_MODE,
    DEFAULT_VAR_CONFIDENCE,
)


def render_risk_settings(portfolio_tickers: list[str] | None = None) -> dict:
    """
    Render the risk settings panel.

    Returns
    -------
    dict with keys:
        lookback_days, horizon_days, var_confidence, es_confidence,
        estimator, ewma_N, n_simulations, calibration_mode,
        manual_market_params, option_vol_shock_mode,
        option_vol_shock_beta, option_vol_shock_floor
    """
    st.subheader("Risk Parameters")

    calibration_mode = st.radio(
        "Calibration mode",
        options=["historical", "manual"],
        index=0 if DEFAULT_CALIBRATION_MODE == "historical" else 1,
        horizontal=True,
        key="calibration_mode",
        help=(
            "'historical' estimates mean/covariance from price history. "
            "'manual' uses user-supplied daily mean, daily volatility, and correlation."
        ),
    )

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
            help=(
                "Used when calibration mode is historical. "
                "'window' = equal-weight rolling; 'ewma' = exponentially weighted."
            ),
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

    if abs(float(es_confidence) - float(var_confidence)) > 1e-12:
        st.info(
            "VaR and ES are currently using different confidence levels. "
            "That is supported here, but ES should then be read as a separate "
            "tail measure rather than 'average loss beyond the displayed VaR'."
        )

    n_simulations = st.number_input(
        "Monte Carlo simulations",
        min_value=1_000,
        max_value=100_000,
        value=DEFAULT_MC_SIMULATIONS,
        step=1_000,
        key="n_simulations",
    )

    st.divider()
    st.caption(
        "Option-volatility scenario handling applies to the full-repricing engines "
        "(historical simulation and Monte Carlo)."
    )

    col3, col4, col5 = st.columns(3)
    with col3:
        option_vol_shock_mode = st.selectbox(
            "Option volatility shock mode",
            options=["fixed", "underlying_beta"],
            index=0 if DEFAULT_OPTION_VOL_SHOCK_MODE == "fixed" else 1,
            key="option_vol_shock_mode",
            help=(
                "'fixed' keeps implied vol unchanged. "
                "'underlying_beta' applies sigma' = max(floor, sigma * (1 - beta * R))."
            ),
        )
    with col4:
        option_vol_shock_beta = st.number_input(
            "Vol shock beta",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_OPTION_VOL_SHOCK_BETA,
            step=0.1,
            key="option_vol_shock_beta",
            disabled=(option_vol_shock_mode == "fixed"),
        )
    with col5:
        option_vol_shock_floor = st.number_input(
            "Vol floor",
            min_value=0.0001,
            max_value=5.0,
            value=DEFAULT_OPTION_VOL_SHOCK_FLOOR,
            step=0.01,
            format="%.4f",
            key="option_vol_shock_floor",
            disabled=(option_vol_shock_mode == "fixed"),
        )

    manual_market_params = None
    if calibration_mode == "manual":
        tickers = _normalise_tickers(portfolio_tickers)
        st.divider()
        st.subheader("Manual Market-Risk Parameters")
        st.info(
            "Manual inputs override the estimated mean/covariance for the parametric and "
            "Monte Carlo engines. Historical simulation still uses the loaded price history."
        )
        manual_market_params = _render_manual_market_inputs(tickers)

    return {
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "var_confidence": float(var_confidence),
        "es_confidence": float(es_confidence),
        "estimator": estimator,
        "ewma_N": int(ewma_N),
        "n_simulations": int(n_simulations),
        "calibration_mode": calibration_mode,
        "manual_market_params": manual_market_params,
        "option_vol_shock_mode": option_vol_shock_mode,
        "option_vol_shock_beta": float(option_vol_shock_beta),
        "option_vol_shock_floor": float(option_vol_shock_floor),
    }


def _normalise_tickers(portfolio_tickers: list[str] | None) -> list[str]:
    seen: set[str] = set()
    tickers: list[str] = []
    for raw in portfolio_tickers or ["AAPL", "MSFT"]:
        ticker = str(raw).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers or ["AAPL", "MSFT"]


def _default_manual_summary(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Mean Log Return (daily)": [0.0] * len(tickers),
            "Volatility (daily)": [0.02] * len(tickers),
        }
    )


def _default_manual_corr(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        np.eye(len(tickers), dtype=float),
        index=tickers,
        columns=tickers,
    )


def _sync_manual_state(tickers: list[str]) -> None:
    summary = st.session_state.get("manual_params_df")
    existing_rows: dict[str, dict[str, float]] = {}
    if isinstance(summary, pd.DataFrame) and "Ticker" in summary.columns:
        for _, row in summary.iterrows():
            ticker = str(row.get("Ticker", "")).strip().upper()
            if ticker:
                existing_rows[ticker] = {
                    "Mean Log Return (daily)": float(row.get("Mean Log Return (daily)", 0.0)),
                    "Volatility (daily)": float(row.get("Volatility (daily)", 0.02)),
                }

    st.session_state["manual_params_df"] = pd.DataFrame(
        {
            "Ticker": tickers,
            "Mean Log Return (daily)": [
                existing_rows.get(t, {}).get("Mean Log Return (daily)", 0.0) for t in tickers
            ],
            "Volatility (daily)": [
                existing_rows.get(t, {}).get("Volatility (daily)", 0.02) for t in tickers
            ],
        }
    )

    corr = st.session_state.get("manual_corr_df")
    corr_df = _default_manual_corr(tickers)
    if isinstance(corr, pd.DataFrame):
        for i in tickers:
            for j in tickers:
                if i in corr.index and j in corr.columns:
                    try:
                        corr_df.loc[i, j] = float(corr.loc[i, j])
                    except Exception:
                        pass
    st.session_state["manual_corr_df"] = corr_df


def _render_manual_market_inputs(tickers: list[str]) -> dict | None:
    current = st.session_state.get("manual_params_df")
    if not isinstance(current, pd.DataFrame) or current.get("Ticker") is None:
        _sync_manual_state(tickers)
    else:
        current_tickers = [str(t).strip().upper() for t in current["Ticker"].tolist() if str(t).strip()]
        if current_tickers != tickers:
            _sync_manual_state(tickers)

    summary = st.data_editor(
        st.session_state["manual_params_df"],
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Mean Log Return (daily)": st.column_config.NumberColumn(
                "Mean Log Return (daily)", format="%.6f"
            ),
            "Volatility (daily)": st.column_config.NumberColumn(
                "Volatility (daily)", format="%.6f"
            ),
        },
        key="manual_params_editor",
    )
    st.session_state["manual_params_df"] = summary

    corr = st.data_editor(
        st.session_state["manual_corr_df"],
        use_container_width=True,
        key="manual_corr_editor",
    )
    st.session_state["manual_corr_df"] = corr

    errors: list[str] = []
    tickers = [str(t).strip().upper() for t in summary["Ticker"].tolist()]

    if len(set(tickers)) != len(tickers):
        errors.append("Manual parameter tickers must be unique.")

    try:
        mu_vals = pd.to_numeric(summary["Mean Log Return (daily)"], errors="raise").astype(float)
        vol_vals = pd.to_numeric(summary["Volatility (daily)"], errors="raise").astype(float)
    except Exception:
        errors.append("Manual mean/vol entries must all be numeric.")
        mu_vals = pd.Series(dtype=float)
        vol_vals = pd.Series(dtype=float)

    if not vol_vals.empty and (vol_vals <= 0).any():
        errors.append("Manual daily volatilities must be strictly positive.")

    corr_df = pd.DataFrame(corr, dtype=float)
    corr_df.index = tickers
    corr_df.columns = tickers

    if corr_df.shape != (len(tickers), len(tickers)):
        errors.append("Manual correlation matrix must be square and match the ticker list.")
    else:
        if not np.allclose(corr_df.values, corr_df.values.T, atol=1e-10):
            errors.append("Manual correlation matrix must be symmetric.")
        if not np.allclose(np.diag(corr_df.values), 1.0, atol=1e-10):
            errors.append("Manual correlation matrix must have ones on the diagonal.")
        if (np.abs(corr_df.values) > 1.0 + 1e-10).any():
            errors.append("Manual correlations must lie between -1 and 1.")
        eigvals = np.linalg.eigvalsh(corr_df.values)
        if np.min(eigvals) < -1e-10:
            errors.append("Manual correlation matrix must be positive semidefinite.")

    for err in errors:
        st.error(err)
    if errors:
        return None

    mu_daily = pd.Series(mu_vals.values, index=tickers, dtype=float)
    sigmas = pd.Series(vol_vals.values, index=tickers, dtype=float)
    cov_daily = pd.DataFrame(
        np.outer(sigmas.values, sigmas.values) * corr_df.values,
        index=tickers,
        columns=tickers,
    )
    return {
        "mu_daily": mu_daily,
        "cov_daily": cov_daily,
        "tickers": tickers,
    }
