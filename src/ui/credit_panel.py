"""
credit_panel.py
Streamlit UI for credit-risk analysis — formula-sheet §8 (reduced-form) + §9 (Merton).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.data.market_data import fetch_risk_free_rate
from src.schemas import Portfolio
from src.services.credit_service import (
    merton_implied_B_for_survival,
    merton_summary,
    reduced_form_summary,
)


def render_credit_panel(
    portfolio: Portfolio,
    prices: pd.DataFrame | None,
) -> None:
    """Render both Reduced-form and Merton sections."""
    st.subheader("Credit Risk")
    st.caption(
        "Reduced-form (§8) — hazard / survival / default density. "
        "Merton (§9) — structural default of a firm modelled as GBM."
    )

    _render_reduced_form_section()
    st.divider()
    _render_merton_section(portfolio, prices)


# ── Section A — Reduced-form ──────────────────────────────────────────────────

def _render_reduced_form_section() -> None:
    st.markdown("### A · Reduced-form (hazard rate)")

    col1, col2, col3 = st.columns(3)
    with col1:
        lam = st.number_input(
            "Hazard rate λ",
            min_value=0.0,
            max_value=2.0,
            value=0.03,
            step=0.005,
            format="%.4f",
            key="rf_lam",
            help="Constant hazard rate (annualised).",
        )
    with col2:
        R = st.number_input(
            "Recovery R",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            key="rf_R",
        )
    with col3:
        r_disc = st.number_input(
            "Discount rate r",
            min_value=0.0,
            max_value=0.25,
            value=0.03,
            step=0.005,
            format="%.4f",
            key="rf_r",
        )

    horizons_str = st.text_input(
        "Horizons (years, comma-separated)",
        value="0.25, 0.5, 1, 2, 3, 5, 10",
        key="rf_horizons",
    )
    try:
        horizons = [float(x.strip()) for x in horizons_str.split(",") if x.strip()]
    except ValueError:
        st.error("Horizons must be comma-separated numbers (e.g. 1, 2, 5).")
        return
    if not horizons:
        st.info("Enter at least one horizon to compute survival/default.")
        return

    summary = reduced_form_summary(lam=lam, horizons=horizons, R=R, r=r_disc)

    col_a, col_b = st.columns(2)
    col_a.metric("LGD = 1 − R", f"{summary['LGD']:.2%}")
    col_b.metric(
        "CDS approx spread  (1−R)·λ",
        f"{summary['approx_cds'] * 1e4:,.1f} bps",
        help="Constant-hazard approximation of the CDS par spread (formula-sheet §10).",
    )

    df = pd.DataFrame(summary["rows"])
    df_fmt = df.copy()
    df_fmt["survival"] = df_fmt["survival"].map(lambda x: f"{x:.4%}")
    df_fmt["cum_default"] = df_fmt["cum_default"].map(lambda x: f"{x:.4%}")
    df_fmt["density"] = df_fmt["density"].map(lambda x: f"{x:.6f}")
    df_fmt["risky_zcb"] = df_fmt["risky_zcb"].map(lambda x: f"{x:.6f}")
    df_fmt["spread"] = df_fmt["spread"].map(
        lambda x: f"{x * 1e4:,.1f} bps" if not np.isnan(x) else "N/A"
    )
    st.dataframe(df_fmt, use_container_width=True, hide_index=True)


# ── Section B — Merton ────────────────────────────────────────────────────────

def _render_merton_section(
    portfolio: Portfolio,
    prices: pd.DataFrame | None,
) -> None:
    st.markdown("### B · Merton structural model")

    # Prefill-from-portfolio helper.
    tickers = sorted(
        {p.ticker for p in portfolio.stocks}
        | {p.underlying_ticker for p in portfolio.options}
    )

    prefill_col1, prefill_col2 = st.columns([2, 1])
    with prefill_col1:
        selected_ticker = st.selectbox(
            "Prefill from ticker (optional)",
            options=["(none)"] + tickers,
            key="merton_ticker",
            help="Uses last price as V₀ and trailing-252-day log returns to set μ, σ.",
        )
    with prefill_col2:
        do_prefill = st.button("Prefill", key="merton_prefill")

    prefill_V0 = None
    prefill_mu = None
    prefill_sigma = None
    prefill_r = None

    if do_prefill and selected_ticker != "(none)":
        if prices is None or selected_ticker not in prices.columns:
            st.warning(
                f"Ticker {selected_ticker} not found in loaded prices. "
                "Load market data first."
            )
        else:
            s = prices[selected_ticker].dropna()
            if len(s) < 30:
                st.warning("Not enough price history to estimate μ and σ.")
            else:
                ret = np.log(s / s.shift(1)).dropna()
                tail = ret.tail(252)
                prefill_V0 = float(s.iloc[-1])
                prefill_mu = float(tail.mean() * 252)
                prefill_sigma = float(tail.std() * np.sqrt(252))
                try:
                    prefill_r = float(fetch_risk_free_rate(date.today(), fallback=0.04))
                except Exception:
                    prefill_r = 0.04
                st.session_state["merton_V0"] = prefill_V0
                st.session_state["merton_mu"] = prefill_mu
                st.session_state["merton_sigma"] = prefill_sigma
                st.session_state["merton_r"] = prefill_r
                st.success(
                    f"Prefilled V₀={prefill_V0:.2f}, μ={prefill_mu:.2%}, "
                    f"σ={prefill_sigma:.2%}, r={prefill_r:.2%}."
                )

    col1, col2, col3 = st.columns(3)
    with col1:
        V0 = st.number_input(
            "Firm value V₀",
            min_value=0.01,
            value=st.session_state.get("merton_V0", 100.0),
            step=1.0,
            format="%.4f",
            key="merton_V0_input",
        )
        B = st.number_input(
            "Default barrier B",
            min_value=0.01,
            value=80.0,
            step=1.0,
            format="%.4f",
            key="merton_B_input",
        )
    with col2:
        mu = st.number_input(
            "Real-world drift μ (P-measure)",
            min_value=-0.5,
            max_value=0.5,
            value=st.session_state.get("merton_mu", 0.08),
            step=0.01,
            format="%.4f",
            key="merton_mu_input",
        )
        r_ = st.number_input(
            "Risk-free r (Q-measure)",
            min_value=0.0,
            max_value=0.25,
            value=st.session_state.get("merton_r", 0.04),
            step=0.005,
            format="%.4f",
            key="merton_r_input",
        )
    with col3:
        sigma = st.number_input(
            "Asset vol σ",
            min_value=0.001,
            max_value=3.0,
            value=st.session_state.get("merton_sigma", 0.25),
            step=0.01,
            format="%.4f",
            key="merton_sigma_input",
        )
        T = st.number_input(
            "Horizon T (years)",
            min_value=0.01,
            max_value=50.0,
            value=1.0,
            step=0.25,
            format="%.4f",
            key="merton_T_input",
        )

    try:
        snap = merton_summary(V0=V0, B=B, r=r_, mu=mu, sigma=sigma, T=T)
    except Exception as exc:
        st.error(f"Merton inputs invalid: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q-PD  (pricing)", f"{snap['Q']['PD']:.4%}")
    c2.metric("P-PD  (real world)", f"{snap['P']['PD']:.4%}")
    c3.metric("Equity E₀", f"{snap['E0']:.4f}")
    c4.metric("Debt D₀", f"{snap['D0']:.4f}")

    det = pd.DataFrame(
        {
            "Measure": ["Q (ν=r)", "P (ν=μ)"],
            "d₁": [snap["Q"]["d1"], snap["P"]["d1"]],
            "d₂": [snap["Q"]["d2"], snap["P"]["d2"]],
            "PD = N(−d₂)": [snap["Q"]["PD"], snap["P"]["PD"]],
        }
    )
    st.dataframe(det, use_container_width=True, hide_index=True)

    # ── Target-survival inversion (HW IX) ─────────────────────────────────────
    st.markdown("**Target-survival inversion** — HW IX")
    target_surv = st.slider(
        "Target survival s*",
        min_value=0.01,
        max_value=0.999,
        value=0.95,
        step=0.01,
        key="merton_target_surv",
    )
    try:
        B_star = merton_implied_B_for_survival(
            V0=V0, target_survival=target_surv, r=r_, sigma=sigma, T=T
        )
        st.metric("Implied barrier B*", f"{B_star:.4f}")
    except Exception as exc:
        st.error(f"Inversion failed: {exc}")
