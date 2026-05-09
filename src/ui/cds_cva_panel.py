"""
cds_cva_panel.py
Streamlit UI for CDS pricing (§10) and CVA (§11).
"""
from __future__ import annotations

from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from src.credit.cva import cva_discounted, epe_profile_from_mc
from src.credit.mitigation import mitigated_cva, netted_exposure
from src.portfolio.portfolio import portfolio_value, reprice_portfolio
from src.risk.estimators import get_mean_cov
from src.risk.returns import compute_log_returns
from src.schemas import Portfolio
from src.services.credit_service import (
    cds_spread_for_schedule,
    cds_summary,
    cva_summary,
)


def render_cds_cva_panel(
    portfolio: Portfolio,
    prices: pd.DataFrame | None,
    risk_params: dict,
) -> None:
    """Render CDS + CVA sections."""
    st.subheader("CDS · CVA")
    st.caption(
        "CDS par spread (§10) under constant- or piecewise-hazard λ. "
        "CVA (§11) = (1−R)·Σ Ē·p̄ against a time-stepped exposure profile. "
        "Mitigated CVA (§11) with netting + CSA collateral."
    )

    _render_cds_section()
    st.divider()
    _render_cva_section(portfolio, prices, risk_params)
    st.divider()
    _render_mitigated_cva_section(portfolio, prices, risk_params)


# ── Section A · CDS ───────────────────────────────────────────────────────────

def _render_cds_section() -> None:
    st.markdown("### A · CDS par spread")

    col1, col2, col3 = st.columns(3)
    with col1:
        lam = st.number_input(
            "Flat hazard λ",
            min_value=0.0,
            max_value=2.0,
            value=0.03,
            step=0.005,
            format="%.4f",
            key="cds_lam",
        )
    with col2:
        R = st.number_input(
            "Recovery R", min_value=0.0, max_value=1.0, value=0.40, step=0.05,
            key="cds_R",
        )
    with col3:
        r_disc = st.number_input(
            "Discount r",
            min_value=0.0,
            max_value=0.25,
            value=0.03,
            step=0.005,
            format="%.4f",
            key="cds_r",
        )

    col4, col5 = st.columns(2)
    with col4:
        tenors_str = st.text_input(
            "Tenors (years)",
            value="1, 2, 3, 5, 7, 10",
            key="cds_tenors",
        )
    with col5:
        freq = st.selectbox(
            "Premium frequency",
            options=[1, 2, 4],
            index=2,
            format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x],
            key="cds_freq",
        )

    try:
        tenors = [float(x.strip()) for x in tenors_str.split(",") if x.strip()]
    except ValueError:
        st.error("Tenors must be comma-separated numbers.")
        return
    if not tenors:
        return

    try:
        summary = cds_summary(
            lam=lam, R=R, tenors=tenors, r=r_disc, premium_freq=float(freq),
        )
    except Exception as exc:
        st.error(f"CDS inputs invalid: {exc}")
        return

    c1, c2 = st.columns(2)
    c1.metric(
        "Constant-hazard approx  (1−R)·λ",
        f"{summary['approx_spread'] * 1e4:,.1f} bps",
        help="§14 landmark: λ=3%, R=40% → 180 bps.",
    )
    curve = summary["curve"]
    curve_df = pd.DataFrame(curve, columns=["Tenor (yr)", "Par spread"])
    curve_df["Par spread (bps)"] = curve_df["Par spread"] * 1e4
    c2.metric(
        "Par spread @ longest tenor",
        f"{curve_df['Par spread (bps)'].iloc[-1]:,.1f} bps",
    )
    st.line_chart(
        curve_df.set_index("Tenor (yr)")["Par spread (bps)"],
        height=260,
    )
    st.dataframe(
        curve_df[["Tenor (yr)", "Par spread (bps)"]].style.format(
            {"Par spread (bps)": "{:,.2f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ── Bespoke schedule + piecewise hazard ───────────────────────────────────
    with st.expander("Bespoke schedule + piecewise-constant hazard curve"):
        default_sched = pd.DataFrame(
            {"t": [1.0, 2.0, 3.0, 5.0], "hazard": [0.02, 0.03, 0.04, 0.05]}
        )
        edited = st.data_editor(
            default_sched,
            key="cds_piecewise",
            num_rows="dynamic",
            use_container_width=True,
        )
        accrual = st.checkbox("Include accrued-premium-at-default", value=True,
                              key="cds_accrual")
        try:
            ts = [float(x) for x in edited["t"].tolist() if pd.notna(x)]
            hs = [float(x) for x in edited["hazard"].tolist() if pd.notna(x)]
            if len(ts) == len(hs) and len(ts) > 0:
                spread = cds_spread_for_schedule(
                    payment_times=ts, hazards=hs, r=r_disc, R=R, accrual=accrual,
                )
                st.metric("Bespoke par spread", f"{spread * 1e4:,.2f} bps")
        except Exception as exc:
            st.error(f"Piecewise CDS failed: {exc}")


# ── Section B · CVA ───────────────────────────────────────────────────────────

def _render_cva_section(
    portfolio: Portfolio,
    prices: pd.DataFrame | None,
    risk_params: dict,
) -> None:
    st.markdown("### B · CVA")

    mode = st.radio(
        "Exposure source",
        options=["Use current portfolio MC", "Upload exposure CSV"],
        horizontal=True,
        key="cva_mode",
    )

    col1, col2 = st.columns(2)
    with col1:
        R = st.number_input(
            "Recovery R",
            min_value=0.0, max_value=1.0, value=0.40, step=0.05,
            key="cva_R",
        )
    with col2:
        lam_cva = st.number_input(
            "Counterparty hazard λ (flat)",
            min_value=0.0,
            max_value=2.0,
            value=0.03,
            step=0.005,
            format="%.4f",
            key="cva_lam",
            help="Used to derive marginal default probability in each bucket.",
        )

    # ── Exposure profile ───────────────────────────────────────────────────────
    exposure_df: pd.DataFrame | None = None

    if mode == "Use current portfolio MC":
        default_grid = "0.083, 0.25, 0.5, 0.75, 1.0"
        grid_str = st.text_input(
            "Horizon grid (years)",
            value=default_grid,
            key="cva_grid",
            help="Expected-positive-exposure is computed at each horizon.",
        )
        n_sims = st.number_input(
            "MC simulations per horizon",
            min_value=500,
            max_value=50_000,
            value=5_000,
            step=500,
            key="cva_nsims",
        )
        if st.button("Build EPE profile from portfolio", key="cva_build"):
            if prices is None:
                st.warning("Load market data first.")
            elif len(portfolio.stocks) + len(portfolio.options) == 0:
                st.warning("Portfolio is empty.")
            else:
                try:
                    horizons = [float(x.strip()) for x in grid_str.split(",") if x.strip()]
                    exposures = _simulate_epe(
                        portfolio=portfolio,
                        prices=prices,
                        horizons=horizons,
                        n_sims=int(n_sims),
                        lookback_days=risk_params["lookback_days"],
                        estimator=risk_params["estimator"],
                        ewma_N=risk_params["ewma_N"],
                    )
                    st.session_state["cva_exposure_df"] = pd.DataFrame(
                        {"t": horizons, "exposure": exposures}
                    )
                    st.success("EPE profile built.")
                except Exception as exc:
                    st.error(f"EPE build failed: {exc}")
        exposure_df = st.session_state.get("cva_exposure_df")

    else:  # CSV upload
        st.caption("CSV must have columns `t` (years) and `exposure` (dollars).")
        upload = st.file_uploader("Exposure CSV", type=["csv"], key="cva_csv")
        if upload is not None:
            try:
                df = pd.read_csv(upload)
                if not {"t", "exposure"}.issubset(df.columns):
                    st.error("CSV must contain columns 't' and 'exposure'.")
                else:
                    df = df[["t", "exposure"]].dropna().sort_values("t")
                    st.session_state["cva_exposure_df"] = df
                    exposure_df = df
                    st.success(f"Loaded {len(df)} exposure rows.")
            except Exception as exc:
                st.error(f"CSV parse error: {exc}")

    if exposure_df is None or exposure_df.empty:
        st.info("No exposure profile yet — build from portfolio MC or upload a CSV.")
        return

    st.markdown("**Exposure profile (EPE)**")
    st.dataframe(exposure_df, use_container_width=True, hide_index=True)
    st.line_chart(
        exposure_df.set_index("t")["exposure"],
        height=220,
    )

    # ── Marginal default probabilities from flat hazard ─────────────────────────
    ts = exposure_df["t"].to_numpy()
    s = np.exp(-lam_cva * ts)
    s_prev = np.concatenate(([1.0], s[:-1]))
    marginal = np.maximum(s_prev - s, 0.0)

    V0 = None
    if prices is not None and (len(portfolio.stocks) + len(portfolio.options)) > 0:
        try:
            V0 = float(portfolio_value(portfolio, prices.iloc[-1], date.today()))
        except Exception:
            V0 = None

    out = cva_summary(
        exposure_profile=exposure_df["exposure"].to_numpy(),
        marginal_default_probs=marginal,
        R=R,
        V0=V0,
    )

    c1, c2 = st.columns(2)
    c1.metric("CVA (USD)", f"{out['cva']:,.2f}")
    if "cva_pct" in out:
        c2.metric("CVA / V₀", f"{out['cva_pct']:.4%}")

    st.dataframe(
        pd.DataFrame(
            {
                "t": ts,
                "exposure": exposure_df["exposure"].to_numpy(),
                "marginal_PD": marginal,
                "CVA contribution": (1.0 - R) * exposure_df["exposure"].to_numpy() * marginal,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ── Internal: simulate expected-positive-exposure profile ─────────────────────

def _simulate_epe(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    horizons: list[float],
    n_sims: int,
    lookback_days: int,
    estimator: str,
    ewma_N: int,
) -> np.ndarray:
    """
    Simulate future portfolio values at each horizon via correlated GBM,
    returning EPE = E[max(V_T − V_0, 0)] at each horizon.
    """
    underlyings = sorted(
        {p.ticker for p in portfolio.stocks}
        | {p.underlying_ticker for p in portfolio.options}
    )
    underlyings = [u for u in underlyings if u in prices.columns]
    if not underlyings:
        raise ValueError("No portfolio tickers found in loaded prices.")

    pricing_date = date.today()
    spots0 = prices.iloc[-1]
    V0 = float(portfolio_value(portfolio, spots0, pricing_date))

    log_ret = compute_log_returns(prices[underlyings])
    mu_daily, cov_daily = get_mean_cov(log_ret, lookback_days, estimator, ewma_N)

    rng = np.random.default_rng(42)
    epe = np.zeros(len(horizons))

    for i, T in enumerate(horizons):
        h_days = max(1, int(round(T * 252.0)))
        mu_h = mu_daily.values * h_days
        cov_h = cov_daily.values * h_days
        R_sim = rng.multivariate_normal(mu_h, cov_h, size=n_sims)

        V_paths = np.empty(n_sims)
        spots0_arr = np.array([float(spots0[u]) for u in underlyings])
        for k in range(n_sims):
            shocked_arr = spots0_arr * np.exp(R_sim[k])
            full_shocked = spots0.copy()
            for u, v in zip(underlyings, shocked_arr):
                full_shocked[u] = v
            V_paths[k] = reprice_portfolio(portfolio, full_shocked, pricing_date)

        epe[i] = float(epe_profile_from_mc(V_paths, V0)[0])

    return epe


# ── Section C · Mitigated CVA ─────────────────────────────────────────────────

def _render_mitigated_cva_section(
    portfolio: Portfolio,
    prices: pd.DataFrame | None,
    risk_params: dict,
) -> None:
    """
    Show unmitigated vs mitigated CVA side-by-side.

    Mitigation:
        1. Netting — portfolio MTM is summed before CVA (already embedded in
           the EPE from Section B).
        2. CSA collateral — user inputs threshold and MTA; collateral is
           subtracted from exposure each period.

    Requires the EPE profile from Section B to be built first.
    """
    st.markdown("### C · Mitigated CVA (netting + CSA)")
    st.caption(
        "Applies CSA collateral and a netting agreement to the Section B "
        "exposure profile to show the CVA reduction from credit risk mitigation."
    )

    exposure_df = st.session_state.get("cva_exposure_df")
    if exposure_df is None or exposure_df.empty:
        st.info("Build or upload an exposure profile in Section B first.")
        return

    lam_cva = float(st.session_state.get("cva_lam", 0.03))
    R = float(st.session_state.get("cva_R", 0.40))

    col1, col2, col3 = st.columns(3)
    with col1:
        collateral = st.number_input(
            "Initial collateral ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
            key="mit_collateral",
            help="Posted collateral reduces gross exposure.",
        )
    with col2:
        threshold = st.number_input(
            "CSA threshold ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
            key="mit_threshold",
            help="Collateral is only called above this exposure level.",
        )
    with col3:
        mta = st.number_input(
            "Minimum transfer amount ($)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
            key="mit_mta",
            help="Minimum margin call size.",
        )

    ts = exposure_df["t"].to_numpy()
    exposures_raw = exposure_df["exposure"].to_numpy()

    # Marginal default probs from flat hazard
    s = np.exp(-lam_cva * ts)
    s_prev = np.concatenate(([1.0], s[:-1]))
    marginal = np.maximum(s_prev - s, 0.0)
    discount_factors = np.ones(len(ts))  # no discounting in base CVA; user can extend

    # Unmitigated CVA (no collateral)
    try:
        cva_unmitigated = float(
            cva_discounted(
                exposures=exposures_raw,
                marginal_default_probs=marginal,
                discount_factors=discount_factors,
                R=R,
            )
        )
    except Exception as exc:
        st.error(f"Unmitigated CVA failed: {exc}")
        return

    # Mitigated CVA (netting + CSA)
    try:
        # Build fake MTM path per bucket: each bucket has a single MTM = exposure
        # (already a netted EPE; CSA will further reduce via threshold/MTA logic)
        mtm_paths = [np.array([e]) for e in exposures_raw]
        cva_mit = float(
            mitigated_cva(
                mtm_paths=mtm_paths,
                marginal_default_probs=marginal,
                discount_factors=discount_factors,
                R=R,
                collateral=float(collateral),
                threshold=float(threshold),
                mta=float(mta),
            )
        )
    except Exception as exc:
        st.error(f"Mitigated CVA failed: {exc}")
        return

    reduction = cva_unmitigated - cva_mit
    reduction_pct = reduction / cva_unmitigated if cva_unmitigated > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Unmitigated CVA ($)", f"{cva_unmitigated:,.2f}")
    c2.metric("Mitigated CVA ($)", f"{cva_mit:,.2f}")
    c3.metric(
        "CVA reduction",
        f"{reduction:,.2f}",
        f"{reduction_pct:+.2%}",
    )

    # Per-period breakdown
    detail_rows = []
    for i, (t_val, exp_raw) in enumerate(zip(ts, exposures_raw)):
        # Compute per-period mitigated exposure: max(E - collateral, 0)
        exp_mit = max(exp_raw - float(collateral), 0.0)
        # Apply threshold/MTA approximately (simplified)
        if exp_raw - float(collateral) > float(threshold) + float(mta):
            exp_csa = float(threshold)
        else:
            exp_csa = exp_mit
        detail_rows.append(
            {
                "t (yr)": t_val,
                "EPE (unmitigated)": exp_raw,
                "EPE (after collateral)": exp_mit,
                "EPE (after CSA)": exp_csa,
                "Marginal PD": marginal[i],
                "CVA contrib (unmit)": (1 - R) * exp_raw * marginal[i],
                "CVA contrib (mit)": (1 - R) * exp_csa * marginal[i],
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    st.dataframe(
        detail_df.style.format(
            {
                "t (yr)": "{:.3f}",
                "EPE (unmitigated)": "{:,.2f}",
                "EPE (after collateral)": "{:,.2f}",
                "EPE (after CSA)": "{:,.2f}",
                "Marginal PD": "{:.6f}",
                "CVA contrib (unmit)": "{:,.4f}",
                "CVA contrib (mit)": "{:,.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
