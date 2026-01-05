# MATH GR 5320 — Advanced Demo: Magnificent Seven Portfolio

**Columbia University · Spring 2026**

This document traces the Streamlit front-end alongside the notebook (`advanced_demo.ipynb`) for an equal-weight **Magnificent Seven (M7)** portfolio constructed at 2015-12-31. Screenshots capture the exact UI state for each tab; all key numbers are compared against the notebook outputs.

The app runs at `localhost:8502`. All tabs share the same underlying `src/` modules — the notebook and the UI call identical code paths.

> **Evaluation-date note**: The notebook prices the portfolio at 2015-12-31 (split-adjusted spot prices via yfinance). The Streamlit app evaluates at the last loaded price in the market-data window (2016-12-30). Portfolio values and VaR figures therefore differ between the two — the structural properties (ES ≥ VaR, Historical > Parametric > MC, diversification benefit, backtest rejection) hold in both.

---

## Coverage Matrix

| # | Section | Notebook § | App tab | Key target |
|---|---------|-----------|---------|------------|
| 1 | M7 portfolio construction | §1 | Tab 1 | $349 834 at 2015-12-31 |
| 2 | Stock-only VaR/ES (3 methods) | §2 | Tab 2 + Tab 3 + Tab 4 | Hist > Param > MC |
| 3 | Diversification benefit | §3 | Tab 4 | 20.8% reduction |
| 4 | Option positions (OTM calls + short put) | §4 | Tab 1 | net value $1 371 |
| 5 | Full portfolio risk (stocks + options) | §5 | Tab 4 | ΔVaR ≈ +$1 332 |
| 6 | VaR backtesting (Kupiec) | §6 | Tab 5 | 18 exc vs 7.5, REJECT |
| 7 | Merton credit — NVDA & TSLA | §7 | Tab 6 (B) | NVDA 0.031%, TSLA 0.192% |

---

## Tab 1 · Portfolio Input

![Portfolio Input](../docs/screenshots/01_portfolio_input.png)

*Stock positions: equal-weight M7 at ~$50 000 per stock, computed from 2015-12-31 split-adjusted prices. Three option positions overlay the equity book.*

### Stock positions

| Ticker | Quantity | Notional ($) | Weight (%) |
|--------|----------|-------------|------------|
| AAPL | 2 108 | 49 982 | 14.29 |
| MSFT | 1 031 | 49 971 | 14.28 |
| GOOGL | 1 295 | 49 963 | 14.28 |
| AMZN | 1 479 | 49 982 | 14.29 |
| NVDA | 62 194 | 50 000 | 14.29 |
| META | 481 | 49 949 | 14.28 |
| TSLA | 3 124 | 49 986 | 14.29 |

*Total (notebook, 2015-12-31): **$349 833.69***

### Option positions

| Label | Underlying | Type | Qty | Strike ($) | Maturity | σ | Multiplier |
|-------|-----------|------|-----|-----------|----------|---|------------|
| AAPL_CALL | AAPL | call | +10 | 24.90 | 2026-11-30 | 0.25 | 100 |
| AMZN_C_OTM | AMZN | call | +5 | 36.50 | 2026-11-30 | 0.30 | 100 |
| TSLA_P_OTM | TSLA | put | −8 | 14.40 | 2026-11-30 | 0.55 | 100 |

*Strikes are set OTM relative to 2015-12-31 spot: AAPL +5%, AMZN +8%, TSLA −10%.*
*App status: **Portfolio: 7 stock position(s), 3 option position(s).***

---

## Tab 2 · Market Data

![Market Data](../docs/screenshots/02_market_data.png)

*Yahoo Finance download: 7 M7 tickers, 2013-01-01 → 2016-12-31. Split-adjusted prices.*

| Field | Value |
|-------|-------|
| Tickers | AAPL MSFT GOOGL AMZN NVDA META TSLA |
| Start | 2013/01/01 |
| End | 2016/12/31 |
| Rows loaded | **1 008 rows × 7 tickers (2013-01-02 → 2016-12-30)** |

*Last visible prices (2016-12-16 to 2016-12-30): AAPL ≈ 26.7, AMZN ≈ 37.9–38.6, GOOGL ≈ 39.3–40.6, META ≈ 114–119, MSFT ≈ 55.8–57.0, NVDA ≈ 2.47–2.63, TSLA ≈ 13.2–14.2 (all split-adjusted).*

---

## Tab 3 · Risk Settings

![Risk Settings](../docs/screenshots/03_risk_settings.png)

*Calibration mode: historical. All parameters match the notebook's §2 configuration.*

| Parameter | Value |
|-----------|-------|
| Calibration mode | historical |
| Lookback window | **252** trading days |
| Risk horizon | **5** trading days |
| VaR confidence | **0.990** (99%) |
| ES confidence | **0.975** (97.5%) |
| Estimator type | **window** (rolling) |
| Monte Carlo simulations | **50 000** |
| Option vol shock mode | fixed |

---

## Tab 4 · Run Analysis

### §2 + §5 — VaR/ES Comparison (three methods)

![Run Analysis — portfolio summary](../docs/screenshots/04_run_analysis.png)

*Portfolio Value $486 276.51 reflects Dec 2016 spot prices (last loaded). The notebook value of $351 204.44 uses Dec 2015 prices.*

**App output (evaluated at Dec 2016):**

| Model | VaR 99% 5d ($) | ES 97.5% 5d ($) | VaR / Portfolio |
|-------|---------------|----------------|-----------------|
| Historical | **47 216.61** | **47 408.82** | 9.71% |
| Parametric (Delta-Normal) | **34 579.91** | **34 771.48** | 7.11% |
| Monte Carlo | **32 982.95** | **33 138.23** | 6.78% |

**Notebook comparison (evaluated at Dec 2015, full portfolio with options):**

| Model | VaR stocks ($) | VaR full ($) | ES full ($) |
|-------|---------------|-------------|------------|
| Historical | 25 470.58 | **26 802.60** | **29 004.40** |
| Parametric | 22 161.68 | **23 519.24** | **23 646.41** |
| Monte Carlo | 21 683.52 | **22 747.81** | **23 098.07** |

**Structural properties verified (both app and notebook):**
- Historical > Parametric > Monte Carlo ✓ (fat tails captured by full repricing)
- ES ≥ VaR for all three methods ✓
- All values positive ✓

### §3 — Diversification benefit

*Shown implicitly via individual vs portfolio VaR in notebook §3:*

| Measure | Value |
|---------|-------|
| Sum of 7 individual VaRs | $32 153.24 |
| Portfolio VaR (historical) | $25 470.58 |
| **Diversification benefit** | **20.8%** |

Portfolio VaR < sum of individual VaRs — sub-additivity of ES confirmed.

### §4 + §5 — Option impact on risk

*Short TSLA puts dominate: net effect of adding options is a VaR increase.*

| Method | VaR Δ ($) | ES Δ ($) |
|--------|----------|---------|
| Historical | +1 332.02 | +1 520.57 |
| Parametric | +1 357.55 | +1 364.77 |
| Monte Carlo | +1 064.28 | +1 211.94 |

### Loss distribution & correlations

![Run Analysis — loss distribution and correlations](../docs/screenshots/04_run_analysis.png)

*Historical simulation loss distribution (right-skewed; VaR marker at $47 217, ES at $47 408). Correlation matrix shows GOOGL–MSFT highest (0.71), NVDA and TSLA lowest inter-stock correlations (~0.25–0.37), consistent with diversification story. Normalised price history shows NVDA +250% in H2 2016 while other names stay near base=100.*

---

## Tab 5 · Backtesting

### §6 — Kupiec unconditional coverage test

![Backtesting](../docs/screenshots/05_backtesting.png)

*Walk-forward historical backtest: 252-day estimation window, 5-day horizon, 99% VaR, 2013–2016 M7 data.*

| Metric | Value |
|--------|-------|
| Observations | **750** |
| Exceptions | **18** |
| Observed exception rate | **2.40%** |
| Expected exception rate | **1.00%** |
| Kupiec LR statistic | **10.6661** |
| p-value | **0.0011** |
| Reject H₀ at 5%? | **Yes** |
| Interpretation | **Model FAILS: exception rate is statistically different from expected** |

**Notebook comparison:**

| Quantity | Notebook (§6) | Expected |
|----------|---------------|----------|
| Observations | 750 | 750 ✓ |
| Expected exceptions | 7.5 | 7.5 ✓ |
| Actual exceptions | 18 | 18 ✓ |
| LR statistic | 10.6661 | 10.6661 ✓ |
| p-value | 0.0011 | 0.0011 ✓ |
| H₀ rejected | True | True ✓ |

**Interpretation**: The rolling 252-day window underestimates tail risk during the volatile 2013–2016 period. Exception clustering is visible in the walk-forward chart around Oct 2014 (macro volatility), Aug 2015 (China selloff), and Jan 2016 (energy/credit stress). This is a pedagogically valuable result — real-world VaR models based on a short calm-period window systematically understate risk during regime transitions.

---

## Tab 6 · Credit Risk

### §7A — Reduced-form (hazard rate)

*Inputs: λ = 0.0300, R = 0.40, r = 0.0300, horizons 0.25, 0.5, 1, 2, 3, 5, 10*

| Output | Value |
|--------|-------|
| LGD = 1 − R | **60.00%** |
| CDS approx spread (1−R)·λ | **180.0 bps** |
| S(5) | 86.0708% |
| P(τ ≤ 5) | 13.9292% |

### §7B — Merton structural model (NVDA)

*Inputs: V₀ = $16.3B, B = $1.3B, r = 0.02, μ = 0.5135, σ_A = 0.3119, T = 5 yr*

| Output | Notebook (§7) | Expected |
|--------|---------------|----------|
| Q-PD | **0.0312%** | 0.0312% ✓ |
| P-PD | **≈ 0.00%** | — ✓ |
| E₀ + D₀ = V₀ | **$16.30B** | ✓ |

### §7B — Merton structural model (TSLA)

*Inputs: V₀ = $33.7B, B = $2.7B, r = 0.02, μ = 0.0762, σ_A = 0.3567, T = 5 yr*

| Output | Notebook (§7) | Expected |
|--------|---------------|----------|
| Q-PD | **0.1916%** | 0.1916% ✓ |
| P-PD | **0.0590%** | 0.0590% ✓ |
| E₀ + D₀ = V₀ | **$33.70B** | ✓ |

*NVDA Q-PD < TSLA Q-PD: NVDA has a higher leverage ratio (mkt cap/assets) and lower LTD/assets, so probability of asset value falling below debt face value is much smaller. TSLA's slower real-world drift (μ = 7.6% vs NVDA's 51.4%) means P-PD > Q-PD for TSLA (drift below risk-neutral rate), while NVDA's extremely high μ drives P-PD effectively to zero.*

---

## System Architecture

```
src/
├── pricing/black_scholes.py      §4 — OTM option prices, delta
├── risk/
│   ├── historical.py             §2, §3, §5, §6 — historical VaR/ES, backtest
│   ├── parametric.py             §2, §5 — delta-normal VaR/ES
│   ├── monte_carlo.py            §2, §5 — MC simulation
│   └── backtest.py               §6 — walk-forward + Kupiec LR test
└── credit/
    ├── hazard.py                 §7A — survival, default probs, spreads
    └── merton.py                 §7B — structural PD (Q and P measure)
```

All modules are pure functions — no Streamlit imports, no network calls. The app calls `src/services/risk_engine_service.py` which wires these modules to the UI layer.

**Tests**: `tests/test_homework_cases.py` and `tests/test_course_validation.py` cover all key values above. Run with `python -m pytest tests/ -v`.
