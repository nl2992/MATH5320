# Combined Final Report
## MATH GR 5320 Portfolio Risk Management System

**Course:** MATH GR 5320 Financial Risk Management  
**Project title:** Portfolio Risk Management System  
**Submission package version:** Combined report assembled from segmented deliverables and relevant material from `FINAL_REPORT.md`  
**Repository:** `MATH5320`  
**Reference commit reviewed in this pass:** `5841589e3f3d2dbd3c1e38b08642eccce201a6a2`  

---

## Executive Summary

This repository implements an academic portfolio risk engine for MATH GR 5320. The system accepts portfolios of stocks and European options, loads historical market data from CSV files or Yahoo Finance, and computes Value at Risk and Expected Shortfall under three market-risk methodologies: historical simulation, parametric delta-normal, and Monte Carlo simulation. It also supports walk-forward VaR backtesting and includes a second layer of course-formula extension modules covering exact GBM/lognormal VaR and ES, reduced-form hazard models, the Merton structural default model, CDS pricing, CVA, counterparty mitigation, and illustrative regulatory capital and DFAST-style calculations.

The repo is organized as a modular Python and Streamlit application. The UI layer gathers inputs and renders outputs; service modules orchestrate end-to-end runs; portfolio and pricing modules provide valuation primitives; and risk, credit, and regulatory modules contain the underlying quantitative logic. This layered design is appropriate for a course risk engine because it supports testing, reuse in notebooks, and clearer separation between user interface code and model code.

Observed no-network and live integration evidence remains strong. The local suite passed `576` no-network tests, both live-data integration scripts passed, and the strict coverage run reported `91.22%` statement coverage across `src/`. That is meaningful validation evidence for coursework, but it does not satisfy the README target of `100%`.

For intended academic use, the system is suitable as a teaching-quality risk and validation framework. Its main limitations are also clear: Black-Scholes repricing uses a simplified option-volatility shock rather than a full implied-volatility surface; the parametric engine is first-order and therefore vulnerable to nonlinear option effects; the Monte Carlo engine uses multivariate normal return shocks; data quality depends on CSV inputs or Yahoo Finance; and several extension modules and UI branches remain below full coverage.

---

## Table of Contents

1. Requirement Coverage Matrix  
2. Introduction and Scope  
3. Model Risk Management Framework  
4. Application Screenshots and User Workflow  
5. Product / System Description  
6. Model Description  
7. Software Design and Implementation  
8. Validation Methodology, Test Plan, and Scope  
9. Validation Results  
10. Limitations and Model Risk  
11. Conclusions and Recommendations  
12. Bibliography / References  
13. Appendices  

---

## 1. Requirement Coverage Matrix

| Project requirement | Implementation | Test evidence | Where covered in this combined report |
|---|---|---|---|
| Portfolio of stocks and options as input | `src/schemas.py`, `src/ui/portfolio_editor.py` | `tests/test_backend.py`, `tests/test_ui_panels.py`, `tests/test_config_and_validation.py` | Sections 2, 4, 5 |
| Historical data and parameter inputs | `src/data/market_data.py`, `src/ui/market_data_panel.py`, `src/ui/risk_settings.py` | `tests/test_market_data.py`, `tests/test_config_and_validation.py` | Sections 4, 5, 7, 8 |
| Historical VaR | `src/risk/historical.py` | `tests/test_backend.py`, `tests/test_course_validation.py`, `tests/test_homework_cases.py` | Section 6 |
| Parametric VaR | `src/risk/parametric.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py` | Section 6 |
| Monte Carlo VaR | `src/risk/monte_carlo.py` | `tests/test_backend.py`, `tests/test_coverage_gaps.py` | Section 6 |
| Historical ES | `src/risk/historical.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py` | Section 6 |
| Parametric ES | `src/risk/parametric.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py` | Section 6 |
| Monte Carlo ES | `src/risk/monte_carlo.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py` | Section 6 |
| VaR backtesting | `src/risk/backtest.py`, `src/services/risk_engine_service.py` | `tests/test_backend.py`, `tests/test_backtest_extensions.py` | Sections 3, 6, 8, 9 |
| European option pricing | `src/pricing/black_scholes.py` | `tests/test_backend.py`, `tests/test_homework_cases.py`, `tests/test_strict_numerics.py` | Section 6 |
| Software design documentation | Layering across `app.py`, `src/`, `tests/`, and `notebooks/` | `tests/test_ui_panels.py`, backend/service tests | Section 7 |
| Test plan | `tests/` plus validation notebooks and artifacts | Local test suite and artifact bundle | Section 8 |
| Test results | Local pytest and coverage runs, artifact bundle, network integration checks | `submission/test_artifacts/` and `submission/coverage_report/` | Section 9 |
| Model documentation | Combined report plus segmented `submission/` reports | Cross-reference to all sections | Entire document |

---

## 2. Introduction and Scope

### 2.1 System Name

**MATH5320 Portfolio Risk Management System**

### 2.2 Business Purpose

The business purpose of the system is educational. It is a local academic risk-calculation platform for MATH GR 5320, intended to help students and analysts:

- value mixed portfolios of stocks and European options,
- compare multiple VaR and ES methodologies,
- validate formulas against course-derived fixtures,
- study model-risk governance through documentation, testing, and limitations.

### 2.3 Intended Users

- Students working through the local Streamlit interface.
- Instructors or markers reviewing the model and its validation evidence.
- Analysts or researchers importing directly from the Python modules.
- Notebook users reproducing course cases or testing assumptions interactively.

### 2.4 Intended Use

The intended uses are:

- constructing portfolios of stock and European option positions,
- loading aligned historical price data from CSV or Yahoo Finance wrappers,
- configuring lookback windows, horizons, VaR and ES confidence levels, estimators, and Monte Carlo path counts,
- computing and comparing historical, parametric, and Monte Carlo VaR and ES,
- running walk-forward VaR backtests and reviewing exception diagnostics,
- validating extension modules for lognormal VaR/ES, hazard, Merton, CDS, CVA, and regulatory calculations.

### 2.5 Non-Intended Use

The system is not intended for:

- production trading or production model governance,
- official regulatory filing or CCAR/DFAST submission,
- production XVA, enterprise-wide credit portfolio modeling, or market data governance,
- pricing or hedging American options, path-dependent options, or volatility-surface-sensitive exotics.

### 2.6 Scope

| Area | In scope | Out of scope |
|---|---|---|
| Instruments | Stocks, European calls, European puts | American options, path-dependent exotics |
| VaR methods | Historical simulation, parametric delta-normal, Monte Carlo | EVT, filtered historical simulation, copula VaR |
| ES methods | Historical, parametric, Monte Carlo, exact GBM extension | Full regulatory ES framework |
| Pricing | Black-Scholes with constant volatility | Local or stochastic volatility, early exercise |
| Credit | Hazard, Merton, CDS, CVA, mitigants | Production issuer portfolio credit model |
| Regulation | RWA, capital ratio, illustrative DFAST path | Official Fed production stress model |

---

## 3. Model Risk Management Framework

This section carries over the most useful model-risk-governance framing from the older `FINAL_REPORT.md` and aligns it with the current segmented deliverables and the current repository state.

### 3.1 Purpose, Scope, and Performance Requirements

| Item | Documentation |
|---|---|
| Purpose | Course-level risk engine for portfolios of stocks and European options |
| Scope | Historical, parametric, and Monte Carlo VaR/ES; walk-forward VaR backtesting; formula-sheet extensions for lognormal, hazard, Merton, CDS, CVA, counterparty mitigation, and illustrative capital/stress |
| Non-scope | Production trading, official regulatory reporting, production XVA, American or path-dependent option support |
| Performance requirement | Core formulas should match deterministic tests closely; current repo shows strong unit coverage through `576` passing tests, while remaining honest that coverage is `91.22%`, not `100%` |
| Data requirement | Aligned price histories, sufficient lookback, positive stock prices, and well-formed option inputs; proxy data sources must be documented |

### 3.2 Model Choice Justification

| Model choice | Why chosen | Alternative | Limitation |
|---|---|---|---|
| Historical simulation VaR | Transparent, nonparametric, based on realised scenarios | Filtered historical simulation, EVT | Sensitive to lookback and regime change |
| Parametric delta-normal VaR | Fast, standard baseline, easy to explain | Delta-gamma, Cornish-Fisher | Weak for nonlinear option portfolios and fat tails |
| Monte Carlo VaR | Allows full repricing under simulated shocks | Bootstrap MC, lattice methods | Depends on multivariate normal shocks and sample size |
| Black-Scholes | Standard closed-form European option pricer | Local vol, stochastic vol, binomial or LSMC | Constant vol, no smile/skew, no early exercise |
| Log-return shock convention | Keeps shocked stock prices positive and matches GBM framing | Arithmetic return shocks | Convexity and aggregation limitations |
| Window and EWMA estimators | Transparent and course-aligned | GARCH-family estimators | Window and decay sensitivity |

**GBM vs ABM note:** GBM is appropriate for non-negative equity prices, which matches the core stock/option scope of this project. Arithmetic Brownian Motion can be more appropriate for factors that may become negative, such as some rate or spread processes, but those are outside the scope of the required stock-and-European-option risk engine.

### 3.3 Data Validation and Proxy Assumptions

The older `FINAL_REPORT.md` framed this area well, but the current repo should be described carefully rather than optimistically.

| Data item | Current design expectation | Current implementation note |
|---|---|---|
| Price histories | Positive prices, parseable dates, aligned columns, adequate history | `src/data/validation.py` checks emptiness, `DatetimeIndex`, all-NaN columns, and non-positive prices; alignment and sufficiency are handled partly in loaders and risk loops |
| Return series | Stable enough to estimate mean/covariance | Return construction is explicit in `src/risk/returns.py`; no separate outlier-cleaning layer is implemented |
| Option inputs | Positive strike, positive volatility, well-defined maturity | Some constraints are enforced naturally in pricing functions; central schema validation is lighter than the older report text implied |
| Data proxy | Yahoo Finance adjusted close or user CSV | Appropriate for coursework, not institutional market-data governance |
| Cache behavior | Avoid repeated downloads and allow more reproducible live runs | Implemented through cached download wrappers in `src/data/market_data.py` |

### 3.4 Conceptual Soundness

| Conceptual soundness check | Evidence in this repo |
|---|---|
| Appropriate for intended task | Core requirement is a stock and European-option risk engine; the repo matches that directly |
| Mathematical specification documented | The segmented model documentation and code docstrings document formulas explicitly |
| Alternative approaches considered | Repo compares historical, parametric, and Monte Carlo methods and includes window versus EWMA estimation |
| Assumptions documented | Black-Scholes, log shocks, multivariate normal Monte Carlo, and covariance estimation conventions are documented |
| Sensitivity and diagnostics available | Lookback, horizon, estimator, confidence levels, and Monte Carlo path count are configurable; backtesting diagnostics are available in code |
| Limitations documented | Model and software limitations are explicit in the submission package |

### 3.5 Ongoing Monitoring and Post-Deployment

For coursework this is a recommended governance plan rather than an operational production process.

| Monitoring item | Suggested action |
|---|---|
| VaR exceptions | Re-run Kupiec and Christoffersen diagnostics periodically when data changes materially |
| Input drift | Track realized volatility, correlation, and outlier behavior across windows |
| Data quality | Detect missing prices, stale prices, and ticker mismatches before analysis |
| Code changes | Rerun the full local suite after all material changes |
| Parameter overrides | Record any manual overrides to volatility, confidence, lookback, or horizon |
| New instruments | Require scope review, pricing-model review, and new tests before adding them |

**Change management principle:** material changes to methodology, pricing, data sources, or risk definitions should trigger revalidation. The current repo already reflects this mindset in practice through its broad regression suite and separate documentation/testing deliverables.

### 3.6 Outcome Analysis and Backtesting

Outcome analysis is centered on VaR backtesting.

The codebase currently supports:

- Kupiec unconditional coverage,
- Christoffersen independence,
- conditional coverage,
- Basel traffic-light classification,
- exception severity diagnostics.

The Streamlit application’s main backtest tab emphasizes the walk-forward VaR path and Kupiec summary, while the deeper diagnostics are present in code and tests.

---

## 4. Application Screenshots and User Workflow

The older `FINAL_REPORT.md` already included a useful screenshot walkthrough, so the relevant pieces are carried over here as representative UI evidence. These figures should be read as a representative live session, not as canonical regression values for all future runs.

### 4.1 Representative Eight-Tab Workflow

| Tab | Name | Purpose |
|---|---|---|
| 1 | Portfolio Input | Enter stock and option positions |
| 2 | Market Data | Load CSV or download from Yahoo Finance |
| 3 | Risk Settings | Configure lookback, horizon, confidence, estimator, calibration mode, option-volatility treatment, and Monte Carlo paths |
| 4 | Run Analysis | Execute all three core market-risk methods |
| 5 | Backtesting | Run walk-forward backtest and review coverage diagnostics |
| 6 | Credit Risk | Hazard-rate and Merton summaries |
| 7 | CDS / CVA | CDS spreads, CVA, and mitigation helpers |
| 8 | Capital & Stress | RWA, capital ratio, DFAST-style scenarios |

### 4.2 Representative Screenshot Walkthrough

#### Portfolio Input

![Portfolio Input](../docs/screenshots/01_portfolio_input.png)

This panel captures stock and option positions and summarizes the current portfolio composition. It is the user’s entry point into the core stock-and-option risk workflow.

#### Market Data

![Market Data](../docs/screenshots/02_market_data.png)

This panel loads CSV data or downloads Yahoo Finance data. The screenshot illustrates the intended “load-then-validate” workflow used before risk calculations are run.

#### Risk Settings

![Risk Settings](../docs/screenshots/03_risk_settings.png)

This panel collects the main market-risk parameters: lookback, horizon, VaR confidence, ES confidence, estimator choice, calibration mode, manual market-risk inputs when selected, option-volatility shock mode, and Monte Carlo path count.

#### Run Analysis

![Run Analysis](../docs/screenshots/04_run_analysis.png)

This panel shows the combined comparison of historical, parametric, and Monte Carlo outputs, along with charts such as loss distributions and correlations.

#### Backtesting

![Backtesting](../docs/screenshots/05_backtesting.png)

This panel illustrates the walk-forward backtest view and the user-facing Kupiec summary. It shows how model performance is exposed visually rather than only numerically.

#### Credit Risk

![Credit Risk](../docs/screenshots/06_credit_risk.png)

This extension panel supports reduced-form hazard and Merton structural calculations. It is useful supplementary functionality but should still be labeled as extension scope relative to the required stock/option risk engine.

#### CDS / CVA

![CDS / CVA](../docs/screenshots/07_cds_cva.png)

This panel collects the CDS and CVA functionality in a single workflow and reflects how the repo expanded beyond the original market-risk scope.

#### Capital & Stress

![Capital & Stress](../docs/screenshots/08_capital_stress.png)

This panel illustrates the regulatory and stress-testing extension layer: RWA, capital ratio, and DFAST-style scenario helpers.

---

## 5. Product / System Description

### 5.1 User Workflow

The operational workflow of the required stock/option risk engine is:

1. Input stock and option positions.
2. Load historical data from CSV or Yahoo Finance.
3. Configure lookback, horizon, VaR confidence, ES confidence, estimator, calibration mode, manual inputs if used, option-volatility treatment, and Monte Carlo settings.
4. Run risk analysis and compare methods.
5. Review plots, correlations, and downloadable outputs.
6. Run backtesting and inspect exception diagnostics.

### 5.2 Input Schema

#### Stock input

| Field | Meaning |
|---|---|
| `ticker` | Equity symbol |
| `quantity` | Signed position size |

#### Option input

| Field | Meaning |
|---|---|
| `ticker` / label | Option identifier |
| `underlying_ticker` | Underlying stock symbol |
| `option_type` | `call` or `put` |
| `quantity` | Signed contract count |
| `strike` | Strike price |
| `maturity_date` | Expiry date |
| `volatility` | Annualized volatility input |
| `risk_free_rate` | Continuous risk-free rate |
| `dividend_yield` | Continuous dividend yield |
| `contract_multiplier` | Shares per contract |

#### Market data input

| Requirement | Meaning |
|---|---|
| Date index | Parseable and ordered dates |
| Price columns | One price series per underlying |
| Positive prices | No non-positive equity prices |
| Adequate history | Enough observations for the chosen lookback and backtest horizon |
| Alignment | Shared index after cleaning/alignment |

#### Risk settings

| Setting | Meaning |
|---|---|
| `lookback_days` | Estimation window size |
| `horizon_days` | Risk horizon |
| `var_confidence` | VaR confidence level |
| `es_confidence` | ES confidence level |
| `estimator` | `window` or `ewma` |
| `ewma_N` | EWMA parameter |
| `calibration_mode` | `historical` or `manual` |
| `manual_market_params` | Daily mean/covariance override bundle for parametric and Monte Carlo runs |
| `option_vol_shock_mode` | `fixed` or `underlying_beta` |
| `option_vol_shock_beta` | Simplified volatility shock sensitivity |
| `option_vol_shock_floor` | Lower bound on shocked volatility |
| `n_simulations` | Monte Carlo path count |
| `backtest model` | Historical, parametric, or Monte Carlo |

### 5.3 Inputs, Sources, and Validation Checks

| Input | Source | Used by | Current validation behavior |
|---|---|---|---|
| Price history | CSV or Yahoo Finance | All risk methods | Checks for empty frames, `DatetimeIndex`, all-NaN columns, and non-positive prices |
| Portfolio positions | Streamlit UI and dataclasses | Valuation and risk | Ticker existence checked against loaded price columns |
| Volatility | User input | Option repricing and parametric/MC assumptions | Domain errors surface through pricing behavior if invalid |
| Risk-free rate | User input or helper | Option pricing and some extension modules | Numeric input expected; live helper available for a Treasury proxy |
| Confidence levels | User input | VaR/ES and backtesting | Behavioral coverage exists in tests, especially for separate ES confidence |
| Manual market-risk parameters | User input in manual mode | Parametric and Monte Carlo engines | Missing underlyings, non-finite values, asymmetry, and non-PSD covariance are rejected |

### 5.4 Outputs

The core outputs are:

- VaR by method,
- ES by method,
- loss distributions,
- correlation and chart views,
- backtest exception statistics,
- JSON and CSV downloads.

The extension outputs add:

- hazard and survival tables,
- Merton PD/equity/debt summaries,
- CDS spread outputs,
- CVA and mitigated CVA summaries,
- RWA, capital ratio, and DFAST-style capital-path outputs.

---

## 6. Model Description

### 6.1 Portfolio Valuation and Loss Convention

Portfolio value is built from stock values plus repriced option values:

```text
V_t = sum_i q_i P_{i,t}
```

The repo uses:

```text
PnL  = V_T - V_0
Loss = V_0 - V_T
```

Positive loss means the portfolio lost value.

### 6.2 Stock Price Model and Shock Construction

Daily log returns are:

```text
r_t = log(S_t / S_{t-1})
```

Overlapping horizon returns are:

```text
R_t^(h) = sum_{k=0}^{h-1} r_{t-k}
```

The shocked stock-price convention is:

```text
S_shocked = S_0 * exp(R_h)
```

### 6.3 Option Pricing Model

European calls and puts are repriced with Black-Scholes under continuous dividends. The key inputs are spot, strike, maturity, volatility, risk-free rate, dividend yield, and option type.

```text
d1 = [log(S/K) + (r - q + 0.5 sigma^2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

```text
Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
```

```text
Delta_call = e^{-qT} N(d1)
Delta_put  = e^{-qT} (N(d1) - 1)
```

Important limitation: the core VaR engines reprice options with user-supplied volatility but do not dynamically shock the volatility surface.

### 6.4 Historical VaR and ES

Historical simulation:

1. computes log returns,
2. builds overlapping `h`-day scenarios,
3. shocks underlying prices,
4. fully reprices the portfolio,
5. builds an empirical loss distribution.

The implementation computes:

```text
VaR_alpha = empirical alpha-quantile of losses
ES_alpha  = average loss beyond the ES threshold
```

This is nonparametric, but it is also highly dependent on the selected history and the number of available tail scenarios.

### 6.5 Parametric VaR and ES

The parametric engine uses a delta-normal approximation:

```text
mu_h    = h * mu
Sigma_h = h * Sigma
```

```text
m   = x' mu_h
s^2 = x' Sigma_h x
```

```text
VaR = -m + s Phi^{-1}(alpha_var)
ES  = -m + s phi(z_es) / (1 - alpha_es)
```

The current code explicitly allows `es_confidence` to differ from `var_confidence`.

The current implementation uses corrected delta-dollar exposure for options:

```text
x_option = quantity × multiplier × BS_delta × spot
```

which aligns the implementation with the documented exposure convention.

### 6.6 Monte Carlo VaR and ES

The Monte Carlo engine estimates `mu` and `Sigma` from historical log returns or accepts them from the manual calibration path, applies horizon scaling, simulates:

```text
R_h ~ N(mu_h, Sigma_h)
```

and then:

1. shocks each underlying with `S_sim = S_0 * exp(R_sim)`,
2. fully reprices the portfolio,
3. constructs a simulated loss distribution,
4. computes empirical VaR and ES.

Observed design choices:

- default Monte Carlo path count is `10,000`,
- deterministic test paths can use a fixed seed,
- manual calibration can override the estimated daily mean/covariance,
- full repricing can use `fixed` or simplified `underlying_beta` option-volatility shock mode,
- backtest Monte Carlo runs cap simulations for speed.

### 6.7 Estimation Methods: Window and EWMA

The repo supports:

- rolling window mean/covariance,
- EWMA mean/covariance with:

```text
lambda = (N - 1) / (N + 1)
```

Window estimation is transparent; EWMA responds faster to recent volatility clustering.

### 6.8 Backtesting

Backtesting is walk-forward and out-of-sample:

1. take the prior lookback window,
2. estimate the model,
3. forecast VaR,
4. observe realized loss over the horizon,
5. record an exception when realized loss exceeds forecast VaR.

The codebase supports:

- Kupiec unconditional coverage,
- Christoffersen independence,
- conditional coverage,
- Basel traffic-light classification,
- exception severity summaries.

### 6.9 Formula-Sheet Extension Modules

These should be described as extension scope, not as a replacement for the core stock/option risk engine.

| Module | Purpose | Inputs | Outputs | Test evidence |
|---|---|---|---|---|
| `src/risk/lognormal.py` | Exact GBM/lognormal VaR and ES | `V0`, `mu`, `sigma`, `h`, `p` | Exact VaR/ES | `test_lognormal.py`, `test_course_validation.py` |
| `src/credit/hazard.py` | Reduced-form hazard and risky bond quantities | Hazard, recovery, time | Survival, PD, spread, risky ZCB | `test_credit.py`, `test_course_validation.py` |
| `src/credit/merton.py` | Structural default model | `V0`, `B`, `r`, `mu`, `sigma`, `T` | PD, equity value, debt value | `test_credit.py`, `test_course_validation.py` |
| `src/credit/cds.py` | CDS spread calculations | Hazard curve, recovery, discounting | CDS spread outputs | `test_credit.py`, `test_course_validation.py` |
| `src/credit/cva.py` | CVA and related exposure helpers | Exposure profile, marginal PD, recovery | CVA outputs | `test_credit.py`, `test_cva_mitigants.py` |
| `src/credit/mitigation.py` | Netting and collateral helpers | MTM/exposure inputs | Netted or collateralized exposures | `test_counterparty_mitigation.py`, `test_cva_mitigants.py` |
| `src/risk/regulatory.py` | RWA, capital ratio, stress helpers | Exposures, risk weights, stress assumptions | Capital and stress outputs | `test_regulatory.py`, `test_dfast_pathing.py`, `test_balance_sheet.py` |

---

## 7. Software Design and Implementation

### 7.1 Architecture Overview

```mermaid
flowchart TD
    U["User"] --> UI["Streamlit UI<br/>app.py<br/>src/ui/*"]
    UI --> SVC["Service Layer<br/>src/services/*"]
    SVC --> DOM["Domain Layer<br/>src/schemas.py<br/>src/portfolio/*"]
    DOM --> MOD["Model Layer<br/>src/pricing/*<br/>src/risk/*<br/>src/credit/*"]
    MOD --> OUT["Outputs<br/>VaR/ES tables<br/>loss distributions<br/>backtest results<br/>JSON/CSV downloads"]
```

The design is layered:

- the UI layer captures inputs and renders outputs,
- the service layer orchestrates end-to-end runs,
- the domain layer defines portfolio objects and valuation structures,
- the model layer contains pricing, returns, risk, credit, and regulatory logic.

### 7.2 Data Flow and Control Flow

```mermaid
flowchart TD
    A["Portfolio input"] --> B["Validation"]
    B --> C["Market data loading"]
    C --> D["Current valuation"]
    D --> E["Return and parameter estimation"]
    E --> F["Historical / Parametric / Monte Carlo"]
    F --> G["Aggregation and charts"]
    G --> H["Backtesting"]
```

### 7.3 Backtesting Control Flow

```mermaid
flowchart TD
    A["Historical data"] --> B["Rolling estimation window"]
    B --> C["VaR forecast"]
    C --> D["Realised loss"]
    D --> E["Exception sequence"]
    E --> F["Kupiec and diagnostics"]
```

### 7.4 Module Map

| Module | Role |
|---|---|
| `app.py` and `src/ui/*` | User interaction, displays, downloads |
| `src/services/risk_engine_service.py` | Core market-risk orchestration |
| `src/services/credit_service.py` | Credit and CVA orchestration |
| `src/services/regulatory_service.py` | RWA and DFAST orchestration |
| `src/schemas.py` | Portfolio dataclasses |
| `src/portfolio/*` | Valuation and exposure aggregation |
| `src/pricing/black_scholes.py` | European option price and delta |
| `src/risk/*` | Returns, estimation, market-risk methods, backtesting |
| `src/credit/*` | Hazard, Merton, CDS, CVA, mitigation |
| `tests/*` | Regression, validation, and UI/service tests |
| `notebooks/*` | Course walkthroughs and supplementary validation narratives |

### 7.5 Key Design Decisions

- Separate formulas from UI rendering.
- Keep orchestration in service modules rather than in Streamlit panels.
- Use reusable pricing and valuation helpers so all three risk engines share the same base valuation logic.
- Keep course extension modules distinct from the required core stock/option engine.
- Maintain a test-heavy repo structure so validation evidence is part of the design, not an afterthought.

### 7.6 Notebook Sequence

One useful section from `FINAL_REPORT.md` that was worth carrying over is the explicit notebook map:

| Notebook | Topic |
|---|---|
| `01_market_risk_var_es_goldens.ipynb` | Market-risk goldens and exact checks |
| `02_aapl_cat_var_es_methods.ipynb` | AAPL/CAT comparison across VaR/ES methods |
| `03_historical_shock_methodology.ipynb` | Historical-shock construction |
| `04_estimation_rolling_vs_ewma.ipynb` | Estimator comparison |
| `05_credit_hazard_risky_bond_spread.ipynb` | Hazard-rate and risky-bond extension |
| `06_credit_merton_structural_default.ipynb` | Merton structural extension |
| `07_cds_pricing_validation.ipynb` | CDS validation |
| `08_cva_counterparty_mitigation.ipynb` | CVA and mitigation |
| `09_regulatory_rwa_dfast_pathing.ipynb` | Capital and DFAST pathing |
| `10_backtesting_validation_dashboard.ipynb` | Backtest diagnostics |
| `11_end_to_end_demo.ipynb` | End-to-end demonstration |

---

## 8. Validation Methodology, Test Plan, and Scope

### 8.1 Validation Objectives

The validation program aims to establish:

- mathematical correctness of formulas,
- correctness of stock and option repricing,
- correctness of historical, parametric, and Monte Carlo VaR/ES outputs,
- correctness of backtesting and exception logic,
- correctness of credit and regulatory extension formulas,
- robustness to invalid inputs and edge cases,
- correct service-layer and UI integration,
- reproducibility through deterministic fixtures and stored artifacts.

### 8.2 Validation Categories

| Category | Purpose |
|---|---|
| Unit tests | Validate pure functions and deterministic model logic |
| Analytical goldens | Compare against closed-form reference values |
| Homework fixtures | Compare against course-derived expected values |
| Behavioral tests | Check monotonicity and financial reasonableness |
| Edge/failure tests | Ensure invalid inputs fail visibly or safely |
| Integration tests | Exercise end-to-end workflows |
| UI tests | Validate Streamlit panels and rendering paths |
| Coverage tests | Measure executed source coverage and identify gaps |

### 8.3 Module-Level Test Scope

| Module area | Main test files |
|---|---|
| Core backend and services | `tests/test_backend.py` |
| Backtest extensions | `tests/test_backtest_extensions.py` |
| Course validation sheet | `tests/test_course_validation.py` |
| Homework cases | `tests/test_homework_cases.py` |
| Credit and CVA | `tests/test_credit.py`, `tests/test_credit_service.py`, `tests/test_cva_mitigants.py`, `tests/test_counterparty_mitigation.py`, `tests/test_merton_timing.py` |
| Regulatory and balance sheet | `tests/test_regulatory.py`, `tests/test_dfast_pathing.py`, `tests/test_balance_sheet.py` |
| Market data and validation | `tests/test_market_data.py`, `tests/test_config_and_validation.py` |
| Numerical and coverage gaps | `tests/test_strict_numerics.py`, `tests/test_coverage_gaps.py`, `tests/test_es_confidence_split.py` |
| UI and charts | `tests/test_ui_panels.py`, `tests/test_charts.py` |
| Live integration | `tests/integration_test.py`, `tests/integration_test_formula_sheet.py` |

### 8.4 Tolerances and Numerical Standards

| Tolerance type | Typical use |
|---|---|
| Machine precision | Deterministic formulas in strict numerics |
| Tight relative tolerance | Homework and course fixtures |
| Structural assertions | Sign, monotonicity, or ordering logic |
| Monte Carlo tolerance | Sampling-based approximate acceptance |
| Live integration acceptance | Positive, finite, and operational workflow checks |

Important repo-specific note: the current code in `tests/test_course_validation.py` uses about `1%` relative tolerance for those fixtures, even though older README language mentions looser tolerance.

### 8.5 Coverage Plan and Known Untested Areas

The README advertises a `100%` statement coverage target across `src/`. The observed current state remains lower than that target, so the test plan should be read as an intended standard rather than an already-met standard.

Known lower-coverage or special-risk areas include:

- deeper CDS branches,
- hazard piecewise paths,
- some historical absolute-shock branches,
- `src/risk/normal.py`,
- some regulatory-service branches,
- selected UI extension paths.

---

## 9. Validation Results

### 9.1 No-Network Test Suite

Observed command:

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py -q
```

Observed result from the current walkthrough:

```text
576 passed, 242 warnings in 14.95s
```

### 9.2 Coverage Run

Observed command:

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:submission/coverage_report --cov-report=xml:submission/coverage_report/coverage.xml --cov-fail-under=100 --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

Observed result:

- all counted tests still passed,
- total statement coverage was `91.22%`,
- the strict coverage gate failed because the README target is `100%`.

Selected lower-coverage files in the current run included:

- `src/credit/cds.py`
- `src/credit/hazard.py`
- `src/risk/historical.py`
- `src/risk/normal.py`
- `src/risk/returns.py`
- `src/services/regulatory_service.py`
- `src/ui/capital_panel.py`
- `src/ui/cds_cva_panel.py`

### 9.3 Selected Analytical and Fixture Evidence

The repo includes strong deterministic and homework-style evidence through:

- `test_strict_numerics.py` for closed-form exactness,
- `test_lognormal.py` for GBM VaR/ES,
- `test_course_validation.py` for formula-sheet fixtures,
- `test_homework_cases.py` for course-derived regression values,
- `test_es_confidence_split.py` for separate VaR/ES confidence behavior.

### 9.4 Representative Backtesting Evidence

The codebase and artifact bundle demonstrate:

- walk-forward backtesting,
- exception counts,
- Kupiec unconditional coverage,
- Christoffersen independence,
- conditional coverage,
- Basel traffic-light status,
- exception severity summaries.

This is stronger than a minimal “just count exceptions” academic implementation.

### 9.5 Live Integration Status

The two live-data integration scripts now pass under the current repo behavior.

Current observed status:

- [tests/integration_test.py](../tests/integration_test.py) passed, including live download, service orchestration, all three market-risk methods, backtesting, EWMA mode, and multi-day horizon checks.
- [tests/integration_test_formula_sheet.py](../tests/integration_test_formula_sheet.py) passed, including live download, risk-free-rate fetch, stock/option portfolio risk run, backtesting, Merton, CDS, CVA, and regulatory helpers.

This means:

1. live-data download and orchestration work end to end,
2. the scripts are aligned with the current separate-confidence VaR/ES design.

---

## 10. Limitations and Model Risk

| Area | Limitation | Impact | Mitigation / interpretation |
|---|---|---|---|
| Historical VaR | Past may not represent future regimes | Under- or over-estimated risk | Compare windows and compare with MC/parametric |
| Parametric VaR | Delta-normal approximation | Weak for nonlinear option portfolios | Use as baseline, not sole authority |
| Parametric implementation | First-order exposure approach remains a linear approximation even after the corrected delta-dollar exposure fix | Nonlinear option effects can still be understated | Treat as a baseline method and compare with full repricing |
| Monte Carlo | Multivariate normal return shocks | Tail risk may be understated | Compare with historical method and larger path counts |
| Options | Simplified option-volatility shock rather than a full implied-vol surface | Smile/skew dynamics remain out of scope | Document clearly as scope limitation |
| Parameter-driven market-risk mode | Manual mean/covariance inputs are available only for parametric and Monte Carlo methods; historical simulation still needs price history by construction | Direct-input support is method-dependent | Document clearly as an inherent historical-simulation constraint |
| Covariance estimation | Sample instability and window dependence | Risk estimates may change materially | Provide window/EWMA comparison |
| Input validation | Central validation layer is lighter than some report wording might suggest | Documentation can overstate hard controls | Keep docs aligned to actual code behavior |
| Data source | Yahoo Finance and user CSVs are imperfect proxies | Stale or noisy prices can distort results | Use validation checks and disclose proxy status |
| Extensions | Credit and regulatory modules are simplified | Not production credit or supervisory engines | Label clearly as course-formula extensions |
| Coverage | Coverage is `91.22%`, not `100%` | Some branches are less tested | Extend tests before making stronger coverage claims |

---

## 11. Conclusions and Recommendations

### 11.1 Conclusion

The repository is acceptable for its intended course-project use. It successfully implements the required stock-and-European-option risk engine, supports multiple VaR and ES methodologies, and provides unusually strong testing and validation evidence for an academic project.

It should not be presented as a production risk platform or production regulatory engine. The most accurate framing is: a teaching-quality risk and validation framework that satisfies the core market-risk brief and extends it with broader course-formula modules.

### 11.2 Recommendations

1. Keep `submission/` as the official segmented deliverable package.
2. Use this combined report as the single “one-document” option for markers who prefer an integrated submission.
3. Align any remaining documentation language with the current code, especially around validation strictness and input checking.
4. Keep the corrected delta-dollar exposure convention explicit in the final write-up so the parametric method is documented accurately.
5. Extend the simplified option-volatility shock logic only if the project is pushed beyond current coursework scope.
6. Extend coverage in lower-tested extension and UI branches.

---

## 12. Bibliography / References

1. Columbia MATH GR 5320 lecture materials and project instructions.
2. Columbia MATH GR 5320 homework and course validation fixtures as encoded in `tests/test_course_validation.py` and `tests/test_homework_cases.py`.
3. Black, F., and Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*.
4. Kupiec, P. (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models*.
5. Christoffersen, P. (1998). *Evaluating Interval Forecasts*.
6. Stein, H. J. (2014). *Model Validation for Municipal Bonds*. Bloomberg Portfolio Risk Analytics. Local reference cited in `docs/references/`.

---

## 13. Appendices

### Appendix A. Core Formula Summary

```text
r_t = log(S_t / S_{t-1})
R_t^(h) = sum_{k=0}^{h-1} r_{t-k}
S_shocked = S_0 * exp(R_h)
PnL = V_T - V_0
Loss = V_0 - V_T
```

```text
VaR_parametric = -m + s Phi^{-1}(alpha_var)
ES_parametric  = -m + s phi(z_es) / (1 - alpha_es)
```

```text
d1 = [log(S/K) + (r - q + 0.5 sigma^2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

### Appendix B. Repository File Tree

```text
MATH5320/
├── app.py
├── src/
│   ├── schemas.py
│   ├── config.py
│   ├── data/
│   ├── pricing/
│   ├── portfolio/
│   ├── risk/
│   ├── credit/
│   ├── services/
│   └── ui/
├── tests/
├── notebooks/
├── docs/
│   ├── references/
│   └── screenshots/
└── submission/
```

### Appendix C. Reproducibility Commands

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:submission/coverage_report --cov-report=xml:submission/coverage_report/coverage.xml --cov-fail-under=100 --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
python tests/integration_test.py
python tests/integration_test_formula_sheet.py
```

### Appendix D. Submission Package Contents

- `00_combined_final_report.md`
- `01_model_documentation.md`
- `02_software_design_documentation.md`
- `03_test_plan.md`
- `04_test_results.md`
- `05_guide_gap_review.md`
- `06_prompt_coverage_matrix.md`
- `coverage_report/`
- `test_artifacts/`

### Appendix E. Checklist

| Item | Included in this combined report? |
|---|---|
| Purpose, intended use, and non-intended use | Yes |
| Requirement coverage matrix | Yes |
| Core model description | Yes |
| Software architecture and orchestration | Yes |
| Test-plan summary | Yes |
| Test-results summary | Yes |
| Screenshots and workflow | Yes |
| Limitations and recommendations | Yes |
| References and appendices | Yes |
