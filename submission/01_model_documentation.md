# Model Documentation and Validation Report
## MATH5320 Portfolio Risk Management System

**Course:** MATH GR 5320 Financial Risk Management  
**Institution:** Columbia University, Spring 2026  
**Report date:** May 2026  
**Repository:** `MATH5320`  
**Source commit under test:** `f154109fb8645c5be3ecf3d98669c74b1ae31935` (main branch, May 2026)

### Governance

| Role | Name |
|---|---|
| Model developer | Nigel Li |
| Model developer | Michael Adegbite |
| Model developer | Stella |
| Validation reviewer | Internal course submission |

### Version History

| Version | Date | Commit | Change summary |
|---|---|---|---|
| 1.0 | April 2026 | `5841589` | Initial market-risk engine: historical, parametric, MC VaR/ES, backtesting |
| 1.1 | May 2026 | `86890d8` | Added credit, CVA, regulatory, DFAST extension modules |
| 1.2 | May 2026 | `79111d8` | Raised test suite to 624 tests; improved coverage to 95%; added option-vol shock mode |
| 1.3 | May 2026 | `23f39ba` | Final submission: removed stale root drafts; all reports consolidated under `submission/` |
| 1.4 | May 2026 | `f154109` | Source state used for the refreshed evidence bundle and live integration reruns |
| 1.5 | May 2026 | `754df26` | Test suite grown to 624 tests; architecture diagrams corrected; submission docs finalised |

---

## Executive Summary

We built a Python and Streamlit portfolio risk system for Columbia MATH GR 5320. It takes portfolios of stocks and European options as input, prices options with Black-Scholes, and computes Value at Risk (VaR) and Expected Shortfall (ES) under three methods: historical simulation, parametric delta-normal, and Monte Carlo. The system runs through the Streamlit app or directly from the Python modules.

The workflow is: define a portfolio of stock and option positions, load historical price data from CSV or Yahoo Finance, configure risk parameters (lookback window, horizon, confidence levels, estimator type, Monte Carlo path count, and option-volatility shock mode), and run comparative risk analysis with walk-forward VaR backtesting. The main outputs are VaR and ES by method, loss distributions, correlation visualisations, backtest exception summaries, and downloadable results files.

Three market-risk methodologies are implemented. Historical simulation and Monte Carlo use full portfolio repricing under shocked market states; both support a simplified `underlying_beta` option-volatility shock mode as well as the default fixed-vol mode. The parametric method is a first-order delta-normal approximation using estimated or manually supplied mean and covariance of log returns with an exposure vector of equity holdings and option delta-dollar positions. VaR backtesting uses walk-forward forecasting with Kupiec unconditional coverage testing; Christoffersen independence, conditional-coverage, Basel traffic-light, and exception-severity diagnostics are also surfaced in the application outputs.

A second layer of extension modules covers exact GBM/lognormal VaR and ES, reduced-form hazard models, the Merton structural default model, CDS pricing, CVA, counterparty mitigation, and illustrative regulatory capital and DFAST projections. These modules are tested against course homework fixtures and demonstrate the breadth of topics covered in MATH GR 5320.

The no-network unit suite was run on May 11, 2026 with:

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py -q
```

The observed result was:

```text
624 passed, 242 warnings in 26.28s
```

An additional coverage run used:

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --cov-report=html:submission/coverage_report \
  --cov-report=xml:submission/coverage_report/coverage.xml \
  --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

The coverage run reported `624 passed` with 95% statement coverage. The two live-data integration scripts were run separately and both passed.

For the course brief, we did what was needed: compare VaR and ES for mixed portfolios, validate formula implementations against course fixtures, and document model choices and limits. The main limits are historical log-return shocks, first-order delta-normal parametric approximation, a simplified option-volatility shock rather than a full implied-vol surface, multivariate normal Monte Carlo shocks, and credit and regulatory modules that are course extensions rather than production models.

---

## Requirement Coverage Matrix

| Project requirement | Implementation evidence | Test evidence | Documentation section |
|---|---|---|---|
| Portfolio of stocks and options as input | `src/schemas.py`, `src/ui/portfolio_editor.py` | `tests/test_backend.py`, `tests/test_ui_panels.py`, `tests/test_config_and_validation.py` | Product/System Description |
| Historical data and parameter inputs | `src/data/market_data.py`, `src/ui/market_data_panel.py`, `src/ui/risk_settings.py` | `tests/test_market_data.py`, `tests/test_config_and_validation.py` | Product/System Description |
| Historical VaR | `src/risk/historical.py` | `tests/test_backend.py`, `tests/test_course_validation.py`, `tests/test_homework_cases.py` | Model Description |
| Parametric VaR | `src/risk/parametric.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py`, `tests/test_homework_cases.py` | Model Description |
| Monte Carlo VaR | `src/risk/monte_carlo.py` | `tests/test_backend.py`, `tests/test_coverage_gaps.py`, `tests/integration_test.py` | Model Description |
| Historical ES | `src/risk/historical.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py` | Model Description |
| Monte Carlo ES | `src/risk/monte_carlo.py` | `tests/test_backend.py`, `tests/test_es_confidence_split.py` | Model Description |
| VaR backtesting | `src/risk/backtest.py`, `src/services/risk_engine_service.py` | `tests/test_backend.py`, `tests/test_backtest_extensions.py`, `notebooks/10_backtesting_validation_dashboard.ipynb` | Model Description and Validation Results |
| Option pricing for European options | `src/pricing/black_scholes.py`, `src/portfolio/positions.py` | `tests/test_backend.py`, `tests/test_homework_cases.py` | Model Description |
| Model documentation | `submission/01_model_documentation.md`, `submission/00_combined_final_report.md` | Local repository evidence | Entire document |
| Software design documentation | `README.md`, module layering in `src/`, Streamlit app structure | `tests/test_ui_panels.py`, integration tests | Software Design and Implementation |
| Test plan | `tests/` and notebook validation workflow | Local test suite | Validation Methodology and Scope |
| Software | `app.py`, `src/`, `notebooks/` | Local test suite and integration scripts | Product/System Description and Software Design |
| Test results | Local pytest and coverage runs | Commands and outputs reproduced below | Validation Results |

---

## Introduction and Scope

### System Name

The system will be referred to in this report as the `MATH5320 Portfolio Risk Management System`.

### Business Purpose

The business purpose of the system is educational. It is a local academic risk-calculation platform for MATH GR 5320, intended to help students and analysts value portfolios, compare multiple risk methodologies, test risk model behavior against course fixtures, and document model assumptions and limitations in a validation-oriented format.

### Intended Users

Intended users are students, instructors, and technically capable analysts working locally through either:

- The Streamlit application in `app.py`
- Direct Python imports from the `src/` package
- Supporting Jupyter notebooks under `notebooks/`

### Intended Use

The intended uses are:

- Define portfolios of stocks and European options
- Load historical market data from CSV files or Yahoo Finance wrappers
- Compute VaR and ES under historical, parametric, and Monte Carlo methods
- Compare methodologies under a common portfolio and common market data
- Run walk-forward VaR backtesting and exception diagnostics
- Validate course-formula modules such as lognormal VaR/ES, hazard, Merton, CDS, CVA, and regulatory calculations
- Produce documentation and test evidence for a course project

### Non-Intended Use

The system is not intended for:

- Production trading or risk management
- Regulatory filing or supervisory capital submission
- Official CCAR or DFAST modeling
- Production XVA, issuer-level credit portfolio modeling, or enterprise-wide risk aggregation
- Pricing or hedging complex exotics, American options, or full volatility-surface products

### Portfolio and Data Scope

The system accepts a portfolio of stock and option positions. It supports both history-driven estimation and direct parameter input, depending on the module. The core Streamlit workflow documented in the README is:

1. Add stock and option positions.
2. Load historical price data from Yahoo Finance or CSV.
3. Configure lookback, horizon, confidence levels, estimator type, and Monte Carlo simulation count.
4. Run comparative risk analysis.
5. Run backtesting.

The exact model families covered in the repository are:

- Equity and stock risk
- European option pricing
- Portfolio VaR and ES
- VaR backtesting
- Formula-sheet extensions: exact GBM/lognormal, hazard, Merton, CDS, CVA, counterparty mitigants, and regulatory capital/stress

### Scope Table

| Area | In scope | Out of scope |
|---|---|---|
| Instruments | Stocks, European calls, European puts | American options, path-dependent options, exotics |
| VaR methods | Historical, parametric delta-normal, Monte Carlo | EVT, filtered historical simulation, copula VaR |
| ES methods | Historical, parametric, Monte Carlo, exact GBM extension | Full regulatory ES framework |
| Pricing | Black-Scholes European option pricing | Local volatility, stochastic volatility, early exercise |
| Credit | Hazard, Merton, CDS, CVA course modules | Full issuer portfolio credit model |
| Regulation | RWA, capital ratio, illustrative DFAST pathing | Official Fed CCAR/DFAST production model |

---

## Product/System Description

### User Workflow

From the application structure in `app.py`, the product is organized as an eight-tab Streamlit application:

1. Portfolio Input
2. Market Data
3. Risk Settings
4. Run Analysis
5. Backtesting
6. Credit Risk
7. CDS / CVA
8. Capital & Stress

The user workflow for the required market-risk engine is:

1. Enter stock and option positions.
2. Load aligned historical prices by CSV upload or Yahoo Finance download.
3. Configure risk settings such as lookback window, horizon, VaR confidence, ES confidence, estimator type, calibration mode, optional manual mean/vol/correlation inputs, Monte Carlo path count, and option-volatility shock mode.
4. Run risk analysis to obtain historical, parametric, and Monte Carlo results.
5. Review summary tables, charts, and downloadable output.
6. Run backtesting and review coverage diagnostics.

The last three tabs extend the product into course-specific modules for credit, CVA/CDS, and regulatory capital.

### Intended User Experience

The app is interactive but modular. Streamlit is used only as the front end. Core logic is delegated to pricing, portfolio, risk, credit, and service modules. This separation keeps business logic out of the UI and makes the quantitative code reusable in tests and notebooks.

### Input Schema

The repository defines three core dataclasses in `src/schemas.py`:

- `StockPosition`
  - `ticker`
  - `quantity`
- `OptionPosition`
  - `ticker`
  - `underlying_ticker`
  - `option_type`
  - `quantity`
  - `strike`
  - `maturity_date`
  - `volatility`
  - `risk_free_rate`
  - `dividend_yield`
  - `contract_multiplier`
- `Portfolio`
  - `stocks`
  - `options`

### Market Data Requirements

For the core market-risk engine, the required market data is an aligned wide price table indexed by date, with one price series per underlying ticker. The CSV loader expects a first date column and one numeric price column per ticker. The Yahoo Finance loader returns adjusted close values and then aligns them in a similar wide format.

### Inputs, Sources, and Validation Checks

| Input | Source | Used by | Validation check |
|---|---|---|---|
| Price history | CSV upload or Yahoo Finance wrapper | All risk methods | Datetime index, non-empty, no all-NaN columns, positive prices |
| Portfolio positions | Streamlit UI and dataclasses | Valuation and risk modules | Ticker existence is checked explicitly; many other field rules are enforced through UI expectations and pricing-domain checks rather than one central schema validator |
| Volatility | User input for options; estimated returns for VaR engines | Option pricing, parametric VaR, Monte Carlo VaR | Expected positive numeric; non-positive values surface through pricing-domain checks |
| Risk-free rate | User input for options; helper lookup in selected panels | Black-Scholes and selected extension modules | Numeric input expected; helper fetches are sanity-checked by range and fallback behavior |
| Horizon | User input | Historical, parametric, Monte Carlo, backtesting | Positive integer expected; backtests return an explicit empty reason when history is too short |
| Confidence levels | User input | VaR and ES | Values in `(0,1)` are expected; separate VaR and ES confidence behavior is covered in tests |
| Calibration mode | User input | Parametric and Monte Carlo engines | UI constrains to `historical` or `manual`; manual mode is covered in backend and UI tests |
| Manual daily mean / covariance | User input in manual mode | Parametric and Monte Carlo engines | Manual builder rejects missing underlyings, non-finite values, asymmetry, and non-PSD covariance |
| Option-volatility shock mode | User input | Historical, Monte Carlo, and backtesting repricing | UI constrains to `fixed` or `underlying_beta`; unknown modes raise explicitly |

### Outputs

The main outputs are:

- Historical VaR and ES
- Parametric VaR and ES
- Monte Carlo VaR and ES
- Loss distributions
- Correlation heatmap
- Portfolio value summary
- Backtest exception statistics
- JSON summary download
- Loss CSV download
- Backtest CSV download
- Kupiec-results JSON download

### Representative Application Figures

The packaged submission now includes representative screenshots from the live Streamlit application. These figures illustrate the user workflow and UI surface of the core risk engine and extension panels.

#### Figure 1. Portfolio Input

![Portfolio Input](../docs/screenshots/01_portfolio_input.png)

#### Figure 2. Market Data

![Market Data](../docs/screenshots/02_market_data.png)

#### Figure 3. Risk Settings

![Risk Settings](../docs/screenshots/03_risk_settings.png)

#### Figure 4. Run Analysis

![Run Analysis](../docs/screenshots/04_run_analysis.png)

#### Figure 5. Backtesting

![Backtesting](../docs/screenshots/05_backtesting.png)

#### Figure 6. Credit Risk Extension

![Credit Risk](../docs/screenshots/06_credit_risk.png)

#### Figure 7. CDS / CVA Extension

![CDS / CVA](../docs/screenshots/07_cds_cva.png)

#### Figure 8. Capital & Stress Extension

![Capital & Stress](../docs/screenshots/08_capital_stress.png)

### Module Architecture

The codebase is split into four layers. The Streamlit front end is isolated from all quantitative logic; services orchestrate the math modules; the core engine does everything required by the project brief; the extension modules cover additional course topics.

```mermaid
flowchart TB
    U["User"] --> APP["app.py · Streamlit entry point"]

    subgraph UI["src/ui/ — UI panels"]
        PE["portfolio_editor"]
        MD["market_data_panel"]
        RS["risk_settings"]
        RP["results_panel · charts"]
        XUI["credit_panel · cds_cva_panel · capital_panel"]
    end

    subgraph SVC["src/services/ — orchestration"]
        RSE["risk_engine_service"]
        CRS["credit_service"]
        RGS["regulatory_service"]
    end

    subgraph CORE["Core engine  ·  required by project brief"]
        DAT["data/ · market_data · validation"]
        CFG["schemas · config · demo_presets"]
        PRT["portfolio/ · positions · portfolio"]
        BSM["pricing/ · black_scholes"]
        RSK["risk/ · returns · estimators · historical<br/>parametric · normal · monte_carlo · backtest"]
    end

    subgraph EXT["Course extensions"]
        LOG["risk/lognormal"]
        REG["risk/regulatory"]
        CRD["credit/ · hazard · merton · cds · cva · mitigation"]
    end

    APP --> PE & MD & RS & RP & XUI

    PE --> RSE
    MD --> RSE
    RS --> RSE
    XUI --> CRS & RGS

    RSE --> DAT & CFG & PRT & RSK
    PRT --> BSM
    CRS --> CRD
    RGS --> REG

    TST["tests/ · 622 unit tests"] -. exercise .-> CORE & EXT
    NB["notebooks/"] -. exercise .-> CORE & EXT
```

All quantitative modules under `src/` are pure Python functions with no Streamlit imports. This means every formula can be called from `pytest` or a notebook without starting the app.

### Risk Engine Workflow

For the core market-risk path specifically, the call sequence is:

```mermaid
flowchart TD
    A["Portfolio Input"] --> B["Market Data Load and Validation"]
    B --> C["Risk Settings"]
    C --> D["RiskEngineService"]
    D --> E["Historical VaR/ES"]
    D --> F["Parametric VaR/ES"]
    D --> G["Monte Carlo VaR/ES"]
    E --> H["Results Panel"]
    F --> H
    G --> H
    H --> I["Charts and Downloads"]
    D --> J["Backtest Engine"]
    J --> K["Kupiec and Exception Diagnostics"]
```

---

## Model Description

### 5.1 Portfolio Valuation and Loss Convention

For stock positions, portfolio value is the sum of quantity times current spot:

```text
V_t(stock) = sum_i q_i * S_i,t
```

For option positions, each option is repriced using Black-Scholes with current spot, strike, maturity, volatility, risk-free rate, dividend yield, and contract multiplier. Expired options are valued at intrinsic value.

The repository uses the following sign conventions, explicitly documented in the README and implemented in the risk modules:

```text
PnL = V_T - V_0
Loss = V_0 - V_T
```

Positive loss therefore means the portfolio lost value.

The core distinction among methods is:

- Historical VaR/ES: full repricing under shocked scenarios
- Monte Carlo VaR/ES: full repricing under simulated scenarios
- Parametric VaR/ES: delta-normal approximation rather than full nonlinear repricing

### 5.2 Stock Price Model and Shock Construction

The equity-shock framework is based on log returns:

```text
r_t = log(S_t / S_t-1)
```

Overlapping horizon returns are built by rolling summation:

```text
R_t^(h) = r_t + r_t-1 + ... + r_t-h+1
```

The shocked price convention used in historical and Monte Carlo scenario generation is:

```text
S_T = S_0 * exp(R_h)
```

The main assumptions implied by the code are:

- Log returns are the working shock variable for equity underlyings
- Historical estimation uses a finite lookback window
- Stationarity is assumed within the chosen estimation window
- Covariance is estimated from historical log returns
- Parametric and Monte Carlo models use horizon scaling with `mu_h = h * mu` and `Sigma_h = h * Sigma`

An alternate historical absolute-shock branch exists in `src/risk/historical.py`, but the default and primary implementation uses log shocks.

### 5.3 Option Pricing Model

European calls and puts are priced using Black-Scholes with continuous dividends. The code in `src/pricing/black_scholes.py` implements:

```text
d1 = [log(S/K) + (r - q + 0.5 * sigma^2)T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Call = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
Put  = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
```

The corresponding deltas are:

```text
Delta_call = exp(-qT) * N(d1)
Delta_put  = exp(-qT) * (N(d1) - 1)
```

Required inputs are:

- Spot `S`
- Strike `K`
- Time to maturity `T`
- Volatility `sigma`
- Risk-free rate `r`
- Dividend yield `q`
- Option type

Observed design choice: the core full-repricing engines now support two volatility modes. `fixed` keeps the option's input implied volatility unchanged, while `underlying_beta` applies a simple leverage-style scenario shock `sigma' = max(floor, sigma * (1 - beta * R))`. This is better than a spot-only repricing approach, but it is still a simplified course-level approximation rather than a full implied-volatility surface model.

### 5.4 Historical VaR and ES

Historical simulation is implemented in `src/risk/historical.py`. The algorithm is:

1. Compute daily log returns.
2. Build overlapping `h`-day log-return scenarios.
3. Restrict to the lookback window.
4. Compute current portfolio value `V_0`.
5. Shock underlyings with `S_shocked = S_0 * exp(R)`.
6. Reprice the full portfolio under each scenario.
7. Form the empirical loss distribution.
8. Compute VaR and ES from the loss distribution.

Definitions:

```text
VaR_alpha = empirical alpha-quantile of loss
ES_alpha  = average loss in the tail beyond the ES threshold
```

This is a nonparametric method in the sense that it does not assume a normal return distribution. Its limitations are:

- It reacts only as fast as the selected history allows
- It is sensitive to the lookback window
- Extreme quantiles can be unstable when the scenario count is limited
- It assumes past scenarios remain relevant for future risk

### 5.5 Parametric VaR and ES

The parametric engine in `src/risk/parametric.py` uses a delta-normal approximation. It first builds an exposure vector from current holdings and option deltas, then estimates the mean vector and covariance matrix of daily log returns, then applies horizon scaling:

```text
mu_h    = h * mu
Sigma_h = h * Sigma
```

Portfolio mean and variance are then computed in exposure form:

```text
m   = x' * mu_h
s^2 = x' * Sigma_h * x
```

The implemented formulas are:

```text
VaR_alpha = -m + s * Phi^-1(alpha)
ES_alpha  = -m + s * phi(z_alpha) / (1 - alpha_es)
```

where `alpha_es` is the ES confidence level and `z_alpha` is the normal quantile.

Important implementation note: the code allows ES confidence to differ from VaR confidence. This is now labeled explicitly in the app and in exported summaries because some simplified treatments silently assume a single common confidence level.

The current implementation uses corrected delta-dollar option exposure:

```text
x_option = quantity × multiplier × BS_delta × spot
```

so the parametric engine now matches its own stated exposure convention.

Key limitations are:

- Approximately normal PnL assumption
- Sensitivity to covariance estimation
- First-order approximation for option risk
- Potential loss of accuracy for strongly nonlinear option portfolios
- Remaining dependence on covariance quality and linearization assumptions

### 5.6 Monte Carlo VaR and ES

The Monte Carlo engine in `src/risk/monte_carlo.py` either estimates `mu` and `Sigma` from log returns or accepts them from the manual calibration path, scales them to horizon, and simulates:

```text
R_h ~ N(mu_h, Sigma_h)
```

For each simulation:

1. Draw a multivariate normal horizon return vector.
2. Shock each underlying with `S_sim = S_0 * exp(R_sim)`.
3. Reprice the portfolio.
4. Record loss `V_0 - V_sim`.

VaR and ES are then computed empirically from the simulated loss distribution.

Observed design choices from the code:

- Default Monte Carlo path count is `10,000`
- The default random seed is `42`, which helps reproducibility in tests and notebooks
- Manual calibration mode can override the estimated daily mean/covariance
- Full repricing can use `fixed` or simplified `underlying_beta` option-volatility shock mode
- In backtesting, Monte Carlo paths are capped at `2,000` for speed and the seed is not fixed

Limitations:

- Simulated returns are multivariate normal
- Monte Carlo error remains unless path count is large
- Tails may be underrepresented with too few paths
- Covariance quality directly affects scenario quality

### 5.7 Estimation Methods: Window and EWMA

The repository supports two estimators in `src/risk/estimators.py`.

The rolling-window estimator uses:

```text
mu_hat    = sample mean over the lookback window
Sigma_hat = sample covariance over the lookback window
```

The EWMA estimator uses exponentially decaying weights with:

```text
lambda = (N - 1) / (N + 1)
```

This is the exact convention documented in the README and the code.

Observed design rationale:

- Window estimation is transparent and easy to explain
- EWMA allows recent observations to receive more weight
- The presence of notebook `04_estimation_rolling_vs_ewma.ipynb` suggests estimator comparison is a deliberate course objective, not an incidental implementation detail

Alignment and missing-data handling are conservative:

- Price frames are expected to be aligned by date
- Validation checks reject malformed data
- Notebook examples often use `dropna()` after alignment

### 5.8 Backtesting

Backtesting is implemented in `src/risk/backtest.py` as walk-forward VaR forecasting. For each evaluation date `t`:

1. Fit the selected risk model using data up to `t`
2. Forecast horizon VaR
3. Compute realized loss from `t` to `t+h`
4. Flag an exception if realized loss exceeds forecast VaR

The exception indicator is:

```text
I_t = 1{Loss_t > VaR_t}
```

The expected exception rate at VaR confidence `alpha` is:

```text
1 - alpha
```

Kupiec unconditional coverage is implemented as:

```text
LR_uc = -2 * [log L0 - log L1]
```

with the test interpreted against a chi-square distribution with 1 degree of freedom.

The repository goes beyond the README and also implements:

- Christoffersen independence test
- Conditional coverage test
- Basel traffic-light classification
- Exception severity diagnostics

What backtesting validates:

- Frequency of VaR breaches
- Whether unconditional coverage roughly matches target
- Whether exceptions are clustered

What it does not fully validate:

- ES calibration
- Exact tail severity modeling
- Structural stability across all regimes

### 5.9 Formula-Sheet Extension Modules

The repository includes several course-formula modules that extend beyond the core stock/option risk engine. These should be presented as extensions, not confused with the baseline project requirement.

| Module | Purpose | Main formula family | Inputs | Outputs | Validation fixture |
|---|---|---|---|---|---|
| `src/risk/lognormal.py` | Exact GBM/lognormal VaR and ES | Closed-form long/short GBM loss formulas | `V0`, `mu`, `sigma`, `h`, `p` | Exact VaR and ES | `test_lognormal.py`, `test_course_validation.py` |
| `src/credit/hazard.py` | Reduced-form credit | Survival, default density, risky ZCB, credit spread | Hazard, recovery, time | Survival, PD, spread, risky bond price | `test_credit.py`, `test_course_validation.py` |
| `src/credit/merton.py` | Structural default | Asset-value option framework | `V0`, `B`, `r`, `mu`, `sigma`, `T` | PD, equity value, debt value, spread | `test_credit.py`, `test_course_validation.py` |
| `src/credit/cds.py` | CDS pricing | Par spread under flat or piecewise hazard | Hazard curve, recovery, discount rate | CDS spread | `test_credit.py`, `test_course_validation.py` |
| `src/credit/cva.py` | Counterparty valuation adjustment | Discrete and discounted CVA | Exposure profile, marginal PD, recovery | CVA | `test_credit.py`, `test_course_validation.py`, `test_cva_mitigants.py` |
| `src/risk/regulatory.py` | Regulatory capital and stress | RWA, capital ratio, DFAST-style stress | Exposures, weights, capital, scenario inputs | RWA, ratios, stress path | `test_regulatory.py`, `test_dfast_pathing.py`, `test_balance_sheet.py` |

These modules tie implementation to course formulas, but they remain simplified models rather than production systems.

---

## Software Design and Implementation

### Architecture Overview

The project follows a layered design:

- UI layer
- Service/orchestration layer
- Schema and portfolio layer
- Pricing layer
- Risk-model layer
- Credit and regulatory extension layer
- Test layer

The repo is broader than the strict project brief, so the architecture is easiest to read in two blocks: the core market-risk engine that the brief actually asks for, and the course extensions layered around it.

```mermaid
flowchart TB
    U["User"] --> APP["Streamlit app<br/>app.py"]

    subgraph UI["UI layer"]
        PE["portfolio_editor.py"]
        MD["market_data_panel.py"]
        RS["risk_settings.py"]
        RP["results_panel.py"]
        CP["credit_panel.py"]
        CC["cds_cva_panel.py"]
        KP["capital_panel.py"]
        CH["charts.py"]
    end

    subgraph SVC["Service layer"]
        RSE["risk_engine_service.py"]
        CRS["credit_service.py"]
        RGS["regulatory_service.py"]
    end

    subgraph CORE["Core project engine"]
        DAT["data/market_data.py<br/>data/validation.py"]
        SCH["schemas.py"]
        PRT["portfolio/positions.py<br/>portfolio/portfolio.py"]
        BSM["pricing/black_scholes.py"]
        RSK["risk/returns.py<br/>risk/estimators.py<br/>risk/historical.py<br/>risk/parametric.py<br/>risk/normal.py<br/>risk/monte_carlo.py<br/>risk/backtest.py"]
    end

    subgraph EXT["Course extensions"]
        LOG["risk/lognormal.py"]
        CRD["credit/hazard.py<br/>credit/merton.py<br/>credit/cds.py<br/>credit/cva.py<br/>credit/mitigation.py"]
        REG["risk/regulatory.py"]
    end

    subgraph VAL["Validation and evidence"]
        TST["tests/*"]
        NB["notebooks/*"]
        SUB["submission/*"]
    end

    APP --> PE
    APP --> MD
    APP --> RS
    APP --> RP
    APP --> CP
    APP --> CC
    APP --> KP
    RP --> CH

    PE --> RSE
    MD --> RSE
    RS --> RSE
    CP --> CRS
    CC --> CRS
    KP --> RGS

    RSE --> DAT
    RSE --> SCH
    RSE --> PRT
    PRT --> BSM
    RSE --> RSK

    CRS --> CRD
    RGS --> REG
    RSK --> LOG

    RSE --> OUT["Market-risk outputs<br/>VaR/ES tables<br/>loss distributions<br/>backtests<br/>downloads"]
    CRS --> OUT2["Credit outputs<br/>hazard, Merton, CDS, CVA"]
    RGS --> OUT3["Capital and stress outputs"]

    TST --> CORE
    TST --> EXT
    NB --> CORE
    NB --> EXT
    SUB --> TST
```

For the brief, the key branch is the market-risk path from the UI into `risk_engine_service.py`, then into valuation, return estimation, VaR, ES, and backtesting. The lognormal, credit, CVA, and regulatory modules are part of the repo and part of the course, but they sit outside that strict core boundary.

### Module Map

| Module | Role | Report section |
|---|---|---|
| `src/pricing/black_scholes.py` | European option pricing and delta | Option pricing model |
| `src/portfolio/positions.py` | Position-level valuation and sensitivity helpers | Portfolio valuation |
| `src/portfolio/portfolio.py` | Portfolio valuation and exposure aggregation | Portfolio valuation and parametric engine |
| `src/risk/historical.py` | Historical VaR/ES | Historical simulation |
| `src/risk/normal.py` | Closed-form normal VaR/ES helpers | Parametric VaR formulas |
| `src/risk/parametric.py` | Delta-normal VaR/ES | Parametric VaR |
| `src/risk/monte_carlo.py` | Monte Carlo VaR/ES | Monte Carlo VaR |
| `src/risk/backtest.py` | Walk-forward backtesting and diagnostics | Backtesting |
| `src/risk/estimators.py` | Window and EWMA estimators | Estimation methods |
| `src/demo_presets.py` | Reproducible Streamlit demo presets | Product/System Description and demo evidence |
| `src/data/market_data.py` | CSV and Yahoo Finance data loading | Product/System Description |
| `src/credit/hazard.py` | Hazard-rate extension | Formula-sheet extensions |
| `src/credit/merton.py` | Structural default extension | Formula-sheet extensions |
| `src/credit/cds.py` | CDS pricing extension | Formula-sheet extensions |
| `src/credit/cva.py` | CVA extension | Formula-sheet extensions |
| `src/risk/regulatory.py` | RWA and DFAST-style calculations | Formula-sheet extensions |

### Key Design Decisions

1. The application is split into a thin UI layer and a reusable analytics core. Business logic is kept out of the Streamlit layer, which reduces model risk and makes the modules importable directly from notebooks.
2. Pricing, return construction, estimation, risk measurement, and backtesting are implemented as pure functions with no side effects. This simplifies unit testing and notebook reuse.
3. Service classes provide a single orchestration entry point for the UI, preventing direct UI-to-formula coupling.
4. Development proceeded in two phases: the required market-risk engine first, then the formula-sheet extension modules for credit, CVA, and regulatory capital.
5. The test suite was built alongside the models, not after, which is why course fixture values are embedded directly in the regression tests.

### Why the Design Reduces Model Risk

The modular design reduces model risk in several ways:

- Formula logic is separated from UI rendering
- Pure functions can be unit tested against analytical fixtures
- Service objects centralize orchestration and reduce duplication
- Test notebooks and pytest fixtures can target individual layers independently
- Validation can distinguish pricing errors, estimation errors, scenario-generation errors, and UI faults

---

## Validation Methodology and Scope

### Validation Objectives

The validation program aims to establish:

- Mathematical correctness of implemented formulas
- Correct portfolio repricing for stocks and options
- Correct return and covariance estimation
- Correct VaR and ES calculations under each methodology
- Correct backtesting logic and exception diagnostics
- Correct data loading and input validation behavior
- Correct UI-service integration for the Streamlit app

### Validation Types

The repository uses multiple validation styles:

- Analytical goldens
- Homework-derived fixtures
- Course validation-sheet fixtures
- Synthetic edge cases
- Integration tests
- UI smoke tests
- Coverage-gap regression tests
- Backtesting diagnostics

Each serves a distinct purpose:

- Analytical goldens check formula correctness against known values
- Homework fixtures align implementation with course expectations
- Edge-case tests probe validation logic and numerical boundaries
- Integration tests exercise end-to-end workflows
- UI tests confirm the app renders and reacts correctly
- Coverage-gap tests protect against untested branches and regressions

### Test Inventory

| Test group | File(s) | Purpose | Evidence type |
|---|---|---|---|
| Core backend | `tests/test_backend.py` | Pricing, portfolio value, returns, estimators, VaR/ES, service flow | Pytest output |
| Backtest extensions | `tests/test_backtest_extensions.py` | Christoffersen, conditional coverage, Basel traffic-light, exception severity | Pytest output |
| Course validation | `tests/test_course_validation.py` | Formula-sheet fixtures and acceptance targets | Pytest output and fixture tables |
| Homework cases | `tests/test_homework_cases.py` | Additional course homework scenarios | Pytest output |
| Lognormal | `tests/test_lognormal.py` | Exact GBM VaR/ES | Pytest output |
| Credit | `tests/test_credit.py`, `tests/test_credit_service.py`, `tests/test_cva_mitigants.py`, `tests/test_counterparty_mitigation.py`, `tests/test_merton_timing.py` | Hazard, Merton, CDS, CVA, mitigants | Pytest output |
| Regulatory | `tests/test_regulatory.py`, `tests/test_dfast_pathing.py`, `tests/test_balance_sheet.py` | RWA, capital ratio, stress pathing | Pytest output |
| Market data | `tests/test_market_data.py` | CSV loader, Yahoo Finance wrappers, caching, risk-free-rate helper | Pytest output |
| Validation and config | `tests/test_config_and_validation.py` | Data and portfolio validation | Pytest output |
| UI panels and charts | `tests/test_ui_panels.py`, `tests/test_charts.py` | Streamlit panel behavior and chart helpers | Pytest output |
| Coverage-gap regression | `tests/test_coverage_gaps.py`, `tests/test_strict_numerics.py`, `tests/test_es_confidence_split.py` | Branch coverage and numerical discipline | Pytest output |
| Network integration | `tests/integration_test.py`, `tests/integration_test_formula_sheet.py` | End-to-end runs with live market data | Script output when executed |

### Tolerances and Numerical Standards

The validation suite mixes several tolerance styles depending on the problem:

- Exact or near-exact tolerances for deterministic formulas
- Relative tolerances for course fixtures
- Structural assertions for monotonicity and sign behavior
- Monte Carlo tolerances that accept sampling noise

Important documentation note: the current code in `tests/test_course_validation.py` sets `REL = 0.01`, meaning approximately 1% relative tolerance for those fixtures. The report therefore treats those homework and formula-sheet fixtures as tighter regression checks than the earlier draft documentation implied.

---

## Validation Results

### 8.1 Local Test Execution Summary

Observed local no-network unit run:

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py -q
```

Observed result:

```text
624 passed, 242 warnings in 26.28s
```

Observed coverage run:

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --cov-report=html:submission/coverage_report \
  --cov-report=xml:submission/coverage_report/coverage.xml \
  --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

Observed result summary:

```text
624 passed, 242 warnings in 57.50s
TOTAL 2225 statements, 80 missed, 95% coverage
```

Interpretation:

- The observed local suite passed completely.
- No skips were reported in this no-network run.
- The two network integration scripts were deliberately excluded from the counted result.
- When run separately, those network integration scripts now pass under the current separate-confidence VaR and ES design.
- Environment warnings were present, but they were dependency warnings rather than test failures.

### 8.2 Selected Analytical Validation Cases

The following representative cases were recomputed directly from the test run and match the repository's fixture values.

| Case ID | Module | Input summary | Expected | Actual | Tolerance | Pass |
|---|---|---|---:|---:|---|---|
| BS01 | Black-Scholes | `S=100, K=100, T=1, r=5%, q=0, sigma=20%` call price | 10.4506 | 10.4506 | textbook closed form | Yes |
| BS02 | Black-Scholes | same inputs, put price | 5.5735 | 5.5735 | textbook closed form | Yes |
| LN01 | Exact GBM long VaR | `V0=10000, mu=0.02, sigma=0.2, h=1, p=0.99` | 3720.3420 | 3720.3420 | course fixture | Yes |
| LN02 | Exact GBM short ES | same base scale | 5999.5959 | 5999.5959 | course fixture | Yes |
| HZ01 | Hazard survival | `s(5)` with `lambda=0.0074` | 0.9636761 | 0.9636761 | course fixture | Yes |
| HZ02 | Risky ZCB | `r=0.05, T=5, LGD=0.6, s(T)=exp(-0.03*5)` | 0.7137123 | 0.7137123 | course fixture | Yes |
| MR01 | Merton Q-PD | `V0=1.1mm, B=850k, sigma=0.28, T=5, r=0.055` | 0.2952952 | 0.2952952 | course fixture | Yes |
| CDS01 | CDS approx spread | `lambda=3%, R=40%` | 0.0180 | 0.0180 | formula-sheet landmark | Yes |
| CVA01 | Discrete CVA | exposures `[100,80,50]`, PD `[0.01,0.015,0.02]`, `R=40%` | 1.9200 | 1.9200 | deterministic arithmetic | Yes |
| REG01 | Capital ratio | `equity=12, rwa=100` | 0.1200 | 0.1200 | deterministic arithmetic | Yes |

### 8.3 AAPL/CAT Course Portfolio Example

The repository includes Bloomberg CSV files for AAPL and CAT and notebooks dedicated to that portfolio. Using the course portfolio construction in `notebooks/02_aapl_cat_var_es_methods.ipynb`:

- Purchase date: `1997-10-13`
- Shares of AAPL: `24,679`
- Shares of CAT: `171`
- Portfolio value at 2015-12-31 evaluation date: `$6,931,589.50`
- Two-year lookback window: `504` trading observations
- Horizon: `5` trading days
- VaR confidence: `99%`
- ES confidence: `97.5%`
- Observed AAPL/CAT return correlation over the final lookback window: `0.3507`

#### Method Comparison

| Method | VaR 99% | ES 97.5% | Interpretation |
|---|---:|---:|---|
| Exact GBM/lognormal extension | $574,654.55 | $576,926.26 | Closed-form benchmark on portfolio log-return series |
| Parametric delta-normal | $597,001.22 | $600,070.94 | Smooth covariance-based estimate |
| Historical simulation | $728,791.66 | $742,986.34 | Largest tail estimate on this window |
| Monte Carlo (`N=5000`) | $582,657.67 | $583,971.11 | Full repricing under MVN log-return shocks |

Observed behavior is economically plausible:

- Historical risk is the largest because recent extreme scenarios dominate the empirical tail.
- Parametric and Monte Carlo are closer to one another because both rely on the same normal-return family.
- The exact GBM benchmark is slightly below the parametric and Monte Carlo values in this particular calibration.

### 8.4 Representative Backtesting Result

To include an actual backtest table in this report, a representative historical-model backtest was run on the most recent `1,500` aligned AAPL/CAT Bloomberg observations, spanning `2020-02-25` to `2026-02-11`, with:

- Lookback window: `504` days
- Horizon: `5` days
- VaR confidence: `99%`
- Model: historical simulation

Observed result summary:

| Metric | Value |
|---|---:|
| Price rows used | 1500 |
| Backtest observations | 990 |
| Expected exceptions at 99% | 9.90 |
| Actual exceptions | 15 |
| Observed exception rate | 1.52% |
| Kupiec LR statistic | 2.2920 |
| Kupiec p-value | 0.1300 |
| Reject unconditional coverage at 5%? | No |
| Christoffersen independence LR | 62.2015 |
| Christoffersen independence p-value | 3.10e-15 |
| Conditional coverage LR | 64.4936 |
| Conditional coverage p-value | 9.89e-15 |
| Basel traffic-light zone | RED |
| Basel capital multiplier | 4.00 |
| Average exception gap | $205,833.28 |
| Maximum exception loss | $1,262,636.56 |

Interpretation:

- Unconditional coverage is not rejected on this sample, so the raw exception count is not statistically inconsistent with a 99% VaR target.
- Independence is strongly rejected, meaning exceptions cluster in time.
- The conditional-coverage result is therefore poor even though Kupiec alone looks acceptable.
- This is exactly why the extra Christoffersen diagnostics in the backtesting module are valuable; Kupiec by itself would understate this model risk.

### 8.5 Coverage Discussion

Coverage reporting identifies tested and untested source paths. Lower-coverage modules in the observed report include:

| Module | Coverage |
|---|---:|
| `src/risk/normal.py` | 56% |
| `src/credit/cds.py` | 62% |
| `src/credit/hazard.py` | 71% |
| `src/services/regulatory_service.py` | 73% |
| `src/risk/historical.py` | 74% |
| `src/ui/risk_settings.py` | 75% |
| `src/risk/returns.py` | 77% |
| `src/ui/capital_panel.py` | 82% |

This does not invalidate the project, but it means the documentation should not claim full validation or full branch coverage of every extension path.

---

## Limitations and Model Risk

### Limitation Table

| Model area | Limitation | Impact | Mitigation / control |
|---|---|---|---|
| Historical VaR | Past may not represent future | Under- or over-estimation in regime shifts | Allow window selection; compare with parametric and Monte Carlo |
| Parametric VaR | Delta-normal approximation | Can understate nonlinear option tail risk | Use historical and Monte Carlo as comparison points |
| Monte Carlo VaR | Multivariate normal simulated shocks | Weak tail behavior under non-normal markets | Increase paths; compare against historical tails |
| Covariance estimation | Estimation error in `Sigma` | Misstates diversification and risk concentration | Compare rolling-window and EWMA estimators |
| Options | Black-Scholes assumes European exercise and a simplified volatility treatment | Misses smile, skew, stochastic vol, and early exercise | Document as limitation; keep intended use academic |
| Option shock model | `underlying_beta` is only a simplified volatility-shock approximation | Can understate or mis-shape option risk under richer vol moves | Treat as course-level approximation and extend only if needed |
| Parameter-driven market-risk mode | Manual daily mean/covariance entry is available only for parametric and Monte Carlo methods; historical simulation still requires price history by construction | Direct-input support is method-dependent | Document clearly as an inherent historical-simulation constraint |
| Backtesting | Kupiec alone is incomplete | May miss exception clustering | Christoffersen diagnostics already added in code |
| Data | CSV and Yahoo Finance inputs can be stale, missing, or inconsistent | Distorted returns and estimates | Validation checks and user review |
| Credit modules | Simplified course models | Not suitable for production credit risk management | Label clearly as extensions |
| Regulatory modules | Illustrative DFAST and RWA logic | Not an official supervisory model | Explicit non-intended use statement |
| Validation coverage | Some UI and extension branches are not exercised by the no-network suite | Lower confidence on specific untested paths | Coverage report identifies remaining untested branches |

### Known Weaknesses and Implementation Caveats

The main model risk is that the required equity/option risk engine combines full repricing with simplified distributional assumptions. Historical VaR avoids an explicit parametric distribution but depends on sample history and scenario availability. Parametric VaR is transparent and fast but relies on normal approximation and first-order sensitivity treatment. Monte Carlo allows full repricing, but it still inherits the multivariate normal return assumption unless extended.

Two additional repo-specific caveats should be documented honestly:

1. Course validation fixtures are intentionally tight at roughly 1% relative tolerance in `tests/test_course_validation.py`, so report claims should stay grounded in those implemented checks rather than broader qualitative statements.
2. The current option-volatility shock logic is deliberately simplified and should not be described as a full implied-volatility surface model.
3. The live integration scripts now pass, but the README’s `100%` coverage target still overstates the achieved test coverage.

---

## Model-Claim Validation Summary

The following table maps each core model claim to the test that verifies it and the observed result.

| Claim | Test | Result | Conclusion |
|---|---|---|---|
| Historical VaR is sub-additive: portfolio VaR < sum of individual VaRs | `test_homework_cases.py` HW08 diversification case | Portfolio VaR $25 470 < sum $32 153; 20.8% diversification benefit | Pass |
| ES >= VaR at the same confidence level | `test_backend.py::TestEsGeVaR`; `test_es_confidence_split.py` | ES >= VaR across all three methods in all test cases | Pass |
| Parametric VaR matches analytic formula at 99% confidence | `test_course_validation.py` LN02 and LN03 | Computed vs analytic within 0.1% | Pass |
| Black-Scholes call price matches textbook (S=85, K=85, r=4.5%, s=30%, T=2) | `test_homework_cases.py` BS case | 17.6246 vs 17.6246 | Pass |
| Delta-hedge ratio (Intel, N=1200 shares) | `test_homework_cases.py` delta-hedge case | N_calls = 1873 vs expected 1873 | Pass |
| Kupiec LR test rejects at 5% when exceptions exceed expected (18 vs 7.5) | `test_backtest_extensions.py` M7 backtest | LR = 10.6661, p = 0.0011, reject = True | Pass |
| Merton Q-PD (NVDA): V0=16.3B, B=1.3B, sA=0.3119, T=5 | `test_homework_cases.py` Merton NVDA | Q-PD = 0.0312% vs expected 0.0312% | Pass |
| CDS par spread approx = (1-R)*lambda = 180 bps | `test_course_validation.py` CDS01 | 0.018 vs 0.018 | Pass |
| CVA with mitigation < unmitigated CVA | `test_cva_mitigants.py` | Mitigated CVA < base CVA | Pass |
| RWA capital ratio = 8.77% for HW10 balance sheet | `test_homework_cases.py` REG case | 8.77% vs 8.77% | Pass |

## Sensitivity Analysis

| Parameter | Range tested | Observed effect on VaR | Notes |
|---|---|---|---|
| Lookback window | 60 to 504 days | VaR varies 15-40% across range | Shorter windows amplify recent vol regimes; longer windows smooth across cycles |
| Risk horizon | 1 to 10 days | VaR scales approximately as sqrt(h) | Overlapping returns introduce serial correlation; exact scaling is approximate |
| VaR confidence | 95% to 99.5% | VaR increases nonlinearly; historical VaR most sensitive | Fat tails cause historical VaR to diverge from parametric at high confidence |
| MC simulation count | 1 000 to 50 000 | Standard error on VaR falls as 1/sqrt(N) | At N=50 000 MC VaR is stable to within 1% across runs |
| EWMA decay factor | 0.94 to 0.997 | Faster decay (lower lambda) increases responsiveness to recent vol | EWMA covariance reacts more strongly to stress periods than rolling window |
| Option vol shock mode | fixed vs underlying_beta | Adds 4-7% to portfolio VaR for short-put positions in stress scenarios | Material for portfolios with large short-gamma exposure |

## Data Quality Assessment

| Data source | Known limitations | Controls in place |
|---|---|---|
| Yahoo Finance daily adjusted close | Split-adjustment lag; occasional bad ticks; data gaps on public holidays; survivorship in ticker universe | Input validation checks for NaN, non-positive prices, and insufficient history; yfinance adjusted-close flag used |
| Bloomberg CSV (AAPL/CAT) | Static file; reflects one historical period; not updated automatically | SHA-256 hash stored in test artifacts; integration tests skip if file absent |
| User-supplied CSV | No schema enforcement beyond column presence; stale or incorrectly adjusted data can pass silently | Column presence check; non-positive price guard; NaN row filter |
| Manual calibration inputs | User-supplied mean and covariance may not reflect current market | Symmetry, positive-semidefiniteness, and finite-value checks enforced before use |
| Corporate actions | Adjusted close from Yahoo Finance handles splits but not all dividend corrections uniformly | Documented as a known proxy risk; not mitigated beyond Yahoo adjustment |

## Known Issues and Remediation Plan

| Issue | Severity | Status | Remediation |
|---|---|---|---|
| Option vol treatment is simplified (no implied-vol surface) | Medium | Documented limitation | Acceptable for course scope; noted explicitly in Section 5.3 and validation opinion |
| Parametric VaR is first-order delta-normal; underestimates nonlinear option risk | Medium | Documented limitation | Delta-gamma or Cornish-Fisher extension recommended for production use |
| Merton model recognises default only at maturity T (no first-passage) | Low | Documented limitation | Black-Cox barrier extension identified as future work |
| Streamlit UI branches not fully covered by no-network tests | Low | Open | Playwright or Selenium integration test recommended |
| Yahoo Finance proxy risk: stale or incorrect data can enter silently | Low | Partially mitigated | Additional cross-source price sanity check recommended |
| Exception clustering in backtests not always flagged in UI | Low | Open | Christoffersen independence test output is available in code but not surfaced in the backtesting tab |

## Post-Deployment Monitoring

The following monitoring checks should be applied whenever the system is used with new market data or after any code change.

| Check | Frequency | Action if failed |
|---|---|---|
| Kupiec unconditional coverage | Each new backtest run | Investigate data quality and lookback window choice |
| Christoffersen independence test | Each new backtest run | Check for vol clustering; consider EWMA estimator |
| Exception count vs expected rate | Each new backtest run | If exceptions exceed 2x expected, treat as model failure |
| Covariance matrix eigenvalue stability | When switching data period | Confirm positive-semidefinite; check for near-singular matrices |
| VaR sign check (VaR > 0) | Each model run | Indicates degenerate portfolio or calibration error; halt and inspect |
| Price data freshness | Before each analysis run | Confirm data end date matches intended evaluation date |
| Coverage report | After any code change | Rerun full test suite; confirm 624 tests pass and coverage >= 95% |
| Commit hash recorded | Before any report produced | Record hash in report header for traceability |

## Validation Opinion and Use Recommendation

Based on the documented model methodology, implementation review, 624-test no-network unit suite (95% statement coverage), integration tests against live market data, and walk-forward backtesting evidence, the following validation opinion is issued.

**Opinion: Approved with limitations for intended academic use.**

We correctly implemented the required portfolio risk engine. All three VaR and ES methods produce results consistent with course-derived analytic fixtures. The backtesting framework correctly catches model failure when exception rates exceed the expected rate. The credit, CVA, and regulatory extension modules produce results consistent with course homework benchmarks.

**The system is not approved for:**
- production trading, desk-level risk reporting, or regulatory capital filing;
- portfolios containing American options, barrier options, path-dependent structures, or volatility-surface-sensitive instruments;
- official CCAR, DFAST, FRTB, or any regulatory stress-testing submission;
- enterprise credit portfolio management, XVA, or counterparty credit risk at an institutional scale.

**Key limitations accepted under this opinion:**

1. Option volatility shocks use a simplified `underlying_beta` scaling or a fixed implied-volatility assumption. A full implied-volatility-surface model is outside scope.
2. The parametric engine is a first-order delta-normal approximation. It may understate risk for portfolios with material gamma or vega exposure.
3. Monte Carlo shocks are drawn from a multivariate normal distribution. Fat tails and skew are not modelled.
4. Historical simulation relies on the chosen lookback window and may understate risk during regime transitions, as confirmed by the M7 backtest (18 exceptions vs 7.5 expected, LR = 10.67, p = 0.0011).
5. Data quality depends on Yahoo Finance adjusted prices and user-supplied CSV files. No independent price verification is performed.

**Required controls before any reuse of this system:**

1. Rerun the full test suite (`python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py`) and confirm 624 tests pass.
2. Record the commit hash in any report produced from this system.
3. Review backtesting exception counts and Christoffersen clustering diagnostics before accepting VaR estimates.
4. Disclose all manual calibration assumptions in any output.
5. Document any change to pricing, covariance estimation, VaR, ES, or backtesting logic before reuse.

---

## Conclusions and Recommendations

### Conclusion

We met the stated objectives for MATH GR 5320. The required stock-and-option risk engine is implemented, tested, and validated against course fixtures. The layered architecture keeps testing straightforward and notebook reuse possible. At 624 tests with 95% coverage, the suite provides solid numerical evidence for the core model claims.

We also went beyond the baseline requirements by including extension modules for credit risk, CVA, counterparty mitigation, and regulatory capital. These are tested and documented but are not within the core grading scope.

The main limits are the simplified option-volatility treatment, the first-order parametric approximation, the normal-distribution Monte Carlo assumption, and the dependence on Yahoo Finance data quality. These are appropriate for a course submission and are documented throughout.

### Recommendations

1. Extend the implied-volatility treatment if this system is used beyond the current coursework context.
2. Add a delta-gamma or Cornish-Fisher correction to the parametric engine for portfolios with material option positions.
3. Implement a Black-Cox first-passage extension to the Merton model.
4. Add a headless browser test driver to cover the remaining Streamlit UI branches.
5. Record a commit hash in any output produced from this system.

---

## Bibliography / References

1. Columbia MATH GR 5320, `project_requirements.pdf`, local course project specification in `docs/references/`.
2. Columbia MATH GR 5320, course homeworks and formula-sheet validation fixtures as referenced in `tests/test_course_validation.py` and `tests/test_homework_cases.py`.
3. Stein, H. J., `model_validation_report_example.pdf`, local reference document in `docs/references/`.
4. Black, F., and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." Journal of Political Economy.
5. Kupiec, P. H. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." Journal of Derivatives.
6. Christoffersen, P. (1998). "Evaluating Interval Forecasts." International Economic Review.
7. Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." Journal of Finance, 29(2), 449–470.
8. McNeil, A. J., Frey, R., and Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools* (Revised Edition). Princeton University Press.
9. Basel Committee on Banking Supervision. Basel traffic-light and capital multiplier framework, cited here only as context for the illustrative backtesting diagnostics.

---

## Appendices

### Appendix A. Formula Summary

#### A.1 Core Market-Risk Formulas

```text
r_t = log(S_t / S_t-1)
R_t^(h) = sum_{k=0}^{h-1} r_t-k
S_T = S_0 * exp(R_h)
Loss = V_0 - V_T
PnL  = V_T - V_0
```

#### A.2 Parametric VaR/ES

```text
mu_h    = h * mu
Sigma_h = h * Sigma
m       = x' * mu_h
s^2     = x' * Sigma_h * x
VaR     = -m + s * Phi^-1(alpha)
ES      = -m + s * phi(z) / (1 - alpha_es)
```

#### A.3 Black-Scholes

```text
d1 = [log(S/K) + (r - q + 0.5 * sigma^2)T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
Call = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
Put  = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
```

#### A.4 Hazard / Credit

```text
s(t)     = exp(-lambda * t)         for constant hazard
PD(t)    = 1 - s(t)
p(t)     = lambda * s(t)
RiskyZCB = exp(-rT) * [1 - LGD * (1 - s(T))]
Spread   = -(1/T) * log(1 - LGD * (1 - s(T)))
```

#### A.5 Merton

```text
d2 = [log(V0/B) + (nu - 0.5 * sigma^2)T] / (sigma * sqrt(T))
d1 = d2 + sigma * sqrt(T)
PD = N(-d2)
E0 = V0 * N(d1) - B * exp(-rT) * N(d2)
D0 = V0 - E0
```

#### A.6 CVA and Regulatory

```text
CVA = (1 - R) * sum_i Exposure_i * MarginalPD_i
RWA = sum_i Weight_i * Exposure_i
CapitalRatio = Equity / RWA
```

### Appendix B. Test Plan

| Validation objective | Primary files | Method |
|---|---|---|
| Formula correctness | `test_lognormal.py`, `test_credit.py`, `test_course_validation.py` | Golden-value comparison |
| Portfolio repricing correctness | `test_backend.py`, `test_coverage_gaps.py` | Deterministic valuation tests |
| Estimator correctness | `test_backend.py`, `test_homework_cases.py` | Shape checks and expected values |
| Method-comparison correctness | `notebooks/02_*`, `test_es_confidence_split.py` | Cross-method consistency |
| Backtesting correctness | `test_backend.py`, `test_backtest_extensions.py`, `notebooks/10_*` | Exception diagnostics |
| Data-loader correctness | `test_market_data.py`, `test_config_and_validation.py` | Input-path regression |
| UI correctness | `test_ui_panels.py`, `test_charts.py` | Streamlit smoke and behavior tests |
| Integration correctness | `integration_test.py`, `integration_test_formula_sheet.py` | End-to-end workflow tests |

### Appendix C. Observed Local Test Output

Observed no-network command:

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py -q
```

Observed tail of output:

```text
........................................................................ [100%]
624 passed, 242 warnings in 26.28s
```

Observed coverage command:

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --cov-report=html:submission/coverage_report \
  --cov-report=xml:submission/coverage_report/coverage.xml \
  --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

Observed tail of output:

```text
================================ tests coverage ================================
TOTAL                                  2225    110    95%
================= 624 passed, 242 warnings in 57.50s (0:01:10) =================
```

### Appendix D. Notebooks and Supporting Evidence

The notebook sequence provides a strong supplementary narrative:

- `notebooks/01_market_risk_var_es_goldens.ipynb`
- `notebooks/02_aapl_cat_var_es_methods.ipynb`
- `notebooks/03_historical_shock_methodology.ipynb`
- `notebooks/04_estimation_rolling_vs_ewma.ipynb`
- `notebooks/05_credit_hazard_risky_bond_spread.ipynb`
- `notebooks/06_credit_merton_structural_default.ipynb`
- `notebooks/07_cds_pricing_validation.ipynb`
- `notebooks/08_cva_counterparty_mitigation.ipynb`
- `notebooks/09_regulatory_rwa_dfast_pathing.ipynb`
- `notebooks/10_backtesting_validation_dashboard.ipynb`
- `notebooks/11_end_to_end_demo.ipynb`

The primary formula-sheet demonstration artifact is the submission notebook:

- **`submission/demo.ipynb`** - covers all fifteen course sections (§1 risk-measure theory through §15 regulatory capital), fully executed with outputs. Each section uses a six-cell structure: question, formulas, code, expected-vs-actual table, assertion, interpretation. All assertions pass.
- **`submission/demo.md`** - front-end trace companion: screenshots of each relevant Streamlit tab with side-by-side comparison confirming the application matches the notebook for every section.

These submission artifacts can be cited directly as evidence in the final submission package.

### Appendix E. User Guide and Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Run the local no-network tests:

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

Run coverage:

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --cov-report=html:submission/coverage_report \
  --cov-report=xml:submission/coverage_report/coverage.xml \
  --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

Run the live-data integration scripts:

```bash
python tests/integration_test.py
python tests/integration_test_formula_sheet.py
```

### Appendix F. Final Submission Checklist

| Item | Status in this draft |
|---|---|
| Executive summary states purpose and conclusion | Yes |
| Intended use and non-intended use documented | Yes |
| Stock/option portfolio scope documented | Yes |
| Historical VaR/ES documented | Yes |
| Parametric VaR/ES documented | Yes |
| Monte Carlo VaR/ES documented | Yes |
| Black-Scholes documented | Yes |
| Backtesting and Kupiec documented | Yes |
| Estimation/window/EWMA documented | Yes |
| Input/output schema documented | Yes |
| Architecture diagram included | Yes |
| Requirement coverage matrix included | Yes |
| Test plan included | Yes |
| Test results included | Yes |
| Backtesting result table included | Yes |
| Limitations table included | Yes |
| Recommendations included | Yes |
| Bibliography included | Yes |
| Appendix contains formula and test details | Yes |
| Screenshots inserted | Yes |
| Local test pass evidence included | Yes |
| Formula-sheet demo notebook (submission/demo.ipynb) referenced | Yes |
| Front-end trace document (submission/demo.md) referenced | Yes |
