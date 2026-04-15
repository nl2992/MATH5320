<div class="titlepage">

<div class="center">

**Combined Final Report**\
**MATH5320 Portfolio Risk Management System**\
Columbia University\
MATH GR 5320 Financial Risk Management\
Spring 2026\

<div class="tabular">

@L4cmL9cm@ **Field** & **Value**\
Authors & Nigel Li, Michael Adegbite, Stella\
Reference commit & `main` branch, May 2026 submission version\
No-network tests & 644 passed, 0 failed, 1 skipped\
Statement coverage & 95%\
Integration scripts & 2 / 2 passed\
PyPI package & `math5320-portfolio-risk-system` v0.2.1\

</div>

<div class="minipage">

*This document consolidates the segmented model documentation, software design, test plan, and test results into one submission report. The crosswalk tables at the front are included so the implementation, validation evidence, and project requirements can be checked without reading the report linearly.*

</div>

</div>

</div>

# Quick Navigation

The table below lets the reader jump directly to the section that provides evidence for each requirement or marking concern.

<div class="center">

<div class="longtable">

L5.5cmL7.5cm **Marker may want to check** & **Direct reference**\
Core project requirements & Section <a href="#sec:req-matrix" data-reference-type="ref" data-reference="sec:req-matrix">2</a>: Requirements Matrix\
Bloomberg MRM template compliance & Section <a href="#sec:mrm-crosswalk" data-reference-type="ref" data-reference="sec:mrm-crosswalk">3</a>: MRM Template Crosswalk\
Deliverable structure and grading & Section <a href="#sec:deliverable-crosswalk" data-reference-type="ref" data-reference="sec:deliverable-crosswalk">4</a>: Deliverable Crosswalk\
System purpose and scope & Section <a href="#sec:scope" data-reference-type="ref" data-reference="sec:scope">6</a>\
Model-risk governance framework & Section <a href="#sec:mrm-framework" data-reference-type="ref" data-reference="sec:mrm-framework">7</a>\
Portfolio payoffs and examples & Section <a href="#sec:product" data-reference-type="ref" data-reference="sec:product">8</a>\
Pricing model, Black-Scholes & Section <a href="#sec:bs" data-reference-type="ref" data-reference="sec:bs">9.2</a>\
Historical simulation VaR/ES & Section <a href="#sec:historical" data-reference-type="ref" data-reference="sec:historical">9.3</a>\
Parametric delta-normal VaR/ES & Section <a href="#sec:parametric" data-reference-type="ref" data-reference="sec:parametric">9.4</a>\
Monte Carlo VaR/ES & Section <a href="#sec:mc" data-reference-type="ref" data-reference="sec:mc">9.5</a>\
Backtesting, Kupiec and Christoffersen & Sections <a href="#sec:backtest-method" data-reference-type="ref" data-reference="sec:backtest-method">9.6</a> and <a href="#sec:backtest-results" data-reference-type="ref" data-reference="sec:backtest-results">12.4</a>\
Extension modules, GBM, credit, regulatory & Section <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
Software architecture and design & Section <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>\
Validation methodology and test plan & Section <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a>\
Numerical precision and behavioural tests & Section <a href="#sec:numerical-behavioural-results" data-reference-type="ref" data-reference="sec:numerical-behavioural-results">13</a>\
Test results and coverage & Section <a href="#sec:test-results" data-reference-type="ref" data-reference="sec:test-results">12</a>\
Known limitations and model risk & Section <a href="#sec:limitations" data-reference-type="ref" data-reference="sec:limitations">15</a>\
Validation opinion and recommendations & Section <a href="#sec:conclusions" data-reference-type="ref" data-reference="sec:conclusions">16</a>\

</div>

</div>

# Requirements Matrix

The table below maps each project requirement to its implementation, test evidence, and the section of this report where it is discussed.

<div class="center">

<div class="longtable">

L3.2cmL4cmL3.5cmL2.5cm **Requirement** & **Implementation** & **Test evidence** & **Report section**\
Portfolio of stocks and options as input & `src/schemas.py`, `src/ui/portfolio_editor.py` & `test_backend.py`, `test_config_and_validation.py` & <a href="#sec:product" data-reference-type="ref" data-reference="sec:product">8</a>, <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>\
Load historical market data, CSV and Yahoo Finance & `src/data/market_data.py`, `src/ui/market_data_panel.py` & `test_market_data.py` & <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>\
Accept manual mean/covariance input & `src/risk/estimators.py` (`manual_mean_cov`) & `test_backend.py`, `test_coverage_gaps.py` & <a href="#sec:parametric" data-reference-type="ref" data-reference="sec:parametric">9.4</a>\
Historical simulation VaR & `src/risk/historical.py` & `test_backend.py`, `test_homework_cases.py` & <a href="#sec:historical" data-reference-type="ref" data-reference="sec:historical">9.3</a>\
Historical simulation ES & `src/risk/historical.py` & `test_backend.py`, `test_es_confidence_split.py` & <a href="#sec:historical" data-reference-type="ref" data-reference="sec:historical">9.3</a>\
Parametric delta-normal VaR & `src/risk/parametric.py`, `src/risk/normal.py` & `test_backend.py`, `test_course_validation.py` & <a href="#sec:parametric" data-reference-type="ref" data-reference="sec:parametric">9.4</a>\
Parametric ES & `src/risk/parametric.py`, `src/risk/normal.py` & `test_backend.py`, `test_es_confidence_split.py` & <a href="#sec:parametric" data-reference-type="ref" data-reference="sec:parametric">9.4</a>\
Monte Carlo VaR and ES & `src/risk/monte_carlo.py` & `test_backend.py`, `test_coverage_gaps.py` & <a href="#sec:mc" data-reference-type="ref" data-reference="sec:mc">9.5</a>\
European option pricing, Black-Scholes & `src/pricing/black_scholes.py` & `test_backend.py`, `test_homework_cases.py` & <a href="#sec:bs" data-reference-type="ref" data-reference="sec:bs">9.2</a>\
Covariance estimation, rolling and EWMA & `src/risk/estimators.py` & `test_backend.py`, `test_homework_cases.py` & <a href="#sec:parametric" data-reference-type="ref" data-reference="sec:parametric">9.4</a>\
Option volatility shock mode & `src/portfolio/positions.py`, `src/risk/historical.py`, `src/risk/monte_carlo.py` & `test_backend.py` & <a href="#sec:bs" data-reference-type="ref" data-reference="sec:bs">9.2</a>, <a href="#sec:mc" data-reference-type="ref" data-reference="sec:mc">9.5</a>\
Walk-forward VaR backtesting & `src/risk/backtest.py` & `test_backend.py`, `test_backtest_extensions.py` & <a href="#sec:backtest-method" data-reference-type="ref" data-reference="sec:backtest-method">9.6</a>, <a href="#sec:backtest-results" data-reference-type="ref" data-reference="sec:backtest-results">12.4</a>\
Kupiec unconditional coverage test & `src/risk/backtest.py` (`kupiec_test`) & `test_backend.py`, `test_backtest_extensions.py` & <a href="#sec:backtest-method" data-reference-type="ref" data-reference="sec:backtest-method">9.6</a>\
Christoffersen independence test & `src/risk/backtest.py` (`christoffersen_test`) & `test_backtest_extensions.py` & <a href="#sec:backtest-method" data-reference-type="ref" data-reference="sec:backtest-method">9.6</a>\
Basel traffic-light classification & `src/risk/backtest.py` (`basel_traffic_light`) & `test_backtest_extensions.py` & <a href="#sec:backtest-method" data-reference-type="ref" data-reference="sec:backtest-method">9.6</a>\
Numerical precision and failure modes & Black-Scholes limits, log-return cancellation, covariance stability, EWMA stability, extreme-tail VaR/ES & `test_numerical_precision.py` (NP_01–NP_07) & <a href="#sec:numerical-behavioural-results" data-reference-type="ref" data-reference="sec:numerical-behavioural-results">13</a>\
Behavioural confirmation tests & Monotonicity, put-call parity, no-arbitrage lower bound, ES/VaR ordering & `test_backend.py` (BEH_01–BEH_08) & <a href="#sec:numerical-behavioural-results" data-reference-type="ref" data-reference="sec:numerical-behavioural-results">13</a>\
Convergence and inversion tests & MC convergence, Merton implied barrier, Kupiec exact-count check & `test_backend.py` (CONV_01, INV_01–INV_02) & <a href="#sec:numerical-behavioural-results" data-reference-type="ref" data-reference="sec:numerical-behavioural-results">13</a>\
P&L attribution and hedge effectiveness & Linear P&L residual and one-day delta hedge check & `test_backend.py` (PNL_01, HEDGE_01) & <a href="#sec:numerical-behavioural-results" data-reference-type="ref" data-reference="sec:numerical-behavioural-results">13</a>\
Exact GBM/lognormal VaR and ES & `src/risk/lognormal.py` & `test_lognormal.py`, `test_course_validation.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
Reduced-form hazard credit model & `src/credit/hazard.py` & `test_credit.py`, `test_course_validation.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
Merton structural default model & `src/credit/merton.py` & `test_credit.py`, `test_homework_cases.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
CDS par spread & `src/credit/cds.py` & `test_credit.py`, `test_course_validation.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
CVA with counterparty mitigation & `src/credit/cva.py`, `src/credit/mitigation.py` & `test_credit.py`, `test_cva_mitigants.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
RWA and regulatory capital & `src/risk/regulatory.py` & `test_regulatory.py`, `test_balance_sheet.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
DFAST-style stress pathing & `src/risk/regulatory.py` & `test_dfast_pathing.py` & <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">9.7</a>\
Software design documentation & Layered architecture across `src/` and `app.py` & All test files & <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>\
Test plan & `tests/` directory and validation framework & 644 passed, 1 skipped, 0 failed & <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a>\
Test results & pytest, coverage run, artifact bundle & `submission/test_artifacts/` & <a href="#sec:test-results" data-reference-type="ref" data-reference="sec:test-results">12</a>\
Model documentation & This combined report and segmented deliverables & All validation evidence & Entire document\

</div>

</div>

# Bloomberg MRM Template Crosswalk

The validation report structure follows the Bloomberg Enterprise Risk Model Validation Report Template . The table below maps each section of that template to the corresponding section of this combined report.

<div class="center">

<div class="longtable">

L5.5cmL7.5cm **MRM template section** & **Where addressed**\
\
Purpose of review & Section <a href="#sec:exec-summary" data-reference-type="ref" data-reference="sec:exec-summary">5</a>: Executive Summary\
Model description & Sections <a href="#sec:product" data-reference-type="ref" data-reference="sec:product">8</a> and <a href="#sec:model-description" data-reference-type="ref" data-reference="sec:model-description">9</a>\
Current and intended usage & Section <a href="#sec:scope" data-reference-type="ref" data-reference="sec:scope">6</a>: Scope and Intended Use\
Validation methodology and scope & Section <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a>\
Critical analysis & Sections <a href="#sec:limitations" data-reference-type="ref" data-reference="sec:limitations">15</a> and <a href="#sec:conclusions" data-reference-type="ref" data-reference="sec:conclusions">16</a>\
\
System reviewed including version ID & Section <a href="#sec:scope" data-reference-type="ref" data-reference="sec:scope">6</a>\
Business unit and user context & Section <a href="#sec:scope" data-reference-type="ref" data-reference="sec:scope">6</a>: academic/course use\
Report purpose & Section <a href="#sec:exec-summary" data-reference-type="ref" data-reference="sec:exec-summary">5</a>\
Version history & Section <a href="#sec:scope" data-reference-type="ref" data-reference="sec:scope">6</a>\
\
Product description and payoff & Section <a href="#sec:product" data-reference-type="ref" data-reference="sec:product">8</a>\
Example portfolios & Section <a href="#sec:product" data-reference-type="ref" data-reference="sec:product">8</a>\
\
Theory and assumptions & Section <a href="#sec:model-description" data-reference-type="ref" data-reference="sec:model-description">9</a>\
Pros and cons of model choice & Sections <a href="#sec:mrm-framework" data-reference-type="ref" data-reference="sec:mrm-framework">7</a> and <a href="#sec:limitations" data-reference-type="ref" data-reference="sec:limitations">15</a>\
Mathematical inputs and outputs & Section <a href="#sec:model-description" data-reference-type="ref" data-reference="sec:model-description">9</a>\
Implementation and numerical methods & Section <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>\
Calibration methodology & Sections <a href="#sec:parametric" data-reference-type="ref" data-reference="sec:parametric">9.4</a> and <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>\
\
Scope and how validation was performed & Section <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a>\
Benchmark model & Section <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a>: Benchmark Comparisons\
Tests performed & Sections <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a> and <a href="#sec:test-results" data-reference-type="ref" data-reference="sec:test-results">12</a>\
Outputs reviewed & Sections <a href="#sec:test-results" data-reference-type="ref" data-reference="sec:test-results">12</a> and <a href="#sec:numerical-behavioural-results" data-reference-type="ref" data-reference="sec:numerical-behavioural-results">13</a>\
\
Presentation and critical discussion & Section <a href="#sec:test-results" data-reference-type="ref" data-reference="sec:test-results">12</a>, including backtesting and numerical precision results\
\
Validation opinion and recommendations & Section <a href="#sec:conclusions" data-reference-type="ref" data-reference="sec:conclusions">16</a>\

</div>

</div>

# Deliverable Crosswalk

<div class="center">

<div class="longtable">

L3.5cmL1.5cmL8cm **Deliverable** & **Points** & **Coverage in this combined report and repository**\
. Model documentation & 30 & Section <a href="#sec:exec-summary" data-reference-type="ref" data-reference="sec:exec-summary">5</a> through <a href="#sec:conclusions" data-reference-type="ref" data-reference="sec:conclusions">16</a>. Formal segmented version in `submission/latex_deliverables/01_model_documentation.tex`.\
2. Software design documentation & 15 & Section <a href="#sec:architecture" data-reference-type="ref" data-reference="sec:architecture">10</a>. Full module inventory, interface contracts, data flow, separation of concerns, and validation strategy. Formal segmented version in `submission/latex_deliverables/02_software_design_documentation.tex`.\
3. Test plan & 20 & Section <a href="#sec:test-plan" data-reference-type="ref" data-reference="sec:test-plan">11</a>. Test categories, benchmark comparisons, numerical precision checks, behavioural tests, convergence/inversion checks, P&L attribution, and hedge-effectiveness testing. Formal segmented version in `submission/latex_deliverables/03_test_plan.tex`.\
4. Software, running & 25 & Repository at `github.com/nl2992/MATH5320`. Install using `pip install math5320-portfolio-risk-system` and run with `streamlit run app.py`.\
5. Test results & 10 & Section <a href="#sec:test-results" data-reference-type="ref" data-reference="sec:test-results">12</a>. 644 tests passed, 1 intentionally skipped, 0 failed, 95% coverage, two live integration scripts passed. Artifact bundle in `submission/test_artifacts/`. Formal segmented version in `submission/latex_deliverables/04_test_results.tex`.\

</div>

</div>

# Executive Summary

The MATH5320 Portfolio Risk Management System is an eight-tab Streamlit application built for Columbia MATH GR 5320 Financial Risk Management, Spring 2026. It takes user-defined portfolios of equities and European options, loads aligned historical price data from CSV or Yahoo Finance, and produces Value at Risk (VaR) and Expected Shortfall (ES) under three independently implemented methods: historical simulation, parametric delta-normal, and Monte Carlo full-repricing. Walk-forward VaR backtesting with Kupiec unconditional coverage and Christoffersen independence diagnostics is integrated and produces reproducible output. A second layer of extension modules demonstrates exact GBM lognormal VaR and ES, reduced-form hazard credit models, the Merton structural default model, CDS pricing, CVA with counterparty mitigation, and illustrative regulatory capital and DFAST-style projections.

**Validation methodology.** Validation was performed through a 644-test passing no-network suite with one intentional skip and 95% statement coverage across `src/`, supplemented by two live-data integration scripts. Core formulas were cross-checked against analytical golden values, course-homework fixtures, benchmark comparisons, and walk-forward backtesting outputs. The final test pass also adds explicit coverage for numerical precision failure modes, behavioural confirmation, Monte Carlo convergence, Merton inversion, Kupiec exact-count behaviour, P&L attribution, and one-day hedge effectiveness.

**Critical analysis.** The system’s main strengths are its three-method comparative framework, the clean separation of pricing, risk, service, and UI layers, and the depth of its test coverage. Black-Scholes with user-supplied implied volatility is the right standard tool for European options and is implemented accurately. Historical simulation and Monte Carlo use full portfolio repricing, which is the right treatment for nonlinear option books. The main limitations are deliberate and documented: option repricing uses fixed implied volatility or a simplified vol-shock approximation rather than a full implied-vol surface; the parametric method is a first-order delta-normal approximation; Monte Carlo shocks are multivariate normal; and the Merton model recognises default only at maturity.

**Validation opinion: approved with limitations for intended academic use.** The system is a sound, well-tested academic risk calculation platform for MATH GR 5320. It is not suitable for production deployment without independent validation, formal governance controls, calibrated implied-volatility surface dynamics, and broader market-data and computational infrastructure.

# Introduction and Scope

## System Reviewed

The system reviewed is the **MATH5320 Portfolio Risk Management System**, implemented in Python and delivered through an eight-tab Streamlit application. The package is published to PyPI as `math5320-portfolio-risk-system` v0.2.1 and is installable with `pip install math5320-portfolio-risk-system`.

## Version History

Version 1.0 delivered the required market-risk engine: historical simulation, parametric delta-normal, Monte Carlo VaR and ES, walk-forward backtesting, and the Streamlit UI. Version 1.1 added credit, CVA, and regulatory extension modules. Version 1.2 added the option-volatility shock mode and substantially expanded test coverage. The final submission version reports 644 passing no-network tests, one intentional skip, 95% statement coverage, and two passing live integration scripts.

## Intended Use

The system is designed for academic analysis by students, instructors, and technically capable analysts working locally through the Streamlit interface or through the Python API. Intended users include students working through the Streamlit interface, instructors or markers reviewing model and validation evidence, analysts importing directly from the `src/` Python package, and notebook users reproducing course cases or testing assumptions interactively.

## Non-Intended Use

The system is **not** intended for production trading, official regulatory filing, CCAR or DFAST submission, enterprise-wide risk aggregation, or any application requiring independent model validation and formal governance controls. Those boundaries are explicit and consistent with the course-level scope.

# Model Risk Management Framework

Following the Lecture 5 framework, the system addresses five pre-deployment model-risk requirements: clear statement of purpose, design documentation, data analysis, testing, and system analysis. The architecture enforces separation of concerns by keeping all quantitative logic in pure Python modules with no Streamlit imports, making independent testing straightforward.

The model choice matrix for the three VaR methods is:

<div class="center">

<div class="tabular">

L2.8cmL4cmL4cmL3cm **Method** & **Strengths** & **Limitations** & **Best for**\
Historical simulation & Nonparametric; captures observed skewness; full repricing & Limited to observed history; window sensitive & Option books; distributional audit\
Parametric delta-normal & Closed-form; fast; analytically clear & Normal assumption; first-order delta approximation & Linear stock portfolios; sensitivity analysis\
Monte Carlo & Full repricing; flexible; convergence-testable & Normal shocks; computational cost; seed dependence & Nonlinear books; scenario analysis\

</div>

</div>

The three-method comparison is itself a model-risk control. Large disagreement between methods is usually telling the user something about nonlinearity, fat-tail exposure, or distributional mismatch.

# Product Description

## Portfolio Payoff and Loss Convention

At evaluation time $`t`$, the portfolio value is:
``` math
\begin{equation}
  V_t = \sum_{i} q_i\, S_{i,t}
      + \sum_{j} n_j\, m_j\,
        \Pi_j\!\left(S_{u(j),t},\, K_j,\, \sigma_j,\, r_j,\, q_j,\, T_j - t\right)
  \label{eq:portfolio_value}
\end{equation}
```
where $`q_i`$ is the number of shares of equity $`i`$, $`n_j`$ the number of contracts for option $`j`$, $`m_j`$ the contract multiplier, $`\Pi_j`$ the Black-Scholes price, $`u(j)`$ the underlying of option $`j`$, and $`K_j,\sigma_j,r_j,q_j,T_j`$ the strike, implied volatility, risk-free rate, dividend yield, and maturity. Negative $`n_j`$ represents a short option position.

The portfolio loss over a horizon of $`h`$ trading days is:
``` math
\begin{equation}
  L = V_0 - V_T, \qquad T = t + h
  \label{eq:loss}
\end{equation}
```
A positive $`L`$ means the portfolio lost value. VaR at confidence level $`\alpha`$ and ES at level $`\alpha_{\mathrm{ES}}`$ are:
``` math
\begin{align}
  \mathop{\mathrm{VaR}}_\alpha &= \inf\bigl\{l : \mathbb{P}(L > l) \leq 1 - \alpha\bigr\}
  \label{eq:var_def}\\[4pt]
  \mathop{\mathrm{ES}}_{\alpha_{\mathrm{ES}}} &= \mathbb{E}\!\left[L \;\big|\; L > \mathop{\mathrm{VaR}}_{\alpha_{\mathrm{ES}}}\right]
  \label{eq:es_def}
\end{align}
```

## Representative Portfolio

The following portfolio, matching the Bloomberg course data used in validation, grounds the analysis throughout this report.

<div class="center">

<div class="tabular">

@L0.20L0.12r L0.12L0.12r@ Position & Type & Quantity & Strike & Maturity & $`\sigma`$\
AAPL equity & Stock & 24,679 shares & n/a & n/a & n/a\
CAT equity & Stock & 171 shares & n/a & n/a & n/a\
AAPL call & Call & $`+10`$ contracts & \$190 & Jun 2026 & 25%\
CAT put & Put & $`-5`$ contracts & \$300 & Dec 2025 & 22%\

</div>

</div>

At reference prices of approximately \$178.50 for AAPL and \$342.60 for CAT, the equity notional is approximately \$4.5M.

# Model Description

## Equity Return Model and Shock Construction

Daily log returns and overlapping $`h`$-day returns are computed as:
``` math
\begin{equation}
  r_{i,t} = \log\!\left(\frac{S_{i,t}}{S_{i,t-1}}\right),
  \qquad
  R_{i,t}^{(h)} = \sum_{k=0}^{h-1} r_{i,t-k}
  \label{eq:returns}
\end{equation}
```
Shocked prices are applied via:
``` math
\begin{equation}
  S_{i,T}^{(\text{shocked})} = S_{i,0}\,e^{R_i^{(h)}}.
\end{equation}
```
The log-return convention keeps shocked prices positive and is consistent with Black-Scholes GBM dynamics.

## Black-Scholes Option Pricing Model

European calls and puts are priced using Black-Scholes with continuous dividend yield:
``` math
\begin{align}
  d_1 &= \frac{\log(S/K) + \bigl(r - q + \tfrac{1}{2}\sigma^2\bigr)T}
              {\sigma\sqrt{T}},
  \qquad d_2 = d_1 - \sigma\sqrt{T}
  \label{eq:bs_d1d2}\\[6pt]
  C   &= S\,e^{-qT} N(d_1) - K\,e^{-rT} N(d_2)
  \label{eq:bs_call}\\[4pt]
  P   &= K\,e^{-rT} N(-d_2) - S\,e^{-qT} N(-d_1)
  \label{eq:bs_put}
\end{align}
```
The option deltas used in the parametric approximation are:
``` math
\begin{equation}
  \Delta_{\mathrm{call}} = e^{-qT} N(d_1),
  \qquad
  \Delta_{\mathrm{put}}  = e^{-qT}\bigl(N(d_1) - 1\bigr).
  \label{eq:bs_delta}
\end{equation}
```

The implied volatility $`\sigma`$ is user-supplied, so the system does not require an option-chain data feed. Two volatility modes are supported. Under `fixed` mode, $`\sigma`$ remains constant across scenarios. Under `underlying_beta` mode, a simplified scenario volatility is applied:
``` math
\begin{equation}
  \sigma'= \max\!\bigl(\sigma_{\mathrm{floor}},\; \sigma \cdot (1 - \beta R)\bigr)
  \label{eq:vol_shock}
\end{equation}
```
where $`R`$ is the underlying log-return scenario and $`\beta`$ is a leverage scaling factor. This is not a full surface model, but it gives a directionally useful way to make adverse equity moves increase implied volatility.

## Historical Simulation VaR and ES

Historical simulation applies overlapping $`h`$-day log-return scenarios to the current portfolio and reprices the full portfolio under each scenario. The empirical loss distribution $`\{L^{(s)}\}`$ is formed from all scenarios in the lookback window. $`\mathop{\mathrm{VaR}}_\alpha`$ is the empirical $`\alpha`$-quantile, and $`\mathop{\mathrm{ES}}_{\alpha_{\mathrm{ES}}}`$ is the mean of losses exceeding the ES threshold. This method is nonparametric and captures skewness and fat tails only to the extent that they are present in the historical sample.

## Parametric Delta-Normal VaR and ES

The parametric engine builds a dollar exposure vector from stock holdings and option deltas:
``` math
\begin{equation}
  x_i^{\mathrm{stock}} = q_i S_{i,0},
  \qquad
  x_j^{\mathrm{option}} = n_j m_j \Delta_j S_{u(j),0}.
  \label{eq:exposure}
\end{equation}
```
Daily mean and covariance $`(\hat{\mu}, \hat{\Sigma})`$ are estimated from historical log returns and scaled to the horizon:
``` math
\hat{\mu}_h = h\hat{\mu}, \qquad \hat{\Sigma}_h = h\hat{\Sigma}.
```
The portfolio mean and standard deviation in dollars are:
``` math
\begin{equation}
  m = \mathbf{x}^{\!\top}\hat{\mu}_h,
  \qquad
  s^2 = \mathbf{x}^{\!\top}\hat{\Sigma}_h\,\mathbf{x}.
  \label{eq:port_moments}
\end{equation}
```
Under the normality assumption:
``` math
\begin{align}
  \mathop{\mathrm{VaR}}_\alpha &= -m + s\,\Phi^{-1}(\alpha)
  \label{eq:parametric_var}\\[4pt]
  \mathop{\mathrm{ES}}_{\alpha_{\mathrm{ES}}} &= -m + s\cdot
    \frac{\phi\!\bigl(\Phi^{-1}(\alpha_{\mathrm{ES}})\bigr)}{1 - \alpha_{\mathrm{ES}}}.
  \label{eq:parametric_es}
\end{align}
```
The system supports separate confidence levels for VaR and ES. Two estimators are available: a rolling-window estimator and an EWMA estimator. The EWMA mean and its recursive update are:
``` math
\begin{equation}
  m_N = (1-\lambda)\sum_{i=0}^{\infty}\lambda^i a_{N-i},
  \qquad
  m_N = (1-\lambda)a_N + \lambda m_{N-1}.
  \label{eq:ewma_course_mean}
\end{equation}
```
The decay parameter follows the project specification convention $`\lambda = (N-1)/(N+1)`$, which gives an effective exponential memory of $(N+1)/2$ observations. For the default $`N=60`$ this yields $`\lambda \approx 0.967`$. This differs from the course / textbook form $`\lambda = 1 - 1/N`$ (which gives $`\lambda \approx 0.983`$ for the same $`N`$ and an effective memory of $`N`$ observations); the specification convention uses a more aggressive decay and reacts more quickly to volatility regime changes.

Both conventions are present in the codebase: `_ewma_lambda(N)` in `src/risk/estimators.py` implements the specification convention and is the active formula used by all production paths. `_ewma_lambda_course(N)` implements the textbook form $`\lambda = 1 - 1/N`$ as a standalone reference function that is not wired into any production path. All risk estimates in this report use the specification convention.

## Monte Carlo VaR and ES

The Monte Carlo engine simulates $`N_{\text{sim}}`$ horizon return vectors:
``` math
\begin{equation}
  \mathbf{R}_h^{(s)} \sim \mathcal{N}\!\left(\hat{\mu}_h, \hat{\Sigma}_h\right),
  \quad s = 1,\ldots,N_{\text{sim}}.
  \label{eq:mc_draws}
\end{equation}
```
For each draw, shocked prices are applied to all underlyings and the full portfolio is repriced. VaR and ES are then computed empirically from the simulated loss distribution. The default is $`N_{\text{sim}}=10{,}000`$ with seed 42 for reproducibility. In walk-forward backtesting, the path count is reduced for computational feasibility.

## Walk-Forward Backtesting

VaR backtesting is implemented as a walk-forward loop. At each evaluation date $`t`$:

1.  Fit the selected risk model using all data up to and including $`t`$.

2.  Forecast the $`h`$-day VaR, $`\widehat{\mathop{\mathrm{VaR}}}_\alpha(t)`$.

3.  Compute the realised $`h`$-day portfolio loss $`L(t,t+h)`$.

4.  Record an exception if $`L(t,t+h) > \widehat{\mathop{\mathrm{VaR}}}_\alpha(t)`$.

The exception indicator is:
``` math
I_t = \mathbf{1}\!\{L(t,t+h) > \widehat{\mathop{\mathrm{VaR}}}_\alpha(t)\}.
```

**Kupiec unconditional coverage test.** At confidence level $`\alpha`$, the expected exception rate is $`p^* = 1-\alpha`$. Kupiec’s likelihood-ratio statistic tests whether the observed exception rate $`\hat{p} = N_e/T`$ is consistent with $`p^*`$:
``` math
\begin{equation}
  \mathrm{LR}_{\mathrm{uc}}
    = -2\log\!\left[\frac{(1-p^*)^{T-N_e}(p^*)^{N_e}}
                        {(1-\hat{p})^{T-N_e}\hat{p}^{N_e}}\right]
    \;\xrightarrow{d}\; \chi^2_1.
  \label{eq:kupiec}
\end{equation}
```
A passing Kupiec test means the exception frequency is statistically reasonable. It does not tell us whether exceptions cluster.

**Christoffersen independence test.** Let $`n_{ij}`$ be the count of transitions from state $`i`$ to state $`j`$, where state 1 means an exception. The test statistic is:
``` math
\begin{equation}
  \mathrm{LR}_{\mathrm{ind}}
    = -2\log\!\left[\frac{(1-\hat{\pi})^{n_{00}+n_{10}}
                          \hat{\pi}^{n_{01}+n_{11}}}
                         {(1-\hat{\pi}_{01})^{n_{00}}\hat{\pi}_{01}^{n_{01}}
                          (1-\hat{\pi}_{11})^{n_{10}}\hat{\pi}_{11}^{n_{11}}}\right]
    \;\xrightarrow{d}\; \chi^2_1.
  \label{eq:christoffersen}
\end{equation}
```
The joint conditional coverage statistic is $`\mathrm{LR}_{\mathrm{cc}} = \mathrm{LR}_{\mathrm{uc}} + \mathrm{LR}_{\mathrm{ind}}`$.

**Basel traffic-light zones.** Under a 99% confidence level over 250 trading days: GREEN if $`N_e \leq 4`$, YELLOW if $`5 \leq N_e \leq 9`$, and RED if $`N_e \geq 10`$.

## Extension Modules

**Exact GBM/lognormal VaR and ES** (`src/risk/lognormal.py`). Under GBM with drift $`\mu`$ and volatility $`\sigma`$, the closed-form VaR for a long position of value $`V_0`$ is:
``` math
\begin{equation}
  \mathop{\mathrm{VaR}}_\alpha^{\mathrm{GBM}} =
  V_0\!\left[1 - \exp\!\left(m_h + s_h z_{1-\alpha}\right)\right],
  \quad
  m_h = \left(\mu - \tfrac{1}{2}\sigma^2\right)h,
  \quad
  s_h = \sigma\sqrt{h}.
  \label{eq:gbm_var}
\end{equation}
```
Short positions require a separate formula because the loss comes from upward price moves.

**Reduced-form hazard model** (`src/credit/hazard.py`). Under constant hazard rate $`\lambda`$:
``` math
\begin{equation}
  s(t) = e^{-\lambda t},
  \qquad
  f(t) = \lambda e^{-\lambda t},
  \qquad
  P(\tau \leq t) = 1 - e^{-\lambda t}.
\end{equation}
```
Piecewise-constant hazard is also implemented.

**Merton structural default model** (`src/credit/merton.py`). Default occurs if firm asset value $`V_T < B`$ at maturity $`T`$:
``` math
\begin{equation}
  \mathrm{PD} = N(-d_2),
  \quad
  d_2 =
  \frac{\log(V_0/B) + (\nu - \tfrac{1}{2}\sigma_A^2)T}
       {\sigma_A\sqrt{T}}.
\end{equation}
```
Setting $`\nu=r`$ gives the Q-measure default probability. Setting $`\nu=\mu`$ gives the P-measure probability.

**CDS par spread** (`src/credit/cds.py`). Under constant hazard and recovery rate $`R`$:
``` math
\begin{equation}
  s_{\mathrm{CDS}} \approx (1-R)\lambda.
\end{equation}
```
For $`\lambda=3\%`$ and $`R=40\%`$, this gives approximately 180 basis points.

**CVA** (`src/credit/cva.py`). Discrete CVA with recovery rate $`R`$ is:
``` math
\begin{equation}
  \mathrm{CVA} = (1-R)\sum_i \bar{E}_i \bar{p}_i,
\end{equation}
```
where $`\bar{E}_i`$ is expected positive exposure and $`\bar{p}_i`$ is the marginal default probability.

**Regulatory capital** (`src/risk/regulatory.py`). The system computes:
``` math
\mathrm{RWA} = \sum_i w_i E_i,
  \qquad
  \kappa = \frac{\text{Equity}}{\text{RWA}}.
```
The illustrative pass threshold is $`\kappa \geq 0.08`$. DFAST-style pathing projects capital ratios through a 9-quarter stress path.

# Software Architecture

## Layered Architecture

The system follows a strict layered architecture:

<div class="center">

</div>

Pure model functions have no Streamlit imports and can be called directly from the test suite or notebooks without running the app.

## Module Inventory

<div class="center">

<div class="tabular">

L2.4cmL3.8cmL2.4cmL2.2cmL2.4cm **Module** & **Purpose** & **Inputs** & **Outputs** & **Test evidence**\
Schemas & Define stock, option, and portfolio objects & User inputs & Structured portfolio & `test_config…`\
Market data & CSV and Yahoo Finance loading & Tickers, dates & Aligned prices & `test_market…`\
Black-Scholes & Option pricing and delta & $`S,K,T,r,q,\sigma`$ & Price, delta & `test_backend.py`\
Portfolio & Aggregate positions and exposures & Portfolio, spots & Value, exposures & `test_backend.py`\
Returns & Log and horizon return construction & Price matrix & Return matrix & `test_backend.py`\
Estimators & Rolling and EWMA covariance & Return matrix & Mean, covariance & `test_backend.py`\
Historical & Historical VaR/ES & Portfolio, history & VaR, ES, losses & `test_backend.py`\
Parametric & Delta-normal VaR/ES & Exposures, covariance & VaR, ES & `test_backend.py`\
Monte Carlo & Simulated VaR/ES & Mean/covariance, portfolio & VaR, ES, losses & `test_backend.py`\
Backtest & Walk-forward VaR validation & History, settings & Exceptions, statistics & `test_backtest…`\
Lognormal & Exact GBM VaR/ES & GBM parameters & Exact VaR, ES & `test_lognormal.py`\
Hazard & Reduced-form default & Hazard, maturity & Survival, spread & `test_credit.py`\
Merton & Structural default & $`V_0,B,T,r,\sigma`$ & PD, equity, debt & `test_credit.py`\
CDS & Par spread & Hazard, recovery & Spread & `test_credit.py`\
CVA & Counterparty adjustment & Exposure, PD & CVA & `test_credit.py`\
Regulatory & RWA, capital, DFAST & Assets, weights & Ratios, stress & `test_regulatory…`\

</div>

</div>

## Key Design Decisions

All quantitative functions are pure. Given the same inputs, they return the same outputs, with no hidden network calls and no UI dependencies. The service layer orchestrates workflows. The UI layer is kept thin and only presents inputs, outputs, charts, and downloads. Credit and regulatory extensions sit behind separate service modules, keeping them independent from the core stock/option market-risk path.

## Application Screenshots

The eight Streamlit tabs below demonstrate the UI workflow. Images are stored in `docs/images/` in the repository root.

<figure data-latex-placement="H">
<p><span><img src="images/01_portfolio_input.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 1: Portfolio Input, stocks and European options editor</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/02_market_data.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 2: Market Data, Yahoo Finance download and CSV upload</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/03_risk_settings.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 3: Risk Settings, lookback, horizon, confidence, EWMA, MC paths</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/04_run_analysis.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 4: Run Analysis, VaR/ES comparison table and loss histogram</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/05_backtesting.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 5: Backtesting, walk-forward exception chart and Kupiec/Christoffersen diagnostics</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/06_credit_risk.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 6: Credit Risk, reduced-form and Merton structural default</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/07_cds_cva.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 7: CDS / CVA, par spread curve and counterparty CVA</figcaption>
</figure>

<figure data-latex-placement="H">
<p><span><img src="images/08_capital_stress.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Tab 8: Capital and Stress, RWA, capital ratio, and DFAST scenarios</figcaption>
</figure>

# Validation Methodology and Test Plan

## Validation Layers

Validation was performed through six complementary layers.

**Analytical golden tests.** Deterministic formulas were compared against hand-calculated or textbook reference values. Black-Scholes pricing was verified against standard benchmark values. Kupiec LR statistics were compared against chi-square critical values. Exact lognormal VaR was compared against closed-form formulas.

**Course-homework fixture tests.** Key scenarios were derived from MATH GR 5320 homework problems and embedded as regression tests in `tests/test_homework_cases.py` and `tests/test_course_validation.py`. If the implementation drifts from the course formulas, the tests fail directly.

**Numerical precision and failure-mode tests.** The final suite adds NP_01–NP_07 in `tests/test_numerical_precision.py`. These tests cover IEEE 754-style floating-point issues, extreme Black-Scholes inputs, log-return cancellation, near-singular covariance matrices, EWMA stability, and extreme-confidence VaR/ES. This is where Goldberg’s floating-point guidance is most relevant .

**Behavioural, convergence, and inversion tests.** The final suite also adds BEH_01–BEH_08, CONV_01, INV_01–INV_02, PNL_01, and HEDGE_01. These tests check monotonicity, put-call parity, no-arbitrage bounds, ES/VaR ordering, Monte Carlo convergence, Merton inversion, Kupiec p-values, linear P&L attribution, and a one-day delta-hedge check.

**Integration tests.** Two live-data scripts (`integration_test.py` and `integration_test_formula_sheet.py`) exercise full end-to-end workflows against Yahoo Finance data. Both scripts passed.

**Walk-forward backtesting.** VaR backtests were run on a 1,500-row AAPL/CAT price panel, producing 990 backtest observations at a 5-day horizon with 99% VaR confidence.

## Benchmark Comparisons

Three formal benchmark comparisons are included.

**Parametric vs. historical.** For a simple equity-only portfolio, the two methods should agree within approximately 10%; larger differences are documented when the historical distribution departs from normality.

**Monte Carlo vs. exact GBM.** For a single-asset position, Monte Carlo VaR should converge to exact lognormal GBM VaR as path count increases. The CONV_01 run records a clear reduction in error as paths increase from 500 to 5,000 to 50,000.

**CDS approximation vs. full formula.** The approximation $`s \approx (1-R)\lambda`$ is checked against the full par-spread formula. At $`\lambda=3\%`$ and $`R=40\%`$, the approximation gives 180 basis points.

## Acceptance Criteria

All required no-network tests should pass with zero failures. The suite may include explicitly documented intentional skips. Statement coverage across `src/` should reach at least 95%. ES must be at least VaR when evaluated at the same confidence level. Black-Scholes price and delta must agree with analytical benchmarks within standard numerical tolerance. Benchmark comparisons must be interpretable and consistent with the model assumptions.

# Validation Results

## Unit Test Suite Results

Observed terminal output:

<div class="shellcode">

644 passed, 1 skipped, 242 warnings in 14.95s

</div>

The no-network suite passed 644 tests with 0 failures and 1 intentional skip. The warnings are deprecation notices in third-party libraries and do not affect the test results. The skipped test is an older placeholder for a Merton inversion path; the implemented `INV_01` round-trip test passes separately.

Selected test distribution:

<div class="center">

<div class="tabular">

L3.5cmL5cmr **Group** & **Files** & **Count**\
Core backend & `test_backend.py` & 49\
Numerical precision & `test_numerical_precision.py` & 7\
Backtest extensions & `test_backtest_extensions.py` & 31\
Course validation & `test_course_validation.py` & 67\
Homework fixtures & `test_homework_cases.py` & 83\
Lognormal & `test_lognormal.py` & 34\
Credit & `test_credit.py` + mitigants + timing & 118\
Credit service & `test_credit_service.py` & 11\
Regulatory & `test_regulatory.py` + balance sheet + DFAST & 46\
Market data & `test_market_data.py` & 25\
Config / validation & `test_config_and_validation.py` + namespace & 22\
Charts & `test_charts.py` & 6\
UI panels & `test_ui_panels.py` & 68\
Coverage / numerics & `test_coverage_gaps.py` + strict numerics + ES split & 77\
**No-network passed tests** & & **644**\

</div>

</div>

<figure data-latex-placement="H">
<p><span><img src="images/advanced_tab5_backtesting.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Backtesting tab: walk-forward exception sequence, Kupiec/Christoffersen results, Basel traffic-light zone</figcaption>
</figure>

## Statement Coverage

Total statement coverage is **95%**. The remaining uncovered lines are concentrated in UI display branches, secondary credit and regulatory branches, selected historical vol-shock paths, and defensive validation branches. These gaps do not change the core validation conclusion: the pricing formulas, VaR/ES engines, covariance handling, backtesting diagnostics, credit formulas, and regulatory arithmetic are all tested directly.

## Analytical Golden Test Results

<div class="center">

<div class="tabular">

L1.5cmL2.2cmL3.8cmL3.8cmL2cml **Case** & **Module** & **Expected** & **Actual** & **Abs. err.** & **Pass**\
BS_01 & B-S call & 10.4505835722 & 10.4505835722 & $`\approx 0`$ & Yes\
BS_02 & B-S put & 5.5735260223 & 5.5735260223 & $`\approx 0`$ & Yes\
LN_01 & Long GBM VaR & 3720.342013894248 & 3720.342013894248 & 0 & Yes\
LN_02 & Short GBM VaR & 5924.434136581646 & 5924.434136581646 & 0 & Yes\
HZ_01 & Hazard $`s(5)`$ & 0.9636761353 & 0.9636761353 & $`\approx 0`$ & Yes\
MR_01 & Merton Q-PD & 0.2952952345 & 0.2952952345 & $`\approx 0`$ & Yes\
CDS_01 & CDS spread & 0.0180 & 0.0180 & 0 & Yes\
CVA_01 & Discrete CVA & 1.92 & 1.92 & 0 & Yes\
REG_01 & Capital ratio & 0.12 & 0.12 & 0 & Yes\

</div>

</div>

## Backtesting Results and Interpretation

Walk-forward VaR backtesting was run on the AAPL/CAT Bloomberg panel at a 5-day horizon and 99% VaR confidence.

<div class="center">

<div class="tabular">

L2.2cmL1.5cmL1.3cmL1.2cmL1.5cmL1.5cmL1.5cmL1.5cm **Model** & **Horizon** & **Conf.** & **Obs.** & **Exp. exc.** & **Act. exc.** & **Exc. rate** & **Kupiec $`p`$**\
Historical & 5d & 0.99 & 990 & 9.90 & 15 & 1.52% & 0.130\

</div>

</div>

Additional diagnostics:

- Kupiec $`\mathrm{LR}_{\mathrm{uc}}`$: **2.2920**

- Christoffersen independence LR: **62.2015**

- Christoffersen independence $`p`$-value: $`3.10 \times 10^{-15}`$

- Conditional coverage LR: **64.4936**

- Conditional coverage $`p`$-value: $`9.89 \times 10^{-15}`$

- Basel zone: **RED**

- Average exception gap: \$205,833.28

- Maximum exception loss: \$1,262,636.56

The Kupiec test does not reject unconditional coverage at the 5% level: 15 exceptions out of 990 observations is higher than the expected 9.90, but the $`p`$-value of 0.130 is not small enough to reject. The Christoffersen result is different. The independence test strongly rejects, meaning exceptions cluster in time. That is exactly the type of weakness expected from rolling-window historical simulation during volatility regime changes. The model can pass an exception-count test while still failing a clustering test.

The Basel RED classification reflects the exception count mechanically. This result is useful rather than cosmetic: it shows why both Kupiec and Christoffersen diagnostics are needed.

# Numerical Precision, Behavioural, Convergence, and Attribution Results

## Numerical Precision Results: NP_01–NP_07

<div class="center">

<div class="tabular">

L1.7cmL5.2cmL6.2cm **Test ID** & **Observed result** & **Interpretation**\
NP_01 & Black-Scholes call at $`\sigma = 10^{-10}`$ returned 4.87705755; discounted intrinsic value was 4.87705755. & Low-volatility limiting case passed to displayed precision.\
NP_02 & Black-Scholes call at $`\sigma = 50`$ returned 100.00000000. & Extreme-volatility value remained finite and did not exceed the spot scale.\
NP_03 & Near-zero maturity call at $`T = 10^{-8}`$ returned 10.00000005. & Near-expiry price converged to intrinsic value without NaN or infinity.\
NP_04 & Tiny price increment produced log return $`9.9920 \times 10^{-13}`$, followed by $`-9.9920 \times 10^{-13}`$. & Catastrophic cancellation did not collapse the return to zero.\
NP_05 & Near-singular covariance VaR returned 1446.1278; ES returned 1453.3279. & Parametric VaR remained finite and positive under a near-singular covariance setting.\
NP_06 & EWMA covariance eigenvalues were $`3.0261 \times 10^{-4}`$ and $`3.9396 \times 10^{-4}`$. & EWMA covariance remained finite, symmetric, and positive semidefinite.\
NP_07 & Extreme-confidence parametric VaR at 99.99% returned 1895.2176; ES returned 2022.6418. & Extreme-tail VaR/ES remained finite, positive, and satisfied ES $`\geq`$ VaR.\

</div>

</div>

The extreme Black-Scholes tests confirm that `scipy.stats.norm.cdf` behaves safely at machine limits. When $`d_1`$ or $`d_2`$ is extremely large in magnitude, the normal CDF saturates to 0 or 1 rather than producing NaN or infinity. That is the desired behaviour for these limiting cases.

For the near-singular covariance case, the test logic includes an explicit regularisation fallback:
``` math
\Sigma_{\mathrm{reg}} = \Sigma + 10^{-8} I.
```
In the observed run, the constructed covariance was already positive definite, so the jitter branch was not needed. The fallback still matters because it states how the engine should behave if a nearly singular matrix crosses the numerical boundary.

## Behavioural Confirmation Results: BEH_01–BEH_08

<div class="center">

<div class="tabular">

L1.7cmL5.2cmL6.2cm **Test ID** & **Observed result** & **Interpretation**\
BEH_01 & Call prices increased as spot increased from 80 to 120. & Spot monotonicity passed.\
BEH_02 & Call prices decreased as strike increased from 80 to 120. & Strike monotonicity passed.\
BEH_03 & Call prices increased as volatility increased from 10% to 40%. & Vega sign behaved correctly.\
BEH_04 & Put-call parity residual was $`0.0`$, below $`10^{-10}`$. & Internal pricing consistency passed: $`C-P = S e^{-qT} - K e^{-rT}`$.\
BEH_05 & Low-volatility call with $`S=110`$, $`K=100`$, $`\sigma=10^{-8}`$ returned approximately 10.0000. & Volatility-to-zero limiting case passed.\
BEH_06 & All tested calls satisfied $`C \geq \max(S e^{-qT} - K e^{-rT},0)`$. & No-arbitrage lower-bound check passed.\
BEH_07 & Historical, parametric, and Monte Carlo methods all satisfied ES $`\geq`$ VaR at matched confidence. & Risk-measure ordering passed.\
BEH_08 & Historical VaR was finite and positive on the representative two-stock fixture. & Historical VaR positivity passed.\

</div>

</div>

For BEH_04, with $`S=100`$, $`K=100`$, $`r=5\%`$, $`q=2\%`$, $`\sigma=25\%`$, and $`T=1`$, the observed call price was 11.12376193, the put price was 8.22683705, and:
``` math
(C-P) - (S e^{-qT} - K e^{-rT}) = 0.0.
```
This is stronger than the required $`10^{-10}`$ tolerance.

The extreme-confidence risk result is also finite: the 99.99% VaR test returned VaR = 1895.2176 and ES = 2022.6418. This confirms that the tail-probability logic remains stable at the tested extreme confidence level.

## Convergence and Inversion Results

<div class="center">

<div class="tabular">

L2cmL4.5cmL6.5cm **Test ID** & **Observed result** & **Interpretation**\
CONV_01 & MC VaR estimates: $`N=500`$: 1044.2859; $`N=5{,}000`$: 1157.5628; $`N=50{,}000`$: 1149.1737. & Fine-grid error was much smaller than coarse-grid error.\
INV_01 & Merton `implied_B` recovered $`B=80.000000`$ from the target survival probability. & Round-trip inversion error was below $`10^{-6}`$.\
INV_02 & Kupiec p-value was 0.4812 in the near-expected exception-count check. & The test correctly did not reject unconditional coverage.\

</div>

</div>

For CONV_01:
``` math
|\mathrm{VaR}_{5k} - \mathrm{VaR}_{500}|
  =
  |1157.5628 - 1044.2859|
  =
  113.2769,
```
and:
``` math
|\mathrm{VaR}_{50k} - \mathrm{VaR}_{5k}|
  =
  |1149.1737 - 1157.5628|
  =
  8.3891.
```
The observed error ratio was:
``` math
\frac{8.3891}{113.2769} = 0.0741.
```
This is comfortably below the rough Monte Carlo scaling benchmark:
``` math
\frac{1}{\sqrt{10}} \approx 0.316.
```
The realised ratio need not equal 0.316 because these are realised quantile estimates under a fixed seed, but the direction and magnitude are consistent with convergence.

For INV_01, the Merton test used $`V_0=100`$, $`B=80`$, $`r=5\%`$, $`\sigma=30\%`$, and $`T=1`$. The computed default probability was 0.2234843, target survival was 0.7765157, and the inverted barrier was 80.000000.

## P&L Attribution and Hedge Effectiveness

<div class="center">

<div class="tabular">

L2cmL5.5cmL5.5cm **Test ID** & **Observed result** & **Interpretation**\
PNL_01 & Maximum absolute residual for the linear stock portfolio was $`2.61 \times 10^{-12}`$; mean residual was $`5.41 \times 10^{-15}`$. & Residual is zero up to floating-point rounding.\
HEDGE_01 & For a +1% ATM call shock, unhedged $`|\mathrm{P\&L}|=0.57422`$ and hedged $`|\mathrm{P\&L}|=0.03398`$. & Delta hedge reduced P&L magnitude.\
HEDGE_01 & For a -1% ATM call shock, unhedged $`|\mathrm{P\&L}|=0.50564`$ and hedged $`|\mathrm{P\&L}|=0.03460`$. & Delta hedge reduced P&L magnitude.\

</div>

</div>

For PNL_01, the portfolio is purely linear in the two stock prices:
``` math
\Delta V_t = \sum_i q_i (S_{i,t+1} - S_{i,t}).
```
The observed residual is effectively zero.

For HEDGE_01, the initial one-month ATM call price was 2.512067 and the Black-Scholes delta was 0.540239. Under a +1% shock, the option P&L was 0.574217 and the delta-hedged net P&L was 0.033978. Under a -1% shock, the option P&L was -0.505635 and the delta-hedged net P&L was 0.034604. In both cases:
``` math
|\mathrm{P\&L}_{\mathrm{hedged}}|
  <
  |\mathrm{P\&L}_{\mathrm{option}}|.
```
This is a one-day local delta-hedge test. It does not claim that the project implements a production dynamic hedging engine.

# Additional Informative Test Cases

## Short VaR Exceeds Long VaR

For the same GBM parameters (HW9 regression: $`V_0=100{,}000`$, $`\mu=0.10`$, $`\sigma=0.25`$, $`h=5`$, $`\alpha=0.99`$), the short GBM VaR (5924.43) exceeds the long GBM VaR (3720.34) by approximately 59%. This is a structural property of the lognormal distribution.

A long position loses at most $`V_0`$. A short position has theoretically unbounded adverse moves. The asymmetry of the lognormal distribution therefore produces a larger upper quantile for the short-loss than for the long-loss.

## Merton Q-PD vs. P-PD Divergence

For the HW7 Merton case ($`V_0=100`$, $`B=80`$, $`r=0.05`$, $`\mu=0.10`$, $`\sigma=0.25`$, $`T=5`$), Q-PD = 0.2953 and P-PD = 0.3888. The distinction matters because risk-neutral probabilities are used for pricing, while physical probabilities are used for real-world risk assessment.

## ES is Always at Least VaR

The ordering ES $`\geq`$ VaR is checked directly across the three model families at matched confidence levels. Treating this as a test rather than an assumption is useful because confidence-level handling bugs can otherwise slip through silently.

# Limitations and Model Risk

The following limitations are deliberate and documented.

1.  **Fixed implied volatility.** Under `fixed` mode, $`\sigma`$ is held constant across scenarios. In practice, implied volatility often rises when the underlying falls. The `underlying_beta` mode gives a simple directional improvement, but a production model would need a full implied-volatility surface.

2.  **First-order delta-normal approximation.** The parametric method linearises option payoffs at current delta. For large moves, near-expiry options, or short-gamma portfolios, it can understate VaR. A delta-gamma extension would improve this without changing the rest of the architecture.

3.  **Multivariate normal Monte Carlo shocks.** Increasing the number of paths improves Monte Carlo precision, but it does not fix the normality assumption. Fat tails and left skewness would require a different shock distribution.

4.  **Historical simulation window sensitivity.** Short windows respond quickly but can be noisy. Long windows are stable but slower to adapt to volatility regime changes. The backtesting evidence shows this tradeoff clearly.

5.  **Exception clustering.** The historical model produces clustered exceptions in the representative backtest. The EWMA estimator is a natural next step because it down-weights stale observations.

6.  **Merton default timing.** The standard Merton model recognises default only at maturity. A Black-Cox barrier model would handle continuous default monitoring.

# Conclusions and Recommendations

## Validation Opinion

Based on the 644 passing no-network tests, one intentional skip, 95% statement coverage, two passing live-data integration scripts, walk-forward backtesting evidence, and analytical benchmark comparisons, we issue the following opinion.

**Approved with limitations for intended academic use.** The MATH5320 Portfolio Risk Management System correctly implements historical simulation, parametric delta-normal, and Monte Carlo VaR and ES for mixed equity and European option portfolios. Black-Scholes pricing, delta-normal exposure construction, EWMA and rolling-window estimation, walk-forward backtesting with Kupiec and Christoffersen diagnostics, exact lognormal VaR/ES, hazard and Merton credit models, CDS and CVA pricing, counterparty mitigation, and regulatory capital calculations are implemented and tested against analytical benchmarks and course-homework fixtures. The system is suitable for MATH GR 5320 risk calculations and educational analysis.

The system is **not suitable** for production trading, regulatory filing, official CCAR or DFAST submission, or enterprise-wide risk aggregation.

## Recommendations

1.  **Add delta-gamma parametric VaR.** A quadratic approximation would improve option-heavy portfolio risk estimates while keeping the method fast.

2.  **Use EWMA inside walk-forward backtesting.** A shorter effective EWMA window should reduce exception clustering in stress periods and improve conditional coverage.

3.  **Add a richer volatility-shock model.** The current `underlying_beta` mode is useful, but a skew-aware or VIX-linked volatility rule would be more realistic for vega-heavy portfolios.

4.  **Add Black-Cox structural credit.** Continuous default monitoring is the natural extension to the current Merton model.

5.  **Add browser-level UI testing.** A Playwright or Selenium harness would close most of the remaining Streamlit coverage gap.

# Bibliography

<div class="thebibliography">

10

Harvey J. Stein. *Model Validation Report Template*. Bloomberg Enterprise Risk, November 2015.

Harvey J. Stein. *Model Validation Municipal Bonds*. Bloomberg Enterprise Risk, 2014.

David Goldberg. “What Every Computer Scientist Should Know About Floating-Point Arithmetic.” *ACM Computing Surveys*, 23(1):5–48, 1991.

John C. Hull. *Options, Futures, and Other Derivatives*, 10th ed. Pearson, 2018.

Paul H. Kupiec. “Techniques for Verifying the Accuracy of Risk Measurement Models.” *Journal of Derivatives*, 3(2):73–84, 1995.

Peter F. Christoffersen. “Evaluating Interval Forecasts.” *International Economic Review*, 39(4):841–862, 1998.

Robert C. Merton. “On the Pricing of Corporate Debt: The Risk Structure of Interest Rates.” *Journal of Finance*, 29(2):449–470, 1974.

Fischer Black and Myron Scholes. “The Pricing of Options and Corporate Liabilities.” *Journal of Political Economy*, 81(3):637–654, 1973.

Alexander J. McNeil, Rüdiger Frey, and Paul Embrechts. *Quantitative Risk Management: Concepts, Techniques and Tools*, revised ed. Princeton University Press, 2015.

Columbia MATH GR 5320. *Project Requirements*. Course reference document, Spring 2026.

</div>
