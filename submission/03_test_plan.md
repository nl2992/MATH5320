<div class="titlepage">

**Test Plan**

MATH GR 5320 Portfolio Risk Management System

Columbia University, Financial Risk Management, Spring 2026

<div class="tabular">

L4cmL9cm **Field** & **Value**\
Deliverable & 3 of 5 (20 points)\
Authors & Nigel Li, Michael Adegbite, Stella\
Reference Commit & `a4aa9e9` (main branch, May 2026)\
Submission Date & May 13, 2026\
Python Version & 3.12.2\
Test Suite & 644 tests, 1 skipped, 0 failures (see Deliverable 5)\
Statement Coverage & 95%\

</div>

<div class="minipage">

*This test plan formalizes the validation program for the MATH5320 risk engine. It treats testing as part of model development and model governance, aligned with Lecture 5’s emphasis on pre-deployment model risk management.*

</div>

</div>

# Executive Summary

This test plan addresses four questions that every model validation program must answer:

1.  What is being tested?

2.  Why is it being tested?

3.  Against what benchmark is it being tested?

4.  What constitutes acceptable behavior?

The `MATH5320` repository contains a broad test suite covering formula correctness, portfolio valuation, VaR/ES calculations, backtesting, market-data loading, UI behavior, and course-formula extensions. The plan below formalizes those tests into a model-risk-oriented validation program. Testing is treated as a first-class component of model development and model governance, not merely as proof that the application executes without crashing.

The plan also includes a dedicated benchmark-comparison section (Section 11) that evaluates our implementations against external analytical references, a requirement highlighted in the Lecture 5 model validation framework.

# Test Objectives

The test plan validates the following properties:

1.  Correctness of mathematical formulas;

2.  Correctness of portfolio valuation and option repricing;

3.  Correctness of VaR/ES calculations under historical, parametric, and Monte Carlo methods;

4.  Correctness of backtesting and exception logic;

5.  Correctness of credit and regulatory formula-sheet extensions;

6.  Correct handling of invalid inputs, edge cases, and numerical failure modes;

7.  Correct service-layer integration and UI behaviour;

8.  Reproducibility through deterministic fixtures, seeded Monte Carlo, and coverage reporting.

These objectives align with Lecture 5’s emphasis on testing as part of model development and validation, rather than mere application execution.

# Scope of Testing

**In scope:** pure pricing formulas; portfolio valuation and aggregation; return and covariance estimation; historical VaR/ES; parametric VaR/ES; Monte Carlo VaR/ES; VaR backtesting and exception diagnostics; course-formula extensions in `src/risk/lognormal.py`, `src/credit/`, and `src/risk/regulatory.py`; market-data loading and validation; and Streamlit UI rendering.

**Out of scope:** production deployment hardening; enterprise access control; performance benchmarking at scale; full external market-data certification; and production volatility-surface validation.

# Test Environment

- Repository root: `MATH5320`

- Python: `3.12.2`; OS: macOS / Darwin arm64

- Key packages: `streamlit 1.37.1`, `numpy 1.26.4`, `pandas 3.0.2`, `scipy 1.17.1`, `plotly 5.24.1`, `yfinance 1.2.0`, `pytest 7.4.4`, `pytest-cov 7.1.0`

Environment snapshot commands:

<div class="shellcode">

git rev-parse HEAD
python --version
pip freeze > test_artifacts/requirements_freeze.txt

</div>

Network requirements: the no-network unit suite runs without external downloads. Live-data integration scripts require network access and are explicitly separated. All Monte Carlo regression tests use fixed seeds; if a seed is randomized for exploratory work, it is logged.

# Test Data and Fixtures

## Synthetic Fixtures

Used for deterministic unit tests: synthetic price histories, toy two-stock portfolios, simple option positions, and deterministic exposures and covariance matrices.

## Course-Derived Fixtures

Used for validation against homework and formula-sheet results: exact GBM/lognormal values, hazard-rate survival and spread values, Merton Q/P default probabilities and valuations, CDS and CVA examples, and regulatory RWA and capital-ratio examples.

## Bloomberg Course Data

Observed local files: `data/AAPL-bloomberg.csv` and `data/CAT-bloomberg.csv`. These support the AAPL/CAT course portfolio notebooks and provide course-accepted regression anchors.

## Live Market Data

Used by integration scripts: Yahoo Finance downloads for equities and rate proxies via the cached download layer in `src/data/market_data.py`.

## Data Proxy Policy

Consistent with Lecture 5’s requirement to document data quality, proxies, and cleaning assumptions:

- CSV-based Bloomberg fixtures are treated as the primary course acceptance data.

- Yahoo Finance is treated as a convenience data source, not a gold-standard benchmark.

- Any cleaning, alignment, or row-dropping is performed explicitly and logged.

# Test Categories

<div class="center">

<div class="tabular">

L3.5cmL5cmL5cm **Category** & **Purpose** & **Example**\
Unit tests & Validate pure functions & Black-Scholes price, hazard survival\
Analytical goldens & Compare against closed-form values & Exact GBM VaR/ES\
Homework fixtures & Course-derived regression values & AAPL/CAT VaR, Merton Q/P\
External benchmarks & Independent comparison source & Option-calculator Black-Scholes, Basel traffic light\
Edge cases & Validate boundary behaviour & Zero hazard, zero volatility\
Failure-mode tests & Ensure controlled errors & Invalid confidence, missing data\
Behavioural tests & Financial monotonicity and logic & Option price increases with vol\
Convergence tests & Numerical stability & MC VaR converging as simulation count increases\
Backtesting tests & VaR forecast logic & Exceptions, Kupiec LR, Christoffersen LR\
Data validation tests & Input quality checking & Missing or stale prices\
Integration tests & Full workflow & Portfolio input to risk output\
UI tests & Streamlit panel behavior & Portfolio editor, settings, results\
Coverage tests & Source execution breadth & Coverage report and missing-line review\

</div>

</div>

# Module-Level Test Matrix

<div class="center">

<div class="tabular">

L5cmL8.5cm **Module** & **Required tests**\
`pricing/black_scholes.py` & Price, delta, put-call parity, invalid inputs, monotonicity\
`portfolio/positions.py` & Stock value, option value, delta exposure, long/short signs\
`portfolio/portfolio.py` & Aggregate value, exposure vector, empty portfolio rejection\
`risk/returns.py` & Log returns, overlapping returns, horizon summation\
`risk/estimators.py` & Rolling mean/cov, EWMA, covariance symmetry\
`risk/historical.py` & Historical VaR/ES, log shock, missing data\
`risk/parametric.py` & Normal VaR/ES, covariance aggregation, ES confidence separation\
`risk/monte_carlo.py` & Seeded reproducibility, MC VaR/ES, covariance validation\
`risk/backtest.py` & Exceptions, Kupiec, Christoffersen, traffic light, severity\
`risk/lognormal.py` & Exact long/short GBM VaR/ES\
`credit/hazard.py` & Constant and piecewise hazard, survival, density, risky ZCB\
`credit/merton.py` & Q/P PD, equity/debt, target-survival inversion\
`credit/cds.py` & Approximation and full par spread\
`credit/cva.py` & EPE, CVA, discounted CVA\
`credit/mitigation.py` & Netting, collateral, CSA logic\
`risk/regulatory.py` & RWA, capital ratio, DFAST pathing\
`services/risk_engine_service.py` & Orchestration and result-object consistency\
`ui/*.py` & Streamlit input handling and result rendering\
`tests/test_numerical_precision.py`& 7 tests covering IEEE 754 numerical precision and failure modes\

</div>

</div>

# Analytical Golden Tests

## Black-Scholes

Required test family:

- **BS_01** call price against known analytical value

- **BS_02** put price against known analytical value

- **BS_03** put-call parity: $`C - P = S e^{-qT} - K e^{-rT}`$

- **BS_04** call delta $`\in [0,1]`$

- **BS_05** put delta $`\in [-1,0]`$

- **BS_06** option price increases with volatility (vega $`> 0`$)

- **BS_07** invalid maturity or volatility raises controlled exception

Acceptance criterion: numerical values agree within analytic tolerance; parity holds within floating-point tolerance; invalid domains fail loudly.

## Exact Lognormal VaR/ES

Required test family:

- **LN_01** long VaR: $`V_0[1 - e^{m_h + s_h z_{1-\alpha}}]`$

- **LN_02** long ES: closed-form formula with $`\mathcal{N}(z_{1-\alpha} - s_h)/(1-\alpha)`$

- **LN_03** short VaR: $`V_0[e^{\mu h + z_\alpha \sigma\sqrt{h}} - 1]`$

- **LN_04** short ES: closed-form short formula

- **LN_05** VaR scales linearly with notional

- **LN_06** short VaR exceeds long VaR for identical base inputs

- **LN_07** zero horizon gives zero-risk limit or controlled rejection

## Normal Parametric VaR/ES

Required test family:

- **NORM_01** VaR formula: $`-\mu_h^\top x + \sqrt{x^\top \Sigma_h x} \cdot \Phi^{-1}(\alpha)`$

- **NORM_02** ES formula: analogous closed-form integral

- **NORM_03** ES $`\geq`$ VaR when using the same confidence level

- **NORM_04** covariance aggregation for multi-asset portfolio

- **NORM_05** perfectly offsetting exposures reduce risk

- **NORM_06** invalid covariance is rejected or handled explicitly

# Homework-Derived Regression Fixtures

The repository contains substantial homework-derived cases that function as formal regression fixtures. These provide direct numerical anchors, exactly the kind of evidence Lecture 5 encourages.

<div class="center">

<div class="tabular">

L4.5cmL3cmL6cm **Case ID** & **Area** & **Expected validation**\
`HW4_SINGLE_STOCK` & GBM VaR & 5-day 99% VaR near homework value\
`HW4_TWO_STOCK` & Parametric covariance VaR & Correct mean/variance/correlation aggregation\
`HW6_EWMA` & Rolling/EWMA estimation & Window and EWMA parameter behavior\
`HW6_HAZARD_CONST` & Constant hazard & Survival/default probability regression\
`HW6_HAZARD_PIECEWISE`& Piecewise hazard & $`\lambda(t)`$, $`\Lambda(t)`$, $`s(t)`$, $`p(t)`$, spread table\
`HW7_MERTON_QP` & Merton & Q vs P PD comparison\
`HW7_MERTON_TIMING` & Merton timing & Zero default probability before maturity interval\
`HW8_CDS` & CDS & Constant-hazard approx spread and full par spread\
`HW8_CVA` & CVA & Exposure and default-probability aggregation\
`HW9_MERTON_INVERSION`& Merton inversion & Target-survival inversion\
`HW9_SHORT` & Short-risk formulas & Short VaR/ES sign and magnitude behavior\
`HW10_RWA` & Regulatory & RWA and capital-ratio arithmetic\
`HW10_DFAST` & Regulatory & 9-quarter stress-path structure\

</div>

</div>

# External Benchmark Tests

External benchmarks are not substitutes for homework fixtures. They provide a second independent source of validation confidence.

## Option Calculator Benchmark Cases

<div class="center">

<div class="tabular">

L2.2cmrrrrrrl **Case** & $`S`$ & $`K`$ & $`T`$ & $`r`$ & $`q`$ & $`\sigma`$ & **Type**\
ATM call & 100 & 100 & 1.0 & 0.05 & 0.00 & 0.20 & call\
ATM put & 100 & 100 & 1.0 & 0.05 & 0.00 & 0.20 & put\
Dividend call & 100 & 105 & 2.0 & 0.03 & 0.02 & 0.25 & call\
ITM put & 90 & 100 & 0.5 & 0.04 & 0.01 & 0.30 & put\
Near-expiry & 100 & 100 & $`1/252`$ & 0.05 & 0.00 & 0.20 & call\

</div>

</div>

Acceptance criterion: price and delta agree within standard numerical rounding tolerance of an external Black-Scholes calculator.

## Basel Traffic-Light Benchmark

If the Basel-zone helper is present, required assertions are:

- `basel_zone(0, 250, 0.99) == green`

- `basel_zone(4, 250, 0.99) == green`

- `basel_zone(5, 250, 0.99) == amber`

- `basel_zone(9, 250, 0.99) == amber`

- `basel_zone(10, 250, 0.99) == red`

## DFAST Structural Benchmark

We do not claim Federal Reserve replication. The structural test checks: 9-quarter stress path; baseline / adverse / severely adverse scenario naming; Tier 1 capital path; RWA path; and minimum capital ratio threshold.

# MRM Benchmark Model Comparison

A model validation program following Lecture 5’s framework requires comparison of the model under review against an independent reference or benchmark approach. We perform three benchmark comparisons.

## Parametric vs. Historical Benchmark

The parametric delta-normal model serves as the primary benchmark for the historical simulation model, and vice versa. For any given portfolio and data window, we verify:

- Both models produce positive VaR and ES.

- ES $`\geq`$ VaR at the same confidence level for both methods.

- For a simple single-stock equity-only portfolio (no options), parametric and historical VaR agree within a pre-specified tolerance of approximately 10%. Larger differences are expected and explicitly documented when the historical return distribution departs significantly from normality.

- Adding options consistently increases both historical and MC VaR relative to the corresponding linear (parametric) estimate when significant nonlinearity is present.

This comparison is implemented in `tests/test_backend.py` and `tests/test_course_validation.py`, and the results are recorded in `submission/test_artifacts/official_benchmark_results.csv`.

## Monte Carlo vs. Exact GBM Benchmark

For a single-asset position with no options, Monte Carlo VaR should converge to the exact GBM/lognormal closed-form VaR as the simulation count increases. We verify:

- At $`n_{\mathrm{sims}} = 100\,000`$ with a fixed seed, Monte Carlo 99% VaR agrees with `var_long_lognormal` within 2% for a representative single-stock case.

- Convergence is monotone in expectation as $`n_{\mathrm{sims}}`$ grows.

This benchmark is implemented in `tests/test_lognormal.py` and `tests/test_coverage_gaps.py`.

## CDS Approximation vs. Full Par Spread Benchmark

The constant-hazard CDS par-spread approximation $`s_{\mathrm{CDS}} \approx (1-R)\lambda`$ is benchmarked against the full discrete-payment formula from `cds_par_spread`. We verify:

- For $`\lambda = 3\%`$ and $`R = 40\%`$, the approximation yields approximately $`180`$ basis points, consistent with the §14 course landmark.

- The full formula produces a result within 5% of the approximation for short tenors and flat hazard curves.

- The full formula diverges predictably from the approximation as the tenor increases or the hazard rate increases, confirming that the approximation error is understood and bounded.

This benchmark is implemented in `tests/test_credit.py` and `tests/test_course_validation.py`.

# Numerical Precision and Failure-Mode Testing

Quantitative developers need to understand both the strengths and the weaknesses of the numerical methods they use. The test suite therefore includes a dedicated numerical-precision file, `tests/test_numerical_precision.py`, designed to address floating-point failure modes discussed in the lectures. The relevant numerical reference is Goldberg’s discussion of IEEE 754 floating-point arithmetic .

These tests are not intended to prove that the model is accurate for every market regime. Their purpose is narrower and more technical: they confirm that the implementation remains finite, stable, and directionally sensible under cases that commonly expose floating-point errors, catastrophic cancellation, unstable covariance calculations, and extreme-tail VaR/ES evaluation.

<div class="longtable">

L2.2cmL4.4cmL6.2cm **Test ID** & **Failure mode** & **Validation purpose**\
**Test ID** & **Failure mode** & **Validation purpose**\
`NP_01` & Black-Scholes underflow at very low volatility & Confirms that an at-the-money call with $`\sigma = 10^{-10}`$ returns a finite value close to discounted intrinsic value rather than NaN or infinity.\
`NP_02` & Black-Scholes extreme high volatility & Confirms that a call with $`\sigma = 50`$ remains finite, positive, and bounded by the underlying spot scale.\
`NP_03` & Near-zero time to maturity & Confirms that a deep in-the-money call with $`T = 10^{-8}`$ prices close to intrinsic value and does not fail from division by a very small denominator.\
`NP_04` & Catastrophic cancellation in log returns & Confirms that very small price increments produce finite nonzero log returns of the correct order of magnitude.\
`NP_05` & Near-singular covariance in VaR & Confirms that a highly correlated, nearly singular covariance setting does not crash the parametric VaR calculation and returns finite positive VaR.\
`NP_06` & EWMA long-series stability & Confirms that EWMA mean and covariance over a 2000-day series remain finite, symmetric, and positive semidefinite up to numerical tolerance.\
`NP_07` & Extreme-confidence VaR/ES & Confirms that parametric VaR and ES at $`\alpha = 0.9999`$ remain finite, positive, and satisfy ES $`\geq`$ VaR at the same confidence level.\

</div>

Acceptance criterion: every NP test must return finite numerical output or a controlled exception. Silent NaNs, infinities, negative risk values where positivity is required, and unstable covariance outputs fail the test plan.

# Behavioral Confirmation Testing

Behavioral confirmation tests encode model facts that should hold before any benchmark comparison is meaningful. These are not external goldens; they are internal consistency checks based on financial logic. The implemented behavioral tests are BEH_01 through BEH_08 in `tests/test_backend.py`.

## Black-Scholes Behavioural Properties

<div class="center">

<div class="tabular">

L2.2cmL4.8cmL6cm **Test ID** & **Property** & **Expected behavior**\
`BEH_01` & Call monotonicity in spot & A European call price should increase as $`S`$ increases.\
`BEH_02` & Call monotonicity in strike & A European call price should decrease as $`K`$ increases.\
`BEH_03` & Call monotonicity in volatility & A European call price should increase as $`\sigma`$ increases.\
`BEH_04` & Put-call parity & The implementation should satisfy $`C - P = S e^{-qT} - K e^{-rT}`$ within floating-point tolerance.\
`BEH_05` & Volatility-to-zero limiting case & As volatility tends to zero, the option value should approach intrinsic / deterministic discounted value.\
`BEH_06` & No-arbitrage lower bound & The call price should not fall below $`\max(S e^{-qT} - K e^{-rT}, 0)`$.\

</div>

</div>

These tests check that the implementation behaves correctly between known limiting cases and does not violate elementary no-arbitrage logic.

## Risk Measure Coherence and Positivity

<div class="center">

<div class="tabular">

L2.2cmL4.8cmL6cm **Test ID** & **Property** & **Expected behavior**\
`BEH_07` & ES and VaR internal consistency & Historical, parametric, and Monte Carlo methods should satisfy ES $`\geq`$ VaR when evaluated at the same confidence level.\
`BEH_08` & Historical VaR positivity & Historical VaR should be finite and positive on the representative two-stock fixture.\

</div>

</div>

Acceptance criterion: behavioural facts must hold before a model output is treated as meaningful. If these basic implications fail, the issue is more serious than a calibration error, because it indicates a pricing, aggregation, or risk-measure implementation defect.

# Convergence and Inversion Testing

Convergence and inversion tests address numerical robustness rather than only formula correctness. A production-style numerical method should be stable as discretisation or simulation parameters change, and an inversion routine should reprice or recover the target quantity when its output is substituted back into the model.

## Monte Carlo Convergence

- **CONV_01** checks that Monte Carlo VaR becomes more stable as the number of simulated paths increases from $`500`$ to $`5{,}000`$ to $`50{,}000`$.

For independent Monte Carlo simulation, the standard error scales as
``` math
\frac{\sigma}{\sqrt{N}},
```
so the expected order of simulation error is approximately
``` math
O\left(\frac{1}{\sqrt{N}}\right).
```
The implemented test therefore checks that the fine-grid difference $`|\mathrm{VaR}_{50k} - \mathrm{VaR}_{5k}|`$ is smaller than the coarse-grid difference $`|\mathrm{VaR}_{5k} - \mathrm{VaR}_{500}|`$ under a fixed random seed.

## Merton and Kupiec Inversion Checks

<div class="center">

<div class="tabular">

L2.2cmL4.8cmL6cm **Test ID** & **Test** & **Expected behavior**\
`INV_01` & Merton implied-barrier round-trip & Given $`V_0`$, $`B`$, $`r`$, $`\sigma`$, and $`T`$, compute the Merton default probability, convert it to target survival, invert for $`B`$, and recover the original barrier within tolerance.\
`INV_02` & Kupiec exact-count p-value & For $`250`$ observations and an exception count close to the expected count under the selected confidence level, the Kupiec test should not reject unconditional coverage.\

</div>

</div>

Acceptance criterion: convergence tests should demonstrate improved stability as simulation count increases, and inversion tests should recover the original input or fail with a clear diagnostic rather than silently returning an inconsistent quantity.

# P&L Attribution and Hedge Effectiveness

Robustness testing in model validation includes checking whether modeled risk factors explain observed or simulated portfolio value changes. This project includes a small P&L attribution test and a one-day hedge effectiveness test. These tests are deliberately scoped to the course risk engine; a full dynamic hedging engine is out of scope.

## P&L Attribution

- **PNL_01** checks a linear stock-only portfolio where the delta-explained P&L equals the actual P&L exactly.

For a linear portfolio with share quantities $`q_i`$ and prices $`S_{i,t}`$, actual one-period P&L is
``` math
\Delta V_t
  =
  \sum_i q_i (S_{i,t+1} - S_{i,t}).
```
Because the position is linear in the underlying prices, the delta-based explanation is identical:
``` math
\Delta V_t^{\mathrm{explained}}
  =
  \sum_i q_i \Delta S_{i,t}.
```
The expected residual is therefore
``` math
\Delta V_t - \Delta V_t^{\mathrm{explained}} = 0.
```

## Hedge Effectiveness

- **HEDGE_01** checks that a one-day delta hedge reduces the absolute P&L of an at-the-money call under both $`+1\%`$ and $`-1\%`$ spot shocks.

The test uses the Black-Scholes delta at the starting spot, applies a small spot shock, and compares the absolute unhedged option P&L with the absolute delta-hedged P&L. The required behavior is:
``` math
|\mathrm{P\&L}_{\mathrm{hedged}}|
  <
  |\mathrm{P\&L}_{\mathrm{unhedged}}|.
```

This is not a claim that the project implements production dynamic hedging. It only confirms that the local delta calculation has the expected first-order hedge interpretation over a small one-day shock.

# Edge-Case and Failure-Mode Tests

The following edge and failure cases are explicitly required and all are implemented in the test suite:

- **EDGE_01** Empty portfolio raises a controlled exception

- **EDGE_02** Missing ticker history raises a controlled exception

- **EDGE_03** Insufficient lookback raises with a descriptive message

- **EDGE_04** NaN prices are handled or rejected explicitly

- **EDGE_05** Duplicate dates are handled or rejected explicitly

- **EDGE_06** Negative or zero prices are rejected

- **EDGE_07** Invalid confidence level is rejected

- **EDGE_08** VaR confidence can be set independently from ES confidence

- **EDGE_09** Non-positive volatility is rejected

- **EDGE_10** Zero maturity is handled or rejected

- **EDGE_11** Non-PSD covariance is handled or rejected

- **EDGE_12** Monte Carlo seed reproducibility is confirmed

- **EDGE_13** $`n_{\mathrm{sims}} \leq 0`$ is rejected

- **EDGE_14** Zero hazard gives survival $`= 1`$ and PD $`= 0`$

- **EDGE_15** Recovery $`R = 1`$ gives zero CDS/CVA loss

- **EDGE_16** Merton survival decreases as debt face value increases

- **EDGE_17** Capital-ratio division by zero is rejected

- **EDGE_18** Netted exposure $`\leq`$ gross exposure

- **EDGE_19** Collateralized exposure $`\leq`$ uncollateralized exposure

- **EDGE_20** ES $`\geq`$ VaR when evaluated at the same confidence

# Integration and UI Tests

## Integration Tests

Required integration paths:

- Portfolio creation to `RiskEngineService.run_all()`;

- End-to-end VaR/ES computation under live or cached market data;

- End-to-end backtesting with Kupiec and Christoffersen diagnostics;

- Formula-sheet integration with live market data.

Relevant files: `tests/integration_test.py` and `tests/integration_test_formula_sheet.py`.

## UI Tests

Required UI test areas: portfolio editor; market data panel; risk settings; results panel; credit panel; CDS/CVA panel; capital panel; and chart helpers.

Relevant files: `tests/test_ui_panels.py` and `tests/test_charts.py`.

Acceptance criterion: panels render without crashing; user inputs are validated; expected outputs appear; and download and data-loading paths behave consistently.

# Backtesting Tests

Required backtesting tests:

- **BT_01** Exception flag: $`\mathrm{loss} > \mathrm{VaR}`$ correctly identified

- **BT_02** No-exception case (all losses below VaR)

- **BT_03** All-exception case (all losses above VaR)

- **BT_04** Expected exception count $`= T \cdot (1-\alpha)`$

- **BT_05** Kupiec $`\mathrm{LR}_{\mathrm{uc}}`$ is finite and non-negative

- **BT_06** Kupiec p-value in $`[0,1]`$

- **BT_07** 95% confidence backtest

- **BT_08** 97.5% confidence backtest

- **BT_09** 99% confidence backtest

- **BT_10** Basel traffic-light zone correctly classified

- **BT_11** Exception severity table

- **BT_12** Christoffersen independence LR finite and non-negative

Lecture 5 emphasizes that a backtest must examine not only the *frequency* of exceptions but also their *clustering* and *behavior across confidence levels*. The repository includes Christoffersen-style diagnostics; this test plan explicitly requires coverage of those diagnostics.

# Data Validation Tests

- **DATA_01** Price series has strictly positive prices

- **DATA_02** No duplicate dates after cleaning

- **DATA_03** Missing-data report is generated for all-NaN columns

- **DATA_04** Stale-price run of $`\geq 10`$ days is detected

- **DATA_05** Extreme return outliers are visible (no silent clipping)

- **DATA_06** Aligned histories share a common date index

- **DATA_07** Insufficient lookback raises with a descriptive message

- **DATA_08** CSV with a bad date column raises

- **DATA_09** CSV with a missing price column raises

- **DATA_10** Data-proxy caveat is documented in the test plan and software design

Lecture 5 explicitly warns that poor data quality leads to poor model outputs. Data validation therefore belongs in the test plan rather than only in the software-design document.

# Coverage Plan

The coverage target is the highest achievable with the no-network unit suite. Streamlit UI branch paths are excluded because they require a live browser context. All other `src/` modules are expected to reach or exceed 95% statement coverage. The acceptance command is:

<div class="shellcode">

python -m pytest tests/  –cov=src  –cov-report=term-missing  –cov-report=html:submission/coverage_report  –cov-report=xml:submission/coverage_report/coverage.xml  –ignore=tests/integration_test.py  –ignore=tests/integration_test_formula_sheet.py

</div>

Coverage steps:

1.  Run coverage and identify all missing lines.

2.  Add tests for missing branches wherever feasible.

3.  Use `pragma: no cover` only for genuinely unreachable defensive code.

4.  Save terminal coverage output and `htmlcov/index.html`.

5.  Include coverage output and missing-line discussion in Deliverable 5.

# Acceptance Criteria

The software satisfies the planned test standard if:

1.  All required no-network unit tests pass.

2.  Analytical golden tests match expected values within stated tolerances.

3.  Homework-derived fixtures pass within stated tolerances.

4.  Invalid inputs produce controlled failures rather than silent bad outputs.

5.  Backtesting diagnostics return finite, interpretable results.

6.  UI smoke tests pass.

7.  Integration workflows either pass or have explicitly documented, understood failures.

8.  Statement coverage across `src/` reaches $`\geq 95\%`$ for all non-UI modules; Streamlit panel branches requiring a live browser session are excluded from the hard target and documented as justified gaps.

9.  Benchmark comparisons (Section <a href="#sec:mrm-benchmark" data-reference-type="ref" data-reference="sec:mrm-benchmark">11</a>) confirm that relative differences between methods are within documented bounds.

# Known Untested Areas

A sound test plan acknowledges residual validation risk honestly:

- Some extension modules have historically lower coverage than the core market-risk engine.

- External option-calculator benchmarking may remain partially manual unless automated.

- Live market-data integration is inherently less deterministic than no-network unit tests.

- Duplicate-date or stale-price handling is not as extensively tested as basic missing-data and positivity checks.

- Coverage target may not reach 100% without additional branch-specific tests in CDS, hazard absolute-shock branches, and selected UI/regulatory paths.

# Robustness Testing

Lecture 5 requires that testing assess *stability and robustness*, including sensitivity to market conditions and deal specifications, behavior over a large range of inputs, and identification of where the model performs poorly. This section formalizes those requirements.

## Parameter Sensitivity

The following test cases probe sensitivity to the key model-configuration parameters that are subject to expert judgment.

- **ROB_01** Lookback window sweep: run historical and parametric VaR at 60, 126, 252, and 504 days. Expected: VaR varies with the captured volatility regime; no crashes; no NaN outputs.

- **ROB_02** EWMA half-life $`N`$ sweep: run EWMA covariance at $`N \in \{10, 20, 60, 120\}`$. Expected: covariance PSD at all values; estimates vary smoothly; $`N=10`$ is more responsive, $`N=120`$ is more stable.

- **ROB_03** Monte Carlo path-count sweep: VaR at $`n \in
      \{100, 1\,000, 10\,000\}`$. Expected: VaR converges as $`n`$ increases; result at $`n = 100`$ is noisier but finite; at $`n = 10\,000`$ is stable.

- **ROB_04** Extreme confidence levels: VaR and ES at $`\alpha \in \{0.95, 0.975, 0.99, 0.999\}`$. Expected: strictly monotonically increasing; ES $`\geq`$ VaR throughout.

- **ROB_05** Very short lookback (30 days): expected to return a descriptive error or reduced precision result; no crash.

- **ROB_06** Very long lookback (2,520 days): expected to handle large covariance matrices without memory error; result finite.

These are implemented in `tests/test_coverage_gaps.py` and `tests/test_strict_numerics.py` (see Appendix CONV_05 and CONV_06).

## Extreme and Boundary Input Tests

- **ROB_07** Near-expiry option ($`T = 1/252`$): BS price at intrinsic-value limit; delta near 0 or 1; no NaN.

- **ROB_08** Deep OTM put (strike = 0.5 $`\times`$ spot): small positive price; delta near zero; VaR dominated by underlying.

- **ROB_09** High-volatility input ($`\sigma = 2.0`$, i.e. 200%): VaR large but finite; no overflow.

- **ROB_10** Zero-drift portfolio ($`\mu = 0`$): parametric VaR still positive from variance term; historical and MC also positive.

- **ROB_11** Single-position portfolio: all three risk engines return positive, finite VaR; no failure from degenerate covariance.

- **ROB_12** Large portfolio (50 positions): covariance assembly, risk computation, and service orchestration complete without crash.

## Known Conditions Where the Model Performs Poorly

A sound validation program explicitly documents where the model is expected to degrade, not just where it succeeds. These cases are *documented limitations*, not unresolved bugs.

- **Historical simulation.** The equally-weighted method is slow to adapt to volatility regime changes. This manifests directly in the backtesting results: the Christoffersen independence test rejects the independence hypothesis at $`p < 10^{-14}`$, indicating clustered exceptions during high-volatility sub-periods. Filtered Historical Simulation or GARCH-based forecasting is the documented improvement path.

- **Delta-normal parametric.** Treats the portfolio as linear in underlying returns. For portfolios with significant gamma (e.g., short options), it systematically underestimates tail losses. Test BEH_01 through BEH_05 confirm directional correctness; behavioral comparison tests confirm that options-heavy portfolios show materially higher historical and MC VaR relative to parametric.

- **Fat-tailed return distributions.** When actual returns exhibit heavy tails (crisis periods, individual-stock events), the parametric normality assumption underestimates extreme losses. The exact GBM lognormal module provides a tractable analytical benchmark; EVT tail fitting is documented as a future enhancement.

- **Model cascading into capital calculations.** The VaR output feeds `src/risk/regulatory.py` capital requirements. ROB tests confirm that extreme VaR inputs (very large, near-zero, and zero) do not produce silent failures in downstream RWA and ratio calculations.

- **Illiquid or discontinuous markets.** Option repricing assumes continuously tradeable underlying without gaps. Stale prices or trading halts in the historical window will affect all three methods; the data validation layer flags stale-price runs but does not adjust the risk estimate.

# Appendix: Behavioural and Convergence Tests

## Behavioural Tests

The primary behavioural confirmation tests are now documented in Section <a href="#sec:behavioral-confirmation" data-reference-type="ref" data-reference="sec:behavioral-confirmation">13</a>. The implemented BEH_01 through BEH_08 cases cover Black-Scholes monotonicity, put-call parity, volatility-to-zero behavior, no-arbitrage lower bounds, ES/VaR internal consistency, and finite positive historical VaR.

## Convergence and Stability Tests

The primary convergence and inversion tests are now documented in Section <a href="#sec:convergence-inversion" data-reference-type="ref" data-reference="sec:convergence-inversion">14</a>. Additional robustness and sensitivity tests remain relevant:

- **CONV_01** MC VaR convergence as simulation count increases

- **CONV_02** One-asset MC ES convergence toward exact lognormal ES as $`n_{\mathrm{sims}}`$ increases

- **CONV_03** Finite-difference delta converges to analytical Black-Scholes delta

- **CONV_04** Too-small bump demonstrates cancellation risk or is explicitly avoided

- **CONV_05** Rolling-window VaR sensitivity across window lengths (2y vs. 5y)

- **CONV_06** EWMA $`\lambda`$ sensitivity

## Traceability to Repository Test Files

- `tests/test_backend.py`

- `tests/test_numerical_precision.py`

- `tests/test_backtest_extensions.py`

- `tests/test_course_validation.py`

- `tests/test_homework_cases.py`

- `tests/test_lognormal.py`

- `tests/test_credit.py`

- `tests/test_credit_service.py`

- `tests/test_cva_mitigants.py`

- `tests/test_counterparty_mitigation.py`

- `tests/test_market_data.py`

- `tests/test_config_and_validation.py`

- `tests/test_ui_panels.py`

- `tests/test_charts.py`

- `tests/test_regulatory.py`

- `tests/test_dfast_pathing.py`

- `tests/test_balance_sheet.py`

- `tests/test_coverage_gaps.py`

- `tests/test_strict_numerics.py`

- `tests/test_es_confidence_split.py`

- `tests/integration_test.py`

- `tests/integration_test_formula_sheet.py`

<div class="thebibliography">

9

Goldberg, D. (1991). *What Every Computer Scientist Should Know About Floating-Point Arithmetic*. ACM Computing Surveys, 23(1), 5–48.

</div>
