# Deliverable 3: Test Plan

## 1. Executive Summary

This test plan answers four questions:

1. What is being tested?
2. Why is it being tested?
3. Against what benchmark is it being tested?
4. What constitutes acceptable behavior?

The `MATH5320` repository contains a broad test suite covering formula correctness, portfolio valuation, VaR/ES calculations, backtesting, market-data loading, UI behavior, and course-formula extensions. The plan below formalizes those tests into a model-risk-oriented validation program. It treats testing as part of model development and model governance, not just as proof that the application runs.

---

## 2. Test Objectives

The test plan validates:

1. correctness of mathematical formulas;
2. correctness of portfolio valuation and option repricing;
3. correctness of VaR/ES calculations under historical, parametric, and Monte Carlo methods;
4. correctness of backtesting and exception logic;
5. correctness of credit/regulatory formula-sheet extensions;
6. robustness to invalid inputs, edge cases, and numerical failure modes;
7. correct service-layer integration and UI behaviour;
8. reproducibility through deterministic fixtures, seeded Monte Carlo, and coverage reporting.

These objectives align with Lecture 5’s emphasis on testing as part of model development and validation, rather than mere application execution.

---

## 3. Scope of Testing

In scope:

- Pure pricing formulas
- Portfolio valuation and aggregation
- Return and covariance estimation
- Historical VaR/ES
- Parametric VaR/ES
- Monte Carlo VaR/ES
- VaR backtesting and exception diagnostics
- Course-formula extensions in `src/risk/lognormal.py`, `src/credit/`, and `src/risk/regulatory.py`
- Market-data loading and validation
- Streamlit UI rendering and panel behavior
- End-to-end service orchestration

Out of scope for this test plan:

- Production deployment hardening
- Enterprise access control
- Performance benchmarking at scale
- Full external-market-data certification
- Production volatility-surface validation

---

## 4. Test Environment

The planned and observed validation environment is local and reproducible:

- Repository root: `MATH5320`
- Python: `3.12.2`
- OS: macOS / Darwin arm64
- Core packages include:
  - `streamlit 1.37.1`
  - `numpy 1.26.4`
  - `pandas 3.0.2`
  - `scipy 1.17.1`
  - `plotly 5.24.1`
  - `yfinance 1.2.0`
  - `pytest 7.4.4`
  - `pytest-cov 7.1.0`

The environment should always be snapshotted with:

```bash
git rev-parse HEAD
python --version
pip freeze > test_artifacts/requirements_freeze.txt
```

Network status matters:

- The no-network unit suite should be runnable without external downloads.
- Live-data integration scripts require network access.

Randomness control:

- Monte Carlo regression-style tests should use fixed seeds.
- If a seed is randomized, it should be recorded.

---

## 5. Test Data and Fixtures

The repository uses several classes of data and fixtures:

### 5.1 Synthetic Fixtures

Used for deterministic unit tests:

- synthetic price histories,
- toy two-stock portfolios,
- simple option positions,
- deterministic exposures and covariance matrices.

### 5.2 Course-Derived Fixtures

Used for validation against homework and formula-sheet results:

- exact GBM/lognormal values,
- hazard-rate survival and spread values,
- Merton PD and valuation cases,
- CDS and CVA examples,
- regulatory RWA and capital-ratio examples.

### 5.3 Bloomberg Course Data

Observed local files:

- `data/AAPL-bloomberg.csv`
- `data/CAT-bloomberg.csv`

These support the AAPL/CAT course portfolio notebooks and related regression-style checks.

### 5.4 Live Market Data

Used by integration scripts:

- Yahoo Finance downloads for equities and rate proxies,
- cached live download paths in `src/data/market_data.py`.

### 5.5 Acceptance of Data Proxies

Lecture 5 requires documentation of data used, data quality, proxies, and cleaning assumptions. Therefore:

- CSV-based Bloomberg fixtures are treated as course acceptance data.
- Yahoo Finance is treated as a convenience data source, not a gold-standard benchmark.
- Any cleaning, alignment, or dropping of rows must be explicit.

---

## 6. Test Categories

| Test category | Purpose | Example |
|---|---|---|
| Unit tests | Validate pure functions | Black-Scholes price, hazard survival |
| Analytical goldens | Compare against closed-form values | Exact GBM VaR/ES |
| Homework fixtures | Course-derived regression values | AAPL/CAT VaR, Merton Q/P |
| External benchmarks | Independent comparison | Option-calculator-style Black-Scholes values, Basel traffic light |
| Edge cases | Validate boundary behaviour | Zero hazard, zero volatility |
| Failure-mode tests | Ensure controlled errors | Invalid confidence, missing data |
| Behavioural tests | Check financial monotonicity and logic | Option price increases with vol |
| Convergence tests | Check numerical stability | MC VaR as simulation count increases |
| Backtesting tests | Validate VaR forecast logic | Exceptions, Kupiec, traffic light |
| Data tests | Validate input quality | Missing or stale prices |
| Integration tests | Validate full workflow | Portfolio input to risk output |
| UI tests | Validate Streamlit panels | Portfolio editor, settings, results |
| Coverage tests | Ensure source execution breadth | Coverage report and missing-line review |

---

## 7. Module-Level Test Matrix

| Module | Required tests |
|---|---|
| `pricing/black_scholes.py` | Price, delta, put-call parity, invalid inputs, monotonicity |
| `portfolio/positions.py` | Stock value, option value, delta exposure, long/short signs |
| `portfolio/portfolio.py` | Aggregate value, exposure vector, empty portfolio rejection |
| `risk/returns.py` | Log returns, overlapping returns, horizon summation |
| `risk/estimators.py` | Rolling mean/cov, EWMA, covariance symmetry |
| `risk/historical.py` | Historical VaR/ES, log shock, absolute shock if used, missing data |
| `risk/parametric.py` | Normal VaR/ES, covariance aggregation, ES confidence separation |
| `risk/monte_carlo.py` | Seeded reproducibility, MC VaR/ES, covariance validation |
| `risk/backtest.py` | Exceptions, Kupiec, edge cases, traffic light, severity, independence |
| `risk/lognormal.py` | Exact long/short GBM VaR/ES |
| `credit/hazard.py` | Constant and piecewise hazard, survival, density, risky ZCB |
| `credit/merton.py` | Q/P PD, equity/debt, target survival inversion |
| `credit/cds.py` | Approximation and full par spread |
| `credit/cva.py` | EPE, CVA, discounted CVA |
| `credit/mitigation.py` | Netting, collateral, CSA logic |
| `risk/regulatory.py` | RWA, capital ratio, DFAST pathing |
| `services/risk_engine_service.py` | Orchestration and result-object consistency |
| `ui/*.py` | Streamlit input handling and result rendering |

---

## 8. Analytical Golden Tests

### 8.1 Black-Scholes

Required test family:

- `BS_01` call price against known value
- `BS_02` put price against known value
- `BS_03` put-call parity
- `BS_04` call delta in `[0,1]`
- `BS_05` put delta in `[-1,0]`
- `BS_06` option price increases with volatility
- `BS_07` invalid maturity or volatility raises

Acceptance criterion:

- numerical values agree within analytic tolerance,
- parity holds within floating-point tolerance,
- invalid domains fail loudly.

### 8.2 Exact Lognormal VaR/ES

Required test family:

- `LN_01` long VaR exact formula
- `LN_02` long ES exact formula
- `LN_03` short VaR exact formula
- `LN_04` short ES exact formula
- `LN_05` VaR scales linearly with notional
- `LN_06` short VaR exceeds long VaR for identical base inputs
- `LN_07` zero horizon gives zero-risk limit or controlled rejection

### 8.3 Normal Parametric VaR/ES

Required test family:

- `NORM_01` VaR formula
- `NORM_02` ES formula
- `NORM_03` ES is at least VaR when using the same confidence
- `NORM_04` covariance aggregation
- `NORM_05` offsetting exposures reduce risk
- `NORM_06` invalid covariance is rejected or handled

These tests are especially important because they validate the baseline approximation engine that is easiest to explain but easiest to misuse.

---

## 9. Homework-Derived Tests

The repository already contains substantial homework-derived cases. They should be presented formally as regression fixtures.

| Case ID | Area | Expected validation |
|---|---|---|
| `HW4_SINGLE_STOCK` | GBM VaR | 5-day 99% VaR near homework value |
| `HW4_TWO_STOCK` | Parametric covariance VaR | Correct mean/variance/correlation aggregation |
| `HW6_EWMA` | Rolling/EWMA estimation | Window and EWMA parameter behavior |
| `HW6_HAZARD_CONST` | Constant hazard | Survival/default probability regression |
| `HW6_HAZARD_PIECEWISE` | Piecewise hazard | `lambda(t)`, `Lambda(t)`, `s(t)`, `p(t)`, spread table |
| `HW7_MERTON_QP` | Merton | Q vs P PD comparison |
| `HW7_MERTON_TIMING` | Merton | Zero default probability before maturity interval |
| `HW8_CDS` | CDS | Rough spread and full par spread |
| `HW8_CVA` | CVA | Exposure and default-probability aggregation |
| `HW9_MERTON_INVERSION` | Merton | Target-survival inversion |
| `HW9_SHORT` | Short-risk formulas | Short VaR/ES sign and magnitude behavior |
| `HW10_RWA` | Regulatory | RWA and capital-ratio arithmetic |
| `HW10_DFAST` | Regulatory | 9-quarter stress-path structure if implemented |

The uploaded homework and course-validation values provide direct numerical anchors for regression-style testing, which is exactly the kind of evidence Lecture 5 encourages.

---

## 10. External/Official Benchmark Tests

These are not substitutes for homework fixtures. They provide a second source of confidence.

### 10.1 External Benchmark: Option Calculator Style Cases

Planned benchmark cases:

| Case | S | K | T | r | q | vol | type |
|---|---:|---:|---:|---:|---:|---:|---|
| ATM call | 100 | 100 | 1.0 | 0.05 | 0.00 | 0.20 | call |
| ATM put | 100 | 100 | 1.0 | 0.05 | 0.00 | 0.20 | put |
| Dividend call | 100 | 105 | 2.0 | 0.03 | 0.02 | 0.25 | call |
| ITM put | 90 | 100 | 0.5 | 0.04 | 0.01 | 0.30 | put |
| Near-expiry | 100 | 100 | `1/252` | 0.05 | 0.00 | 0.20 | call |

Acceptance criterion:

- price and delta within reasonable rounding tolerance of an external calculator or standard reference.

### 10.2 Official-Style Benchmark: Basel Traffic Light

If the Basel-style helper is present, required checks are:

- `basel_zone(0, 250, 0.99) == green`
- `basel_zone(4, 250, 0.99) == green`
- `basel_zone(5, 250, 0.99) == amber/yellow`
- `basel_zone(9, 250, 0.99) == amber/yellow`
- `basel_zone(10, 250, 0.99) == red`

### 10.3 Official-Style DFAST Structure

Do not claim Federal Reserve replication. Test only the structural expectations:

- 9-quarter path
- baseline / adverse / severely adverse naming
- Tier 1 capital path
- RWA path
- minimum capital ratio
- hurdle pass/fail
- overlay-style loss components where implemented

---

## 11. Edge-Case and Failure-Mode Tests

The following edge and failure cases should exist explicitly in the plan:

- `EDGE_01` empty portfolio raises
- `EDGE_02` missing ticker history raises
- `EDGE_03` insufficient lookback raises
- `EDGE_04` NaN prices handled or rejected
- `EDGE_05` duplicate dates handled or rejected
- `EDGE_06` negative or zero price rejected
- `EDGE_07` invalid confidence rejected
- `EDGE_08` VaR confidence separate from ES confidence
- `EDGE_09` negative volatility rejected
- `EDGE_10` zero maturity handled or rejected
- `EDGE_11` non-PSD covariance handled or rejected
- `EDGE_12` Monte Carlo seed reproducibility
- `EDGE_13` `n_sims <= 0` rejected
- `EDGE_14` zero hazard gives survival 1 and PD 0
- `EDGE_15` recovery 1 gives zero CDS/CVA loss
- `EDGE_16` Merton survival decreases with debt face value
- `EDGE_17` capital-ratio division by zero rejected
- `EDGE_18` netted exposure is no larger than gross exposure
- `EDGE_19` collateralized exposure is no larger than uncollateralized exposure
- `EDGE_20` ES is at least VaR when evaluated at the same confidence

These are high-value because they test not just correctness but failure discipline.

---

## 12. Integration and UI Tests

### 12.1 Integration Tests

Required integration paths:

- portfolio creation to `RiskEngineService.run_all()`,
- end-to-end VaR/ES computation under live or cached data,
- end-to-end backtesting,
- formula-sheet integration with live market data.

Relevant files:

- `tests/integration_test.py`
- `tests/integration_test_formula_sheet.py`

### 12.2 UI Tests

Required UI test areas:

- portfolio editor
- market data panel
- risk settings
- results panel
- credit panel
- CDS/CVA panel
- capital panel
- chart helpers

Relevant files:

- `tests/test_ui_panels.py`
- `tests/test_charts.py`

Acceptance criterion:

- panels render,
- user inputs are validated,
- expected outputs appear without crashing,
- download and data-loading paths behave consistently.

---

## 13. Backtesting Tests

Required backtesting tests:

- `BT_01` exception flag: `loss > VaR`
- `BT_02` no-exception case
- `BT_03` all-exception case
- `BT_04` expected exception count
- `BT_05` Kupiec statistic finite
- `BT_06` Kupiec p-value in `[0,1]`
- `BT_07` 95% confidence backtest
- `BT_08` 97.5% confidence backtest
- `BT_09` 99% confidence backtest
- `BT_10` Basel traffic light, if implemented
- `BT_11` exception severity table
- `BT_12` exception clustering / Christoffersen, if implemented

Lecture 5’s guidance emphasizes not only the frequency of exceptions but also their clustering and behavior across confidence levels. The repository already includes Christoffersen-style diagnostics, so the test plan should explicitly cover them.

---

## 14. Data Validation Tests

Required data-quality checks:

- `DATA_01` price series has positive prices
- `DATA_02` no duplicate dates after cleaning
- `DATA_03` missing-data report is generated
- `DATA_04` stale-price run is detected where practical
- `DATA_05` extreme return outliers are flagged or at least visible
- `DATA_06` aligned histories share common dates
- `DATA_07` insufficient lookback raises
- `DATA_08` CSV with bad date column raises
- `DATA_09` CSV with missing price column raises
- `DATA_10` proxy/data-source caveat is documented

Lecture 5 explicitly warns that poor data quality leads to poor model outputs, so data validation belongs in the test plan rather than only in the software-design document.

---

## 15. Coverage Plan

The coverage target for this project is the highest achievable with the no-network unit suite. Streamlit UI branch paths are excluded from the target because they require a live browser context; all other `src/` modules are expected to reach or exceed 95% statement coverage. The acceptance command is:

```bash
python -m pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:submission/coverage_report \
  --cov-report=xml:submission/coverage_report/coverage.xml \
  --ignore=tests/integration_test.py \
  --ignore=tests/integration_test_formula_sheet.py
```

Planned coverage steps:

1. Run coverage.
2. Identify all missing lines.
3. Add tests for missing branches.
4. Use `pragma: no cover` only for genuinely unreachable defensive code.
5. Save terminal coverage output.
6. Save `htmlcov/index.html`.
7. Include coverage output and missing-line discussion in the Test Results report.

---

## 16. Acceptance Criteria

The software is considered to satisfy the planned test standard if:

1. All required no-network unit tests pass.
2. Analytical golden tests match expected values within stated tolerances.
3. Homework-derived fixtures pass within stated tolerances.
4. Invalid inputs produce controlled failures rather than silent bad outputs.
5. Backtesting diagnostics return finite, interpretable results.
6. UI smoke tests pass.
7. Integration workflows either pass or have explicitly documented, understood failures.
8. Statement coverage across `src/` reaches ≥ 95% for all non-UI modules. Streamlit UI panel branches that require a live browser session are excluded from the hard target and documented as justified gaps in the Test Results report.

---

## 17. Known Untested Areas

Even before execution, the plan should flag areas that are likely to remain weak or incomplete unless extra work is done:

- Some extension modules have historically lower coverage than the core market-risk engine.
- External option-calculator benchmarking may remain manual unless automated.
- Live market-data integration is inherently less deterministic than no-network unit tests.
- Duplicate-date or stale-price handling is not as explicitly tested as basic missing-data and positivity checks.
- Coverage target may remain unmet without additional branch-specific tests in CDS, hazard, historical absolute-shock branches, `risk/normal.py`, and selected UI/regulatory paths.

This section is important because a good test plan should be honest about residual validation risk.

---

## 18. Appendix: Full Test Case Register

### 18.1 Behavioural Tests

- `BEH_01` call price increases with spot
- `BEH_02` put price decreases with spot
- `BEH_03` option price increases with volatility
- `BEH_04` call price decreases with strike
- `BEH_05` put price increases with strike
- `BEH_06` VaR increases with notional
- `BEH_07` VaR increases with volatility
- `BEH_08` CDS spread increases with hazard
- `BEH_09` CVA increases with exposure
- `BEH_10` CVA decreases with recovery
- `BEH_11` capital ratio decreases after losses
- `BEH_12` stressed capital-path minimum is correctly selected

### 18.2 Convergence and Stability Tests

- `CONV_01` one-asset MC VaR converges toward exact lognormal VaR as `n_sims` increases
- `CONV_02` one-asset MC ES converges toward exact lognormal ES as `n_sims` increases
- `CONV_03` finite-difference delta converges to analytical Black-Scholes delta
- `CONV_04` too-small bump demonstrates cancellation risk or is explicitly avoided
- `CONV_05` rolling-window VaR sensitivity: 2y vs 5y vs 10y
- `CONV_06` EWMA lambda sensitivity

### 18.3 Traceability to Repository Tests

The following files currently provide most of the plan coverage:

- `tests/test_backend.py`
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
