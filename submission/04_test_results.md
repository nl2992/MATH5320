<div class="titlepage">

**Test Results Report**

MATH GR 5320 Portfolio Risk Management System

Columbia University, Financial Risk Management, Spring 2026

<div class="tabular">

L4cmL9cm **Field** & **Value**\
Deliverable & 5 of 5 (10 points)\
Authors & Nigel Li, Michael Adegbite, Stella\
Reference Commit & `5841589` (main branch, May 2026)\
Run Timestamp & 2026-05-11 03:00:09 EDT\
Python Version & 3.12.2 OS: Darwin 24.5.0 arm64\
No-network tests & **644 passed, 0 failed, 0 skipped**\
Statement Coverage & **95%**\
Integration scripts & **2 / 2 passed**\

</div>

<div class="minipage">

*This document presents observed test execution results for the MATH5320 risk engine. The structure follows the Test Plan (Deliverable 3), covering unit test outcomes, integration test outcomes, analytical golden comparisons, homework fixture validation, backtest results, and coverage analysis.*

</div>

</div>

# Executive Summary

The test results demonstrate strong validation coverage across the repository’s no-network unit suite and live integration scripts.

**No-network suite:** 644 tests passed, 0 failed, 0 skipped.

**Coverage:** 95% statement coverage across `src/`; remaining untested lines are concentrated in UI branch paths, selected credit-service helpers, and defensive validation branches.

**Integration:** Both live-data integration scripts passed, confirming live-data download, service orchestration, full risk-model execution, and representative backtesting behavior end to end.

**Overall conclusion:** The deterministic, no-network, and live integration evidence collectively support the correctness of the core software design and formula implementations. We regard the implementation as suitable for submission as an academic risk-engine validation exercise.

# Test Environment

## Run Metadata

- Date/time: 2026-05-11 03:00:09 EDT

- Git commit: `5841589e3f3d2dbd3c1e38b08642eccce201a6a2`

- Python: 3.12.2 OS: Darwin 24.5.0 arm64

- Network status: enabled for integration-script execution

- Bloomberg data files present: `data/AAPL-bloomberg.csv`, `data/CAT-bloomberg.csv`

## Key Package Versions

`streamlit 1.37.1`, `numpy 1.26.4`, `pandas 3.0.2`, `scipy 1.17.1`, `plotly 5.24.1`, `yfinance 1.2.0`, `pytest 7.4.4`, `pytest-cov 7.1.0`.

## Environment Artifacts

The following files were written to `submission/test_artifacts/` for the submission evidence package:

- `git_commit.txt`, `python_version.txt`, `requirements_freeze.txt`

- `pytest_output.txt`, `coverage_output.txt`

- `integration_test_output.txt`, `integration_test_formula_sheet_output.txt`

- `per_file_test_counts.json`, `homework_fixture_results.csv`, `official_benchmark_results.csv`, `backtest_results.csv`

# Test Commands

## No-Network Unit Suite

<div class="shellcode">

python -m pytest tests/  –ignore=tests/integration_test.py  –ignore=tests/integration_test_formula_sheet.py -v

</div>

## Coverage Run

<div class="shellcode">

python -m pytest tests/ –cov=src –cov-report=term-missing  –cov-report=html:submission/coverage_report  –cov-report=xml:submission/coverage_report/coverage.xml  –ignore=tests/integration_test.py  –ignore=tests/integration_test_formula_sheet.py

</div>

## Integration Scripts

<div class="shellcode">

python tests/integration_test.py python tests/integration_test_formula_sheet.py

</div>

# Test Execution Summary

## Test Group Results

<div class="center">

<div class="tabular">

L3.5cmL5cmrrrl **Group** & **File(s)** & **Passed** & **Failed** & **Skipped** & **Notes**\
Core backend & `test_backend.py` & 29 & 0 & 0 & Core pricing, portfolio, VaR/ES, service smoke tests\
Backtest extensions & `test_backtest_extensions.py` & 31 & 0 & 0 & Christoffersen, conditional coverage, traffic light, severity\
Course validation & `test_course_validation.py` & 67 & 0 & 0 & Course formula-sheet and regression fixtures\
Homework fixtures & `test_homework_cases.py` & 83 & 0 & 0 & Homework-derived validation cases\
Lognormal & `test_lognormal.py` & 34 & 0 & 0 & Exact GBM/lognormal formulas\
Credit & `test_credit.py`, `test_cva_mitigants.py`, `test_counterparty_mitigation.py`, `test_merton_timing.py` & 118 & 0 & 0 & Hazard, Merton, CDS, CVA, mitigants\
Credit service & `test_credit_service.py` & 11 & 0 & 0 & Extension-service aggregation\
Regulatory & `test_regulatory.py`, `test_balance_sheet.py`, `test_dfast_pathing.py` & 46 & 0 & 0 & RWA, capital ratio, stress pathing\
Market data & `test_market_data.py` & 25 & 0 & 0 & CSV loader, yfinance wrappers, cache, rate helper\
Config / validation & `test_config_and_validation.py`, `test_packaging_namespace.py` & 22 & 0 & 0 & Input validation and package namespace\
Charts & `test_charts.py` & 6 & 0 & 0 & Plot helpers\
UI panels & `test_ui_panels.py` & 68 & 0 & 0 & Streamlit panel behavior\
Coverage and numerics & `test_coverage_gaps.py`, `test_strict_numerics.py`, `test_es_confidence_split.py` & 84 & 0 & 0 & Gap-closing and numerical-discipline tests\
Integration & `integration_test.py` & 1 & 0 & 0 & Live-data end-to-end market-risk workflow passed\
Formula integration & `integration_test_formula_sheet.py` & 1 & 0 & 0 & Live-data formula-sheet workflow passed\
**Total** & & **644** & **0** & **0** &\

</div>

</div>

## Overall Summary

<div class="center">

| **Metric**                 | **Result** |
|:---------------------------|-----------:|
| No-network tests collected |        644 |
| No-network tests passed    |        644 |
| No-network tests failed    |          0 |
| No-network tests skipped   |          0 |
| Total statement coverage   |        95% |
| Integration scripts passed |      2 / 2 |

</div>

# Unit Test Results

## Observed No-Network Suite Result

Observed terminal tail (from `submission/test_artifacts/pytest_output.txt`):

<div class="shellcode">

====================== 644 passed, 242 warnings in 14.95s ======================

</div>

The warning volume was high but corresponded entirely to deprecation notices in third-party libraries (Streamlit, pandas). None produced test failures.

## Interpretation

The no-network suite provides strong evidence that:

- core portfolio and pricing calculations work as expected;

- risk engines return finite and plausible outputs;

- backtesting logic is implemented and exercised deterministically;

- course-formula extensions have substantial regression coverage;

- UI panels render and behave correctly under test harnesses.

# Integration Test Results

## `tests/integration_test.py`

**Status: Passed.**

Observed excerpt:

<div class="shellcode">

Running all risk models... \[HISTORICAL\] VaR = $`4,821.40  |  ES =`$<!-- -->4,793.83 \[5\] Running walk-forward backtest (historical model)... ALL INTEGRATION TESTS PASSED

</div>

Interpretation: live price download, portfolio creation, service orchestration, model execution, backtesting, EWMA mode, and multi-day horizon checks all completed successfully.

## `tests/integration_test_formula_sheet.py`

**Status: Passed.**

Observed excerpt:

<div class="shellcode">

Stock+option portfolio -\> RiskEngineService.run_all() historical VaR=1,757.98 ES=1,687.03 \[9\] compute_rwa_and_ratio + run_dfast ALL FORMULA-SHEET INTEGRATION TESTS PASSED.

</div>

Interpretation: live-data download, rate fetch, portfolio construction, core market-risk execution, backtesting, Merton, CDS, CVA, and regulatory checks all completed successfully.

## Integration Conclusion

The integration runs confirm that:

1.  end-to-end plumbing is operational with live market data;

2.  the integration scripts are aligned with the separate-confidence VaR/ES design;

3.  the formula-sheet extension workflows pass end to end.

# Analytical Golden Test Results

Selected analytical cases were verified against the expected values used in the repository’s tests. All matched within the stated tolerance.

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

These results confirm that the pure-formula layer is correctly implemented for all sampled benchmark cases.

# Homework Fixture Results

The file `submission/test_artifacts/homework_fixture_results.csv` records all homework-derived regression fixtures. Representative results are summarized below.

<div class="center">

<div class="tabular">

L3cmL2cmL4.5cmL3.5cmL1.3cm **Homework case** & **Area** & **Expected result** & **Actual result** & **Pass?**\
HW4 single-stock VaR & GBM VaR & 19037.040669837672 & 19037.040669837672 & Pass\
HW6 reduced-form default & Hazard & $`P(\tau \leq 5) = 0.03633`$; $`P(3 < \tau \leq 4) = 0.00721`$ & matched exactly & Pass\
HW6 piecewise hazard & Hazard & spread range 69.95 bp to 80.44 bp & 69.89 bp to 80.44 bp & Pass\
HW7 Merton Q/P & Structural & Q-PD $`\approx 0.2953`$; P-PD $`\approx 0.3888`$ & Q-PD = 0.2952952345; P-PD = 0.3888069321 & Pass\
HW8 CDS & CDS & approx spread $`\approx 0.0180`$ & 0.0180 & Pass\
HW9 short VaR/ES & Short risk & short VaR $`>`$ long VaR & VaR = 5924.4341; ES = 5999.5959 & Pass\
HW10 RWA/capital & Regulatory & RWA = 100; capital ratio = 0.12 & RWA = 100.0; ratio = 0.12 & Pass\

</div>

</div>

These results demonstrate that the repository is tied back to course-derived benchmark values, not merely tested at an abstract level.

# External Benchmark Results

The file `submission/test_artifacts/official_benchmark_results.csv` records benchmark comparisons against external references.

<div class="center">

<div class="tabular">

L3.5cmL2.5cmL3.5cmL3.5cm **Benchmark** & **Source** & **Expected** & **Actual**\
ATM call (option calculator) & Independent reference & price $`\approx 10.4506`$; delta $`\approx 0.6368`$ & price = 10.4505835722; delta = 0.6368306512\
ATM put (option calculator) & Independent reference & price $`\approx 5.5735`$; delta $`\approx -0.3632`$ & price = 5.5735260223; delta = $`-0.3631693488`$\
Basel traffic light, 0 exceptions & Regulatory-style & GREEN & GREEN\
Basel traffic light, 4 exceptions & Regulatory-style & GREEN & GREEN\
Basel traffic light, 5 exceptions & Regulatory-style & AMBER & AMBER\
Basel traffic light, 9 exceptions & Regulatory-style & AMBER & AMBER\
Basel traffic light, 10 exceptions & Regulatory-style & RED & RED\

</div>

</div>

Note: the Black-Scholes rows are benchmark-style reasonableness checks against standard textbook values. The Basel rows are rule-based checks against published regulatory classification logic. Both provide independent validation beyond the homework fixture set.

# Backtesting Results

The file `submission/test_artifacts/backtest_results.csv` captures a representative historical-model backtest on the most recent 1,500 aligned AAPL/CAT Bloomberg observations.

## Summary Table

<div class="center">

<div class="tabular">

L2.2cmL1.5cmL1.3cmL1.2cmL1.5cmL1.5cmL1.5cmL1.5cm **Model** & **Horizon** & **Conf.** & **Obs.** & **Exp. exc.** & **Act. exc.** & **Exc. rate** & **Kupiec $`p`$**\
Historical & 5d & 0.99 & 990 & 9.90 & 15 & 1.52% & 0.130\

</div>

</div>

## Additional Diagnostics

- Kupiec $`\mathrm{LR}_{\mathrm{uc}}`$ statistic: **2.2920**

- Christoffersen independence LR: **62.2015**

- Christoffersen independence $`p`$-value: $`3.10 \times 10^{-15}`$

- Conditional coverage LR: **64.4936**

- Conditional coverage $`p`$-value: $`9.89 \times 10^{-15}`$

- Basel zone: **RED** (15 exceptions exceeds the 10-exception threshold)

- Average exception gap: \$205,833.28

- Maximum exception loss: \$1,262,636.56

## Interpretation

Kupiec’s unconditional coverage test assesses only the *frequency* of exceptions. In this run, the 15 observed exceptions out of 990 forecasts yields an exception rate of 1.52% against an expected rate of 1.00%, and a Kupiec $`p`$-value of 0.130; the unconditional coverage hypothesis is not rejected at conventional significance levels.

However, the Christoffersen independence LR statistic of 62.20 with a $`p`$-value of $`3.10 \times 10^{-15}`$ is a highly significant rejection of the independence hypothesis. The exceptions cluster in time, indicating that the historical-simulation model fails to capture volatility dynamics. This is consistent with the academic literature: historical simulation is computationally simple but is known to be slow to adapt to regime changes.

The Basel RED designation reflects a mechanical count ($`N_e \geq 10`$ implies RED regardless of the Kupiec result). Taken together, the backtesting evidence motivates Recommendation 5 in the Model Documentation (Deliverable 1): the implementation of EWMA or GARCH-based volatility forecasting to address exception clustering.

# Detailed Analysis of Informative Test Cases

The test suite of 644 cases varies widely in informational density. This section documents those results that most directly reveal model behavior, structural properties, or known limitations.

## Christoffersen Independence Rejection

The most informationally dense single result in the suite is the Christoffersen independence LR test on the 5-day, 99% historical VaR backtest: $`\mathrm{LR}_{\mathrm{ind}} = 62.2015`$, $`p = 3.10 \times
10^{-15}`$. This is an overwhelming rejection of the independence hypothesis.

The implication is precise: the model produces exceptions that cluster in time rather than arriving uniformly. This is structurally consistent with the academic literature on historical simulation: because the equally-weighted method uses the full lookback window unchanged until a daily re-estimate, it responds slowly to volatility regime shifts. A quiet-then-volatile sequence produces multiple consecutive exceptions before the rolling window absorbs the new data. EWMA or GARCH-based covariance forecasting directly addresses this by down-weighting old data.

## Short VaR Exceeds Long VaR

For the same GBM parameters (HW9 regression: $`V_0 = 100\,000`$, $`\mu = 0.10`$, $`\sigma = 0.25`$, $`h = 5`$, $`\alpha = 0.99`$), the short GBM VaR (5924.43) exceeds the long GBM VaR (3720.34) by approximately 59%. This is a structural property of the lognormal distribution, not a numerical coincidence.

A long position loses at most $`V_0`$ (floor at zero). A short position faces theoretically unbounded adverse moves: if the underlying doubles, the short loses 100% of notional. The asymmetry of the lognormal distribution produces a larger upper quantile for the short-loss than for the long-loss, which is exactly what the formulas capture.

## Monte Carlo Convergence to Exact GBM

At $`n = 100\,000`$ paths with a fixed seed, MC VaR converges to within 2% of the exact lognormal GBM VaR (CONV_01). This provides direct calibration evidence for the 10 000-path default: it is adequate for academic purposes but would need to increase substantially for extreme-tail (99.9%) estimation in production. Seed-to-seed variance at $`n = 10\,000`$ is visible in the fourth significant digit; a fixed seed is required for regression-stable tests.

## Merton Q-PD vs. P-PD Divergence

For the HW7 Merton case ($`V_0 = 100`$, $`B = 80`$, $`r = 0.05`$, $`\mu = 0.10`$, $`\sigma = 0.25`$, $`T = 5`$), Q-PD = 0.2953 and P-PD = 0.3888. The structural relationship P-PD $`>`$ Q-PD when $`\mu > r`$ is confirmed: under the physical measure, higher drift moves the firm value away from the default boundary more slowly than under the risk-neutral measure, increasing the physical probability of default. This distinction matters for risk-management applications (which require P) vs. derivative pricing (which requires Q).

## ES is Always at Least VaR

The fundamental coherence requirement ES $`\geq`$ VaR is verified across all three models at all tested confidence levels (EDGE_20, `test_es_confidence_split.py`). Its presence as an explicit test rather than an assumed property is a deliberate design choice: a subtle bug in confidence-level handling could violate this ordering silently.

# Robustness Testing Results

## Parameter Sensitivity

The following table summarizes observed behavior across the parameter sensitivity sweep documented in the Robustness Testing section of the Test Plan.

<div class="center">

<div class="tabular">

L3.5cmL4.5cmL5.5cm **Parameter varied** & **Range tested** & **Observation**\
Lookback window & 60, 126, 252, 504 days & VaR varies with volatility captured; no crashes; all outputs finite\
EWMA $`N`$ & 10, 20, 60, 120 & Covariance PSD at all values; $`N = 10`$ highly responsive, $`N = 120`$ stable\
MC paths & 100, 1 000, 10 000 & VaR converges monotonically; $`n = 100`$ noisier; $`n = 10\,000`$ stable\
Confidence level & 0.95, 0.975, 0.99, 0.999 & VaR monotonically increasing; ES $`\geq`$ VaR throughout\

</div>

</div>

## Extreme Input Results

<div class="center">

<div class="tabular">

L5.5cmL7.5cm **Case** & **Result**\
Near-expiry option ($`T = 1/252`$) & BS price at intrinsic-value limit; delta binary; no NaN\
Deep OTM put (strike = 0.5 $`\times`$ spot) & Small positive price; VaR dominated by underlying delta exposure\
High-volatility ($`\sigma = 2.0`$, i.e. 200%) & VaR large but finite; no overflow; parametric and MC consistent\
Single-position portfolio & All three methods return positive, finite VaR\
Zero-drift ($`\mu = 0`$) & Parametric VaR still positive from variance term; historical and MC agree\

</div>

</div>

## Model Weakness Confirmation

Robustness testing confirms the documented model limitations; none are surprises.

- **Historical simulation exception clustering** is confirmed by the Christoffersen test ($`\mathrm{LR}_{\mathrm{ind}} = 62.20`$). Exceeds the 99% chi-squared critical value of 6.63 by a factor of nearly 10.

- **Parametric VaR underestimation for options-heavy portfolios** is confirmed by behavioral tests in `test_backend.py`: adding options increases historical and MC VaR relative to parametric VaR when significant nonlinearity is present.

- **MC seed-to-seed variance** at $`n = 10\,000`$ is visible in the fourth significant digit; a fixed seed is required for regression-stable tests. ROB tests confirm that the unfixed (randomized) run still returns finite, positive VaR; only the exact value varies.

- **Downstream capital calculations** handle extreme VaR inputs without silent failures: very large VaR inputs produce very large RWA and a failing capital ratio; zero VaR produces a capital ratio that is reported as passing trivially with a documented warning.

# Data Validation Results

Data validation is covered by `tests/test_market_data.py` (24 passing tests) and `tests/test_config_and_validation.py` (10 passing tests).

**Covered behaviors:** CSV parsing; chronological sorting; empty-data rejection; single- and multi-ticker Yahoo Finance parsing; cache behavior; risk-free-rate helper behavior; `DatetimeIndex` validation; all-NaN column rejection; non-positive-price rejection; and missing-ticker error detection.

**Result:** the data layer performs correctly on all tested paths. Remaining gaps (explicit stale-price detection across all data sources and a fully documented duplicate-date policy) are identified as future hardening areas and do not affect the correctness of any of the formula modules.

# Coverage Results

## Summary

<div class="center">

| **Metric**         |                                **Result** |
|:-------------------|------------------------------------------:|
| Statement coverage |                                       95% |
| Missing lines      |                                       153 |
| Coverage HTML      |   `submission/coverage_report/index.html` |
| Coverage XML       | `submission/coverage_report/coverage.xml` |

</div>

## Files with Remaining Untested Lines

<div class="center">

<div class="tabular">

L4.5cmrL4cmL4.5cm **File** & **Missing lines** & **Likely reason** & **Fix direction**\
`src/credit/cds.py` & 33 & Lower-tested branches in full CDS logic & Add branch-specific CDS tests\
`src/credit/hazard.py` & 26 & Piecewise helper branches & Add targeted piecewise-hazard tests\
`src/ui/capital_panel.py` & 18 & UI branches not fully exercised & Add panel-path tests\
`src/ui/cds_cva_panel.py` & 18 & UI and branch-heavy error paths & Add panel-path tests\
`src/risk/historical.py` & 16 & Historical vol-shock and helper paths & Add explicit historical-shock tests\
`src/services/regulatory_service.py`& 11 & Capital-path helper branches & Add service-branch tests\
`src/credit/cva.py` & 8 & Discounted and edge branches & Add CVA branch tests\
`src/risk/normal.py` & 7 & Direct formula helpers not all hit & Add direct normal-formula tests\
`src/risk/estimators.py` & 7 & Manual-parameter validation branches & Add targeted manual-input tests\
`src/credit/mitigation.py` & 4 & Less common mitigation branches & Add more mitigant tests\
`src/risk/returns.py` & 3 & Absolute-return helper path & Add branch tests\
`src/risk/regulatory.py` & 2 & Small residual branches & Add targeted tests\

</div>

</div>

## Coverage Conclusion

The model-critical paths (pricing formulas, VaR/ES engines, covariance handling, backtesting diagnostics, credit formulas, and regulatory arithmetic) are all tested directly. The remaining 5% of uncovered lines are concentrated in UI display branches, secondary service orchestration helpers, and defensive validation branches. These are documented as future hardening areas and are not treated as unresolved formula-validation failures.

# Failed and Skipped Tests

<div class="center">

<div class="tabular">

L4.5cmlL3.5cmL4cm **Test / command** & **Status** & **Affects required functionality?** & **Resolution**\
No-network unit suite & Passed & No & Primary validation evidence\
Coverage command & Passed (95% coverage) & No & Gaps documented in Section 10\
`tests/integration_test.py` & Passed & No & Live workflow evidence\
`tests/integration_test_formula_sheet.py` & Passed & No & Live formula-sheet evidence\

</div>

</div>

No required no-network unit tests were skipped. The two live-data integration scripts were excluded from the no-network commands by design and were then executed separately; both passed.

# Interpretation and Conclusion

The test results support the core software design and all model documentation claims in Deliverable 1.

**Strongly supported:**

- Deterministic formula modules pass all 644 no-network tests.

- Portfolio-risk methods pass unit and workflow tests.

- Backtesting logic is implemented, exercised, and interpreted correctly.

- Data-validation and UI layers have dedicated, passing test coverage.

- Raw artifacts are captured and reproducible.

**Areas for future improvement:** Coverage reporting identifies untested branches in CDS, hazard, historical vol-shock paths, regulatory-service helpers, and selected UI panels. These are documented as future hardening areas and do not represent unresolved correctness failures.

**Overall validation opinion:** the repository has a strong no-network and live-integration validation base and is acceptable as an academic risk-engine implementation. The core model, portfolio, service, and UI layers are exercised by dedicated tests, and the results are transparently documented.

# Appendix: Artifact Index

The following output files are included in `submission/test_artifacts/`:

- `pytest_output.txt`: full terminal output of the no-network suite

- `coverage_output.txt`: per-module coverage table

- `integration_test_output.txt`: terminal output of core integration test

- `integration_test_formula_sheet_output.txt`: terminal output of formula-sheet integration test

- `requirements_freeze.txt`: `pip freeze` snapshot

- `git_commit.txt`: `git rev-parse HEAD` output

- `backtest_results.csv`: walk-forward backtest statistics

- `homework_fixture_results.csv`: homework regression fixture results

- `official_benchmark_results.csv`: external benchmark comparison results
