# Deliverable 4/5: Test Results Report

## 1. Executive Summary

The test results show a strong outcome for the repository’s no-network validation suite and a mixed outcome for full integration and coverage-gate enforcement.

Observed local no-network suite result:

- `569 passed`
- `0 failed`
- `0 skipped`

Observed strict coverage-gate result:

- all no-network tests still passed,
- but the coverage command failed because total statement coverage reached only `92.49%` against the stated `100%` target.

Observed integration-script result:

- `tests/integration_test.py` failed
- `tests/integration_test_formula_sheet.py` failed

Both integration failures came from the same assumption: the scripts assert `ES >= VaR` even when the repository is configured with separate VaR and ES confidence levels. In the current design, a `97.5%` ES can be lower than a `99%` VaR, so these failures appear to reflect an outdated integration expectation rather than a general breakdown of the core risk engine.

Overall conclusion: the deterministic and no-network validation evidence strongly supports the core software design and formula implementations, but the coverage target is not yet met and the live integration scripts should be updated before final submission.

---

## 2. Test Environment

### 2.1 Run Metadata

- Date/time of observed run: `2026-05-10 04:54:54 EDT`
- Git commit hash: `50143fde53c1d6b4d9bee277b96d97c2ef870dca`
- Python version: `3.12.2`
- OS: `Darwin 24.5.0 arm64`
- Network status: enabled for integration-script execution
- Data files present:
  - `data/AAPL-bloomberg.csv`
  - `data/CAT-bloomberg.csv`

### 2.2 Key Package Versions

- `streamlit 1.37.1`
- `numpy 1.26.4`
- `pandas 3.0.2`
- `scipy 1.17.1`
- `plotly 5.24.1`
- `yfinance 1.2.0`
- `pytest 7.4.4`
- `pytest-cov 7.1.0`

### 2.3 Captured Environment Artifacts

The following files were written to `test_artifacts/`:

- `git_commit.txt`
- `python_version.txt`
- `requirements_freeze.txt`
- `pytest_output.txt`
- `coverage_output.txt`
- `integration_test_output.txt`
- `integration_test_formula_sheet_output.txt`
- `per_file_test_counts.json`
- `homework_fixture_results.csv`
- `official_benchmark_results.csv`
- `backtest_results.csv`

---

## 3. Test Commands

### 3.1 No-Network Unit Suite

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py -v
```

### 3.2 Strict Coverage Run

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=100 --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

### 3.3 Integration Scripts

```bash
python tests/integration_test.py
python tests/integration_test_formula_sheet.py
```

### 3.4 Environment Capture Commands

```bash
git rev-parse HEAD
python --version
pip freeze > test_artifacts/requirements_freeze.txt
```

---

## 4. Test Execution Summary

### 4.1 Top-Level Result Table

| Test group | File(s) | Passed | Failed | Skipped | Notes |
|---|---|---:|---:|---:|---|
| Core backend | `test_backend.py` | 25 | 0 | 0 | Core pricing, portfolio, VaR/ES, service smoke tests |
| Backtest extensions | `test_backtest_extensions.py` | 31 | 0 | 0 | Christoffersen, conditional coverage, traffic light, severity |
| Course validation | `test_course_validation.py` | 67 | 0 | 0 | Course formula-sheet and regression fixtures |
| Homework fixtures | `test_homework_cases.py` | 83 | 0 | 0 | Additional homework-derived validation cases |
| Lognormal | `test_lognormal.py` | 34 | 0 | 0 | Exact GBM/lognormal formulas |
| Credit | `test_credit.py`, `test_cva_mitigants.py`, `test_counterparty_mitigation.py`, `test_merton_timing.py` | 118 | 0 | 0 | Hazard, Merton, CDS, CVA, mitigants |
| Credit service | `test_credit_service.py` | 11 | 0 | 0 | Extension-service aggregation |
| Regulatory | `test_regulatory.py`, `test_balance_sheet.py`, `test_dfast_pathing.py` | 46 | 0 | 0 | RWA, capital ratio, stress pathing |
| Market data | `test_market_data.py` | 24 | 0 | 0 | CSV loader, yfinance wrappers, cache, rate helper |
| Config/validation | `test_config_and_validation.py` | 10 | 0 | 0 | Input and data validation |
| Charts | `test_charts.py` | 6 | 0 | 0 | Plot helpers |
| UI panels | `test_ui_panels.py` | 67 | 0 | 0 | Streamlit panel behavior |
| Coverage and numerics | `test_coverage_gaps.py`, `test_strict_numerics.py`, `test_es_confidence_split.py` | 47 | 0 | 0 | Gap-closing and numerical-discipline tests |
| Integration | `integration_test.py` | 0 | 1 | 0 | Failed on historical `ES >= VaR` assertion |
| Formula integration | `integration_test_formula_sheet.py` | 0 | 1 | 0 | Failed on historical `ES >= VaR` assertion |

### 4.2 Overall Summary

| Metric | Result |
|---|---:|
| No-network tests collected | 569 |
| No-network tests passed | 569 |
| No-network tests failed | 0 |
| No-network tests skipped | 0 |
| Coverage gate target | 100% |
| Actual total statement coverage | 92.49% |
| Integration scripts passed | 0 / 2 |

---

## 5. Unit Test Results

### 5.1 No-Network Suite Outcome

Observed tail of `test_artifacts/pytest_output.txt`:

```text
====================== 569 passed, 242 warnings in 15.28s ======================
```

The warning volume was high but did not correspond to test failures. Most warnings came from dependency-version and deprecation notices in third-party libraries used by Streamlit or pandas.

### 5.2 Unit-Suite Interpretation

The no-network suite provides strong evidence that:

- core portfolio and pricing calculations work as expected,
- risk engines return finite and plausible outputs,
- backtesting logic is implemented and testable,
- course-formula extensions have substantial regression coverage,
- UI panels render and behave correctly under test harnesses.

---

## 6. Integration Test Results

### 6.1 `tests/integration_test.py`

Status: `Failed`

Observed failure excerpt:

```text
[4] Running all risk models...
    [HISTORICAL]  VaR = $4,821.40  |  ES = $4,793.83
AssertionError: historical ES must be >= VaR
```

Interpretation:

- The script reached live data download, portfolio creation, service instantiation, and model execution successfully.
- Failure occurred only at the assertion that historical `ES >= VaR`.
- Under the current repository design, VaR and ES confidence levels are separate, so this assertion is not always valid.

### 6.2 `tests/integration_test_formula_sheet.py`

Status: `Failed`

Observed failure excerpt:

```text
[3] Stock+option portfolio → RiskEngineService.run_all()
    historical   VaR=1,757.98  ES=1,687.03
AssertionError: historical ES < VaR
```

Interpretation:

- Live-data download, rate fetch, portfolio construction, and service execution all worked.
- The same outdated assumption caused failure.

### 6.3 Integration-Test Conclusion

These are meaningful failures, but they do not point to a complete workflow crash. Instead, they indicate that:

1. the end-to-end plumbing is operational up to the assertion point, and
2. the integration scripts need to be reconciled with the current separate-confidence VaR/ES design.

---

## 7. Analytical Golden Test Results

Selected analytical cases were recomputed from the current workspace and matched the expected values used in the repository’s tests.

| Case ID | Module | Expected | Actual | Abs error | Tolerance | Pass |
|---|---|---:|---:|---:|---|---|
| `BS_01` | Black-Scholes call | 10.4505835722 | 10.4505835722 | ~0 | analytic / rounding | Yes |
| `BS_02` | Black-Scholes put | 5.5735260223 | 5.5735260223 | ~0 | analytic / rounding | Yes |
| `LN_01` | Long GBM VaR | 3720.342013894248 | 3720.342013894248 | 0 | course fixture | Yes |
| `LN_02` | Short GBM VaR | 5924.434136581646 | 5924.434136581646 | 0 | course fixture | Yes |
| `HZ_01` | Hazard survival `s(5)` | 0.9636761353490535 | 0.9636761353490535 | 0 | course fixture | Yes |
| `MR_01` | Merton Q-PD | 0.2952952345271121 | 0.29529523452711204 | ~0 | course fixture | Yes |
| `CDS_01` | Flat-hazard CDS spread | 0.0180 | 0.0180 | 0 | formula benchmark | Yes |
| `CVA_01` | Discrete CVA | 1.92 | 1.92 | 0 | deterministic arithmetic | Yes |
| `REG_01` | Capital ratio | 0.12 | 0.12 | 0 | deterministic arithmetic | Yes |

These values support the claim that the pure-formula layer is implemented correctly for the sampled benchmark cases.

---

## 8. Homework Fixture Results

The file [homework_fixture_results.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/homework_fixture_results.csv) was generated during this pass. Representative rows are summarized below.

| Homework case | Area | Expected result | Actual result | Pass? |
|---|---|---|---|---|
| HW4 single-stock VaR | GBM VaR | `19037.040669837672` | `19037.040669837672` | Pass |
| HW6 reduced-form default | Hazard | `P(tau<=5)=0.0363238646509465`, `P(3<tau<=4)=0.0072108171597774495` | matched exactly | Pass |
| HW6 piecewise hazard | Hazard table | spread range `69.95bp` to `80.44bp` | `69.89bp` to `80.44bp` | Pass |
| HW7 Merton Q/P | Structural credit | `Q-PD≈0.295295`, `P-PD≈0.3888` | `Q-PD=0.2952952345`, `P-PD=0.3888069321` | Pass |
| HW8 CDS | CDS | approx spread `0.018` | `0.018` | Pass |
| HW9 short VaR/ES | Short risk | short VaR > long VaR | `VaR=5924.4341`, `ES=5999.5959` | Pass |
| HW10 RWA/capital | Regulatory | `RWA=100`, `capital ratio=0.12` | `RWA=100.0`, `ratio=0.12` | Pass |

These results reinforce that the repository is not just unit-tested at an abstract level; it is tied back to course-derived benchmark values.

---

## 9. External/Official Benchmark Results

The file [official_benchmark_results.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/official_benchmark_results.csv) was generated during this pass.

| Benchmark | Source type | Module | Expected behaviour | Result |
|---|---|---|---|---|
| Option calculator style ATM call | Independent option-pricing reference | Black-Scholes | Call price near `10.4506`; delta near `0.6368` | `price=10.4505835722`, `delta=0.6368306512` |
| Option calculator style ATM put | Independent option-pricing reference | Black-Scholes | Put price near `5.5735`; delta near `-0.3632` | `price=5.5735260223`, `delta=-0.3631693488` |
| Basel traffic light 0 exceptions | Regulatory-style benchmark | Backtesting | Green | `GREEN` |
| Basel traffic light 4 exceptions | Regulatory-style benchmark | Backtesting | Green | `GREEN` |
| Basel traffic light 5 exceptions | Regulatory-style benchmark | Backtesting | Amber/yellow | `YELLOW` |
| Basel traffic light 9 exceptions | Regulatory-style benchmark | Backtesting | Amber/yellow | `YELLOW` |
| Basel traffic light 10 exceptions | Regulatory-style benchmark | Backtesting | Red | `RED` |

Important nuance:

- The Black-Scholes rows are benchmark-style reasonableness checks using standard textbook values.
- They were not independently scraped from an external calculator during this run.
- The Basel rows are effectively independent rule-based checks against regulatory-style classification logic.

---

## 10. Backtesting Results

The file [backtest_results.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/backtest_results.csv) captures a representative historical-model backtest on the most recent `1,500` aligned AAPL/CAT Bloomberg observations.

### 10.1 Summary Table

| Model | Horizon | Confidence | Observations | Expected exceptions | Actual exceptions | Exception rate | Kupiec p-value | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| historical | 5 | 0.99 | 990 | 9.90 | 15 | 0.01515 | 0.1300 | No unconditional-coverage rejection |

### 10.2 Additional Diagnostics from the Same Run

These extra values were computed during the documentation pass:

- Kupiec LR statistic: `2.2920`
- Christoffersen independence LR: `62.2015`
- Christoffersen independence p-value: `3.10e-15`
- Conditional coverage LR: `64.4936`
- Conditional coverage p-value: `9.89e-15`
- Basel zone: `RED`
- Average exception gap: `$205,833.28`
- Maximum exception loss: `$1,262,636.56`

### 10.3 Interpretation

A model with too many exceptions may underestimate risk; a model with too few may be overly conservative. Kupiec assesses unconditional coverage but does not by itself test clustering or dependence. In this representative run:

- unconditional coverage was not rejected,
- but exception clustering was severe,
- so the broader backtesting picture is materially weaker than Kupiec alone suggests.

That makes the Christoffersen-style diagnostics already present in the repository particularly valuable.

---

## 11. Data Validation Results

Data validation is covered primarily by:

- `tests/test_market_data.py` with `24` passing tests
- `tests/test_config_and_validation.py` with `10` passing tests

### 11.1 Covered Behaviors

These tests exercise:

- CSV parsing
- chronological sorting
- empty-data rejection
- single- and multi-ticker Yahoo Finance parsing
- cache behavior
- risk-free-rate helper behavior
- DatetimeIndex validation
- all-NaN column rejection
- non-positive-price rejection
- missing-ticker error detection

### 11.2 Result Interpretation

The data layer appears strong on:

- basic input structure,
- malformed-file rejection,
- missing-series detection,
- positivity checks,
- yfinance response-shape handling.

Remaining gaps are more about richer data-quality logic, such as explicit stale-price detection or documented duplicate-date policy.

---

## 12. Coverage Results

### 12.1 Coverage Summary Table

| Metric | Result | Target | Pass? |
|---|---:|---:|---|
| Statement coverage | 92.49% | 100% | No |
| Branch coverage | Not measured in this command | Not stated | N/A |
| Missing lines | 144 | 0 or justified | No |
| Coverage HTML | `htmlcov/index.html` generated | Required | Yes |
| Coverage XML | `coverage.xml` generated | Required | Yes |

### 12.2 Coverage-Failure Excerpt

Observed tail of `test_artifacts/coverage_output.txt`:

```text
ERROR: Coverage failure: total of 92 is less than fail-under=100
FAIL Required test coverage of 100% not reached. Total coverage: 92.49%
```

### 12.3 Files with Missing Coverage

| File | Missing lines | Why missing (likely) | Fix direction |
|---|---|---|---|
| `src/credit/cds.py` | 33 | Lower-tested branches in full CDS logic | Add branch-specific CDS tests |
| `src/credit/hazard.py` | 26 | Piecewise helper branches | Add targeted piecewise-hazard tests |
| `src/ui/capital_panel.py` | 18 | UI branches not fully exercised | Add panel-path tests |
| `src/ui/cds_cva_panel.py` | 18 | UI and branch-heavy error paths | Add panel-path tests |
| `src/risk/historical.py` | 14 | Absolute-shock and branch paths | Add explicit historical-shock tests |
| `src/services/regulatory_service.py` | 11 | Capital-path helper branches | Add service-branch tests |
| `src/credit/cva.py` | 8 | Discounted / edge branches | Add CVA branch tests |
| `src/risk/normal.py` | 7 | Direct formula helpers not all hit | Add direct normal-formula tests |
| `src/credit/mitigation.py` | 4 | Less common mitigation branches | Add more mitigant tests |
| `src/risk/returns.py` | 3 | Absolute-return helper path | Add branch tests |
| `src/risk/regulatory.py` | 2 | Small residual branches | Add target tests |

### 12.4 Coverage Conclusion

Coverage is good for an academic repository, but it does not satisfy the formal target stated in the README. That gap should be disclosed plainly in the final submission.

---

## 13. Failed or Skipped Tests

### 13.1 Status Table

| Test / command | Status | Reason | Does it affect required functionality? | Resolution |
|---|---|---|---|---|
| No-network unit suite | Passed | N/A | No negative effect | Keep as primary validation evidence |
| Strict coverage command | Failed | Coverage `92.49% < 100%` target | Yes, affects stated coverage target | Add missing branch tests |
| `tests/integration_test.py` | Failed | Historical `ES >= VaR` assertion invalid under separate VaR/ES confidence levels | Partially; integration script needs update | Update integration expectations |
| `tests/integration_test_formula_sheet.py` | Failed | Same `ES >= VaR` assumption | Partially; integration script needs update | Update integration expectations |

### 13.2 Skip Statement

No required no-network unit tests were skipped in the observed unit-suite run. The only excluded checks were the two live-data integration scripts when running the no-network commands, and those were then executed separately and documented above.

---

## 14. Interpretation and Conclusion

The test results support the core software design and most of the model documentation claims.

What is strongly supported:

- deterministic formula modules passed their goldens,
- the portfolio-risk methods passed unit and workflow tests,
- backtesting logic is implemented and exercised,
- data-validation and UI layers are both covered by dedicated tests,
- raw artifacts were captured for reproducibility.

What remains incomplete:

- the stated 100% coverage target was not met,
- both live integration scripts contain a now-invalid assumption about ES and VaR ordering when confidence levels differ,
- some extension-module and UI branches remain uncovered.

Overall conclusion:

The current repository has a strong no-network validation base and is acceptable as an academic risk-engine implementation, but it is not yet at the final testing standard implied by the strict `--cov-fail-under=100` target. The main remaining work is to close coverage gaps and bring the integration scripts into alignment with the current separate-confidence VaR/ES design.

---

## 15. Appendix: Raw Terminal Outputs

The following raw outputs should be included or linked when converting this markdown into the final submission package:

- [pytest_output.txt](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/pytest_output.txt)
- [coverage_output.txt](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/coverage_output.txt)
- [integration_test_output.txt](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/integration_test_output.txt)
- [integration_test_formula_sheet_output.txt](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/integration_test_formula_sheet_output.txt)
- [requirements_freeze.txt](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/requirements_freeze.txt)
- [git_commit.txt](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/git_commit.txt)
- [backtest_results.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/backtest_results.csv)
- [homework_fixture_results.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/homework_fixture_results.csv)
- [official_benchmark_results.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/test_artifacts/official_benchmark_results.csv)
