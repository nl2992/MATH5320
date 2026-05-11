# Prompt Coverage Matrix

This memo maps the current repository and refreshed submission package against the core project prompt summarized in [docs/references/README.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/docs/references/README.md).

## 1. Core Software Requirements

| Prompt requirement | Current status | Evidence | Remaining action |
|---|---|---|---|
| Accept a portfolio of stock and option positions | Implemented | `src/schemas.py`, `src/ui/portfolio_editor.py`, `tests/test_backend.py`, `tests/test_ui_panels.py` | None beyond normal proofreading in reports |
| Calibrate to historical data | Implemented | `src/risk/returns.py`, `src/risk/estimators.py`, `src/risk/historical.py`, `src/risk/parametric.py`, `src/risk/monte_carlo.py` | None |
| Accept parameters as direct input | Implemented with method-specific limits | `src/ui/risk_settings.py`, `src/risk/estimators.py::manual_mean_cov`, `src/services/risk_engine_service.py`, `tests/test_backend.py`, `tests/test_ui_panels.py`, `tests/test_coverage_gaps.py` | Keep reports explicit that historical simulation still requires actual price history by construction |
| Compute historical VaR | Implemented | `src/risk/historical.py`, `tests/test_backend.py`, `tests/test_homework_cases.py` | None |
| Compute parametric VaR | Implemented | `src/risk/parametric.py`, corrected delta-dollar exposure in `src/portfolio/positions.py`, `tests/test_backend.py` | None |
| Compute Monte Carlo VaR | Implemented | `src/risk/monte_carlo.py`, `tests/test_backend.py`, `tests/integration_test.py` | None |
| Compute historical ES | Implemented | `src/risk/historical.py`, `tests/test_backend.py`, `tests/test_es_confidence_split.py` | None |
| Compute Monte Carlo ES | Implemented | `src/risk/monte_carlo.py`, `tests/test_backend.py`, `tests/test_es_confidence_split.py` | None |
| Backtest computed VaRs against history | Implemented and extended beyond minimum | `src/risk/backtest.py`, `tests/test_backend.py`, `tests/test_backtest_extensions.py`, `tests/integration_test.py` | None |

## 2. Deliverable Coverage

| Prompt deliverable | Current status | Evidence | Remaining action |
|---|---|---|---|
| Model documentation | Implemented | [submission/01_model_documentation.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/01_model_documentation.md) | Convert to Word/PDF if needed |
| Software design documentation | Implemented | [submission/02_software_design_documentation.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/02_software_design_documentation.md) | Convert to Word/PDF if needed |
| Test plan | Implemented | [submission/03_test_plan.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/03_test_plan.md) | Convert to Word/PDF if needed |
| Software | Implemented | `app.py`, `src/`, `notebooks/`, current `main` branch | None |
| Test results | Implemented and refreshed | [submission/04_test_results.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/04_test_results.md), [submission/test_artifacts](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/test_artifacts), [submission/coverage_report](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/coverage_report) | Convert to Word/PDF if needed |

## 3. Grading-Risk Items Called Out in the Guide

| Guide risk item | Current status | Evidence | Remaining action |
|---|---|---|---|
| Not modeling changes in option volatility | Addressed in simplified form | `src/portfolio/positions.py::shocked_option_volatility`, `src/risk/historical.py`, `src/risk/monte_carlo.py`, `tests/test_backend.py::test_historical_option_vol_shock_changes_result` | Keep reports honest that `underlying_beta` is not a full implied-vol surface |
| Using historical vol instead of implied vol for option pricing | Acceptable for coursework design | Options are priced off user-supplied option vol inputs in `src/schemas.py` / `src/portfolio/positions.py` rather than estimated stock-return volatility | None, but keep wording precise in the report |
| Incorrect covariance calculation | Addressed with tests and manual-input checks | `src/risk/estimators.py`, `tests/test_homework_cases.py`, `tests/test_coverage_gaps.py` | Add more branch tests if targeting 100% coverage |
| Inappropriate parametric VaR design | Addressed and documented | `src/risk/parametric.py`, corrected delta-dollar exposure in `src/portfolio/positions.py`, explicit documentation in model/software-design reports | None beyond keeping the first-order limitation explicit |
| Tests do not support report conclusions | Addressed | `576` no-network tests passed; both live integration scripts passed; refreshed [submission/04_test_results.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/submission/04_test_results.md) | Continue keeping reports synchronized with the latest artifacts |

## 4. Current Quantitative Validation Snapshot

| Item | Current observed result |
|---|---|
| No-network suite | `576 passed, 242 warnings` |
| Live integration | `tests/integration_test.py` passed |
| Formula-sheet integration | `tests/integration_test_formula_sheet.py` passed |
| Total statement coverage | `91.22%` |
| Strict 100% coverage gate | Fails because `91.22% < 100%` |

## 5. Remaining Actions

1. Raise statement coverage if the README’s `100%` target is intended to be enforced strictly.
2. Keep all final report wording explicit that manual direct-input mode applies to the parametric and Monte Carlo engines, while historical simulation still depends on actual scenarios.
3. Keep all final report wording explicit that the implemented option-volatility shock is simplified and course-appropriate, not a full implied-volatility surface model.
4. Convert the refreshed markdown package into the final Word/PDF submission set when the wording is agreed.
