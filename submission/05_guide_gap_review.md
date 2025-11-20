# Guide Gap Review

This memo compares the current repository and the packaged submission reports against the coursework guide summarized in [docs/references/README.md](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/docs/references/README.md).

It is not a graded deliverable by itself. It is a working review to answer three questions:

1. What core project requirements are already implemented?
2. What implementation areas are still missing, weak, or ambiguous?
3. What report additions were needed or may still be worth adding?

---

## 1. Core Requirement Status

| Guide requirement | Current status | Evidence | Gap status |
|---|---|---|---|
| Accept portfolio of stock and option positions | Implemented | `src/schemas.py`, `src/ui/portfolio_editor.py`, backend/UI tests | No major gap |
| Calibrate to historical data | Implemented | Historical, parametric, and MC methods all estimate from prices | No major gap |
| Accept parameters as direct input | Implemented with method-specific limits | Manual daily mean/covariance input mode is available for parametric and Monte Carlo; historical simulation still requires price history by construction | No major gap if documented clearly |
| Compute historical VaR | Implemented | `src/risk/historical.py` | No major gap |
| Compute parametric VaR | Implemented | `src/risk/parametric.py` | No major gap |
| Compute Monte Carlo VaR | Implemented | `src/risk/monte_carlo.py` | No major gap |
| Compute historical ES | Implemented | `src/risk/historical.py` | No major gap |
| Compute Monte Carlo ES | Implemented | `src/risk/monte_carlo.py` | No major gap |
| Backtest computed VaRs against history | Implemented | `src/risk/backtest.py`, service layer, app backtest tab | No major gap |
| Model documentation | Implemented | `submission/00` and `submission/01` | Ready, subject to normal proofreading |
| Software design documentation | Implemented | `submission/02` | Ready |
| Test plan | Implemented | `submission/03` | Ready |
| Test results | Implemented | `submission/04`, artifact bundle | Ready, but includes honest failures/gaps |

Bottom line: the core required market-risk workflow is present. The remaining caution is not missing functionality, but documenting clearly that historical simulation still depends on actual historical scenarios even though the parametric and Monte Carlo engines now support manual mean/covariance input.

---

## 2. Missing or Weak Implementation Areas

These are the main implementation items that remain incomplete or still worth calling out relative to the guide and grading notes.

### 2.1 High Priority

| Item | Why it matters | Current state |
|---|---|---|
| Coverage target | README advertises 100% statement coverage | Current observed coverage is `91.22%`, so the strict gate still fails |
| Simplified option-volatility shock model | The guide warns against weak option-volatility modelling | Implemented as `fixed` or simplified `underlying_beta`, but not a full vol surface |
| Distributed validation controls | Report wording must match the actual code | Validation is spread across UI, loaders, manual-parameter checks, and numerical modules rather than one schema layer |

### 2.2 Medium Priority

| Item | Why it matters | Current state |
|---|---|---|
| Richer manual calibration workflow | Could improve usability if the guide is interpreted very strictly | Manual mean/covariance is implemented, but only for parametric and Monte Carlo methods |
| Stronger centralized input validation | Helps software-design credibility and reduces report/code mismatch | Validation remains distributed across UI, loader, pricing, and manual-parameter checks |
| Broader coverage in extension/UI branches | Tightens the testing story | Lowest coverage remains in CDS, hazard, historical branch paths, regulatory-service, and selected UI panels |

### 2.3 Lower Priority / Nice to Have

| Item | Why it matters | Current state |
|---|---|---|
| Stale-price/outlier diagnostics | Strengthens data-quality discussion | Not a major dedicated implemented subsystem |
| Broader backtesting UI exposure | Christoffersen/Basel/severity exist in code but not all are surfaced prominently in the main app flow | Implemented in backend, only partially surfaced in UI |
| More direct benchmark automation | Nice independent evidence | Some benchmark-style cases exist, but not a broad automated external-benchmark layer |

---

## 3. Report Gaps That Were Present and Are Now Addressed

These were weaknesses in the packaged reports that have now been tightened.

| Item | Previous status | Current status |
|---|---|---|
| Combined all-in-one report | Missing from `submission/` | Added as `submission/00_combined_final_report.md` |
| Screenshot placeholders in model documentation | Placeholder-only | Replaced with actual representative screenshots |
| Stale test metadata in submission docs | Older date, commit, and timings | Updated to the latest observed repo pass |
| Overstated validation language in software design doc | Too strict relative to code | Softened to match distributed enforcement reality |
| Integration-status mismatch in reports | Older package still said integration failed | Refreshed so all submission docs now show both integration scripts passing |

---

## 4. Report Additions Still Worth Considering

These are not necessarily mandatory, but they could further improve the submission if there is time.

| Suggested addition | Why it may help | Priority |
|---|---|---|
| A short “marker note” in the combined report intro explaining that `submission/00` is the integrated version and `submission/01-04` are segmented deliverables | Makes the package easier to navigate | Low |
| A brief explicit statement in the model report that the core app is primarily history-calibrated rather than a full manual mean/covariance tool | Helps with the “direct input” interpretation issue | Medium |
| A one-paragraph note in the test results report explaining that warnings were dependency warnings, not model/test failures | Improves scanability for a marker | Low |

---

## 5. Recommended Implementation Order If We Decide To Code More

If there is time for further non-report work, this is the order that gives the best payoff relative to grading risk.

1. Add targeted tests to improve coverage in `src/credit/cds.py`, `src/credit/hazard.py`, `src/risk/historical.py`, `src/services/regulatory_service.py`, and selected UI branches.
2. If desired, enrich the current simplified option-volatility shock into a more structured implied-volatility stress model.
3. If desired, add more user-friendly export/import helpers around the new manual mean/covariance path.
4. Keep the report language aligned with the fact that direct-input support is now present for parametric and Monte Carlo, while historical simulation still needs actual history.

---

## 6. Practical Submission Readiness View

If submission happened now:

- the report package would be credible and organized,
- the core required functionality would be present,
- the testing evidence would be strong for coursework,
- but the package would still carry a few honest caveats:
  - 91.22% coverage instead of 100%,
  - simplified rather than full-surface option-volatility shocks,
  - distributed rather than fully centralized validation controls,
  - remaining low-coverage areas in CDS, hazard, historical branch paths, regulatory-service helpers, and selected UI panels.

That means the current state is submission-capable, but not yet “nothing left to question.”
