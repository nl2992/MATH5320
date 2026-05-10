# Project Reference Documents

These two PDFs are the authoritative guides for the MATH GR 5320 final project.
**Refer back here whenever there is any doubt about scope, deliverables, or report format.**

---

## 1. `project_requirements.pdf` — Representative Guide

**What it is:** The official Columbia MATH 5320 project specification.

**Key points:**

### Requirements (§2)
The risk calculation system must:
- Accept a portfolio of stock and option positions as input
- Both calibrate to historical data *and* accept parameters as direct input
- Compute **Monte Carlo**, **historical**, and **parametric VaR**
- Compute **Monte Carlo** and **historical ES**
- **Backtest** computed VaRs against history

### Deliverables (§3) — 5 items, graded out of 100 pts

| # | Deliverable | Points |
|---|---|---|
| 1 | Model documentation | 30 |
| 2 | Software design documentation | 15 |
| 3 | Test plan | 20 |
| 4 | Software | 25 |
| 5 | Test results | 10 |

### Grading emphasis
- **Model documentation (30 pts):** completeness, clarity, justification of *all* modelling choices and limitations (modelled on the Stein validation template — see below)
- **Software design doc (15 pts):** clear and complete architecture description
- **Test plan (20 pts):** validates model performance; the test suite we have built maps directly to this
- **Software (25 pts):** complete, correct, well-written; handles arbitrary securities; both history- and parameter-driven modes; all three VaR models + ES + backtest
- **Test results (10 pts):** accurately carries out the test plan and records/evaluates results

### Grading penalties to avoid
- Not modelling changes in volatility for options
- Using historical volatility instead of implied vol to price options
- Incorrect covariance calculation
- Inappropriate choices for parametric VaR
- Tests that don't support the model-doc conclusions

---

## 2. `model_validation_report_example.pdf` — Report Template

**What it is:** Harvey J. Stein, *Model Validation Municipal Bonds* (2014) — the example model validation report cited in the project spec (reference [Ste14a]).  This is the **report format and content standard** we must follow for Deliverable 1.

**Use this as the template for:**
- Structure of the model documentation report
- How to document model assumptions and their limitations
- How to present and justify methodology choices
- How to record test results and link them back to model claims
- The level of rigour expected in the "validation" sections

---

## Quick reference: what our repo covers

| Project requirement | Where it lives |
|---|---|
| Historical VaR / ES | `src/risk/historical.py` |
| Parametric (Delta-Normal) VaR / ES | `src/risk/parametric.py` |
| Monte Carlo VaR / ES | `src/risk/monte_carlo.py` |
| Backtest + Kupiec + Christoffersen | `src/risk/backtest.py` |
| Black-Scholes pricing | `src/pricing/black_scholes.py` |
| Portfolio & option valuation | `src/portfolio/` |
| Credit (hazard, Merton, CDS, CVA) | `src/credit/` |
| Regulatory capital + DFAST | `src/risk/regulatory.py` |
| Lognormal exact VaR / ES | `src/risk/lognormal.py` |
| Test plan + test results | `tests/` |
| Notebooks (formula-sheet walkthroughs) | `notebooks/` |
| Streamlit app (UI) | `app.py` + `src/ui/` |
