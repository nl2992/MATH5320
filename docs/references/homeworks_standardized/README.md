# Standardized Homeworks Reference Pack

This folder is the consolidated reference area for the standardized MATH GR 5320 homeworks that were supplied outside the repo.

## What is in here

- `MATH5320_standardised_homeworks/`
  - The unpacked PDF files from `MATH5320_Homeworks_02-11_Standardised.zip`
- `extracted_text/`
  - Text extracted from each PDF with `pypdf`
  - This is not image OCR in the strictest sense, but the PDFs contain enough embedded text for reliable extraction

## Where the homework solutions already live in this repo

The repo already had homework-derived implementations and validation fixtures in several places:

- `tests/test_homework_cases.py`
  - Additional homework-style numeric checks
- `tests/test_course_validation.py`
  - Course-sheet goldens for lognormal, hazard, Merton, CDS, CVA, regulatory, and AAPL/CAT acceptance targets
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
- `submission/test_artifacts/homework_fixture_results.csv`

In short: the new standardized PDFs now live under `docs/references/homeworks_standardized/`, while the implemented answers already live mainly in `tests/` and `notebooks/`.

## What the project prompt demanded

Per `docs/references/README.md`, the core project requirements are:

1. Accept a portfolio of stock and option positions
2. Calibrate to historical data
3. Accept parameters as direct input
4. Compute historical VaR
5. Compute parametric VaR
6. Compute Monte Carlo VaR
7. Compute historical ES
8. Compute Monte Carlo ES
9. Backtest VaRs against history

The homework set covers almost all of this directly. The one area that is less naturally covered by the standardized PDFs is the required stock-plus-European-option angle for the app itself, so the repo's additional homework-style Black-Scholes and delta-hedge tests should be used alongside the standardized homeworks.

## Recommended comprehensive demo coverage

This is the cleanest "full story" if we want one demo that covers the prompt requirements and then extends into the Phase 2 / formula-sheet material.

| Demo step | Area covered | Homework source | Given answer / target | Repo answer path | Why it belongs |
|---|---|---|---|---|---|
| 1 | Risk-measure definitions and theory | HW03 Q1 | VaR definition, coherence discussion, ES coherence | `MATH5320_Homework_03.txt`, `submission/01_model_documentation.md` | Good opening framing for the marker |
| 2 | European option pricing and delta | Repo homework-style case `HW5_BS_DELTA_FD` | ATM call price `17.62456`, delta `0.664313` | `tests/test_homework_cases.py::TestHW5_BS_DeltaFiniteDiff`, `src/pricing/black_scholes.py` | Covers the stock-and-option requirement explicitly |
| 3 | Option hedge intuition | Repo homework-style case `HW3_INTEL_BSM_DELTA_HEDGE` | Intel call price `5.34508`, delta `0.640605`, about `1873` calls to write | `tests/test_homework_cases.py::TestHW3_IntelBSM_DeltaHedge` | Shows the repo understands option sensitivities, not just prices |
| 4 | Historical scenario VaR and ES | HW03 scenario problem | `VaR_90 = 3931.2`, `ES_80 = 3428.6` | `tests/test_homework_cases.py::TestHW3_ScenarioVaR_ES` | Deterministic, easy first historical-simulation demo |
| 5 | Parameter-driven single-stock GBM VaR | HW04 Q1 | 5-day 99% VaR about `$19,040` | `tests/test_homework_cases.py::TestHW4_SingleStockParamVaR`, `tests/test_course_validation.py::TestLN02_HomeworkIV` | Shows direct parameter input mode and exact GBM logic |
| 6 | Two-stock parametric VaR | HW04 Q2 | 2-week 99% VaR about `$9,007.37` | `tests/test_homework_cases.py::TestHW4_TwoStockNormalVaR` | Clean baseline for covariance-based parametric VaR |
| 7 | Window vs EWMA calibration | HW05 Q1-Q4 | Equivalent lambdas, AAPL/CAT window vs EWMA behavior | `notebooks/04_estimation_rolling_vs_ewma.ipynb`, `tests/test_homework_cases.py::TestHW5_Lambda20PctHeuristic` | Explains estimator choice, a major model-risk decision |
| 8 | Historical AAPL/CAT VaR and ES on real data | HW07 Q2-style AAPL/CAT section | Latest normalized historical 5-day VaR around `$905.78` for AAPL, `$969.13` for CAT, `$898.76` for portfolio | `notebooks/03_historical_shock_methodology.ipynb`, `src/risk/historical.py` | Best real-data historical-simulation demo |
| 9 | Monte Carlo VaR and ES | HW08 Q3-Q4 | Direct MC: `VaR 593`, `ES 850`; component MC: `VaR 587`, `ES 829` on scaled `$10k` setup | `src/risk/monte_carlo.py`, `notebooks/02_aapl_cat_var_es_methods.ipynb`, `notebooks/11_end_to_end_demo.ipynb` | Covers the required MC branch cleanly |
| 10 | VaR backtesting | HW11 | Expected exceptions `96.33`; long/short exception tables and EWMA comparison | `src/risk/backtest.py`, `notebooks/10_backtesting_validation_dashboard.ipynb`, `notebooks/11_end_to_end_demo.ipynb` | This is the required validation leg of the project |
| 11 | Reduced-form credit / hazard | HW06 | `P(tau <= 5) = 3.6324%`, `P(3 < tau <= 4) = 0.7211%`, piecewise spreads `69.95bp` to `80.44bp` | `tests/test_course_validation.py::TestHZ01_ConstantHazard`, `TestHZ02_PiecewiseHazard` | Strong extension module with precise goldens |
| 12 | Merton structural model | HW07 and HW09 | HW07: `PD_Q = 29.53%`, `PD_P = 38.88%`; HW09: `B* = 4,612,960.81` | `tests/test_course_validation.py::TestMR01_HomeworkVII_QvsP`, `TestMR02_TargetSurvivalInversion` | Best structural-credit extension demo |
| 13 | CDS pricing | HW08 | Constant-hazard approx `180bp`; full-form annual par spread `184.55bp` | `tests/test_course_validation.py::TestCDS01_FlatApprox`, `TestCDS02_FullAnnualPaymentParSpread` | Clean pricing extension with known answers |
| 14 | CVA and mitigation | HW08 and HW09 | Discrete CVA around `5.21`; mitigation concepts: netting, collateral, CCP | `tests/test_homework_cases.py::TestHW9_DiscreteCVA`, `src/credit/cva.py`, `src/credit/mitigation.py` | Strong counterparty-risk extension story |
| 15 | Regulatory capital / RWA | HW10 | assets `189,000`, capital `7,000`, RWA `79,850`, capital ratio `8.77%` | `tests/test_homework_cases.py::TestHW10_RWA_Capital`, `src/risk/regulatory.py` | Good ending extension: risk translated into supervision/capital |

## Best end-to-end demo story

If we want one exhaustive demo that covers the required core system first and extensions second, use the following order.

### Part A. Start with the core prompt requirements

1. **Open with the project scope**
   - State that the system is a portfolio risk engine for stocks and European options
   - State that the project prompt required historical, parametric, and Monte Carlo VaR, plus historical and Monte Carlo ES, plus backtesting

2. **Show the option-pricing foundation first**
   - Use `HW5_BS_DELTA_FD`
   - Inputs: `S=85`, `K=85`, `r=0.045`, `sigma=0.30`, `T=2`
   - Given answer: call price `17.62456`, delta `0.664313`
   - Repo evidence: `src/pricing/black_scholes.py`, `tests/test_homework_cases.py::TestHW5_BS_DeltaFiniteDiff`
   - Why this matters: it proves the repo can handle the "option" half of the required stock-and-option portfolio scope

3. **Then show an option risk intuition case**
   - Use `HW3_INTEL_BSM_DELTA_HEDGE`
   - Given answer path: call price `5.34508`, delta `0.640605`, about `1873` calls to neutralize `1200` shares
   - Why this matters: it demonstrates Greeks and hedge logic, not just static pricing

4. **Show direct parameter-input single-stock VaR**
   - Use HW04 Q1
   - Given answer: 5-day 99% VaR for `1,400` shares at `S0=82` is about `$19,040`
   - Repo evidence:
     - `tests/test_homework_cases.py::TestHW4_SingleStockParamVaR`
     - `tests/test_course_validation.py::TestLN02_HomeworkIV`
   - This is the cleanest "parameter-driven market-risk" case

5. **Show direct parameter-input two-stock parametric VaR**
   - Use HW04 Q2
   - Given answer: 2-week 99% normal VaR about `$9,007.37`
   - Repo evidence: `tests/test_homework_cases.py::TestHW4_TwoStockNormalVaR`
   - This covers covariance aggregation and the parametric branch

6. **Show pure historical empirical VaR/ES on scenarios**
   - Use HW03 scenario portfolio
   - Given answers:
     - 90% VaR `3931.2`
     - 80% ES `3428.6`
   - Repo evidence: `tests/test_homework_cases.py::TestHW3_ScenarioVaR_ES`
   - This is the easiest deterministic illustration of historical VaR and ES

7. **Show the real-data AAPL/CAT historical methodology**
   - Use the AAPL/CAT historical sections reflected in HW07 and the historical notebook
   - Given homework-style values:
     - AAPL latest 5-day historical VaR around `$905.78` per `$10k`
     - CAT around `$969.13`
     - Portfolio around `$898.76`
   - Repo evidence:
     - `src/risk/historical.py`
     - `notebooks/03_historical_shock_methodology.ipynb`
   - This upgrades the historical story from toy arithmetic to actual portfolio data

8. **Show estimator choice and model-risk governance**
   - Use HW05
   - Main point: rolling-window and EWMA estimates are not identical even when lambdas are matched
   - Repo evidence:
     - `notebooks/04_estimation_rolling_vs_ewma.ipynb`
     - `tests/test_homework_cases.py::TestHW5_Lambda20PctHeuristic`
   - Why this matters: it answers the "why did you choose this estimation method?" question that markers often ask

9. **Show Monte Carlo VaR/ES**
   - Use HW08 Q3-Q4
   - Given answers:
     - direct portfolio MC: `95% VaR = 593`, `97.5% ES = 850`
     - component AAPL/CAT MC: `95% VaR = 587`, `97.5% ES = 829`
   - Repo evidence:
     - `src/risk/monte_carlo.py`
     - `notebooks/02_aapl_cat_var_es_methods.ipynb`
     - `notebooks/11_end_to_end_demo.ipynb`
   - This completes the final core required VaR/ES branch

10. **Finish the required core demo with backtesting**
    - Use HW11
    - Given answers:
      - expected exceptions at 99%: `96.33`
      - long AAPL: `156`
      - long CAT: `181`
      - direct-portfolio long: `163`
      - component Monte Carlo long: `149`
      - short AAPL: `98`
      - direct-portfolio short: `91`
      - component Monte Carlo short: `82`
      - short CAT: `130`
    - Repo evidence:
      - `src/risk/backtest.py`
      - `notebooks/10_backtesting_validation_dashboard.ipynb`
      - `notebooks/11_end_to_end_demo.ipynb`
    - Why this matters: it closes the loop from forecast to validation, which the prompt explicitly required

### Part B. Then show the extension modules

11. **Reduced-form hazard**
    - Use HW06
    - Given answers:
      - `P(tau <= 5) = 0.036324`
      - `P(3 < tau <= 4) = 0.007211`
      - piecewise-credit spreads rising from about `69.95bp` to `80.44bp`
    - Repo evidence:
      - `src/credit/hazard.py`
      - `tests/test_course_validation.py::TestHZ01_ConstantHazard`
      - `tests/test_course_validation.py::TestHZ02_PiecewiseHazard`

12. **Merton structural credit**
    - Use HW07 first:
      - `PD_Q = 29.53%`
      - `PD_P = 38.88%`
    - Then use HW09:
      - `B* = 4,612,960.81`
      - `d2 = 1.795086`
      - `Equity = 11,435,404.64`
      - `Debt = 3,564,595.36`
    - Repo evidence:
      - `src/credit/merton.py`
      - `tests/test_course_validation.py::TestMR01_HomeworkVII_QvsP`
      - `tests/test_course_validation.py::TestMR02_TargetSurvivalInversion`

13. **CDS pricing**
    - Use HW08
    - Given answers:
      - simple spread `(1-R)lambda = 180bp`
      - full annual par spread around `184.55bp`
    - Repo evidence:
      - `src/credit/cds.py`
      - `tests/test_course_validation.py::TestCDS01_FlatApprox`
      - `tests/test_course_validation.py::TestCDS02_FullAnnualPaymentParSpread`

14. **CVA and mitigation**
    - Use HW09 discrete CVA:
      - risk-neutral CVA about `5.21`
    - Use HW08 mitigation write-up:
      - netting
      - collateralization
      - CCP / central clearing
    - Repo evidence:
      - `src/credit/cva.py`
      - `src/credit/mitigation.py`
      - `tests/test_homework_cases.py::TestHW9_DiscreteCVA`

15. **Regulatory arithmetic**
    - Use HW10 Q2
    - Given answers:
      - assets `189,000`
      - capital `7,000`
      - RWA `79,850`
      - capital ratio `8.77%`
      - leverage ratio about `3.7%`
    - Repo evidence:
      - `src/risk/regulatory.py`
      - `tests/test_homework_cases.py::TestHW10_RWA_Capital`

## Best "marker-facing" demo package

If we only want one polished demo that still feels comprehensive, the strongest trimmed subset is:

1. BS option price and delta (`HW5_BS_DELTA_FD`)
2. HW04 single-stock 5-day 99% VaR
3. HW04 two-stock parametric VaR
4. HW03 scenario historical VaR/ES
5. HW05 EWMA/window estimator choice
6. HW08 Monte Carlo VaR/ES
7. HW11 backtesting
8. HW06 hazard
9. HW07 Merton
10. HW10 regulatory capital

That sequence covers every core prompt requirement, plus enough extensions to show the repo is materially richer than the minimum.

## Current live repo outputs worth citing

These are not homework answer keys. They are current repo-side demonstration outputs observed from the live code and checked data on `2026-05-11`.

### AAPL/CAT 1-day live engine demo

Using the fixed-share portfolio:

- `24,679` shares of AAPL
- `171` shares of CAT
- common-history dataset starting `1997-10-13`
- pricing date `2026-02-11`
- lookback `252`
- horizon `1`
- VaR confidence `0.99`
- ES confidence `0.975`

Observed outputs:

| Method | VaR | ES |
|---|---:|---:|
| Historical | `336,476.19` | `375,648.94` |
| Parametric (window) | `311,028.70` | `312,587.51` |
| Parametric (EWMA, N=60) | `198,605.79` | `199,633.79` |
| Monte Carlo | `315,922.66` | `310,212.39` |

Observed 1-day historical backtest summary:

- observations: `6873`
- exceptions: `96`
- observed exception rate: `1.3968%`
- Kupiec LR statistic: `9.7286`
- p-value: `0.001814`
- interpretation: reject unconditional coverage at the 5% level under this specific 1-day setup

These numbers are helpful if we want to demo the app and the Python engine live, rather than only citing homework fixtures.

## What the standardized homeworks reveal that is still missing or weak in the repo

The homework pack is also useful as a gap finder.

### Clear gaps or weaknesses

1. **Dynamic option-volatility shock logic is still missing in the core VaR engines**
   - HW10 explicitly explores how higher tail implied volatility changes the hedge and VaR
   - The current core engine prices options with user-supplied volatility, but does not dynamically shock the volatility surface in historical or Monte Carlo repricing

2. **The core market-risk engine is still mostly history-calibrated, not a full direct-input mean/covariance tool**
   - The project prompt wanted both historical calibration and direct parameter input
   - The direct-input story is strong for formula modules and option pricing, but only partial for the main app workflow

3. **The parametric option-exposure path deserves review**
   - This has already been flagged in `submission/05_guide_gap_review.md`
   - It matters if we want to rely heavily on option-bearing parametric demos

4. **Live integration scripts still need alignment with the separate VaR/ES-confidence design**
   - The reports already document this

### Good future implementation ideas directly suggested by the homeworks

1. Add an **option-volatility shock mode** for historical and Monte Carlo option repricing
2. Add a **manual parameter-input mode** for the main market-risk engine
3. Add a **demo notebook or script** that executes the full sequence listed in this document and prints side-by-side expected versus actual values
4. Add a **report appendix table** that directly maps homework question numbers to tests and notebook sections

## Recommended next artifact

If we want this to become a polished teaching/demo asset, the next best thing to add is:

- `notebooks/12_homework_demo_story.ipynb`

That notebook should run the exact sequence above and print:

- the homework question or problem label,
- the given target answer,
- the repo-computed answer,
- the comparison tolerance,
- and a short interpretation of what the case proves.
