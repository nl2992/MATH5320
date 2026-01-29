# MATH GR 5320 - Advanced Demo: Live Front-End Trace

**Columbia University · Spring 2026**

This is the live Streamlit front-end trace for the equal-weight Magnificent Seven portfolio used in [advanced_demo.ipynb](./advanced_demo.ipynb). The point is simple: show that the app loads the same portfolio and data, runs the same market-risk workflow, and reproduces the notebook story on screen.

The app was run locally at `http://localhost:8502`.

## Demo Setup

| Item | Value |
|------|-------|
| Notebook reference | `submission/advanced_demo.ipynb` |
| Reference artifact | `submission/test_artifacts/advanced_demo_reference.json` |
| Source market-data export | `data/m7_2015.csv` |
| Analysis preset | `advanced_m7_full` |
| Analysis data file | `data/m7_2015_eval.csv` |
| Backtest preset | `advanced_m7_stocks` |
| Backtest data file | `data/m7_2015.csv` |
| Analysis date | 2015-12-31 |
| Backtest sample | 2013-01-02 to 2016-12-30 |

Two presets were used on purpose:

1. `advanced_m7_full` for the point-in-time portfolio analysis with the stock book plus the three-option overlay.
2. `advanced_m7_stocks` for the historical backtest, which keeps the same M7 stock book but avoids mixing option expiries into the walk-forward VaR test.

The two CSV inputs used by the app are derived from the notebook export `data/m7_2015.csv`:

1. `data/m7_2015_eval.csv` ends at 2015-12-31 so the live valuation date matches the notebook.
2. `data/m7_2015.csv` keeps the extra 2016 rows needed for the walk-forward backtest.

After rerunning both the notebook and the live app, the remaining differences are only a few cents on some summary numbers. That is small enough to treat as numerical rounding rather than a model discrepancy.

## Coverage Matrix

| Step | Notebook topic | App tab | Evidence |
|------|----------------|---------|----------|
| 1 | M7 portfolio construction | Tab 1 | Portfolio preset screenshot |
| 2 | Market-data load | Tab 2 | CSV load screenshot |
| 3 | Risk settings | Tab 3 | Parameter screenshot |
| 4 | Full-portfolio VaR / ES | Tab 4 | Live result screenshot + notebook/app comparison table |
| 5 | Manual direct-input calibration | Notebook only | Rerun notebook output table in §5 |
| 6 | Option-volatility shock sensitivity | Notebook only | Rerun notebook output table in §6 |
| 7 | Stock-only backtest | Tab 5 | Live backtest screenshot + reference match table |
| 8 | Credit tab smoke check | Tab 6 | Reduced-form screenshot |

## Tab 1 - Portfolio Input

![Portfolio Input](../docs/screenshots/advanced_tab1_portfolio.png)

The preset loads seven long equity positions and three option positions.

### Stock positions

| Ticker | Quantity |
|--------|----------|
| AAPL | 2108 |
| MSFT | 1031 |
| GOOGL | 1295 |
| AMZN | 1479 |
| NVDA | 62194 |
| META | 481 |
| TSLA | 3124 |

### Option overlay

| Label | Underlying | Type | Qty | Strike ($) | Maturity | Vol | r |
|-------|------------|------|-----|------------|----------|-----|---|
| AAPL_C_OTM | AAPL | call | 10 | 24.90 | 2016-06-30 | 0.25 | 0.02 |
| AMZN_C_OTM | AMZN | call | 5 | 36.50 | 2016-06-30 | 0.30 | 0.02 |
| TSLA_P_OTM | TSLA | put | -8 | 14.40 | 2016-03-31 | 0.55 | 0.02 |

The app status line confirms the expected book:

`Portfolio: 7 stock position(s), 3 option position(s).`

Notebook-side book values after rerun:

| Measure | Value |
|---------|-------|
| Stock book value | $349,833.69 |
| Net option value | $1,370.74 |
| Full portfolio value | $351,204.44 |

## Tab 2 - Market Data

![Market Data](../docs/screenshots/advanced_tab2_market_data.png)

For the point-in-time analysis run, the app loads [data/m7_2015_eval.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/data/m7_2015_eval.csv), which stops at the notebook valuation date.

| Field | Value |
|------|-------|
| File | `data/m7_2015_eval.csv` |
| Tickers | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA |
| Rows loaded | 756 |
| Date range | 2013-01-02 to 2015-12-31 |

That matters because the notebook prices the portfolio at 2015-12-31, so the front-end run has to stop at the same date if we want a clean one-to-one comparison.

## Tab 3 - Risk Settings

![Risk Settings](../docs/screenshots/advanced_tab3_risk_settings.png)

The preset drives the app with the same market-risk setup used for the reference run.

| Parameter | Value |
|-----------|-------|
| Calibration mode | historical |
| Lookback window | 252 trading days |
| Horizon | 5 trading days |
| VaR confidence | 99.0% |
| ES confidence | 97.5% |
| Estimator | window |
| Monte Carlo simulations | 10000 |
| Option-volatility shock mode | fixed |
| Vol shock beta | 1.00 |
| Vol floor | 0.0500 |

## Tab 4 - Run Analysis

![Run Analysis](../docs/screenshots/advanced_tab4_run_analysis.png)

This is the key screen. After the preset fix, the live app and the rerun notebook now line up to within a few cents.

### Full-portfolio result comparison

| Quantity | Notebook | App | Difference |
|----------|----------|-----|------------|
| Portfolio value | $351,204.44 | $351,204.43 | $0.01 |
| Historical VaR | $26,802.59 | $26,802.58 | $0.01 |
| Historical ES | $29,004.39 | $29,004.40 | $0.01 |
| Parametric VaR | $23,519.24 | $23,519.24 | $0.00 |
| Parametric ES | $23,646.41 | $23,646.41 | $0.00 |
| Monte Carlo VaR | $22,747.81 | $22,747.82 | $0.01 |
| Monte Carlo ES | $23,098.07 | $23,098.07 | $0.00 |

### Structural checks visible in the live run

| Check | Result |
|-------|--------|
| Historical VaR > Parametric VaR > Monte Carlo VaR | Yes |
| Historical ES > Historical VaR | Yes |
| Parametric ES > Parametric VaR | Yes |
| Monte Carlo ES > Monte Carlo VaR | Yes |
| All risk numbers positive | Yes |

### Stock-only versus full-portfolio effect

The notebook also studies what happens when the three options are added on top of the stock book.

| Model | Stock-only VaR ($) | Full VaR ($) | Increase ($) |
|-------|--------------------|--------------|--------------|
| Historical | 25,470.58 | 26,802.59 | 1,332.01 |
| Parametric | 22,161.68 | 23,519.24 | 1,357.55 |
| Monte Carlo | 21,683.52 | 22,747.81 | 1,064.29 |

| Model | Stock-only ES ($) | Full ES ($) | Increase ($) |
|-------|-------------------|-------------|--------------|
| Historical | 27,483.81 | 29,004.39 | 1,520.58 |
| Parametric | 22,281.64 | 23,646.41 | 1,364.77 |
| Monte Carlo | 21,886.13 | 23,098.07 | 1,211.94 |

The option overlay increases risk across all three methods, which is what we expect here because the short TSLA put adds downside exposure.

### Diversification check from the same run

| Measure | Value |
|---------|-------|
| Sum of individual stock historical VaRs | $32,153.24 |
| Stock-only portfolio historical VaR | $25,470.58 |
| Diversification benefit | 20.78% |

So the app-backed reference run preserves the same diversification story as the notebook: the portfolio VaR is below the sum of the standalone stock VaRs.

## Notebook-Only Validation for Remaining Prompt Items

The live app trace proves the main workflow, but two prompt-sensitive items are cleaner to show directly in the rerun notebook:

1. the manual direct-input calibration path;
2. the option-volatility shock path.

### Manual direct-input calibration

The project brief requires direct parameter input, not only historical calibration. In [advanced_demo.ipynb](./advanced_demo.ipynb) §6, the notebook takes the exact trailing-window daily mean and covariance from the M7 sample, feeds them back through `calibration_mode='manual'`, and compares the results with the standard historical-calibration run.

| Measure | Historical calibration | Manual input path | Absolute difference |
|---------|------------------------|-------------------|---------------------|
| Parametric VaR | $23,519.24 | $23,519.24 | $0.00 |
| Parametric ES | $23,646.41 | $23,646.41 | $0.00 |
| Monte Carlo VaR | $22,747.81 | $22,747.81 | $0.00 |
| Monte Carlo ES | $23,098.07 | $23,098.07 | $0.00 |

That is the exact result we want. It shows the manual calibration path is wired correctly for the model families that consume `mu` and `Sigma` directly.

### Option-volatility shock sensitivity

The project guide also warns against leaving option volatility completely fixed without addressing the effect. In [advanced_demo.ipynb](./advanced_demo.ipynb) §7, the notebook reruns the full portfolio with `option_vol_shock_mode='underlying_beta'` and compares it with the base `fixed` run.

| Measure | Fixed vol | `underlying_beta` | Change |
|---------|-----------|-------------------|--------|
| Historical VaR | $26,802.59 | $26,746.55 | -$56.05 |
| Historical ES | $29,004.39 | $28,958.09 | -$46.30 |
| Parametric VaR | $23,519.24 | $23,519.24 | $0.00 |
| Parametric ES | $23,646.41 | $23,646.41 | $0.00 |
| Monte Carlo VaR | $22,747.81 | $22,651.26 | -$96.55 |
| Monte Carlo ES | $23,098.07 | $23,034.35 | -$63.71 |

This is also the right behavior:

1. the full-repricing engines change when option vol is shocked;
2. the delta-normal parametric engine does not change, because it does not reprice options scenario by scenario.

## Tab 5 - Backtesting

![Backtesting](../docs/screenshots/advanced_tab5_backtesting.png)

For the live backtest proof, the app is reopened with the `advanced_m7_stocks` preset and [data/m7_2015.csv](/Users/nigelli/Desktop/Columbia%20MAFN/26Spring/MATH5320/Project/MATH5320/data/m7_2015.csv), which provides the extra 2016 realised returns needed for the walk-forward test.

### Backtest result comparison

| Quantity | Notebook | App | Difference |
|----------|----------|-----|------------|
| Observations | 750 | 750 | 0 |
| Exceptions | 18 | 18 | 0 |
| Observed exception rate | 2.40% | 2.40% | 0.00% |
| Expected exception rate | 1.00% | 1.00% | 0.00% |
| Kupiec LR statistic | 10.6661 | 10.6661 | 0.0000 |
| p-value | 0.0011 | 0.0011 | 0.0000 |
| Reject H0 at 5% | Yes | Yes | No difference |

This is a strong front-end check because the backtest is not a static table. The app has to estimate rolling VaR forecasts, compare them with realised losses, count exceptions, and then compute the Kupiec test on top.

## Tab 6 - Credit Risk Smoke Check

![Credit Risk](../docs/screenshots/advanced_tab6_credit_reduced_form.png)

The M7 advanced demo is mainly about the market-risk workflow, but the credit tab was also opened to confirm that the live front-end credit panel renders and computes outputs.

The reduced-form section shows:

| Quantity | Value |
|----------|-------|
| Hazard rate λ | 0.0300 |
| Recovery R | 0.40 |
| Discount rate r | 0.0300 |
| LGD | 60.00% |
| CDS approximation | 180.0 bps |
| 5-year cumulative default | 13.9292% |

The Merton subsection is not used as formal evidence in this note because this front-end pass was built to mirror the M7 market-risk notebook. The structural-credit modules remain covered elsewhere in the main submission package and test suite.

## Conclusion

This front-end trace does what it needs to do:

1. It loads the same balanced M7 portfolio as the advanced notebook.
2. It uses date-aligned CSV inputs so the live app and notebook share the same valuation date.
3. It reproduces the headline portfolio VaR and ES numbers to within a few cents.
4. It reproduces the stock-only backtest result exactly, including the exception count, LR statistic, and p-value.
5. The rerun notebook now covers the two extra prompt-sensitive checks that are less natural to prove with a single static UI trace: manual direct-input calibration and option-vol shock sensitivity.

So for the main market-risk story, the Streamlit front-end is now properly evidenced rather than just described.
