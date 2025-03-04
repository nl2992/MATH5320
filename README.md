# MATH5320 Portfolio Risk System

A Streamlit application for portfolio risk analysis supporting stocks and European options.

## Features

| Feature | Details |
|---|---|
| **Historical VaR / ES** | Full portfolio repricing under overlapping h-day log-return scenarios |
| **Parametric VaR / ES** | Delta-Normal with horizon scaling; window or EWMA estimator |
| **Monte Carlo VaR / ES** | Full repricing under N(μ_h, Σ_h) simulated log-return shocks |
| **Black-Scholes Pricing** | European calls and puts with continuous dividends |
| **VaR Backtesting** | Walk-forward forecasting with Kupiec unconditional coverage test |
| **Downloads** | JSON risk summary, losses CSV, backtest CSV |

## Architecture

```
math5320_risk_app/
├── app.py                          # Streamlit entry point (UI only)
├── requirements.txt
├── README.md
├── src/
│   ├── schemas.py                  # StockPosition, OptionPosition, Portfolio
│   ├── config.py                   # Global defaults
│   ├── data/
│   │   ├── market_data.py          # CSV loader + yfinance downloader
│   │   └── validation.py           # Input validation
│   ├── pricing/
│   │   └── black_scholes.py        # BS price and delta
│   ├── portfolio/
│   │   ├── positions.py            # Per-position value and delta
│   │   └── portfolio.py            # Portfolio valuation and exposure vector
│   ├── risk/
│   │   ├── returns.py              # Log returns, overlapping horizon returns
│   │   ├── estimators.py           # Window and EWMA mean/covariance
│   │   ├── historical.py           # Historical VaR/ES
│   │   ├── parametric.py           # Delta-Normal VaR/ES
│   │   ├── monte_carlo.py          # Monte Carlo VaR/ES
│   │   └── backtest.py             # Walk-forward backtest + Kupiec test
│   ├── services/
│   │   └── risk_engine_service.py  # Orchestration layer
│   └── ui/
│       ├── portfolio_editor.py     # Portfolio input tables
│       ├── market_data_panel.py    # Data loading UI
│       ├── risk_settings.py        # Parameter controls
│       ├── results_panel.py        # Results display and downloads
│       └── charts.py               # Plotly chart helpers
└── tests/
    ├── test_backend.py             # 19 unit tests
    └── integration_test.py         # End-to-end integration test
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage Workflow

1. **Portfolio Input** — Add stock positions (ticker + quantity) and option positions (label, underlying, type, quantity, strike, maturity, vol, r, q, multiplier).
2. **Market Data** — Download price history from Yahoo Finance or upload a CSV file.
3. **Risk Settings** — Configure lookback window, horizon, VaR/ES confidence levels, estimator type (window or EWMA), and Monte Carlo simulation count.
4. **Run Analysis** — Click "Run Risk Analysis" to compute all three VaR/ES models. Results include a comparison table, loss histograms, correlation heatmap, and download buttons.
5. **Backtesting** — Select a model and click "Run Backtest" for walk-forward VaR backtesting with Kupiec test results.

## Key Modelling Conventions

| Convention | Specification |
|---|---|
| **Returns** | Daily log returns: r_t = log(S_t / S_{t-1}) |
| **Horizon returns** | Overlapping rolling sum: R_t^(h) = Σ r_{t-k} for k=0..h-1 |
| **Price shock** | S_shocked = S_0 × exp(R) |
| **PnL** | pnl = V_T − V_0 |
| **Loss** | loss = V_0 − V_T (positive = loss) |
| **EWMA λ** | λ = (N−1)/(N+1) |
| **Horizon scaling** | μ_h = μ × h, Σ_h = Σ × h |
| **Parametric VaR** | −m + s × Φ⁻¹(confidence) |
| **Parametric ES** | −m + s × φ(z) / α |
| **Option pricing** | Black-Scholes with continuous dividends |
| **Kupiec test** | LR_uc ~ χ²(1) |

## Running Tests

All commands below are run from the project root.

### Full unit-test suite (no network)

```bash
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

### With coverage report

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py
```

Target: 100% statement coverage across `src/`.

### Individual test files

```bash
python -m pytest tests/test_backend.py -v            # Core engine + service layer
python -m pytest tests/test_course_validation.py -v  # PDF validation-sheet fixtures
python -m pytest tests/test_charts.py -v             # Plotly chart helpers
python -m pytest tests/test_ui_panels.py -v          # Streamlit UI panels (AppTest)
python -m pytest tests/test_credit.py -v             # hazard / Merton / CDS / CVA
python -m pytest tests/test_regulatory.py -v         # RWA / capital / DFAST
python -m pytest tests/test_lognormal.py -v          # Exact GBM VaR / ES
python -m pytest tests/test_market_data.py -v        # CSV loader + yfinance wrappers
python -m pytest tests/test_config_and_validation.py -v
python -m pytest tests/test_credit_service.py -v
python -m pytest tests/test_coverage_gaps.py -v
```

### Running a single class or test

```bash
python -m pytest tests/test_course_validation.py::TestLN02_HomeworkIV -v
python -m pytest tests/test_course_validation.py::TestMR01_HomeworkVII_QvsP::test_pd_Q -v
```

### Network integration tests

```bash
python tests/integration_test.py                  # End-to-end with real market data
python tests/integration_test_formula_sheet.py    # Full formula-sheet integration
```

### Useful pytest flags

| Flag | Effect |
|---|---|
| `-v` | Verbose (one line per test) |
| `-x` | Stop at first failure |
| `-k "merton"` | Only tests matching the keyword |
| `--lf` | Re-run only last failures |
| `-s` | Don't capture stdout (useful for debugging prints) |

### Course validation fixtures

`tests/test_course_validation.py` encodes the course-supplied fixtures from
`risk_engine_validation_test_sheet.pdf` (LN01–LN04, HZ01–HZ04, MR01–MR02,
CDS01–CDS04, CVA01–CVA05, REG01–REG02, plus non-numeric monotonicity /
methodology checks). Numerical goldens are compared at ~10% relative tolerance.

The two AAPL/CAT acceptance tests (ACC01, ACC02) skip cleanly unless
`data/AAPL-bloomberg.csv` and `data/CAT-bloomberg.csv` are present.
