# MATH5320 Portfolio Risk System

A Streamlit application for portfolio risk analysis supporting stocks and European options.

## Features

| Feature | Details |
|---|---|
| **Historical VaR / ES** | Full portfolio repricing under overlapping h-day log-return scenarios |
| **Parametric VaR / ES** | Delta-Normal with horizon scaling; window or EWMA estimator |
| **Monte Carlo VaR / ES** | Full repricing under N(μ_h, Σ_h) simulated log-return shocks |
| **Black-Scholes Pricing** | European calls and puts with continuous dividends |
| **VaR Backtesting** | Walk-forward forecasting with Kupiec, Christoffersen, Basel traffic-light, and exception-severity diagnostics |
| **Downloads** | JSON risk summary with run metadata, tidy losses CSV, backtest CSV, backtest summary JSON |

## What Matters for the Brief

The project brief is narrower than the full repo. The core graded system is the stock and European-option risk engine:

- portfolio input,
- historical or manual parameter calibration,
- historical, parametric, and Monte Carlo VaR,
- historical, parametric, and Monte Carlo ES,
- walk-forward backtesting.

The credit, CVA, lognormal, and regulatory pieces are still part of the repo and still tested, but they should be read as course extensions rather than the main project boundary.

## Architecture Overview

```mermaid
flowchart TB
    U["User"] --> APP["app.py · Streamlit entry point"]

    subgraph UI["src/ui/ — UI panels"]
        PE["portfolio_editor"]
        MD["market_data_panel"]
        RS["risk_settings"]
        RP["results_panel · charts"]
        XUI["credit_panel · cds_cva_panel · capital_panel"]
    end

    subgraph SVC["src/services/ — orchestration"]
        RSE["risk_engine_service"]
        CRS["credit_service"]
        RGS["regulatory_service"]
    end

    subgraph CORE["Core engine  ·  required by project brief"]
        DAT["data/ · market_data · validation"]
        CFG["schemas · config · demo_presets"]
        PRT["portfolio/ · positions · portfolio"]
        BSM["pricing/ · black_scholes"]
        RSK["risk/ · returns · estimators · historical<br/>parametric · normal · monte_carlo · backtest"]
    end

    subgraph EXT["Course extensions"]
        LOG["risk/lognormal"]
        REG["risk/regulatory"]
        CRD["credit/ · hazard · merton · cds · cva · mitigation"]
    end

    APP --> PE & MD & RS & RP & XUI

    PE --> RSE
    MD --> RSE
    RS --> RSE
    XUI --> CRS & RGS

    RSE --> DAT & CFG & PRT & RSK
    PRT --> BSM
    CRS --> CRD
    RGS --> REG

    RSE --> MOUT["VaR/ES · loss distributions · backtest results · downloads"]
    CRS --> COUT["hazard · Merton · CDS · CVA"]
    RGS --> ROUT["RWA · capital · DFAST"]

    TST["tests/ · 622 unit tests"] -. exercise .-> CORE & EXT
    NB["notebooks/"] -. exercise .-> CORE & EXT
```

The main split is simple: `app.py` handles Streamlit rendering and calls services when the user clicks "Run". The services orchestrate the math modules. All quantitative logic (pricing, risk, credit, regulatory) lives in pure Python modules under `src/` with no Streamlit imports, so tests and notebooks can call them directly without running the app.

## Repository Layout

```
MATH5320/
├── app.py                          # Streamlit entry point (UI only)
├── requirements.txt
├── README.md
├── src/
│   ├── schemas.py                  # StockPosition, OptionPosition, Portfolio
│   ├── config.py                   # Global defaults
│   ├── demo_presets.py             # Reproducible Streamlit demo presets
│   ├── data/
│   │   ├── market_data.py          # CSV loader + yfinance downloader + cache
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
│   │   ├── normal.py               # Closed-form normal VaR/ES helpers
│   │   ├── monte_carlo.py          # Monte Carlo VaR/ES
│   │   ├── lognormal.py            # Exact GBM VaR/ES
│   │   ├── regulatory.py           # RWA, capital ratio, DFAST helpers
│   │   └── backtest.py             # Walk-forward backtest + Kupiec/Christoffersen
│   ├── credit/
│   │   ├── hazard.py               # Reduced-form hazard and survival
│   │   ├── merton.py               # Merton structural default model
│   │   ├── cds.py                  # CDS par spread
│   │   ├── cva.py                  # CVA and exposure helpers
│   │   └── mitigation.py           # Netting and collateral
│   ├── services/
│   │   ├── risk_engine_service.py  # Market-risk orchestration
│   │   ├── credit_service.py       # Credit and CVA orchestration
│   │   └── regulatory_service.py   # RWA and DFAST orchestration
│   └── ui/
│       ├── portfolio_editor.py     # Portfolio input
│       ├── market_data_panel.py    # Data loading
│       ├── risk_settings.py        # Parameter controls
│       ├── results_panel.py        # Results and downloads
│       ├── credit_panel.py         # Hazard / Merton panel
│       ├── cds_cva_panel.py        # CDS / CVA panel
│       ├── capital_panel.py        # Capital and stress panel
│       └── charts.py               # Plotly chart helpers
├── tests/                          # 622 no-network unit and regression tests
├── notebooks/                      # Course walkthrough notebooks
├── docs/
│   ├── references/
│   └── screenshots/
└── submission/                     # Final submission package
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Or install the packaged distribution once published
pip install math5320-portfolio-risk-system

# Run the app
streamlit run app.py
```

## Usage Workflow

1. **Portfolio Input** - Add stock positions (ticker + quantity) and option positions (label, underlying, type, quantity, strike, maturity, vol, r, q, multiplier).
   The schema layer now rejects structurally invalid positions before they reach pricing or risk code.
2. **Market Data** - Download price history from Yahoo Finance or upload a CSV file.
   The loader rejects duplicate dates, non-positive prices, and active columns with missing values. It also warns on suspiciously long stale-price runs.
3. **Risk Settings** - Configure lookback window, horizon, VaR/ES confidence levels, estimator type (window or EWMA), and Monte Carlo simulation count. If VaR and ES use different confidence levels, the app labels that explicitly in the results.
4. **Run Analysis** - Click "Run Risk Analysis" to compute all three VaR/ES models. Results include a comparison table, loss histograms, correlation heatmap, and download buttons.
5. **Backtesting** - Select a model and click "Run Backtest" for walk-forward VaR backtesting with Kupiec test results. If any forecast dates fail, the app now reports how many dates were skipped instead of hiding them silently.

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

The service layer now validates that the loaded price history is structurally clean and long enough for the requested lookback and horizon before a risk run starts.

---

## Programmatic API

All quantitative modules are plain Python — no Streamlit dependency — so you can call them directly from a notebook, script, or test without running the app.  The canonical entry points are described below, grouped by layer.

### 1 · Data structures

```python
from src.schemas import Portfolio, StockPosition, OptionPosition
from datetime import date

# Long 100 AAPL + long 1 call on AAPL
portfolio = Portfolio(
    stocks=[StockPosition(ticker="AAPL", quantity=100)],
    options=[
        OptionPosition(
            ticker="AAPL_C_200",
            underlying_ticker="AAPL",
            option_type="call",
            quantity=1,
            strike=200.0,
            maturity=date(2026, 12, 19),
            volatility=0.25,          # implied vol
            risk_free_rate=0.045,
            dividend_yield=0.0,
            multiplier=100,
        )
    ],
)
```

`StockPosition` validates that `ticker` is non-empty and `quantity` is finite; `OptionPosition` validates all seven positivity and range constraints at construction time.

---

### 2 · Market data

```python
from src.data.market_data import (
    load_price_history_csv,
    download_adjusted_close,
    download_adjusted_close_cached,
    fetch_risk_free_rate,
)

# From CSV (Bloomberg wide format)
prices = load_price_history_csv("data/AAPL-bloomberg.csv")

# From Yahoo Finance (plain, no cache)
prices = download_adjusted_close(["AAPL", "MSFT"], start="2022-01-01", end="2025-01-01")

# Fault-tolerant: parquet cache + retry + per-ticker fallback
prices = download_adjusted_close_cached(
    ["AAPL", "MSFT", "^GSPC"],
    start="2022-01-01", end="2025-01-01",
    cache_dir=".cache/prices", max_retries=3,
)

# Risk-free rate proxy from ^TNX (falls back to 0.04 on any failure)
from datetime import date
r = fetch_risk_free_rate(asof=date.today(), fallback=0.04)
```

**Returns:** `pd.DataFrame` with `DatetimeIndex` and ticker columns; values are adjusted-close prices.

---

### 3 · Service layer (recommended entry point)

`RiskEngineService` orchestrates all three VaR/ES models and the backtest:

```python
from src.services.risk_engine_service import RiskEngineService

svc = RiskEngineService(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date(2025, 1, 2),
    lookback_days=252,
    horizon_days=10,
    var_confidence=0.99,
    es_confidence=0.975,
    estimator="window",      # or "ewma"
    ewma_N=60,
    n_simulations=10_000,
)

# Current portfolio mark-to-market
V0 = svc.portfolio_value()

# All three models in one call
results = svc.run_all()
print(results["historical"]["var"])    # Historical VaR ($)
print(results["parametric"]["es"])     # Parametric ES ($)
print(results["monte_carlo"]["var"])   # Monte Carlo VaR ($)
```

`results["historical"]` / `results["parametric"]` / `results["monte_carlo"]` each contain at minimum `{"var": float, "es": float}`.

#### Backtesting

```python
bt = svc.run_backtest(model="historical")  # or "parametric" / "monte_carlo"

df = bt["backtest_df"]          # pd.DataFrame: date, var_forecast, realized_loss, exception
kupiec = bt["kupiec"]           # {"p_hat", "lr_stat", "p_value", "reject_h0", ...}
cc = bt["conditional_coverage"] # Christoffersen (independence + joint coverage)
basel = bt["basel"]             # {"zone": "GREEN"|"YELLOW"|"RED", "n_exceptions": int, ...}
severity = bt["severity"]       # {"exception_gap", "average_exception_loss", ...}
```

---

### 4 · Risk modules (direct calls)

Each risk module is a pure function; use these when you need fine-grained control.

#### Historical VaR / ES

```python
from src.risk.historical import historical_var_es

res = historical_var_es(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date(2025, 1, 2),
    lookback_days=252,
    horizon_days=10,
    var_confidence=0.99,
    es_confidence=0.975,
)
# res["var"], res["es"], res["losses"] (np.ndarray), res["n_scenarios"]
```

#### Parametric (Delta-Normal) VaR / ES

```python
from src.risk.parametric import parametric_var_es

res = parametric_var_es(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date(2025, 1, 2),
    lookback_days=252,
    horizon_days=10,
    var_confidence=0.99,
    es_confidence=0.975,
    estimator="ewma",
    ewma_N=60,
)
# res["var"], res["es"], res["mean_pnl"], res["std_pnl"]
```

You can also pass **manual** mean / covariance parameters (e.g. from the course homework inputs):

```python
res = parametric_var_es(
    ...,
    calibration_mode="manual",
    manual_market_params={
        "mu_daily": {"AAPL": 0.0003, "MSFT": 0.0004},
        "cov_daily": {
            "AAPL": {"AAPL": 4e-4, "MSFT": 2e-4},
            "MSFT": {"AAPL": 2e-4, "MSFT": 3e-4},
        },
    },
)
```

#### Monte Carlo VaR / ES

```python
from src.risk.monte_carlo import monte_carlo_var_es

res = monte_carlo_var_es(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date(2025, 1, 2),
    lookback_days=252,
    horizon_days=10,
    var_confidence=0.99,
    es_confidence=0.975,
    n_simulations=20_000,
    random_seed=42,
)
# res["var"], res["es"], res["losses"] (np.ndarray, length n_simulations)
```

---

### 5 · Closed-form normal VaR / ES

```python
from src.risk.normal import normal_var, normal_es, portfolio_delta_normal_mean_var
import numpy as np

# If you already have exposures (x), daily μ, and daily Σ:
x   = np.array([100_000.0, -50_000.0])   # dollar-delta exposures
mu  = np.array([0.0003, 0.0004]) * 10    # 10-day mean log returns
cov = np.eye(2) * 1e-3 * 10             # 10-day covariance

m, s = portfolio_delta_normal_mean_var(x, mu, cov)
var  = normal_var(m, s, confidence=0.99)
es   = normal_es(m, s, confidence=0.975)
```

---

### 6 · Exact lognormal (GBM) VaR / ES

```python
from src.risk.lognormal import (
    var_long_lognormal, es_long_lognormal,
    var_short_lognormal, es_short_lognormal,
)

# 5-trading-day 99% VaR on a $100 000 long GBM position
var = var_long_lognormal(V0=100_000, mu=0.08, sigma=0.20, h=5/252, p=0.99)
es  = es_long_lognormal( V0=100_000, mu=0.08, sigma=0.20, h=5/252, p=0.975)

# Short position — losses come from upward moves
var_short = var_short_lognormal(V0=100_000, mu=0.08, sigma=0.20, h=5/252, p=0.99)
```

---

### 7 · Credit modules

#### Reduced-form hazard model (§8)

```python
from src.credit.hazard import survival, cumulative_default_prob, credit_spread, survival_piecewise

# Constant hazard
s5 = survival(t=5, lam=0.03)                     # P(τ > 5) under λ=3%
pd5 = cumulative_default_prob(t=5, lam=0.03)     # P(τ ≤ 5)
spread = credit_spread(T=5, LGD=0.60, s_T=s5)   # Implied credit spread

# Piecewise-constant hazard
s = survival_piecewise(t=3.5, grid=[0,1,3,5,10], hazards=[0.01,0.02,0.03,0.04])
```

#### Merton structural model (§9)

```python
from src.credit.merton import merton_pd, merton_equity, merton_debt, merton_implied_B
from src.services.credit_service import merton_summary

# Full snapshot: Q-PD, P-PD, E0, D0 in one call
snap = merton_summary(V0=100, B=80, r=0.05, mu=0.08, sigma=0.25, T=1)
print(snap["Q"]["PD"])   # Risk-neutral default probability
print(snap["P"]["PD"])   # Real-world default probability
print(snap["E0"])        # Equity value
print(snap["D0"])        # Risky debt value

# Find the barrier B* that implies a target 1-year survival of 90%
B_star = merton_implied_B(V0=100, target_survival=0.90, r=0.05, sigma=0.25, T=1)
```

#### CDS par spread (§10)

```python
from src.credit.cds import cds_par_spread_constant_hazard, cds_par_spread

# Quick approximation: C ≈ (1−R)·λ  (≈180 bps at λ=3%, R=40%)
spread_approx = cds_par_spread_constant_hazard(lam=0.03, R=0.40)

# Full numerical formula with piecewise hazard
spread_full = cds_par_spread(
    payment_times=[1, 2, 3, 4, 5],
    hazards=[0.03]*5,
    r=0.03, R=0.40,
)
```

#### CVA (§11)

```python
from src.credit.cva import cva_discrete, cva_continuous_constant_exposure
from src.services.credit_service import cva_summary

# Discrete CVA from EPE profile
cva = cva_discrete(
    exposures=[1_000, 800, 600, 400],          # EPE at t=1,2,3,4
    marginal_default_probs=[0.02, 0.019, 0.018, 0.017],
    R=0.40,
)

# Full summary with CVA as % of V0
summary = cva_summary(exposures, marginal_default_probs, R=0.40, V0=50_000)
print(summary["cva_pct"])
```

---

### 8 · Regulatory capital (§12)

```python
from src.risk.regulatory import risk_weighted_assets, capital_ratio, apply_stress_scenario
from src.services.regulatory_service import compute_rwa_and_ratio, run_dfast

import pandas as pd
from datetime import date

spots = pd.Series({"AAPL": 200.0, "MSFT": 420.0})

# RWA + capital ratio (uses BS-delta exposures for options)
rwa_result = compute_rwa_and_ratio(
    portfolio=portfolio,
    prices=spots,
    risk_weights={"AAPL": 1.0, "MSFT": 1.0},
    equity=50_000.0,
    pricing_date=date(2025, 1, 2),
)
print(rwa_result["ratio"], rwa_result["pass"])

# DFAST equity stress scenarios (baseline / adverse / severely_adverse)
dfast = run_dfast(portfolio=portfolio, prices=spots, pricing_date=date(2025, 1, 2))
for name, res in dfast.items():
    print(f"{name}: PnL = ${res['pnl']:,.0f}  ({res['equity_shock']:+.0%})")
```

---

### 9 · Backtest statistics in isolation

```python
from src.risk.backtest import kupiec_test, christoffersen_test, conditional_coverage_test, basel_traffic_light
import numpy as np

# Suppose 250 backtest days with 6 exceptions
kupiec = kupiec_test(n_observations=250, n_exceptions=6, var_confidence=0.99)
# {"lr_stat", "p_value", "reject_h0", "p_hat", ...}

exceptions = np.zeros(250, dtype=int)
exceptions[[10, 50, 100, 150, 200, 240]] = 1
cc = conditional_coverage_test(
    n_observations=250, n_exceptions=6,
    var_confidence=0.99, exceptions=exceptions,
)
# Adds {"lr_cc", "p_value_cc", "reject_cc", ...}

zone = basel_traffic_light(n_exceptions=6)
# {"zone": "GREEN", "n_exceptions": 6, "multiplier": 0.0, ...}
```

---

### 10 · Covariance estimators

```python
from src.risk.estimators import get_mean_cov, manual_mean_cov
from src.risk.returns import compute_log_returns

log_ret = compute_log_returns(prices)

# Rolling window
mu, cov = get_mean_cov(log_ret, lookback_days=252, estimator="window")

# EWMA (λ = (N-1)/(N+1), here N=60 → λ≈0.967)
mu, cov = get_mean_cov(log_ret, lookback_days=252, estimator="ewma", ewma_N=60)

# Manual override
mu, cov = manual_mean_cov(
    manual_market_params={
        "mu_daily": {"AAPL": 0.0003, "MSFT": 0.0004},
        "cov_daily": {
            "AAPL": {"AAPL": 4e-4, "MSFT": 2e-4},
            "MSFT": {"AAPL": 2e-4, "MSFT": 3e-4},
        },
    },
    underlyings=["AAPL", "MSFT"],
)
```

---

### 11 · Bloomberg CSV format

The system accepts any wide-format CSV where the first column is the date and subsequent columns are ticker symbols.  This matches the Bloomberg terminal export with minor pre-processing:

```
Date,AAPL US Equity,CAT US Equity
2023-01-03,125.07,228.47
2023-01-04,126.36,231.29
...
```

Rename the header row so tickers match what you pass to `download_adjusted_close` or specify in `StockPosition` / `OptionPosition`.  No other transformation is needed; the loader handles date parsing and sort order automatically.

---

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

Coverage is reported with pytest-cov and reviewed through the terminal missing-line report.

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
methodology checks). Numerical goldens are compared at approximately 1%
relative tolerance (`REL = 0.01` in `tests/test_course_validation.py`).

The two AAPL/CAT acceptance tests (ACC01, ACC02) skip cleanly unless
`data/AAPL-bloomberg.csv` and `data/CAT-bloomberg.csv` are present.

## Project Requirements Coverage Matrix

Requirements from `docs/references/project_requirements.pdf` (MATH GR 5320).

| Requirement | Status | Implementation | Tests |
|---|---|---|---|
| Accept stock and option positions as input | ✅ | [src/schemas.py](src/schemas.py), [src/ui/portfolio_editor.py](src/ui/portfolio_editor.py) | [tests/test_backend.py](tests/test_backend.py) |
| Calibrate to historical price data | ✅ | [src/data/market_data.py](src/data/market_data.py), [src/risk/returns.py](src/risk/returns.py), [src/risk/estimators.py](src/risk/estimators.py) | [tests/test_market_data.py](tests/test_market_data.py), [tests/test_backend.py](tests/test_backend.py) |
| Accept manual parameters as input | ✅ | [src/risk/estimators.py](src/risk/estimators.py) (`manual_mean_cov`) | [tests/test_coverage_gaps.py](tests/test_coverage_gaps.py) |
| Compute historical VaR and ES | ✅ | [src/risk/historical.py](src/risk/historical.py) | [tests/test_backend.py](tests/test_backend.py), [tests/test_homework_cases.py](tests/test_homework_cases.py) |
| Compute parametric (delta-normal) VaR and ES | ✅ | [src/risk/parametric.py](src/risk/parametric.py) | [tests/test_backend.py](tests/test_backend.py), [tests/test_course_validation.py](tests/test_course_validation.py) |
| Compute Monte Carlo VaR and ES | ✅ | [src/risk/monte_carlo.py](src/risk/monte_carlo.py) | [tests/test_backend.py](tests/test_backend.py), [tests/test_course_validation.py](tests/test_course_validation.py) |
| Backtest VaR against historical exceptions | ✅ | [src/risk/backtest.py](src/risk/backtest.py), [src/ui/results_panel.py](src/ui/results_panel.py) | [tests/test_backtest_extensions.py](tests/test_backtest_extensions.py), [tests/test_course_validation.py](tests/test_course_validation.py) |
| Option pricing (Black-Scholes) | ✅ | [src/pricing/black_scholes.py](src/pricing/black_scholes.py) | [tests/test_backend.py](tests/test_backend.py), [tests/test_coverage_gaps.py](tests/test_coverage_gaps.py) |
| Covariance estimation (window and EWMA) | ✅ | [src/risk/estimators.py](src/risk/estimators.py) | [tests/test_backend.py](tests/test_backend.py), [tests/test_coverage_gaps.py](tests/test_coverage_gaps.py) |
| Option volatility shock (not just fixed vol) | ✅ | [src/risk/historical.py](src/risk/historical.py) (`option_vol_shock_mode="underlying_beta"`) | [tests/test_backend.py](tests/test_backend.py) |

**Grading penalty flags addressed:**

| Penalty flag (project guide) | How it is addressed |
|---|---|
| Not modelling volatility changes for options | `underlying_beta` shock mode scales option vol with the underlying return: `σ' = max(floor, σ·(1 − β·R))`. Default remains `fixed`; `underlying_beta` is available and demonstrated in `submission/advanced_demo.ipynb §7`. |
| Using historical volatility instead of implied volatility | Option positions carry a user-supplied `volatility` field (implied vol). The system does not back out implied vol from market prices; this limitation is documented in `submission/00_combined_final_report.md §10`. |
| Incorrect covariance | Covariance is estimated from historical log returns using a rolling window or EWMA; the delta-dollar exposure vector is computed correctly as `x = n·S·Δ`. See [src/risk/parametric.py](src/risk/parametric.py) and [tests/test_strict_numerics.py](tests/test_strict_numerics.py). |
| Inappropriate parametric VaR | Parametric VaR uses the correct delta-normal formula: `VaR = −m + s·Φ⁻¹(α)` with proper h-day horizon scaling. See [src/risk/normal.py](src/risk/normal.py) and [tests/test_course_validation.py](tests/test_course_validation.py). |
| Tests not supporting model-doc conclusions | 610 no-network tests with 96% statement coverage. Course fixture values (HW03–HW11) are embedded in [tests/test_homework_cases.py](tests/test_homework_cases.py) and [tests/test_course_validation.py](tests/test_course_validation.py). |

---

## Final Submission Documents

The final submission package is in `submission/`.

| File | Purpose |
|---|---|
| `submission/00_combined_final_report.md` | Primary final report |
| `submission/01_model_documentation.md` | Model documentation |
| `submission/02_software_design_documentation.md` | Software design documentation |
| `submission/03_test_plan.md` | Test plan |
| `submission/04_test_results.md` | Test results |
| `submission/demo.ipynb` | Formula-sheet demonstration notebook (15 sections, fully executed) |
| `submission/demo.md` | Front-end workflow trace with screenshots |
| `submission/advanced_demo.ipynb` | Advanced demo: equal-weight M7 portfolio, §1-§10 including manual calibration and option-vol shock checks |
| `submission/advanced_demo.md` | M7 portfolio front-end trace with screenshots plus notebook-only proof for prompt-sensitive checks |
