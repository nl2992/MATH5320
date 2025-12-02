# MATH5320 Portfolio Risk System

A Streamlit application for end-to-end portfolio risk analysis covering market risk, credit risk, and regulatory capital. Supports mixed portfolios of stocks and European options.

**Columbia University · MATH GR 5320 · Spring 2026**

---

## Features

| Module | Capability |
|--------|-----------|
| **Historical VaR / ES** | Full portfolio repricing under overlapping h-day log-return scenarios |
| **Parametric VaR / ES** | Delta-Normal with horizon scaling; rolling window or EWMA covariance |
| **Monte Carlo VaR / ES** | Full repricing under N(μ_h, Σ_h) correlated log-return shocks (Cholesky) |
| **Black-Scholes Pricing** | European calls and puts with continuous dividends; Greeks |
| **VaR Backtesting** | Walk-forward forecasting with Kupiec unconditional coverage LR test |
| **Hazard / Reduced-form** | Survival function, default density, risky ZCB, CDS approximation |
| **Merton Structural** | Q-PD and P-PD via GBM asset model; implied barrier inversion |
| **CDS Pricing** | Par spread curve across tenors via hazard-rate bootstrapping |
| **CVA & Mitigation** | Discrete CVA from exposure profile; netting and collateral mitigation |
| **Regulatory Capital** | Basel III RWA, Tier-1 capital ratio, DFAST scenario PnL |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (default port 8501)
streamlit run app.py

# Or on a specific port
streamlit run app.py --server.port 8502
```

---

## Application Tabs

### Tab 1 · Portfolio Input
Enter stock positions (ticker + quantity) and European option positions. Each option requires: label, underlying ticker, type (`call`/`put`), quantity (negative = short), strike ($), maturity (YYYY-MM-DD), implied vol σ, risk-free rate r, dividend yield q, and contract multiplier. The panel validates that option maturities are in the future and confirms the total position count.

### Tab 2 · Market Data
Load price history via Yahoo Finance (enter tickers space-separated, start/end dates, click **Download**) or upload a pre-formatted CSV. The panel shows row count, date range, and a scrollable price preview. Cached downloads avoid repeated network calls.

### Tab 3 · Risk Settings
Configure all risk parameters:

| Parameter | Description |
|-----------|-------------|
| Calibration mode | `historical` (estimates μ, Σ from data) or `manual` (enter directly) |
| Lookback window | Trading days used to estimate μ and Σ |
| Risk horizon | Holding period in trading days (h) |
| VaR confidence | e.g. 0.99 for 99% VaR |
| ES confidence | e.g. 0.975 for 97.5% ES |
| Estimator type | `window` (equal-weight rolling) or `ewma` (exponentially weighted) |
| EWMA N parameter | λ = (N−1)/(N+1) |
| Monte Carlo simulations | Number of Cholesky MC paths |
| Option vol shock mode | `fixed` (vol constant) or `beta` (vol scales with underlying shock) |

### Tab 4 · Run Analysis
Click **Run Risk Analysis** to compute all three VaR/ES models simultaneously. Output includes:
- Portfolio value and per-model VaR/ES comparison table
- Bar chart: VaR and ES by model
- Loss distribution histogram (historical simulation) with VaR/ES markers
- Monte Carlo loss distribution tab
- Return correlation heatmap (last lookback window)
- Normalised price history chart
- Download buttons: JSON summary, losses CSV

### Tab 5 · Backtesting
Select a model (`historical`, `parametric`, or `monte_carlo`) and click **Run Backtest** for a walk-forward out-of-sample backtest. Outputs:
- Walk-forward VaR forecast vs realised loss chart with exception markers
- Exception count, observed vs expected exception rate
- Kupiec unconditional coverage LR statistic, p-value, and PASS/FAIL
- Download: backtest results CSV

### Tab 6 · Credit Risk
**Section A — Reduced-form (hazard rate):** Enter constant hazard rate λ, recovery rate R, discount rate r, and a comma-separated list of horizons. Outputs survival function S(t), cumulative default probability F(t), default density f(t), risky ZCB price, and par spread by tenor. Headline metric: CDS approximation (1−R)λ in bps.

**Section B — Merton structural model:** Enter firm value V₀, default barrier B, risk-free rate r (Q-measure), real-world drift μ (P-measure), asset volatility σ_A, and horizon T. Outputs Q-PD, P-PD, Merton equity E₀, Merton debt D₀, d₁, d₂. Also includes target-survival inversion: given a target survival probability s*, returns the implied barrier B*.

### Tab 7 · CDS / CVA
**Section A — CDS pricing:** Computes the full par-spread curve by bootstrapping a piecewise-constant hazard-rate term structure. Inputs: hazard rate λ, recovery R, discount rate r, tenor list. Chart shows spread vs tenor.

**Section B — CVA & mitigation:** Computes discrete CVA from an exposure profile (exposure at each time step) and a hazard-rate curve. Mitigation options: netting (add an offsetting exposure) and collateral (reduce exposure by a posted amount). Outputs: CVA before and after each mitigation step.

### Tab 8 · Capital & Stress
Enter risk weights per ticker (Basel III standard approach) and Tier-1 equity capital. Outputs:
- Risk-Weighted Assets (RWA) = Σ (position notional × risk weight)
- Tier-1 capital ratio = equity / RWA
- PASS if ratio ≥ 8% (Basel III minimum), FAIL otherwise
- DFAST scenario PnL: applies user-defined shock factors to each position

---

## Module Reference

### `src/pricing/`

| Module | Function | Description |
|--------|----------|-------------|
| `black_scholes.py` | `bs_price(S, K, T, r, q, sigma, option_type)` | Black-Scholes price for European call or put |
| | `bs_delta(S, K, T, r, q, sigma, option_type)` | Option delta Δ |
| | `bs_greeks(S, K, T, r, q, sigma, option_type)` | Full Greeks dict (delta, gamma, vega, theta, rho) |

### `src/risk/`

| Module | Key function | Description |
|--------|-------------|-------------|
| `returns.py` | `log_returns(prices)` | Daily log returns from price DataFrame |
| | `overlapping_horizon_returns(log_ret, h)` | Rolling h-day overlapping returns |
| `estimators.py` | `rolling_mean_cov(returns, window)` | Equal-weight μ̂, Σ̂ over trailing window |
| | `ewma_mean_cov(returns, lam)` | EWMA μ̂, Σ̂ with decay parameter λ |
| `historical.py` | `historical_var_es(portfolio, prices, pricing_date, lookback_days, horizon_days, var_confidence, es_confidence)` | Historical simulation VaR and ES |
| `parametric.py` | `parametric_var_es(portfolio, prices, pricing_date, lookback_days, horizon_days, var_confidence, es_confidence, estimator)` | Delta-Normal VaR and ES |
| `monte_carlo.py` | `monte_carlo_var_es(portfolio, prices, pricing_date, lookback_days, horizon_days, var_confidence, es_confidence, n_simulations, seed)` | Monte Carlo VaR and ES |
| `lognormal.py` | `lognormal_var(V0, mu_daily, sigma_daily, horizon, confidence)` | Analytical GBM VaR for single lognormal position |
| `backtest.py` | `run_backtest(...)` | Walk-forward VaR backtest |
| | `kupiec_test(n_obs, n_exceptions, confidence)` | Kupiec LR statistic and p-value |
| `regulatory.py` | `compute_rwa(positions, risk_weights)` | Basel III RWA |
| | `capital_ratio(equity, rwa)` | Tier-1 capital ratio |
| | `dfast_pnl(positions, shocks)` | DFAST scenario PnL |

### `src/credit/`

| Module | Key function | Description |
|--------|-------------|-------------|
| `hazard.py` | `survival(t, lam)` | S(t) = exp(−λt) |
| | `cum_default(t, lam)` | F(t) = 1 − S(t) |
| | `default_density(t, lam)` | f(t) = λ exp(−λt) |
| | `risky_zcb(t, lam, r, R)` | Risky zero-coupon bond price |
| | `cds_par_spread(T, lam, r, R)` | CDS par spread at tenor T |
| `merton.py` | `merton_pd(V0, B, v, sigma, T)` | Default probability N(−d₂) under drift v |
| | `merton_equity(V0, B, r, sigma, T)` | Merton equity value E₀ |
| | `merton_debt(V0, B, r, sigma, T)` | Merton debt value D₀ |
| | `implied_barrier(V0, sigma, T, r, target_survival)` | Invert Merton for implied barrier B* |
| `cds.py` | `cds_full_spread(tenors, lam, r, R)` | Full par-spread curve across tenors |
| `cva.py` | `compute_cva(exposure_profile, lam, r, R)` | Discrete CVA from exposure profile |
| `mitigation.py` | `apply_netting(exposure, offset)` | Net exposure after offsetting trade |
| | `apply_collateral(exposure, collateral)` | Exposure after collateral posting |

### `src/services/`

| Module | Class / function | Description |
|--------|-----------------|-------------|
| `risk_engine_service.py` | `RiskEngineService(portfolio, prices, ...)` | Orchestration layer — wires portfolio + data + parameters to all risk modules |
| | `.run_all()` | Returns dict with `historical`, `parametric`, `monte_carlo` results |
| | `.run_backtest(model)` | Returns backtest DataFrame and Kupiec dict |
| | `.portfolio_value()` | Current mark-to-market portfolio value |

### `src/schemas.py`

```python
StockPosition(ticker: str, quantity: int)

OptionPosition(
    ticker: str,
    underlying_ticker: str,
    option_type: Literal["call", "put"],
    quantity: float,           # negative = short
    strike: float,
    maturity_date: date,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    contract_multiplier: float,
)

Portfolio(
    stocks: list[StockPosition] = [],
    options: list[OptionPosition] = [],
)
```

---

## Programmatic Usage

All `src/` modules are pure functions — no Streamlit imports. Use them directly from Python:

```python
import sys
sys.path.insert(0, '.')

from src.schemas import Portfolio, StockPosition, OptionPosition
from src.services.risk_engine_service import RiskEngineService
from src.data.market_data import download_adjusted_close
from datetime import date

# Build a portfolio
portfolio = Portfolio(stocks=[
    StockPosition(ticker='AAPL', quantity=500),
    StockPosition(ticker='MSFT', quantity=300),
])

# Load price data
prices = download_adjusted_close(['AAPL', 'MSFT'], '2022-01-01', '2024-12-31')

# Run risk analysis
svc = RiskEngineService(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date(2024, 12, 31),
    lookback_days=252,
    horizon_days=5,
    var_confidence=0.99,
    es_confidence=0.975,
    n_simulations=50_000,
    estimator='window',
)

results = svc.run_all()
print(f"Historical VaR: ${results['historical']['var']:,.2f}")
print(f"Historical ES:  ${results['historical']['es']:,.2f}")

# Backtest
bt = svc.run_backtest(model='historical')
print(f"Kupiec p-value: {bt['kupiec']['p_value']:.4f}")
```

```python
# Credit risk — Merton model
from src.credit.merton import merton_pd, merton_equity, merton_debt

pd_q = merton_pd(V0=16.3, B=1.3, v=0.02, sigma=0.3119, T=5)
print(f"NVDA Q-PD: {pd_q:.4%}")

# CDS par spread curve
from src.credit.cds import cds_full_spread
spreads = cds_full_spread(tenors=[1, 2, 3, 5, 10], lam=0.03, r=0.03, R=0.40)
```

---

## Architecture

```
math5320/
├── app.py                          # Streamlit entry point (8 tabs)
├── requirements.txt
├── README.md
├── src/
│   ├── schemas.py                  # StockPosition, OptionPosition, Portfolio
│   ├── config.py                   # Global defaults
│   ├── data/
│   │   ├── market_data.py          # yfinance downloader + CSV loader
│   │   └── validation.py           # Input validation helpers
│   ├── pricing/
│   │   └── black_scholes.py        # BS price, delta, Greeks
│   ├── portfolio/
│   │   ├── positions.py            # Per-position value and delta
│   │   └── portfolio.py            # Portfolio valuation and exposure vector
│   ├── risk/
│   │   ├── returns.py              # Log returns, overlapping horizon returns
│   │   ├── estimators.py           # Window and EWMA covariance
│   │   ├── historical.py           # Historical simulation VaR/ES
│   │   ├── parametric.py           # Delta-Normal VaR/ES
│   │   ├── monte_carlo.py          # Cholesky Monte Carlo VaR/ES
│   │   ├── lognormal.py            # Analytical GBM VaR/ES (single stock)
│   │   ├── backtest.py             # Walk-forward backtest + Kupiec test
│   │   └── regulatory.py          # RWA, capital ratio, DFAST
│   ├── credit/
│   │   ├── hazard.py               # Survival, default density, risky ZCB
│   │   ├── merton.py               # Structural PD (Q and P measure)
│   │   ├── cds.py                  # CDS par spread curve
│   │   ├── cva.py                  # Discrete CVA
│   │   └── mitigation.py          # Netting and collateral
│   ├── services/
│   │   └── risk_engine_service.py  # Orchestration layer
│   └── ui/
│       ├── portfolio_editor.py     # Tab 1 — portfolio input tables
│       ├── market_data_panel.py    # Tab 2 — data loading
│       ├── risk_settings.py        # Tab 3 — parameter controls
│       ├── results_panel.py        # Tab 4 — results display + downloads
│       ├── charts.py               # Plotly chart helpers
│       ├── credit_panel.py         # Tab 6 — credit risk UI
│       ├── cds_cva_panel.py        # Tab 7 — CDS / CVA UI
│       └── capital_panel.py        # Tab 8 — capital & stress UI
├── tests/
│   ├── test_backend.py
│   ├── test_course_validation.py   # PDF fixture goldens (LN01–REG02)
│   ├── test_charts.py
│   ├── test_ui_panels.py
│   ├── test_credit.py
│   ├── test_regulatory.py
│   ├── test_lognormal.py
│   ├── test_market_data.py
│   ├── test_config_and_validation.py
│   ├── test_credit_service.py
│   ├── test_coverage_gaps.py
│   ├── integration_test.py
│   └── integration_test_formula_sheet.py
└── submission/
    ├── advanced_demo.ipynb         # M7 portfolio advanced demo (fully executed)
    ├── advanced_demo.md            # Advanced demo front-end trace
    ├── demo.ipynb                  # 15-section formula-sheet demo
    ├── demo.md                     # Demo front-end trace with screenshots
    └── *.md                        # Final report, model docs, test plan/results
```

---

## Key Modelling Conventions

| Convention | Specification |
|------------|--------------|
| Returns | Daily log returns: r_t = log(S_t / S_{t−1}) |
| Horizon returns | Overlapping rolling sum: R_t^(h) = Σ r_{t−k} for k=0..h−1 |
| Price shock | S_shocked = S₀ × exp(R) |
| PnL | pnl = V_T − V₀ |
| Loss | loss = V₀ − V_T (positive = loss) |
| EWMA λ | λ = (N−1)/(N+1) |
| Horizon scaling | μ_h = μ × h, Σ_h = Σ × h |
| Parametric VaR | −μ_h + σ_h × Φ⁻¹(α) |
| Parametric ES | −μ_h + σ_h × φ(z_α) / (1−α) |
| Option pricing | Black-Scholes-Merton with continuous dividends |
| Kupiec LR | LR_uc ~ χ²(1) under H₀: p = 1 − α |
| Merton Q-PD | N(−d₂) with drift ν = r |
| Merton P-PD | N(−d₂) with drift ν = μ (real-world) |
| CDS approx spread | (1 − R) × λ |
| CVA (discrete) | Σ_t LGD × PD(t−1, t) × EE(t) × D(t) |
| RWA | Σ notional_i × risk_weight_i |
| Capital ratio | Tier-1 equity / RWA |

---

## Running Tests

```bash
# Full unit-test suite (no network)
python -m pytest tests/ --ignore=tests/integration_test.py \
  --ignore=tests/integration_test_formula_sheet.py -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --ignore=tests/integration_test.py \
  --ignore=tests/integration_test_formula_sheet.py

# By domain
python -m pytest tests/test_backend.py -v              # Core engine + service layer
python -m pytest tests/test_course_validation.py -v    # PDF fixture goldens
python -m pytest tests/test_credit.py -v               # Hazard / Merton / CDS / CVA
python -m pytest tests/test_regulatory.py -v           # RWA / capital / DFAST
python -m pytest tests/test_lognormal.py -v            # Analytical GBM VaR/ES
python -m pytest tests/test_charts.py -v               # Plotly chart helpers
python -m pytest tests/test_ui_panels.py -v            # Streamlit panels (AppTest)

# Network integration tests
python tests/integration_test.py
python tests/integration_test_formula_sheet.py

# Useful flags
# -x        stop at first failure
# -k "merton"  filter by keyword
# --lf      re-run last failures
# -s        show stdout
```

### Course validation fixtures

`tests/test_course_validation.py` encodes goldens from `risk_engine_validation_test_sheet.pdf` covering LN01–LN04, HZ01–HZ04, MR01–MR02, CDS01–CDS04, CVA01–CVA05, REG01–REG02. Numerical comparisons use ~10% relative tolerance. The two AAPL/CAT acceptance tests (ACC01, ACC02) skip unless `data/AAPL-bloomberg.csv` and `data/CAT-bloomberg.csv` are present.
