"""
integration_test_formula_sheet.py
End-to-end integration test exercising the full formula-sheet stack against
real Yahoo data. Network-required — skip if `MATH5320_SKIP_NETWORK=1`.

Run with:
    python tests/integration_test_formula_sheet.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

if os.environ.get("MATH5320_SKIP_NETWORK") == "1":
    print("Skipped (MATH5320_SKIP_NETWORK=1).")
    sys.exit(0)

from src.credit.cds import cds_par_spread_constant_hazard
from src.credit.cva import cva_discrete, epe_profile_from_mc
from src.data.market_data import (
    download_adjusted_close_cached,
    fetch_risk_free_rate,
)
from src.risk.lognormal import (
    var_long_lognormal,
    var_short_lognormal,
)
from src.risk.regulatory import DFAST_SCENARIOS
from src.schemas import OptionPosition, Portfolio, StockPosition
from src.services.credit_service import cva_summary, merton_summary
from src.services.regulatory_service import compute_rwa_and_ratio, run_dfast
from src.services.risk_engine_service import RiskEngineService

print("=" * 70)
print("MATH5320 Formula-Sheet Integration Test")
print("=" * 70)

# ── 1. Cached price + benchmark download ──────────────────────────────────────
print("\n[1] download_adjusted_close_cached(AAPL, MSFT, ^GSPC, ^TNX) · 3y")
end = date.today()
start = end - timedelta(days=3 * 365)
tickers = ["AAPL", "MSFT", "^GSPC", "^TNX"]
prices = download_adjusted_close_cached(tickers, str(start), str(end))
print(f"    Shape: {prices.shape}, columns: {list(prices.columns)}")
assert len(prices) > 500, "Expected at least 500 price rows for 3y."
for t in tickers:
    assert t in prices.columns, f"Missing column {t}"
assert prices.notna().any().all(), "Some tickers are all-NaN."

# ── 2. Risk-free rate ─────────────────────────────────────────────────────────
print("\n[2] fetch_risk_free_rate(today)")
r = fetch_risk_free_rate(date.today())
print(f"    r = {r:.4%}")
assert 0.0 < r < 0.20, f"Risk-free rate out of sensible range: {r}"

# ── 3. Stock + option portfolio — all three VaR/ES models ─────────────────────
print("\n[3] Stock+option portfolio → RiskEngineService.run_all()")
eq_prices = prices[["AAPL", "MSFT"]].dropna()
portfolio = Portfolio(
    stocks=[
        StockPosition(ticker="AAPL", quantity=100),
        StockPosition(ticker="MSFT", quantity=50),
    ],
    options=[
        OptionPosition(
            ticker="AAPL-C",
            underlying_ticker="AAPL",
            option_type="call",
            quantity=1,
            strike=float(eq_prices["AAPL"].iloc[-1]) * 1.05,
            maturity_date=date.today() + timedelta(days=90),
            volatility=0.25,
            risk_free_rate=r,
        ),
    ],
)
service = RiskEngineService(
    portfolio=portfolio,
    prices=eq_prices,
    pricing_date=date.today(),
    lookback_days=252,
    horizon_days=1,
    var_confidence=0.99,
    es_confidence=0.975,
    estimator="window",
    n_simulations=5_000,
)
res = service.run_all()
for model in ("historical", "parametric", "monte_carlo"):
    v = res[model]["var"]
    e = res[model]["es"]
    print(f"    {model:<12} VaR={v:,.2f}  ES={e:,.2f}")
    assert v > 0, f"{model} VaR non-positive"
    assert e >= v - 1e-6, f"{model} ES < VaR"

# ── 4. Kupiec on historical backtest ──────────────────────────────────────────
print("\n[4] run_backtest('historical') → Kupiec")
bt = service.run_backtest(model="historical")
kupiec = bt["kupiec"]
print(
    f"    n={kupiec['n_observations']}, exc={kupiec['n_exceptions']}, "
    f"p_hat={kupiec['p_hat']:.4f}, LR={kupiec['lr_stat']:.4f}"
)
assert kupiec["n_observations"] > 0

# ── 5. Lognormal long vs short VaR ────────────────────────────────────────────
print("\n[5] var_short > var_long for AAPL (5d, 99%)")
aapl_ret = np.log(eq_prices["AAPL"] / eq_prices["AAPL"].shift(1)).dropna()
mu = float(aapl_ret.mean() * 252)
sigma = float(aapl_ret.std() * np.sqrt(252))
V0 = float(eq_prices["AAPL"].iloc[-1]) * 100
h = 5 / 252
var_long = var_long_lognormal(V0=V0, mu=mu, sigma=sigma, h=h, p=0.99)
var_short = var_short_lognormal(V0=V0, mu=mu, sigma=sigma, h=h, p=0.99)
print(f"    var_long={var_long:,.2f}  var_short={var_short:,.2f}")
assert var_short > var_long, "Short VaR should exceed long VaR under GBM."

# ── 6. Merton — structural Q-PD ∈ (0,1) and P-PD > Q-PD when μ < r ────────────
print("\n[6] merton_summary — Q-PD ∈ (0,1), structural μ vs r relation")
snap = merton_summary(V0=100.0, B=80.0, r=0.05, mu=0.02, sigma=0.25, T=1.0)
print(
    f"    Q-PD={snap['Q']['PD']:.4%}  P-PD={snap['P']['PD']:.4%}  "
    f"E₀={snap['E0']:.4f}  D₀={snap['D0']:.4f}"
)
assert 0.0 < snap["Q"]["PD"] < 1.0
assert snap["P"]["PD"] > snap["Q"]["PD"], "μ<r should imply P-PD > Q-PD."

# ── 7. CDS §14 landmark: λ=3%, R=40% → ~180 bps ───────────────────────────────
print("\n[7] cds_par_spread_constant_hazard(0.03, 0.40) ≈ 180 bps")
approx_spread = cds_par_spread_constant_hazard(0.03, 0.40)
bps = approx_spread * 1e4
print(f"    ≈ {bps:.1f} bps")
assert abs(bps - 180.0) < 10.0, f"Expected ~180 bps, got {bps}"

# ── 8. CVA on MC-derived exposure ─────────────────────────────────────────────
print("\n[8] cva_discrete on MC-derived exposure profile")
rng = np.random.default_rng(0)
spots0 = eq_prices.iloc[-1]
mu_daily = np.log(eq_prices / eq_prices.shift(1)).dropna().mean().values
cov_daily = np.log(eq_prices / eq_prices.shift(1)).dropna().cov().values
horizons = [1 / 12, 3 / 12, 6 / 12, 1.0]
exposures = np.zeros(len(horizons))
from src.portfolio.portfolio import portfolio_value, reprice_portfolio

V0_port = float(portfolio_value(portfolio, spots0, date.today()))
for i, T in enumerate(horizons):
    h_days = max(1, int(round(T * 252.0)))
    R_sim = rng.multivariate_normal(mu_daily * h_days, cov_daily * h_days, 2_000)
    V_sim = np.empty(len(R_sim))
    for k in range(len(R_sim)):
        shocked = spots0.copy()
        for j, u in enumerate(eq_prices.columns):
            shocked[u] = float(spots0[u]) * float(np.exp(R_sim[k][j]))
        V_sim[k] = reprice_portfolio(portfolio, shocked, date.today())
    exposures[i] = float(epe_profile_from_mc(V_sim, V0_port)[0])

lam = 0.03
s = np.exp(-lam * np.array(horizons))
marginal = np.concatenate(([1.0 - s[0]], -np.diff(s)))
cva = cva_discrete(exposures, marginal, R=0.40)
print(f"    EPE={exposures}  CVA={cva:,.4f}")
assert cva > 0, "CVA must be strictly positive when exposures are."

# ── 9. RWA + DFAST ────────────────────────────────────────────────────────────
print("\n[9] compute_rwa_and_ratio + run_dfast")
rwa_res = compute_rwa_and_ratio(
    portfolio=portfolio,
    prices=spots0,
    risk_weights={"AAPL": 1.0, "MSFT": 1.0},
    equity=0.10 * V0_port,
    pricing_date=date.today(),
)
print(
    f"    RWA={rwa_res['rwa']:,.2f}  ratio={rwa_res['ratio']:.4%}  "
    f"pass={rwa_res['pass']}"
)
assert np.isfinite(rwa_res["rwa"])
assert isinstance(rwa_res["pass"], bool)

dfast = run_dfast(portfolio, spots0, date.today())
for name, r_ in dfast.items():
    print(f"    {name:<20} PnL={r_['pnl']:,.2f} ({r_['pnl_pct']:+.2%})")
    assert np.isfinite(r_["pnl"])
# Severely adverse should be worse than baseline.
assert dfast["severely_adverse"]["pnl"] < dfast["baseline"]["pnl"]

print("\n" + "=" * 70)
print("ALL FORMULA-SHEET INTEGRATION TESTS PASSED.")
print("=" * 70)
