"""
integration_test.py
Full integration test: download real data, run all risk models, backtest.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta
import pandas as pd
import numpy as np

from src.schemas import StockPosition, OptionPosition, Portfolio
from src.data.market_data import download_adjusted_close
from src.services.risk_engine_service import RiskEngineService

print("=" * 60)
print("MATH5320 Risk System — Integration Test")
print("=" * 60)

# ── 1. Download market data ────────────────────────────────────────────────────
print("\n[1] Downloading AAPL + MSFT price history (2 years)...")
end = date.today()
start = end - timedelta(days=730)
prices = download_adjusted_close(
    tickers=["AAPL", "MSFT"],
    start=str(start),
    end=str(end),
)
print(f"    Loaded {len(prices)} rows, columns: {list(prices.columns)}")
assert len(prices) > 200, "Expected at least 200 price rows"

# ── 2. Build portfolio ─────────────────────────────────────────────────────────
print("\n[2] Building portfolio...")
maturity = date.today() + timedelta(days=90)
portfolio = Portfolio(
    stocks=[
        StockPosition(ticker="AAPL", quantity=100),
        StockPosition(ticker="MSFT", quantity=50),
    ],
    options=[
        OptionPosition(
            ticker="AAPL_CALL",
            underlying_ticker="AAPL",
            option_type="call",
            quantity=10,
            strike=float(prices["AAPL"].iloc[-1]) * 1.05,  # 5% OTM
            maturity_date=maturity,
            volatility=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            contract_multiplier=100,
        )
    ],
)
print(f"    {len(portfolio.stocks)} stocks, {len(portfolio.options)} options")

# ── 3. Instantiate service ─────────────────────────────────────────────────────
print("\n[3] Creating RiskEngineService...")
service = RiskEngineService(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date.today(),
    lookback_days=252,
    horizon_days=1,
    var_confidence=0.99,
    es_confidence=0.975,
    estimator="window",
    ewma_N=60,
    n_simulations=5000,
)

pv = service.portfolio_value()
print(f"    Portfolio Value: ${pv:,.2f}")
assert pv > 0

# ── 4. Run all risk models ─────────────────────────────────────────────────────
print("\n[4] Running all risk models...")
results = service.run_all()

for model_name, res in results.items():
    print(f"    [{model_name.upper()}]  VaR = ${res['var']:,.2f}  |  ES = ${res['es']:,.2f}")
    assert res["var"] > 0, f"{model_name} VaR must be positive"
    assert res["es"] >= res["var"], f"{model_name} ES must be >= VaR"

# ── 5. Run backtest ────────────────────────────────────────────────────────────
print("\n[5] Running walk-forward backtest (historical model)...")
bt_result = service.run_backtest(model="historical")
bt_df = bt_result["backtest_df"]
kupiec = bt_result["kupiec"]

print(f"    Observations: {kupiec['n_observations']}")
print(f"    Exceptions:   {kupiec['n_exceptions']}")
print(f"    p̂ = {kupiec['p_hat']:.4f}  (expected α = {kupiec['alpha']:.4f})")
print(f"    LR stat = {kupiec['lr_stat']:.4f},  p-value = {kupiec['p_value']:.4f}")
print(f"    Reject H₀: {kupiec['reject_h0']}")

assert kupiec["n_observations"] > 0
assert 0.0 <= kupiec["p_hat"] <= 1.0

# ── 6. EWMA estimator test ─────────────────────────────────────────────────────
print("\n[6] Testing EWMA estimator...")
service_ewma = RiskEngineService(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date.today(),
    lookback_days=252,
    horizon_days=1,
    var_confidence=0.99,
    es_confidence=0.975,
    estimator="ewma",
    ewma_N=60,
    n_simulations=2000,
)
results_ewma = service_ewma.run_all()
print(f"    EWMA Parametric VaR = ${results_ewma['parametric']['var']:,.2f}")
assert results_ewma["parametric"]["var"] > 0

# ── 7. Multi-day horizon test ──────────────────────────────────────────────────
print("\n[7] Testing 5-day horizon...")
service_5d = RiskEngineService(
    portfolio=portfolio,
    prices=prices,
    pricing_date=date.today(),
    lookback_days=252,
    horizon_days=5,
    var_confidence=0.99,
    es_confidence=0.975,
    n_simulations=2000,
)
results_5d = service_5d.run_all()
print(f"    5-day Historical VaR = ${results_5d['historical']['var']:,.2f}")
print(f"    1-day Historical VaR = ${results['historical']['var']:,.2f}")
# 5-day VaR should generally be larger than 1-day
assert results_5d["historical"]["var"] > 0

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED")
print("=" * 60)
