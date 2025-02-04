"""
config.py
Global defaults for the MATH5320 Risk System.
All values can be overridden from the Streamlit UI.
"""

# ── Risk defaults ──────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS: int = 252          # 1 trading year
DEFAULT_HORIZON_DAYS: int = 1            # 1-day horizon
DEFAULT_VAR_CONFIDENCE: float = 0.99     # 99% VaR
DEFAULT_ES_CONFIDENCE: float = 0.975     # 97.5% ES
DEFAULT_ESTIMATOR: str = "window"        # "window" | "ewma"
DEFAULT_EWMA_N: int = 60                 # EWMA half-life parameter N
DEFAULT_MC_SIMULATIONS: int = 10_000    # Monte Carlo paths

# ── Market data defaults ───────────────────────────────────────────────────────
DEFAULT_YFINANCE_PERIOD: str = "5y"      # yfinance download period
TRADING_DAYS_PER_YEAR: int = 252

# ── Backtest defaults ──────────────────────────────────────────────────────────
DEFAULT_BACKTEST_MODEL: str = "historical"   # "historical" | "parametric" | "monte_carlo"
MIN_BACKTEST_OBSERVATIONS: int = 30          # minimum windows for meaningful test
