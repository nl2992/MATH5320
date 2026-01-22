"""
demo_presets.py
Small front-end presets used for reproducible Streamlit demos.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    DEFAULT_OPTION_VOL_SHOCK_BETA,
    DEFAULT_OPTION_VOL_SHOCK_FLOOR,
)


_OPTION_COLUMNS = [
    "Label",
    "Underlying",
    "Type",
    "Quantity",
    "Strike",
    "Maturity",
    "Volatility",
    "Risk-Free Rate",
    "Dividend Yield",
    "Multiplier",
]


def load_demo_preset(name: str) -> dict | None:
    """Return a session-state payload for a named demo preset."""
    key = str(name).strip().lower()
    if key == "advanced_m7_full":
        params = _advanced_m7_risk_params()
        return {
            "stock_df": _advanced_m7_stocks_df(),
            "option_df": _advanced_m7_options_df(),
            "risk_params": params,
            **params,
            "allow_expired_options": True,
            "loaded_demo_preset": key,
        }
    if key == "advanced_m7_stocks":
        params = _advanced_m7_risk_params()
        return {
            "stock_df": _advanced_m7_stocks_df(),
            "option_df": pd.DataFrame(columns=_OPTION_COLUMNS),
            "risk_params": params,
            **params,
            "allow_expired_options": True,
            "loaded_demo_preset": key,
        }
    return None


def _advanced_m7_stocks_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
            "Quantity": [2108.0, 1031.0, 1295.0, 1479.0, 62194.0, 481.0, 3124.0],
        }
    )


def _advanced_m7_options_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Label": ["AAPL_CALL", "AMZN_C_OTM", "TSLA_P_OTM"],
            "Underlying": ["AAPL", "AMZN", "TSLA"],
            "Type": ["call", "call", "put"],
            "Quantity": [10.0, 5.0, -8.0],
            "Strike": [24.8962, 36.4981, 14.4006],
            "Maturity": ["2016-06-30", "2016-06-30", "2016-03-31"],
            "Volatility": [0.25, 0.30, 0.55],
            "Risk-Free Rate": [0.02, 0.02, 0.02],
            "Dividend Yield": [0.0, 0.0, 0.0],
            "Multiplier": [100.0, 100.0, 100.0],
        }
    )


def _advanced_m7_risk_params() -> dict:
    return {
        "lookback_days": 252,
        "horizon_days": 5,
        "var_confidence": 0.99,
        "es_confidence": 0.975,
        "estimator": "window",
        "ewma_N": 60,
        "n_simulations": 10_000,
        "calibration_mode": "historical",
        "manual_market_params": None,
        "option_vol_shock_mode": "fixed",
        "option_vol_shock_beta": DEFAULT_OPTION_VOL_SHOCK_BETA,
        "option_vol_shock_floor": DEFAULT_OPTION_VOL_SHOCK_FLOOR,
    }
