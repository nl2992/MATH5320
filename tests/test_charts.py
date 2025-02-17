"""
test_charts.py
Unit tests for src/ui/charts.py. Chart helpers return plotly Figures, so we
instantiate them with synthetic inputs and assert trace/layout structure.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.ui.charts import (
    backtest_chart,
    correlation_heatmap,
    loss_histogram,
    price_history_chart,
    var_comparison_bar,
)


def test_loss_histogram_returns_figure():
    losses = np.random.default_rng(0).normal(0, 100, 500)
    fig = loss_histogram(losses, var=200.0, es=300.0, title="Test", var_confidence=0.99)
    assert isinstance(fig, go.Figure)
    # One histogram trace
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Test"


def test_correlation_heatmap_returns_figure():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"A": np.arange(10, dtype=float), "B": np.arange(10, dtype=float) * 0.5},
        index=idx,
    )
    fig = correlation_heatmap(df, lookback_days=5)
    assert isinstance(fig, go.Figure)


def test_backtest_chart_with_exceptions():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": idx,
            "var_forecast": [100.0] * 5,
            "realized_loss": [50.0, 150.0, 60.0, 200.0, 70.0],
            "exception": [0, 1, 0, 1, 0],
        }
    )
    fig = backtest_chart(df, var_confidence=0.99)
    assert isinstance(fig, go.Figure)
    # VaR + realised + exceptions = 3 traces
    assert len(fig.data) == 3


def test_backtest_chart_without_exceptions():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "date": idx,
            "var_forecast": [100.0] * 3,
            "realized_loss": [50.0] * 3,
            "exception": [0, 0, 0],
        }
    )
    fig = backtest_chart(df, var_confidence=0.99)
    # No exceptions branch -> 2 traces only
    assert len(fig.data) == 2


def test_var_comparison_bar():
    results = {
        "historical": {"var": 100.0, "es": 120.0},
        "parametric": {"var": 110.0, "es": 130.0},
        "monte_carlo": {"var": 105.0, "es": 125.0},
    }
    fig = var_comparison_bar(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # VaR and ES bars


def test_price_history_chart():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"AAPL": np.linspace(100, 120, 10), "MSFT": np.linspace(300, 320, 10)},
        index=idx,
    )
    fig = price_history_chart(df, lookback_days=5)
    assert isinstance(fig, go.Figure)
    # One trace per column
    assert len(fig.data) == 2
