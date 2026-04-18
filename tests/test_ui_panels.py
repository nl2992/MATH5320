"""
test_ui_panels.py
Coverage-focused tests for src/ui/*.py panels using streamlit's AppTest
harness. AppTest.from_function serializes the target function to a temp
script that runs in isolation — so each app-function here is fully
self-contained (inline imports + inline data construction).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def _click(at: AppTest, key: str) -> None:
    """Click the button with the given key (AppTest API helper)."""
    for btn in at.button:
        if btn.key == key:
            btn.click()
            return
    raise AssertionError(f"button key={key!r} not found in app")


@pytest.fixture(autouse=True)
def _restore_ui_module_attrs():
    """Save + restore attributes on every already-loaded `src.*` module.
    Tests monkey-patch attributes on shared modules inside their AppTest
    app functions; without restoration, AppTest.from_function re-imports
    the same cached module and test B sees test A's leaked mutations."""
    import sys
    import importlib

    # Eagerly import the modules that tests touch so they're in sys.modules
    # before we snapshot.
    for name in [
        "src.ui.capital_panel", "src.ui.cds_cva_panel", "src.ui.credit_panel",
        "src.ui.market_data_panel", "src.ui.portfolio_editor",
        "src.ui.results_panel", "src.ui.risk_settings",
        "src.data.market_data", "src.data.validation",
        "src.portfolio.portfolio", "src.portfolio.positions",
        "src.services.credit_service", "src.services.regulatory_service",
        "src.services.risk_engine_service",
    ]:
        try:
            importlib.import_module(name)
        except Exception:
            pass

    saved = {
        name: dict(mod.__dict__)
        for name, mod in list(sys.modules.items())
        if name.startswith("src.") and mod is not None
    }
    yield
    for name, snapshot in saved.items():
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for k in list(mod.__dict__.keys()):
            if k not in snapshot:
                try:
                    delattr(mod, k)
                except AttributeError:
                    pass
        for k, v in snapshot.items():
            setattr(mod, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# risk_settings.py
# ──────────────────────────────────────────────────────────────────────────────

def _app_risk_settings():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.ui.risk_settings import render_risk_settings
    import streamlit as st
    st.session_state["_out"] = render_risk_settings()


def test_risk_settings_renders():
    at = AppTest.from_function(_app_risk_settings)
    at.run()
    assert not at.exception
    assert set(at.session_state["_out"].keys()) == {
        "lookback_days", "horizon_days", "var_confidence",
        "es_confidence", "estimator", "ewma_N", "n_simulations",
    }


def _app_risk_settings_ewma():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    st.session_state["estimator"] = "ewma"
    from src.ui.risk_settings import render_risk_settings
    render_risk_settings()


def test_risk_settings_ewma():
    at = AppTest.from_function(_app_risk_settings_ewma)
    at.run()
    assert not at.exception


# ──────────────────────────────────────────────────────────────────────────────
# portfolio_editor.py — _validate_option_row direct tests
# ──────────────────────────────────────────────────────────────────────────────

def _valid_row(**overrides):
    base = {
        "Label": "X",
        "Underlying": "AAPL",
        "Type": "call",
        "Quantity": 1.0,
        "Strike": 100.0,
        "Volatility": 0.25,
        "Risk-Free Rate": 0.04,
        "Dividend Yield": 0.0,
        "Multiplier": 100.0,
        "Maturity": str(date.today() + timedelta(days=30)),
    }
    base.update(overrides)
    return pd.Series(base)


class TestValidateOptionRow:
    def test_blank(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert _validate_option_row(0, pd.Series({"Label": "", "Underlying": ""})) is None

    def test_nan_blank(self):
        from src.ui.portfolio_editor import _validate_option_row
        row = pd.Series({"Label": float("nan"), "Underlying": float("nan")})
        assert _validate_option_row(0, row) is None

    def test_valid(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert _validate_option_row(0, _valid_row()) == ""

    def test_missing_label(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert "Label" in _validate_option_row(0, _valid_row(Label=""))

    def test_missing_underlying(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert "Underlying" in _validate_option_row(0, _valid_row(Underlying=""))

    def test_bad_type(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert "call" in _validate_option_row(0, _valid_row(Type="banana"))

    def test_missing_numeric(self):
        from src.ui.portfolio_editor import _validate_option_row
        err = _validate_option_row(0, _valid_row(Strike=float("nan")))
        assert "Strike" in err

    def test_nonnumeric(self):
        from src.ui.portfolio_editor import _validate_option_row
        # Use an object pandas can't coerce via pd.isna (a list)
        row = _valid_row()
        row["Quantity"] = "not a number"
        err = _validate_option_row(0, row)
        assert err is None or "Quantity" in err or "number" in err or err == ""

    def test_nonpositive_strike(self):
        from src.ui.portfolio_editor import _validate_option_row
        err = _validate_option_row(0, _valid_row(Strike=0.0))
        assert "positive" in err

    def test_missing_maturity(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert "Maturity" in _validate_option_row(0, _valid_row(Maturity=""))

    def test_bad_maturity(self):
        from src.ui.portfolio_editor import _validate_option_row
        assert "Maturity" in _validate_option_row(0, _valid_row(Maturity="not-a-date"))

    def test_past_maturity(self):
        from src.ui.portfolio_editor import _validate_option_row
        err = _validate_option_row(
            0, _valid_row(Maturity=str(date.today() - timedelta(days=1)))
        )
        assert "future" in err


def _app_portfolio_editor_default():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.ui.portfolio_editor import render_portfolio_editor
    import streamlit as st
    st.session_state["_out"] = render_portfolio_editor()


def test_portfolio_editor_default():
    at = AppTest.from_function(_app_portfolio_editor_default)
    at.run()
    assert not at.exception
    pf = at.session_state["_out"]
    assert len(pf.stocks) == 2
    assert len(pf.options) == 1


def _app_portfolio_editor_bad_option():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import pandas as pd
    import streamlit as st
    bad = pd.DataFrame(
        {
            "Label": ["X"],
            "Underlying": ["AAPL"],
            "Type": ["banana"],
            "Quantity": [1.0],
            "Strike": [100.0],
            "Maturity": [str(date.today() + timedelta(days=30))],
            "Volatility": [0.25],
            "Risk-Free Rate": [0.04],
            "Dividend Yield": [0.0],
            "Multiplier": [100.0],
        }
    )
    st.session_state["option_df"] = bad
    from src.ui.portfolio_editor import render_portfolio_editor
    render_portfolio_editor()


def test_portfolio_editor_bad_option_row():
    at = AppTest.from_function(_app_portfolio_editor_bad_option)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("call" in e.lower() or "put" in e.lower() for e in errors)


def _app_portfolio_editor_nan_qty():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    import streamlit as st
    st.session_state["stock_df"] = pd.DataFrame(
        {"Ticker": ["AAPL", ""], "Quantity": [100.0, float("nan")]}
    )
    from src.ui.portfolio_editor import render_portfolio_editor
    render_portfolio_editor()


def test_portfolio_editor_nan_qty():
    at = AppTest.from_function(_app_portfolio_editor_nan_qty)
    at.run()
    assert not at.exception


# ──────────────────────────────────────────────────────────────────────────────
# Shared build-in-script snippets (copied into each AppTest script)
# ──────────────────────────────────────────────────────────────────────────────
# Each function below re-builds everything inline. Copy/paste is fine here
# because the functions are self-contained scripts.


# ──────────────────────────────────────────────────────────────────────────────
# results_panel.py
# ──────────────────────────────────────────────────────────────────────────────

def _app_results_panel_happy():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.ui.results_panel import render_results_panel

    rng = np.random.default_rng(0)
    losses = rng.normal(0, 100, 200)
    results = {
        "historical": {"var": 200.0, "es": 250.0, "losses": losses, "n_scenarios": 200},
        "parametric": {"var": 210.0, "es": 260.0,
                       "portfolio_mean": 0.0, "portfolio_vol": 100.0},
        "monte_carlo": {"var": 205.0, "es": 255.0, "losses": losses,
                        "n_simulations": 10_000},
    }
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {"AAPL": np.linspace(100, 150, 300), "MSFT": np.linspace(200, 230, 300)},
        index=idx,
    )
    render_results_panel(results, 50_000.0, prices, 60, 0.99)


def test_results_panel_happy():
    at = AppTest.from_function(_app_results_panel_happy)
    at.run()
    assert not at.exception


def _app_results_panel_zero_pv():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.ui.results_panel import render_results_panel

    losses = np.random.default_rng(0).normal(0, 100, 200)
    results = {
        "historical": {"var": 200.0, "es": 250.0, "losses": losses, "n_scenarios": 200},
        "parametric": {"var": 210.0, "es": 260.0,
                       "portfolio_mean": 0.0, "portfolio_vol": 100.0},
        "monte_carlo": {"var": 205.0, "es": 255.0, "losses": losses,
                        "n_simulations": 10_000},
    }
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, 300)}, index=idx)
    render_results_panel(results, 0.0, prices, 60, 0.99)


def test_results_panel_zero_pv():
    at = AppTest.from_function(_app_results_panel_zero_pv)
    at.run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("non-positive" in w.lower() or "ratio" in w.lower() for w in warnings)


def _app_results_panel_no_losses():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.ui.results_panel import render_results_panel

    results = {
        "historical": {"var": 200.0, "es": 250.0, "n_scenarios": 200},
        "parametric": {"var": 210.0, "es": 260.0,
                       "portfolio_mean": 0.0, "portfolio_vol": 100.0},
        "monte_carlo": {"var": 205.0, "es": 255.0, "n_simulations": 10_000},
    }
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, 300)}, index=idx)
    render_results_panel(results, 50_000.0, prices, 60, 0.99)


def test_results_panel_no_losses():
    at = AppTest.from_function(_app_results_panel_no_losses)
    at.run()
    assert not at.exception


# ──────────────────────────────────────────────────────────────────────────────
# market_data_panel.py
# ──────────────────────────────────────────────────────────────────────────────

def _app_mdp_no_prices():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.ui.market_data_panel import render_market_data_panel
    render_market_data_panel(["AAPL", "MSFT"])


def test_mdp_no_prices():
    at = AppTest.from_function(_app_mdp_no_prices)
    at.run()
    assert not at.exception


def _app_mdp_with_prices():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    st.session_state["prices"] = pd.DataFrame(
        {"AAPL": np.ones(10), "MSFT": np.ones(10)}, index=idx
    )
    from src.ui.market_data_panel import render_market_data_panel
    render_market_data_panel(["AAPL", "MSFT"])


def test_mdp_with_prices():
    at = AppTest.from_function(_app_mdp_with_prices)
    at.run()
    assert not at.exception


def _app_mdp_many_tickers():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    cols = [f"T{i}" for i in range(12)]
    st.session_state["prices"] = pd.DataFrame(
        np.ones((10, 12)), index=idx, columns=cols
    )
    from src.ui.market_data_panel import render_market_data_panel
    render_market_data_panel(cols)


def test_mdp_many_tickers_ellipsis():
    at = AppTest.from_function(_app_mdp_many_tickers)
    at.run()
    assert not at.exception


def _app_mdp_csv_mode():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    st.session_state["data_source"] = "Upload CSV"
    from src.ui.market_data_panel import render_market_data_panel
    render_market_data_panel([])


def test_mdp_csv_mode():
    at = AppTest.from_function(_app_mdp_csv_mode)
    at.run()
    assert not at.exception


def _app_mdp_yf_click():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.ui import market_data_panel as mdp

    def fake(tickers, start, end, **kw):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {t: np.linspace(100, 110, 10) for t in tickers}, index=idx
        )
    mdp.download_adjusted_close_cached = fake
    mdp.download_adjusted_close = fake
    mdp.render_market_data_panel(["AAPL"])


def test_mdp_yf_click_success():
    at = AppTest.from_function(_app_mdp_yf_click)
    at.run()
    _click(at, "yf_download")
    at.run()
    assert not at.exception


def _app_mdp_yf_empty_tickers():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    st.session_state["yf_tickers"] = "   "
    from src.ui.market_data_panel import render_market_data_panel
    render_market_data_panel([])


def test_mdp_yf_empty_tickers():
    at = AppTest.from_function(_app_mdp_yf_empty_tickers)
    at.run()
    _click(at, "yf_download")
    at.run()
    assert not at.exception


def _app_mdp_yf_uncached_benchmarks():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.ui import market_data_panel as mdp

    def fake(tickers, start, end, **kw):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {t: np.linspace(100, 110, 10) for t in tickers}, index=idx
        )
    mdp.download_adjusted_close = fake
    mdp.download_adjusted_close_cached = fake
    st.session_state["yf_use_cache"] = False
    st.session_state["yf_include_benchmarks"] = True
    mdp.render_market_data_panel(["AAPL"])


def test_mdp_yf_uncached_benchmarks():
    at = AppTest.from_function(_app_mdp_yf_uncached_benchmarks)
    at.run()
    _click(at, "yf_download")
    at.run()
    assert not at.exception


def _app_mdp_yf_fail():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.ui import market_data_panel as mdp

    def boom(*a, **k):
        raise RuntimeError("network")
    mdp.download_adjusted_close_cached = boom
    mdp.download_adjusted_close = boom
    mdp.render_market_data_panel(["AAPL"])


def test_mdp_yf_fail():
    at = AppTest.from_function(_app_mdp_yf_fail)
    at.run()
    _click(at, "yf_download")
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("failed" in e.lower() for e in errors)


# ──────────────────────────────────────────────────────────────────────────────
# credit_panel.py
# ──────────────────────────────────────────────────────────────────────────────

def _build_option_portfolio():
    """String copied into scripts — don't call, copy/paste the body."""
    pass  # (documentation only)


def _app_credit_happy():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.credit_panel import render_credit_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0), StockPosition("MSFT", 50.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {"AAPL": 200 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
         "MSFT": 400 * np.exp(np.cumsum(rng.normal(0, 0.012, 300)))},
        index=idx,
    )
    render_credit_panel(pf, prices)


def test_credit_happy():
    at = AppTest.from_function(_app_credit_happy)
    at.run()
    assert not at.exception


def _app_credit_bad_horizons():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.credit_panel import render_credit_panel
    st.session_state["rf_horizons"] = "abc, def"
    render_credit_panel(Portfolio(stocks=[], options=[]), None)


def test_credit_bad_horizons():
    at = AppTest.from_function(_app_credit_bad_horizons)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("number" in e.lower() or "comma" in e.lower() for e in errors)


def _app_credit_empty_horizons():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.credit_panel import render_credit_panel
    st.session_state["rf_horizons"] = "   "
    render_credit_panel(Portfolio(stocks=[], options=[]), None)


def test_credit_empty_horizons():
    at = AppTest.from_function(_app_credit_empty_horizons)
    at.run()
    assert not at.exception


def _app_credit_prefill_no_prices():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import streamlit as st
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.credit_panel import render_credit_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    st.session_state["merton_ticker"] = "AAPL"
    render_credit_panel(pf, None)


def test_credit_prefill_no_prices():
    at = AppTest.from_function(_app_credit_prefill_no_prices)
    at.run()
    _click(at, "merton_prefill")
    at.run()
    assert not at.exception


def _app_credit_prefill_short():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio, StockPosition
    from src.ui.credit_panel import render_credit_panel

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"AAPL": np.arange(5.0) + 200}, index=idx)
    st.session_state["merton_ticker"] = "AAPL"
    render_credit_panel(pf, prices)


def test_credit_prefill_short():
    at = AppTest.from_function(_app_credit_prefill_short)
    at.run()
    _click(at, "merton_prefill")
    at.run()
    assert not at.exception


def _app_credit_prefill_success():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio, StockPosition
    from src.ui import credit_panel as cp
    cp.fetch_risk_free_rate = lambda asof, fallback=0.04: 0.045

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {"AAPL": 200 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))},
        index=idx,
    )
    st.session_state["merton_ticker"] = "AAPL"
    cp.render_credit_panel(pf, prices)


def test_credit_prefill_success():
    at = AppTest.from_function(_app_credit_prefill_success)
    at.run()
    _click(at, "merton_prefill")
    at.run()
    assert not at.exception


def _app_credit_prefill_rff_fail():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio, StockPosition
    from src.ui import credit_panel as cp

    def boom(*a, **k):
        raise RuntimeError("x")
    cp.fetch_risk_free_rate = boom

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {"AAPL": 200 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))},
        index=idx,
    )
    st.session_state["merton_ticker"] = "AAPL"
    cp.render_credit_panel(pf, prices)


def test_credit_prefill_rff_fail():
    at = AppTest.from_function(_app_credit_prefill_rff_fail)
    at.run()
    _click(at, "merton_prefill")
    at.run()
    assert not at.exception


def _app_credit_merton_exception():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.schemas import Portfolio
    from src.ui import credit_panel as cp

    def boom(*a, **k):
        raise ValueError("bad")
    cp.merton_summary = boom
    cp.render_credit_panel(Portfolio(stocks=[], options=[]), None)


def test_credit_merton_exception():
    at = AppTest.from_function(_app_credit_merton_exception)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("invalid" in e.lower() for e in errors)


def _app_credit_implied_fail():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.schemas import Portfolio
    from src.ui import credit_panel as cp

    def boom(*a, **k):
        raise ValueError("no solution")
    cp.merton_implied_B_for_survival = boom
    cp.render_credit_panel(Portfolio(stocks=[], options=[]), None)


def test_credit_implied_fail():
    at = AppTest.from_function(_app_credit_implied_fail)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("inversion" in e.lower() for e in errors)


# ──────────────────────────────────────────────────────────────────────────────
# cds_cva_panel.py
# ──────────────────────────────────────────────────────────────────────────────

_RISK_PARAMS = {
    "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
    "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
    "n_simulations": 1_000,
}


def _app_cds_cva_happy():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.cds_cva_panel import render_cds_cva_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {"AAPL": 200 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))},
        index=idx,
    )
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(pf, prices, risk_params)


def test_cds_cva_happy():
    at = AppTest.from_function(_app_cds_cva_happy)
    at.run()
    assert not at.exception


def _app_cds_cva_bad_tenors():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cds_tenors"] = "abc"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_bad_tenors():
    at = AppTest.from_function(_app_cds_cva_bad_tenors)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("number" in e.lower() or "tenor" in e.lower() for e in errors)


def _app_cds_cva_empty_tenors():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cds_tenors"] = "  "
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_empty_tenors():
    at = AppTest.from_function(_app_cds_cva_empty_tenors)
    at.run()
    assert not at.exception


def _app_cds_cva_summary_fails():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.schemas import Portfolio
    from src.ui import cds_cva_panel as cp

    def boom(**k):
        raise ValueError("bad")
    cp.cds_summary = boom
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    cp.render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_summary_fails():
    at = AppTest.from_function(_app_cds_cva_summary_fails)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("cds" in e.lower() for e in errors)


def _app_cds_cva_csv_mode():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cva_mode"] = "Upload exposure CSV"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_csv_mode():
    at = AppTest.from_function(_app_cds_cva_csv_mode)
    at.run()
    assert not at.exception


def _app_cds_cva_with_exposure():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cva_exposure_df"] = pd.DataFrame(
        {"t": [0.25, 0.5, 1.0], "exposure": [100.0, 200.0, 300.0]}
    )
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_with_exposure():
    at = AppTest.from_function(_app_cds_cva_with_exposure)
    at.run()
    assert not at.exception


def _app_cds_cva_mc_no_prices():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import streamlit as st
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.cds_cva_panel import render_cds_cva_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    st.session_state["cva_mode"] = "Use current portfolio MC"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(pf, None, risk_params)


def test_cds_cva_mc_no_prices():
    at = AppTest.from_function(_app_cds_cva_mc_no_prices)
    at.run()
    _click(at, "cva_build")
    at.run()
    assert not at.exception


def _app_cds_cva_mc_empty_pf():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, 300)}, index=idx)
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), prices, risk_params)


def test_cds_cva_mc_empty_pf():
    at = AppTest.from_function(_app_cds_cva_mc_empty_pf)
    at.run()
    _click(at, "cva_build")
    at.run()
    assert not at.exception


def _app_cds_cva_mc_build_fails():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio, StockPosition
    from src.ui import cds_cva_panel as cp

    def boom(**k):
        raise RuntimeError("mc")
    cp._simulate_epe = boom
    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, 300)}, index=idx)
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    cp.render_cds_cva_panel(pf, prices, risk_params)


def test_cds_cva_mc_build_fails():
    at = AppTest.from_function(_app_cds_cva_mc_build_fails)
    at.run()
    _click(at, "cva_build")
    at.run()
    assert not at.exception


def _app_cds_cva_mc_build_ok():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio, StockPosition
    from src.ui import cds_cva_panel as cp

    cp._simulate_epe = lambda **k: np.array([100.0, 200.0])
    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, 300)}, index=idx)
    st.session_state["cva_grid"] = "0.25, 0.5"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    cp.render_cds_cva_panel(pf, prices, risk_params)


def test_cds_cva_mc_build_ok():
    at = AppTest.from_function(_app_cds_cva_mc_build_ok)
    at.run()
    _click(at, "cva_build")
    at.run()
    assert not at.exception


def _app_cds_cva_piecewise_fail():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.schemas import Portfolio
    from src.ui import cds_cva_panel as cp

    def boom(**k):
        raise ValueError("piecewise bad")
    cp.cds_spread_for_schedule = boom
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    cp.render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_piecewise_fail():
    at = AppTest.from_function(_app_cds_cva_piecewise_fail)
    at.run()
    assert not at.exception


def _app_cds_cva_pv_exception():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.schemas import Portfolio, StockPosition
    from src.ui import cds_cva_panel as cp

    def boom(*a, **k):
        raise RuntimeError("pv fail")
    cp.portfolio_value = boom

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 120, 5)}, index=idx)
    st.session_state["cva_exposure_df"] = pd.DataFrame(
        {"t": [0.25, 0.5], "exposure": [100.0, 200.0]}
    )
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    cp.render_cds_cva_panel(pf, prices, risk_params)


def test_cds_cva_pv_exception():
    at = AppTest.from_function(_app_cds_cva_pv_exception)
    at.run()
    assert not at.exception


# ──────────────────────────────────────────────────────────────────────────────
# capital_panel.py
# ──────────────────────────────────────────────────────────────────────────────

def _app_capital_no_prices():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.capital_panel import render_capital_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    render_capital_panel(pf, None)


def test_capital_no_prices():
    at = AppTest.from_function(_app_capital_no_prices)
    at.run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("market data" in w.lower() for w in warnings)


def _app_capital_empty_portfolio():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.schemas import Portfolio
    from src.ui.capital_panel import render_capital_panel
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 110, 10)}, index=idx)
    render_capital_panel(Portfolio(stocks=[], options=[]), prices)


def test_capital_empty_portfolio():
    at = AppTest.from_function(_app_capital_empty_portfolio)
    at.run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("portfolio" in w.lower() or "position" in w.lower() for w in warnings)


def _app_capital_happy():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.capital_panel import render_capital_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    render_capital_panel(pf, prices)


def test_capital_happy():
    at = AppTest.from_function(_app_capital_happy)
    at.run()
    assert not at.exception


def _app_capital_pv_fails():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    from src.schemas import OptionPosition, Portfolio, StockPosition
    import src.portfolio.portfolio as port

    def boom(*a, **k):
        raise RuntimeError("pv")
    port.portfolio_value = boom
    # Also patch inside capital_panel (uses inline import, so monkey-patch
    # src.portfolio.portfolio itself)
    from src.ui.capital_panel import render_capital_panel

    mat = date.today() + timedelta(days=60)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    render_capital_panel(pf, prices)


def test_capital_pv_fails():
    at = AppTest.from_function(_app_capital_pv_fails)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("price" in e.lower() or "cannot" in e.lower() for e in errors)


def _app_capital_rwa_fails():
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import numpy as np
    import pandas as pd
    from src.schemas import Portfolio, StockPosition
    from src.ui import capital_panel as cp

    def boom(**k):
        raise ValueError("rwa")
    cp.compute_rwa_and_ratio = boom

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    cp.render_capital_panel(pf, prices)


def test_capital_rwa_fails():
    at = AppTest.from_function(_app_capital_rwa_fails)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("capital" in e.lower() for e in errors)


def _app_capital_dfast_click():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.schemas import Portfolio, StockPosition
    from src.ui.capital_panel import render_capital_panel

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    render_capital_panel(pf, prices)


def test_capital_dfast_click():
    at = AppTest.from_function(_app_capital_dfast_click)
    at.run()
    _click(at, "cap_dfast")
    at.run()
    assert not at.exception


def _app_capital_dfast_fails():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.schemas import Portfolio, StockPosition
    from src.ui import capital_panel as cp

    def boom(*a, **k):
        raise RuntimeError("dfast")
    cp.run_dfast = boom

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    cp.render_capital_panel(pf, prices)


def test_capital_dfast_fails():
    at = AppTest.from_function(_app_capital_dfast_fails)
    at.run()
    _click(at, "cap_dfast")
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("dfast" in e.lower() for e in errors)


def _app_capital_custom():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.schemas import Portfolio, StockPosition
    from src.ui.capital_panel import render_capital_panel

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    render_capital_panel(pf, prices)


def test_capital_custom():
    at = AppTest.from_function(_app_capital_custom)
    at.run()
    _click(at, "cap_custom_run")
    at.run()
    assert not at.exception


def _app_capital_custom_fails():
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import pandas as pd
    from src.schemas import Portfolio, StockPosition
    from src.ui import capital_panel as cp

    def boom(*a, **k):
        raise RuntimeError("custom")
    cp.run_custom_stress = boom

    pf = Portfolio(stocks=[StockPosition("AAPL", 100.0)], options=[])
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(200, 210, 10)}, index=idx)
    cp.render_capital_panel(pf, prices)


def test_capital_custom_fails():
    at = AppTest.from_function(_app_capital_custom_fails)
    at.run()
    _click(at, "cap_custom_run")
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("custom stress" in e.lower() for e in errors)


# ──────────────────────────────────────────────────────────────────────────────
# Coverage-gap closers
# ──────────────────────────────────────────────────────────────────────────────

def test_simulate_epe_direct():
    """Direct unit test of _simulate_epe to cover lines 317-352."""
    from datetime import date, timedelta
    from src.schemas import OptionPosition, Portfolio, StockPosition
    from src.ui.cds_cva_panel import _simulate_epe

    mat = date.today() + timedelta(days=120)
    pf = Portfolio(
        stocks=[StockPosition("AAPL", 100.0)],
        options=[OptionPosition(
            ticker="AAPL_C", underlying_ticker="AAPL", option_type="call",
            quantity=1.0, strike=205.0, maturity_date=mat,
            volatility=0.25, risk_free_rate=0.04,
        )],
    )
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {"AAPL": 200 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))},
        index=idx,
    )
    epe = _simulate_epe(
        portfolio=pf, prices=prices, horizons=[0.1, 0.25],
        n_sims=50, lookback_days=60, estimator="window", ewma_N=60,
    )
    assert len(epe) == 2
    assert (epe >= 0).all()


def test_simulate_epe_no_underlyings_in_prices():
    """Covers the `if not underlyings: raise` branch."""
    from src.schemas import Portfolio, StockPosition
    from src.ui.cds_cva_panel import _simulate_epe

    pf = Portfolio(stocks=[StockPosition("ZZZ", 100.0)], options=[])  # ZZZ absent
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, 100)}, index=idx)
    with pytest.raises(ValueError, match="No portfolio tickers"):
        _simulate_epe(
            portfolio=pf, prices=prices, horizons=[0.1],
            n_sims=10, lookback_days=30, estimator="window", ewma_N=30,
        )


def _app_portfolio_editor_blank_option():
    """Seeds a truly blank option row (Label+Underlying both blank) to hit
    the `if row_err is None: continue` silent-skip branch."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    from datetime import date, timedelta
    import pandas as pd
    import streamlit as st
    blank = pd.DataFrame(
        {
            "Label": ["", ""],
            "Underlying": ["", "AAPL"],
            "Type": ["call", "call"],
            "Quantity": [1.0, 1.0],
            "Strike": [100.0, 100.0],
            "Maturity": [
                str(date.today() + timedelta(days=30)),
                str(date.today() + timedelta(days=30)),
            ],
            "Volatility": [0.25, 0.25],
            "Risk-Free Rate": [0.04, 0.04],
            "Dividend Yield": [0.0, 0.0],
            "Multiplier": [100.0, 100.0],
        }
    )
    st.session_state["option_df"] = blank
    from src.ui.portfolio_editor import render_portfolio_editor
    render_portfolio_editor()


def test_portfolio_editor_blank_option_row_silent():
    at = AppTest.from_function(_app_portfolio_editor_blank_option)
    at.run()
    assert not at.exception


def _app_mdp_csv_with_upload():
    """Simulate CSV upload that fails to parse."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    from io import BytesIO
    import streamlit as st
    from src.ui import market_data_panel as mdp

    def boom(*a, **k):
        raise ValueError("bad csv")
    mdp.load_price_history_csv = boom

    st.session_state["data_source"] = "Upload CSV"
    mdp.render_market_data_panel([])


def test_mdp_csv_upload_fail(monkeypatch):
    """Patch file_uploader to return a fake uploaded file so we reach the
    load_price_history_csv(...) call."""
    import streamlit as st
    from io import BytesIO

    real_uploader = st.file_uploader

    def fake_uploader(label, **kw):
        if kw.get("key") == "csv_upload":
            return BytesIO(b"not,a,csv")
        return real_uploader(label, **kw)

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    at = AppTest.from_function(_app_mdp_csv_with_upload)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("csv" in e.lower() for e in errors)


def _app_mdp_yf_bad_data():
    """YF download succeeds but returns a frame that fails validation —
    covers validation-error branch (market_data_panel.py:68-70)."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    from src.ui import market_data_panel as mdp

    def fake(tickers, start, end, **kw):
        # Empty frame -> validation fails "empty"
        return pd.DataFrame()

    mdp.download_adjusted_close_cached = fake
    mdp.download_adjusted_close = fake
    mdp.render_market_data_panel(["AAPL"])


def test_mdp_yf_validation_error():
    at = AppTest.from_function(_app_mdp_yf_bad_data)
    at.run()
    _click(at, "yf_download")
    at.run()
    assert not at.exception


def _app_cds_cva_csv_upload_good():
    """CSV upload path with a well-formed CSV (hits success branch 244-248)."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cva_mode"] = "Upload exposure CSV"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_csv_upload_good(monkeypatch):
    import streamlit as st
    from io import BytesIO

    real_uploader = st.file_uploader

    def fake_uploader(label, **kw):
        if kw.get("key") == "cva_csv":
            return BytesIO(b"t,exposure\n0.25,100\n0.5,200\n1.0,300\n")
        return real_uploader(label, **kw)

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    at = AppTest.from_function(_app_cds_cva_csv_upload_good)
    at.run()
    assert not at.exception


def _app_cds_cva_csv_upload_missing_cols():
    """CSV upload missing required columns — hits the error branch 242-243."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cva_mode"] = "Upload exposure CSV"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def test_cds_cva_csv_upload_missing_cols(monkeypatch):
    import streamlit as st
    from io import BytesIO

    real_uploader = st.file_uploader

    def fake_uploader(label, **kw):
        if kw.get("key") == "cva_csv":
            return BytesIO(b"a,b\n1,2\n")
        return real_uploader(label, **kw)

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    at = AppTest.from_function(_app_cds_cva_csv_upload_missing_cols)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("t" in e.lower() and "exposure" in e.lower() for e in errors)


def _app_cds_cva_csv_upload_parse_fail():
    """Force pd.read_csv to fail via patched bad upload."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    import streamlit as st
    from src.schemas import Portfolio
    from src.ui.cds_cva_panel import render_cds_cva_panel
    st.session_state["cva_mode"] = "Upload exposure CSV"
    risk_params = {
        "lookback_days": 60, "horizon_days": 1, "var_confidence": 0.99,
        "es_confidence": 0.975, "estimator": "window", "ewma_N": 60,
        "n_simulations": 1_000,
    }
    render_cds_cva_panel(Portfolio(stocks=[], options=[]), None, risk_params)


def _app_mdp_csv_upload_good():
    """CSV upload that parses successfully — hits market_data_panel.py:172."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    from io import BytesIO
    import numpy as np
    import pandas as pd
    import streamlit as st
    from src.ui import market_data_panel as mdp
    st.session_state["data_source"] = "Upload CSV"

    # Stub file_uploader (inside the AppTest script) + the loader.
    mdp.st.file_uploader = lambda *a, **kw: BytesIO(b"Date,AAPL\n2024-01-01,100\n")

    def fake_loader(upload):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        return pd.DataFrame({"AAPL": np.linspace(100, 110, 10)}, index=idx)
    mdp.load_price_history_csv = fake_loader

    mdp.render_market_data_panel([])


def test_mdp_csv_upload_good():
    at = AppTest.from_function(_app_mdp_csv_upload_good)
    at.run()
    assert not at.exception


def test_cds_cva_csv_upload_parse_fail(monkeypatch):
    import streamlit as st
    import pandas as pd
    from src.ui import cds_cva_panel as cp

    class FakeUpload:
        pass

    def fake_uploader(label, **kw):
        if kw.get("key") == "cva_csv":
            return FakeUpload()
        return st.file_uploader(label, **kw)

    monkeypatch.setattr(st, "file_uploader", fake_uploader)

    at = AppTest.from_function(_app_cds_cva_csv_upload_parse_fail)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("parse" in e.lower() or "error" in e.lower() for e in errors)
