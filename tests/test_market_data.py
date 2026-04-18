"""
test_market_data.py
Unit tests for src/data/market_data.py with yfinance mocked.
Covers: CSV load, plain downloader (multi+single+error paths), cached
downloader (cache hit/miss/corrupt, retries, per-ticker fallback),
and fetch_risk_free_rate (success + fallback).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import logging
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import market_data


# ── CSV loader ────────────────────────────────────────────────────────────────

class TestLoadCSV:
    def test_load_simple_csv(self):
        csv = "Date,AAPL,MSFT\n2024-01-02,100,200\n2024-01-03,101,201\n"
        df = market_data.load_price_history_csv(io.StringIO(csv))
        assert list(df.columns) == ["AAPL", "MSFT"]
        assert len(df) == 2
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_csv_sorts_by_date(self):
        csv = "Date,AAPL\n2024-01-03,101\n2024-01-02,100\n"
        df = market_data.load_price_history_csv(io.StringIO(csv))
        assert df.index[0] < df.index[1]

    def test_load_csv_all_nan_raises(self):
        csv = "Date,AAPL\n2024-01-02,\n2024-01-03,\n"
        with pytest.raises(ValueError, match="empty DataFrame"):
            market_data.load_price_history_csv(io.StringIO(csv))


# ── download_adjusted_close — mocked yfinance ─────────────────────────────────

def _multi_ticker_frame(tickers):
    """Build a fake yfinance multi-ticker response."""
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    n = len(dates)
    cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
    data = np.arange(n * 2 * len(tickers), dtype=float).reshape(n, 2 * len(tickers)) + 100
    return pd.DataFrame(data, index=dates, columns=cols)


def _single_ticker_frame():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0], "Open": [99, 100, 101]}, index=dates)


class TestDownloadAdjustedClose:
    def test_multi_ticker_success(self):
        with patch("src.data.market_data.yf.download",
                   return_value=_multi_ticker_frame(["AAPL", "MSFT"])):
            df = market_data.download_adjusted_close(["AAPL", "MSFT"],
                                                     "2024-01-01", "2024-01-05")
        assert set(df.columns) == {"AAPL", "MSFT"}
        assert len(df) == 3

    def test_single_ticker_success(self):
        with patch("src.data.market_data.yf.download",
                   return_value=_single_ticker_frame()):
            df = market_data.download_adjusted_close(["AAPL"],
                                                     "2024-01-01", "2024-01-05")
        assert list(df.columns) == ["AAPL"]

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers list is empty"):
            market_data.download_adjusted_close([], "2024-01-01", "2024-01-05")

    def test_empty_response_raises(self):
        with patch("src.data.market_data.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="no rows"):
                market_data.download_adjusted_close(["AAPL"],
                                                    "2024-01-01", "2024-01-05")

    def test_none_response_raises(self):
        with patch("src.data.market_data.yf.download", return_value=None):
            with pytest.raises(ValueError, match="no rows"):
                market_data.download_adjusted_close(["AAPL"],
                                                    "2024-01-01", "2024-01-05")

    def test_multi_missing_close_level(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        cols = pd.MultiIndex.from_product([["Open"], ["AAPL"]])
        bad = pd.DataFrame(np.ones((3, 1)), index=dates, columns=cols)
        with patch("src.data.market_data.yf.download", return_value=bad):
            with pytest.raises(ValueError, match="missing 'Close' level"):
                market_data.download_adjusted_close(["AAPL"], "a", "b")

    def test_single_missing_close_column(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        bad = pd.DataFrame({"Open": [1, 2, 3]}, index=dates)
        with patch("src.data.market_data.yf.download", return_value=bad):
            with pytest.raises(ValueError, match="missing 'Close' column"):
                market_data.download_adjusted_close(["AAPL"], "a", "b")

    def test_all_nan_close_raises(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        nan_frame = pd.DataFrame({"Close": [np.nan, np.nan, np.nan]}, index=dates)
        with patch("src.data.market_data.yf.download", return_value=nan_frame):
            with pytest.raises(ValueError, match="all Close prices are NaN"):
                market_data.download_adjusted_close(["AAPL"], "a", "b")

    def test_missing_ticker_warning_logged(self, caplog):
        # Multi-ticker frame with only AAPL → MSFT listed but missing.
        frame = _multi_ticker_frame(["AAPL"])
        with patch("src.data.market_data.yf.download", return_value=frame):
            with caplog.at_level(logging.WARNING, logger="src.data.market_data"):
                df = market_data.download_adjusted_close(
                    ["AAPL", "MSFT"], "a", "b"
                )
        assert "MSFT" in df.columns or any("MSFT" in r.message for r in caplog.records)


# ── download_adjusted_close_cached ────────────────────────────────────────────

class TestCachedDownloader:
    def test_cache_hit(self, tmp_path):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        frame = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=dates)
        # Pre-populate cache.
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = market_data._cache_key(["AAPL"], "2024-01-01", "2024-01-05")
        frame.to_parquet(cache_dir / f"{key}.parquet")
        # Ensure yfinance is NOT called.
        with patch("src.data.market_data.yf.download",
                   side_effect=RuntimeError("should not be called")):
            df = market_data.download_adjusted_close_cached(
                ["AAPL"], "2024-01-01", "2024-01-05", cache_dir=str(cache_dir),
            )
        pd.testing.assert_frame_equal(df, frame, check_freq=False)

    def test_cache_miss_writes_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        with patch("src.data.market_data.yf.download",
                   return_value=_multi_ticker_frame(["AAPL", "MSFT"])):
            df = market_data.download_adjusted_close_cached(
                ["AAPL", "MSFT"], "2024-01-01", "2024-01-05",
                cache_dir=str(cache_dir),
            )
        assert len(df) > 0
        # Parquet file should now exist.
        parquet_files = list(cache_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_use_cache_false_bypasses_existing(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Put a wrong frame in the cache.
        key = market_data._cache_key(["AAPL"], "2024-01-01", "2024-01-05")
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        wrong = pd.DataFrame({"AAPL": [999.0, 999.0, 999.0]}, index=dates)
        wrong.to_parquet(cache_dir / f"{key}.parquet")
        # Fresh download should replace it.
        with patch("src.data.market_data.yf.download",
                   return_value=_single_ticker_frame()):
            df = market_data.download_adjusted_close_cached(
                ["AAPL"], "2024-01-01", "2024-01-05",
                cache_dir=str(cache_dir), use_cache=False,
            )
        assert df["AAPL"].iloc[0] != 999.0

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers list is empty"):
            market_data.download_adjusted_close_cached([], "a", "b", cache_dir=None)

    def test_corrupt_cache_falls_through(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = market_data._cache_key(["AAPL"], "2024-01-01", "2024-01-05")
        # Write garbage to the parquet path.
        (cache_dir / f"{key}.parquet").write_bytes(b"not a parquet")
        with patch("src.data.market_data.yf.download",
                   return_value=_single_ticker_frame()):
            df = market_data.download_adjusted_close_cached(
                ["AAPL"], "2024-01-01", "2024-01-05", cache_dir=str(cache_dir),
            )
        assert len(df) == 3

    def test_retry_then_per_ticker_fallback(self, tmp_path, monkeypatch):
        """Batch fails N times, then per-ticker succeeds for one of two."""
        cache_dir = tmp_path / "cache"

        calls = {"batch": 0, "per_ticker": 0}

        def flaky_download(tickers, **kw):
            # Multi-ticker batch call fails every time.
            if isinstance(tickers, list) and len(tickers) > 1:
                calls["batch"] += 1
                raise RuntimeError("batch bust")
            # Per-ticker: AAPL succeeds, MSFT still fails.
            calls["per_ticker"] += 1
            if tickers == ["AAPL"] or tickers == "AAPL":
                return _single_ticker_frame()
            raise RuntimeError("msft bust")

        # Skip sleeps to keep tests fast.
        monkeypatch.setattr(market_data.time, "sleep", lambda *a, **kw: None)
        with patch("src.data.market_data.yf.download", side_effect=flaky_download):
            df = market_data.download_adjusted_close_cached(
                ["AAPL", "MSFT"], "2024-01-01", "2024-01-05",
                cache_dir=str(cache_dir), max_retries=2,
            )
        assert "AAPL" in df.columns
        assert calls["batch"] >= 2  # retried
        assert calls["per_ticker"] >= 1

    def test_all_fail_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(market_data.time, "sleep", lambda *a, **kw: None)
        with patch("src.data.market_data.yf.download",
                   side_effect=RuntimeError("nope")):
            with pytest.raises(ValueError, match="All download attempts failed"):
                market_data.download_adjusted_close_cached(
                    ["AAPL", "MSFT"], "2024-01-01", "2024-01-05",
                    cache_dir=None, max_retries=1,
                )

    def test_cache_write_failure_swallowed(self, tmp_path, monkeypatch):
        """If the cache path can't be written, the function still returns data."""
        cache_dir = tmp_path / "cache"

        def bad_parquet(self, path, **kw):  # noqa: ARG001
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", bad_parquet)
        with patch("src.data.market_data.yf.download",
                   return_value=_single_ticker_frame()):
            df = market_data.download_adjusted_close_cached(
                ["AAPL"], "2024-01-01", "2024-01-05", cache_dir=str(cache_dir),
            )
        assert len(df) == 3


# ── fetch_risk_free_rate ──────────────────────────────────────────────────────

class TestFetchRiskFreeRate:
    def test_success(self, tmp_path):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        frame = pd.DataFrame({"Close": [4.2, 4.25, 4.3]}, index=dates)
        with patch("src.data.market_data.yf.download", return_value=frame):
            r = market_data.fetch_risk_free_rate(
                date(2024, 1, 5), fallback=0.04, cache_dir=str(tmp_path),
            )
        assert r == pytest.approx(0.043)

    def test_fallback_on_empty(self, tmp_path):
        with patch("src.data.market_data.yf.download", return_value=pd.DataFrame()):
            r = market_data.fetch_risk_free_rate(
                date(2024, 1, 5), fallback=0.05, cache_dir=str(tmp_path),
            )
        assert r == 0.05

    def test_fallback_out_of_range(self, tmp_path):
        # ^TNX = 9999 ⇒ rate = 99.99 ⇒ sanity check fails ⇒ fallback.
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        frame = pd.DataFrame({"Close": [9999.0, 9999.0, 9999.0]}, index=dates)
        with patch("src.data.market_data.yf.download", return_value=frame):
            r = market_data.fetch_risk_free_rate(
                date(2024, 1, 5), fallback=0.07, cache_dir=str(tmp_path),
            )
        assert r == 0.07


# ── Benchmark constants ───────────────────────────────────────────────────────

def test_benchmark_tickers_present():
    assert market_data.BENCHMARK_TICKERS["SP500"] == "^GSPC"
    assert market_data.BENCHMARK_TICKERS["TNX"] == "^TNX"
    assert market_data.BENCHMARK_TICKERS["VIX"] == "^VIX"
    assert market_data.BENCHMARK_TICKERS["NASDAQ100"] == "^NDX"
