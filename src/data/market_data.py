"""
market_data.py
Market data ingestion: CSV upload and yfinance download.

Layers, from simplest to most fault-tolerant:

    load_price_history_csv(file)
    download_adjusted_close(tickers, start, end)
    download_adjusted_close_cached(tickers, start, end, cache_dir=..., max_retries=3)
    fetch_risk_free_rate(asof, fallback=0.04)

The cached variant adds a parquet on-disk cache, retry with exponential
backoff, and a per-ticker fallback when a batch download returns malformed
or partial data.  The plain downloader keeps its original signature so every
existing caller (UI, tests, integration scripts) keeps working.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Benchmark reference tickers ────────────────────────────────────────────────

BENCHMARK_TICKERS: dict[str, str] = {
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "VIX": "^VIX",
    "TNX": "^TNX",  # 10-year Treasury yield (in percent × 10, i.e. raw = pct*100)
}


# ── CSV path ──────────────────────────────────────────────────────────────────

def load_price_history_csv(file_obj: Union[str, io.IOBase]) -> pd.DataFrame:
    """Load adjusted-close price history from a wide-format CSV file.

    Expected CSV format:
        Date,AAPL,MSFT,SPY
        2020-01-02,300.1,150.2,320.5
        ...

    The first column must be parseable as dates; subsequent columns are ticker
    symbols.  All non-numeric values are coerced to NaN; rows with all-NaN
    prices are dropped.

    Args:
        file_obj (str | io.IOBase): File path string or any file-like object
            accepted by :func:`pandas.read_csv`.

    Returns:
        pd.DataFrame: DatetimeIndex (ascending) × ticker columns;
            values are adjusted-close prices.

    Raises:
        ValueError: If the CSV parses to an empty DataFrame or the index
            contains duplicate dates.
    """
    df = pd.read_csv(file_obj, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("CSV parsed to an empty DataFrame.")
    _assert_unique_datetime_index(df, source="CSV")
    return df


# ── yfinance path (plain) ─────────────────────────────────────────────────────

def download_adjusted_close(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted-close prices from Yahoo Finance (plain, no cache).

    Args:
        tickers (list[str]): Ticker symbols, e.g. ``["AAPL", "MSFT"]``.
            Benchmark tickers such as ``"^GSPC"`` and ``"^TNX"`` are
            accepted by yfinance.
        start (str): Start date in ``"YYYY-MM-DD"`` format (inclusive).
        end (str): End date in ``"YYYY-MM-DD"`` format (exclusive).

    Returns:
        pd.DataFrame: DatetimeIndex (ascending) × ticker columns;
            values are adjusted-close prices.

    Raises:
        ValueError: If ``tickers`` is empty, yfinance returns no rows,
            the ``"Close"`` column/level is missing, all prices are NaN,
            or the index contains duplicate dates.
    """
    if not tickers:
        raise ValueError("tickers list is empty.")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no rows for tickers={tickers} "
            f"over {start}→{end}. Check the ticker spellings, weekend-only "
            "date range, or network access."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker: columns are (field, ticker). Pull the Close level.
        if "Close" not in raw.columns.levels[0]:
            raise ValueError(
                f"yfinance response missing 'Close' level for tickers={tickers}. "
                f"Got top-level fields: {list(raw.columns.levels[0])}"
            )
        prices = raw["Close"]
    else:
        # Single ticker: flat DataFrame. auto_adjust=True returns 'Close'.
        if "Close" not in raw.columns:
            raise ValueError(
                f"yfinance response missing 'Close' column for ticker={tickers[0]}. "
                f"Got columns: {list(raw.columns)}"
            )
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    _assert_unique_datetime_index(prices, source="yfinance")

    if prices.empty:
        raise ValueError(
            f"yfinance returned rows but all Close prices are NaN for "
            f"tickers={tickers} over {start}→{end}."
        )

    # Warn about any ticker that came back entirely empty.
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning("yfinance returned no data for: %s", missing)

    return prices


def _assert_unique_datetime_index(df: pd.DataFrame, *, source: str) -> None:
    """Reject duplicate timestamps explicitly so downstream return logic is stable."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        sample = ", ".join(
            str(ts.date()) if hasattr(ts, "date") else str(ts) for ts in dupes[:5]
        )
        suffix = " ..." if len(dupes) > 5 else ""
        raise ValueError(f"{source} data contains duplicate dates: {sample}{suffix}")


# ── yfinance path (cached + retry) ────────────────────────────────────────────

def _cache_key(tickers: Iterable[str], start: str, end: str) -> str:
    """Stable hash of (sorted tickers, start, end) for cache-file naming."""
    joined = ",".join(sorted(tickers)) + f"|{start}|{end}"
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def download_adjusted_close_cached(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Union[str, Path, None] = ".cache/prices",
    max_retries: int = 3,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fault-tolerant price downloader with parquet cache and retry logic.

    Extends :func:`download_adjusted_close` with:
    - **Parquet cache**: responses are stored keyed on a hash of
      ``(sorted(tickers), start, end)``; cache hits avoid network calls.
    - **Exponential backoff**: up to ``max_retries`` batch attempts with
      waits of 1, 2, 4… seconds between failures.
    - **Per-ticker fallback**: if all batch attempts fail, downloads each
      ticker individually and concatenates results.

    Args:
        tickers (list[str]): Ticker symbols; may include benchmarks such as
            ``"^GSPC"`` or ``"^TNX"``.
        start (str): Start date ``"YYYY-MM-DD"`` (inclusive).
        end (str): End date ``"YYYY-MM-DD"`` (exclusive).
        cache_dir (str | Path | None): Directory for the parquet cache.
            Pass ``None`` to disable caching entirely.
        max_retries (int): Batch download attempts before falling back to
            per-ticker downloads (default 3).
        use_cache (bool): If ``False``, bypass the cache read (but still
            write on success).

    Returns:
        pd.DataFrame: Same shape as :func:`download_adjusted_close`.

    Raises:
        ValueError: If ``tickers`` is empty, or all download attempts fail
            for every ticker.
    """
    if not tickers:
        raise ValueError("tickers list is empty.")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{_cache_key(tickers, start, end)}.parquet"
        if use_cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as exc:  # corrupt cache — fall through to refetch
                logger.warning("Cache read failed (%s); refetching.", exc)

    # Batch attempt with exponential backoff.
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            prices = download_adjusted_close(tickers, start, end)
            break
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Batch download attempt %d/%d failed (%s); retrying in %ds.",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    else:
        # All batch retries failed — try per-ticker.
        logger.warning(
            "Batch download exhausted retries; falling back to per-ticker. "
            "Last error: %s", last_exc,
        )
        per_ticker_frames: list[pd.DataFrame] = []
        for t in tickers:
            try:
                per_ticker_frames.append(download_adjusted_close([t], start, end))
            except Exception as exc:
                logger.warning("Skipping %s (per-ticker failed): %s", t, exc)
        if not per_ticker_frames:
            raise ValueError(
                f"All download attempts failed for tickers={tickers}."
            ) from last_exc
        prices = pd.concat(per_ticker_frames, axis=1).sort_index()
        prices = prices.dropna(how="all")

    # Write cache (best-effort).
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(cache_path)
        except Exception as exc:
            logger.warning("Cache write failed (%s); continuing.", exc)

    return prices


# ── Risk-free rate helper ─────────────────────────────────────────────────────

def fetch_risk_free_rate(
    asof: date,
    fallback: float = 0.04,
    cache_dir: Union[str, Path, None] = ".cache/prices",
) -> float:
    """Fetch a risk-free rate proxy from Yahoo Finance's ^TNX series.

    ^TNX is the CBOE 10-year Treasury yield index, quoted as yield × 100
    (e.g. 4.25% appears as 42.50).  This function divides by 100 to return
    a decimal rate suitable for Black-Scholes and discounting.

    On any failure (network error, empty response, sanity-range violation)
    the ``fallback`` rate is returned with a ``WARNING``-level log entry so
    that downstream calculations are never blocked.

    Args:
        asof (date): Date for which to retrieve the rate.  A 14-day look-back
            buffer is used to handle weekends and holidays.
        fallback (float): Rate to return on failure (default 0.04 = 4%).
        cache_dir (str | Path | None): Parquet cache directory passed to
            :func:`download_adjusted_close_cached` (default ``".cache/prices"``).

    Returns:
        float: Risk-free rate as a decimal in [0, 0.25], or ``fallback`` on
            any error.
    """
    try:
        start = (asof - timedelta(days=14)).isoformat()  # small buffer for weekends
        end = (asof + timedelta(days=1)).isoformat()
        df = download_adjusted_close_cached(
            ["^TNX"], start=start, end=end,
            cache_dir=cache_dir, max_retries=2,
        )
        if df.empty or "^TNX" not in df.columns:
            raise ValueError("^TNX frame empty or column missing.")
        last = float(df["^TNX"].dropna().iloc[-1])
        rate = last / 100.0
        if not (0.0 <= rate <= 0.25):
            raise ValueError(f"^TNX-derived rate {rate} outside sanity range.")
        return rate
    except Exception as exc:
        logger.warning(
            "fetch_risk_free_rate(%s) failed (%s); using fallback %.4f.",
            asof, exc, fallback,
        )
        return fallback
