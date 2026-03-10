"""
market_data.py
Market data ingestion: CSV upload and yfinance download.
"""
from __future__ import annotations

import io
from typing import Union

import pandas as pd
import yfinance as yf


def load_price_history_csv(file_obj: Union[str, io.IOBase]) -> pd.DataFrame:
    """
    Load adjusted-close price history from a CSV file.

    Expected CSV format (wide):
        Date,AAPL,MSFT,SPY
        2020-01-02,300.1,150.2,320.5
        ...

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (ascending), Columns: ticker symbols, Values: prices.
    """
    df = pd.read_csv(file_obj, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def download_adjusted_close(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted-close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols, e.g. ["AAPL", "MSFT"].
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex, Columns: ticker symbols, Values: adjusted close prices.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns MultiIndex when multiple tickers requested
        prices = raw["Close"]
    else:
        # Single ticker — raw is a flat DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    return prices
