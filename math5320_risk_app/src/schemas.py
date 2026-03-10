"""
schemas.py
Data model definitions for the MATH5320 Risk System.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class StockPosition:
    """A long or short equity position."""
    ticker: str
    quantity: float  # positive = long, negative = short


@dataclass
class OptionPosition:
    """A European call or put option position."""
    ticker: str                  # option identifier label
    underlying_ticker: str       # must match a column in the price DataFrame
    option_type: str             # "call" or "put"
    quantity: float              # number of contracts (positive = long)
    strike: float
    maturity_date: date
    volatility: float            # annualised implied vol (e.g. 0.20 for 20%)
    risk_free_rate: float        # continuously compounded (e.g. 0.05)
    dividend_yield: float = 0.0  # continuous dividend yield
    contract_multiplier: float = 100.0  # shares per contract


@dataclass
class Portfolio:
    """Container for a mixed stock / option portfolio."""
    stocks: list[StockPosition] = field(default_factory=list)
    options: list[OptionPosition] = field(default_factory=list)
