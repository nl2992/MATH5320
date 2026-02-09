"""
schemas.py
Data model definitions for the MATH5320 Risk System.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date


@dataclass
class StockPosition:
    """A long or short equity position."""
    ticker: str
    quantity: float  # positive = long, negative = short

    def __post_init__(self) -> None:
        self.ticker = str(self.ticker).strip().upper()
        self.quantity = float(self.quantity)
        if not self.ticker:
            raise ValueError("Stock ticker must be non-empty.")
        if not math.isfinite(self.quantity):
            raise ValueError("Stock quantity must be finite.")


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

    def __post_init__(self) -> None:
        self.ticker = str(self.ticker).strip()
        self.underlying_ticker = str(self.underlying_ticker).strip().upper()
        self.option_type = str(self.option_type).strip().lower()
        self.quantity = float(self.quantity)
        self.strike = float(self.strike)
        self.volatility = float(self.volatility)
        self.risk_free_rate = float(self.risk_free_rate)
        self.dividend_yield = float(self.dividend_yield)
        self.contract_multiplier = float(self.contract_multiplier)

        if not self.ticker:
            raise ValueError("Option label must be non-empty.")
        if not self.underlying_ticker:
            raise ValueError("Option underlying ticker must be non-empty.")
        if self.option_type not in {"call", "put"}:
            raise ValueError("Option type must be 'call' or 'put'.")
        if not isinstance(self.maturity_date, date):
            raise ValueError("Option maturity_date must be a date.")
        if not math.isfinite(self.quantity):
            raise ValueError("Option quantity must be finite.")
        if not math.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("Option strike must be strictly positive and finite.")
        if not math.isfinite(self.volatility) or self.volatility <= 0.0:
            raise ValueError("Option volatility must be strictly positive and finite.")
        if not math.isfinite(self.risk_free_rate):
            raise ValueError("Option risk_free_rate must be finite.")
        if not math.isfinite(self.dividend_yield):
            raise ValueError("Option dividend_yield must be finite.")
        if not math.isfinite(self.contract_multiplier) or self.contract_multiplier <= 0.0:
            raise ValueError("Option contract_multiplier must be strictly positive and finite.")


@dataclass
class Portfolio:
    """Container for a mixed stock / option portfolio."""
    stocks: list[StockPosition] = field(default_factory=list)
    options: list[OptionPosition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stocks = list(self.stocks)
        self.options = list(self.options)
