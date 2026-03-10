"""
black_scholes.py
Black-Scholes pricing and delta for European calls and puts.

Formulas (from spec):
    d1 = [ln(S/K) + (r - q + 0.5 σ²) T] / (σ √T)
    d2 = d1 - σ √T

    Call price : C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  price : P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    Call delta : Δ = e^{-qT} N(d1)
    Put  delta : Δ = e^{-qT} (N(d1) - 1)
"""
from __future__ import annotations

import math

from scipy.stats import norm


def _d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0.0:
        raise ValueError("Time to maturity T must be positive.")
    if sigma <= 0.0:
        raise ValueError("Volatility sigma must be positive.")
    if S <= 0.0:
        raise ValueError("Spot price S must be positive.")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Annualised volatility.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Option price per share.
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type.lower() == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type '{option_type}'. Use 'call' or 'put'.")


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black-Scholes delta for a European option.

    Parameters
    ----------
    (same as bs_price)

    Returns
    -------
    float
        Delta (∂V/∂S per share).
    """
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)

    if option_type.lower() == "call":
        return disc_q * norm.cdf(d1)
    elif option_type.lower() == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError(f"Unknown option_type '{option_type}'. Use 'call' or 'put'.")
