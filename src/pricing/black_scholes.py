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
    """Black-Scholes price for a European call or put.

    Uses the Garman-Kohlhagen form with a continuous dividend yield q:
        C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
        P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    Args:
        S (float): Current spot price of the underlying (must be > 0).
        K (float): Strike price (must be > 0).
        T (float): Time to maturity in years (must be > 0).
        r (float): Continuously compounded risk-free rate (annualised).
        q (float): Continuous dividend yield (annualised; use 0.0 if none).
        sigma (float): Annualised implied volatility (must be > 0).
        option_type (str): ``"call"`` or ``"put"`` (case-insensitive).

    Returns:
        float: Option price per underlying share (not per contract).

    Raises:
        ValueError: If T <= 0, sigma <= 0, S <= 0, or option_type is unknown.

    Example:
        >>> round(bs_price(S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.20, option_type="call"), 4)
        10.4506
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
    """Black-Scholes delta (∂V/∂S) for a European call or put.

        Call delta : Δ = e^{-qT} N(d1)      ∈ (0, 1)
        Put  delta : Δ = e^{-qT} (N(d1)−1)  ∈ (-1, 0)

    Args:
        S (float): Current spot price of the underlying (must be > 0).
        K (float): Strike price (must be > 0).
        T (float): Time to maturity in years (must be > 0).
        r (float): Continuously compounded risk-free rate (annualised).
        q (float): Continuous dividend yield (annualised).
        sigma (float): Annualised implied volatility (must be > 0).
        option_type (str): ``"call"`` or ``"put"`` (case-insensitive).

    Returns:
        float: Delta ∂V/∂S per underlying share.

    Raises:
        ValueError: If T <= 0, sigma <= 0, S <= 0, or option_type is unknown.

    Example:
        >>> round(bs_delta(S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.20, option_type="call"), 4)
        0.6368
    """
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)

    if option_type.lower() == "call":
        return disc_q * norm.cdf(d1)
    elif option_type.lower() == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError(f"Unknown option_type '{option_type}'. Use 'call' or 'put'.")
