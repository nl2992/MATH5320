"""
merton.py
Structural default model (formula-sheet §9).

Firm value follows GBM:  dV = μ V dt + σ V dW.
Default is declared at maturity T iff V_T < B.

    d₂(ν) = [ log(V₀/B) + (ν − ½σ²) T ] / (σ √T)
    d₁(ν) = d₂(ν) + σ √T
    PD     = P(V_T < B)     = N(-d₂(ν))

Measure choice:
    Q-measure (risk-neutral, pricing):  ν = r.
    P-measure (real-world, historical): ν = μ.

Valuation (under Q):
    E₀ = V₀ N(d₁(r)) − B e^{−rT} N(d₂(r))
    D₀ = V₀ − E₀

Target-survival inversion (HW IX):
    If target survival is s*, z = N⁻¹(s*), then
        B* = V₀ exp( −z σ √T + (r − ½σ²) T )

Sanity-check landmarks from formula-sheet §14:
    HW VII inputs → Q-default ≈ 29.53%, P-default ≈ 38.88%.
"""
from __future__ import annotations

import math

from scipy.stats import norm


def merton_d1_d2(V0: float, B: float, nu: float, sigma: float, T: float) -> tuple[float, float]:
    """
    Merton d₁ and d₂ under an arbitrary drift ν.

    For pricing (Q), pass ``nu = r``.  For real-world (P), pass ``nu = μ``.
    """
    _validate(V0, B, sigma, T)
    sqrtT = math.sqrt(T)
    d2 = (math.log(V0 / B) + (nu - 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d1 = d2 + sigma * sqrtT
    return d1, d2


def merton_pd(V0: float, B: float, nu: float, sigma: float, T: float) -> float:
    """Probability of default under the chosen measure: PD = N(-d₂(ν))."""
    _, d2 = merton_d1_d2(V0, B, nu, sigma, T)
    return float(norm.cdf(-d2))


def merton_equity(V0: float, B: float, r: float, sigma: float, T: float) -> float:
    """
    Equity value as a call on the firm (under Q):
        E₀ = V₀ N(d₁(r)) − B e^{−rT} N(d₂(r))
    """
    d1, d2 = merton_d1_d2(V0, B, r, sigma, T)
    return V0 * norm.cdf(d1) - B * math.exp(-r * T) * norm.cdf(d2)


def merton_debt(V0: float, B: float, r: float, sigma: float, T: float) -> float:
    """Debt value (under Q): D₀ = V₀ − E₀."""
    return V0 - merton_equity(V0, B, r, sigma, T)


def merton_credit_spread(V0: float, B: float, r: float, sigma: float, T: float) -> float:
    """
    Merton-implied credit spread on zero-coupon debt of face B (under Q):
        D₀ = B e^{−(r + s) T}   ⇒   s = −(1/T) log(D₀ / B) − r
    """
    D0 = merton_debt(V0, B, r, sigma, T)
    if D0 <= 0:
        raise ValueError("Merton D0 is non-positive; cannot invert spread.")
    return -math.log(D0 / B) / T - r


def merton_implied_B(
    V0: float, target_survival: float, r: float, sigma: float, T: float
) -> float:
    """
    Invert §9's target-survival relation:

        If target survival is s*, z = N⁻¹(s*), then
            B* = V₀ exp( −z σ √T + (r − ½σ²) T )
    """
    if not (0.0 < target_survival < 1.0):
        raise ValueError(
            f"target_survival must be in (0, 1) (got {target_survival})."
        )
    _validate(V0, B=1.0, sigma=sigma, T=T)  # B irrelevant here, pass stub
    z = norm.ppf(target_survival)
    return V0 * math.exp(-z * sigma * math.sqrt(T) + (r - 0.5 * sigma ** 2) * T)


# ── Internal ──────────────────────────────────────────────────────────────────

def _validate(V0: float, B: float, sigma: float, T: float) -> None:
    if V0 <= 0:
        raise ValueError(f"V0 must be positive (got {V0}).")
    if B <= 0:
        raise ValueError(f"B must be positive (got {B}).")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")


# ── Merton timing helpers (§9 / HW VII) ──────────────────────────────────────
#
# Basic Merton: default occurs ONLY at maturity T.
# Therefore P(default between t1 and t2) = 0 for any t2 ≤ T,
# and = pd_T for t1 < T ≤ t2.  This unrealistic feature motivates Black-Cox.

def merton_survival_step(u: float, T: float, pd_T: float) -> float:
    """
    Merton survival at time u given maturity T and default probability pd_T.

    Because default can only occur at T:
        s(u) = 1          for u < T
        s(u) = 1 − pd_T   for u >= T
    """
    if u < 0:
        raise ValueError(f"u must be non-negative (got {u}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    if not (0.0 <= pd_T <= 1.0):
        raise ValueError(f"pd_T must be in [0, 1] (got {pd_T}).")
    return 1.0 if u < T else (1.0 - pd_T)


def merton_interval_default_prob(
    t1: float, t2: float, T: float, pd_T: float
) -> float:
    """
    P(t₁ < τ ≤ t₂) under Merton (default only at maturity T).

    Returns pd_T if the interval [t1, t2] straddles T (i.e. t1 < T ≤ t2),
    otherwise returns 0.  This is the Merton timing defect: default between
    years 3 and 4 is zero when maturity T = 5.
    """
    if t2 < t1:
        raise ValueError(f"t2 must be >= t1 (got {t1}, {t2}).")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T}).")
    if not (0.0 <= pd_T <= 1.0):
        raise ValueError(f"pd_T must be in [0, 1] (got {pd_T}).")
    # Default at T falls in (t1, t2] iff t1 < T <= t2.
    if t1 < T <= t2:
        return float(pd_T)
    return 0.0
