"""
monte_carlo.py
Monte Carlo VaR and ES via full portfolio repricing.

Spec §13:
    Simulate  R_sim ~ N(μ_h, Σ_h)
    For each simulation:
        S_sim = S0 × exp(R_sim)
        Reprice portfolio.
        Compute losses.
    VaR and ES computed empirically.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.portfolio.portfolio import reprice_portfolio
from src.risk.estimators import get_mean_cov, manual_mean_cov
from src.risk.returns import compute_log_returns
from src.schemas import Portfolio


def monte_carlo_var_es(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    pricing_date: date,
    lookback_days: int,
    horizon_days: int,
    var_confidence: float,
    es_confidence: float,
    n_simulations: int = 10_000,
    estimator: str = "window",
    ewma_N: int = 60,
    random_seed: int | None = 42,
    calibration_mode: str = "historical",
    manual_market_params: dict | None = None,
    option_vol_shock_mode: str = "fixed",
    option_vol_shock_beta: float = 1.0,
    option_vol_shock_floor: float = 0.05,
) -> dict:
    """Monte Carlo VaR and ES via full portfolio repricing.

    Algorithm:
        1. Estimate daily μ and Σ from trailing lookback_days log returns.
        2. Scale to horizon: μ_h = μ×h, Σ_h = Σ×h.
        3. Draw R_sim ~ N(μ_h, Σ_h) for n_simulations paths.
        4. For each path: S_sim = S0 × exp(R_sim), reprice full portfolio.
        5. loss_i = V0 − V_sim_i.
        6. VaR = quantile(losses, var_confidence).
        7. ES  = mean(losses | loss ≥ quantile(losses, es_confidence)).

    Args:
        portfolio (Portfolio): Stock and option positions.
        prices (pd.DataFrame): Aligned price history (DatetimeIndex × tickers).
        pricing_date (date): Valuation date for option pricing.
        lookback_days (int): Trailing observations for mean/cov estimation.
        horizon_days (int): Risk horizon h in trading days.
        var_confidence (float): VaR tail probability, e.g. 0.99.
        es_confidence (float): ES averaging threshold, e.g. 0.975.
        n_simulations (int): Number of Monte Carlo paths. Default 10 000.
        estimator (str): ``"window"`` or ``"ewma"`` for mean/cov estimation.
        ewma_N (int): EWMA half-life parameter; only used when estimator=``"ewma"``.
        random_seed (int | None): Seed for the numpy RNG for reproducibility.
            Pass None for a random seed.
        calibration_mode (str): ``"historical"`` (estimate from data) or
            ``"manual"`` (supply μ and Σ via manual_market_params).
        manual_market_params (dict | None): Required when calibration_mode=``"manual"``.
            Must contain ``"mu_daily"`` and ``"cov_daily"`` keyed by ticker.
        option_vol_shock_mode (str): ``"fixed"`` or ``"underlying_beta"``.
        option_vol_shock_beta (float): Beta for ``"underlying_beta"`` mode.
        option_vol_shock_floor (float): Minimum shocked vol.

    Returns:
        dict: Result dictionary with keys:
            - ``"var"`` (float): Monte Carlo VaR in dollars.
            - ``"es"`` (float): Monte Carlo ES in dollars.
            - ``"var_confidence"`` (float): var_confidence passed in.
            - ``"es_confidence"`` (float): es_confidence passed in.
            - ``"losses"`` (np.ndarray): Simulated loss distribution (length n_simulations).
            - ``"n_simulations"`` (int): n_simulations used.
            - ``"calibration_mode"`` (str): Mode used.
            - ``"option_vol_shock_mode"`` (str): Vol shock mode used.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # Current spots
    spots_0 = prices.iloc[-1]

    # Identify underlyings with price data
    underlyings = _portfolio_underlyings(portfolio)
    underlyings = [u for u in underlyings if u in prices.columns]

    # Estimate daily mean and covariance
    if calibration_mode == "manual":
        mu_daily, cov_daily = manual_mean_cov(manual_market_params or {}, underlyings)
    else:
        log_ret = compute_log_returns(prices[underlyings])
        mu_daily, cov_daily = get_mean_cov(log_ret, lookback_days, estimator, ewma_N)

    # Horizon scaling
    mu_h = mu_daily.values * horizon_days          # shape (k,)
    cov_h = cov_daily.values * horizon_days        # shape (k, k)

    # Simulate R_sim ~ N(μ_h, Σ_h)
    R_sim = rng.multivariate_normal(mu_h, cov_h, size=n_simulations)  # (n_sim, k)

    # Current portfolio value
    from src.portfolio.portfolio import portfolio_value
    V0 = portfolio_value(portfolio, spots_0, pricing_date)

    # Compute losses
    losses = np.empty(n_simulations)
    spots_0_arr = np.array([float(spots_0[u]) for u in underlyings])

    for i in range(n_simulations):
        shocked_arr = spots_0_arr * np.exp(R_sim[i])
        shocked = pd.Series(shocked_arr, index=underlyings)
        # Merge with full spots_0 (in case portfolio has more tickers)
        full_shocked = spots_0.copy()
        for u, v in shocked.items():
            full_shocked[u] = v
        simulated_returns = pd.Series(R_sim[i], index=underlyings)
        V_sim = reprice_portfolio(
            portfolio,
            full_shocked,
            pricing_date,
            underlying_returns=simulated_returns,
            option_vol_shock_mode=option_vol_shock_mode,
            option_vol_shock_beta=option_vol_shock_beta,
            option_vol_shock_floor=option_vol_shock_floor,
        )
        losses[i] = V0 - V_sim

    var = float(np.quantile(losses, var_confidence))
    es_threshold = float(np.quantile(losses, es_confidence))
    tail_losses = losses[losses >= es_threshold]
    es = float(tail_losses.mean()) if len(tail_losses) > 0 else es_threshold

    return {
        "var": var,
        "es": es,
        "var_confidence": var_confidence,
        "es_confidence": es_confidence,
        "losses": losses,
        "n_simulations": n_simulations,
        "calibration_mode": calibration_mode,
        "option_vol_shock_mode": option_vol_shock_mode,
    }


def _portfolio_underlyings(portfolio: Portfolio) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for pos in portfolio.stocks:
        if pos.ticker not in seen:
            seen.add(pos.ticker)
            result.append(pos.ticker)
    for pos in portfolio.options:
        if pos.underlying_ticker not in seen:
            seen.add(pos.underlying_ticker)
            result.append(pos.underlying_ticker)
    return result
