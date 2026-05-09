"""
mitigation.py
Counterparty risk mitigants: netting, collateral, CSA, CCP (§11, HW VIII).

These are pure exposure-transformation functions, independent of CVA pricing.
They are applied BEFORE CVA computation to reduce the effective exposure.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


# ── Netting ───────────────────────────────────────────────────────────────────

def gross_positive_exposure(mtms: Sequence[float]) -> float:
    """Sum max(MTM_j, 0) — no netting."""
    return float(sum(max(v, 0.0) for v in mtms))


def netted_exposure(mtms: Sequence[float]) -> float:
    """max(Σ MTM_j, 0) — legally enforceable netting set."""
    return float(max(sum(mtms), 0.0))


def netting_benefit(mtms: Sequence[float]) -> float:
    """gross_positive_exposure − netted_exposure."""
    return float(gross_positive_exposure(mtms) - netted_exposure(mtms))


def netted_exposure_by_counterparty(
    trade_mtms: Sequence[float],
    counterparty_ids: Sequence[str],
) -> dict[str, float]:
    """
    Aggregate trades by counterparty, apply netting within each group.

    Returns
    -------
    dict mapping counterparty → netted exposure.
    """
    if len(trade_mtms) != len(counterparty_ids):
        raise ValueError(
            f"trade_mtms and counterparty_ids must have the same length "
            f"(got {len(trade_mtms)} vs {len(counterparty_ids)})."
        )
    mtms_arr = np.asarray(trade_mtms, dtype=float)
    if np.any(np.isnan(mtms_arr)):
        raise ValueError("trade_mtms must not contain NaN.")
    groups: dict[str, float] = defaultdict(float)
    for mtm, cid in zip(trade_mtms, counterparty_ids):
        groups[cid] += float(mtm)
    return {cid: max(net, 0.0) for cid, net in groups.items()}


# ── Collateral ────────────────────────────────────────────────────────────────

def simple_collateralized_exposure(exposure: float, collateral: float) -> float:
    """max(exposure − collateral, 0)."""
    return float(max(exposure - collateral, 0.0))


def csa_call_amount(
    exposure: float,
    collateral: float,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> float:
    """
    Variation-margin call amount under a CSA.

    A call is triggered when:
        exposure − collateral − threshold > MTA

    If triggered, call = exposure − collateral − threshold.
    Otherwise, call = 0.
    """
    raw = exposure - collateral - threshold
    if raw > mta:
        return float(raw)
    return 0.0


def csa_residual_exposure_after_margin_call(
    exposure: float,
    collateral: float,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> float:
    """
    Residual exposure after a CSA margin call.

    If call is triggered: collateral is topped up to (exposure − threshold),
    leaving residual = threshold.
    If not triggered: residual = max(exposure − collateral, 0).
    """
    call = csa_call_amount(exposure, collateral, threshold, mta)
    if call > 0:
        return float(threshold)
    return float(max(exposure - collateral, 0.0))


# ── CCP ───────────────────────────────────────────────────────────────────────

def ccp_cleared_exposure(
    mtms: Sequence[float],
    initial_margin: float,
    variation_margin: float,
) -> float:
    """
    Residual exposure after CCP clearing.

        residual = max(netted_exposure(mtms) − initial_margin − variation_margin, 0)
    """
    return float(max(netted_exposure(mtms) - initial_margin - variation_margin, 0.0))


def default_waterfall_loss_allocation(
    loss: float,
    defaulter_margin: float,
    default_fund: float,
    ccp_capital: float,
) -> dict[str, float]:
    """
    Allocate a CCP member default loss through the standard waterfall.

    Waterfall:
        1. Defaulter's margin
        2. Default fund
        3. CCP capital
        4. Unfunded / systemic residual

    Returns
    -------
    dict with keys: covered_by_margin, covered_by_default_fund,
                    covered_by_ccp_capital, unfunded_loss.
    """
    if loss < 0:
        raise ValueError(f"loss must be non-negative (got {loss}).")
    for name, v in [("defaulter_margin", defaulter_margin),
                    ("default_fund", default_fund),
                    ("ccp_capital", ccp_capital)]:
        if v < 0:
            raise ValueError(f"{name} must be non-negative (got {v}).")

    remaining = loss
    covered_margin = min(remaining, defaulter_margin)
    remaining -= covered_margin

    covered_fund = min(remaining, default_fund)
    remaining -= covered_fund

    covered_ccp = min(remaining, ccp_capital)
    remaining -= covered_ccp

    return {
        "covered_by_margin": float(covered_margin),
        "covered_by_default_fund": float(covered_fund),
        "covered_by_ccp_capital": float(covered_ccp),
        "unfunded_loss": float(remaining),
    }


# ── CVA with mitigants wrapper ─────────────────────────────────────────────────

def mitigated_cva(
    mtm_paths: Sequence[Sequence[float]],
    marginal_default_probs: Sequence[float],
    discount_factors: Sequence[float],
    R: float,
    collateral: Sequence[float] | None = None,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> float:
    """
    CVA after applying netting (all trades in one netting set) and collateral.

    Parameters
    ----------
    mtm_paths : list of length n_times, each element a list of trade MTMs
        at that time bucket.
    marginal_default_probs : length n_times.
    discount_factors : length n_times.
    R : recovery.
    collateral : posted collateral at each time bucket (length n_times). None → 0.
    threshold : CSA threshold.
    mta : minimum transfer amount.
    """
    from src.credit.cva import cva_discounted

    n = len(mtm_paths)
    if collateral is None:
        collateral = [0.0] * n
    coll_arr = list(collateral)
    if len(coll_arr) != n:
        raise ValueError("collateral must have same length as mtm_paths.")

    exposures = []
    for t_idx, trade_mtms in enumerate(mtm_paths):
        net = netted_exposure(trade_mtms)
        if threshold == 0.0 and mta == 0.0:
            # Simple collateral netting: max(net - collateral, 0)
            res = simple_collateralized_exposure(net, coll_arr[t_idx])
        else:
            res = csa_residual_exposure_after_margin_call(
                net, coll_arr[t_idx], threshold, mta
            )
        exposures.append(res)

    return cva_discounted(exposures, marginal_default_probs, discount_factors, R)
