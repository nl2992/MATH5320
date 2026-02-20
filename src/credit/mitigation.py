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
    """Gross positive exposure (GPE) without any netting.

    GPE = Σ_j max(MTM_j, 0)

    Args:
        mtms (Sequence[float]): Mark-to-market values of individual trades
            (positive = in-the-money to us).

    Returns:
        float: Sum of positive MTMs.
    """
    return float(sum(max(v, 0.0) for v in mtms))


def netted_exposure(mtms: Sequence[float]) -> float:
    """Exposure after legally enforceable netting within a single netting set.

    Netted exposure = max(Σ_j MTM_j, 0)

    Args:
        mtms (Sequence[float]): MTM values of all trades in the netting set.

    Returns:
        float: Net positive exposure; 0 when the netting set is net
            in-the-money to the counterparty.
    """
    return float(max(sum(mtms), 0.0))


def netting_benefit(mtms: Sequence[float]) -> float:
    """Exposure reduction attributable to netting.

    Netting benefit = gross_positive_exposure − netted_exposure

    Args:
        mtms (Sequence[float]): MTM values of all trades in the netting set.

    Returns:
        float: Dollar benefit of netting ≥ 0.
    """
    return float(gross_positive_exposure(mtms) - netted_exposure(mtms))


def netted_exposure_by_counterparty(
    trade_mtms: Sequence[float],
    counterparty_ids: Sequence[str],
) -> dict[str, float]:
    """Aggregate trades by counterparty and apply netting within each group.

    Args:
        trade_mtms (Sequence[float]): MTM values for each trade.  Must not
            contain NaN.  Length must match ``counterparty_ids``.
        counterparty_ids (Sequence[str]): Counterparty identifier for each
            trade (same index as ``trade_mtms``).

    Returns:
        dict[str, float]: Mapping ``{counterparty_id: netted_exposure}``
            where each value = max(Σ MTMs for that counterparty, 0).

    Raises:
        ValueError: If ``trade_mtms`` and ``counterparty_ids`` have different
            lengths, or ``trade_mtms`` contains NaN.
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
    """Residual exposure after subtracting posted collateral.

    Args:
        exposure (float): Current net positive exposure before collateral.
        collateral (float): Dollar value of collateral already posted.

    Returns:
        float: max(exposure − collateral, 0) — residual uncovered exposure.
    """
    return float(max(exposure - collateral, 0.0))


def csa_call_amount(
    exposure: float,
    collateral: float,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> float:
    """Compute the variation-margin call amount under a Credit Support Annex (CSA).

    A margin call is triggered when:
        exposure − collateral − threshold > MTA

    If triggered: call = exposure − collateral − threshold.
    If not triggered: call = 0.

    Args:
        exposure (float): Current net positive exposure.
        collateral (float): Collateral already posted by the counterparty.
        threshold (float): CSA threshold below which no margin call is made
            (default 0.0 = fully collateralised).
        mta (float): Minimum Transfer Amount — calls below this size are
            not triggered (default 0.0).

    Returns:
        float: Dollar margin call amount (0 if not triggered).
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
    """Compute residual exposure after a CSA margin call is (or is not) made.

    If the call is triggered: collateral is topped up to exposure − threshold,
    leaving residual = threshold.
    If not triggered: residual = max(exposure − collateral, 0).

    Args:
        exposure (float): Current net positive exposure.
        collateral (float): Collateral already posted.
        threshold (float): CSA threshold (default 0.0).
        mta (float): Minimum Transfer Amount (default 0.0).

    Returns:
        float: Residual uncollateralised exposure after the margin-call
            mechanism is applied.
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
    """Residual exposure after CCP clearing.

    residual = max(netted_exposure(mtms) − initial_margin − variation_margin, 0)

    Args:
        mtms (Sequence[float]): MTM values for all cleared trades.
        initial_margin (float): IM posted at the CCP to cover potential
            future exposure.
        variation_margin (float): VM collected daily to cover current MtM
            exposure.

    Returns:
        float: Residual exposure not covered by CCP margins ≥ 0.
    """
    return float(max(netted_exposure(mtms) - initial_margin - variation_margin, 0.0))


def default_waterfall_loss_allocation(
    loss: float,
    defaulter_margin: float,
    default_fund: float,
    ccp_capital: float,
) -> dict[str, float]:
    """Allocate a CCP member default loss through the standard loss waterfall.

    Waterfall order:
        1. Defaulter's initial margin (first absorber).
        2. Mutualized default fund contributions.
        3. CCP equity / "skin in the game" capital.
        4. Unfunded residual (systemic loss).

    Args:
        loss (float): Total default loss to be allocated (must be ≥ 0).
        defaulter_margin (float): Defaulting member's initial margin
            available to cover losses (must be ≥ 0).
        default_fund (float): Mutualized default fund available (must be ≥ 0).
        ccp_capital (float): CCP's own capital committed to the waterfall
            (must be ≥ 0).

    Returns:
        dict[str, float]: Loss allocation with keys:
            - ``"covered_by_margin"`` (float)
            - ``"covered_by_default_fund"`` (float)
            - ``"covered_by_ccp_capital"`` (float)
            - ``"unfunded_loss"`` (float): residual not covered by any layer.

    Raises:
        ValueError: If ``loss``, ``defaulter_margin``, ``default_fund``, or
            ``ccp_capital`` are negative.
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
    """CVA after applying netting and CSA collateral (§11, HW VIII).

    All trades are assumed to belong to a single legally enforceable netting
    set.  Netting is applied first; then CSA collateral reduces the net
    exposure at each time bucket.

    Args:
        mtm_paths (Sequence[Sequence[float]]): Outer index = time bucket,
            inner = MTM values of individual trades at that time.
            Length n_times.
        marginal_default_probs (Sequence[float]): Per-interval marginal PD
            (length n_times; must sum to ≤ 1).
        discount_factors (Sequence[float]): Risk-free discount factors
            D(t_i) ∈ (0, 1] (length n_times).
        R (float): Recovery rate ∈ [0, 1].
        collateral (Sequence[float] | None): Collateral posted at each time
            bucket (length n_times).  Pass None for zero collateral.
        threshold (float): CSA threshold (default 0 = full collateralisation).
        mta (float): Minimum Transfer Amount (default 0).

    Returns:
        float: Mitigated CVA in the same currency as the MTM values.
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
