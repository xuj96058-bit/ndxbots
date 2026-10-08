"""Complete position targets shared by research execution and pool exports.

Quantities are fixed from the signal close and NAV. They use the same units
as the input price series; QFQ-adjusted inputs are not broker share counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-10


@dataclass(frozen=True)
class TargetDecision:
    quantities: np.ndarray
    effective_gross: float
    reduction_targets: dict[int, float]
    reasons: dict[int, str]


def equal_target_weights(signs: np.ndarray, gross: float) -> np.ndarray:
    """Signed equal weights over the complete selected set, not TopN watch rows."""
    signs = np.asarray(signs)
    if not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError("Selected directions must be -1, 0 or 1")
    if not np.isfinite(gross) or gross < 0:
        raise ValueError("Gross target must be finite and nonnegative")
    count = int(np.count_nonzero(signs))
    return signs.astype(float) * float(gross) / count if count else np.zeros(signs.shape)


def decide_quantity_targets(
    day: pd.Timestamp,
    signs: np.ndarray,
    quantities: np.ndarray,
    prices: np.ndarray,
    nav: float,
    gross: float,
    effective_gross: float,
    reduction_targets: Mapping[int, float] | None = None,
    review_weekdays: Sequence[int] = (1, 4),
) -> TargetDecision:
    """Weekly normal sizing; daily exits and proportional sentiment cuts.

Failed risk reductions retain their signed ceilings until reconsidered.
Ordinary openings or increases wait for the next scheduled review.
Missing signal quotes preserve current quantities, never use future prices.
"""
    signs = np.asarray(signs)
    quantities = np.asarray(quantities, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if signs.shape != quantities.shape or prices.shape != quantities.shape:
        raise ValueError("Selected directions, quantities and prices must align")
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("Signal NAV must be positive and finite")
    weights = equal_target_weights(signs, gross)
    ceilings = dict(reduction_targets or {})
    target = quantities.copy()
    reasons: dict[int, str] = {}
    normal_review = pd.Timestamp(day).weekday() in review_weekdays
    if normal_review:
        target[:] = 0.0
        for j in np.flatnonzero(signs):
            if np.isfinite(prices[j]) and prices[j] > 0:
                target[j] = weights[j] * nav / prices[j]
            else:
                target[j] = quantities[j]
        ceilings = {}
        effective_gross = float(gross)
        reasons = {j: "ordinary_review" for j in range(len(target))}
    else:
        for j, ceiling in ceilings.items():
            if ((abs(ceiling) <= EPS or np.sign(ceiling) == np.sign(quantities[j]))
                    and abs(ceiling) < abs(target[j])):
                target[j] = ceiling
                reasons[j] = "retry_risk_reduction"
        for j in np.flatnonzero(np.abs(quantities) > EPS):
            if signs[j] == 0 or signs[j] != np.sign(quantities[j]):
                target[j] = 0.0
                reasons[j] = "mandatory_selection_exit"
        if gross < effective_gross - EPS:
            scale = float(gross / effective_gross) if effective_gross > 0 else 0.0
            target *= scale
            effective_gross = float(gross)
            for j in np.flatnonzero(np.abs(quantities) > EPS):
                reasons.setdefault(j, "daily_sentiment_reduction")
        for j in np.flatnonzero(np.abs(target) < np.abs(quantities) - EPS):
            ceilings[int(j)] = float(target[j])
        if (np.abs(target) > np.abs(quantities) + EPS).any():
            raise AssertionError("Unscheduled position increase")
    return TargetDecision(target, float(effective_gross), ceilings, reasons)
