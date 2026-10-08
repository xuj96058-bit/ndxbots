"""Fixed-quantity research fills with cash reconciliation and two reversal legs.

Short proceeds are available as cash. This is not a short-collateral,
borrow-availability, financing-interest or broker-lot model.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

EPS = 1e-10


@dataclass(frozen=True)
class Order:
    index: int
    target_quantity: float
    reason: str
    signal_date: pd.Timestamp
    signal_nav: float


def execute_orders(
    cash: float,
    quantities: np.ndarray,
    orders: Sequence[Order],
    quotes: np.ndarray,
    cost_bps: float,
    codes: Sequence[str],
    max_hold: int,
) -> tuple[float, np.ndarray, list[dict], list[dict]]:
    """Close/trim first; credit short openings; then pro-rate cash-limited buys.

Only additions are cash-limited: risk-reducing covers must be executed.
If covers require borrowing, the caller raises rather than silently hide
the financing requirement. A missing close cannot permit a reversal open.
"""
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Cost must be finite and nonnegative")
    quantities = np.asarray(quantities, dtype=float).copy()
    reductions, additions, blocked, fills = [], [], [], []
    for order in orders:
        j, target = order.index, float(order.target_quantity)
        current = quantities[j]
        if not np.isfinite(target):
            raise ValueError("Requested quantity must be finite")
        if np.sign(current) != np.sign(target) and abs(current) > EPS and abs(target) > EPS:
            reductions.append((order, -current, "close_before_reverse"))
            additions.append((order, target, "open_after_reverse"))
        else:
            delta = target - current
            if abs(delta) <= EPS:
                continue
            row = (order, delta, "rebalance")
            (reductions if abs(target) < abs(current) or abs(target) <= EPS else additions).append(row)
    reductions.sort(key=lambda row: codes[row[0].index])
    additions.sort(key=lambda row: (row[1] > 0, codes[row[0].index]))
    cash_scale = None
    for is_addition, rows in ((False, reductions), (True, additions)):
        for order, delta, leg in rows:
            j = order.index
            quote = quotes[j]
            if not np.isfinite(quote) or quote <= 0:
                blocked.append({"index": j, "event": "missing_execution_quote", "leg": leg, "reason": order.reason})
                continue
            if leg == "open_after_reverse" and abs(quantities[j]) > EPS:
                blocked.append({"index": j, "event": "reverse_close_not_filled", "leg": leg, "reason": order.reason})
                continue
            if abs(quantities[j]) <= EPS and np.count_nonzero(np.abs(quantities) > EPS) >= max_hold:
                blocked.append({"index": j, "event": "slot_cap_after_blocked_exit", "leg": leg, "reason": order.reason})
                continue
            requested = float(delta)
            if is_addition and delta > 0:
                if cash_scale is None:
                    required = sum(
                        item[1] * quotes[item[0].index] * (1 + cost_bps / 10_000.0)
                        for item in additions if item[1] > 0
                        and np.isfinite(quotes[item[0].index]) and quotes[item[0].index] > 0
                    )
                    cash_scale = min(1.0, max(cash, 0.0) / required) if required > 0 else 1.0
                delta *= cash_scale
                if delta <= EPS:
                    blocked.append({"index": j, "event": "insufficient_cash_for_addition", "leg": leg, "reason": order.reason})
                    continue
            notional = float(delta * quote)
            cost = abs(notional) * cost_bps / 10_000.0
            cash -= notional + cost
            if abs(cash) < 1e-8:
                cash = 0.0
            quantities[j] += delta
            if abs(quantities[j]) <= EPS:
                quantities[j] = 0.0
            fills.append({"index": j, "quantity": float(delta), "requested_quantity": requested,
                          "price": float(quote), "signed_notional": notional, "cost": cost,
                          "cash_constrained_partial": abs(delta - requested) > EPS,
                          "leg": leg, "reason": order.reason,
                          "signal_date": order.signal_date, "signal_nav": float(order.signal_nav)})
            if np.count_nonzero(np.abs(quantities) > EPS) > max_hold:
                raise AssertionError("Filled holdings exceed configured position count")
    return float(cash), quantities, fills, blocked
