"""Immutable after-close orders in adjusted research price units.

New entries use their signal-time notional as a budget; reductions preserve
requested quantities. This module contains no broker order API.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EPS = 1e-10


@dataclass(frozen=True)
class Order:
    index: int
    target_quantity: float
    reason: str
    signal_date: pd.Timestamp
    signal_nav: float
