"""The confirmed first-group T+1 schedule; this module never sends orders.

Five-minute labels are bar END times in America/New_York. Prices, T close,
and T ATR20 must use the same adjusted units. A resting limit crossing inside
a bar is recorded at its end, with its earlier activation time retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    delay: int = 0
    first_fraction: float = 1.0
    gap_threshold: float | None = None
    remainder: str = "immediate"
    exit_mode: str = "immediate"


@dataclass(frozen=True)
class PreparedBars:
    """Validated read-only adjusted quotes for one complete regular session."""
    starts: np.ndarray
    ends: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    vwap: np.ndarray


def prepare_bars(bars: pd.DataFrame | PreparedBars) -> PreparedBars:
    if isinstance(bars, PreparedBars):
        return bars
    required = {"bar_end", "open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        raise ValueError(f"Missing five-minute columns: {sorted(required - set(bars.columns))}")
    frame = bars.copy()
    if frame.empty:
        raise ValueError("A complete regular session is required")
    if pd.api.types.is_numeric_dtype(frame.bar_end):
        values = pd.to_numeric(frame.bar_end).to_numpy(float)
        if not (np.isfinite(values) & (values == np.floor(values))).all():
            raise ValueError("Numeric bar endings must be integer exchange minutes")
        frame["end_minute"] = values.astype(int)
    else:
        stamps = pd.to_datetime(frame.bar_end, errors="raise")
        if stamps.dt.tz is not None:
            stamps = stamps.dt.tz_convert("America/New_York")
        if (stamps.isna().any() or stamps.dt.date.nunique() != 1
                or not stamps.eq(stamps.dt.floor("min")).all()):
            raise ValueError("Bar endings must be whole minutes of one exchange session")
        frame["end_minute"] = stamps.dt.hour * 60 + stamps.dt.minute
    frame = frame.sort_values("end_minute").reset_index(drop=True)
    ends = frame.end_minute.to_numpy(int)
    if (ends[0] != 575 or ends[-1] not in (780, 960)
            or not np.all(np.diff(ends) == 5)):
        raise ValueError("A complete continuous full or half regular session is required")
    prices = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError("OHLC must be finite and positive")
    if not ((frame.low <= frame.open) & (frame.open <= frame.high)
            & (frame.low <= frame.close) & (frame.close <= frame.high)).all():
        raise ValueError("OHLC prices violate bar bounds")

    def frozen(values, dtype=float):
        result = np.array(values, dtype=dtype, copy=True)
        result.setflags(write=False)
        return result

    return PreparedBars(frozen(ends - 5, int), frozen(ends, int),
                        frozen(frame.open), frozen(frame.high), frozen(frame.low),
                        frozen(frame.close), frozen(frame.vwap if "vwap" in frame
                                                     else np.full(len(frame), np.nan)))


def execution_schedule(bars: pd.DataFrame | PreparedBars, side: int, policy: Policy,
                       prev_close: float, atr: float, action: str = "entry",
                       urgent: bool = False) -> list[dict]:
    """Return fractions of the required entry budget or exit quantity.

    ``side`` is the position direction: +1 long, -1 short. ATR is the fixed
    signal-day arithmetic 20-day true-range average, not an intraday indicator.
    Strict trade-through by MORE than one basis point is required for limits;
    a touched limit is insufficient. These are historical bar-model fills.
    """
    if side not in (-1, 1) or action not in ("entry", "exit"):
        raise ValueError("side must be +/-1 and action must be entry/exit")
    if (not isinstance(policy.delay, int) or policy.delay < 0 or policy.delay % 5
            or not math.isfinite(policy.first_fraction) or not 0 < policy.first_fraction <= 1):
        raise ValueError("Delay must be nonnegative whole five minutes and fraction in (0,1]")
    if policy.gap_threshold is not None and (
            not math.isfinite(policy.gap_threshold) or policy.gap_threshold < 0):
        raise ValueError("Gap threshold must be finite and nonnegative")
    frame = prepare_bars(bars)
    last = len(frame.starts) - 1
    requested = 570 + policy.delay
    if requested > frame.starts[last]:
        raise ValueError("Requested execution starts after the session completion deadline")
    start = int(np.searchsorted(frame.starts, requested))
    rows: list[dict] = []

    def fill(index, fraction, trigger, *, forced=False, price=None, at_end=False):
        if fraction <= 1e-12:
            return
        rows.append({"time": int(frame.ends[index] if at_end else frame.starts[index]),
                     "active_from": int(frame.starts[index]), "fraction": float(fraction),
                     "price": float(frame.opens[index] if price is None else price),
                     "forced": bool(forced), "trigger": trigger})

    if action == "exit":
        if policy.exit_mode != "immediate":
            raise ValueError("Only the confirmed immediate exit policy is supported")
        fill(0 if urgent else start, 1.0, "urgent_exit" if urgent else "immediate_exit")
        return rows
    if policy.remainder not in ("atr", "immediate"):
        raise ValueError("Only the confirmed ATR entry and immediate execution are supported")
    fraction = policy.first_fraction
    session_open = float(frame.opens[0])
    if policy.gap_threshold is not None:
        if not math.isfinite(prev_close) or prev_close <= 0:
            raise ValueError("A positive T close is required for adverse-gap conditioning")
        if side * (session_open / prev_close - 1.0) <= policy.gap_threshold + 1e-12:
            fraction = 1.0
    if policy.remainder == "immediate":
        fraction = 1.0
    fill(start, fraction, "initial_entry")
    remaining = 1.0 - fraction
    if remaining > 1e-12:
        if not math.isfinite(atr) or atr <= 0:
            raise ValueError("A positive T-day ATR20 is required for the resting limit")
        limit = session_open - side * .2 * atr
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("The ATR limit must be finite and positive")
        boundary = limit * (1 - side * .0001)
        crossed = (frame.lows[start:last] < boundary if side == 1
                   else frame.highs[start:last] > boundary)
        indices = np.flatnonzero(crossed)
        if len(indices):
            i = start + int(indices[0])
            through_open = frame.opens[i] < boundary if side == 1 else frame.opens[i] > boundary
            fill(i, remaining, "atr_limit", price=frame.opens[i] if through_open else limit,
                 at_end=not bool(through_open))
            remaining = 0.0
        fill(last, remaining, "deadline_entry", forced=True)
    if not math.isclose(sum(row["fraction"] for row in rows), 1.0, rel_tol=0, abs_tol=1e-10):
        raise AssertionError("Execution schedule must complete the full required budget")
    return sorted(rows, key=lambda row: row["time"])


def policy_kind(policy: Policy, action: str = "entry") -> str:
    """All production executions require intraday data, including urgent exits."""
    if action not in ("entry", "exit"):
        raise ValueError("action must be entry/exit")
    return "intraday"
