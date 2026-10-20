"""Build immutable paper-account instructions from the existing daily strategy.

The score and allocation functions remain shared with the research engine.
Broker positions are whole, unadjusted shares; only the signal ATR is converted
from the adjusted research panel into raw T-day price units. No broker calls,
order submission, or backtest-account exports are used here.
"""
from __future__ import annotations

import math
import re
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
import pandas as pd

from ndxbots.config import Settings
from ndxbots.regime.exposure import exposure_by_date, load_regime_table
from ndxbots.strategy.scores import build_score_table
from ndxbots.strategy.targets import EPS, decide_quantity_targets


class PlanError(ValueError):
    """Signal or broker state is insufficient to create an executable plan."""


_TERMINAL_ORDERS = {
    "FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DISABLED", "DELETED",
}
_CODE = re.compile(r"US\.[A-Z]+(?:[.-][A-Z]+)*")


def config_fingerprint(settings: Settings) -> str:
    """Identify strategy settings without tying the plan to a machine path."""
    config = asdict(settings)
    for key in ("futu_host", "futu_port", "data_root", "intraday_root", "kline_end",
                "request_sleep_sec", "max_count"):
        config.pop(key, None)
    content = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _number(value, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise PlanError(f"{name} must be finite") from None
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(number):
        raise PlanError(f"{name} must be finite")
    if positive and number <= 0:
        raise PlanError(f"{name} must be positive")
    return number


def _integer(value, name: str, *, positive: bool = False) -> int:
    number = _number(value, name, positive=positive)
    if not number.is_integer() or abs(number) > 2**53 - 1:
        raise PlanError(f"{name} must be an exact whole-share integer")
    return int(number)


def _day(value, name: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise PlanError(f"{name} must be a calendar date") from None
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise PlanError(f"{name} must be a timezone-free calendar date")
    return day


def _whole_ceiling(value: float) -> int:
    """Round magnitude down without turning floating-point 10-epsilon into 9."""
    magnitude = math.floor(abs(value) + EPS)
    return -magnitude if value < 0 else magnitude


def _frame(frame: pd.DataFrame, name: str, required: set[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise PlanError(f"{name} is missing columns: {sorted(required)}")
    out = frame.copy()
    try:
        dates = pd.to_datetime(out["date"], errors="raise")
        if dates.dt.tz is not None:
            raise ValueError("timezone-aware date")
        out["date"] = dates.dt.normalize()
    except (ValueError, TypeError, AttributeError):
        raise PlanError(f"{name} contains invalid dates") from None
    if out["date"].isna().any() or out.duplicated(["date", "code"]).any():
        raise PlanError(f"{name} must contain unique valid date/code rows")
    if not out["code"].map(lambda code: isinstance(code, str) and bool(_CODE.fullmatch(code))).all():
        raise PlanError(f"{name} contains unsupported US stock codes")
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def _atr(panel: pd.DataFrame, factors: pd.DataFrame, day: pd.Timestamp,
         code: str, raw_close: float) -> float:
    adjusted = panel.loc[panel["date"].eq(day) & panel["code"].eq(code)]
    if adjusted.empty:
        raise PlanError(f"{code}: missing adjusted T-day close for ATR conversion")
    adjusted_close = _number(adjusted.iloc[0]["close"], f"{code} adjusted close", positive=True)
    if "my_atr_pct_20" in factors:
        rows = factors.loc[factors["date"].eq(day) & factors["code"].eq(code)]
        if rows.empty:
            raise PlanError(f"{code}: missing T-day ATR20 factor")
        atr = _number(rows.iloc[0]["my_atr_pct_20"], f"{code} ATR20 percentage", positive=True)
        return _number(atr * raw_close, f"{code} raw ATR20", positive=True)
    if not {"high", "low"}.issubset(panel.columns):
        raise PlanError(f"{code}: ATR20 needs factors or complete daily OHLC")
    history = panel.loc[panel["code"].eq(code) & panel["date"].le(day)].copy()
    if len(history) < 20:
        raise PlanError(f"{code}: ATR20 needs at least 20 complete signal-day bars")
    try:
        for column in ("close", "high", "low"):
            history[column] = pd.to_numeric(history[column], errors="raise")
    except (ValueError, TypeError):
        raise PlanError(f"{code}: invalid OHLC for ATR20") from None
    values = history[["close", "high", "low"]].tail(21).to_numpy(float)
    if (not np.isfinite(values).all() or (values <= 0).any()
            or (values[:, 2] > values[:, 0]).any() or (values[:, 0] > values[:, 1]).any()):
        raise PlanError(f"{code}: invalid OHLC for ATR20")
    previous = history["close"].shift(1)
    ranges = pd.concat([
        history["high"] - history["low"],
        (history["high"] - previous).abs(),
        (history["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    adjusted_atr = _number(ranges.tail(20).mean(), f"{code} ATR20", positive=True)
    return _number(adjusted_atr * raw_close / adjusted_close, f"{code} raw ATR20", positive=True)


def build_plan(
    settings: Settings, *, acc_id: int, signal_date: str, execution_date: str,
    close_minute: int, snapshot: dict, raw_closes: dict[str, float],
    previous_state: dict | None = None, panel=None, factors=None, scored=None,
    regime=None,
) -> dict:
    """Create T+1 entry budgets and whole-share exits for one SIMULATE account.

    The caller verifies the official next-session calendar, the completion of
    signal-day data and the timestamp of the broker snapshot. This pure planner
    rejects stale caches and excludes rows after its explicit signal date. It does
    not infer trading holidays from weekdays. Account state is for a dedicated
    strategy account: every held symbol participates in reconciliation.

    ``previous_state`` is the prior persisted plan (or its effective_gross and
    reduction_targets fields). Outstanding risk ceilings survive non-review
    days exactly as in decide_quantity_targets.
    """
    account = _integer(acc_id, "acc_id", positive=True)
    if not isinstance(snapshot, Mapping) or snapshot.get("env") != "SIMULATE":
        raise PlanError("Paper planning requires a SIMULATE broker snapshot")
    if _integer(snapshot.get("acc_id"), "snapshot acc_id", positive=True) != account:
        raise PlanError("Snapshot account does not match acc_id")
    nav = _number(snapshot.get("nav"), "broker NAV", positive=True)
    cash = _number(snapshot.get("cash"), "broker cash")
    if cash < 0:
        raise PlanError("Negative broker cash is unsupported")
    raw_positions = snapshot.get("positions")
    if not isinstance(raw_positions, Mapping):
        raise PlanError("Broker positions must be a code/quantity mapping")
    positions = {}
    for code, quantity in raw_positions.items():
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise PlanError(f"Unsupported broker position code: {code!r}")
        quantity = _integer(quantity, f"{code} position")
        if quantity:
            positions[code] = quantity
    orders = snapshot.get("orders", [])
    if not isinstance(orders, list):
        raise PlanError("Broker orders must be a list")
    for order in orders:
        if (not isinstance(order, Mapping)
                or str(order.get("status", "")).split(".")[-1] not in _TERMINAL_ORDERS):
            raise PlanError("Resolve all open or uncertain broker orders before creating a new plan")

    day = _day(signal_date, "signal_date")
    execution = _day(execution_date, "execution_date")
    if execution <= day or day.weekday() > 4 or execution.weekday() > 4:
        raise PlanError("Execution must follow the signal on a valid weekday session")
    if isinstance(close_minute, bool) or close_minute not in (780, 960):
        raise PlanError("Calendar close_minute must be 780 or 960")
    fee_bps = _number(settings.cost_bps, "cost_bps")
    if fee_bps < 0:
        raise PlanError("cost_bps must be nonnegative")
    max_hold = _integer(settings.max_hold, "max_hold", positive=True)
    if panel is None:
        panel = pd.read_parquet(settings.data_root / "panel" / "daily.parquet")
    if factors is None:
        factors = pd.read_parquet(settings.data_root / "factors" / "daily.parquet")
    panel = _frame(panel, "Daily panel", {"date", "code", "close"})
    factors = _frame(factors, "Factor table", {"date", "code"})
    panel = panel.loc[panel["date"].le(day)].copy()
    factors = factors.loc[factors["date"].le(day)].copy()
    benchmark_rows = panel.loc[panel["code"].eq(settings.benchmark)]
    if benchmark_rows.empty or benchmark_rows["date"].max() != day:
        raise PlanError("Signal date must equal the latest benchmark daily session")
    if factors.empty or factors["date"].max() != day:
        raise PlanError("Factor cache is stale for the signal date")
    benchmark_close = benchmark_rows.loc[benchmark_rows["date"].eq(day)].iloc[0]["close"]
    _number(benchmark_close, "benchmark T-day close", positive=True)
    if scored is None:
        # All historical rows are essential to selection retention/hysteresis.
        scored = build_score_table(factors, settings)
    scored = _frame(scored, "Score table", {"date", "code", "side", "in_hold"})
    scored = scored.loc[scored["date"].le(day)].copy()
    if scored.empty or scored["date"].max() != day:
        raise PlanError("Score table is stale or empty for the signal date")
    if not scored["in_hold"].map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise PlanError("Score in_hold values must be booleans")
    today = scored.loc[scored["date"].eq(day)].set_index("code")
    selected = today.loc[today["in_hold"]]
    if not selected["side"].isin(["long", "short"]).all():
        raise PlanError("Selected positions require a long or short direction")
    if len(selected) > max_hold:
        raise PlanError("Strategy selection exceeds max_hold")
    if not settings.allow_short and selected["side"].eq("short").any():
        raise PlanError("Short selection conflicts with allow_short=False")
    if settings.benchmark in selected.index:
        raise PlanError("Benchmark must not be included in the strategy selection")

    calendar = pd.DatetimeIndex(benchmark_rows["date"].unique()).sort_values()
    gross = 1.0
    regime_gross = 1.0
    gross_reason = "ladder_disabled"
    if settings.use_gross_ladder:
        table = load_regime_table(settings) if regime is None else regime
        if table is not None:
            table = table.copy()
            table["date"] = pd.to_datetime(table["date"]).dt.normalize()
            table = table.loc[table["date"].le(day)]
        diagnostics = exposure_by_date(
            table, settings, dates=calendar, required_dates=pd.DatetimeIndex([day]),
        )
        row = diagnostics.loc[diagnostics["date"].eq(day)].iloc[0]
        gross = _number(row["gross_target"], "regime gross")
        regime_gross = gross
        gross_reason = str(row["gross_reason"])
    hurst_scale = 1.0
    factors_today = factors.loc[factors["date"].eq(day)].set_index("code")
    if "my_hurst_126" in factors:
        if settings.benchmark in factors_today.index:
            value = factors_today.at[settings.benchmark, "my_hurst_126"]
            try:
                hurst = float(value)
            except (ValueError, TypeError):
                raise PlanError("Benchmark Hurst must be numeric or missing") from None
            if math.isinf(hurst):
                raise PlanError("Benchmark Hurst must be finite or missing")
            hurst_scale = 1.3 if hurst > .75 else .5 if hurst < .55 else 1.0
        gross = float(np.clip(gross * hurst_scale, 0, 1.5))
    if not math.isfinite(gross) or gross < 0:
        raise PlanError("Gross exposure must be nonnegative and finite")
    previous = previous_state or {}
    if not isinstance(previous, Mapping):
        raise PlanError("previous_state must be a mapping")
    if previous.get("env", "SIMULATE") != "SIMULATE":
        raise PlanError("Previous state is not SIMULATE")
    if _integer(previous.get("acc_id", account), "previous acc_id", positive=True) != account:
        raise PlanError("Previous state belongs to another account")
    if "signal_date" in previous and _day(previous["signal_date"], "previous signal_date") >= day:
        raise PlanError("Previous state must predate the new signal")
    effective = _number(previous.get("effective_gross", 1.0), "previous effective_gross")
    if effective < 0:
        raise PlanError("previous effective_gross must be nonnegative")
    prior_ceilings = previous.get("reduction_targets", {})
    if not isinstance(prior_ceilings, Mapping):
        raise PlanError("Previous reduction_targets must be a mapping")
    codes = sorted(set(positions) | set(selected.index))
    code_indices = {code: i for i, code in enumerate(codes)}
    ceilings = {}
    for code, value in prior_ceilings.items():
        ceiling = _number(value, f"{code} reduction target")
        if code in code_indices:
            ceilings[code_indices[code]] = ceiling
    if not isinstance(raw_closes, Mapping):
        raise PlanError("raw_closes must be a mapping")
    prices = np.array([
        _number(raw_closes.get(code), f"{code} raw T close", positive=True) for code in codes
    ], dtype=float)
    quantities = np.array([positions.get(code, 0) for code in codes], dtype=float)
    signs = np.array([
        (1 if selected.at[code, "side"] == "long" else -1) if code in selected.index else 0
        for code in codes
    ], dtype=np.int8)
    decision = decide_quantity_targets(
        day, signs, quantities, prices, nav, gross, effective, ceilings, settings.rebalance_weekdays,
    )
    legs = []
    targets = {}
    for j, code in enumerate(codes):
        current = int(quantities[j])
        target = float(decision.quantities[j])
        if not math.isfinite(target):
            raise PlanError(f"{code}: non-finite strategy target")
        reason = decision.reasons.get(j, "hold_quantity")
        if (target > EPS and current <= 0 and settings.hurst_gate_mode == "strict_entry"
                and "my_hurst_126" in factors):
            value = (factors_today.at[code, "my_hurst_126"]
                     if code in factors_today.index else np.nan)
            try:
                hurst = float(value)
            except (ValueError, TypeError):
                hurst = np.nan
            if not math.isfinite(hurst) or hurst <= .67:
                target = 0.0
                reason = "new_long_hurst_blocked"
        # Reductions preserve a conservative whole-share ceiling. Additions
        # keep their exact signal notional until the executor sizes at T+1.
        same_direction = target * current >= 0
        if same_direction and abs(target) < abs(current):
            target = float(_whole_ceiling(target))
        targets[code] = target
        direction = 0
        if code in today.index:
            direction = {"long": 1, "short": -1}.get(today.at[code, "side"], 0)
        urgent = bool(current and (
            direction == 0 or direction != np.sign(current) or gross < effective - EPS
            or reason == "daily_sentiment_reduction"
        ))
        reverse = target * current < 0
        reduction = (abs(current) if reverse else
                     max(0, abs(current) - int(abs(target))) if abs(target) < abs(current) else 0)
        if reduction:
            legs.append({
                "key": f"{code}:exit", "code": code, "action": "exit",
                "side": 1 if current > 0 else -1, "quantity": int(reduction),
                "notional": 0.0, "prev_close": float(prices[j]), "atr": 0.0,
                "urgent": urgent, "reason": reason,
            })
        addition = abs(target) if reverse else max(0.0, abs(target) - abs(current))
        if addition > EPS:
            legs.append({
                "key": f"{code}:entry", "code": code, "action": "entry",
                "side": 1 if target > 0 else -1, "quantity": 0,
                "notional": float(addition * prices[j]), "prev_close": float(prices[j]),
                "atr": float(_atr(panel, factors, day, code, float(prices[j]))),
                "urgent": False, "reason": reason,
            })
    reductions = {
        codes[j]: float(_whole_ceiling(value))
        for j, value in decision.reduction_targets.items()
        if j < len(codes)
    }
    return {
        "version": 1, "env": "SIMULATE", "acc_id": account,
        "signal_date": day.strftime("%Y-%m-%d"),
        "execution_date": execution.strftime("%Y-%m-%d"),
        "close_minute": int(close_minute), "max_hold": max_hold, "cost_bps": fee_bps,
        "gross": float(gross), "effective_gross": float(decision.effective_gross),
        "reduction_targets": reductions, "positions": positions,
        "signal_nav": nav, "signal_cash": cash, "targets": targets,
        "regime_gross": regime_gross, "hurst_scale": hurst_scale,
        "gross_reason": gross_reason, "strategy_config_hash": config_fingerprint(settings),
        "legs": legs,
    }
