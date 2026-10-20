"""Validate durable paper instructions without contacting a broker.

These checks deliberately accept optional strategy/audit metadata. They reject
ambiguous JSON types and inconsistent execution identities before any mutation.
The engine separately verifies its immutable-plan hash and broker state.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import math
import re


_CODE = re.compile(r"US\.[A-Z]+(?:[.-][A-Z]+)*")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _integer(value, name, *, minimum=None):
    # JSON booleans and floating-point values must not masquerade as quantities.
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an exact integer"
                         + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _number(value, name, *, minimum=None, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or (minimum is not None and value < minimum)
            or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite"
                         + (" positive" if positive else "")
                         + " number"
                         + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _code(value, name="code"):
    if not isinstance(value, str) or not _CODE.fullmatch(value):
        raise ValueError(f"{name} must be a supported US stock code")
    return value


def _day(value, name):
    if not isinstance(value, str) or not _DAY.fullmatch(value):
        raise ValueError(f"{name} must be a canonical YYYY-MM-DD date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be a valid calendar date") from None
    if parsed.weekday() > 4:
        raise ValueError(f"{name} must be a weekday session")
    return parsed


def _timestamp(value, name):
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except ValueError:
        raise ValueError(f"{name} must be an aware ISO timestamp") from None


def validate_plan(plan) -> None:
    """Raise ValueError unless a plan is safe to interpret as paper instructions."""
    plan = _mapping(plan, "plan")
    if plan.get("env") != "SIMULATE":
        raise ValueError("Paper plans require the SIMULATE environment")
    _integer(plan.get("acc_id"), "acc_id", minimum=1)
    if "version" in plan and _integer(plan["version"], "plan version") != 1:
        raise ValueError("Unsupported plan version")
    signal = _day(plan.get("signal_date"), "signal_date")
    execution = _day(plan.get("execution_date"), "execution_date")
    if execution <= signal:
        raise ValueError("execution_date must follow signal_date")
    close = _integer(plan.get("close_minute"), "close_minute")
    if close not in (780, 960):
        raise ValueError("close_minute must be 780 or 960")
    max_hold = _integer(plan.get("max_hold"), "max_hold", minimum=1)
    cost = _number(plan.get("cost_bps"), "cost_bps", minimum=0)
    if cost >= 10_000:
        raise ValueError("cost_bps must be below 10000")
    for name in ("nav", "signal_nav"):
        if name in plan:
            _number(plan[name], name, positive=True)
    for name in ("cash", "signal_cash", "gross", "effective_gross",
                 "regime_gross", "hurst_scale"):
        if name in plan:
            _number(plan[name], name, minimum=0)
    positions = _mapping(plan.get("positions"), "positions")
    for code, quantity in positions.items():
        _code(code, "position code")
        _integer(quantity, f"{code} position")
    for name in ("targets", "reduction_targets"):
        if name in plan:
            for code, quantity in _mapping(plan[name], name).items():
                _code(code, f"{name} code")
                _number(quantity, f"{name} {code}")
    legs = plan.get("legs")
    if not isinstance(legs, list):
        raise ValueError("legs must be a list")
    by_key = {}
    entries = {}
    exits = {}
    for leg in legs:
        _mapping(leg, "leg")
        code = _code(leg.get("code"))
        action = leg.get("action")
        if action not in ("entry", "exit"):
            raise ValueError("Leg action must be entry or exit")
        key = leg.get("key")
        if key != f"{code}:{action}" or key in by_key:
            raise ValueError("Leg keys must be unique canonical code:action values")
        by_key[key] = leg
        side = _integer(leg.get("side"), f"{key} side")
        if side not in (-1, 1):
            raise ValueError(f"{key} side must be -1 or 1")
        _number(leg.get("prev_close"), f"{key} prev_close", positive=True)
        _boolean(leg.get("urgent"), f"{key} urgent")
        if action == "entry":
            _number(leg.get("notional"), f"{key} notional", positive=True)
            _number(leg.get("atr"), f"{key} atr", positive=True)
            if "quantity" in leg and _integer(leg["quantity"], f"{key} quantity") != 0:
                raise ValueError("Entry quantity must be zero until broker sizing")
            if leg["urgent"]:
                raise ValueError("Entry cannot use an urgent exit schedule")
            entries[code] = leg
        else:
            quantity = _integer(leg.get("quantity"), f"{key} quantity", minimum=1)
            current = positions.get(code, 0)
            if current * side <= 0 or quantity > abs(current):
                raise ValueError(f"{key} must reduce its existing position without reversing it")
            for name in ("notional", "atr"):
                if name in leg and _number(leg[name], f"{key} {name}") != 0:
                    raise ValueError(f"{key} {name} must be zero")
            exits[code] = leg
    if len(entries) > max_hold:
        raise ValueError("Entry legs exceed max_hold")
    # Existing holdings can exceed max_hold; all may still need to be reduced.
    if len(legs) > len(positions) + max_hold:
        raise ValueError("Too many execution legs for positions and max_hold")
    for code, entry in entries.items():
        current = positions.get(code, 0)
        exit_leg = exits.get(code)
        if current * entry["side"] < 0:
            if exit_leg is None or exit_leg["quantity"] != abs(current):
                raise ValueError(f"{code} reversal needs a full exit before entry")
        elif exit_leg is not None:
            raise ValueError(f"{code} cannot have an exit and entry in the same direction")
        if exit_leg is not None and exit_leg["prev_close"] != entry["prev_close"]:
            raise ValueError(f"{code} reversal legs must share the same previous close")


def validate_state(state) -> None:
    """Check saved structure and order identities; broker statuses stay engine-owned."""
    state = _mapping(state, "state")
    if _integer(state.get("version"), "state version") != 1:
        raise ValueError("Unsupported state version")
    validate_plan(state.get("plan"))
    _text(state.get("plan_id"), "plan_id")
    _text(state.get("status"), "state status")
    if "completed" in state:
        _boolean(state["completed"], "completed")
    if "last_tick" in state:
        _timestamp(state["last_tick"], "last_tick")
    plan_legs = {leg["key"]: leg for leg in state["plan"]["legs"]}
    saved_legs = _mapping(state.get("legs"), "state legs")
    if saved_legs.keys() != plan_legs.keys():
        raise ValueError("Saved leg keys differ from immutable plan")
    for key, values in saved_legs.items():
        _mapping(values, f"{key} state")
        for flag in ("observed_due", "initial_done", "done"):
            _boolean(values.get(flag), f"{key} {flag}")
        if "budget" not in values:
            raise ValueError(f"{key} budget is missing")
        budget = values["budget"]
        if budget is not None:
            _number(budget, f"{key} budget", positive=True)
            if plan_legs[key]["action"] != "entry" or budget > plan_legs[key]["notional"] + .01:
                raise ValueError(f"{key} funded budget exceeds the planned entry")
        for name in ("open", "limit", "sizing_price"):
            if name in values:
                _number(values[name], f"{key} {name}", positive=True)
        if "dust" in values:
            _number(values["dust"], f"{key} dust", minimum=0)
        if "split" in values:
            _boolean(values["split"], f"{key} split")
        if "observed_at" in values:
            _timestamp(values["observed_at"], f"{key} observed_at")
    orders = state.get("orders")
    if not isinstance(orders, list):
        raise ValueError("state orders must be a list")
    remarks, order_ids, exit_fills = set(), set(), {}
    for order in orders:
        _mapping(order, "saved order")
        key = order.get("leg")
        if not isinstance(key, str) or key not in plan_legs:
            raise ValueError("Saved order references an unknown plan leg")
        leg = plan_legs[key]
        direction = leg["side"] if leg["action"] == "entry" else -leg["side"]
        if order.get("code") != leg["code"] or order.get("side") != ("BUY" if direction == 1 else "SELL"):
            raise ValueError("Saved order code/side differs from its plan leg")
        qty = _integer(order.get("qty"), "order qty", minimum=1)
        filled = _integer(order.get("dealt_qty"), "order dealt_qty", minimum=0)
        if filled > qty:
            raise ValueError("Saved fill quantity exceeds requested shares")
        average = _number(order.get("dealt_avg_price"), "order dealt_avg_price", minimum=0)
        if filled and average <= 0:
            raise ValueError("A filled order requires a positive average price")
        if not filled and average != 0:
            raise ValueError("An unfilled order must have zero average price")
        _text(order.get("status"), "order status")
        remark = _text(order.get("remark"), "order remark")
        if remark in remarks:
            raise ValueError("Duplicate saved order remarks")
        remarks.add(remark)
        order_id = order.get("order_id")
        if order_id is not None:
            if not isinstance(order_id, str) or not re.fullmatch(r"[1-9][0-9]*", order_id):
                raise ValueError("order_id must be a positive numeric string")
            if order_id in order_ids:
                raise ValueError("Duplicate saved broker order IDs")
            order_ids.add(order_id)
        kind = order.get("order_type")
        if kind not in ("MARKET", "LIMIT"):
            raise ValueError("Saved order type must be MARKET or LIMIT")
        if kind == "LIMIT":
            _number(order.get("price"), "limit order price", positive=True)
        elif order.get("price") is not None:
            _number(order["price"], "market order price", minimum=0)
        if "attempt" in order:
            _integer(order["attempt"], "order attempt", minimum=0)
        if "cancel_requested" in order:
            _boolean(order["cancel_requested"], "cancel_requested")
        if "created_at" in order:
            _timestamp(order["created_at"], "order created_at")
        if leg["action"] == "exit":
            exit_fills[key] = exit_fills.get(key, 0) + filled
            if qty > leg["quantity"] or exit_fills[key] > leg["quantity"]:
                raise ValueError("Saved exit orders exceed the frozen reduction quantity")
        elif saved_legs[key]["budget"] is None:
            raise ValueError("Saved entry order has no funded budget")
