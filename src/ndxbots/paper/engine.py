"""One reconciled paper execution step, with write-ahead order intentions.

Actual broker cumulative fills are authoritative. A timeout is NEVER evidence
of rejection. An intent that cannot be located blocks further automatic writes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json
import math
from zoneinfo import ZoneInfo

from ndxbots.order import OrderError, OrderRequest
from .store import PaperError, StateStore
from .validation import validate_plan, validate_state

ET = ZoneInfo("America/New_York")
TERMINAL = {"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DISABLED", "DELETED"}
REJECTED = {"FAILED", "DISABLED", "DELETED"}
ACTIVE = {"UNSUBMITTED", "WAITING_SUBMIT", "SUBMITTING", "SUBMITTED", "FILLED_PART",
          "CANCELLING_PART", "CANCELLING_ALL"}


def aware_et(now):
    if now.tzinfo is None or now.utcoffset() is None:
        raise PaperError("Execution clock must include a timezone")
    return now.astimezone(ET)


def plan_id(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def install_plan(store: StateStore, plan: dict):
    try:
        validate_plan(plan)
    except ValueError as exc:
        raise PaperError(str(exc)) from exc
    ident = plan_id(plan)
    with store.locked():
        old = store.load()
        if old:
            validate_state(old)
        if old and old["plan_id"] == ident:
            return old
        if old:
            if old["status"] != "complete" or any(o["status"] not in TERMINAL for o in old["orders"]):
                raise PaperError("Previous plan is unresolved; inspect status before preparing another")
            if plan["signal_date"] <= old["plan"]["signal_date"]:
                raise PaperError("Cannot replace or replay an already prepared signal date")
            store.archive(old)
        state = {"version": 1, "plan_id": ident, "plan": plan, "status": "prepared",
                 "orders": [], "legs": {leg["key"]: {"budget": None, "observed_due": False,
                                                    "initial_done": False, "done": False}
                                       for leg in plan["legs"]},
                 "last_message": "", "last_tick": None}
        store.save(state)
        return state


class PaperEngine:
    def __init__(self, store: StateStore, broker, *, quote_max_age=90, late_grace=120, clock=None):
        self.store, self.broker = store, broker
        self.quote_max_age, self.late_grace = quote_max_age, late_grace
        self.clock = clock

    def _save(self):
        if self.submit:
            self.store.save(self.state)

    def _message(self, message, status="waiting"):
        self.state["status"] = status
        if status == "complete":
            self.state["completed"] = True
        self.state["last_message"] = message
        self._save()
        return self.state

    def _orders(self, key):
        return [o for o in self.state["orders"] if o["leg"] == key]

    def _totals(self, key):
        orders = self._orders(key)
        return (sum(o["dealt_qty"] for o in orders),
                sum(o["dealt_qty"] * o["dealt_avg_price"] for o in orders))

    def _reconcile(self, snapshot, *, stopping=False):
        plan = self.state["plan"]
        if snapshot.get("env") != "SIMULATE" or snapshot.get("acc_id") != plan["acc_id"]:
            raise PaperError("Broker account/environment differs from the immutable plan")
        observed = snapshot["orders"]
        for intent in self.state["orders"]:
            matches = [o for o in observed if
                       (intent.get("order_id") and o["order_id"] == intent["order_id"])
                       or o.get("remark") == intent["remark"]]
            if len(matches) > 1:
                raise PaperError("Duplicate broker orders match one intent; manual reconciliation required")
            if not matches:
                if intent["status"] in TERMINAL:
                    continue  # OpenD only retains recent closed orders.
                raise PaperError(f"Unresolved order {intent['remark']}: do not resubmit or delete state")
            order = matches[0]
            if any(order[k] != intent[k] for k in ("code", "side", "qty")):
                raise PaperError("Broker order identity/quantity differs from saved intent")
            if order.get("remark") != intent["remark"]:
                raise PaperError("Broker order remark differs from saved intent")
            filled = order["dealt_qty"]
            if (not isinstance(filled, int) or filled < intent["dealt_qty"] or filled > intent["qty"]
                    or (filled and (not math.isfinite(order["dealt_avg_price"])
                                    or order["dealt_avg_price"] <= 0))):
                raise PaperError("Invalid or regressing cumulative broker fills")
            status = order["status"]
            if status not in TERMINAL | ACTIVE:
                raise PaperError(f"Unknown broker order status: {status}")
            if intent["status"] in TERMINAL and status not in TERMINAL:
                raise PaperError("Broker terminal order reverted to a working state")
            if intent["status"] in TERMINAL and (
                    filled != intent["dealt_qty"]
                    or not math.isclose(order["dealt_avg_price"], intent["dealt_avg_price"], abs_tol=1e-8)):
                raise PaperError("Previously final broker fills were revised; reconcile manually")
            if status == "FILLED_ALL" and filled != intent["qty"]:
                raise PaperError("FILLED_ALL does not match the requested quantity")
            intent.update(order_id=order["order_id"], status=status,
                          dealt_qty=filled, dealt_avg_price=float(order["dealt_avg_price"]))
        self._save()  # Persist fills even if a later consistency check blocks.
        known_ids = {o.get("order_id") for o in self.state["orders"]}
        if not stopping and any(o["status"] not in TERMINAL and o["order_id"] not in known_ids for o in observed):
            raise PaperError("Unmanaged working orders exist in this dedicated paper account")
        expected = dict(plan["positions"])
        for order in self.state["orders"]:
            expected[order["code"]] = expected.get(order["code"], 0) + (
                order["dealt_qty"] * (1 if order["side"] == "BUY" else -1))
        expected = {k: v for k, v in expected.items() if v}
        actual = {k: v for k, v in snapshot["positions"].items() if v}
        if not stopping and expected != actual:
            raise PaperError("Positions do not yet match cumulative fills (or were changed externally)")

    def _quote(self, code, quotes, now, leg):
        q = quotes[code]
        if q.get("suspended") or q.get("lot_size") != 1:
            raise PaperError(f"{code}: suspended or unsupported stock lot size")
        stamp = aware_et(datetime.fromisoformat(q["updated_at"]))
        age = (now - stamp).total_seconds()
        if stamp.date() != now.date() or age < -5 or age > self.quote_max_age:
            raise PaperError(f"{code}: stale or future quote")
        if not all(math.isfinite(q[k]) and q[k] > 0 for k in ("price", "open", "prev_close")):
            raise PaperError(f"{code}: invalid quote prices")
        tick = .0001 if min(q["prev_close"], leg["prev_close"]) < 1 else .01
        if not math.isclose(q["prev_close"], leg["prev_close"], rel_tol=1e-4, abs_tol=tick):
            raise PaperError(f"{code}: previous close changed; check split/corporate-action units")
        self.quote_stamps[code] = stamp
        return q

    def _send(self, leg, phase, qty, kind, price, now):
        if self.store.stop_path.exists():
            raise PaperError("STOP requested before submission; no new order sent")
        if self.clock is not None:
            current = aware_et(self.clock())
            minute = current.hour * 60 + current.minute + current.second / 60
            if current.date() != now.date() or minute >= self.state["plan"]["close_minute"]:
                raise PaperError("Session ended during broker reads; no order sent")
            if phase in {"initial", "atr_limit"} and minute >= self.state["plan"]["close_minute"] - 5:
                return self._message("Deadline reached during broker reads; re-evaluate next tick")
            if (current - now).total_seconds() < 0 or (current - now).total_seconds() > self.quote_max_age:
                raise PaperError("Broker reads took too long; refresh before submitting")
            stamp = self.quote_stamps.get(leg["code"])
            if stamp is None or (current - stamp).total_seconds() > self.quote_max_age:
                raise PaperError("Quote became stale during broker reads; no order sent")
        count = len(self._orders(leg["key"]))
        remark = f"ndxp-{self.state['plan_id']}-{len(self.state['orders']):04d}"
        side = ("BUY" if leg["side"] == 1 else "SELL") if leg["action"] == "entry" else (
            "SELL" if leg["side"] == 1 else "BUY")
        request = OrderRequest(leg["code"], side, qty, kind, price, remark)
        if not self.submit:
            return self._message(f"PREVIEW {phase}: {side} {qty} {leg['code']} {kind} {price}", "preview")
        intent = {"leg": leg["key"], "phase": phase, "attempt": count, "remark": remark,
                  "code": leg["code"], "side": side, "qty": qty, "order_type": kind,
                  "price": price, "dealt_qty": 0, "dealt_avg_price": 0.0,
                  "status": "INTENT", "order_id": None, "created_at": now.isoformat(),
                  "cancel_requested": False}
        if leg["action"] == "entry":
            intent["sizing_price"] = self.state["legs"][leg["key"]].get("sizing_price")
        self.state["orders"].append(intent)
        self._save()  # Must commit before the first byte of the order is sent.
        try:
            intent["order_id"] = str(self.broker.place(request))
            if not intent["order_id"] or intent["order_id"] == "None":
                raise PaperError("Broker did not return an order ID")
            intent["status"] = "SUBMITTED"
            self._save()
        except Exception as exc:
            return self._message(f"Submission uncertain: {exc}; reconcile saved intent before continuing", "blocked")
        return self._message(f"Submitted {phase} {remark}", "working")

    def _cancel(self, order):
        if not self.submit:
            return self._message(f"PREVIEW cancel {order['order_id']}", "preview")
        if order.get("cancel_requested"):
            return self._message("Waiting for final broker cancellation/fill confirmation", "working")
        order["cancel_requested"] = True
        self._save()
        try:
            self.broker.cancel(order["order_id"])
        except Exception as exc:
            return self._message(f"Cancellation uncertain: {exc}; awaiting broker confirmation", "blocked")
        return self._message("Cancellation requested; no replacement until terminal confirmation", "working")

    def tick(self, now: datetime, *, submit=False):
        if not isinstance(submit, bool):
            raise PaperError("submit must be a boolean")
        with self.store.locked():
            self.state = deepcopy(self.store.load())
            if not self.state:
                raise PaperError("Prepare a paper plan first")
            self.submit = submit
            self.quote_stamps = {}
            now = aware_et(now)
            self.state["last_tick"] = now.isoformat()
            try:
                validate_state(self.state)
                return self._tick(now)
            except (PaperError, OrderError, ValueError, KeyError, TypeError) as exc:
                return self._message(str(exc), "blocked")

    def _tick(self, now):
        plan = self.state["plan"]
        if plan["env"] != "SIMULATE":
            raise PaperError("Paper engine refuses non-SIMULATE plans")
        if plan_id(plan) != self.state["plan_id"]:
            raise PaperError("Saved immutable plan was modified; manual reconciliation required")
        snapshot = self.broker.snapshot()
        self._reconcile(snapshot, stopping=self.store.stop_path.exists())
        active = [o for o in self.state["orders"] if o["status"] not in TERMINAL]
        if self.store.stop_path.exists():
            if active:
                return self._cancel(active[0])
            return self._message("Stopped by local STOP file; holdings preserved", "stopped")
        if (self.state.get("completed") or self.state["status"] == "complete") and not active:
            return self._message("Previously completed plan remains complete", "complete")
        if now.date().isoformat() < plan["execution_date"]:
            return self._message("Waiting for the next official session")
        if now.date().isoformat() > plan["execution_date"]:
            if active:
                return self._cancel(active[0])
            self._refresh_done_without_quotes()
            if all(s["done"] for s in self.state["legs"].values()):
                return self._message("Expired session reconciled with no remaining orders", "complete")
            raise PaperError("Execution date expired; never replay yesterday's orders")
        minute = now.hour * 60 + now.minute + now.second / 60
        close = plan["close_minute"]
        if minute < 570:
            return self._message("Waiting for regular market open")
        if minute >= close:
            # Re-evaluate completion from final fills before reporting expiration.
            if active:
                return self._cancel(active[0])
            self._refresh_done_without_quotes()
            if all(s["done"] for s in self.state["legs"].values()):
                return self._message("All paper legs reconciled", "complete")
            raise PaperError("Market closed with incomplete execution; review before next plan")
        # Record that the worker observed a due time even while waiting for a
        # pending exit or temporarily unavailable quote. Deliberate deferral
        # must not look like a worker first started hours late.
        for leg in plan["legs"]:
            ls = self.state["legs"][leg["key"]]
            due = (570 if leg["urgent"] else 585) if leg["action"] == "exit" else 600
            if minute >= due and not ls["observed_due"]:
                if (minute - due) * 60 > self.late_grace:
                    raise PaperError(f"Missed {due//60:02d}:{due%60:02d} start for {leg['key']}; no silent catch-up")
                ls["observed_due"] = True
                ls["observed_at"] = now.isoformat()
        self._save()
        if any(o["status"] in REJECTED for o in self.state["orders"]):
            raise PaperError("Broker rejected an order; automatic retries are disabled")
        # At the deadline, reconcile/cancel all resting limits before replacement.
        if minute >= close - 5:
            limits = [o for o in active if o["order_type"] == "LIMIT"]
            if limits:
                return self._cancel(limits[0])
        if any(o["order_type"] == "MARKET" for o in active):
            return self._message("Waiting for market-order final fills", "working")
        codes = sorted({leg["code"] for leg in plan["legs"]})
        if not codes:
            return self._message("No strategy changes", "complete")
        quotes = {}
        def quote_for(leg):
            if leg["code"] not in quotes:
                quotes.update(self.broker.quote([leg["code"]]))
            return self._quote(leg["code"], quotes, now, leg)
        fee = plan["cost_bps"] / 10000
        # Risk reducing sells precede covers, then short entries and cash buys.
        legs = sorted(plan["legs"], key=lambda l: (
            0 if l["action"] == "exit" and l["side"] == 1 else
            1 if l["action"] == "exit" else 2 if l["side"] == -1 else 3, l["code"]))
        for leg in legs:
            if leg["action"] != "exit":
                continue
            ls = self.state["legs"][leg["key"]]
            filled, _ = self._totals(leg["key"])
            remaining = leg["quantity"] - filled
            ls["done"] = remaining == 0
            if remaining < 0:
                raise PaperError("Exit filled more than its frozen quantity")
            if not remaining or not ls["observed_due"]:
                continue
            if any(o["status"] not in TERMINAL for o in self._orders(leg["key"])):
                continue
            q = quote_for(leg)
            side = "SELL" if leg["side"] == 1 else "BUY"
            cap = self.broker.max_quantity(leg["code"], side, q["price"], "MARKET", action="exit")
            quantity = min(remaining, cap, abs(snapshot["positions"].get(leg["code"], 0)))
            if side == "BUY":
                free = max(0, snapshot["cash"] - self._reserved_free_cash(legs, fee))
                quantity = min(quantity, math.floor(free / (q["price"] * (1 + fee))))
            if quantity > 0:
                return self._send(leg, "exit", quantity, "MARKET", None, now)
            ls["blocked_reason"] = "Broker capacity/cash cannot complete the reduction"
        if minute < 600:
            return self._message("Exits reconciled; entries begin at 10:00")
        # No entry can take a slot from an unfilled exit or reverse a non-flat position.
        reserved_names = {l["code"] for l in legs if l["action"] == "entry"
                          and self.state["legs"][l["key"]]["budget"] is not None
                          and not self.state["legs"][l["key"]]["done"]}
        occupied = {k for k, v in snapshot["positions"].items() if v} | reserved_names
        eligible = []
        for leg in legs:
            if leg["action"] != "entry":
                continue
            ls = self.state["legs"][leg["key"]]
            if ls["done"]:
                continue
            position = snapshot["positions"].get(leg["code"], 0)
            if position * leg["side"] < 0:
                ls["blocked_reason"] = "Waiting for reversal exit"
                continue
            if any(l["action"] == "exit" and l["code"] == leg["code"]
                   and not self.state["legs"][l["key"]]["done"] for l in legs):
                continue
            if leg["code"] not in occupied and len(occupied) >= plan["max_hold"]:
                ls["blocked_reason"] = "Waiting for a confirmed vacant holding slot"
                continue
            occupied.add(leg["code"])
            eligible.append(leg)
            if "open" not in ls:
                q = quote_for(leg)
                ls["open"] = q["open"]
                ls["split"] = leg["side"] * (q["open"] / leg["prev_close"] - 1) > .01 + 1e-12
                raw = q["open"] - leg["side"] * .2 * leg["atr"]
                tick = Decimal(".0001") if raw < 1 else Decimal(".01")
                ls["limit"] = float(Decimal(str(raw)).quantize(
                    tick, rounding=ROUND_FLOOR if leg["side"] == 1 else ROUND_CEILING))
                if ls["limit"] <= 0:
                    raise PaperError("ATR limit is not positive")
        for leg in eligible:
            ls = self.state["legs"][leg["key"]]
            if leg["side"] == -1 and ls["budget"] is None:
                ls["budget"] = leg["notional"]
        self._save()
        longs_funded = False
        for leg in eligible:
            # The simulator credits short fills and covers before reserving
            # cash for simultaneous longs. Never reserve anticipated proceeds.
            if leg["side"] == 1 and not longs_funded:
                available = max(0, snapshot["cash"] - self._reserved_free_cash(legs, fee)) / (1 + fee)
                unreserved = [l for l in eligible if l["side"] == 1
                              and self.state["legs"][l["key"]]["budget"] is None]
                needed = sum(l["notional"] for l in unreserved)
                if available > 0 and needed:
                    for buy in unreserved:
                        self.state["legs"][buy["key"]]["budget"] = buy["notional"] * min(1, available / needed)
                longs_funded = True
                self._save()
            ls = self.state["legs"][leg["key"]]
            if ls["budget"] is None:
                continue
            if any(o["status"] not in TERMINAL for o in self._orders(leg["key"])):
                continue
            q = quote_for(leg)
            _, spent = self._totals(leg["key"])
            if spent > ls["budget"] + .01:
                raise PaperError("Market slippage exceeded funded budget; no further automatic entries")
            remaining = max(0, ls["budget"] - spent)
            if remaining < q["price"] and (not ls["split"] or minute >= close - 5 or ls["initial_done"]):
                # Before deadline a split limit may still afford a share below market.
                if not ls["split"] or minute >= close - 5 or remaining < ls["limit"]:
                    ls["done"], ls["dust"] = True, remaining
                    continue
            side = "BUY" if leg["side"] == 1 else "SELL"
            if minute >= close - 5:
                kind, price, amount, phase = "MARKET", q["price"], remaining, "deadline"
            elif not ls["initial_done"]:
                ceiling = ls["budget"] * (.5 if ls["split"] else 1)
                amount = max(0, ceiling - spent)
                if amount < q["price"]:
                    ls["initial_done"] = True
                    self._save()
                    if not ls["split"]:
                        ls["done"], ls["dust"] = True, remaining
                        continue
                    kind, price, amount, phase = "LIMIT", ls["limit"], remaining, "atr_limit"
                else:
                    kind, price, phase = "MARKET", q["price"], "initial"
            elif ls["split"]:
                kind, price, amount, phase = "LIMIT", ls["limit"], remaining, "atr_limit"
            else:
                kind, price, amount, phase = "MARKET", q["price"], remaining, "initial"
            quantity = math.floor((amount + 1e-9) / price)
            if quantity <= 0:
                ls["dust"] = remaining
                continue
            cap = self.broker.max_quantity(leg["code"], side, price, kind, action="entry")
            quantity = min(quantity, cap)
            if quantity <= 0:
                ls["blocked_reason"] = "Broker capacity is zero (cash, short availability or margin)"
                continue
            ls["done"] = False
            ls["sizing_price"] = price
            return self._send(leg, phase, quantity, kind, price if kind == "LIMIT" else None, now)
        self._save()
        if all(ls["done"] for ls in self.state["legs"].values()) and not active:
            return self._message("All paper legs reconciled; whole-share residuals recorded", "complete")
        return self._message("Waiting for due fills, cash or broker capacity", "working")

    def _refresh_done_without_quotes(self):
        for leg in self.state["plan"]["legs"]:
            ls = self.state["legs"][leg["key"]]
            filled, spent = self._totals(leg["key"])
            if leg["action"] == "exit":
                ls["done"] = filled == leg["quantity"]
            elif ls["budget"] is not None and abs(spent - ls["budget"]) < .01:
                ls["done"] = True
            elif ls["budget"] is not None:
                orders = self._orders(leg["key"])
                if orders and all(o["status"] in TERMINAL for o in orders):
                    last = orders[-1]
                    remainder = ls["budget"] - spent
                    # Use the quantity decision actually made while the market
                    # was open, not an invented post-close quote.
                    if (last["phase"] in {"deadline", "initial", "atr_limit"} and last["status"] == "FILLED_ALL"
                            and last.get("sizing_price") and 0 <= remainder < last["sizing_price"]
                            and (last["phase"] != "initial" or not ls.get("split"))):
                        ls["done"], ls["dust"] = True, remainder

    def _reserved_free_cash(self, legs, fee):
        reserved = 0.0
        for leg in legs:
            ls = self.state["legs"][leg["key"]]
            if leg["action"] != "entry" or leg["side"] != 1 or ls["budget"] is None or ls["done"]:
                continue
            _, spent = self._totals(leg["key"])
            # Working limits are already frozen by the broker; reserve only
            # the part that is not yet committed, avoiding double subtraction.
            committed = sum((o["qty"] - o["dealt_qty"]) * o["price"]
                            for o in self._orders(leg["key"])
                            if o["status"] not in TERMINAL and o["order_type"] == "LIMIT")
            reserved += max(0, ls["budget"] - spent - committed) * (1 + fee)
        return reserved
