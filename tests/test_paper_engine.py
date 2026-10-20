"""Offline broker scenarios for write-ahead paper execution and recovery."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from ndxbots.paper.engine import PaperEngine, aware_et, install_plan
from ndxbots.paper.store import PaperError, StateStore


ET = ZoneInfo("America/New_York")
FINAL = {"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "DISABLED", "DELETED"}


def entry(code="US.A", *, side=1, budget=1_000., prev=100., atr=10.):
    return {"key": code + ":entry", "code": code, "action": "entry", "side": side,
            "quantity": 0, "notional": budget, "prev_close": prev, "atr": atr,
            "urgent": False, "reason": "ordinary_review"}


def exit_leg(code="US.A", *, side=1, qty=10, urgent=False):
    return {"key": code + ":exit", "code": code, "action": "exit", "side": side,
            "quantity": qty, "notional": 0., "prev_close": 100., "atr": 0.,
            "urgent": urgent, "reason": "ordinary_review"}


def plan(legs, *, positions=None, close=960, max_hold=2):
    return {"version": 1, "env": "SIMULATE", "acc_id": 123,
            "signal_date": "2024-01-05", "execution_date": "2024-01-08",
            "close_minute": close, "max_hold": max_hold, "cost_bps": 0.,
            "gross": 1., "effective_gross": 1., "reduction_targets": {},
            "positions": positions or {}, "legs": legs}


class FakeBroker:
    """The broker owns fills; the runner can never pretend an ACK is a fill."""
    def __init__(self, *, cash=1_000., positions=None, immediate_market=True):
        self.acc_id, self.env = 123, "SIMULATE"
        self.cash = cash
        self.positions = dict(positions or {})
        self.immediate_market = immediate_market
        self.rows = []
        self.requests = []
        self.cancel_calls = []
        self.now = datetime(2024, 1, 8, 10, tzinfo=ET)
        self.quotes = {}
        self.caps = {}
        self.fail_place = None
        self.place_hook = None
        self.quantity_hook = None
        self.fail_cancel = False
        self.quote_calls = 0

    def q(self, code, **changes):
        row = self.quotes.setdefault(code, {
            "price": 100., "open": 100., "prev_close": 100.,
            "lot_size": 1, "suspended": False,
        })
        row.update(changes)
        return row

    def snapshot(self):
        reserved = sum((row["qty"] - row["dealt_qty"]) * row["price"]
                       for row in self.rows if row["status"] not in FINAL
                       and row["side"] == "BUY" and row["order_type"] == "LIMIT")
        return {"acc_id": self.acc_id, "env": self.env,
                "cash": max(0., self.cash - reserved), "nav": 1_000.,
                "positions": {code: qty for code, qty in self.positions.items() if qty},
                "sellable": {code: max(0, qty) for code, qty in self.positions.items()},
                "orders": deepcopy(self.rows)}

    def quote(self, codes):
        self.quote_calls += 1
        return {code: {**self.q(code),
                       "updated_at": self.q(code).get("updated_at", self.now.isoformat())}
                for code in codes}

    def place(self, request):
        self.requests.append(request)
        if self.place_hook:
            self.place_hook()
        if self.fail_place == "before":
            raise TimeoutError("No broker acknowledgement")
        row = {
            "order_id": str(len(self.rows) + 1), "remark": request.remark,
            "code": request.code, "side": request.side, "qty": request.qty,
            "dealt_qty": 0, "dealt_avg_price": 0., "status": "SUBMITTED",
            "order_type": request.order_type, "price": request.price,
        }
        self.rows.append(row)
        if request.order_type == "MARKET" and self.immediate_market:
            self.fill(row["order_id"], request.qty, self.q(request.code)["price"])
        if self.fail_place == "after":
            raise TimeoutError("Order accepted but reply lost")
        return row["order_id"]

    def cancel(self, order_id):
        self.cancel_calls.append(order_id)
        row = self.order(order_id)
        row["status"] = "CANCELLING_PART" if row["dealt_qty"] else "CANCELLING_ALL"
        if self.fail_cancel:
            raise TimeoutError("Cancel reply lost")

    def confirm_cancel(self, order_id):
        row = self.order(order_id)
        if row["dealt_qty"] < row["qty"]:
            row["status"] = "CANCELLED_PART" if row["dealt_qty"] else "CANCELLED_ALL"

    def order(self, order_id):
        return next(row for row in self.rows if row["order_id"] == str(order_id))

    def fill(self, order_id, qty, price):
        row = self.order(order_id)
        assert 0 < qty <= row["qty"] - row["dealt_qty"]
        old_value = row["dealt_qty"] * row["dealt_avg_price"]
        row["dealt_qty"] += qty
        row["dealt_avg_price"] = (old_value + qty * price) / row["dealt_qty"]
        direction = 1 if row["side"] == "BUY" else -1
        self.positions[row["code"]] = self.positions.get(row["code"], 0) + direction * qty
        self.cash -= direction * qty * price
        row["status"] = "FILLED_ALL" if row["dealt_qty"] == row["qty"] else "FILLED_PART"

    def max_quantity(self, code, side, price, kind, *, action):
        if self.quantity_hook:
            self.quantity_hook()
        if (code, action) in self.caps:
            return self.caps[code, action]
        if action == "exit":
            return abs(self.positions.get(code, 0))
        return math.floor(self.snapshot()["cash"] / price) if side == "BUY" else 100_000


class PaperEngineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix=".paper-engine-", dir=Path(__file__).parent)
        self.store = StateStore(Path(self.folder.name) / "account.sqlite3")
        self.broker = FakeBroker()
        self.engine = PaperEngine(self.store, self.broker)

    def tearDown(self):
        gc.collect()
        self.folder.cleanup()

    def install(self, legs=None, **kwargs):
        doc = plan([entry()] if legs is None else legs, **kwargs)
        self.broker.positions = dict(doc["positions"])
        install_plan(self.store, doc)
        return doc

    def tick(self, when="10:00:00", *, date="2024-01-08", submit=True):
        now = datetime.fromisoformat(f"{date}T{when}").replace(tzinfo=ET)
        self.broker.now = now
        return self.engine.tick(now, submit=submit)

    def restart(self):
        self.store = StateStore(self.store.path)
        self.engine = PaperEngine(self.store, self.broker)

    def test_exact_one_percent_adverse_gap_uses_full_budget_for_both_sides(self):
        for side, opening in ((1, 101.), (-1, 99.)):
            with self.subTest(side=side):
                # Distinct state files simulate independent strategy accounts.
                self.store = StateStore(Path(self.folder.name) / f"exact-{side}.sqlite3")
                self.broker = FakeBroker()
                self.engine = PaperEngine(self.store, self.broker)
                self.broker.q("US.A", price=opening, open=opening)
                self.install([entry(side=side)])
                self.tick()
                self.assertEqual(len(self.broker.requests), 1)
                self.assertEqual(self.broker.requests[0].qty, math.floor(1_000. / opening))
                self.assertFalse(self.store.load()["legs"]["US.A:entry"]["split"])

    def test_adverse_gap_splits_notional_then_activates_atr_limit_after_ten(self):
        self.broker.q("US.A", price=102., open=102.)
        self.install()
        self.tick("09:59:59")
        self.assertEqual(self.broker.requests, [])
        self.tick()
        first = self.broker.requests[0]
        self.assertEqual((first.order_type, first.qty), ("MARKET", 4))
        self.tick("10:00:05")
        second = self.broker.requests[1]
        self.assertEqual((second.order_type, second.price, second.qty), ("LIMIT", 100., 5))
        state = self.store.load()
        self.assertEqual(state["orders"][0]["dealt_qty"], 4)
        self.assertEqual(state["legs"]["US.A:entry"]["budget"], 1_000.)

    def test_short_adverse_gap_has_symmetric_sell_limit(self):
        self.broker.q("US.A", price=98., open=98.)
        self.install([entry(side=-1)])
        self.tick()
        self.assertEqual((self.broker.requests[0].side, self.broker.requests[0].qty), ("SELL", 5))
        self.tick("10:00:05")
        rest = self.broker.requests[1]
        self.assertEqual((rest.side, rest.order_type, rest.price, rest.qty), ("SELL", "LIMIT", 100., 5))

    def test_dry_run_never_places_or_persists_an_order(self):
        self.install()
        before = self.store.load()
        result = self.tick(submit=False)
        self.assertEqual(result["status"], "preview")
        self.assertEqual(self.broker.requests, [])
        self.assertEqual(self.store.load(), before)

    def test_real_plan_or_broker_account_is_rejected(self):
        doc = plan([entry()])
        doc["env"] = "REAL"
        with self.assertRaisesRegex(PaperError, "SIMULATE"):
            install_plan(self.store, doc)
        self.install()
        self.broker.env = "REAL"
        self.assertEqual(self.tick()["status"], "blocked")
        self.assertEqual(self.broker.requests, [])

    def test_account_mismatch_blocks_orders(self):
        self.install()
        self.broker.acc_id = 456
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("account/environment", result["last_message"])

    def test_preopen_future_session_and_expired_plan_cannot_place(self):
        self.install()
        self.assertEqual(self.tick("09:29:59")["status"], "waiting")
        self.assertEqual(self.tick("10:00:00", date="2024-01-06")["status"], "waiting")
        self.assertEqual(self.tick("10:00:00", date="2024-01-09")["status"], "blocked")
        self.assertEqual(self.broker.requests, [])

    def test_next_morning_cancels_surviving_own_order_without_replaying_entry(self):
        self.broker.immediate_market = False
        self.install()
        self.tick()
        self.restart()
        result = self.tick("09:00:00", date="2024-01-09")
        self.assertEqual(result["status"], "working")
        self.assertEqual(self.broker.cancel_calls, ["1"])
        self.assertEqual(len(self.broker.requests), 1)
        self.broker.fill("1", 2, 100.)
        self.broker.confirm_cancel("1")
        result = self.tick("10:00:00", date="2024-01-09")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Execution date expired", result["last_message"])
        self.assertEqual(self.broker.positions["US.A"], 2)
        self.assertEqual(len(self.broker.requests), 1)

    def test_stop_created_during_capacity_query_prevents_new_intent_and_order(self):
        self.install()
        self.broker.quantity_hook = self.store.stop_path.touch
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("STOP requested before submission", result["last_message"])
        self.assertEqual(self.broker.requests, [])
        self.assertEqual(self.store.load()["orders"], [])

    def test_close_reached_during_broker_reads_prevents_deadline_market(self):
        self.install()
        self.broker.caps["US.A", "entry"] = 0
        self.tick()
        self.broker.caps["US.A", "entry"] = 10
        self.engine = PaperEngine(self.store, self.broker,
                                  clock=lambda: datetime(2024, 1, 8, 16, tzinfo=ET))
        result = self.tick("15:59:59")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Session ended during broker reads", result["last_message"])
        self.assertEqual(self.broker.requests, [])
        self.assertEqual(self.store.load()["orders"], [])

    def test_deadline_reached_during_reads_replaces_proposed_limit_on_next_tick(self):
        self.broker.q("US.A", price=102., open=102.)
        self.install()
        self.tick()
        self.engine = PaperEngine(self.store, self.broker,
                                  clock=lambda: datetime(2024, 1, 8, 15, 55, tzinfo=ET))
        result = self.tick("15:54:59")
        self.assertEqual(result["status"], "waiting")
        self.assertIn("Deadline reached during broker reads", result["last_message"])
        self.assertEqual(len(self.broker.requests), 1)
        self.assertEqual(len(self.store.load()["orders"]), 1)
        self.tick("15:55:00")
        self.assertEqual(len(self.broker.requests), 2)
        self.assertEqual((self.broker.requests[1].order_type, self.broker.requests[1].qty),
                         ("MARKET", 5))

    def test_reads_exceeding_ninety_seconds_prevent_initial_market(self):
        self.install()
        self.engine = PaperEngine(self.store, self.broker,
                                  clock=lambda: datetime(2024, 1, 8, 10, 1, 31, tzinfo=ET))
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Broker reads took too long", result["last_message"])
        self.assertEqual(self.broker.requests, [])
        self.assertEqual(self.store.load()["orders"], [])

    def test_quote_aging_past_ninety_seconds_during_reads_prevents_order(self):
        self.install()
        self.broker.q("US.A", updated_at="2024-01-08T09:59:00-05:00")
        self.engine = PaperEngine(self.store, self.broker,
                                  clock=lambda: datetime(2024, 1, 8, 10, 0, 31, tzinfo=ET))
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.broker.requests, [])
        self.assertEqual(self.store.load()["orders"], [])

    def test_ordinary_and_urgent_exits_use_distinct_times(self):
        self.install([exit_leg("US.A", urgent=True), exit_leg("US.B", urgent=False)],
                     positions={"US.A": 10, "US.B": 10})
        self.tick("09:30:00")
        self.assertEqual([r.code for r in self.broker.requests], ["US.A"])
        self.tick("09:44:59")
        self.assertEqual(len(self.broker.requests), 1)
        self.tick("09:45:00")
        self.assertEqual([r.code for r in self.broker.requests], ["US.A", "US.B"])

    def test_partial_limit_cancel_race_reconciles_extra_fill_before_halfday_replacement(self):
        self.broker.q("US.A", price=102., open=102.)
        self.install(close=780)
        self.tick()
        self.tick("10:00:05")
        limit_id = self.broker.rows[1]["order_id"]
        self.broker.fill(limit_id, 2, 100.)
        self.tick("12:54:59")
        self.assertEqual(self.broker.cancel_calls, [])
        result = self.tick("12:55:00")
        self.assertEqual(result["status"], "working")
        self.assertEqual(self.broker.cancel_calls, [limit_id])
        self.tick("12:55:01")
        self.assertEqual(len(self.broker.requests), 2)
        self.assertEqual(self.broker.cancel_calls, [limit_id])
        # One additional fill races the cancellation acknowledgement.
        self.broker.fill(limit_id, 1, 100.)
        self.broker.confirm_cancel(limit_id)
        self.tick("12:55:02")
        replacement = self.broker.requests[-1]
        self.assertEqual((replacement.order_type, replacement.qty), ("MARKET", 2))
        self.assertEqual(len(self.broker.requests), 3)
        result = self.tick("12:55:03")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.broker.positions["US.A"], 9)
        self.assertAlmostEqual(result["legs"]["US.A:entry"]["dust"], 88.)

    def test_regular_deadline_is_1555_and_not_halfday_time(self):
        self.broker.q("US.A", price=102., open=102.)
        self.install()
        self.tick()
        self.tick("10:00:05")
        self.tick("12:55:00")
        self.assertEqual(self.broker.cancel_calls, [])
        self.tick("15:55:00")
        self.assertEqual(len(self.broker.cancel_calls), 1)

    def test_lost_place_acknowledgement_is_found_by_remark_without_duplicate(self):
        self.broker.fail_place = "after"
        self.install()
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.store.load()["orders"][0]["status"], "INTENT")
        self.restart()
        self.broker.fail_place = None
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.broker.requests), 1)
        self.assertEqual(self.store.load()["orders"][0]["dealt_qty"], 10)

    def test_unlocated_uncertain_intent_blocks_retry_after_restart(self):
        self.broker.fail_place = "before"
        self.install()
        self.tick()
        self.restart()
        self.broker.fail_place = None
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Unresolved order", result["last_message"])
        self.assertEqual(len(self.broker.requests), 1)

    def test_fill_reconciliation_after_restart_is_cumulative_not_additive(self):
        self.broker.immediate_market = False
        self.install()
        self.tick()
        self.broker.fill("1", 4, 100.)
        self.tick("10:00:05")
        self.restart()
        self.tick("10:00:10")
        self.assertEqual(self.store.load()["orders"][0]["dealt_qty"], 4)
        self.assertEqual(len(self.broker.requests), 1)
        self.broker.fill("1", 6, 100.)
        self.assertEqual(self.tick("10:00:15")["status"], "complete")
        self.assertEqual(self.broker.positions["US.A"], 10)
        self.assertEqual(self.store.load()["orders"][0]["dealt_qty"], 10)

    def test_second_worker_cannot_place_while_first_holds_account_lock(self):
        self.install()
        second = PaperEngine(StateStore(self.store.path), self.broker)
        failures = []

        def competing_tick():
            try:
                second.tick(self.broker.now, submit=True)
            except PaperError as exc:
                failures.append(str(exc))

        self.broker.place_hook = competing_tick
        self.tick()
        self.assertEqual(len(failures), 1)
        self.assertIn("Another paper worker", failures[0])
        self.assertEqual(len(self.broker.requests), 1)

    def test_manual_position_change_or_unmanaged_order_blocks(self):
        self.install()
        self.broker.positions["US.B"] = 1
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Positions", result["last_message"])
        self.broker.positions.clear()
        self.broker.rows.append({
            "order_id": "999", "remark": "manual", "code": "US.B", "side": "BUY",
            "qty": 1, "dealt_qty": 0, "dealt_avg_price": 0., "status": "SUBMITTED",
            "order_type": "MARKET", "price": None,
        })
        result = self.tick("10:00:01")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Unmanaged", result["last_message"])
        self.assertEqual(self.broker.requests, [])

    def test_reversal_waits_until_cover_is_fully_filled(self):
        self.broker.immediate_market = False
        self.install([exit_leg(side=-1, urgent=True), entry()],
                     positions={"US.A": -10})
        self.tick("09:30:00")
        self.tick()
        self.assertEqual(len(self.broker.requests), 1)
        self.broker.fill("1", 5, 100.)
        self.tick("10:00:05")
        self.assertEqual(len(self.broker.requests), 1)
        self.broker.fill("1", 5, 100.)
        self.broker.cash = 1_000.
        self.tick("10:00:10")
        self.assertEqual(len(self.broker.requests), 2)
        self.assertEqual(self.broker.requests[1].side, "BUY")

    def test_observed_entry_waiting_for_exit_is_not_mistaken_for_late_restart(self):
        self.broker.immediate_market = False
        self.install([exit_leg("US.A", urgent=True), entry("US.B")],
                     positions={"US.A": 10})
        self.tick("09:30:00")
        self.tick("10:00:00")
        self.assertEqual(len(self.broker.requests), 1)
        self.broker.fill("1", 10, 100.)
        result = self.tick("10:03:00")
        self.assertEqual(result["status"], "working")
        self.assertEqual(len(self.broker.requests), 2)
        self.assertEqual(self.broker.requests[1].code, "US.B")

    def test_stale_future_entry_quote_does_not_block_due_urgent_exit(self):
        self.install([exit_leg("US.A", urgent=True), entry("US.B")],
                     positions={"US.A": 10})
        self.broker.q("US.B", updated_at="2024-01-05T16:00:00-05:00")
        result = self.tick("09:30:00")
        self.assertEqual(result["status"], "working")
        self.assertEqual(len(self.broker.requests), 1)
        self.assertEqual(self.broker.requests[0].code, "US.A")

    def test_no_new_name_takes_an_occupied_slot(self):
        self.install([entry("US.C")], positions={"US.A": 1, "US.B": 1})
        result = self.tick()
        self.assertEqual(self.broker.requests, [])
        self.assertIn("vacant holding slot", result["legs"]["US.C:entry"]["blocked_reason"])

    def test_simultaneous_long_budgets_are_allocated_proportionally_and_persist(self):
        self.install([entry("US.A", budget=500.), entry("US.B", budget=1_500.)])
        self.tick()
        state = self.store.load()
        self.assertEqual(state["legs"]["US.A:entry"]["budget"], 250.)
        self.assertEqual(state["legs"]["US.B:entry"]["budget"], 750.)
        self.assertEqual((self.broker.requests[0].code, self.broker.requests[0].qty), ("US.A", 2))
        self.restart()
        self.tick("10:00:05")
        self.assertEqual((self.broker.requests[1].code, self.broker.requests[1].qty), ("US.B", 7))
        self.assertEqual(self.tick("10:00:10")["status"], "complete")
        self.assertEqual(self.broker.positions, {"US.A": 2, "US.B": 7})

    def test_short_proceeds_are_available_before_long_budget_is_frozen(self):
        self.broker.cash = 500.
        self.install([entry("US.A", side=-1, budget=500.),
                      entry("US.B", budget=1_000.)])
        self.tick()
        first = self.broker.requests[0]
        self.assertEqual((first.code, first.side, first.qty), ("US.A", "SELL", 5))
        self.assertIsNone(self.store.load()["legs"]["US.B:entry"]["budget"])
        self.restart()
        self.tick("10:00:05")
        second = self.broker.requests[1]
        self.assertEqual((second.code, second.side, second.qty), ("US.B", "BUY", 10))
        self.assertEqual(self.store.load()["legs"]["US.B:entry"]["budget"], 1_000.)

    def test_cover_cannot_spend_uncommitted_long_split_reserve(self):
        self.broker.q("US.A", price=102., open=102.)
        self.broker.caps["US.B", "exit"] = 0
        self.install([exit_leg("US.B", side=-1, qty=5, urgent=True), entry("US.A")],
                     positions={"US.B": -5})
        self.tick("09:30:00")
        self.tick()
        self.assertEqual((self.broker.requests[0].code, self.broker.requests[0].qty),
                         ("US.A", 4))
        self.assertEqual(self.broker.cash, 592.)
        # The broker has not frozen a limit yet, but the remaining $592 belongs
        # to A's split entry. A newly available cover must not spend that budget.
        self.broker.caps["US.B", "exit"] = 5
        self.tick("10:00:05")
        self.assertEqual(len(self.broker.requests), 2)
        second = self.broker.requests[1]
        self.assertEqual((second.code, second.order_type, second.qty),
                         ("US.A", "LIMIT", 5))
        self.assertEqual(self.broker.positions["US.B"], -5)
        self.tick("10:00:10")
        self.assertEqual(len(self.broker.requests), 2)

    def test_final_deadline_fill_observed_at_close_completes_with_share_dust(self):
        self.broker.q("US.A", price=102., open=102.)
        self.install()
        self.tick()
        self.tick("10:00:05")
        self.tick("15:55:00")
        self.broker.confirm_cancel("2")
        self.broker.immediate_market = False
        self.tick("15:55:01")
        replacement = self.broker.requests[-1]
        self.assertEqual((replacement.order_type, replacement.qty), ("MARKET", 5))
        self.broker.fill("3", 5, 102.)
        self.restart()
        quotes_before_close = self.broker.quote_calls
        result = self.tick("16:00:00")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["legs"]["US.A:entry"]["dust"], 82.)
        self.assertEqual(self.broker.quote_calls, quotes_before_close)
        self.assertEqual(len(self.broker.requests), 3)
        self.assertEqual(self.broker.positions["US.A"], 9)

    def test_final_atr_limit_fill_observed_at_close_completes_with_share_dust(self):
        self.broker.q("US.A", price=102., open=102.)
        self.install()
        self.tick()
        self.tick("10:00:05")
        self.broker.fill("2", 5, 100.)
        self.restart()
        result = self.tick("16:00:00")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["legs"]["US.A:entry"]["dust"], 92.)
        self.assertEqual(len(self.broker.requests), 2)
        self.assertEqual(self.broker.positions["US.A"], 9)

    def test_terminal_fill_correction_is_detected_even_after_plan_completed(self):
        self.install()
        self.tick()
        self.assertEqual(self.tick("10:00:05")["status"], "complete")
        self.broker.rows[0]["dealt_avg_price"] = 99.
        self.restart()
        result = self.tick("10:00:10")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Previously final broker fills were revised", result["last_message"])
        self.assertEqual(self.store.load()["orders"][0]["dealt_avg_price"], 100.)
        self.assertEqual(len(self.broker.requests), 1)

    def test_stop_and_resume_after_completion_never_reopens_share_dust(self):
        self.broker.q("US.A", price=110., open=100.)
        self.install()
        self.tick()
        self.assertEqual(self.tick("10:00:05")["status"], "complete")
        self.store.stop_path.touch()
        self.assertEqual(self.tick("10:00:10")["status"], "stopped")
        self.store.stop_path.unlink()
        self.broker.q("US.A", price=5.)
        self.restart()
        self.assertEqual(self.tick("10:00:15")["status"], "complete")
        self.assertEqual(len(self.broker.requests), 1)
        self.assertEqual(self.broker.positions["US.A"], 9)

    def test_stale_quote_and_split_reference_block_entry(self):
        self.install()
        self.broker.q("US.A", updated_at="2024-01-08T09:58:00-05:00")
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("stale", result["last_message"])
        self.broker.quotes.clear()
        self.broker.q("US.A", prev_close=25., open=25., price=25.)
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("split/corporate", result["last_message"])
        self.assertEqual(self.broker.requests, [])

    def test_low_price_previous_close_change_blocks_entry(self):
        self.install([entry(prev=.50, atr=.02)])
        self.broker.q("US.A", prev_close=.505, open=.505, price=.505)
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("split/corporate", result["last_message"])
        self.assertEqual(self.broker.requests, [])
        self.assertEqual(self.store.load()["orders"], [])

    def test_late_initial_start_is_blocked_but_exact_grace_boundary_is_allowed(self):
        self.install()
        result = self.tick("10:02:01")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Missed 10:00", result["last_message"])
        self.assertEqual(self.broker.requests, [])
        # No intention was created; a fresh eligible clock tests the exact boundary.
        self.assertEqual(self.tick("10:02:00")["status"], "working")
        self.assertEqual(len(self.broker.requests), 1)

    def test_stop_file_cancels_pending_order_without_liquidation(self):
        self.broker.immediate_market = False
        self.install()
        self.tick()
        self.store.stop_path.touch()
        self.tick("10:00:05")
        self.assertEqual(self.broker.cancel_calls, ["1"])
        self.broker.fill("1", 2, 100.)
        self.broker.confirm_cancel("1")
        result = self.tick("10:00:10")
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(self.broker.positions["US.A"], 2)
        self.assertEqual(len(self.broker.requests), 1)

    def test_completed_plan_is_latched_when_price_drop_makes_dust_buyable(self):
        self.broker.q("US.A", price=110., open=100.)
        self.install()
        self.tick()
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.broker.positions["US.A"], 9)
        self.broker.q("US.A", price=5.)
        self.restart()
        result = self.tick("10:01:00")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.broker.requests), 1)

    def test_completed_entry_is_latched_while_another_entry_waits_for_capacity(self):
        self.broker.cash = 2_000.
        self.broker.q("US.A", price=110., open=100.)
        self.broker.caps["US.B", "entry"] = 0
        self.install([entry("US.A"), entry("US.B")])
        self.tick()
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "working")
        self.assertTrue(result["legs"]["US.A:entry"]["done"])
        self.assertEqual(result["legs"]["US.A:entry"]["dust"], 10.)
        self.assertFalse(result["legs"]["US.B:entry"]["done"])
        self.broker.q("US.A", price=5.)
        self.restart()
        self.tick("10:00:10")
        self.assertEqual(len(self.broker.requests), 1)
        self.assertEqual(self.broker.positions["US.A"], 9)

    def test_terminal_order_disappearing_from_recent_history_does_not_repeat_it(self):
        self.install()
        self.tick()
        self.tick("10:00:05")
        self.broker.rows.clear()
        self.restart()
        self.assertEqual(self.tick("10:00:10")["status"], "complete")
        self.assertEqual(len(self.broker.requests), 1)

    def test_matching_duplicate_broker_remarks_block_recovery(self):
        self.broker.fail_place = "after"
        self.install()
        self.tick()
        duplicate = deepcopy(self.broker.rows[0])
        duplicate["order_id"] = "2"
        self.broker.rows.append(duplicate)
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Duplicate broker orders", result["last_message"])
        self.assertEqual(len(self.broker.requests), 1)

    def test_rejected_order_never_automatically_retries(self):
        self.broker.immediate_market = False
        self.install()
        self.tick()
        self.broker.rows[0]["status"] = "FAILED"
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("rejected", result["last_message"])
        self.assertEqual(len(self.broker.requests), 1)

    def test_zero_short_capacity_is_explicit_and_never_sends_unavailable_short(self):
        self.install([entry(side=-1)])
        self.broker.caps["US.A", "entry"] = 0
        result = self.tick()
        self.assertEqual(self.broker.requests, [])
        self.assertIn("Broker capacity is zero", result["legs"]["US.A:entry"]["blocked_reason"])
        result = self.tick("16:00:00")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("incomplete execution", result["last_message"])

    def test_market_slippage_over_budget_blocks_further_entries(self):
        self.broker.immediate_market = False
        self.install()
        self.tick()
        self.broker.fill("1", 10, 101.)
        result = self.tick("10:00:05")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("slippage exceeded funded budget", result["last_message"])
        self.assertEqual(len(self.broker.requests), 1)

    def test_stop_still_cancels_own_working_order_after_external_position_change(self):
        self.broker.immediate_market = False
        self.install()
        self.tick()
        self.broker.positions["US.B"] = 1
        self.store.stop_path.touch()
        self.tick("10:00:05")
        self.assertEqual(self.broker.cancel_calls, ["1"])
        self.assertEqual(len(self.broker.requests), 1)

    def test_immutable_plan_modification_is_detected_before_broker_write(self):
        self.install()
        state = self.store.load()
        state["plan"]["legs"][0]["notional"] = 100_000.
        self.store.save(state)
        result = self.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertIn("immutable plan was modified", result["last_message"])
        self.assertEqual(self.broker.requests, [])

    def test_exchange_clock_tracks_dst_without_fixed_utc_offset(self):
        winter = aware_et(datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc))
        summer = aware_et(datetime(2024, 7, 8, 13, 30, tzinfo=timezone.utc))
        self.assertEqual((winter.hour, winter.minute), (9, 30))
        self.assertEqual((summer.hour, summer.minute), (9, 30))
        self.assertNotEqual(winter.utcoffset(), summer.utcoffset())

    def test_installation_is_idempotent_but_cannot_replace_active_plan(self):
        doc = self.install()
        self.assertEqual(install_plan(self.store, deepcopy(doc))["plan_id"],
                         self.store.load()["plan_id"])
        self.tick()
        altered = deepcopy(doc)
        altered["legs"][0]["notional"] = 500.
        with self.assertRaisesRegex(PaperError, "unresolved"):
            install_plan(self.store, altered)

    def test_naive_clock_is_rejected_without_order(self):
        self.install()
        with self.assertRaisesRegex(PaperError, "timezone"):
            self.engine.tick(datetime(2024, 1, 8, 10), submit=True)
        self.assertEqual(self.broker.requests, [])


if __name__ == "__main__":
    unittest.main()
