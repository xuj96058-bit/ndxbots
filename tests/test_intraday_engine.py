"""Independent first-group ledger checks; no Futu process or network required."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from ndxbots.backtest.engine import run_backtest
from ndxbots.backtest.execution import Order
from ndxbots.backtest.intraday import ENTRY_POLICY, EXIT_POLICY, IncompleteExecutionError, IntradaySimulator
from ndxbots.backtest.intraday_schedule import prepare_bars
from ndxbots.config import load_settings


def session(price=10., end=960, overrides=None):
    ends = np.arange(575, end + 1, 5)
    frame = pd.DataFrame({"bar_end": ends, "open": price, "high": price,
                          "low": price, "close": price, "vwap": price})
    for minute, fields in (overrides or {}).items():
        index = np.flatnonzero(ends == minute + 5)[0]
        for field, value in fields.items():
            frame.loc[index, field] = value
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


class FakeMinuteStore:
    def __init__(self, frames):
        self.frames, self.calls = frames, []

    def bars(self, code, day, daily_open):
        day = pd.Timestamp(day).normalize()
        self.calls.append((code, day, daily_open))
        if (code, day) not in self.frames:
            raise ValueError(f"{code} {day.date()}: minute data absent; no daily fallback")
        return prepare_bars(self.frames[(code, day)])


def fixture(prices=None, opens=None, active=None, sides=None, cost=0., days=None, frames=None):
    days = pd.DatetimeIndex(days if days is not None else
                           ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    prices = prices if prices is not None else [10.] * len(days)
    opens = opens if opens is not None else prices
    active = active if active is not None else [True] * len(days)
    sides = sides if sides is not None else ["long"] * len(days)
    panel = pd.DataFrame([
        {"date": day, "code": "US.A", "open": op, "close": px, "high": max(op, px), "low": min(op, px)}
        for day, op, px in zip(days, opens, prices)] + [
        {"date": day, "code": "US.QQQ", "open": 100., "close": 100., "high": 100., "low": 100.}
        for day in days])
    scored = pd.DataFrame({"date": days, "code": "US.A", "side": sides,
                           "in_hold": active, "in_pool": active, "score": .8})
    factors = pd.DataFrame({"date": days, "code": "US.A", "my_atr_pct_20": .1})
    settings = replace(load_settings(), bt_exec="next_intraday", bt_start=str(days[0].date()),
                       bt_end=str(days[-1].date()), initial_cash=1_000., cost_bps=cost,
                       max_hold=2, use_gross_ladder=False, rebalance_weekdays=(1, 4))
    if frames is None:
        frames = {("US.A", day): session(float(px)) for day, px in zip(days, prices) if np.isfinite(px)}
    return settings, panel, factors, scored, FakeMinuteStore(frames)


def run_fixture(parts):
    settings, panel, factors, scored, store = parts
    return run_backtest(settings, panel, factors, scored=scored, minute_store=store)


class IntradayIntegrationTests(unittest.TestCase):
    def test_budget_shares_follow_price_and_intraday_pnl_marks_to_close(self):
        parts = fixture(prices=[10., 10., 15., 15.])
        parts[-1].frames[("US.A", pd.Timestamp("2024-01-03"))] = session(
            10., overrides={600: {"open": 20., "close": 20.}})
        result = run_fixture(parts)
        fill = result["fills"].iloc[0]
        self.assertEqual(fill.signal_date, pd.Timestamp("2024-01-02"))
        self.assertEqual(fill.date, pd.Timestamp("2024-01-03"))
        self.assertEqual(fill.time, 600)
        self.assertAlmostEqual(fill.quantity, 50.)
        self.assertAlmostEqual(fill.signed_notional, 1_000.)
        np.testing.assert_allclose(result["curve"].equity.iloc[:3], [1_000., 500., 750.])
        self.assertAlmostEqual(result["curve"].execution_pnl.iloc[1], -500.)
        self.assertAlmostEqual(result["curve"].mark_pnl.iloc[2], 250.)
        self.assertEqual(result["targets"].target_quantity.iloc[0], 100.)
        self.assertTrue(result["completion"].complete.all())

    def test_full_available_budget_reserves_fees(self):
        result = run_fixture(fixture(cost=10.))
        fill, expected = result["fills"].iloc[0], 1_000. / 1.001
        self.assertAlmostEqual(fill.signed_notional, expected)
        self.assertAlmostEqual(fill.cost, expected * .001)
        self.assertAlmostEqual(result["curve"].cash.iloc[1], 0.)
        complete = result["completion"].iloc[0]
        self.assertAlmostEqual(complete.planned_addition_notional, 1_000.)
        self.assertAlmostEqual(complete.funded_budget, expected)
        self.assertAlmostEqual(complete.spent_notional, expected)
        self.assertTrue(complete.cash_capped)
        self.assertTrue(complete.complete)
        self.assertFalse(fill.cash_constrained_partial)

    def test_short_entry_uses_budget_and_cover_preserves_actual_short_quantity(self):
        parts = fixture(active=[True, False, False, False], sides=["short", None, None, None])
        parts[-1].frames[("US.A", pd.Timestamp("2024-01-03"))] = session(
            10., overrides={600: {"open": 20., "close": 20.}})
        result = run_fixture(parts)
        entry, cover = result["fills"].iloc[0], result["fills"].iloc[1]
        self.assertEqual(entry.side, -1)
        self.assertEqual(entry.action, "entry")
        self.assertAlmostEqual(entry.quantity, -50.)
        self.assertAlmostEqual(abs(entry.signed_notional), 1_000.)
        self.assertEqual(cover.action, "exit")
        self.assertEqual(cover.time, 570)
        self.assertAlmostEqual(cover.quantity, 50.)
        self.assertAlmostEqual(cover.price, 10.)
        np.testing.assert_allclose(result["curve"].equity, [1_000., 1_500., 1_500., 1_500.])
        self.assertEqual(result["curve"].n_short.iloc[1], 1)
        self.assertEqual(result["curve"].n_hold.iloc[2], 0)
        self.assertTrue(result["completion"].complete.all())

    def test_large_gap_forces_remainder_before_full_or_half_day_close(self):
        for end, deadline in [(960, 955), (780, 775)]:
            with self.subTest(session_end=end):
                days = ["2024-01-02", "2024-01-03"]
                frames = {("US.A", pd.Timestamp(days[1])): session(11., end=end)}
                result = run_fixture(fixture(prices=[10., 11.], opens=[10., 11.], days=days, frames=frames))
                fills = result["fills"]
                self.assertEqual(fills.time.tolist(), [600, deadline])
                np.testing.assert_allclose(fills.signed_notional, [500., 500.])
                self.assertEqual(fills.trigger.iloc[-1], "deadline_entry")
                self.assertTrue(fills.forced.iloc[-1])
                self.assertTrue(result["completion"].complete.all())
                self.assertEqual(result["audit"]["late_entry_orders"], 0)

    def test_atr_limit_uses_signal_day_atr_and_trade_through(self):
        parts = fixture(prices=[10., 10.2, 10.2, 10.2], opens=[10., 10.2, 10.2, 10.2])
        parts[2].loc[parts[2].date.eq(pd.Timestamp("2024-01-03")), "my_atr_pct_20"] = 1.
        parts[-1].frames[("US.A", pd.Timestamp("2024-01-03"))] = session(
            10.2, overrides={605: {"low": 9.99, "high": 10.3}})
        result = run_fixture(parts)
        fills = result["fills"]
        self.assertEqual(fills.time.tolist(), [600, 610])
        self.assertAlmostEqual(fills.price.iloc[-1], 10.)
        self.assertEqual(fills.trigger.iloc[-1], "atr_limit")

    def test_valid_replacement_exits_normally_invalid_direction_urgently(self):
        for side, exit_time, trigger in [("long", 585, "immediate_exit"), (None, 570, "urgent_open")]:
            with self.subTest(side=side):
                result = run_fixture(fixture(active=[True, False, False, False], sides=["long", side, side, side]))
                exits = result["fills"].loc[result["fills"].action.eq("exit")]
                self.assertEqual(len(exits), 1)
                self.assertEqual(exits.time.iloc[0], exit_time)
                self.assertEqual(exits.trigger.iloc[0], trigger)
                self.assertEqual(exits.date.iloc[0], pd.Timestamp("2024-01-04"))

    def test_gross_cut_reduces_fixed_quantity_at_urgent_open(self):
        parts = fixture()
        parts = (replace(parts[0], use_gross_ladder=True), *parts[1:])
        gross = pd.Series([1., .5, .5, .5], index=pd.to_datetime(parts[1].date.unique()))
        with patch("ndxbots.backtest.intraday.load_regime_table", return_value=pd.DataFrame()), \
             patch("ndxbots.backtest.intraday.gross_by_date", return_value=gross):
            result = run_fixture(parts)
        reduced = result["fills"].loc[result["fills"].action.eq("exit")].iloc[0]
        self.assertEqual(reduced.time, 570)
        self.assertEqual(reduced.trigger, "urgent_open")
        self.assertAlmostEqual(reduced.quantity, -50.)
        self.assertAlmostEqual(result["curve"].gross_exp.iloc[2], .5)

    def test_future_prices_cannot_change_earlier_fills_nav_or_targets(self):
        original = run_fixture(fixture())
        future = run_fixture(fixture(prices=[10., 10., 10., 999.]))
        pd.testing.assert_frame_equal(original["curve"].iloc[:3], future["curve"].iloc[:3])
        pd.testing.assert_frame_equal(original["fills"], future["fills"])
        pd.testing.assert_frame_equal(original["targets"].iloc[:3], future["targets"].iloc[:3])

    def test_output_contract_and_terminal_signal_not_counted_failure(self):
        result = run_fixture(fixture(active=[True, True, True, False]))
        self.assertTrue({"curve", "holdings", "stats", "scored", "fills", "targets", "logs", "audit", "completion"}.issubset(result))
        self.assertTrue({"date", "code", "side", "quantity", "close", "w", "value", "score"}.issubset(result["holdings"].columns))
        self.assertTrue({"current_weight", "target_weight", "scheduled_review", "in_pool", "in_selected",
                         "order_status", "planned_addition_notional", "sizing_mode"}.issubset(result["targets"].columns))
        self.assertTrue({"time", "time_et", "trigger", "signal_date", "reserved_budget"}.issubset(result["fills"].columns))
        self.assertGreater(result["audit"]["last_unexecuted_order_rows"], 0)
        self.assertEqual(result["audit"]["incomplete_order_rows"], 0)
        self.assertIn("第一組", result["stats"])

    def test_empty_account_keeps_exportable_table_schemas(self):
        result = run_fixture(fixture(active=[False] * 4))
        self.assertTrue(result["fills"].empty)
        self.assertTrue(result["targets"].empty)
        for key in ["fills", "targets", "completion", "holdings", "logs", "scored"]:
            self.assertTrue({"date", "code"}.issubset(result[key].columns), key)
        np.testing.assert_allclose(result["curve"].equity, 1_000.)

    def test_empty_account_does_not_require_or_create_default_minute_cache(self):
        parts = fixture(active=[False] * 4)
        with tempfile.TemporaryDirectory(prefix=".intraday-empty-", dir=Path(__file__).resolve().parent) as folder:
            data_root = Path(folder) / "no-data"
            minute_root = Path(folder) / "no-minutes"
            settings = replace(parts[0], data_root=data_root, intraday_root=minute_root)
            result = run_backtest(settings, parts[1], parts[2], scored=parts[3])
            self.assertTrue(result["fills"].empty)
            self.assertFalse(data_root.exists())
            self.assertFalse(minute_root.exists())

    def test_missing_minutes_raise_without_daily_fallback(self):
        with self.assertRaisesRegex(ValueError, "minute data absent"):
            run_fixture(fixture(frames={}))

    def test_replaced_next_close_and_other_modes_fail_clearly(self):
        for mode in ["next_close", "next_open"]:
            with self.subTest(mode=mode):
                parts = fixture()
                with self.assertRaisesRegex(ValueError, "next_intraday"):
                    run_fixture((replace(parts[0], bt_exec=mode), *parts[1:]))

    def test_invalid_inputs_and_duplicate_rows_fail_clearly(self):
        parts = fixture()
        for changes in [{"initial_cash": 0.}, {"cost_bps": -1.}, {"max_hold": 0}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_fixture((replace(parts[0], **changes), *parts[1:]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_fixture((parts[0], pd.concat([parts[1], parts[1].iloc[:1]]), *parts[2:]))

    def test_wrapper_rejects_incomplete_or_zero_funded_budget(self):
        failed = {"audit": {"incomplete_order_rows": 1}, "completion": pd.DataFrame([{
            "complete": False, "code": "US.A", "date": pd.Timestamp("2024-01-03"),
            "funded_budget": 0., "spent_notional": 0.}])}
        with patch.object(IntradaySimulator, "simulate", return_value=failed), \
             self.assertRaisesRegex(IncompleteExecutionError, r"T\+1 execution incomplete"):
            run_fixture(fixture())


def day_ledger(schedules, opens=None):
    simulator = IntradaySimulator.__new__(IntradaySimulator)
    simulator.codes = np.array(["US.A", "US.B", "US.C"])
    simulator.code_index = {code: j for j, code in enumerate(simulator.codes)}
    simulator.dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    simulator.raw = np.full((2, 3), 100.)
    simulator.max_hold = 2
    bars = session(100., overrides={m: {"open": value} for m, value in (opens or {}).items()})
    bars["bar_start"] = bars.bar_end - 5
    simulator._schedule = types.MethodType(lambda self, code, t, side, action, policy, urgent:
        [dict(row) for row in schedules[(code, action)]], simulator)
    simulator.bars = types.MethodType(lambda self, code, day, t: bars, simulator)
    return simulator


def tranche(time, fraction=1., price=100., trigger="test_market", active_from=None):
    row = {"time": time, "fraction": fraction, "price": price, "forced": False, "trigger": trigger}
    if active_from is not None:
        row["active_from"] = active_from
    return row


def ledger_order(index, quantity):
    return Order(index, quantity, "test", pd.Timestamp("2024-01-02"), 200.)


class IntradayLedgerTests(unittest.TestCase):
    def execute(self, simulator, cash, quantity, orders, fee=0.):
        return simulator._execute_day(1, cash, np.asarray(quantity, float), orders, {},
                                      ENTRY_POLICY, EXIT_POLICY, fee, "budget")

    def test_limit_cannot_use_slot_or_cash_released_only_at_bar_end(self):
        schedules = {("US.A", "exit"): [tranche(620)],
                     ("US.B", "entry"): [tranche(570, .5), tranche(620, .5, 90., "test_limit", 615)]}
        simulator = day_ledger(schedules, opens={620: 120.})
        cash, quantity, fills, completion = self.execute(
            simulator, 0., [1., 0., 1.], [ledger_order(0, 0.), ledger_order(1, 1.)])
        entry = next(fill for fill in fills if fill["action"] == "entry")
        self.assertEqual(entry["time"], 620)
        self.assertEqual(entry["price"], 120.)
        self.assertEqual(entry["trigger"], "unfunded_limit_repriced_at_current_open")
        self.assertAlmostEqual(quantity[1], 100. / 120.)
        self.assertAlmostEqual(cash, 0.)
        self.assertEqual(np.count_nonzero(quantity), 2)
        self.assertTrue(all(row["complete"] for row in completion))
        self.assertTrue(completion[1]["discarded_unfunded_limit"])

    def test_reserved_remainder_cannot_be_spent_by_another_entry(self):
        simulator = day_ledger({("US.A", "entry"): [tranche(570, .5), tranche(955, .5)],
                                ("US.B", "entry"): [tranche(600)]})
        cash, quantity, fills, completion = self.execute(
            simulator, 100., [0., 0., 0.], [ledger_order(0, 1.), ledger_order(1, 1.)])
        np.testing.assert_allclose(quantity, [1., 0., 0.])
        self.assertAlmostEqual(cash, 0.)
        self.assertTrue(completion[0]["complete"])
        self.assertFalse(completion[1]["complete"])
        self.assertEqual(completion[1]["funded_budget"], 0.)
        self.assertFalse(any(fill["index"] == 1 for fill in fills))

    def test_same_time_short_proceeds_exist_before_long_addition(self):
        simulator = day_ledger({("US.A", "entry"): [tranche(600)],
                                ("US.B", "entry"): [tranche(600)]})
        cash, quantity, fills, completion = self.execute(
            simulator, 100., [0., 0., 0.], [ledger_order(0, 1.), ledger_order(1, -1.)], fee=.01)
        self.assertEqual([fill["index"] for fill in fills], [1, 0])
        np.testing.assert_allclose(quantity, [1., -1., 0.])
        self.assertAlmostEqual(cash, 98.)
        self.assertAlmostEqual(sum(fill["cost"] for fill in fills), 2.)
        self.assertTrue(all(row["complete"] for row in completion))

    def test_cash_limited_cover_reports_requested_and_actual_quantities(self):
        simulator = day_ledger({("US.A", "exit"): [tranche(570)]})
        cash, quantity, fills, completion = self.execute(
            simulator, 40., [-1., 0., 0.], [ledger_order(0, 0.)])
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0]["requested_quantity"], 1.)
        self.assertAlmostEqual(fills[0]["quantity"], .4)
        self.assertTrue(fills[0]["cash_constrained_partial"])
        self.assertAlmostEqual(quantity[0], -.6)
        self.assertAlmostEqual(cash, 0.)
        self.assertFalse(completion[0]["complete"])
        self.assertTrue(completion[0]["cash_deferred"])

    def test_reversal_closes_before_spending_new_direction_budget(self):
        simulator = day_ledger({("US.A", "exit"): [tranche(585)],
                                ("US.A", "entry"): [tranche(600, price=120.)]})
        cash, quantity, fills, completion = self.execute(
            simulator, 100., [1., 0., 0.], [ledger_order(0, -1.)])
        self.assertEqual([fill["leg"] for fill in fills], ["close_before_reverse", "open_after_reverse"])
        self.assertEqual(fills[0]["quantity"], -1.)
        self.assertAlmostEqual(quantity[0], -100. / 120.)
        self.assertAlmostEqual(cash, 300.)
        self.assertTrue(completion[0]["complete"])

    def test_slot_release_after_deadline_cannot_count_as_complete(self):
        simulator = day_ledger({("US.A", "exit"): [tranche(960)], ("US.B", "entry"): [tranche(570)]})
        _, quantity, fills, completion = self.execute(
            simulator, 0., [1., 0., 1.], [ledger_order(0, 0.), ledger_order(1, 1.)])
        self.assertEqual(np.count_nonzero(quantity), 2)
        self.assertEqual(fills[-1]["time"], 960)
        self.assertTrue(completion[1]["late_entry"])
        self.assertFalse(completion[1]["complete"])


if __name__ == "__main__":
    unittest.main()
