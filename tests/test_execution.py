from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from ndxbots.backtest.engine import run_backtest
from ndxbots.backtest.execution import Order, execute_orders
from ndxbots.config import load_settings
from ndxbots.strategy.targets import decide_quantity_targets


def order(index, target):
    return Order(index, target, "test", pd.Timestamp("2024-01-02"), 1_000.0)


class ExecutionTests(unittest.TestCase):
    def test_reversal_two_legs_conserve_account_value(self):
        cash, qty, fills, blocked = execute_orders(
            1_000.0, np.array([10.0]), [order(0, -5)], np.array([20.0]), 100, ["A"], 4
        )
        self.assertFalse(blocked)
        self.assertEqual([fill["leg"] for fill in fills], ["close_before_reverse", "open_after_reverse"])
        self.assertEqual(qty[0], -5)
        self.assertAlmostEqual(cash + qty[0] * 20, 1_197)

    def test_short_cover_then_long_open(self):
        cash, qty, fills, blocked = execute_orders(
            1_000.0, np.array([-10.0]), [order(0, 5)], np.array([20.0]), 100, ["A"], 4
        )
        self.assertFalse(blocked)
        self.assertEqual(qty[0], 5)
        self.assertEqual(len(fills), 2)
        self.assertAlmostEqual(cash + qty[0] * 20, 797)

    def test_cash_guard_scales_buys_after_short_proceeds_including_fees(self):
        cash, qty, fills, blocked = execute_orders(
            100.0, np.zeros(3), [order(0, 10), order(1, 10), order(2, -5)],
            np.array([20.0, 20.0, 20.0]), 100, ["A", "B", "C"], 4
        )
        self.assertFalse(blocked)
        self.assertAlmostEqual(cash, 0)
        self.assertAlmostEqual(qty[0], qty[1])
        self.assertEqual(qty[2], -5)
        self.assertEqual(fills[0]["index"], 2)
        self.assertTrue(all(fill["cash_constrained_partial"] for fill in fills[1:]))
        self.assertAlmostEqual(cash + np.dot(qty, [20, 20, 20]), 100 - sum(fill["cost"] for fill in fills))

    def test_missing_quote_cannot_close_or_reverse_or_use_future_price(self):
        cash, qty, fills, blocked = execute_orders(
            1_000.0, np.array([10.0]), [order(0, -5)], np.array([np.nan]), 10, ["A"], 4
        )
        self.assertEqual(cash, 1_000)
        self.assertEqual(qty[0], 10)
        self.assertFalse(fills)
        self.assertEqual(len(blocked), 2)

    def test_blocked_exit_cannot_create_extra_fifth_holding(self):
        cash, qty, fills, blocked = execute_orders(
            100.0, np.array([1, 1, 1, 1, 0.0]), [order(0, 0), order(4, 1)],
            np.array([np.nan, 10, 10, 10, 10]), 10, ["A", "B", "C", "D", "E"], 4
        )
        self.assertEqual(np.count_nonzero(qty), 4)
        self.assertFalse(fills)
        self.assertIn("slot_cap_after_blocked_exit", [item["event"] for item in blocked])

    def test_off_review_keeps_quantities_and_only_cuts_or_exits(self):
        decision = decide_quantity_targets(
            pd.Timestamp("2024-01-03"), np.array([1, 0, 1]), np.array([10.0, 10.0, 0]),
            np.array([15.0, 5.0, 10.0]), 1_000, 0.6, 1.0
        )
        np.testing.assert_array_equal(decision.quantities, [6, 0, 0])
        self.assertEqual(decision.reasons[1], "mandatory_selection_exit")
        retry = decide_quantity_targets(
            pd.Timestamp("2024-01-04"), np.array([1, 0, 1]), np.array([10.0, 10.0, 0]),
            np.array([15.0, 5.0, 10.0]), 1_000, 0.6, 0.6, decision.reduction_targets
        )
        np.testing.assert_array_equal(retry.quantities, [6, 0, 0])

    def _run(self, prices, active, pool=None, exec_mode="next_close"):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        panel = pd.DataFrame([
            {"date": day, "code": code, "close": price}
            for day, price in zip(dates, prices) for code in ["A"]
        ] + [{"date": day, "code": "US.QQQ", "close": 100.0} for day in dates])
        scores = pd.DataFrame({"date": dates, "code": "A", "side": "long", "in_hold": active,
                               "in_pool": pool if pool is not None else active})
        settings = replace(load_settings(), bt_start="2024-01-02", bt_end="2024-01-05",
                           initial_cash=1_000, cost_bps=0, use_gross_ladder=False, bt_exec=exec_mode)
        return run_backtest(settings, panel=panel, scored=scores)

    def test_next_close_quantity_fixed_from_signal_and_price_gap_not_earned(self):
        result = self._run([10, 5, 10, 10], [True] * 4)
        curve = result["curve"]
        self.assertEqual(curve.equity.iloc[0], 1_000)
        self.assertEqual(curve.equity.iloc[1], 1_000)
        self.assertEqual(curve.equity.iloc[2], 1_500)
        fill = result["fills"].iloc[0]
        self.assertEqual(fill.quantity, 100)
        self.assertEqual(fill.price, 5)
        self.assertEqual(fill.signal_date, pd.Timestamp("2024-01-02"))
        self.assertEqual(fill.date, pd.Timestamp("2024-01-03"))

    def test_zero_exit_outside_pool_is_in_complete_target_and_retries(self):
        result = self._run([10, 10, np.nan, 10], [True, False, False, False], [True, False, False, False])
        targets = result["targets"].set_index("date")
        self.assertEqual(targets.loc["2024-01-03", "target_quantity"], 0)
        self.assertFalse(targets.loc["2024-01-03", "in_pool"])
        self.assertEqual(targets.loc["2024-01-04", "order_status"], "blocked_missing_signal_quote")
        self.assertEqual(result["audit"]["execution_missing_quotes"], 1)
        self.assertEqual(result["curve"].n_hold.iloc[-1], 1)

    def test_execution_mode_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "next_close"):
            self._run([10] * 4, [True] * 4, exec_mode="next_open")

    def test_missing_entry_quote_keeps_selected_long_label_without_future_fill(self):
        result = self._run([np.nan, 10, 10, 10], [True] * 4)
        first = result["targets"].iloc[0]
        self.assertEqual(first.side, "long")
        self.assertEqual(first.target_quantity, 0)
        self.assertEqual(first.order_status, "blocked_missing_signal_quote")
        self.assertTrue(result["fills"].empty)

    def test_prefix_uses_no_future_execution_price_or_nav(self):
        first = self._run([10, 5, 10, 10], [True] * 4)
        changed_future = self._run([10, 5, 10, 999], [True] * 4)
        pd.testing.assert_frame_equal(first["curve"].iloc[:3], changed_future["curve"].iloc[:3])
        pd.testing.assert_frame_equal(first["fills"], changed_future["fills"])


if __name__ == "__main__":
    unittest.main()
