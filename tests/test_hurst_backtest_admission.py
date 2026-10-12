"""New-long Hurst checks use actual account holdings and signal-day data."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from ndxbots.backtest.engine import run_backtest
from ndxbots.backtest.intraday_schedule import prepare_bars
from ndxbots.config import load_settings
from ndxbots.strategy.scores import apply_holding_hysteresis


class FlatMinuteStore:
    def bars(self, code, day, daily_open):
        return prepare_bars(pd.DataFrame({
            "bar_end": np.arange(575, 961, 5),
            "open": 10., "high": 10., "low": 10., "close": 10., "vwap": 10.,
        }))


def fixture(stock_hurst, *, benchmark_hurst=None, sides=None, start=None):
    days = pd.bdate_range("2024-01-02", periods=len(stock_hurst))
    benchmark_hurst = benchmark_hurst if benchmark_hurst is not None else [.65] * len(days)
    sides = sides if sides is not None else ["long"] * len(days)
    panel = pd.DataFrame([
        {"date": day, "code": code, "open": price, "close": price,
         "high": price, "low": price}
        for day in days for code, price in (("US.A", 10.), ("US.QQQ", 100.))
    ])
    factors = pd.concat([
        pd.DataFrame({"date": days, "code": "US.A", "my_atr_pct_20": .1,
                      "my_hurst_126": stock_hurst}),
        pd.DataFrame({"date": days, "code": "US.QQQ", "my_atr_pct_20": .1,
                      "my_hurst_126": benchmark_hurst}),
    ], ignore_index=True)
    # These rows can originate from selection warmed up before the evaluation
    # account exists. Actual holdings, not in_hold alone, control new entry.
    scored = pd.DataFrame({"date": days, "code": "US.A", "side": sides,
                           "in_hold": True, "in_pool": True, "score": .8})
    settings = replace(load_settings(), benchmark="US.QQQ", bt_exec="next_intraday",
                       hurst_gate_mode="strict_entry",
                       bt_start=start or str(days[0].date()), bt_end=str(days[-1].date()),
                       initial_cash=1_000., cost_bps=0., max_hold=2,
                       use_gross_ladder=False, rebalance_weekdays=(1, 4))
    return settings, panel, factors, scored


def simulate(parts):
    return run_backtest(parts[0], parts[1], parts[2], scored=parts[3],
                        minute_store=FlatMinuteStore())


class HurstAccountAdmissionTests(unittest.TestCase):
    def test_legacy_mode_preserves_selection_and_native_execution_with_benchmark_scale(self):
        parts = fixture([.5] * 5, benchmark_hurst=[.5] * 5)
        legacy = replace(parts[0], hurst_gate_mode="legacy_regime")
        result = simulate((legacy, *parts[1:]))
        fill = result["fills"].iloc[0]
        self.assertEqual(fill.signal_date, pd.Timestamp("2024-01-02"))
        self.assertEqual(fill.date, pd.Timestamp("2024-01-03"))
        self.assertEqual(fill.time, 600)
        self.assertEqual(fill.trigger, "initial_entry")
        self.assertEqual(fill.quantity, 50.)
        self.assertEqual(result["curve"].cash.iloc[1], 500.)
        self.assertTrue(result["completion"].complete.all())

    def test_warmup_selection_cannot_open_a_low_hurst_long_in_fresh_account(self):
        parts = fixture([.8, .5, .5, .5, .5], start="2024-01-03")
        selection_input = parts[3].assign(
            regime_side="long", pool_side="long", side_score=.8,
            combo_rank=1., long_score=.8, short_score=.2,
            my_hurst_126=[.8, .5, .5, .5, .5],
        )
        scored = apply_holding_hysteresis(selection_input, parts[0])
        self.assertTrue(scored.in_hold.all())
        result = simulate((*parts[:3], scored))
        self.assertTrue(result["fills"].empty)
        self.assertTrue(result["holdings"].empty)
        np.testing.assert_allclose(result["curve"].cash, 1_000.)
        review = result["targets"].loc[result["targets"].date.eq("2024-01-05")].iloc[0]
        self.assertEqual(review.current_quantity, 0.)
        self.assertEqual(review.target_quantity, 0.)
        self.assertEqual(review.planned_addition_notional, 0.)
        self.assertEqual(review.reason, "new_long_hurst_blocked")
        self.assertEqual(review.order_status, "blocked_hurst_entry")

    def test_allowed_signal_is_not_cancelled_by_execution_day_low_hurst(self):
        result = simulate(fixture([.8, .5, .5, .5, .5]))
        fill = result["fills"].iloc[0]
        self.assertEqual(fill.signal_date, pd.Timestamp("2024-01-02"))
        self.assertEqual(fill.date, pd.Timestamp("2024-01-03"))
        self.assertEqual(fill.action, "entry")
        self.assertEqual(fill.quantity, 100.)
        self.assertEqual(len(result["fills"]), 1)

    def test_low_hurst_does_not_exit_existing_long_or_block_scheduled_increase(self):
        result = simulate(fixture([.8, .5, .5, .5, .5],
                                  benchmark_hurst=[.5, .5, .5, .65, .65]))
        fills = result["fills"]
        self.assertEqual(list(fills.action), ["entry", "entry"])
        np.testing.assert_allclose(fills.quantity, [50., 50.])
        self.assertEqual(list(fills.signal_date), [pd.Timestamp("2024-01-02"),
                                                  pd.Timestamp("2024-01-05")])
        self.assertEqual(list(fills.date), [pd.Timestamp("2024-01-03"),
                                           pd.Timestamp("2024-01-08")])
        self.assertEqual(result["holdings"].iloc[-1].quantity, 100.)
        self.assertFalse(result["targets"].reason.eq("new_long_hurst_blocked").any())

    def test_blocked_short_to_long_reversal_still_exits_existing_short(self):
        result = simulate(fixture([.5] * 5,
                                  sides=["short", "short", "short", "long", "long"]))
        fills = result["fills"]
        self.assertEqual(list(fills.action), ["entry", "exit"])
        np.testing.assert_allclose(fills.quantity, [-100., 100.])
        self.assertEqual(fills.iloc[1].signal_date, pd.Timestamp("2024-01-05"))
        self.assertEqual(fills.iloc[1].date, pd.Timestamp("2024-01-08"))
        review = result["targets"].loc[result["targets"].date.eq("2024-01-05")].iloc[0]
        self.assertEqual(review.current_quantity, -100.)
        self.assertEqual(review.target_quantity, 0.)
        self.assertEqual(review.delta_quantity, 100.)
        self.assertEqual(review.reason, "new_long_hurst_blocked")
        self.assertEqual(review.order_status, "queued_next_intraday")
        self.assertEqual(result["curve"].cash.iloc[-1], 1_000.)
        self.assertEqual(result["curve"].gross_exp.iloc[-1], 0.)

    def test_missing_stock_hurst_blocks_entry_even_with_high_benchmark_hurst(self):
        for missing in ("value", "row"):
            with self.subTest(missing=missing):
                parts = fixture([np.nan] * 5, benchmark_hurst=[.8] * 5)
                factors = parts[2]
                if missing == "row":
                    factors = factors.loc[factors.code.ne("US.A")].copy()
                result = simulate((parts[0], parts[1], factors, parts[3]))
                self.assertTrue(result["fills"].empty)
                self.assertTrue(result["holdings"].empty)
                reviews = result["targets"].loc[result["targets"].scheduled_review]
                self.assertTrue(reviews.reason.eq("new_long_hurst_blocked").all())

    def test_hard_threshold_is_strict_for_actual_new_long(self):
        denied = simulate(fixture([.67] * 5))
        self.assertTrue(denied["fills"].empty)
        allowed = simulate(fixture([.670001, .5, .5, .5, .5]))
        self.assertEqual(allowed["fills"].iloc[0].quantity, 100.)


if __name__ == "__main__":
    unittest.main()
