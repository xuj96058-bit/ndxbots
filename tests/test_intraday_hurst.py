"""Benchmark Hurst sizing preserves next-session execution and cash accounting."""
from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ndxbots.backtest.engine import run_backtest
from ndxbots.backtest.intraday import IntradaySimulator
from ndxbots.backtest.intraday_schedule import prepare_bars
from ndxbots.config import load_settings


class FlatMinuteStore:
    def bars(self, code, day, daily_open):
        return prepare_bars(pd.DataFrame({
            "bar_end": np.arange(575, 961, 5),
            "open": 10., "high": 10., "low": 10., "close": 10., "vwap": 10.,
        }))


def fixture(hurst=(.65, .65, .65, .65), *, cost=0., benchmark="US.QQQ"):
    days = pd.bdate_range("2024-01-02", periods=len(hurst))
    panel = pd.DataFrame([
        {"date": day, "code": code, "open": price, "close": price,
         "high": price, "low": price}
        for day in days for code, price in [("US.A", 10.), (benchmark, 100.)]
    ])
    scored = pd.DataFrame({"date": days, "code": "US.A", "side": "long",
                           "in_hold": True, "in_pool": True, "score": .8})
    factors = pd.concat([
        pd.DataFrame({"date": days, "code": "US.A", "my_atr_pct_20": .1,
                      "my_hurst_126": .1}),
        pd.DataFrame({"date": days, "code": benchmark, "my_atr_pct_20": .1,
                      "my_hurst_126": hurst}),
    ], ignore_index=True)
    settings = replace(load_settings(), benchmark=benchmark, bt_exec="next_intraday",
                       bt_start=str(days[0].date()), bt_end=str(days[-1].date()),
                       initial_cash=1_000., cost_bps=cost, max_hold=2,
                       use_gross_ladder=False, rebalance_weekdays=(1, 4))
    return settings, panel, factors, scored


def simulate(parts):
    return run_backtest(parts[0], parts[1], parts[2], scored=parts[3],
                        minute_store=FlatMinuteStore())


class IntradayHurstTests(unittest.TestCase):
    def test_thresholds_boundaries_and_missing_values_use_benchmark_only(self):
        parts = fixture((.750001, .549999, .55, .75, .65, np.nan, .8))
        factors = parts[2]
        # A missing benchmark row must not carry yesterday's scale forward.
        factors = factors.drop(factors.index[-1])
        simulator = IntradaySimulator(parts[0], parts[1], factors, parts[3])
        np.testing.assert_allclose(simulator.gross, [1.3, .5, 1., 1., 1., 1., 1.])
        np.testing.assert_array_equal(simulator.selected, np.ones((7, 1)))

    def test_configured_benchmark_controls_scale(self):
        parts = fixture((.8, .5, .65, .65), benchmark="US.SPY")
        simulator = IntradaySimulator(*parts)
        np.testing.assert_allclose(simulator.gross, [1.3, .5, 1., 1.])

    def test_scale_multiplies_regime_gross_then_caps_at_one_point_five(self):
        parts = fixture((.8, .8, .5, .65))
        settings = replace(parts[0], use_gross_ladder=True)
        gross = pd.Series([1.4, .8, 1.4, 0.], index=parts[3].date)
        with patch("ndxbots.backtest.intraday.load_regime_table", return_value=pd.DataFrame()), \
             patch("ndxbots.backtest.intraday.gross_by_date", return_value=gross):
            simulator = IntradaySimulator(settings, *parts[1:])
        np.testing.assert_allclose(simulator.gross, [1.5, 1.04, .7, 0.])

    def test_string_dates_are_normalized_without_mutating_factors(self):
        parts = fixture((.8, .5, .55, .75))
        factors = parts[2].copy()
        factors["date"] = factors.date.dt.strftime("%Y-%m-%d 16:00:00")
        factors = factors.iloc[::-1].reset_index(drop=True)
        original = factors.copy(deep=True)
        simulator = IntradaySimulator(parts[0], parts[1], factors, parts[3])
        np.testing.assert_allclose(simulator.gross, [1.3, .5, 1., 1.])
        pd.testing.assert_frame_equal(factors, original)

    def test_duplicate_benchmark_sessions_after_normalization_are_rejected(self):
        parts = fixture()
        # Omit ATR so its existing duplicate guard cannot mask missing Hurst validation.
        factors = parts[2].drop(columns="my_atr_pct_20")
        duplicate = factors.loc[factors.code.eq(parts[0].benchmark)].iloc[[0]].copy()
        duplicate["date"] += pd.Timedelta(hours=16)
        factors = pd.concat([factors, duplicate], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            IntradaySimulator(parts[0], parts[1], factors, parts[3])

    def test_unavailable_hurst_preserves_existing_regime_targets(self):
        parts = fixture()
        settings = replace(parts[0], use_gross_ladder=True)
        gross = pd.Series([1.8, .7, 1., 0.], index=parts[3].date)
        unavailable = {
            "no_column": parts[2].drop(columns="my_hurst_126"),
            "no_benchmark_rows": parts[2].loc[parts[2].code.eq("US.A")],
            "no_factors": None,
        }
        for reason, factors in unavailable.items():
            with self.subTest(reason=reason), \
                 patch("ndxbots.backtest.intraday.load_regime_table", return_value=pd.DataFrame()), \
                 patch("ndxbots.backtest.intraday.gross_by_date", return_value=gross):
                simulator = IntradaySimulator(settings, parts[1], factors, parts[3])
                np.testing.assert_allclose(simulator.gross, gross.to_numpy())

    def test_nullable_hurst_missing_value_keeps_neutral_scale(self):
        parts = fixture((.8, pd.NA, .5, .65))
        parts[2]["my_hurst_126"] = parts[2].my_hurst_126.astype("Float64")
        simulator = IntradaySimulator(*parts)
        np.testing.assert_allclose(simulator.gross, [1.3, 1., .5, 1.])

    def test_infinite_hurst_is_rejected_instead_of_becoming_a_scale(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                IntradaySimulator(*fixture((value, .65, .65, .65)))

    def test_malformed_hurst_is_rejected_instead_of_becoming_missing(self):
        with self.assertRaises(ValueError):
            IntradaySimulator(*fixture(("not-a-number", .65, .65, .65)))

    def test_invalid_regime_gross_cannot_be_hidden_by_hurst_clipping(self):
        parts = fixture((.8, .8, .8, .8))
        settings = replace(parts[0], use_gross_ladder=True)
        for value in (-.1, np.inf, -np.inf):
            gross = pd.Series([value, 1., 1., 1.], index=parts[3].date)
            with self.subTest(value=value), \
                 patch("ndxbots.backtest.intraday.load_regime_table", return_value=pd.DataFrame()), \
                 patch("ndxbots.backtest.intraday.gross_by_date", return_value=gross), \
                 self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                IntradaySimulator(settings, *parts[1:])

    def test_low_hurst_sets_signal_budget_filled_on_next_session(self):
        result = simulate(fixture((.5, .5, .5, .5), cost=10.))
        target, fill = result["targets"].iloc[0], result["fills"].iloc[0]
        self.assertEqual(target.date, pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(target.planned_addition_notional, 500.)
        self.assertEqual(fill.signal_date, target.date)
        self.assertEqual(fill.date, pd.Timestamp("2024-01-03"))
        self.assertEqual(fill.time, 600)
        self.assertAlmostEqual(fill.quantity, 50.)
        self.assertAlmostEqual(fill.signed_notional, 500.)
        self.assertAlmostEqual(fill.cost, .5)
        self.assertAlmostEqual(result["curve"].cash.iloc[1], 499.5)
        self.assertAlmostEqual(result["curve"].equity.iloc[1], 999.5)
        self.assertFalse(result["holdings"].empty)
        self.assertTrue(result["completion"].complete.all())
        self.assertEqual(result["audit"]["incomplete_order_rows"], 0)
        self.assertLess(result["audit"]["max_fill_nav_conservation_error"], 1e-7)

    def test_high_hurst_budget_still_reserves_fees_and_cannot_borrow_cash(self):
        result = simulate(fixture((.8, .8, .8, .8), cost=10.))
        funded = 1_000. / 1.001
        completion, fill = result["completion"].iloc[0], result["fills"].iloc[0]
        self.assertAlmostEqual(completion.planned_addition_notional, 1_300.)
        self.assertAlmostEqual(completion.funded_budget, funded)
        self.assertAlmostEqual(completion.spent_notional, funded)
        self.assertAlmostEqual(fill.cost, funded * .001)
        self.assertTrue(completion.cash_capped)
        self.assertTrue(completion.complete)
        self.assertGreaterEqual(result["curve"].cash.min(), 0.)
        self.assertAlmostEqual(result["curve"].cash.iloc[1], 0.)
        self.assertGreater(result["audit"]["cash_capped_budget_legs"], 0)

    def test_future_hurst_does_not_change_earlier_signals_or_execution(self):
        baseline = simulate(fixture())
        changed = simulate(fixture((.65, .65, .5, .65)))
        cutoff = pd.Timestamp("2024-01-04")
        for table in ("curve", "targets", "holdings"):
            pd.testing.assert_frame_equal(
                baseline[table].loc[baseline[table].date.lt(cutoff)].reset_index(drop=True),
                changed[table].loc[changed[table].date.lt(cutoff)].reset_index(drop=True),
            )
        # Even cutoff-day fills were fixed by the preceding session's signal.
        pd.testing.assert_frame_equal(
            baseline["fills"].loc[baseline["fills"].date.le(cutoff)].reset_index(drop=True),
            changed["fills"].loc[changed["fills"].date.le(cutoff)].reset_index(drop=True),
        )
        reduction = changed["fills"].loc[changed["fills"].action.eq("exit")].iloc[0]
        self.assertEqual(reduction.signal_date, cutoff)
        self.assertEqual(reduction.date, pd.Timestamp("2024-01-05"))
        self.assertEqual(reduction.time, 570)
        self.assertAlmostEqual(reduction.quantity, -50.)
        self.assertAlmostEqual(changed["curve"].gross_exp.iloc[-1], .5)


if __name__ == "__main__":
    unittest.main()
