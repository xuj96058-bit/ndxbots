from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ndxbots.config import load_settings
from ndxbots.strategy.scores import apply_holding_hysteresis, latest_pool


def settings(**changes):
    values = dict(max_hold=4, min_hold_days=10, keep_rank=8,
                  strategy_factors=("f1", "f2"), strategy_top_n=10,
                  rebalance_weekdays=(1, 4), replacement_score_gap=0.05)
    values.update(changes)
    return SimpleNamespace(**values)


def row(day, code, side="long", score=0.5, rank=1, *, regime="same", eligible=True, raw=1):
    long_score = score if side == "long" else 1 - score
    in_pool = eligible and rank is not None and rank <= 10
    return dict(date=pd.Timestamp(day), code=code,
                regime_side=side if regime == "same" else regime,
                pool_side=side if eligible else None,
                side_score=score if eligible else np.nan,
                combo_rank=rank if eligible and rank is not None else np.nan,
                in_pool=in_pool, in_pool_long=in_pool and side == "long",
                in_pool_short=in_pool and side == "short",
                score=long_score, long_score=long_score, short_score=1 - long_score,
                side=side, f1=raw, f2=raw)


def held(scored, day):
    rows = scored[scored.date.eq(pd.Timestamp(day)) & scored.in_hold]
    return set(zip(rows.code, rows.side))


class SelectionTests(unittest.TestCase):
    def select(self, rows, **kwargs):
        return apply_holding_hysteresis(pd.DataFrame(rows), settings(**kwargs))

    def test_unified_order_ignores_raw_magnitude_and_excludes_rank_eleven(self):
        result = self.select([row("2024-01-02", "A", score=.9, raw=.001),
                              row("2024-01-02", "B", score=.8, rank=2, raw=100),
                              row("2024-01-02", "C", score=.99, rank=11, raw=1000)], max_hold=1)
        self.assertEqual(held(result, "2024-01-02"), {("A", "long")})

    def test_ties_follow_combined_rank_across_sides(self):
        result = self.select([row("2024-01-02", "A", "short", .9, 1),
                              row("2024-01-02", "B", "long", .9, 2)], max_hold=1)
        self.assertEqual(held(result, "2024-01-02"), {("A", "short")})

    def test_code_is_final_tie_breaker_independent_of_input_order(self):
        result = self.select([row("2024-01-02", "B", score=.9),
                              row("2024-01-02", "A", score=.9)], max_hold=1)
        self.assertEqual(held(result, "2024-01-02"), {("A", "long")})

    def test_latest_display_keeps_combined_order_and_scores(self):
        scored = self.select([row("2024-01-02", "A", "long", .8, 2),
                              row("2024-01-02", "B", "short", .9, 1),
                              row("2024-01-02", "C", score=.5, eligible=False)])
        display = latest_pool(scored)
        self.assertEqual(display.code.tolist(), ["B", "A", "C"])
        self.assertTrue({"side_score", "combo_rank", "pool_side"}.issubset(display.columns))
        self.assertTrue(latest_pool(scored.iloc[:0]).empty)

    def test_minimum_age_then_scheduled_replacement_preserves_retained_age(self):
        rows = []
        for i, day in enumerate(pd.bdate_range("2024-01-02", "2024-01-16")):
            rows += [row(day, "A", score=.5, rank=1 if i == 0 else 9),
                     row(day, "B", score=.9, rank=1, eligible=i > 0)]
        result = self.select(rows, max_hold=1)
        self.assertEqual(held(result, "2024-01-12"), {("A", "long")})
        self.assertEqual(held(result, "2024-01-15"), {("A", "long")})
        age = result.loc[result.date.eq(pd.Timestamp("2024-01-15")) & result.code.eq("A"), "hold_days_long"].iloc[0]
        self.assertEqual(age, 10)
        self.assertEqual(held(result, "2024-01-16"), {("B", "long")})

    def test_exact_gap_replaces_same_side_but_smaller_gap_and_opposite_side_do_not(self):
        rows = []
        for i, day in enumerate(pd.bdate_range("2024-01-02", "2024-01-19")):
            improved = .65 if day == pd.Timestamp("2024-01-19") else .649
            rows += [row(day, "A", score=.6, rank=1 if i == 0 else 9),
                     row(day, "B", score=improved, rank=2, eligible=i > 0),
                     row(day, "C", "short", .99, 1, eligible=i > 0)]
        result = self.select(rows, max_hold=1)
        self.assertEqual(held(result, "2024-01-16"), {("A", "long")})
        self.assertEqual(held(result, "2024-01-19"), {("B", "long")})

    def test_rank_eight_remains_protected_after_minimum_age(self):
        rows = []
        for i, day in enumerate(pd.bdate_range("2024-01-02", "2024-01-19")):
            rows += [row(day, "A", score=.5, rank=1 if i == 0 else 8),
                     row(day, "B", score=.99, rank=1, eligible=i > 0)]
        result = self.select(rows, max_hold=1)
        self.assertEqual(held(result, "2024-01-19"), {("A", "long")})

    def test_pool_exit_alone_does_not_liquidate_safe_incumbent(self):
        rows = [row("2024-01-02", "A", score=.5),
                row("2024-01-03", "A", score=.5, eligible=False),
                row("2024-01-05", "A", score=.5, eligible=False)]
        result = self.select(rows, max_hold=1, min_hold_days=1)
        self.assertEqual(held(result, "2024-01-05"), {("A", "long")})

    def test_missing_old_score_cannot_establish_a_replacement_gap(self):
        rows = [row("2024-01-02", "A", score=.5),
                row("2024-01-05", "A", score=np.nan, eligible=False),
                row("2024-01-05", "B", score=.99)]
        result = self.select(rows, max_hold=1, min_hold_days=1)
        self.assertEqual(held(result, "2024-01-05"), {("A", "long")})

    def test_missing_or_opposite_regime_exits_locked_holding_without_same_day_refill(self):
        for regime in [None, "short"]:
            with self.subTest(regime=regime):
                result = self.select([row("2024-01-02", "A"),
                                      row("2024-01-03", "A", regime=regime),
                                      row("2024-01-03", "B", score=.9),
                                      row("2024-01-05", "B", score=.9)], max_hold=1)
                self.assertEqual(held(result, "2024-01-03"), set())
                self.assertEqual(held(result, "2024-01-05"), {("B", "long")})

    def test_empty_stock_session_in_benchmark_calendar_clears_position_and_age(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        rows = [row("2024-01-02", "A"), row("2024-01-04", "A"), row("2024-01-05", "A")]
        result = apply_holding_hysteresis(pd.DataFrame(rows), settings(max_hold=1), trading_dates=dates)
        self.assertEqual(held(result, "2024-01-04"), set())
        age = result.loc[result.date.eq(pd.Timestamp("2024-01-05")), "hold_days_long"].iloc[0]
        self.assertEqual(age, 1)

    def test_vacancy_is_filled_before_challenging_an_existing_holding(self):
        result = self.select([row("2024-01-02", "A", score=.6),
                              row("2024-01-05", "A", score=.6, rank=9),
                              row("2024-01-05", "C", score=.9, rank=1),
                              row("2024-01-05", "D", score=.649, rank=2)],
                             max_hold=2, min_hold_days=1)
        self.assertEqual(held(result, "2024-01-05"), {("A", "long"), ("C", "long")})

    def test_replacement_evicts_weakest_unprotected_same_side_only(self):
        result = self.select([row("2024-01-02", "A", score=.5, rank=1),
                              row("2024-01-02", "B", score=.4, rank=2),
                              row("2024-01-05", "A", score=.5, rank=9),
                              row("2024-01-05", "B", score=.4, rank=10),
                              row("2024-01-05", "C", score=.55, rank=1)],
                             max_hold=2, min_hold_days=1)
        self.assertEqual(held(result, "2024-01-05"), {("A", "long"), ("C", "long")})

    def test_holiday_does_not_shift_review_to_previous_or_next_session(self):
        rows = []
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08", "2024-01-09"]
        for i, day in enumerate(dates):
            rows += [row(day, "A", score=.5, rank=1 if i == 0 else 9),
                     row(day, "B", score=.9, rank=1, eligible=i > 0)]
        result = self.select(rows, max_hold=1, min_hold_days=1)
        self.assertEqual(held(result, "2024-01-08"), {("A", "long")})
        self.assertEqual(held(result, "2024-01-09"), {("B", "long")})

    def test_capacity_reduction_overrides_locked_age_even_off_review(self):
        rows = [row(day, code, score=score, rank=rank)
                for day in ["2024-01-02", "2024-01-03"]
                for code, score, rank in [("A", .9, 1), ("B", .8, 2), ("C", .7, 3), ("D", .6, 4)]]
        result = apply_holding_hysteresis(pd.DataFrame(rows), settings(),
                                          max_hold_by_date={pd.Timestamp("2024-01-03"): 2})
        self.assertEqual(held(result, "2024-01-02"), {("A", "long"), ("B", "long"), ("C", "long"), ("D", "long")})
        self.assertEqual(held(result, "2024-01-03"), {("A", "long"), ("B", "long")})

    def test_future_rows_do_not_change_past_holdings(self):
        rows = [row("2024-01-02", "A"), row("2024-01-03", "A"),
                row("2024-01-05", "A", regime="short"), row("2024-01-05", "B", score=.99)]
        full = self.select(rows, max_hold=1)
        prefix = self.select(rows[:2], max_hold=1)
        columns = ["date", "code", "side", "in_hold", "hold_days_long", "hold_days_short"]
        pd.testing.assert_frame_equal(prefix[columns], full.iloc[:2][columns])


class ConfigurationTests(unittest.TestCase):
    def test_new_defaults_are_canonical(self):
        with patch("ndxbots.config._load_yaml", return_value={}):
            actual = load_settings()
        self.assertEqual(actual.rebalance_weekdays, (1, 4))
        self.assertEqual(actual.replacement_score_gap, .05)

    def test_invalid_weekdays_or_gap_are_rejected(self):
        for values in [{"rebalance_weekdays": [7]}, {"rebalance_weekdays": [True]},
                       {"rebalance_weekdays": [1.5]}, {"rebalance_weekdays": []},
                       {"replacement_score_gap": np.nan}, {"replacement_score_gap": np.inf},
                       {"replacement_score_gap": -0.01}, {"replacement_score_gap": 1.01}]:
            with self.subTest(values=values), patch("ndxbots.config._load_yaml", return_value={"strategy": values}):
                with self.assertRaises(ValueError):
                    load_settings()


if __name__ == "__main__":
    unittest.main()
