"""Hurst controls new long admission without changing pool or incumbent rules."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ndxbots.config import load_settings
from ndxbots.strategy.scores import apply_holding_hysteresis, build_score_table


def selection_settings(**changes):
    values = dict(max_hold=1, min_hold_days=1, keep_rank=0,
                  strategy_factors=("f1",), strategy_top_n=10,
                  rebalance_weekdays=(1, 4), replacement_score_gap=.05)
    values.update(changes)
    return SimpleNamespace(**values)


def candidate(day, code, hurst, *, side="long", score=.8, rank=1):
    long_score = score if side == "long" else 1 - score
    return dict(date=pd.Timestamp(day), code=code, regime_side=side,
                pool_side=side, side_score=score, combo_rank=rank,
                in_pool=True, in_pool_long=side == "long", in_pool_short=side == "short",
                score=long_score, long_score=long_score, short_score=1 - long_score,
                side=side, f1=score, my_hurst_126=hurst)


class HurstAdmissionTests(unittest.TestCase):
    def select(self, rows, **changes):
        return apply_holding_hysteresis(pd.DataFrame(rows), selection_settings(**changes))

    def test_strict_threshold_and_missing_values_block_only_new_longs(self):
        for hurst in (.4, .67, np.nan, np.inf, -np.inf):
            with self.subTest(hurst=hurst):
                long = self.select([candidate("2024-01-02", "A", hurst)])
                short = self.select([candidate("2024-01-02", "A", hurst, side="short")])
                self.assertTrue(long.in_pool.iloc[0])
                self.assertFalse(long.in_hold.iloc[0])
                self.assertTrue(short.in_hold.iloc[0])
        allowed = self.select([candidate("2024-01-02", "A", .670001)])
        self.assertTrue(allowed.in_hold_long.iloc[0])

    def test_absent_hurst_column_preserves_legacy_selection(self):
        frame = pd.DataFrame([candidate("2024-01-02", "A", np.nan)]).drop(columns="my_hurst_126")
        result = apply_holding_hysteresis(frame, selection_settings())
        self.assertTrue(result.in_hold_long.iloc[0])

    def test_low_hurst_challenger_cannot_replace_and_high_hurst_can(self):
        result = self.select([
            candidate("2024-01-02", "A", .8, score=.5),
            candidate("2024-01-05", "A", .2, score=.5, rank=2),
            candidate("2024-01-05", "B", .67, score=.9),
            candidate("2024-01-09", "A", np.nan, score=.5, rank=2),
            candidate("2024-01-09", "B", .8, score=.9),
        ])
        selected = result[result.in_hold].set_index("date").code.to_dict()
        self.assertEqual(selected, {pd.Timestamp("2024-01-02"): "A",
                                    pd.Timestamp("2024-01-05"): "A",
                                    pd.Timestamp("2024-01-09"): "B"})

    def test_blocked_rank_one_does_not_promote_names_outside_pool(self):
        rows = [candidate("2024-01-02", "A", .4, rank=1),
                candidate("2024-01-02", "B", .9, rank=2)]
        result = self.select(rows, strategy_top_n=1)
        self.assertFalse(result.in_hold.any())
        self.assertEqual(result.combo_rank.tolist(), [1, 2])


class HurstScoreIntegrationTests(unittest.TestCase):
    def settings(self, **changes):
        with patch("ndxbots.config._load_yaml", return_value={}):
            original = load_settings()
        values = dict(strategy_factors=("f1",), strategy_quality_weight=0,
                      strategy_top_n=10, qqq_regime_gate=False,
                      require_above_ma200=False, allow_short=True,
                      space_col="", short_space_col="", min_atr_pct=0,
                      rsi_gate=False, macd_gate=False, bb_width_gate=False,
                      max_hold=1, min_hold_days=1, keep_rank=8)
        values.update(changes)
        return replace(original, **values)

    def test_repeated_long_regime_cannot_bypass_gate_after_first_day(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-09"]),
                              "code": "A", "f1": 1., "my_hurst_126": [.4, .4, .8]})
        result = build_score_table(frame, self.settings())
        self.assertTrue(result.in_pool.all())
        self.assertEqual(result.combo_rank.tolist(), [1., 1., 1.])
        self.assertEqual(result.in_hold_long.tolist(), [False, False, True])

    def test_incumbent_low_hurst_preserves_rank_protection_and_pool_diagnostics(self):
        frame = pd.DataFrame([
            dict(date=pd.Timestamp(day), code=code, f1=score, my_hurst_126=hurst,
                 my_ma200_slope_atr=2., my_ma200_gap=.1)
            for day, code, score, hurst in [
                ("2024-01-02", "A", 2., .8), ("2024-01-02", "B", 1., .8),
                ("2024-01-05", "A", 1., .3), ("2024-01-05", "B", 2., .8)]])
        result = build_score_table(frame, self.settings())
        baseline = build_score_table(frame.assign(my_hurst_126=.8), self.settings())
        diagnostic_columns = ["side", "eligible", "score", "long_score", "short_score",
                              "pool_side", "side_score", "combo_rank", "in_pool",
                              "in_pool_long", "in_pool_short"]
        pd.testing.assert_frame_equal(result[diagnostic_columns], baseline[diagnostic_columns])
        selected = result.loc[result.in_hold, "code"].tolist()
        self.assertEqual(selected, ["A", "A"])
        self.assertEqual(result.loc[result.in_hold, "hold_days_long"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
