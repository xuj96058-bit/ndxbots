from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
with patch("dotenv.load_dotenv", return_value=False):
    from ndxbots.factors.price import rsi
    from ndxbots.regime.stats import classify_state, rolling_percentile


class RegimeStatsTests(unittest.TestCase):
    def test_double_extreme_label_requires_both_hot_thresholds(self):
        self.assertEqual(classify_state(0.5, 0.5, 0.8, 0.2, 0.5), "情绪/拥挤偏高")
        self.assertEqual(classify_state(0.79, 0.79, 0.8, 0.2, 0.5), "情绪/拥挤偏高")
        self.assertNotIn("双重极端", classify_state(0.8, 0.79, 0.8, 0.2, 0.5))
        self.assertTrue(classify_state(0.8, 0.8, 0.8, 0.2, 0.5).startswith("双重极端"))
        self.assertEqual(classify_state(np.nan, 0.8, 0.8, 0.2, 0.5), "数据不足")

    def test_constant_negative_series_is_neutral_and_ties_use_midrank(self):
        result = rolling_percentile(pd.Series([-0.5] * 300), 252, 252)
        self.assertTrue(result.iloc[:251].isna().all())
        self.assertTrue(result.iloc[251:].eq(0.5).all())
        tied = rolling_percentile(pd.Series([1.0, 2.0, 2.0, 3.0, 2.0]), 5, 5)
        self.assertAlmostEqual(tied.iloc[-1], (1 + 0.5 * 3) / 5)

    def test_missing_current_and_minimum_history_remain_missing(self):
        result = rolling_percentile(pd.Series([1.0, np.nan, 2.0, np.nan]), 3, 2)
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertEqual(result.iloc[2], 0.75)
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_percentile_has_no_future_dependency(self):
        series = pd.Series(np.sin(np.arange(80) / 4), index=pd.bdate_range("2020-01-01", periods=80))
        changed = series.copy()
        changed.iloc[50:] = 1e6
        baseline = rolling_percentile(series, 20, 10)
        assert_series_equal(baseline.iloc[:50], rolling_percentile(changed, 20, 10).iloc[:50])
        assert_series_equal(baseline.iloc[:50], rolling_percentile(series.iloc[:50], 20, 10))

    def test_rsi_rising_falling_flat_and_warmup(self):
        close = pd.DataFrame({"UP": 100 + np.arange(40), "DOWN": 200 - np.arange(40), "FLAT": np.full(40, 100.0)})
        result = rsi(close, 14)
        self.assertTrue(result.iloc[:14].isna().all().all())
        self.assertTrue(result.UP.iloc[14:].eq(100).all())
        self.assertTrue(result.DOWN.iloc[14:].eq(0).all())
        self.assertTrue(result.FLAT.iloc[14:].eq(50).all())

    def test_rsi_future_changes_leave_history_unchanged(self):
        close = pd.DataFrame({"A": 100 + np.arange(80) / 2 + np.sin(np.arange(80))})
        changed = close.copy()
        changed.iloc[50:] *= 20
        assert_frame_equal(rsi(close, 14).iloc[:50], rsi(changed, 14).iloc[:50])
        assert_frame_equal(rsi(close, 14).iloc[:50], rsi(close.iloc[:50], 14))


if __name__ == "__main__":
    unittest.main()
