from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Factor tests never need to read a local .env or configuration.
with patch("dotenv.load_dotenv", return_value=False):
    from ndxbots.factors.custom import _efficiency_21, my_factors


def prices(values: dict[str, np.ndarray]) -> pd.DataFrame:
    count = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.bdate_range("2020-01-01", periods=count), dtype=float)


class QualityFactorTests(unittest.TestCase):
    def test_signed_extremes_flat_path_and_exact_warmup(self):
        close = prices({"UP": 100 + np.arange(35), "DOWN": 200 - np.arange(35),
                        "FLAT": np.full(35, 100)})
        quality = _efficiency_21(close)
        self.assertTrue(quality.iloc[:21].isna().all().all())
        self.assertTrue(quality.UP.iloc[21:].eq(1.0).all())
        self.assertTrue(quality.DOWN.iloc[21:].eq(-1.0).all())
        self.assertTrue(quality.FLAT.iloc[21:].eq(0.0).all())
        self.assertTrue(quality.index.equals(close.index))
        self.assertTrue(quality.columns.equals(close.columns))

    def test_same_endpoint_noisy_path_has_lower_quality(self):
        # Both move from 100 to 112 over 21 changes. The noisy path travels
        # 32 price units, so its hand-calculated quality is 12/32, not 1.
        changes = np.array([2.0, -1.0] * 10 + [2.0])
        close = prices({"NOISY": 100 + np.r_[0, changes.cumsum()],
                        "SMOOTH": 100 + np.linspace(0, changes.sum(), 22)})
        quality = _efficiency_21(close)
        self.assertAlmostEqual(quality.NOISY.iloc[-1], 12.0 / 32.0)
        self.assertAlmostEqual(quality.SMOOTH.iloc[-1], 1.0)
        self.assertLess(quality.NOISY.iloc[-1], quality.SMOOTH.iloc[-1])

    def test_missing_or_invalid_close_requires_complete_path_to_recover(self):
        for bad in (np.nan, np.inf, 0.0, -1.0):
            with self.subTest(invalid_close=bad):
                close = prices({"A": 100 + np.arange(45)})
                close.iloc[10, 0] = bad
                quality = _efficiency_21(close)
                # The missing close invalidates both changes at indices10/11.
                self.assertTrue(quality.iloc[21:32, 0].isna().all())
                self.assertEqual(quality.iloc[32, 0], 1.0)
                self.assertFalse(np.isinf(quality.to_numpy()).any())

    def test_future_prices_and_appended_rows_do_not_change_past(self):
        rng = np.random.default_rng(21)
        close = prices({"A": 100 + rng.normal(.2, 1, 90).cumsum(),
                        "B": 150 + rng.normal(-.1, 2, 90).cumsum()})
        baseline = _efficiency_21(close)
        changed = close.copy()
        changed.iloc[60:] *= 17
        assert_frame_equal(baseline.iloc[:60], _efficiency_21(changed).iloc[:60], check_exact=True)
        assert_frame_equal(baseline.iloc[:60], _efficiency_21(close.iloc[:60]), check_exact=True)

    def test_scale_invariance_and_bounds(self):
        t = np.arange(90)
        close = prices({"A": 100 + .4 * t + 4 * np.sin(t / 3),
                        "B": 200 - .2 * t + 7 * np.sin(t / 5)})
        quality = _efficiency_21(close)
        assert_frame_equal(quality, _efficiency_21(close * 13),
                           check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertTrue(quality.stack().dropna().between(-1, 1).all())

    def test_factor_output_is_wired_without_mutating_input(self):
        close = prices({"A": 100 + np.arange(60)})
        original = close.copy(deep=True)
        factors = my_factors(close, None, None, None, None, None)
        self.assertIn("my_efficiency_21", factors)
        assert_frame_equal(factors["my_efficiency_21"], _efficiency_21(close), check_exact=True)
        assert_frame_equal(close, original, check_exact=True)


if __name__ == "__main__":
    unittest.main()
