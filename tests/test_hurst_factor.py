"""Regression coverage for the requested log-price H = 2 * slope convention."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ndxbots.factors.custom import _hurst_classic, my_factors


def quadratic_prices(periods=300):
    t = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {"US.A": np.exp(4.0 + .002 * t + .00001 * t * t),
         "US.B": np.exp(5.0 + .003 * t - .000015 * t * t)},
        index=pd.bdate_range("2023-01-02", periods=periods),
    )


def quadratic_log_price_reference(window=126):
    """Closed-form lag dispersion plus scalar OLS, without numerical fitting.

    For log(price[t]) = a + b*t + c*t**2, lag differences are an
    arithmetic progression with step 2*c*lag and window-lag terms.
    Its population standard deviation is proportional to
    lag * sqrt(((window-lag)**2 - 1) / 12). The constant factor
    changes only the regression intercept, so it can be omitted.
    """
    x = [math.log(lag) for lag in range(2, 21)]
    y = [math.log(lag * math.sqrt(((window - lag) ** 2 - 1) / 12))
         for lag in range(2, 21)]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = sum((a - mean_x) ** 2 for a in x)
    return 2 * numerator / denominator


class HurstFactorTests(unittest.TestCase):
    def test_public_factor_matches_analytic_log_price_reference(self):
        prices = quadratic_prices()
        factors = my_factors(prices, None, None, None, None, None)
        actual = factors["my_hurst_126"]
        pd.testing.assert_index_equal(actual.index, prices.index)
        pd.testing.assert_index_equal(actual.columns, prices.columns)
        expected = quadratic_log_price_reference()
        # This convention can exceed one; do not silently substitute another
        # Hurst convention or clip the result into [0, 1].
        self.assertGreater(expected, 1.8)
        np.testing.assert_allclose(actual.iloc[125:], expected, atol=1e-10, rtol=0)

    def test_exact_126_observation_warmup(self):
        prices = quadratic_prices(126)
        actual = _hurst_classic(prices)
        self.assertTrue(actual.iloc[:125].isna().all().all())
        self.assertTrue(np.isfinite(actual.iloc[125]).all())
        self.assertTrue(_hurst_classic(prices.iloc[:125]).isna().all().all())

    def test_bad_price_invalidates_its_entire_window_then_recovers(self):
        prices = quadratic_prices()
        baseline = _hurst_classic(prices)
        for bad_price in (np.nan, 0.0, -1.0, np.inf, -np.inf):
            with self.subTest(bad_price=bad_price):
                damaged = prices.copy()
                damaged.iloc[140, 0] = bad_price
                actual = _hurst_classic(damaged)
                self.assertTrue(actual["US.A"].iloc[140:266].isna().all())
                pd.testing.assert_series_equal(actual["US.A"].iloc[:140],
                                               baseline["US.A"].iloc[:140])
                pd.testing.assert_series_equal(actual["US.A"].iloc[266:],
                                               baseline["US.A"].iloc[266:])
                pd.testing.assert_series_equal(actual["US.B"], baseline["US.B"])

    def test_constant_prices_have_no_estimable_hurst(self):
        prices = quadratic_prices()
        prices.loc[:, :] = 100.0
        self.assertTrue(_hurst_classic(prices).isna().all().all())

    def test_appending_or_replacing_future_prices_never_changes_history(self):
        t = np.arange(280, dtype=float)
        prices = pd.DataFrame({"US.A": np.exp(4 + .003 * t + .03 * np.sin(t / 9))},
                              index=pd.bdate_range("2023-01-02", periods=len(t)))
        cutoff = 180
        prefix = my_factors(prices.iloc[:cutoff], None, None, None, None, None)["my_hurst_126"]
        full = my_factors(prices, None, None, None, None, None)["my_hurst_126"]
        pd.testing.assert_frame_equal(full.iloc[:cutoff], prefix)
        changed = prices.copy()
        changed.iloc[cutoff:, 0] = np.exp(8 + .01 * t[cutoff:] + np.sin(t[cutoff:]))
        altered = my_factors(changed, None, None, None, None, None)["my_hurst_126"]
        pd.testing.assert_frame_equal(altered.iloc[:cutoff], prefix)
        self.assertFalse(np.allclose(altered.iloc[cutoff:], full.iloc[cutoff:]))


if __name__ == "__main__":
    unittest.main()
