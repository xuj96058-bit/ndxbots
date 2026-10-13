import unittest

import numpy as np
import pandas as pd

from ndxbots.regime.leverage import _normalize, leverage_on_trading_days, yoy_on_trading_days


class LeverageTests(unittest.TestCase):
    def monthly(self):
        return pd.DataFrame({"month": pd.date_range("2019-01-01", periods=15, freq="MS").strftime("%Y-%m"),
                             "margin_debt": np.arange(100.0, 115.0)})

    def test_yoy_uses_calendar_month_despite_gap(self):
        table = self.monthly()
        table = table.loc[table.month.ne("2019-06")]
        out = yoy_on_trading_days(table, pd.DatetimeIndex(["2020-03-23"]), 21)
        self.assertAlmostEqual(out.iloc[0], 113 / 101 - 1)

    def test_actual_release_date_boundary_and_provenance(self):
        table = self.monthly()
        table.loc[table.month.eq("2020-01"), "available_on"] = "2020-02-25"
        out = leverage_on_trading_days(table, pd.DatetimeIndex(["2020-02-24", "2020-02-25"]), 21)
        self.assertTrue(pd.isna(out.leverage_yoy.iloc[0]))
        self.assertAlmostEqual(out.leverage_yoy.iloc[1], 0.12)
        self.assertEqual(out.leverage_availability_source.iloc[1], "provided_release_date")

    def test_stale_data_expires_and_keeps_metadata(self):
        table = self.monthly().iloc[:13]
        out = leverage_on_trading_days(table, pd.DatetimeIndex(["2020-02-24", "2020-05-04"]), 21, 90)
        self.assertFalse(out.leverage_stale.iloc[0])
        self.assertTrue(out.leverage_stale.iloc[1])
        self.assertTrue(pd.isna(out.leverage_yoy.iloc[1]))
        self.assertEqual(out.leverage_reference_month.iloc[1], "2020-01")
        self.assertEqual(out.leverage_age_days.iloc[1], 94)

    def test_missing_comparison_does_not_reuse_previous_yoy(self):
        table = self.monthly().loc[lambda x: x.month.ne("2019-02")]
        out = leverage_on_trading_days(table, pd.DatetimeIndex(["2020-02-24", "2020-03-23"]), 21)
        self.assertTrue(pd.notna(out.leverage_yoy.iloc[0]))
        self.assertTrue(pd.isna(out.leverage_yoy.iloc[1]))
        self.assertEqual(out.leverage_reference_month.iloc[1], "2020-02")

    def test_requested_date_order_is_preserved(self):
        dates = pd.DatetimeIndex(["2020-04-23", "2020-02-24", "2020-03-23"])
        out = leverage_on_trading_days(self.monthly(), dates, 21)
        self.assertTrue(out.index.equals(dates))
        np.testing.assert_allclose(out.leverage_yoy.to_numpy(), [114 / 102 - 1, .12, 113 / 101 - 1])

    def test_future_months_do_not_change_past(self):
        dates = pd.bdate_range("2020-02-01", "2020-03-20")
        full = leverage_on_trading_days(self.monthly(), dates, 21)
        limited = leverage_on_trading_days(self.monthly().iloc[:13], dates, 21)
        pd.testing.assert_frame_equal(full, limited)

    def test_empty_data_is_missing_not_stale(self):
        out = leverage_on_trading_days(pd.DataFrame(columns=["month", "margin_debt"]),
                                       pd.DatetimeIndex(["2025-01-02"]), 21)
        self.assertTrue(pd.isna(out.leverage_yoy.iloc[0]))
        self.assertFalse(out.leverage_stale.iloc[0])
        self.assertEqual(out.leverage_availability_source.iloc[0], "missing")

    def test_normalization_preserves_actual_dates_and_rejects_nonpositive_debt(self):
        table = pd.DataFrame({"month": ["2020-01", "2020-02"], "margin_debt": [100, 0],
                              "available_on": ["2020-02-25", "2020-03-25"]})
        out = _normalize(table)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.available_on.iloc[0], pd.Timestamp("2020-02-25"))

    def test_release_before_reference_month_is_rejected(self):
        table = self.monthly()
        table.loc[table.month.eq("2020-01"), "available_on"] = "2019-12-01"
        with self.assertRaisesRegex(ValueError, "發布日"):
            leverage_on_trading_days(table, pd.DatetimeIndex(["2020-02-24"]), 21)

    def test_release_dates_cannot_regress_to_older_reference_month(self):
        table = self.monthly()
        table.loc[table.month.eq("2020-01"), "available_on"] = "2020-05-01"
        with self.assertRaisesRegex(ValueError, "遞增"):
            leverage_on_trading_days(table, pd.DatetimeIndex(["2020-05-01"]), 21)

    def test_invalid_calendar_month_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "01 至 12"):
            _normalize(pd.DataFrame({"month": ["2025-13"], "margin_debt": [100]}))

    def test_invalid_supplied_date_is_not_silently_estimated(self):
        with self.assertRaisesRegex(ValueError, "有效發布日期"):
            _normalize(pd.DataFrame({"month": ["2025-01"], "margin_debt": [100], "available_on": ["bad"]}))


if __name__ == "__main__":
    unittest.main()
