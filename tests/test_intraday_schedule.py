from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ndxbots.backtest.intraday_prices import (
    IntradayPriceStore, InvalidIntradayData, MissingIntradayData,
)
from ndxbots.backtest.intraday_schedule import (
    Policy, execution_schedule, policy_kind, prepare_bars,
)


ENTRY = Policy("first_group", delay=30, first_fraction=.5, gap_threshold=.01, remainder="atr")
EXIT = Policy("first_group_exit", delay=15, exit_mode="immediate")


def quotes(price=103.0, close_minute=960):
    price = float(price)
    return pd.DataFrame({"bar_end": np.arange(575, close_minute + 1, 5),
                         "open": price, "high": price + 1, "low": price - .5,
                         "close": price, "volume": 100.0, "turnover": price * 100})


def native(day="2024-01-03", close_minute=960):
    frame = quotes(100.0, close_minute)
    stamps = pd.Timestamp(day, tz="America/New_York") + pd.to_timedelta(frame.bar_end, unit="min")
    frame["bar_end"] = stamps
    frame["time_key"] = stamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["date"] = pd.Timestamp(day)
    frame["code"] = "US.AAPL"
    return frame


class ScheduleTests(unittest.TestCase):
    def entry(self, frame, side=1, atr=10.0, prev_close=100.0):
        result = execution_schedule(frame, side, ENTRY, prev_close, atr)
        self.assertAlmostEqual(sum(row["fraction"] for row in result), 1.0)
        return result

    def test_gap_equal_one_percent_completes_at_ten_and_needs_no_atr(self):
        result = self.entry(quotes(101), atr=np.nan)
        self.assertEqual([(row["time"], row["fraction"]) for row in result], [(600, 1.0)])
        result = self.entry(quotes(101.00001))
        self.assertEqual(result[0]["fraction"], .5)

    def test_favorable_gap_completes_at_ten_in_both_directions(self):
        for side, price in [(1, 97), (-1, 103)]:
            result = self.entry(quotes(price), side=side)
            self.assertEqual([(row["time"], row["fraction"]) for row in result], [(600, 1.0)])

    def test_low_before_ten_cannot_fill_remainder(self):
        frame = quotes()
        frame.loc[frame.bar_end <= 600, "low"] = 90
        result = self.entry(frame)
        self.assertEqual([row["trigger"] for row in result], ["initial_entry", "deadline_entry"])
        self.assertEqual(result[-1]["time"], 955)
        self.assertTrue(result[-1]["forced"])

    def test_atr_anchor_is_session_open_and_signal_atr(self):
        frame = quotes()
        frame.loc[frame.bar_end.eq(605), ["open", "high", "low", "close"]] = [104, 105, 100.9, 104]
        result = self.entry(frame, atr=10)
        self.assertEqual(result[-1]["price"], 101)
        self.assertEqual(result[-1]["time"], 605)
        self.assertEqual(result[-1]["active_from"], 600)
        self.assertFalse(result[-1]["forced"])
        self.assertEqual(self.entry(frame, atr=5)[-1]["price"], 102)

    def test_limit_is_active_in_first_ten_oclock_bar(self):
        frame = quotes()
        frame.loc[frame.bar_end.eq(605), "low"] = 100.9
        result = self.entry(frame)
        self.assertEqual([row["time"] for row in result], [600, 605])
        self.assertEqual(result[-1]["trigger"], "atr_limit")
        self.assertEqual(result[-1]["active_from"], 600)

    def test_touch_or_exact_one_bp_trade_through_is_insufficient(self):
        for low in [101.0, 101 * .9999]:
            frame = quotes()
            frame.loc[frame.bar_end.eq(605), "low"] = low
            self.assertEqual(self.entry(frame)[-1]["trigger"], "deadline_entry")
        frame.loc[frame.bar_end.eq(605), "low"] = 101 * .9999 - .00001
        self.assertEqual(self.entry(frame)[-1]["trigger"], "atr_limit")

    def test_opened_through_limit_uses_better_open_price(self):
        frame = quotes()
        frame.loc[frame.bar_end.eq(610), ["open", "high", "low", "close"]] = [100, 101, 99, 100]
        result = self.entry(frame)
        self.assertEqual((result[-1]["time"], result[-1]["active_from"], result[-1]["price"]),
                         (605, 605, 100))

    def test_short_entry_reverses_limit_direction_and_strict_boundary(self):
        frame = quotes(97)
        frame.loc[frame.bar_end.eq(605), "high"] = 99 * 1.0001
        self.assertEqual(self.entry(frame, side=-1)[-1]["trigger"], "deadline_entry")
        frame.loc[frame.bar_end.eq(605), "high"] += .00001
        result = self.entry(frame, side=-1)
        self.assertEqual((result[-1]["price"], result[-1]["time"]), (99, 605))

    def test_deadline_is_last_open_not_last_close_for_full_and_half_sessions(self):
        for close_minute in [960, 780]:
            frame = quotes(close_minute=close_minute)
            frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [110, 121, 100, 120]
            result = self.entry(frame)
            self.assertEqual((result[-1]["time"], result[-1]["price"]), (close_minute - 5, 110))
            self.assertTrue(result[-1]["forced"])

    def test_ordinary_and_urgent_exit_complete_required_quantity(self):
        for side in [-1, 1]:
            for urgent, time in [(False, 585), (True, 570)]:
                result = execution_schedule(quotes(), side, EXIT, np.nan, np.nan,
                                            action="exit", urgent=urgent)
                self.assertEqual(len(result), 1)
                self.assertEqual((result[0]["time"], result[0]["fraction"]), (time, 1))

    def test_utc_bar_end_converts_to_exchange_time_across_dst(self):
        for day in ["2024-01-03", "2024-07-03"]:
            frame = quotes(close_minute=780 if day.endswith("07-03") else 960)
            stamps = pd.Timestamp(day, tz="America/New_York") + pd.to_timedelta(frame.bar_end, unit="min")
            frame["bar_end"] = stamps.dt.tz_convert("UTC")
            self.assertEqual(self.entry(frame)[0]["time"], 600)

    def test_missing_duplicate_mixed_dates_and_subminute_bars_rejected(self):
        bad = [quotes().drop(2), pd.concat([quotes(), quotes().iloc[[2]]])]
        frame = native()
        shifted = frame.copy()
        shifted.loc[0, "bar_end"] += pd.Timedelta(days=1)
        bad.append(shifted)
        shifted = frame.copy()
        shifted.loc[0, "bar_end"] += pd.Timedelta(seconds=1)
        bad.append(shifted)
        for frame in bad:
            with self.assertRaises(ValueError):
                prepare_bars(frame)

    def test_prepared_arrays_are_read_only_and_invalid_atr_does_not_fallback(self):
        prepared = prepare_bars(quotes())
        self.assertIs(prepare_bars(prepared), prepared)
        with self.assertRaises(ValueError):
            prepared.opens[0] = 1
        with self.assertRaisesRegex(ValueError, "ATR20"):
            self.entry(prepared, atr=np.nan)
        self.assertEqual(policy_kind(ENTRY), "intraday")


class PriceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name) / "intraday" / "kline"
        self.root.mkdir(parents=True)
        self.calendar = pd.DataFrame({"date": [pd.Timestamp("2024-01-03")], "close_minute": [960]})
        self.calendar.to_parquet(self.root.parent / "calendar.parquet", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, frame):
        frame.to_parquet(self.root / "US_AAPL.parquet", index=False)

    def test_scale_uses_known_daily_open_and_never_last_close(self):
        frame = native()
        self.write(frame)
        first = IntradayPriceStore(self.root).bars("US.AAPL", "2024-01-03", 120)
        self.assertAlmostEqual(first.opens[0], 120)
        frame.loc[frame.index[-1], ["high", "close"]] = [151, 150]
        self.write(frame)
        changed = IntradayPriceStore(self.root).bars("US.AAPL", "2024-01-03", 120)
        np.testing.assert_array_equal(first.opens, changed.opens)
        np.testing.assert_array_equal(first.closes[:-1], changed.closes[:-1])
        self.assertAlmostEqual(changed.closes[-1], 180)

    def test_truncated_normal_day_is_not_inferred_as_half_day(self):
        self.write(native(close_minute=780))
        with self.assertRaisesRegex(InvalidIntradayData, "78 continuous"):
            IntradayPriceStore(self.root).bars("US.AAPL", "2024-01-03", 100)
        self.calendar["close_minute"] = 780
        self.calendar.to_parquet(self.root.parent / "calendar.parquet", index=False)
        prepared = IntradayPriceStore(self.root).bars("US.AAPL", "2024-01-03", 100)
        self.assertEqual(prepared.ends[-1], 780)

    def test_zero_volume_first_bar_is_rejected(self):
        frame = native()
        frame.loc[0, "volume"] = 0
        self.write(frame)
        with self.assertRaisesRegex(InvalidIntradayData, "first 09:30-09:35"):
            IntradayPriceStore(self.root).bars("US.AAPL", "2024-01-03", 100)

    def test_missing_calendar_cache_or_date_never_uses_daily_quote(self):
        store = IntradayPriceStore(self.root)
        with self.assertRaises(MissingIntradayData):
            store.bars("US.AAPL", "2024-01-03", 100)
        self.write(native())
        with self.assertRaises(MissingIntradayData):
            store.bars("US.AAPL", "2024-01-04", 100)
        (self.root.parent / "calendar.parquet").unlink()
        with self.assertRaisesRegex(MissingIntradayData, "calendar"):
            IntradayPriceStore(self.root)

    def test_native_utc_timestamps_and_cache_preserve_read_only_units(self):
        frame = native()
        frame["bar_end"] = frame.bar_end.dt.tz_convert("UTC")
        self.write(frame)
        store = IntradayPriceStore(self.root)
        prepared = store.bars("US.AAPL", "2024-01-03", 120)
        self.assertIs(store.bars("US.AAPL", "2024-01-03", 120), prepared)
        self.assertEqual(prepared.starts[0], 570)
        self.assertEqual(store.bars("US.AAPL", "2024-01-03", 100).opens[0], 100)
        with self.assertRaises(ValueError):
            prepared.closes[-1] = 1

    def test_bad_ohlc_volume_and_missing_five_minute_bar_rejected(self):
        frames = [native().drop(3)]
        bad = native()
        bad.loc[4, "close"] = 1000
        frames.append(bad)
        bad = native()
        bad.loc[4, "volume"] = -1
        frames.append(bad)
        for frame in frames:
            self.write(frame)
            with self.assertRaises(InvalidIntradayData):
                IntradayPriceStore(self.root).bars("US.AAPL", "2024-01-03", 100)


if __name__ == "__main__":
    unittest.main()
