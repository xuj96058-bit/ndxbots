from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.data.intraday import (
    IntradayDataError, calendar_path, collect_intraday, kline_path,
    load_intraday, normalize_calendar, normalize_intraday, required_intraday_days,
)

API = SimpleNamespace(RET_OK=0, KLType=SimpleNamespace(K_5M="K_5M"),
                      AuType=SimpleNamespace(NONE="NONE"), Session=SimpleNamespace(RTH="RTH"),
                      TradeDateMarket=SimpleNamespace(US="US"))


def bars(day, close=960, *, utc=False):
    start = pd.Timestamp(day).tz_localize("America/New_York")
    stamps = [start + pd.Timedelta(minutes=minute) for minute in range(575, close + 1, 5)]
    time_key = pd.DatetimeIndex(stamps).tz_convert("UTC") if utc else [s.strftime("%Y-%m-%d %H:%M:%S") for s in stamps]
    return pd.DataFrame({"code": "US.AAPL", "time_key": time_key, "open": 100., "high": 101.,
                         "low": 99., "close": 100.5, "volume": 200., "turnover": 20050.})


class FakeQuote:
    def __init__(self, days, pages):
        self.days = days
        self.pages = list(pages)
        self.calls = []
        self.calendar_calls = 0

    def request_trading_days(self, **kwargs):
        self.calendar_calls += 1
        return 0, self.days

    def request_history_kline(self, code, **kwargs):
        self.calls.append((code, kwargs))
        if not self.pages:
            raise AssertionError("Unexpected extra page request")
        return self.pages.pop(0)


class IntradayIngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(load_settings(), data_root=root, intraday_root=root / "intraday" / "kline", request_sleep_sec=0)
        self.days = [{"time": "2024-11-27", "trade_date_type": "WHOLE"},
                     {"time": "2024-11-29", "trade_date_type": "MORNING"}]
        self.calendar = normalize_calendar(self.days)

    def test_pagination_halfday_cache_and_raw_units(self):
        full = pd.concat([bars("2024-11-27"), bars("2024-11-29", 780)], ignore_index=True)
        quote = FakeQuote(self.days, [(0, full.iloc[:70], b"page1"), (0, full.iloc[65:], None)])
        result = collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-29", quote_context=quote, api=API)
        self.assertEqual(result, {"US.AAPL": 120})
        self.assertEqual(quote.calls[1][1]["page_req_key"], b"page1")
        for _, request in quote.calls:
            self.assertEqual(request["ktype"], "K_5M")
            self.assertEqual(request["autype"], "NONE")
            self.assertEqual(request["session"], "RTH")
        cached = load_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-29")
        self.assertEqual(str(cached.bar_end.dt.tz), "America/New_York")
        self.assertTrue(cached.open.eq(100).all())
        self.assertTrue(cached.volume.eq(200).all())
        self.assertEqual(cached.loc[cached.date.eq(pd.Timestamp("2024-11-29")), "bar_end"].max().hour, 13)
        offline = FakeQuote(self.days, [])
        self.assertEqual(collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-29", quote_context=offline, api=API), result)
        self.assertEqual(offline.calendar_calls, 0)
        self.assertEqual(len(offline.calls), 0)

    def test_utc_timestamps_follow_new_york_dst(self):
        days = normalize_calendar([{"time": day, "trade_date_type": "WHOLE"} for day in ("2024-03-08", "2024-03-11")])
        frame = pd.concat([bars("2024-03-08", utc=True), bars("2024-03-11", utc=True)], ignore_index=True)
        parsed = normalize_intraday(frame, "US.AAPL", days)
        self.assertTrue((parsed.groupby("date").bar_end.first().dt.hour == 9).all())
        self.assertEqual(parsed.bar_end.iloc[0].utcoffset().total_seconds(), -5 * 3600)
        self.assertEqual(parsed.bar_end.iloc[-1].utcoffset().total_seconds(), -4 * 3600)

    def test_truncated_normal_day_is_not_inferred_halfday(self):
        with self.assertRaisesRegex(IntradayDataError, "expected 78, got 42"):
            normalize_intraday(bars("2024-11-27", 780), "US.AAPL", self.calendar)

    def test_missing_first_bar_rejected(self):
        with self.assertRaisesRegex(IntradayDataError, "incomplete"):
            normalize_intraday(bars("2024-11-27").iloc[1:], "US.AAPL", self.calendar)

    def test_error_after_page_preserves_existing_cache(self):
        first = FakeQuote(self.days, [(0, bars("2024-11-27"), None)])
        collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-27", quote_context=first, api=API)
        cache = kline_path(self.settings, "US.AAPL")
        before = hashlib.sha256(cache.read_bytes()).hexdigest()
        failing = FakeQuote(self.days, [(0, bars("2024-11-29", 780).iloc[:10], b"next"), (1, "permission denied", None)])
        with self.assertRaisesRegex(IntradayDataError, "permission denied"):
            collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-29", quote_context=failing, api=API)
        self.assertEqual(before, hashlib.sha256(cache.read_bytes()).hexdigest())

    def test_repeated_page_key_fails_without_stock_cache(self):
        quote = FakeQuote(self.days, [(0, bars("2024-11-27").iloc[:20], b"same"), (0, bars("2024-11-27").iloc[20:40], b"same")])
        with self.assertRaisesRegex(IntradayDataError, "repeated history pagination"):
            collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-27", quote_context=quote, api=API)
        self.assertFalse(kline_path(self.settings, "US.AAPL").exists())

    def test_empty_source_and_missing_offline_cache_fail_clearly(self):
        with self.assertRaisesRegex(IntradayDataError, "calendar absent"):
            load_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-29")
        quote = FakeQuote(self.days, [(0, pd.DataFrame(), None)])
        with self.assertRaisesRegex(IntradayDataError, "absent minute history"):
            collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-27", quote_context=quote, api=API)
        self.assertFalse(kline_path(self.settings, "US.AAPL").exists())

    def test_outside_rth_is_excluded_and_bad_prices_rejected(self):
        full = bars("2024-11-27")
        extended = full.iloc[[0]].copy()
        extended["time_key"] = "2024-11-27 08:00:00"
        result = normalize_intraday(pd.concat([extended, full]), "US.AAPL", self.calendar)
        self.assertEqual(len(result), 78)
        full.loc[0, "high"] = 90
        with self.assertRaisesRegex(IntradayDataError, "invalid raw OHLC"):
            normalize_intraday(full, "US.AAPL", self.calendar)

    def test_calendar_uses_daytype_and_rejects_unknown_types(self):
        self.assertEqual(list(self.calendar.close_minute), [960, 780])
        with self.assertRaisesRegex(IntradayDataError, "Unsupported"):
            normalize_calendar([{"time": "2024-11-29", "trade_date_type": "AFTERNOON"}])

    def test_conflicting_timestamp_and_date_cache_fields_fail(self):
        frame = normalize_intraday(bars("2024-11-27"), "US.AAPL", self.calendar)
        frame.loc[0, "time_key"] = "2024-11-27 09:40:00"
        with self.assertRaisesRegex(IntradayDataError, "time_key disagrees"):
            normalize_intraday(frame, "US.AAPL", self.calendar)
        frame = normalize_intraday(bars("2024-11-27"), "US.AAPL", self.calendar)
        frame.loc[0, "date"] = pd.Timestamp("2024-11-29")
        with self.assertRaisesRegex(IntradayDataError, "stored date disagrees"):
            normalize_intraday(frame, "US.AAPL", self.calendar)

    def test_unneeded_incomplete_source_day_does_not_block_needed_day(self):
        unneeded = bars("2024-11-27").iloc[:3]
        full = pd.concat([unneeded, bars("2024-11-29", 780)], ignore_index=True)
        quote = FakeQuote(self.days, [(0, full, None)])
        plan = pd.DataFrame({"code": ["US.AAPL"], "date": [pd.Timestamp("2024-11-29")]})
        result = collect_intraday(self.settings, ["US.AAPL"], "2024-11-27", "2024-11-29", required_days=plan, quote_context=quote, api=API)
        self.assertEqual(result, {"US.AAPL": 42})
        cached = pd.read_parquet(kline_path(self.settings, "US.AAPL"))
        self.assertTrue(cached.date.eq(pd.Timestamp("2024-11-29")).all())

    def test_pure_signal_plan_includes_next_sessions_without_build(self):
        factor_file = self.settings.data_root / "factors" / "daily.parquet"
        panel_file = self.settings.data_root / "panel" / "daily.parquet"
        factor_file.parent.mkdir(parents=True)
        panel_file.parent.mkdir(parents=True)
        pd.DataFrame({"placeholder": [1]}).to_parquet(factor_file, index=False)
        pd.DataFrame({"code": ["US.QQQ"] * 4, "date": pd.to_datetime(["2024-11-26", "2024-11-27", "2024-11-29", "2024-12-02"])}).to_parquet(panel_file, index=False)
        scored = pd.DataFrame({"code": ["US.AAPL", "US.MSFT"], "date": pd.to_datetime(["2024-11-26", "2024-11-27"]), "in_hold": [True, False]})
        with patch("ndxbots.strategy.scores.build_score_table", return_value=scored):
            plan = required_intraday_days(self.settings, "2024-11-27", "2024-12-02")
        self.assertEqual(plan.code.unique().tolist(), ["US.AAPL"])
        self.assertEqual(plan.date.tolist(), list(pd.to_datetime(["2024-11-27", "2024-11-29"])))


if __name__ == "__main__":
    unittest.main()
