"""Strict local native five-minute data, normalized using the known daily open.

Input files contain Futu AuType.NONE regular-session OHLC and END timestamps.
No network fetch, daily-price substitute, forward fill, or closing-price scale
is used. Calendar metadata is required to distinguish a true half day from a
truncated ordinary session.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ndxbots.backtest.intraday_schedule import PreparedBars, prepare_bars


class IntradayDataError(ValueError):
    """Required local intraday inputs are unavailable or invalid."""


class MissingIntradayData(IntradayDataError):
    pass


class InvalidIntradayData(IntradayDataError):
    pass


def _date(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise InvalidIntradayData("Session date is missing")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    return stamp.normalize()


def _timestamps(values) -> pd.Series:
    stamps = pd.to_datetime(values, errors="raise")
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
    else:
        stamps = stamps.dt.tz_convert("America/New_York")
    if stamps.isna().any() or not stamps.eq(stamps.dt.floor("min")).all():
        raise ValueError("Bar END timestamps must be whole exchange minutes")
    return stamps


class IntradayPriceStore:
    """Lazy native cache reader; ``bars(code, day, daily_open)`` is read-only.

    Root normally resolves to settings.data_root / 'intraday' / 'kline'. A
    per-code file is US_AAPL.parquet; calendar defaults to its parent's
    calendar.parquet with naive ``date`` and ``close_minute`` 780 or 960.
    """
    def __init__(self, root: str | Path, calendar: pd.DataFrame | str | Path | None = None):
        self.root = Path(root)
        self.calendar_path = self.root.parent / "calendar.parquet"
        if calendar is None:
            calendar = self.calendar_path
        if not isinstance(calendar, pd.DataFrame):
            path = Path(calendar)
            if not path.is_file():
                raise MissingIntradayData(f"Missing session calendar: {path}. Collect data.intraday first.")
            try:
                calendar = pd.read_parquet(path)
            except Exception as exc:
                raise InvalidIntradayData(f"Cannot read session calendar {path}: {exc}") from exc
        if not {"date", "close_minute"}.issubset(calendar.columns):
            raise InvalidIntradayData("Calendar requires date and close_minute columns")
        try:
            dates = calendar.date.map(_date)
            closes = pd.to_numeric(calendar.close_minute, errors="raise")
        except Exception as exc:
            raise InvalidIntradayData(f"Invalid session calendar: {exc}") from exc
        if dates.duplicated().any() or not closes.isin([780, 960]).all():
            raise InvalidIntradayData("Calendar dates must be unique and close_minute must be 780/960")
        self.calendar = dict(zip(dates, closes.astype(int)))
        self._native: dict[str, dict[pd.Timestamp, pd.DataFrame]] = {}
        self._prepared: dict[tuple[str, pd.Timestamp, float], PreparedBars] = {}

    def _load(self, code: str) -> dict[pd.Timestamp, pd.DataFrame]:
        if code in self._native:
            return self._native[code]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", code):
            raise InvalidIntradayData(f"Invalid instrument code: {code!r}")
        path = self.root / f"{code.replace('.', '_')}.parquet"
        if not path.is_file():
            raise MissingIntradayData(f"Missing native K5 cache for {code}: {path}. Collect data.intraday first.")
        try:
            frame = pd.read_parquet(path)
            required = {"open", "high", "low", "close", "volume", "turnover"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing native columns: {sorted(required - set(frame.columns))}")
            stamp_column = "bar_end" if "bar_end" in frame else "time_key"
            if stamp_column not in frame or pd.api.types.is_numeric_dtype(frame[stamp_column]):
                raise ValueError("Native data requires END timestamps, not numeric minutes")
            frame = frame.copy()
            stamps = _timestamps(frame[stamp_column])
            if "time_key" in frame and stamp_column != "time_key":
                if not _timestamps(frame.time_key).eq(stamps).all():
                    raise ValueError("time_key and bar_end disagree")
            if "code" in frame and not frame.code.eq(code).all():
                raise ValueError("Instrument column disagrees with cache filename")
            dates = stamps.dt.tz_localize(None).dt.normalize()
            if "date" in frame and not frame.date.map(_date).eq(dates).all():
                raise ValueError("Session date disagrees with exchange END timestamp")
            frame["bar_end"] = stamps
            frame["date"] = dates
            if frame.bar_end.duplicated().any():
                raise ValueError("Duplicate native five-minute END timestamps")
            blocks = {day: block.sort_values("bar_end").reset_index(drop=True)
                      for day, block in frame.groupby("date", sort=False)}
        except Exception as exc:
            raise InvalidIntradayData(f"Invalid native K5 cache for {code}: {path}: {exc}") from exc
        self._native[code] = blocks
        return blocks

    def frame(self, code: str, day, daily_open: float) -> pd.DataFrame:
        """Validate a needed session and return its causally adjusted quotes."""
        day = _date(day)
        label = f"{code} {day.date()}"
        if day not in self.calendar:
            raise MissingIntradayData(f"No verified regular-session calendar for {label}")
        if not math.isfinite(daily_open) or daily_open <= 0:
            raise InvalidIntradayData(f"A positive adjusted daily open is required for {label}")
        blocks = self._load(code)
        if day not in blocks:
            raise MissingIntradayData(f"Missing native K5 session {label}; no daily substitute is allowed")
        frame = blocks[day].copy()
        try:
            end = frame.bar_end.dt.hour * 60 + frame.bar_end.dt.minute
            expected = np.arange(575, self.calendar[day] + 1, 5)
            if not np.array_equal(end.to_numpy(), expected):
                raise ValueError(f"Expected {len(expected)} continuous regular-session K5 bars ending "
                                 f"at {self.calendar[day] // 60:02d}:{self.calendar[day] % 60:02d}")
            for column in ("volume", "turnover"):
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
                if not (np.isfinite(values) & (values >= 0)).all():
                    raise ValueError(f"{column} must be finite and nonnegative")
            if float(frame.volume.iloc[0]) <= 0:
                raise ValueError("The first 09:30-09:35 native bar has no positive volume; its open is untrusted")
            prepare_bars(frame)
            factor = float(daily_open) / float(frame.open.iloc[0])
            for column in ("open", "high", "low", "close"):
                frame[column] = frame[column].astype(float) * factor
            cumulative_volume = frame.volume.astype(float).cumsum()
            frame["vwap"] = frame.turnover.astype(float).cumsum().div(cumulative_volume) * factor
            # Turnover conventions can differ by vendor. Invalid cumulative
            # VWAP is unavailable; it is never used to fabricate OHLC or fills.
            valid_vwap = ((frame.vwap >= frame.low.cummin() * .9999)
                          & (frame.vwap <= frame.high.cummax() * 1.0001))
            frame.loc[~valid_vwap, "vwap"] = np.nan
            frame["adjustment_factor"] = factor
            frame["bar_start"] = end.to_numpy() - 5
            frame["bar_end"] = end.to_numpy()
            frame["minute"] = end.to_numpy()
            prepare_bars(frame)
        except Exception as exc:
            raise InvalidIntradayData(f"Invalid native K5 session {label}: {exc}") from exc
        return frame

    def bars(self, code: str, day, daily_open: float) -> PreparedBars:
        day = _date(day)
        key = (code, day, float(daily_open))
        if key not in self._prepared:
            self._prepared[key] = prepare_bars(self.frame(code, day, daily_open))
        return self._prepared[key]
