"""Read-only OpenD K_5M ingestion and strictly validated local RTH caches.

Prices remain AuType.NONE. The backtest price store performs its own causal
daily-open calibration. No trade context or order API is used in this module.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from ndxbots.config import Settings, load_settings

ET = "America/New_York"
PRICE_COLUMNS = ("open", "high", "low", "close")


class IntradayDataError(RuntimeError):
    """Missing or invalid minute source; callers must not substitute daily bars."""


def intraday_root(settings: Settings) -> Path:
    return Path(settings.intraday_root) if settings.intraday_root is not None else settings.data_root / "intraday" / "kline"


def calendar_path(settings: Settings) -> Path:
    return intraday_root(settings).parent / "calendar.parquet"


def kline_path(settings: Settings, code: str) -> Path:
    if not re.fullmatch(r"US\.[A-Z0-9._-]+", code):
        raise IntradayDataError(f"Only US security codes are supported: {code!r}")
    return intraday_root(settings) / f"{code.replace('.', '_')}.parquet"


def _date(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(ET).tz_localize(None)
    return stamp.normalize()


def normalize_calendar(rows) -> pd.DataFrame:
    """Use the API day type, never infer a half day from missing minute bars."""
    if isinstance(rows, pd.DataFrame) and {"date", "close_minute"}.issubset(rows):
        result = rows[["date", "close_minute"]].copy()
        result["date"] = result.date.map(_date)
    else:
        records = []
        for row in rows:
            kind = str(row["trade_date_type"]).upper().split(".")[-1]
            if kind not in {"WHOLE", "MORNING", "HALF"}:
                raise IntradayDataError(f"Unsupported US trading-day type: {kind}")
            records.append({"date": _date(row["time"]), "close_minute": 960 if kind == "WHOLE" else 780})
        result = pd.DataFrame(records, columns=["date", "close_minute"])
    if result.empty or result.date.duplicated().any() or not result.close_minute.isin([780, 960]).all():
        raise IntradayDataError("Invalid or empty US session calendar")
    return result.sort_values("date").reset_index(drop=True)


def normalize_intraday(frame: pd.DataFrame, code: str, calendar: pd.DataFrame, *, required_dates=None) -> pd.DataFrame:
    """Validate bar END labels, retaining raw NONE price/volume units.

    Naive Futu time_key is New York local time; timezone-aware input is converted
    to New York. Regular grids are 09:35..16:00 or 09:35..13:00, inclusive.
    """
    required = {*PRICE_COLUMNS, "volume", "turnover"}
    if frame.empty or not required.issubset(frame):
        raise IntradayDataError(f"{code}: absent 5-minute source or missing OHLCV/turnover fields")
    out = frame.copy()
    if "code" in out and not out.code.eq(code).all():
        raise IntradayDataError(f"{code}: cache contains a different security")
    source = "bar_end" if "bar_end" in out else "time_key"
    if source not in out:
        raise IntradayDataError(f"{code}: no bar_end/time_key column")
    stamp = pd.to_datetime(out[source])
    try:
        out["bar_end"] = stamp.dt.tz_localize(ET, ambiguous="raise", nonexistent="raise") if stamp.dt.tz is None else stamp.dt.tz_convert(ET)
    except (ValueError, TypeError) as exc:
        raise IntradayDataError(f"{code}: invalid minute timezone") from exc
    if source == "bar_end" and "time_key" in out:
        native = pd.to_datetime(out.time_key)
        native = native.dt.tz_localize(ET) if native.dt.tz is None else native.dt.tz_convert(ET)
        if not native.eq(out.bar_end).all():
            raise IntradayDataError(f"{code}: time_key disagrees with bar_end")
    source_date = out["date"].map(_date) if "date" in out else None
    out["date"] = out.bar_end.dt.tz_localize(None).dt.normalize()
    if source_date is not None and not source_date.eq(out.date).all():
        raise IntradayDataError(f"{code}: stored date disagrees with minute timestamp")
    out["code"] = code
    out = out.sort_values("bar_end").drop_duplicates("bar_end", keep="last").reset_index(drop=True)
    calendar = normalize_calendar(calendar)
    closes = calendar.set_index("date").close_minute
    minute = out.bar_end.dt.hour * 60 + out.bar_end.dt.minute
    limit = out.date.map(closes)
    # Explicit Session.RTH request plus calendar filter: extended rows cannot
    # enter a cache even if a provider unexpectedly returns them.
    out = out.loc[limit.notna() & minute.ge(575) & minute.le(limit)].copy()
    if out.empty:
        raise IntradayDataError(f"{code}: no regular-session 5-minute bars returned")
    try:
        for col in (*PRICE_COLUMNS, "volume", "turnover"):
            out[col] = pd.to_numeric(out[col], errors="raise")
    except (ValueError, TypeError) as exc:
        raise IntradayDataError(f"{code}: nonnumeric raw OHLC/volume/turnover") from exc
    numeric = out[[*PRICE_COLUMNS, "volume", "turnover"]].to_numpy(float)
    prices = numeric[:, :4]
    bad = (~np.isfinite(numeric).all(axis=1) | (prices <= 0).any(axis=1)
           | (numeric[:, 4:] < 0).any(axis=1) | out.high.lt(out.open).to_numpy()
           | out.high.lt(out.close).to_numpy() | out.low.gt(out.open).to_numpy()
           | out.low.gt(out.close).to_numpy())
    if bad.any():
        raise IntradayDataError(f"{code}: invalid raw OHLC/volume/turnover rows={int(bad.sum())}")
    for day, part in out.groupby("date", sort=False):
        expected = np.arange(575, int(closes.loc[day]) + 1, 5)
        observed = part.bar_end.dt.hour * 60 + part.bar_end.dt.minute
        if (not np.array_equal(observed.to_numpy(), expected)
                or not part.bar_end.dt.second.eq(0).all() or not part.bar_end.dt.microsecond.eq(0).all()):
            raise IntradayDataError(f"{code} {day.date()}: incomplete 5-minute RTH grid; expected {len(expected)}, got {len(part)}")
    if required_dates is not None:
        wanted = {_date(day) for day in required_dates}
        absent = sorted(wanted - set(out.date))
        if absent:
            label = ", ".join(str(day.date()) for day in absent[:5])
            raise IntradayDataError(f"{code}: absent minute sessions ({len(absent)}): {label}; run python -m ndxbots.data.intraday")
    out["time_key"] = out.bar_end.dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.sort_values("bar_end").reset_index(drop=True)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sdk():
    try:
        import futu
    except ImportError as exc:
        raise IntradayDataError("futu-api is required for downloads; install the project dependencies first") from exc
    return futu


def _eligible_dates(settings: Settings, codes: list[str], calendar: pd.DataFrame, lo, hi) -> dict[str, set]:
    sessions = set(calendar.loc[calendar.date.between(lo, hi), "date"])
    panel_path = settings.data_root / "panel" / "daily.parquet"
    if not panel_path.exists():
        return {code: sessions.copy() for code in codes}
    panel = pd.read_parquet(panel_path, columns=["code", "date", "open", "close"])
    panel["date"] = panel.date.map(_date)
    result = {}
    for code in codes:
        valid = panel.loc[panel.code.eq(code) & panel.open.gt(0) & panel.close.gt(0), "date"]
        if valid.empty:
            raise IntradayDataError(f"{code}: absent daily source; ingest daily history before minute history")
        result[code] = sessions.intersection(valid)
    return result


def required_intraday_days(settings: Settings, start: str, end: str, codes: list[str] | None = None) -> pd.DataFrame:
    """Pure signal plan, avoiding a strategy-build/minute-data dependency cycle.

    Include each selected holding day plus its next two QQQ sessions. This
    covers T+1 entries/resizes and the exit after selection ends, while avoiding
    IPO/pre-warm-up sessions that cannot generate the existing daily signals.
    No portfolio simulation, broker connection or file mutation occurs here.
    """
    from ndxbots.strategy.scores import build_score_table

    factor_file = settings.data_root / "factors" / "daily.parquet"
    panel_file = settings.data_root / "panel" / "daily.parquet"
    if not factor_file.exists() or not panel_file.exists():
        raise IntradayDataError("Daily panel/factors absent; run data.ingest and factors.compute before data.intraday")
    factors = pd.read_parquet(factor_file)
    scored = build_score_table(factors, settings)
    scored["date"] = scored.date.map(_date)
    selected = scored.loc[scored.in_hold, ["code", "date"]].drop_duplicates()
    if codes is not None:
        selected = selected.loc[selected.code.isin(codes)]
    panel = pd.read_parquet(panel_file, columns=["code", "date"])
    sessions = pd.DatetimeIndex(sorted(set(panel.loc[panel.code.eq(settings.benchmark), "date"].map(_date))))
    if sessions.empty:
        raise IntradayDataError(f"Benchmark daily calendar absent: {settings.benchmark}")
    pairs = [selected]
    for offset in (1, 2):
        forward = dict(zip(sessions[:-offset], sessions[offset:]))
        later = selected.copy()
        later["date"] = later.date.map(forward)
        pairs.append(later)
    result = pd.concat(pairs, ignore_index=True).dropna(subset=["date"]).drop_duplicates(["code", "date"])
    return result.loc[result.date.between(_date(start), _date(end))].sort_values(["code", "date"]).reset_index(drop=True)


def _required_dates(settings, codes, calendar, lo, hi, required_days):
    if required_days is None:
        return _eligible_dates(settings, codes, calendar, lo, hi)
    plan = required_days[["code", "date"]].copy()
    plan["date"] = plan.date.map(_date)
    plan = plan.loc[plan.date.between(lo, hi)]
    absent_calendar = set(plan.date) - set(calendar.date)
    if absent_calendar:
        raise IntradayDataError(f"Required sessions absent from OpenD calendar: {sorted(absent_calendar)[:5]}")
    return {code: set(plan.loc[plan.code.eq(code), "date"]) for code in codes}


def load_intraday(settings: Settings, codes: list[str], start: str, end: str) -> pd.DataFrame:
    """Offline validation/load only. Missing sources fail without API fallback."""
    path = calendar_path(settings)
    if not path.exists():
        raise IntradayDataError(f"Minute session calendar absent: {path}; run python -m ndxbots.data.intraday")
    calendar = normalize_calendar(pd.read_parquet(path))
    lo, hi = _date(start), _date(end)
    dates = _eligible_dates(settings, codes, calendar, lo, hi)
    frames = []
    for code in codes:
        file = kline_path(settings, code)
        if not file.exists():
            raise IntradayDataError(f"{code}: minute cache absent: {file}; run python -m ndxbots.data.intraday")
        frame = normalize_intraday(pd.read_parquet(file), code, calendar, required_dates=dates[code])
        frames.append(frame.loc[frame.date.between(lo, hi)])
    if not frames:
        raise IntradayDataError("No minute security codes requested")
    return pd.concat(frames, ignore_index=True).sort_values(["code", "bar_end"]).reset_index(drop=True)


def collect_intraday(settings: Settings, codes: list[str], start: str, end: str, *, required_days=None, quote_context=None, api=None) -> dict[str, int]:
    """Download missing regular sessions; persist a code only after full QA.

    Opaque pagination keys stay within this connection and are never persisted.
    Failed pages leave the previous stock cache intact. A rerun replays missing
    dates with a new connection. Existing complete sessions are reused.
    """
    lo, hi = _date(start), _date(end)
    if lo > hi or not codes:
        raise IntradayDataError("Nonempty codes and start <= end are required")
    codes = list(dict.fromkeys(codes))
    for code in codes:
        kline_path(settings, code)
    root = intraday_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    calendar_file = calendar_path(settings)
    meta_file = calendar_file.with_suffix(".json")
    calendar = normalize_calendar(pd.read_parquet(calendar_file)) if calendar_file.exists() else None
    metadata = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    known = bool(calendar is not None and metadata and _date(metadata["start"]) <= lo and _date(metadata["end"]) >= hi)
    own_context = False
    context = quote_context
    sdk = api

    def connect():
        nonlocal context, own_context, sdk
        if sdk is None:
            sdk = _sdk()
        if context is None:
            context = sdk.OpenQuoteContext(host=settings.futu_host, port=settings.futu_port)
            own_context = True
        return context

    try:
        if not known:
            fetch_lo = min(lo, _date(metadata["start"])) if metadata else lo
            fetch_hi = max(hi, _date(metadata["end"])) if metadata else hi
            ctx = connect()
            ret, days = ctx.request_trading_days(market=sdk.TradeDateMarket.US, start=str(fetch_lo.date()), end=str(fetch_hi.date()))
            if ret != sdk.RET_OK:
                raise IntradayDataError(f"US calendar request failed: {days}")
            calendar = normalize_calendar(days)
            if calendar_file.exists():
                calendar = normalize_calendar(pd.concat([pd.read_parquet(calendar_file), calendar]).drop_duplicates("date", keep="last"))
            _write_parquet(calendar_file, calendar)
            _write_json(meta_file, {"start": str(fetch_lo.date()), "end": str(fetch_hi.date()),
                                   "source": "OpenD request_trading_days(TradeDateMarket.US)", "timezone": ET,
                                   "notes": "API excludes weekends/holidays; temporary closures need separate source verification."})
        expected = _required_dates(settings, codes, calendar, lo, hi, required_days)
        first_requests = deque()
        counts = {}
        for code in codes:
            path = kline_path(settings, code)
            old = normalize_intraday(pd.read_parquet(path), code, calendar) if path.exists() else pd.DataFrame()
            have = set(old.date) if len(old) else set()
            missing = sorted(expected[code] - have)
            if not expected[code]:
                raise IntradayDataError(f"{code}: no archive-supported sessions in requested range")
            if not missing:
                counts[code] = len(old.loc[old.date.between(lo, hi)])
                continue
            ctx = connect()
            new_pages = []
            for year in sorted({day.year for day in missing}):
                dates = [day for day in missing if day.year == year]
                request_lo, request_hi = dates[0], dates[-1]
                key = None
                seen_keys = set()
                while True:
                    if key is None:
                        now = time.monotonic()
                        while first_requests and now - first_requests[0] >= 31:
                            first_requests.popleft()
                        if len(first_requests) >= 55:
                            time.sleep(max(0, 31 - (now - first_requests[0])))
                        first_requests.append(time.monotonic())
                    for attempt in range(4):
                        ret, page, next_key = ctx.request_history_kline(
                            code, start=str(request_lo.date()), end=str(request_hi.date()),
                            ktype=sdk.KLType.K_5M, autype=sdk.AuType.NONE, session=sdk.Session.RTH,
                            max_count=settings.max_count, page_req_key=key)
                        if ret == sdk.RET_OK:
                            break
                        message = str(page)
                        if attempt == 3 or not any(word in message.lower() for word in ("频率", "頻率", "too frequent")):
                            raise IntradayDataError(f"{code}: history request failed: {message}")
                        time.sleep(31)
                    if page is not None and not page.empty:
                        new_pages.append(page)
                    if next_key is None:
                        break
                    token = repr(next_key)
                    if token in seen_keys:
                        raise IntradayDataError(f"{code}: repeated history pagination key; cache not modified")
                    seen_keys.add(token)
                    key = next_key
                    time.sleep(settings.request_sleep_sec)
            if not new_pages:
                raise IntradayDataError(f"{code}: absent minute history in requested sessions; cache not modified")
            # Parse each new page before combining with a normalized cache so an
            # existing bar_end column cannot mask the new raw time_key values.
            fresh = pd.concat(new_pages, ignore_index=True)
            source = "bar_end" if "bar_end" in fresh else "time_key"
            stamp = pd.to_datetime(fresh[source])
            local = stamp.dt.tz_convert(ET).dt.tz_localize(None) if stamp.dt.tz is not None else stamp
            # Requested windows can contain unneeded IPO/vendor-anomalous
            # sessions. They must not block otherwise valid selected sessions.
            fresh = fresh.loc[local.dt.normalize().isin(expected[code])].copy()
            normalized = normalize_intraday(fresh, code, calendar)
            merged = normalized if old.empty else pd.concat([old, normalized], ignore_index=True)
            merged = normalize_intraday(merged, code, calendar, required_dates=expected[code])
            _write_parquet(path, merged)
            _write_json(path.with_suffix(".json"), {"code": code, "autype": "NONE", "interval": "K_5M", "session": "RTH",
                                                   "timestamps": "bar END America/New_York", "rows": len(merged),
                                                   "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            counts[code] = len(merged.loc[merged.date.between(lo, hi)])
            print(f"{code}: {counts[code]} validated RTH bars", flush=True)
            time.sleep(settings.request_sleep_sec)
        return counts
    finally:
        if own_context and context is not None:
            context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache read-only US regular-session 5-minute history")
    parser.add_argument("--codes", nargs="+", help="Optional filter of signal-plan securities, e.g. US.AAPL US.MSFT")
    parser.add_argument("--start", help="Default backtest.start")
    parser.add_argument("--end", help="Default backtest.end or latest New York date")
    args = parser.parse_args()
    settings = load_settings()
    start = args.start or settings.bt_start
    end = args.end or settings.bt_end or str(pd.Timestamp.now(tz=ET).date())
    required = required_intraday_days(settings, start, end, codes=args.codes)
    codes = sorted(required.code.unique())
    if not codes:
        print("No selected signal sessions require minute downloads in this range")
        return
    print(f"Signal plan: {len(codes)} securities, {len(required)} stock sessions")
    result = collect_intraday(settings, codes, start, end, required_days=required)
    print(f"Complete: {len(result)} securities, {sum(result.values())} cached 5-minute bars")


if __name__ == "__main__":
    main()
