from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from ndxbots.config import Settings, load_settings


class FutuError(RuntimeError):
    pass


def _is_rate_limit(msg: object) -> bool:
    text = str(msg)
    return "频率太高" in text or "太多" in text or "Too frequent" in text


class FutuClient:
    """唯一和富途 OpenD 说话的地方。策略层不要直接 import futu。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._ctx = None

    def connect(self) -> None:
        try:
            from futu import OpenQuoteContext
        except ImportError as exc:
            raise FutuError(
                "未安装 futu-api。请先: pip install -r requirements.txt"
            ) from exc

        self._ctx = OpenQuoteContext(
            host=self.settings.futu_host,
            port=self.settings.futu_port,
        )

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    @contextmanager
    def session(self) -> Iterator["FutuClient"]:
        self.connect()
        try:
            yield self
        finally:
            self.close()

    def _ctx_or_raise(self):
        if self._ctx is None:
            raise FutuError("尚未 connect()。请用 client.session() 或先调用 connect()。")
        return self._ctx

    def ping(self) -> str:
        from futu import AuType, KLType, RET_OK

        ctx = self._ctx_or_raise()
        ret, data, _ = ctx.request_history_kline(
            "US.AAPL",
            start=None,
            end=None,
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            max_count=5,
        )
        if ret != RET_OK:
            raise FutuError(f"连接失败: {data}")
        last = data.iloc[-1]
        return f"OK  AAPL {last['time_key']} close={last['close']}"

    def list_us_index_plates(self) -> pd.DataFrame:
        from futu import Market, Plate, RET_OK

        ctx = self._ctx_or_raise()
        frames = []
        wanted = ("ALL", "INDUSTRY", "CONCEPT", "OTHER")
        plate_types = [getattr(Plate, name) for name in wanted if hasattr(Plate, name)]
        for plate_cls in plate_types:
            ret, data = ctx.get_plate_list(Market.US, plate_cls)
            if ret != RET_OK or data is None or len(data) == 0:
                continue
            tmp = data.copy()
            tmp["plate_class"] = str(plate_cls)
            frames.append(tmp)
            time.sleep(self.settings.request_sleep_sec)
        if not frames:
            return pd.DataFrame(columns=["code", "plate_name", "plate_id", "plate_class"])
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(subset=["code"])

    def plate_stocks(self, plate_code: str) -> pd.DataFrame:
        from futu import RET_OK

        ctx = self._ctx_or_raise()
        ret, data = ctx.get_plate_stock(plate_code)
        if ret != RET_OK:
            raise FutuError(f"get_plate_stock({plate_code}) 失败: {data}")
        return data

    def history_kline(
        self,
        code: str,
        start: str | None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if not start:
            start = self.settings.kline_start
        if not end:
            end = self.settings.kline_end or pd.Timestamp.today().strftime("%Y-%m-%d")

        frames: list[pd.DataFrame] = []
        for win_start, win_end in _year_windows(start, end):
            piece = self._history_kline_window(code, win_start, win_end)
            if piece is not None and not piece.empty:
                frames.append(piece)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = _normalize_kline(df, code)
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end)
        df = df[(df["date"] >= lo) & (df["date"] <= hi)]
        return df.reset_index(drop=True)

    def _history_kline_window(self, code: str, start: str, end: str) -> pd.DataFrame:
        from futu import AuType, KLType, RET_OK

        ctx = self._ctx_or_raise()
        frames: list[pd.DataFrame] = []
        page = None
        retries = 0

        while True:
            ret, data, page = ctx.request_history_kline(
                code,
                start=start,
                end=end,
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
                max_count=self.settings.max_count,
                page_req_key=page,
            )
            if ret != RET_OK:
                if _is_rate_limit(data) and retries < 6:
                    retries += 1
                    wait = 31
                    print(f"    限频 {code} {start}~{end}，等 {wait}s 后重试 ({retries}/6)")
                    time.sleep(wait)
                    continue
                raise FutuError(f"K线失败 {code}: {data}")

            retries = 0
            if data is not None and len(data) > 0:
                frames.append(data)
            if page is None:
                break
            time.sleep(self.settings.request_sleep_sec)

        time.sleep(self.settings.request_sleep_sec)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _year_windows(start: str, end: str) -> list[tuple[str, str]]:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    if lo > hi:
        return []
    windows = []
    year = lo.year
    while year <= hi.year:
        w_lo = max(lo, pd.Timestamp(year=year, month=1, day=1))
        w_hi = min(hi, pd.Timestamp(year=year, month=12, day=31))
        windows.append((w_lo.strftime("%Y-%m-%d"), w_hi.strftime("%Y-%m-%d")))
        year += 1
    return windows


def _normalize_kline(df: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy()
    if "time_key" in out.columns:
        out["date"] = pd.to_datetime(out["time_key"]).dt.tz_localize(None).dt.normalize()
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    else:
        raise FutuError(f"{code} K线没有 time_key/date 字段: {list(out.columns)}")

    keep = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "turnover": "turnover",
        "pe_ratio": "pe_ratio",
        "turnover_rate": "turnover_rate",
        "change_rate": "change_rate",
        "last_close": "last_close",
        "name": "name",
    }
    cols = {src: dst for src, dst in keep.items() if src in out.columns}
    out = out[list(cols.keys())].rename(columns=cols)
    out["code"] = code
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    return out.reset_index(drop=True)