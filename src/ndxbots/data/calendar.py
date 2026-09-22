from __future__ import annotations

from pathlib import Path

import pandas as pd


def trading_days_from_benchmark(panel_or_kline: pd.DataFrame) -> pd.DatetimeIndex:
    """用 QQQ（或任意基准）的交易日当美股日历。"""
    df = panel_or_kline.copy()
    if "date" not in df.columns:
        raise ValueError("需要 date 列")
    dates = pd.to_datetime(df["date"]).dt.normalize().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def last_complete_session(dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    if len(dates) == 0:
        return None
    return dates.max()


def save_calendar(dates: pd.DatetimeIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates}).to_parquet(path, index=False)
