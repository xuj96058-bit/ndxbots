from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """含今天的 trailing empirical midrank；相同值取中點，常數序列為 0.5。"""

    def _pct(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values[-1]):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) < min_periods:
            return np.nan
        current = values[-1]
        return float((np.sum(valid < current) + 0.5 * np.sum(valid == current)) / len(valid))

    return series.rolling(window, min_periods=min_periods).apply(_pct, raw=True)


def compound_return(daily: pd.Series, window: int) -> pd.Series:
    return (1 + daily).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1


def classify_state(
    sentiment: float,
    crowding: float,
    hot: float,
    cold: float,
    split: float,
) -> str:
    if pd.isna(sentiment) or pd.isna(crowding):
        return "数据不足"
    sent_high = sentiment >= split
    crowd_high = crowding >= split
    if sent_high and crowd_high:
        label = "情绪/拥挤偏高"
    elif sent_high and not crowd_high:
        label = "健康上涨"
    elif (not sent_high) and crowd_high:
        label = "去杠杆/下跌中"
    else:
        label = "潜在底部区"
    if sentiment >= hot and crowding >= hot:
        label = "双重极端"
    tags = []
    if sentiment >= hot:
        tags.append("情绪过热")
    elif sentiment <= cold:
        tags.append("情绪过冷")
    if crowding >= hot:
        tags.append("拥挤极端")
    elif crowding <= cold:
        tags.append("拥挤很低")
    if tags:
        return f"{label}（{'，'.join(tags)}）"
    return label

