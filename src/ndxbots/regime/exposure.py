from __future__ import annotations

import numpy as np
import pandas as pd

from ndxbots.config import Settings


def regime_table_path(settings: Settings):
    return settings.data_root / "regime" / "daily.parquet"


def load_regime_table(settings: Settings) -> pd.DataFrame | None:
    path = regime_table_path(settings)
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    if table.empty or "date" not in table.columns:
        return None
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"]).dt.normalize()
    return table.drop_duplicates("date").sort_values("date")


def _sentiment_no_upvol(table: pd.DataFrame, settings: Settings) -> pd.Series:
    parts = [c for c in ("pct_ma50", "pct_nhnl", "pct_rsi") if c in table.columns]
    if not parts:
        return pd.Series(np.nan, index=table.index)
    breadth = table[parts].mean(axis=1, skipna=False)
    if "leverage_pct" not in table.columns:
        return breadth
    sentiment = breadth * float(settings.regime_breadth_weight) + table["leverage_pct"] * float(settings.regime_leverage_weight)
    missing = table["leverage_pct"].isna() & breadth.notna()
    return sentiment.where(~missing, breadth)


def _ladder(sentiment: float, settings: Settings) -> float:
    if sentiment is None or (isinstance(sentiment, float) and np.isnan(sentiment)):
        return 1.0
    s = float(sentiment)
    if np.isnan(s):
        return 1.0
    start = float(settings.gross_ladder_start)
    step = float(settings.gross_ladder_step)
    cut = float(settings.gross_step)
    floor = float(settings.gross_floor)
    if step <= 0 or s > start:
        return 1.0
    n = int(np.floor((start - s + 1e-9) / step)) + 1
    max_steps = max(1, int(round((1.0 - floor) / cut))) if cut > 0 else 1
    n = max(1, min(n, max_steps))
    return max(floor, 1.0 - n * cut)


def gross_by_date(table: pd.DataFrame, settings: Settings) -> pd.Series:
    sentiment = _sentiment_no_upvol(table, settings)
    window = int(settings.gross_smooth)
    s5 = sentiment.rolling(window, min_periods=window).mean()
    c5 = table["crowding"].rolling(window, min_periods=1).mean() if "crowding" in table.columns else pd.Series(np.nan, index=table.index)
    gross = s5.map(lambda s: _ladder(s, settings))
    hot = float(settings.dual_hot)
    dual = s5.ge(hot) & c5.ge(hot)
    gross = gross.mask(dual, np.minimum(gross, float(settings.gross_floor)))
    gross.index = pd.to_datetime(table["date"]).dt.normalize()
    return gross.rename("gross_target")
