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


def _legacy_ladder(table: pd.DataFrame, settings: Settings) -> pd.Series:
    parts = [c for c in ("pct_ma50", "pct_nhnl", "pct_rsi") if c in table.columns]
    if not parts:
        gross = pd.Series(1.0, index=table.index)
    else:
        breadth = table[parts].mean(axis=1, skipna=False)
        if "leverage_pct" in table.columns:
            sentiment = breadth * float(settings.regime_breadth_weight) + table["leverage_pct"] * float(
                settings.regime_leverage_weight
            )
            missing = table["leverage_pct"].isna() & breadth.notna()
            sentiment = sentiment.where(~missing, breadth)
        else:
            sentiment = breadth
        window = int(settings.gross_smooth)
        s5 = sentiment.rolling(window, min_periods=window).mean()
        start = float(settings.gross_ladder_start)
        step = float(settings.gross_ladder_step)
        cut = float(settings.gross_step)
        floor = float(settings.gross_floor)

        def _one(value: float) -> float:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return 1.0
            if step <= 0 or value > start:
                return 1.0
            n = int(np.floor((start - value + 1e-9) / step)) + 1
            max_steps = max(1, int(round((1.0 - floor) / cut))) if cut > 0 else 1
            n = max(1, min(n, max_steps))
            return max(floor, 1.0 - n * cut)

        gross = s5.map(_one)
    gross.index = pd.to_datetime(table["date"]).dt.normalize()
    return gross.rename("gross_target")


def _z_overlay(table: pd.DataFrame, settings: Settings) -> pd.Series:
    """Step gross only when prior Z crosses +/- 1.5. Extreme greed and dual-hot are caps."""
    sentiment = table["sentiment"]
    crowding = table["crowding"] if "crowding" in table.columns else pd.Series(np.nan, index=table.index)
    z = table["sentiment_z"]
    window = int(settings.gross_smooth)
    s5 = sentiment.rolling(window, min_periods=window).mean()
    c5 = crowding.rolling(window, min_periods=1).mean()
    hot = float(settings.dual_hot)
    floor = float(settings.gross_floor)
    step = float(settings.gross_step)
    z_hot = float(getattr(settings, "z_hot", 1.5))
    z_cold = float(getattr(settings, "z_cold", -1.5))
    zone_hot = float(getattr(settings, "zone_hot", 80.0)) / 100.0

    cur = 1.0
    prev_z = np.nan
    out = []
    for sent, crowd5, sent5, zz in zip(sentiment, c5, s5, z, strict=True):
        if pd.isna(sent) or pd.isna(zz):
            out.append(np.nan)
            continue
        if pd.notna(prev_z):
            if prev_z <= z_hot < zz:
                cur = max(floor, cur - step)
            elif prev_z >= z_cold > zz:
                cur = min(1.0, cur + step)
        prev_z = float(zz)
        cap_zone = floor if sent >= zone_hot else 1.0
        cap_crowd = floor if (pd.notna(sent5) and sent5 >= hot and pd.notna(crowd5) and crowd5 >= hot) else 1.0
        out.append(min(cap_zone, cap_crowd, cur))
    return pd.Series(out, index=pd.to_datetime(table["date"]).dt.normalize(), name="gross_target")


def gross_by_date(table: pd.DataFrame, settings: Settings) -> pd.Series:
    if not settings.use_gross_ladder:
        return pd.Series(1.0, index=pd.to_datetime(table["date"]).dt.normalize(), name="gross_target")
    if "sentiment_z" in table.columns and table["sentiment_z"].notna().any():
        return _z_overlay(table, settings)
    return _legacy_ladder(table, settings)
