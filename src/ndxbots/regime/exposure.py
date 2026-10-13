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
    if table["date"].isna().any() or table.duplicated("date").any():
        raise ValueError("Regime cache must contain unique, valid dates")
    return table.sort_values("date")


def _sentiment_no_upvol(table: pd.DataFrame, settings: Settings) -> pd.Series:
    """Compatibility formula for old caches; never average incomplete columns."""
    parts = ["pct_ma50", "pct_nhnl", "pct_rsi"]
    if not set(parts).issubset(table.columns):
        return pd.Series(np.nan, index=table.index)
    components = table[parts].apply(pd.to_numeric, errors="coerce")
    components = components.where((components >= 0) & (components <= 1))
    breadth = components.mean(axis=1, skipna=False)
    if "leverage_pct" not in table.columns:
        return breadth
    leverage = pd.to_numeric(table["leverage_pct"], errors="coerce")
    leverage = leverage.where(leverage.between(0, 1))
    sentiment = breadth * float(settings.regime_breadth_weight) + leverage * float(settings.regime_leverage_weight)
    missing = table["leverage_pct"].isna() & breadth.notna()
    return sentiment.where(~missing, breadth)


def _ladder(sentiment: float, settings: Settings) -> float:
    if sentiment is None or not np.isfinite(float(sentiment)):
        return float(settings.gross_floor)
    s = float(sentiment)
    start = float(settings.gross_ladder_start)
    step = float(settings.gross_ladder_step)
    cut = float(settings.gross_step)
    floor = float(settings.gross_floor)
    if s > start:
        return 1.0
    n = int(np.floor((start - s + 1e-9) / step)) + 1
    max_steps = max(1, int(round((1.0 - floor) / cut)))
    n = max(1, min(n, max_steps))
    return max(floor, 1.0 - n * cut)


def exposure_by_date(
    table: pd.DataFrame | None,
    settings: Settings,
    *,
    dates: pd.DatetimeIndex | None = None,
    required_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Causal exposure diagnostics, including explicit missing-data decisions.

    Supply the benchmark calendar to include absent sessions before smoothing.
    Existing numeric caches retain the original no-up-volume trade formula;
    their unavailable quality metadata is identified as unverified.
    """
    cache_reason = "regime_cache_missing" if table is None else "regime_date_missing"
    if table is None or table.empty:
        raw = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        if table is not None:
            cache_reason = "regime_cache_empty"
    else:
        if "date" not in table.columns:
            raise ValueError("Regime table has no date column")
        raw = table.copy()
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        if raw["date"].isna().any() or raw.duplicated("date").any():
            raise ValueError("Regime table must contain unique, valid dates")
        raw = raw.set_index("date").sort_index()
    available_dates = raw.index
    if dates is not None:
        calendar = pd.DatetimeIndex(dates).normalize().unique().sort_values()
        raw = raw.reindex(calendar)
    raw.index.name = "date"
    out = raw.copy()
    present = pd.Series(out.index.isin(available_dates), index=out.index)
    if "sentiment_trade" in raw.columns:
        trade = pd.to_numeric(raw["sentiment_trade"], errors="coerce")
    else:
        trade = _sentiment_no_upvol(raw, settings)
    crowding = (pd.to_numeric(raw["crowding"], errors="coerce") if "crowding" in raw.columns
                else pd.Series(np.nan, index=raw.index))
    numeric_valid = trade.between(0, 1) & crowding.between(0, 1)
    quality_known = "regime_data_valid" in raw.columns
    quality_valid = (raw["regime_data_valid"].eq(True).fillna(False) if quality_known
                     else pd.Series(True, index=raw.index))
    data_valid = present & numeric_valid & quality_valid
    quality_reason = (raw["regime_data_reason"].fillna("quality_unverified").astype(str)
                      if "regime_data_reason" in raw else
                      pd.Series("quality_unverified", index=raw.index))
    window = int(settings.gross_smooth)
    trade_smooth = trade.where(data_valid).rolling(window, min_periods=window).mean()
    # Missing current data cannot erase the dual-hot guard and become a
    # full-risk signal; valid rows retain the existing crowding smoother.
    crowd_smooth = crowding.where(data_valid).rolling(window, min_periods=1).mean()
    usable = data_valid & trade_smooth.notna() & crowd_smooth.notna()
    out["sentiment_trade"] = trade
    out["sentiment_trade_smooth"] = trade_smooth
    out["crowding_smooth"] = crowd_smooth
    out["regime_data_valid"] = data_valid
    out["regime_data_reason"] = quality_reason
    out["regime_quality_known"] = quality_known & present
    out["exposure_data_valid"] = usable
    out["gross_data_fallback"] = ~usable
    out["gross_target"] = trade_smooth.map(lambda s: _ladder(s, settings))
    out["gross_reason"] = "normal"
    cold = usable & trade_smooth.le(float(settings.gross_ladder_start))
    out.loc[cold, "gross_reason"] = "cold_ladder"
    dual = usable & trade_smooth.ge(float(settings.dual_hot)) & crowd_smooth.ge(float(settings.dual_hot))
    out.loc[dual, "gross_target"] = np.minimum(out.loc[dual, "gross_target"], float(settings.gross_floor))
    out.loc[dual, "gross_reason"] = "dual_hot"
    missing_reason = pd.Series("smoothing_warmup_or_gap", index=raw.index)
    missing_reason.loc[~quality_valid] = "regime_quality:" + quality_reason.loc[~quality_valid]
    missing_reason.loc[~numeric_valid] = "regime_score_missing_or_invalid"
    missing_reason.loc[~present] = cache_reason
    required = (pd.Series(True, index=raw.index) if required_dates is None else
                pd.Series(raw.index.isin(pd.DatetimeIndex(required_dates).normalize()), index=raw.index))
    unavailable = ~usable & required
    if unavailable.any() and getattr(settings, "regime_missing_policy", "floor") == "error":
        first = out.index[unavailable][0]
        raise ValueError(f"Regime exposure unavailable on {first.date()}: {missing_reason.loc[first]}")
    out.loc[~usable, "gross_target"] = float(settings.gross_floor)
    out.loc[~usable, "gross_reason"] = "missing_data_floor:" + missing_reason.loc[~usable]
    out.loc[~present, "regime_data_reason"] = cache_reason
    return out.reset_index()


def gross_by_date(
    table: pd.DataFrame | None,
    settings: Settings,
    *,
    dates: pd.DatetimeIndex | None = None,
    required_dates: pd.DatetimeIndex | None = None,
) -> pd.Series:
    diagnostics = exposure_by_date(table, settings, dates=dates, required_dates=required_dates)
    return diagnostics.set_index("date")["gross_target"].rename("gross_target")

