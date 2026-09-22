from __future__ import annotations

import numpy as np
import pandas as pd

from ndxbots.config import Settings


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def _space_ok(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].between(lo, hi).fillna(False)


def build_score_table(factors: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """给每天每只股票打综合分，并标多空观察池。

    allow_short=True 时，MA200 是方向闸门：
      my_ma200_gap > 0 → 只允许进多头池
      my_ma200_gap < 0 → 只允许进空头池
      缺 MA200（前 199 天）两边都不进。
    """
    df = factors.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    bench = settings.benchmark
    df = df[df["code"] != bench].copy()

    missing = [c for c in settings.strategy_factors if c not in df.columns]
    if missing:
        raise SystemExit(
            f"因子表缺少 {missing}。请先运行 python -m ndxbots.factors.compute"
        )

    invert = set(settings.strategy_invert)
    parts = []
    for col in settings.strategy_factors:
        ranked = df.groupby("date", sort=False)[col].transform(_cs_rank)
        if col in invert:
            ranked = 1.0 - ranked
        parts.append(ranked.rename(f"rank_{col}"))

    score = parts[0]
    for p in parts[1:]:
        score = score.add(p, fill_value=np.nan)
    df["score"] = score / len(parts)
    df["long_score"] = df["score"]
    df["short_score"] = 1.0 - df["score"]

    if "my_ma200_gap" in df.columns:
        gap = df["my_ma200_gap"]
        df["above_ma200"] = (gap > 0).fillna(False)
        df["below_ma200"] = (gap < 0).fillna(False)
    else:
        df["above_ma200"] = True
        df["below_ma200"] = False

    df["side"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["above_ma200"], "side"] = "long"
    df.loc[df["below_ma200"], "side"] = "short"

    df["space_ok"] = _space_ok(
        df, settings.space_col, settings.space_min, settings.space_max
    )
    df["short_space_ok"] = _space_ok(
        df,
        settings.short_space_col,
        settings.short_space_min,
        settings.short_space_max,
    )

    if "my_atr_pct" in df.columns and settings.min_atr_pct > 0:
        df["atr_ok"] = (df["my_atr_pct"] >= settings.min_atr_pct).fillna(False)
    else:
        df["atr_ok"] = True

    long_eligible = df["long_score"].notna() & df["atr_ok"] & df["space_ok"]
    if settings.require_above_ma200 or settings.allow_short:
        long_eligible &= df["above_ma200"]

    short_eligible = pd.Series(False, index=df.index)
    if settings.allow_short:
        short_eligible = (
            df["short_score"].notna()
            & df["below_ma200"]
            & df["atr_ok"]
            & df["short_space_ok"]
        )

    df["eligible"] = long_eligible | short_eligible
    df["pool_rank"] = pd.NA
    df["short_pool_rank"] = pd.NA

    long_picked = df.loc[long_eligible].copy()
    if not long_picked.empty:
        long_picked["pool_rank"] = long_picked.groupby("date")["long_score"].rank(
            method="first", ascending=False
        )
        df.loc[long_picked.index, "pool_rank"] = long_picked["pool_rank"]

    short_picked = df.loc[short_eligible].copy()
    if not short_picked.empty:
        short_picked["short_pool_rank"] = short_picked.groupby("date")[
            "short_score"
        ].rank(method="first", ascending=False)
        df.loc[short_picked.index, "short_pool_rank"] = short_picked["short_pool_rank"]

    df["in_pool_long"] = df["pool_rank"].le(settings.strategy_top_n).fillna(False)
    df["in_pool_short"] = df["short_pool_rank"].le(settings.strategy_top_n).fillna(False)
    df["in_hold_long"] = df["pool_rank"].le(settings.max_hold).fillna(False)
    df["in_hold_short"] = df["short_pool_rank"].le(settings.max_hold).fillna(False)
    df["in_pool"] = df["in_pool_long"] | df["in_pool_short"]
    df["in_hold"] = df["in_hold_long"] | df["in_hold_short"]
    return df


def latest_pool(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    last = scored["date"].max()
    cols = [
        c
        for c in [
            "date",
            "code",
            "side",
            "score",
            "long_score",
            "short_score",
            "pool_rank",
            "short_pool_rank",
            "in_pool",
            "in_hold",
            "in_pool_long",
            "in_pool_short",
            "above_ma200",
            "below_ma200",
            "space_ok",
            "short_space_ok",
            "atr_ok",
            "my_ma50_gap",
            "my_struct_gap",
            "my_dd_from_high_21",
            "my_dist_from_low_21",
            "my_ma200_gap",
            "my_atr_pct",
        ]
        if c in scored.columns
    ]
    out = scored.loc[scored["date"] == last, cols].copy()
    return out.sort_values(
        ["in_pool", "side", "pool_rank", "short_pool_rank", "score"],
        ascending=[False, True, True, True, False],
    )
