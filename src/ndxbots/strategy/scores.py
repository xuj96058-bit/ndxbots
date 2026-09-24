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


def _classify_slope_zone(df: pd.DataFrame, settings: Settings) -> pd.Series:
    """互斥五段。缺斜率数据时为 NA（两边都不开）。"""
    zone = pd.Series(pd.NA, index=df.index, dtype="object")
    if "my_ma200_slope_atr" not in df.columns:
        return zone

    x = df["my_ma200_slope_atr"]
    strong = settings.slope_atr_strong
    flat = settings.slope_atr_flat
    zone = pd.Series(
        np.select(
            [
                x.gt(strong),
                x.ge(flat) & x.le(strong),
                x.gt(-flat) & x.lt(flat),
                x.ge(-strong) & x.le(-flat),
                x.lt(-strong),
            ],
            ["strong_long", "weak_long", "flat", "weak_short", "strong_short"],
            default=pd.NA,
        ),
        index=df.index,
        dtype="object",
    )
    zone = zone.mask(x.isna(), pd.NA)

    if "my_ma200_slope" in df.columns and settings.slope_pct_floor > 0:
        tiny = df["my_ma200_slope"].abs() < settings.slope_pct_floor
        zone = zone.mask(tiny.fillna(False), "flat")
    return zone


def _lock_regime_side(df: pd.DataFrame, settings: Settings) -> pd.Series:
    """按股票时间序锁定方向。走平日默认沿用昨天；数据不足则清空。"""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    ordered = df.sort_values(["code", "date"])
    last_code = None
    last_side: str | None = None
    for i in ordered.index:
        code = df.at[i, "code"]
        zone = df.at[i, "slope_zone"]
        if code != last_code:
            last_code = code
            last_side = None
        if zone in {"strong_long", "weak_long"}:
            last_side = "long"
        elif zone in {"strong_short", "weak_short"}:
            last_side = "short"
        elif zone == "flat":
            if not settings.flat_keep_prev:
                last_side = None
        else:
            last_side = None
        out.at[i] = last_side
    return out


def build_score_table(factors: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """给每天每只股票打综合分，并用 MA200 斜率当多空闸门。

    MA200 不参与排名。Slope_ATR 五段：
      > strong          强多，允许新开多
      flat ~ strong     弱多，只顺势、默认不纳新
      -flat ~ +flat     走平，禁止反手和新开（默认沿用昨天方向）
      -strong ~ -flat   弱空，只顺势、默认不纳新
      < -strong         强空，允许新开空
    """
    df = factors.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    bench = settings.benchmark
    df = df[df["code"] != bench].copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

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

    buf = settings.ma200_buffer
    if "my_ma200_gap" in df.columns:
        gap = df["my_ma200_gap"]
        df["above_ma200"] = (gap > 0).fillna(False)
        df["below_ma200"] = (gap < 0).fillna(False)
        df["above_ma200_buf"] = (gap > buf).fillna(False)
        df["below_ma200_buf"] = (gap < -buf).fillna(False)
    else:
        df["above_ma200"] = True
        df["below_ma200"] = False
        df["above_ma200_buf"] = True
        df["below_ma200_buf"] = False

    use_slope = "my_ma200_slope_atr" in df.columns
    if use_slope:
        df["slope_zone"] = _classify_slope_zone(df, settings)
        df["regime_side"] = _lock_regime_side(df, settings)
        df["prev_regime"] = df.groupby("code", sort=False)["regime_side"].shift(1)
        df["side"] = df["regime_side"]
    else:
        df["slope_zone"] = pd.NA
        df["regime_side"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[df["above_ma200"], "regime_side"] = "long"
        df.loc[df["below_ma200"], "regime_side"] = "short"
        df["prev_regime"] = df.groupby("code", sort=False)["regime_side"].shift(1)
        df["side"] = df["regime_side"]

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

    prev_long = df["prev_regime"].eq("long")
    prev_short = df["prev_regime"].eq("short")
    zone = df["slope_zone"] if use_slope else pd.Series(pd.NA, index=df.index)

    if use_slope:
        long_dir = (
            (zone.eq("strong_long") & df["above_ma200_buf"])
            | (
                zone.eq("weak_long")
                & (prev_long | ((not settings.slope_weak_no_new) & df["above_ma200_buf"]))
            )
            | (zone.eq("flat") & settings.flat_keep_prev & prev_long)
        )
        short_dir = (
            (zone.eq("strong_short") & df["below_ma200_buf"])
            | (
                zone.eq("weak_short")
                & (
                    prev_short
                    | ((not settings.slope_weak_no_new) & df["below_ma200_buf"])
                )
            )
            | (zone.eq("flat") & settings.flat_keep_prev & prev_short)
        )
    else:
        long_dir = df["above_ma200"]
        short_dir = df["below_ma200"]

    if settings.require_above_ma200 and not use_slope:
        long_dir = long_dir & df["above_ma200"]

    long_eligible = df["long_score"].notna() & df["atr_ok"] & df["space_ok"] & long_dir
    if settings.require_above_ma200 and use_slope:
        new_long = zone.eq("strong_long") | (zone.eq("weak_long") & ~prev_long)
        long_eligible &= (~new_long) | df["above_ma200"]

    short_eligible = pd.Series(False, index=df.index)
    if settings.allow_short:
        short_eligible = (
            df["short_score"].notna()
            & df["atr_ok"]
            & df["short_space_ok"]
            & short_dir
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
            "slope_zone",
            "regime_side",
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
            "my_ma200_slope",
            "my_ma200_slope_atr",
            "my_atr_pct",
        ]
        if c in scored.columns
    ]
    out = scored.loc[scored["date"] == last, cols].copy()
    return out.sort_values(
        ["in_pool", "side", "pool_rank", "short_pool_rank", "score"],
        ascending=[False, True, True, True, False],
    )
