from __future__ import annotations

import numpy as np
import pandas as pd

from ndxbots.config import Settings


def classify_qqq_regime(factors: pd.DataFrame, settings: Settings) -> pd.Series:
    """用基准（默认 QQQ）的 21 日涨跌 + 布林带宽标市场状态。返回 date -> 标签。"""
    bench = settings.benchmark
    raw = factors.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    q = raw.loc[raw["code"] == bench].drop_duplicates("date").set_index("date")
    ret_col = settings.qqq_ret_col
    if q.empty or ret_col not in q.columns:
        return pd.Series(dtype="object")

    ret = q[ret_col]
    width = q["my_bb_width"] if "my_bb_width" in q.columns else pd.Series(np.nan, index=q.index)
    chop = ret.abs().lt(settings.qqq_chop_ret) & width.lt(settings.qqq_chop_width)
    rally = ret.ge(settings.qqq_rally_ret)
    dump = ret.le(settings.qqq_dump_ret)
    label = pd.Series("过渡", index=q.index, dtype="object")
    label = label.mask(chop, "震荡")
    label = label.mask(rally, "大涨")
    label = label.mask(dump, "下跌")
    both = rally & dump
    if both.any():
        label = label.mask(both & ret.ge(0), "大涨")
        label = label.mask(both & ret.lt(0), "下跌")
    return label


def apply_qqq_regime_gates(
    df: pd.DataFrame,
    settings: Settings,
    new_long: pd.Series,
    new_short: pd.Series,
    long_eligible: pd.Series,
    short_eligible: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """按 QQQ 状态收紧新开，不改个股打分。大涨时空头更严但不禁止。"""
    if not settings.qqq_regime_gate or "qqq_regime" not in df.columns:
        return long_eligible, short_eligible

    chop = df["qqq_regime"].eq("震荡")
    rally = df["qqq_regime"].eq("大涨")
    dump = df["qqq_regime"].eq("下跌")

    if settings.qqq_chop_no_new:
        long_eligible = long_eligible & ~(chop & new_long)
        short_eligible = short_eligible & ~(chop & new_short)

    if "my_ma200_gap" in df.columns:
        short_tight = df["my_ma200_gap"].lt(-settings.qqq_rally_short_ma200_buffer)
        short_eligible = short_eligible & ~(rally & new_short & ~short_tight)
        long_tight = df["my_ma200_gap"].gt(settings.qqq_dump_ma200_buffer)
        long_eligible = long_eligible & ~(dump & new_long & ~long_tight)

    return long_eligible, short_eligible


def assign_combined_pool(
    df: pd.DataFrame,
    settings: Settings,
    long_eligible: pd.Series,
    short_eligible: pd.Series,
) -> pd.DataFrame:
    """多空合并成一个观察池，一共 top_n 档。"""
    df = df.copy()
    df["eligible"] = long_eligible | short_eligible
    df["pool_side"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["side_score"] = np.nan
    df.loc[long_eligible, "pool_side"] = "long"
    df.loc[long_eligible, "side_score"] = df.loc[long_eligible, "long_score"]
    take_short = short_eligible & (
        ~long_eligible | df["short_score"].ge(df["long_score"].fillna(-np.inf))
    )
    df.loc[take_short, "pool_side"] = "short"
    df.loc[take_short, "side_score"] = df.loc[take_short, "short_score"]

    if settings.qqq_regime_gate and settings.qqq_rally_short_top_n >= 0:
        is_short = df["pool_side"].eq("short")
        short_day_rank = pd.Series(np.nan, index=df.index)
        if is_short.any():
            short_day_rank.loc[is_short] = (
                df.loc[is_short].groupby("date")["side_score"].rank(
                    method="first", ascending=False
                )
            )
        drop_short = (
            df["qqq_regime"].eq("大涨")
            & is_short
            & short_day_rank.gt(settings.qqq_rally_short_top_n)
        )
        df.loc[drop_short, "pool_side"] = pd.NA
        df.loc[drop_short, "side_score"] = np.nan

    df["combo_rank"] = df.groupby("date")["side_score"].rank(
        method="first", ascending=False
    )
    df["pool_rank"] = pd.NA
    df["short_pool_rank"] = pd.NA
    long_rows = df["pool_side"].eq("long")
    short_rows = df["pool_side"].eq("short")
    df.loc[long_rows, "pool_rank"] = df.loc[long_rows, "combo_rank"]
    df.loc[short_rows, "short_pool_rank"] = df.loc[short_rows, "combo_rank"]
    in_combo = df["combo_rank"].le(settings.strategy_top_n).fillna(False)
    df["in_pool_long"] = long_rows & in_combo
    df["in_pool_short"] = short_rows & in_combo
    return df


def max_hold_map(qqq_regime: pd.Series, settings: Settings) -> dict | None:
    if not settings.qqq_regime_gate or qqq_regime.empty:
        return None
    return {
        day: (
            settings.qqq_chop_max_hold
            if label == "震荡"
            else settings.max_hold
        )
        for day, label in qqq_regime.items()
    }
