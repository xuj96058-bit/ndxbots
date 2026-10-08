from __future__ import annotations

import numpy as np
import pandas as pd

from ndxbots.config import Settings
from ndxbots.strategy.qqq_pool import (
    assign_combined_pool,
    apply_qqq_regime_gates,
    classify_qqq_regime,
    max_hold_map,
)


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def _apply_factor_scores(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Rank trend and signed efficiency groups independently on each date."""
    missing = [c for c in settings.strategy_factors if c not in df.columns]
    quality_weight = float(settings.strategy_quality_weight)
    if not np.isfinite(quality_weight) or not 0 <= quality_weight <= 1:
        raise ValueError("strategy.quality_weight 必须是 0~1 的有限数值")
    quality_col = "my_efficiency_21"
    if quality_weight > 0 and quality_col not in df.columns:
        missing.append(quality_col)
    if missing:
        raise SystemExit(
            f"因子表缺少 {missing}。更新策略后请先运行 python -m ndxbots.factors.compute"
        )

    invert = set(settings.strategy_invert)
    parts = []
    for col in settings.strategy_factors:
        ranked = df.groupby("date", sort=False)[col].transform(_cs_rank)
        if col in invert:
            ranked = 1.0 - ranked
        parts.append(ranked)
    trend_score = parts[0]
    for part in parts[1:]:
        trend_score = trend_score.add(part, fill_value=np.nan)
    trend_score = trend_score / len(parts)
    df["trend_score"] = trend_score
    df["quality_long_score"] = np.nan
    df["quality_short_score"] = np.nan
    if quality_weight > 0:
        df["quality_long_score"] = df.groupby("date", sort=False)[quality_col].transform(_cs_rank)
        df["quality_short_score"] = (-df[quality_col]).groupby(df["date"], sort=False).rank(
            method="average", pct=True
        )
        df["long_score"] = (1.0 - quality_weight) * trend_score + quality_weight * df["quality_long_score"]
        # Rank the negative efficiency separately; it is not 1-rank(efficiency).
        df["short_score"] = (1.0 - quality_weight) * (1.0 - trend_score) + quality_weight * df["quality_short_score"]
    else:
        df["long_score"] = trend_score
        df["short_score"] = 1.0 - trend_score
    df["score"] = df["long_score"]
    return df


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
        if pd.isna(zone):
            last_side = None
        elif zone in {"strong_long", "weak_long"}:
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


def _macd_new_ok(df: pd.DataFrame, settings: Settings) -> tuple[pd.Series, pd.Series]:
    """连续两根 hist_chg 同向才许新开。开关关或缺栏则全通过。"""
    ok_long = pd.Series(True, index=df.index)
    ok_short = pd.Series(True, index=df.index)
    if not settings.macd_gate or "my_macd_hist_chg" not in df.columns:
        return ok_long, ok_short
    chg = df["my_macd_hist_chg"]
    prev = df.groupby("code", sort=False)["my_macd_hist_chg"].shift(1)
    ok_long = (chg > 0) & (prev > 0)
    ok_short = (chg < 0) & (prev < 0)
    return ok_long.fillna(False), ok_short.fillna(False)


def _rsi_new_ok(df: pd.DataFrame, settings: Settings) -> tuple[pd.Series, pd.Series]:
    ok_long = pd.Series(True, index=df.index)
    ok_short = pd.Series(True, index=df.index)
    if not settings.rsi_gate or "my_rsi_14" not in df.columns:
        return ok_long, ok_short
    rsi = df["my_rsi_14"]
    ok_long = rsi.le(settings.rsi_long_max)
    ok_short = rsi.ge(settings.rsi_short_min)
    return ok_long.fillna(False), ok_short.fillna(False)


def _bb_width_ok(df: pd.DataFrame, settings: Settings) -> pd.Series:
    if not settings.bb_width_gate or "my_bb_width" not in df.columns:
        return pd.Series(True, index=df.index)
    width = df["my_bb_width"]
    if settings.bb_width_min is not None:
        return (width >= settings.bb_width_min).fillna(False)
    ranked = df.groupby("date", sort=False)["my_bb_width"].transform(_cs_rank)
    return (ranked >= settings.bb_width_pct_min).fillna(False)


def build_score_table(factors: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """给每天每只股票打综合分，并用 MA200 斜率当多空闸门。

    MA200 不参与排名。Slope_ATR 五段：
      > strong          强多，允许新开多
      flat ~ strong     弱多，只顺势、默认不纳新
      -flat ~ +flat     走平，禁止反手和新开（默认沿用昨天方向）
      -strong ~ -flat   弱空，只顺势、默认不纳新
      < -strong         强空，允许新开空
    MACD / RSI 只卡新开；带宽卡整票。开关默认关。
    """
    qqq_regime = classify_qqq_regime(factors, settings)
    df = factors.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    bench = settings.benchmark
    trading_dates = pd.DatetimeIndex(df.loc[df["code"] == bench, "date"]).unique().sort_values()
    if trading_dates.empty:
        trading_dates = pd.DatetimeIndex(df["date"]).unique().sort_values()
    df = df[df["code"] != bench].copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["qqq_regime"] = df["date"].map(qqq_regime)

    df = _apply_factor_scores(df, settings)

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
        new_long = zone.eq("strong_long") | (zone.eq("weak_long") & ~prev_long)
        new_short = zone.eq("strong_short") | (zone.eq("weak_short") & ~prev_short)
    else:
        long_dir = df["above_ma200"]
        short_dir = df["below_ma200"]
        new_long = long_dir & ~prev_long
        new_short = short_dir & ~prev_short

    if settings.require_above_ma200 and not use_slope:
        long_dir = long_dir & df["above_ma200"]

    macd_long_ok, macd_short_ok = _macd_new_ok(df, settings)
    rsi_long_ok, rsi_short_ok = _rsi_new_ok(df, settings)
    df["bb_width_ok"] = _bb_width_ok(df, settings)
    df["macd_ok"] = (~new_long | macd_long_ok) if settings.macd_gate else True
    df["rsi_ok"] = (~new_long | rsi_long_ok) if settings.rsi_gate else True

    long_eligible = (
        df["long_score"].notna()
        & df["atr_ok"]
        & df["space_ok"]
        & df["bb_width_ok"]
        & long_dir
        & (~new_long | macd_long_ok)
        & (~new_long | rsi_long_ok)
    )
    if settings.require_above_ma200 and use_slope:
        long_eligible &= (~new_long) | df["above_ma200"]

    short_eligible = pd.Series(False, index=df.index)
    if settings.allow_short:
        short_eligible = (
            df["short_score"].notna()
            & df["atr_ok"]
            & df["short_space_ok"]
            & df["bb_width_ok"]
            & short_dir
            & (~new_short | macd_short_ok)
            & (~new_short | rsi_short_ok)
        )

    long_eligible, short_eligible = apply_qqq_regime_gates(
        df, settings, new_long, new_short, long_eligible, short_eligible
    )
    df = assign_combined_pool(df, settings, long_eligible, short_eligible)
    df["in_pool"] = df["in_pool_long"] | df["in_pool_short"]
    df = apply_holding_hysteresis(
        df, settings, max_hold_by_date=max_hold_map(qqq_regime, settings),
        trading_dates=trading_dates,
    )
    df["in_pool"] = df["in_pool_long"] | df["in_pool_short"]
    df["in_hold"] = df["in_hold_long"] | df["in_hold_short"]
    return df


def apply_holding_hysteresis(
    df: pd.DataFrame,
    settings: Settings,
    max_hold_by_date: dict | None = None,
    trading_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """统一 side_score 的 Top N 新开池，按固定交易日同侧分差换股。

    方向缺失或反转每天退出，空位等到下一次普通调仓再补。安全旧仓
    平日续持；未满最少持有日或合并排名 <= keep_rank 的旧仓不换。
    调仓日先补空位，再用新股替换分差足够的同侧最弱、未保护旧仓。
    trading_dates 用基准会话推进状态，股票整日缺行也会触发退出。
    """
    min_days = int(getattr(settings, "min_hold_days", 10))
    keep_rank = int(getattr(settings, "keep_rank", 8))
    max_hold = int(settings.max_hold)
    weekdays = frozenset(getattr(settings, "rebalance_weekdays", (1, 4)))
    score_gap = float(getattr(settings, "replacement_score_gap", 0.05))
    if min_days < 1 or max_hold < 0 or keep_rank < 0:
        raise ValueError("持有日、持仓上限及留仓名次配置无效")
    if not weekdays or any(isinstance(day, bool) or not isinstance(day, int) or day not in range(7)
                           for day in weekdays):
        raise ValueError("rebalance_weekdays 必须包含 0~6 的整数")
    if not np.isfinite(score_gap) or not 0 <= score_gap <= 1:
        raise ValueError("replacement_score_gap 必须是 0~1 的有限数值")
    factor_cols = tuple(settings.strategy_factors)

    df = df.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if df.duplicated(["date", "code"]).any():
        raise ValueError("打分表同一日期/股票存在重复行")
    df["hold_days_long"] = 0
    df["hold_days_short"] = 0
    df["in_hold_long"] = False
    df["in_hold_short"] = False
    # 保留旧导出的诊断字段，原始绝对值不参与持仓选择。
    available_factors = [col for col in factor_cols if col in df.columns]
    if available_factors:
        df["abs_strength"] = df[available_factors].abs().mean(axis=1).fillna(df["score"].abs()).fillna(0.0)
    else:
        df["abs_strength"] = df["score"].abs().fillna(0.0)
    if "in_pool" not in df.columns:
        df["in_pool"] = df["in_pool_long"] | df["in_pool_short"]

    dates = pd.DatetimeIndex(df["date"]) if trading_dates is None else pd.DatetimeIndex(trading_dates)
    dates = dates.normalize().unique().sort_values()
    cap_by_date = {pd.Timestamp(day).normalize(): int(cap)
                   for day, cap in (max_hold_by_date or {}).items()}
    columns = ["code", "regime_side", "pool_side", "side_score", "combo_rank",
               "in_pool", "long_score", "short_score"]
    by_date = {}
    for day, frame in df.groupby("date", sort=False):
        meta = {}
        for row in frame[columns].itertuples(index=True):
            meta[str(row.code)] = {
                "i": row.Index,
                "regime": row.regime_side if pd.notna(row.regime_side) else None,
                "pool_side": row.pool_side if pd.notna(row.pool_side) else None,
                "side_score": float(row.side_score) if pd.notna(row.side_score) else np.nan,
                "rank": float(row.combo_rank) if pd.notna(row.combo_rank) else np.nan,
                "in_pool": bool(row.in_pool) if pd.notna(row.in_pool) else False,
                "long_score": float(row.long_score) if pd.notna(row.long_score) else np.nan,
                "short_score": float(row.short_score) if pd.notna(row.short_score) else np.nan,
            }
        by_date[pd.Timestamp(day)] = meta
    prev: dict[str, tuple[str, int]] = {}

    for day in dates:
        meta = by_date.get(pd.Timestamp(day), {})
        cap = max(0, min(max_hold, cap_by_date.get(pd.Timestamp(day), max_hold)))
        kept = {code: (side, age + 1) for code, (side, age) in prev.items()
                if code in meta and meta[code]["regime"] == side}

        def protected(code: str) -> bool:
            side, age = kept[code]
            info = meta[code]
            rank = info["rank"] if info["pool_side"] == side else np.nan
            return age < min_days or (np.isfinite(rank) and rank <= keep_rank)

        def holding_order(code: str):
            side, _ = kept[code]
            score = meta[code][side + "_score"]
            score = score if np.isfinite(score) else -np.inf
            rank = meta[code]["rank"]
            return (not protected(code), -score, rank if np.isfinite(rank) else np.inf, code)

        if len(kept) > cap:
            # 容量是硬上限，减少名额时也可退出仍处于最少持有期的旧仓。
            allowed = sorted(kept, key=holding_order)[:cap]
            kept = {code: kept[code] for code in allowed}

        if day.weekday() in weekdays:
            candidates = [code for code, info in meta.items()
                          if info["in_pool"] and info["pool_side"] in {"long", "short"}
                          and info["regime"] == info["pool_side"]
                          and np.isfinite(info["side_score"]) and np.isfinite(info["rank"])
                          and info["rank"] <= settings.strategy_top_n]
            candidates.sort(key=lambda code: (-meta[code]["side_score"], meta[code]["rank"], code))
            challengers = [code for code in candidates if code not in kept]
            available_old = {code: kept[code] for code in kept if not protected(code)}
            used = set()
            for code in challengers:
                if len(kept) >= cap:
                    break
                kept[code] = (meta[code]["pool_side"], 1)
                used.add(code)
            for code in challengers:
                if code in used:
                    continue
                side = meta[code]["pool_side"]
                same_side = [old for old, (old_side, _) in available_old.items()
                             if old_side == side and np.isfinite(meta[old][side + "_score"])]
                if not same_side:
                    continue
                old = min(same_side, key=lambda candidate: (meta[candidate][side + "_score"], candidate))
                improvement = meta[code]["side_score"] - meta[old][side + "_score"]
                if improvement >= score_gap - 1e-12:
                    del kept[old]
                    del available_old[old]
                    kept[code] = (side, 1)
                    used.add(code)

        for code, (side, days) in kept.items():
            i = meta[code]["i"]
            df.at[i, "side"] = side
            if side == "long":
                df.at[i, "in_hold_long"] = True
                df.at[i, "hold_days_long"] = days
            else:
                df.at[i, "in_hold_short"] = True
                df.at[i, "hold_days_short"] = days

        prev = kept

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
            "qqq_regime",
            "slope_zone",
            "regime_side",
            "score",
            "long_score",
            "short_score",
            "trend_score",
            "quality_long_score",
            "quality_short_score",
            "side_score",
            "combo_rank",
            "pool_side",
            "pool_rank",
            "short_pool_rank",
            "in_pool",
            "in_hold",
            "in_pool_long",
            "in_pool_short",
            "in_hold_long",
            "in_hold_short",
            "hold_days_long",
            "hold_days_short",
            "abs_strength",
            "above_ma200",
            "below_ma200",
            "space_ok",
            "short_space_ok",
            "atr_ok",
            "bb_width_ok",
            "my_ma50_gap",
            "my_struct_gap",
            "my_efficiency_21",
            "my_dd_from_high_21",
            "my_dist_from_low_21",
            "my_ma200_gap",
            "my_ma200_slope",
            "my_ma200_slope_atr",
            "my_atr_pct",
            "my_rsi_14",
            "my_bb_width",
            "my_macd_hist_chg",
        ]
        if c in scored.columns
    ]
    out = scored.loc[scored["date"] == last, cols].copy()
    return out.sort_values(
        [c for c in ["in_pool", "combo_rank", "code"] if c in out.columns],
        ascending=[False if c == "in_pool" else True
                   for c in ["in_pool", "combo_rank", "code"] if c in out.columns],
    )
