from __future__ import annotations
import numpy as np
import pandas as pd

from ndxbots.config import Settings


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def build_score_table(factors: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """给每天每只股票打综合分，并标是否进入观察池。"""
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

    if "my_ma200_gap" in df.columns:
        df["above_ma200"] = df["my_ma200_gap"] > 0
    else:
        df["above_ma200"] = True

    eligible = df["score"].notna()
    if settings.require_above_ma200:
        eligible &= df["above_ma200"].fillna(False)

    space_col = settings.space_col
    if space_col and space_col in df.columns:
        space_ok = df[space_col].between(settings.space_min, settings.space_max)
        df["space_ok"] = space_ok.fillna(False)
        eligible &= df["space_ok"]
    else:
        df["space_ok"] = True

    if "my_atr_pct" in df.columns and settings.min_atr_pct > 0:
        atr_ok = df["my_atr_pct"] >= settings.min_atr_pct
        df["atr_ok"] = atr_ok.fillna(False)
        eligible &= df["atr_ok"]
    else:
        df["atr_ok"] = True

    df["eligible"] = eligible
    df["pool_rank"] = pd.NA
    picked = df[df["eligible"]].copy()
    picked["pool_rank"] = picked.groupby("date")["score"].rank(
        method="first", ascending=False
    )
    df.loc[picked.index, "pool_rank"] = picked["pool_rank"]
    df["in_pool"] = df["pool_rank"].le(settings.strategy_top_n).fillna(False)
    df["in_hold"] = df["pool_rank"].le(settings.max_hold).fillna(False)
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
            "score",
            "pool_rank",
            "in_pool",
            "in_hold",
            "above_ma200",
            "space_ok",
            "atr_ok",
            "my_ma50_gap",
            "my_struct_gap",
            "my_dd_from_high_21",
            "my_ma200_gap",
            "my_atr_pct",
        ]
        if c in scored.columns
    ]
    out = scored.loc[scored["date"] == last, cols].copy()
    return out.sort_values(["in_pool", "pool_rank", "score"], ascending=[False, True, False])
