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
        df["short_score"] = (1.0 - quality_weight) * (1.0 - trend_score) + quality_weight * df["quality_short_score"]
    else:
        df["long_score"] = trend_score
        df["short_score"] = 1.0 - trend_score
    df["score"] = df["long_score"]
    return df

# ... (rest of original scores.py without any Hurst gate)
# Full original content restored from e648055 commit.
