from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.factors.compute import factor_path
from ndxbots.strategy.scores import build_score_table, latest_pool


def pool_dir(settings) -> Path:
    return settings.data_root / "strategy"


def _print_side(watch: pd.DataFrame, side: str, max_hold: int) -> None:
    part = watch[watch["side"] == side]
    if part.empty:
        print(f"\n{side} 观察池为空。")
        return
    show = part.copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].round(4)
    print(f"\n{side} 池（in_pool）:")
    print(show.to_string(index=False))
    hold = part[part["in_hold"]]
    print(f"回测本侧会持有前 {len(hold)} 档（max_hold={max_hold}）")


def run_build() -> None:
    settings = load_settings()
    path = factor_path(settings)
    if not path.exists():
        raise SystemExit(f"还没有因子表: {path}\n请先运行 python -m ndxbots.factors.compute")

    factors = pd.read_parquet(path)
    scored = build_score_table(factors, settings)

    out_dir = pool_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "scores.parquet"
    pool_path = out_dir / "observation_pool.csv"
    scored.to_parquet(scored_path, index=False)

    pool = latest_pool(scored)
    watch = pool[pool["in_pool"]].copy()
    watch.to_csv(pool_path, index=False)

    print(f"打分表已写 {scored_path}  行数={len(scored)}")
    print(f"观察池已写 {pool_path}  日期={pool['date'].max().date() if len(pool) else '-'}")
    print(
        f"规则: 因子={list(settings.strategy_factors)}  "
        f"TopN={settings.strategy_top_n}  每侧最多持仓={settings.max_hold}  "
        f"MA200过滤={settings.require_above_ma200}  "
        f"允许做空={settings.allow_short}"
    )
    if watch.empty:
        print("今日观察池为空。")
        return
    _print_side(watch, "long", settings.max_hold)
    if settings.allow_short:
        _print_side(watch, "short", settings.max_hold)


def main() -> None:
    parser = argparse.ArgumentParser(description="用日线因子生成纳指100观察池")
    parser.parse_args()
    run_build()


if __name__ == "__main__":
    main()
