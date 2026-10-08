from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.backtest.engine import run_backtest
from ndxbots.data.ingest import panel_path
from ndxbots.factors.compute import factor_path
from ndxbots.regime.compute import regime_path
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
    print(f"本侧观察候选 {len(part)} 档；完整模拟持仓与退出目标见 exec_pool.csv（max_hold={max_hold}）。")


def _print_regime(settings) -> None:
    path = regime_path(settings)
    if not path.exists():
        print("還沒有情緒/擁擠度。可先跑 python -m ndxbots.regime")
        return
    table = pd.read_parquet(path)
    if table.empty:
        return
    row = table.iloc[-1]
    sentiment = row.get("sentiment")
    crowding = row.get("crowding")
    sent_txt = f"{sentiment:.3f}" if pd.notna(sentiment) else "nan"
    crowd_txt = f"{crowding:.3f}" if pd.notna(crowding) else "nan"
    print(
        f"市場狀態 {pd.Timestamp(row['date']).date()}: {row.get('state')}  "
        f"情緒={sent_txt}  擁擠={crowd_txt}  "
        f"成分={row.get('membership_source')}"
    )
    if bool(row.get("qqq_sentiment_divergence")):
        print("背離: QQQ 在 252 日高位，情緒沒有同步過熱。")


def run_build() -> None:
    settings = load_settings()
    path = factor_path(settings)
    if not path.exists():
        raise SystemExit(f"还没有因子表: {path}\n请先运行 python -m ndxbots.factors.compute")

    factors = pd.read_parquet(path)
    scored = build_score_table(factors, settings)
    prices_path = panel_path(settings)
    if not prices_path.exists():
        raise SystemExit(f"还没有行情面板: {prices_path}\n请先运行 python -m ndxbots.data.ingest")
    panel = pd.read_parquet(prices_path)
    simulation = run_backtest(settings, panel=panel, factors=factors, scored=scored)
    if simulation["curve"].empty:
        raise SystemExit("回测区间没有可用交易日，无法生成执行目标")
    signal_date = pd.Timestamp(simulation["curve"]["date"].max()).normalize()

    out_dir = pool_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "scores.parquet"
    pool_path = out_dir / "observation_pool.csv"
    exec_path = out_dir / "exec_pool.csv"
    scored.to_parquet(scored_path, index=False)

    pool = latest_pool(scored.loc[scored["date"].eq(signal_date)])
    watch = pool[pool["in_pool"]].copy()
    watch.to_csv(pool_path, index=False)

    # Reconcile the complete simulated account, including retained names and
    # zero-target exits. The observation pool is not an order list.
    targets = simulation["targets"]
    exec_pool = targets.loc[targets["date"].eq(signal_date)].copy()
    exec_pool["model_source"] = "backtest"
    exec_pool.to_csv(exec_path, index=False)

    print(f"打分表已写 {scored_path}  行数={len(scored)}")
    print(f"观察池已写 {pool_path}  日期={pool['date'].max().date() if len(pool) else '-'}")
    print(f"执行池已写 {exec_path}  行数={len(exec_pool)}")
    print("执行数量来自本地模拟；真实账户需按实际持仓、现金及成交重新对账。")
    print(
        f"规则: 因子={list(settings.strategy_factors)}  "
        f"TopN={settings.strategy_top_n}  全场持仓上限={settings.max_hold}  "
        f"MA200过滤={settings.require_above_ma200}  "
        f"允许做空={settings.allow_short}  "
        f"MACD闸={settings.macd_gate}  RSI闸={settings.rsi_gate}  "
        f"带宽闸={settings.bb_width_gate}  "
        f"QQQ制度闸={settings.qqq_regime_gate}  "
        f"普通调仓星期={settings.rebalance_weekdays}  "
        f"同方向换股分差={settings.replacement_score_gap}"
    )
    if settings.qqq_regime_gate and "qqq_regime" in pool.columns and not pool.empty:
        print(f"今日 QQQ 状态: {pool['qqq_regime'].iloc[0]}")
    _print_regime(settings)
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
