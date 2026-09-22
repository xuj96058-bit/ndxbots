from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ndxbots.config import Settings
from ndxbots.data.ingest import panel_path
from ndxbots.factors.compute import factor_path
from ndxbots.strategy.scores import build_score_table


def bt_dir(settings: Settings) -> Path:
    return settings.data_root / "backtest"


def _slice_dates(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    start = pd.Timestamp(settings.bt_start)
    out = df[df["date"] >= start]
    if settings.bt_end:
        out = out[out["date"] <= pd.Timestamp(settings.bt_end)]
    return out


def run_backtest(
    settings: Settings,
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | str]:
    if panel is None:
        p = panel_path(settings)
        if not p.exists():
            raise SystemExit(f"还没有行情面板: {p}")
        panel = pd.read_parquet(p)
    if factors is None:
        f = factor_path(settings)
        if not f.exists():
            raise SystemExit(f"还没有因子表: {f}\n请先运行 python -m ndxbots.factors.compute")
        factors = pd.read_parquet(f)

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()

    scored = build_score_table(factors, settings)
    scored = _slice_dates(scored, settings)
    panel = _slice_dates(panel, settings)

    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    bench = settings.benchmark
    if bench not in close.columns:
        raise SystemExit(f"面板里没有基准 {bench}")

    hold = scored[scored["in_hold"]][["date", "code", "score", "pool_rank"]].copy()
    if hold.empty:
        raise SystemExit("观察池在回测区间内是空的，检查因子是否算出来、过滤是否过严")

    # 每天等权；T 日名单用于 T+1 的收益（避免收盘价当根成交）
    hold["w"] = hold.groupby("date")["code"].transform(lambda s: 1.0 / len(s))
    weights = hold.pivot(index="date", columns="code", values="w").sort_index()
    weights = weights.reindex(close.index).fillna(0.0)

    ret = close.pct_change(fill_method=None)
    applied = weights.shift(1).fillna(0.0)
    common = [c for c in applied.columns if c in ret.columns]
    port_ret_gross = (applied[common] * ret[common]).sum(axis=1)

    traded = applied.diff().abs().sum(axis=1).fillna(applied.sum(axis=1))
    turnover = 0.5 * traded
    cost = traded * (settings.cost_bps / 10_000.0)
    port_ret = port_ret_gross - cost

    bench_ret = ret[bench].reindex(port_ret.index)
    equity = (1 + port_ret.fillna(0)).cumprod() * settings.initial_cash
    bench_eq = (1 + bench_ret.fillna(0)).cumprod() * settings.initial_cash

    curve = pd.DataFrame(
        {
            "date": port_ret.index,
            "port_ret": port_ret.to_numpy(),
            "port_ret_gross": port_ret_gross.reindex(port_ret.index).to_numpy(),
            "bench_ret": bench_ret.to_numpy(),
            "excess": (port_ret - bench_ret).to_numpy(),
            "turnover": turnover.reindex(port_ret.index).fillna(0).to_numpy(),
            "n_hold": applied.gt(0).sum(axis=1).reindex(port_ret.index).to_numpy(),
            "equity": equity.to_numpy(),
            "bench_equity": bench_eq.to_numpy(),
        }
    )

    extra = [
        c
        for c in ["my_ma50_gap", "my_struct_gap", "my_dd_from_high_21", "my_atr_pct"]
        if c in scored.columns
    ]
    holdings = hold.merge(
        scored[["date", "code", *extra]].drop_duplicates(["date", "code"]),
        on=["date", "code"],
        how="left",
    )

    stats = _summarize(curve, settings)
    return {"curve": curve, "holdings": holdings, "stats": stats, "scored": scored}


def _ann_factor(n_days: int) -> float:
    return 252.0


def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else 0.0


def _summarize(curve: pd.DataFrame, settings: Settings) -> str:
    c = curve.dropna(subset=["port_ret"]).copy()
    if c.empty:
        return "回测无有效收益行"
    p = c["port_ret"].fillna(0)
    b = c["bench_ret"].fillna(0)
    x = p - b
    days = max(len(c), 1)
    years = days / 252.0
    port_end = float(c["equity"].iloc[-1])
    bench_end = float(c["bench_equity"].iloc[-1])
    port_tot = port_end / settings.initial_cash - 1
    bench_tot = bench_end / settings.initial_cash - 1
    port_ann = (1 + port_tot) ** (1 / max(years, 1e-9)) - 1
    bench_ann = (1 + bench_tot) ** (1 / max(years, 1e-9)) - 1
    vol = float(p.std(ddof=1) * np.sqrt(252)) if len(p) > 1 else float("nan")
    sharpe = float(p.mean() / p.std(ddof=1) * np.sqrt(252)) if p.std(ddof=1) else float("nan")
    ir = float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) else float("nan")
    dd = _max_dd(c["equity"])
    win = float((p > 0).mean())
    lines = [
        "引擎 ndxbots.backtest  第一版日线观察池",
        f"区间 {c['date'].iloc[0].date()} ~ {c['date'].iloc[-1].date()}  交易日 {days}",
        f"期初 {settings.initial_cash:,.2f}  期末 {port_end:,.2f}  收益 {port_tot*100:.2f}%  年化 {port_ann*100:.2f}%",
        f"QQQ  期末 {bench_end:,.2f}  收益 {bench_tot*100:.2f}%  年化 {bench_ann*100:.2f}%",
        f"超额 { (port_tot-bench_tot)*100:.2f}%  信息比 {ir:.3f}",
        f"波动 {vol*100:.2f}%  Sharpe {sharpe:.3f}  最大回撤 {dd*100:.2f}%  日胜率 {win*100:.1f}%",
        f"日均换手 {c['turnover'].mean()*100:.2f}%  日均持股 {c['n_hold'].mean():.2f}",
        f"成本单边 {settings.cost_bps:.1f}bp  TopN观察 {settings.strategy_top_n}  持仓上限 {settings.max_hold}",
        f"因子 {list(settings.strategy_factors)}  MA200过滤={settings.require_above_ma200}",
        "成交假设: T 日收盘定池，权重滞后 1 日再乘收益（近似 T+1 开盘调仓）",
        "本版只做多、等权、不做 v7 的 15m 回踩/加仓。那是下一层。",
    ]
    return "\n".join(lines)
