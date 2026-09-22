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


def _signed_weights(hold: pd.DataFrame) -> pd.Series:
    """同侧等权。两侧都有仓时各占 50%；只有一侧时该侧 100%。空头为负权重。"""
    is_long = hold["side"].eq("long")
    is_short = hold["side"].eq("short")
    n_long = hold.groupby("date")["side"].transform(lambda s: int(s.eq("long").sum()))
    n_short = hold.groupby("date")["side"].transform(lambda s: int(s.eq("short").sum()))
    both = (n_long > 0) & (n_short > 0)
    w = pd.Series(0.0, index=hold.index)
    w.loc[is_long & both] = 0.5 / n_long.loc[is_long & both]
    w.loc[is_short & both] = -0.5 / n_short.loc[is_short & both]
    w.loc[is_long & ~both] = 1.0 / n_long.loc[is_long & ~both]
    w.loc[is_short & ~both] = -1.0 / n_short.loc[is_short & ~both]
    return w


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

    cols = ["date", "code", "score", "pool_rank", "short_pool_rank", "side"]
    cols = [c for c in cols if c in scored.columns]
    hold = scored[scored["in_hold"]][cols].copy()
    if hold.empty:
        raise SystemExit("观察池在回测区间内是空的，检查因子是否算出来、过滤是否过严")

    if "side" not in hold.columns:
        hold["side"] = "long"
    hold["w"] = _signed_weights(hold)
    weights = hold.pivot(index="date", columns="code", values="w").sort_index()
    weights = weights.reindex(close.index).fillna(0.0)

    ret = close.pct_change(fill_method=None)
    applied = weights.shift(1).fillna(0.0)
    common = [c for c in applied.columns if c in ret.columns]
    port_ret_gross = (applied[common] * ret[common]).sum(axis=1)

    traded = applied.diff().abs().sum(axis=1).fillna(applied.abs().sum(axis=1))
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
            "n_hold": applied.ne(0).sum(axis=1).reindex(port_ret.index).to_numpy(),
            "n_long": applied.gt(0).sum(axis=1).reindex(port_ret.index).to_numpy(),
            "n_short": applied.lt(0).sum(axis=1).reindex(port_ret.index).to_numpy(),
            "net_exp": applied.sum(axis=1).reindex(port_ret.index).to_numpy(),
            "gross_exp": applied.abs().sum(axis=1).reindex(port_ret.index).to_numpy(),
            "equity": equity.to_numpy(),
            "bench_equity": bench_eq.to_numpy(),
        }
    )

    extra = [
        c
        for c in [
            "my_ma50_gap",
            "my_struct_gap",
            "my_dd_from_high_21",
            "my_dist_from_low_21",
            "my_ma200_gap",
            "my_atr_pct",
            "long_score",
            "short_score",
        ]
        if c in scored.columns
    ]
    holdings = hold.merge(
        scored[["date", "code", *extra]].drop_duplicates(["date", "code"]),
        on=["date", "code"],
        how="left",
    )

    stats = _summarize(curve, settings)
    return {"curve": curve, "holdings": holdings, "stats": stats, "scored": scored}


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
    cash_days = float((c["n_hold"] == 0).mean()) if "n_hold" in c.columns else 0.0
    lines = [
        "引擎 ndxbots.backtest  日线观察池（MA200 多空闸门）",
        f"区间 {c['date'].iloc[0].date()} ~ {c['date'].iloc[-1].date()}  交易日 {days}",
        f"期初 {settings.initial_cash:,.2f}  期末 {port_end:,.2f}  收益 {port_tot*100:.2f}%  年化 {port_ann*100:.2f}%",
        f"QQQ  期末 {bench_end:,.2f}  收益 {bench_tot*100:.2f}%  年化 {bench_ann*100:.2f}%",
        f"超额 { (port_tot-bench_tot)*100:.2f}%  信息比 {ir:.3f}",
        f"波动 {vol*100:.2f}%  Sharpe {sharpe:.3f}  最大回撤 {dd*100:.2f}%  日胜率 {win*100:.1f}%",
        f"日均换手 {c['turnover'].mean()*100:.2f}%  日均持股 {c['n_hold'].mean():.2f}  空仓天占比 {cash_days*100:.1f}%",
        f"日均多 {c['n_long'].mean():.2f}  日均空 {c['n_short'].mean():.2f}  "
        f"日均净曝光 {c['net_exp'].mean():.2f}  日均毛曝光 {c['gross_exp'].mean():.2f}",
        f"成本单边 {settings.cost_bps:.1f}bp  TopN观察 {settings.strategy_top_n}  每侧持仓上限 {settings.max_hold}",
        f"因子 {list(settings.strategy_factors)}  MA200过滤={settings.require_above_ma200}  做空={settings.allow_short}",
        "成交假设: T 日收盘定池，权重滞后 1 日再乘收益（近似 T+1 开盘调仓）",
        "空头为负权重；两侧同时有仓时各占 50% 资金。本版不做 15m 回踩/加仓。",
    ]
    return "\n".join(lines)
