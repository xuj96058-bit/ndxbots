from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ndxbots.config import load_settings
from ndxbots.data.ingest import panel_path
from ndxbots.factors.price import (
    daily_return,
    dist_from_high,
    pivot,
    rolling_vol,
    rsi,
    trailing_return,
    volume_ratio,
)


def report_path(settings) -> Path:
    return settings.data_root / "mining" / "factor_ranking.csv"


def earnings_path(settings) -> Path:
    return settings.data_root / "raw" / "earnings.parquet"


def _forward_excess(close: pd.DataFrame, benchmark: str, horizon: int) -> pd.DataFrame:
    fwd = close.shift(-horizon) / close - 1
    if benchmark not in fwd.columns:
        raise SystemExit(f"面板里没有基准 {benchmark}")
    return fwd.sub(fwd[benchmark], axis=0)


def _rank_ic(factor: pd.DataFrame, label: pd.DataFrame) -> tuple[float, float, float, int]:
    ics = []
    common_days = factor.index.intersection(label.index)
    for day in common_days:
        x = factor.loc[day]
        y = label.loc[day]
        pair = pd.concat([x, y], axis=1, keys=["f", "y"]).dropna()
        pair = pair[pair["f"].abs() + pair["y"].abs() > 0]
        if len(pair) < 20:
            continue
        ics.append(pair["f"].rank().corr(pair["y"].rank()))
    series = pd.Series(ics, dtype=float).dropna()
    if series.empty:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    ir = mean / std if std and std > 0 else float("nan")
    pos = float((series > 0).mean())
    return mean, ir, pos, int(len(series))


def _cs_rank(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.rank(axis=1, pct=True)


def _macd_hist(close: pd.DataFrame) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _days_to_earnings(close: pd.DataFrame, earn: pd.DataFrame) -> pd.DataFrame:
    if "code" not in earn.columns or "earnings_date" not in earn.columns:
        return pd.DataFrame()
    dates = close.index
    codes = close.columns
    earn = earn.copy()
    earn["code"] = earn["code"].astype(str)
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"]).dt.normalize()
    earn = earn.dropna(subset=["code", "earnings_date"])
    out = pd.DataFrame(index=dates, columns=codes, dtype=float)
    by_code = {
        code: sorted(grp["earnings_date"].unique())
        for code, grp in earn.groupby("code")
    }
    for code in codes:
        events = by_code.get(code, [])
        if not events:
            continue
        ev = pd.Series(events)
        for day in dates:
            later = ev[ev >= day]
            if later.empty:
                continue
            out.loc[day, code] = (later.iloc[0] - day).days
    return out


def _candidates(panel: pd.DataFrame, close: pd.DataFrame, earn: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    volume = pivot(panel, "volume") if "volume" in panel.columns else None
    volume = volume.reindex(index=close.index, columns=close.columns) if volume is not None else None
    ret = daily_return(close)
    out: dict[str, pd.DataFrame] = {}

    for win in (5, 10, 21, 63, 126, 252):
        out[f"ret_{win}"] = trailing_return(close, win)
        out[f"vol_{win}"] = rolling_vol(ret, win)
        out[f"dd_from_high_{win}"] = dist_from_high(close, win)
        if volume is not None:
            out[f"vol_ratio_{win}"] = volume_ratio(volume, win)

    for win in (6, 14, 21):
        out[f"rsi_{win}"] = rsi(close, win)

    out["mom_12_1"] = trailing_return(close, 252) - trailing_return(close, 21)
    out["ma_gap_21"] = close / close.rolling(21, min_periods=21).mean() - 1
    out["ma_gap_63"] = close / close.rolling(63, min_periods=63).mean() - 1
    out["macd_hist"] = _macd_hist(close)

    if "open" in panel.columns:
        opn = pivot(panel, "open").reindex(index=close.index, columns=close.columns)
        out["gap"] = opn / close.shift(1) - 1
        out["intraday"] = close / opn - 1
    if "high" in panel.columns and "low" in panel.columns:
        high = pivot(panel, "high").reindex(index=close.index, columns=close.columns)
        low = pivot(panel, "low").reindex(index=close.index, columns=close.columns)
        out["amplitude"] = (high - low) / close.replace(0, np.nan)
    if "turnover_rate" in panel.columns:
        tr = pivot(panel, "turnover_rate").reindex(index=close.index, columns=close.columns)
        out["turnover"] = tr
        out["turnover_21"] = tr.rolling(21, min_periods=10).mean()
    if "pe_ratio" in panel.columns:
        pe = pivot(panel, "pe_ratio").reindex(index=close.index, columns=close.columns)
        pe = pe.mask(pe <= 0)
        out["pe"] = pe
        out["pe_rank"] = _cs_rank(pe)
        out["pe_z_63"] = (pe - pe.rolling(63, min_periods=20).mean()) / pe.rolling(63, min_periods=20).std()

    if earn is not None and not earn.empty:
        dte = _days_to_earnings(close, earn)
        if not dte.empty:
            out["days_to_earn"] = dte
            out["earn_within_7"] = (dte <= 7).astype(float)
            out["earn_within_21"] = (dte <= 21).astype(float)

    from ndxbots.factors.custom import my_factors

    high_w = pivot(panel, "high").reindex(index=close.index, columns=close.columns) if "high" in panel.columns else None
    low_w = pivot(panel, "low").reindex(index=close.index, columns=close.columns) if "low" in panel.columns else None
    opn_w = pivot(panel, "open").reindex(index=close.index, columns=close.columns) if "open" in panel.columns else None
    pe_w = None
    if "pe_ratio" in panel.columns:
        pe_w = pivot(panel, "pe_ratio").reindex(index=close.index, columns=close.columns)
    out.update(my_factors(close, volume, high_w, low_w, opn_w, pe_w))

    return out



def run_mine(horizon: int = 21) -> None:
    settings = load_settings()
    path = panel_path(settings)
    if not path.exists():
        raise SystemExit(f"还没有行情面板: {path}")

    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    close = pivot(panel, "close")
    if settings.benchmark in close.columns:
        close_for_ic = close.drop(columns=[settings.benchmark])
        panel_ic = panel[panel["code"] != settings.benchmark]
    else:
        close_for_ic = close
        panel_ic = panel

    earn = None
    ep = earnings_path(settings)
    if ep.exists():
        earn = pd.read_parquet(ep)
        print(f"已加载财报日历 {ep}  行数={len(earn)}")
    else:
        print("没有财报日历，先跳过财报类因子。需要时运行: python -m ndxbots.data.earnings")

    label = _forward_excess(close, settings.benchmark, horizon)
    label = label.reindex(columns=close_for_ic.columns)

    rows = []
    cands = _candidates(panel_ic, close_for_ic, earn)
    print(f"候选因子 {len(cands)} 个，预测未来 {horizon} 个交易日是否跑赢 QQQ")
    for name, wide in cands.items():
        wide = wide.reindex(index=label.index, columns=label.columns)
        ic, ir, pos, n = _rank_ic(wide, label)
        rows.append(
            {
                "factor": name,
                "rank_ic": None if np.isnan(ic) else round(ic, 4),
                "icir": None if np.isnan(ir) else round(ir, 3),
                "ic_positive_pct": None if np.isnan(pos) else round(pos * 100, 1),
                "sample_days": n,
            }
        )
        print(f"  {name:18s}  RankIC={rows[-1]['rank_ic']}  ICIR={rows[-1]['icir']}")

    report = pd.DataFrame(rows).sort_values(
        "rank_ic", key=lambda s: s.abs(), ascending=False
    )
    out = report_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"\n排名已写 {out}")
    print("\n按 |RankIC| 排序，前 15 名：")
    print(report.head(15).to_string(index=False))
    print("\n怎么读：")
    print("  RankIC 离 0 越远，越能分开后来跑赢/跑输的股票")
    print("  正数：因子越大，后来越容易跑赢")
    print("  负数：因子越大，后来越容易跑输（反过来用也行）")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量试因子并按预测力排序")
    parser.add_argument("--horizon", type=int, default=21)
    args = parser.parse_args()
    run_mine(horizon=args.horizon)


if __name__ == "__main__":
    main()
