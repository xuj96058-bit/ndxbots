from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.data.ingest import panel_path
from ndxbots.factors.custom import my_factors
from ndxbots.factors.price import (
    daily_return,
    dist_from_high,
    excess_return,
    melt_factor,
    pivot,
    rolling_vol,
    rsi,
    trailing_return,
    volume_ratio,
)


def factor_path(settings) -> Path:
    return settings.data_root / "factors" / "daily.parquet"


def _wide(panel: pd.DataFrame, col: str, close: pd.DataFrame) -> pd.DataFrame | None:
    if col not in panel.columns:
        return None
    return pivot(panel, col).reindex(index=close.index, columns=close.columns)


def compute_factor_table(panel: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    close = pivot(panel, "close")
    volume = _wide(panel, "volume", close)
    pe = _wide(panel, "pe_ratio", close)
    high = _wide(panel, "high", close)
    low = _wide(panel, "low", close)
    opn = _wide(panel, "open", close)

    ret = daily_return(close)
    pieces = [
        melt_factor(trailing_return(close, 21), "ret_21"),
        melt_factor(trailing_return(close, 63), "ret_63"),
        melt_factor(trailing_return(close, 126), "ret_126"),
        melt_factor(trailing_return(close, 252), "ret_252"),
        melt_factor(trailing_return(close, 252) - trailing_return(close, 21), "mom_12_1"),
        melt_factor(rolling_vol(ret, 63), "vol_63"),
        melt_factor(dist_from_high(close, 252), "dd_from_high_252"),
        melt_factor(rsi(close, 14), "rsi_14"),
    ]
    if volume is not None:
        pieces.append(melt_factor(volume_ratio(volume, 21), "vol_ratio_21"))
    if pe is not None:
        pieces.append(melt_factor(pe, "pe_ratio"))

    if benchmark in close.columns:
        bench_ret21 = trailing_return(close[[benchmark]], 21)[benchmark]
        excess = excess_return(trailing_return(close, 21), bench_ret21)
        pieces.append(melt_factor(excess, "excess_qqq_21"))

    for name, wide in my_factors(close, volume, high, low, opn, pe).items():
        pieces.append(melt_factor(wide.reindex(index=close.index, columns=close.columns), name))

    table = pieces[0]
    for part in pieces[1:]:
        table = table.merge(part, on=["date", "code"], how="outer")
    table = table.sort_values(["date", "code"]).reset_index(drop=True)
    return table


def run_compute() -> None:
    settings = load_settings()
    path = panel_path(settings)
    if not path.exists():
        raise SystemExit(
            f"还没有行情面板: {path}\n请先运行 python -m ndxbots.data.ingest"
        )

    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    print(f"读取面板 {path}")
    print(f"行数={len(panel)}  股票={panel['code'].nunique()}")

    table = compute_factor_table(panel, settings.benchmark)
    out = factor_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    latest = table["date"].max()
    latest_rows = table[table["date"] == latest].dropna(subset=["ret_21"])
    print(f"因子表已写 {out}")
    print(f"行数={len(table)}  日期 {table['date'].min().date()} → {latest.date()}")
    print(f"最近一天 {latest.date()} 有效股票: {len(latest_rows)}")
    print("\n最近一天样本（前 8 行）:")
    cols = [
        c
        for c in [
            "code",
            "my_ma50_gap",
            "my_struct_gap",
            "my_dd_from_high_21",
            "my_dist_from_low_21",
            "my_ma200_gap",
            "my_atr_pct",
            "ret_21",
            "rsi_14",
        ]
        if c in table.columns
    ]
    print(latest_rows[cols].head(8).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="从日K计算因子（含 custom.py）")
    parser.parse_args()
    run_compute()


if __name__ == "__main__":
    main()
