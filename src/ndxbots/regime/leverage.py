from __future__ import annotations

"""FINRA 客戶融資餘額（Debit Balances），單位百萬美元。

官方月表：
https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx

月度數字對齊到日頻時默認晚 21 個日曆日再用，避免把當月末才知道的數
提前用到當月交易日。權重和滯後都可以在 config.yaml 的 regime 段改。
"""

from pathlib import Path

import pandas as pd

from ndxbots.config import PROJECT_ROOT, Settings

FINRA_XLSX = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
DEBIT_COL = "Debit Balances in Customers' Securities Margin Accounts"


def leverage_candidates(settings: Settings) -> list[Path]:
    return [
        settings.data_root / "meta" / "finra_margin_debt.csv",
        PROJECT_ROOT / "meta" / "finra_margin_debt.csv",
    ]


def leverage_cache_path(settings: Settings) -> Path:
    return settings.data_root / "meta" / "finra_margin_debt.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}
    month_col = cols.get("month") or cols.get("year-month") or cols.get("date")
    debt_col = cols.get("margin_debt") or cols.get(DEBIT_COL.lower())
    if month_col is None or debt_col is None:
        raise ValueError(f"融資盤表缺少 month/margin_debt 欄: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "month": df[month_col].astype(str).str.strip(),
            "margin_debt": pd.to_numeric(df[debt_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["margin_debt"])
    out = out[out["month"].str.match(r"^\d{4}-\d{2}$", na=False)]
    out = out.drop_duplicates(subset=["month"], keep="last").sort_values("month")
    return out.reset_index(drop=True)


def load_margin_debt(settings: Settings) -> tuple[pd.DataFrame, str]:
    for path in leverage_candidates(settings):
        if path.exists():
            return _normalize(pd.read_csv(path)), str(path)
    return pd.DataFrame(columns=["month", "margin_debt"]), ""


def refresh_margin_debt(settings: Settings) -> Path:
    """從 FINRA 官方 xlsx 拉最新月表，寫到 data/meta，不進 git。"""
    raw = pd.read_excel(FINRA_XLSX)
    table = _normalize(raw)
    path = leverage_cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def yoy_on_trading_days(
    monthly: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    lag_days: int,
) -> pd.Series:
    """月度 YoY，ffill 到交易日。available_on = 月末 + lag_days。"""
    if monthly.empty:
        return pd.Series(index=trading_days, dtype=float, name="leverage_yoy")
    frame = monthly.copy()
    frame["month_end"] = pd.to_datetime(frame["month"] + "-01") + pd.offsets.MonthEnd(0)
    frame = frame.sort_values("month_end")
    frame["yoy"] = frame["margin_debt"] / frame["margin_debt"].shift(12) - 1
    frame["available_on"] = frame["month_end"] + pd.Timedelta(days=int(lag_days))
    usable = frame.dropna(subset=["yoy"]).sort_values("available_on")
    days = pd.DataFrame({"date": pd.DatetimeIndex(trading_days)}).sort_values("date")
    merged = pd.merge_asof(
        days,
        usable[["available_on", "yoy"]],
        left_on="date",
        right_on="available_on",
        direction="backward",
    )
    return pd.Series(merged["yoy"].to_numpy(), index=trading_days, name="leverage_yoy")
