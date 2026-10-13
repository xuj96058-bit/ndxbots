from __future__ import annotations

"""FINRA 客戶融資餘額（Debit Balances），單位百萬美元。

官方月表：
https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx

優先使用 available_on 欄中的實際發布日期；缺少時以月末 + 設定滯後估計，
並明確標成 estimated_lag。固定 21 天不是 FINRA 的歷史發布日曆。
日頻輸出保留參考月份、可用日期與資料年齡，超過期限的值不再有效。
"""

from pathlib import Path

import pandas as pd
import numpy as np

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
    available_col = cols.get("available_on")
    if available_col is not None:
        out["available_on"] = pd.to_datetime(df[available_col], errors="coerce")
        supplied = df[available_col].notna() & df[available_col].astype(str).str.strip().ne("")
        if (supplied & out["available_on"].isna()).any():
            raise ValueError("融資盤 available_on 必須為有效發布日期")
    shaped = out["month"].str.match(r"^\d{4}-\d{2}$", na=False)
    calendar_month = out["month"].str.match(r"^\d{4}-(0[1-9]|1[0-2])$", na=False)
    if (shaped & ~calendar_month).any():
        raise ValueError("融資盤 month 的月份必須為 01 至 12")
    out = out[out["margin_debt"].notna() & np.isfinite(out["margin_debt"]) & out["margin_debt"].gt(0)]
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
    max_age_days: int = 90,
) -> pd.Series:
    """相容入口：按十二個日曆月比較，發布後才可用，過期則 NaN。"""
    return leverage_on_trading_days(monthly, trading_days, lag_days, max_age_days)["leverage_yoy"]


def leverage_on_trading_days(
    monthly: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    lag_days: int,
    max_age_days: int = 90,
) -> pd.DataFrame:
    """Return dated YoY and provenance without unlimited forward filling.

    Age is measured from the reference month end, not its publication date.
    A missing twelve-calendar-month comparison is kept missing; it never
    falls back to the twelfth preceding row or an older valid observation.
    The supplied table is a latest-vintage snapshot, not a revision archive.
    """
    if lag_days < 0 or max_age_days <= 0:
        raise ValueError("融資盤滯後須非負，資料有效天數須大於零")
    index = pd.DatetimeIndex(trading_days).normalize()
    out = pd.DataFrame(index=index)
    out["leverage_yoy"] = np.nan
    out["leverage_reference_month"] = pd.Series(pd.NA, index=index, dtype="string")
    out["leverage_available_on"] = pd.NaT
    out["leverage_age_days"] = np.nan
    out["leverage_stale"] = False
    out["leverage_availability_source"] = "missing"
    if monthly.empty or index.empty:
        return out
    frame = _normalize(monthly)
    if frame.empty:
        return out
    frame["period"] = pd.PeriodIndex(frame["month"], freq="M")
    frame["month_end"] = frame["period"].dt.to_timestamp(how="end").dt.normalize()
    debt = frame.set_index("period")["margin_debt"]
    previous = (frame["period"] - 12).map(debt)
    frame["yoy"] = frame["margin_debt"] / previous - 1
    estimate = frame["month_end"] + pd.Timedelta(days=int(lag_days))
    actual = (pd.to_datetime(frame["available_on"]).dt.normalize()
              if "available_on" in frame else pd.Series(pd.NaT, index=frame.index))
    if (actual.notna() & actual.lt(frame["month_end"])).any():
        raise ValueError("融資盤發布日不能早於參考月末")
    frame["available_on"] = actual.fillna(estimate)
    if not frame["available_on"].is_monotonic_increasing:
        raise ValueError("融資盤發布日期須隨參考月份遞增；修訂資料需另行保存版本")
    frame["availability_source"] = np.where(actual.notna(), "provided_release_date", "estimated_lag")
    # Enforce reference-month order when publication dates coincide; preserve
    # the requested day order even if trading_days was not sorted.
    frame = frame.sort_values(["available_on", "month_end"])
    days = pd.DataFrame({"date": index, "position": np.arange(len(index))}).sort_values("date")
    days["date"] = days["date"].astype("datetime64[ns]")
    frame["available_on"] = frame["available_on"].astype("datetime64[ns]")
    merged = pd.merge_asof(days, frame[["available_on", "month", "month_end", "yoy", "availability_source"]],
                           left_on="date", right_on="available_on", direction="backward")
    merged = merged.sort_values("position")
    age = (merged["date"] - merged["month_end"]).dt.days
    stale = age.gt(max_age_days)
    out["leverage_yoy"] = merged["yoy"].mask(stale).to_numpy()
    out["leverage_reference_month"] = pd.array(merged["month"], dtype="string")
    out["leverage_available_on"] = merged["available_on"].to_numpy()
    out["leverage_age_days"] = age.to_numpy()
    out["leverage_stale"] = stale.to_numpy()
    out["leverage_availability_source"] = merged["availability_source"].fillna("missing").to_numpy()
    return out

