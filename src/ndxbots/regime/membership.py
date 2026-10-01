from __future__ import annotations

"""納指100成分的 point-in-time 掩碼。

優先讀 meta/ndx_membership.csv（或 data/meta 覆蓋）：
    code,start,end
    US.AAPL,2016-01-01,2020-12-31
    US.NVDA,2016-01-01,

end 空著表示仍在指數裡。沒有這份歷史名單時，退回當前 universe，
並把 membership_source 標成 static_snapshot。那不是 point-in-time，
廣度與擁擠度會有成分偏差。
"""

from pathlib import Path

import pandas as pd

from ndxbots.config import PROJECT_ROOT, Settings


def membership_candidates(settings: Settings) -> list[Path]:
    return [
        settings.data_root / "meta" / "ndx_membership.csv",
        PROJECT_ROOT / "meta" / "ndx_membership.csv",
    ]


def load_membership_intervals(settings: Settings) -> tuple[pd.DataFrame | None, str]:
    for path in membership_candidates(settings):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c.strip().lower(): c for c in df.columns}
        if "code" not in cols or "start" not in cols:
            continue
        out = pd.DataFrame(
            {
                "code": df[cols["code"]].astype(str).str.strip(),
                "start": pd.to_datetime(df[cols["start"]], errors="coerce"),
                "end": pd.to_datetime(df[cols["end"]], errors="coerce")
                if "end" in cols
                else pd.NaT,
            }
        )
        out = out.dropna(subset=["code", "start"])
        out = out[out["code"].ne("") & out["code"].ne("nan")]
        if out.empty:
            continue
        return out.reset_index(drop=True), f"point_in_time:{path}"
    return None, "static_snapshot"


def member_mask(
    dates: pd.DatetimeIndex,
    codes: list[str],
    settings: Settings,
    benchmark: str,
) -> tuple[pd.DataFrame, str]:
    """回傳 date x code 的 bool 表。True = 當天算進成分。"""
    intervals, source = load_membership_intervals(settings)
    mask = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=list(codes))
    if intervals is None:
        static = [c for c in codes if c != benchmark]
        if static:
            mask.loc[:, static] = True
        return mask, source

    for rec in intervals.itertuples(index=False):
        if rec.code not in mask.columns or rec.code == benchmark:
            continue
        start = pd.Timestamp(rec.start).normalize()
        end = pd.Timestamp(rec.end).normalize() if pd.notna(rec.end) else None
        slot = mask.index >= start
        if end is not None:
            slot = slot & (mask.index <= end)
        mask.loc[slot, rec.code] = True
    return mask, source
