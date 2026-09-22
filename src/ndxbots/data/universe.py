from __future__ import annotations

from pathlib import Path

import pandas as pd

from ndxbots.config import Settings, load_settings
from ndxbots.data.futu_client import FutuClient

# 兜底名单：当前纳指100变动很快，仅在富途板块找不到时使用。
# 第一阶段允许「当前成分 + 静态名单」，历史成分偏差下一阶段再补。
NDX_FALLBACK_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC",
    "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "MAR", "MCHP", "MDLZ",
    "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA", "NXPI",
    "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL",
    "QCOM", "REGN", "ROP", "ROST", "SBUX", "SHOP", "SMCI", "SNPS", "TEAM", "TMUS",
    "TRI", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "WMT",
    "XEL", "ZS",
]


def to_futu_code(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith("US."):
        return ticker
    return f"US.{ticker}"


def fallback_codes(settings: Settings | None = None) -> list[str]:
    settings = settings or load_settings()
    codes = [to_futu_code(t) for t in NDX_FALLBACK_TICKERS]
    for extra in settings.extra_codes:
        if extra not in codes:
            codes.append(extra)
    return codes


def find_ndx_plate(client: FutuClient, keywords: tuple[str, ...]) -> pd.DataFrame:
    plates = client.list_us_index_plates()
    if plates.empty:
        return plates
    name = plates["plate_name"].astype(str)
    mask = False
    for kw in keywords:
        mask = mask | name.str.contains(kw, case=False, na=False)
    return plates.loc[mask].copy()


def resolve_universe(
    client: FutuClient | None = None,
    settings: Settings | None = None,
    save_to: Path | None = None,
) -> pd.DataFrame:
    """返回成分表：code, name, source, plate_code。"""
    settings = settings or load_settings()
    rows: list[dict] = []
    source = "fallback"
    plate_code = ""

    if settings.prefer_futu_plate and client is not None:
        hits = find_ndx_plate(client, settings.plate_keywords)
        if not hits.empty:
            plate_code = str(hits.iloc[0]["code"])
            stocks = client.plate_stocks(plate_code)
            for rec in stocks.to_dict("records"):
                code = rec.get("code")
                if not code:
                    continue
                rows.append(
                    {
                        "code": code,
                        "name": rec.get("stock_name", ""),
                        "list_time": rec.get("list_time", ""),
                        "source": "futu_plate",
                        "plate_code": plate_code,
                    }
                )
            source = "futu_plate"

    if not rows:
        for code in fallback_codes(settings):
            rows.append(
                {
                    "code": code,
                    "name": "",
                    "list_time": "",
                    "source": "fallback",
                    "plate_code": "",
                }
            )

    # 基准必须在下载名单里
    codes = {r["code"] for r in rows}
    if settings.benchmark not in codes:
        rows.append(
            {
                "code": settings.benchmark,
                "name": "Invesco QQQ Trust",
                "list_time": "",
                "source": source,
                "plate_code": plate_code,
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code")
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
    return df.reset_index(drop=True)
