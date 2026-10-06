from __future__ import annotations

"""VIX / HYG / LQD / TLT.

Prefer data/meta/sentiment_external.csv. If missing, read Yahoo daily bars and
cache them under data/meta. Panel codes such as US.HYG win when present.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from ndxbots.config import Settings

YAHOO = {
    "vix": "^VIX",
    "hyg": "HYG",
    "lqd": "LQD",
    "tlt": "TLT",
}
PANEL_CODES = {
    "vix": ("US.VIX", "^VIX"),
    "hyg": ("US.HYG", "HYG"),
    "lqd": ("US.LQD", "LQD"),
    "tlt": ("US.TLT", "TLT"),
}
UA = {"User-Agent": "Mozilla/5.0"}


def external_path(settings: Settings) -> Path:
    return settings.data_root / "meta" / "sentiment_external.csv"


def _from_panel(panel: pd.DataFrame | None) -> pd.DataFrame:
    if panel is None or not {"date", "code", "close"}.issubset(panel.columns):
        return pd.DataFrame()
    wide = panel.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    wide.index = pd.to_datetime(wide.index).normalize()
    picked = {}
    for name, codes in PANEL_CODES.items():
        for code in codes:
            if code in wide.columns:
                picked[name] = wide[code]
                break
    return pd.DataFrame(picked, index=wide.index) if picked else pd.DataFrame()


def _from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "date" not in raw.columns:
        return pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    keep = [c for c in YAHOO if c in raw.columns]
    return raw.set_index("date")[keep].sort_index()


def _yahoo_close(symbol: str, range_: str = "15y") -> pd.Series:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?range={range_}&interval=1d"
    )
    req = urllib.request.Request(url, headers=UA)
    payload = json.loads(urllib.request.urlopen(req, timeout=30).read())
    result = payload["chart"]["result"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s").normalize()
    close = result["indicators"]["quote"][0]["close"]
    series = pd.Series(close, index=idx, dtype="float64")
    return series[~series.index.duplicated()].sort_index().dropna()


def _fetch_yahoo() -> pd.DataFrame:
    columns = {}
    for name, symbol in YAHOO.items():
        columns[name] = _yahoo_close(symbol)
    return pd.DataFrame(columns).sort_index()


def load_external(settings: Settings, panel: pd.DataFrame | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return vix/hyg/lqd/tlt indexed by date. Missing columns stay empty."""
    path = external_path(settings)
    cached = pd.DataFrame() if refresh else _from_csv(path)
    if cached.empty or any(col not in cached.columns for col in YAHOO):
        try:
            fetched = _fetch_yahoo()
        except Exception:
            fetched = pd.DataFrame()
        if not fetched.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            fetched.to_csv(path, index_label="date")
            cached = fetched
    panel_px = _from_panel(panel)
    if panel_px.empty:
        return cached
    if cached.empty:
        return panel_px
    return panel_px.combine_first(cached).sort_index()
