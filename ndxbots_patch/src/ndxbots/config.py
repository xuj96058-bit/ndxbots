from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml() -> dict:
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Settings:
    futu_host: str
    futu_port: int
    data_root: Path
    kline_start: str
    request_sleep_sec: float
    max_count: int
    prefer_futu_plate: bool
    plate_keywords: tuple[str, ...]
    benchmark: str
    extra_codes: tuple[str, ...]
    strategy_factors: tuple[str, ...]
    strategy_invert: tuple[str, ...]
    strategy_top_n: int
    require_above_ma200: bool
    allow_short: bool
    space_col: str
    space_min: float
    space_max: float
    min_atr_pct: float
    max_hold: int
    bt_start: str
    bt_end: str | None
    cost_bps: float
    bt_exec: str
    initial_cash: float


def load_settings() -> Settings:
    raw = _load_yaml()
    futu = raw.get("futu", {})
    data = raw.get("data", {})
    universe = raw.get("universe", {})
    strategy = raw.get("strategy", {})
    backtest = raw.get("backtest", {})

    data_root = Path(os.getenv("DATA_DIR", data.get("root", "data")))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    bt_end = backtest.get("end")
    if bt_end is not None:
        bt_end = str(bt_end)

    return Settings(
        futu_host=os.getenv("FUTU_HOST", str(futu.get("host", "127.0.0.1"))),
        futu_port=int(os.getenv("FUTU_PORT", futu.get("port", 11111))),
        data_root=data_root,
        kline_start=os.getenv("KLINE_START", str(data.get("kline_start", "2016-01-01"))),
        request_sleep_sec=float(
            os.getenv("REQUEST_SLEEP", data.get("request_sleep_sec", 0.35))
        ),
        max_count=int(data.get("max_count", 1000)),
        prefer_futu_plate=bool(universe.get("prefer_futu_plate", True)),
        plate_keywords=tuple(universe.get("plate_keywords", ("NASDAQ 100", "NDX"))),
        benchmark=str(universe.get("benchmark", "US.QQQ")),
        extra_codes=tuple(universe.get("extra_codes", ("US.QQQ",))),
        strategy_factors=tuple(strategy.get("factors", ("my_ma50_gap", "my_struct_gap"))),
        strategy_invert=tuple(strategy.get("invert", ())),
        strategy_top_n=int(strategy.get("top_n", 10)),
        require_above_ma200=bool(strategy.get("require_above_ma200", True)),
        allow_short=bool(strategy.get("allow_short", False)),
        space_col=str(strategy.get("space_col", "my_dd_from_high_21")),
        space_min=float(strategy.get("space_min", -0.12)),
        space_max=float(strategy.get("space_max", -0.01)),
        min_atr_pct=float(strategy.get("min_atr_pct", 0.01)),
        max_hold=int(strategy.get("max_hold", 4)),
        bt_start=str(backtest.get("start", "2018-01-01")),
        bt_end=bt_end,
        cost_bps=float(backtest.get("cost_bps", 10)),
        bt_exec=str(backtest.get("exec", "next_close")),
        initial_cash=float(backtest.get("initial_cash", 20_000)),
    )
