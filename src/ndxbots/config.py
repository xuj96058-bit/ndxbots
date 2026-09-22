from __future__ import annotations

"""
统一读取项目配置。

优先级（后者覆盖前者，仅对标了环境变量的字段生效）：
    1. 代码里的默认值（下面 Settings / load_settings 的 fallback）
    2. 项目根目录 config.yaml
    3. 环境变量 或 .env（python-dotenv 在 import 时已 load_dotenv）

本文件只负责「读出来变成 Settings」，不要在这里写业务逻辑。
改日期区间、选股参数、回测费用，优先改 config.yaml，不要改代码默认值。
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# src/ndxbots/config.py → parents[0]=ndxbots, [1]=src, [2]=项目根（和 config.yaml 同级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml() -> dict:
    """读项目根目录的 config.yaml；文件不存在时返回空 dict，让后面走默认值。"""
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _optional_str(value: object | None) -> str | None:
    """YAML 里写 null / 空字符串都当成「未设置」。"""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "~"}:
        return None
    return text


@dataclass(frozen=True)
class Settings:
    """
    全项目只认这一个配置对象。各模块用 load_settings() 拿一份，不要自己再读 yaml。

    字段按模块分组，和 config.yaml 的段落一一对应。
    """

    # ---------- futu：OpenD 连接 ----------
    futu_host: str
    futu_port: int

    # ---------- data：本地行情 ----------
    data_root: Path
    kline_start: str
    kline_end: str | None
    request_sleep_sec: float
    max_count: int

    # ---------- universe：股票池 ----------
    prefer_futu_plate: bool
    plate_keywords: tuple[str, ...]
    benchmark: str
    extra_codes: tuple[str, ...]

    # ---------- strategy：选股层（日线观察池） ----------
    strategy_factors: tuple[str, ...]
    strategy_invert: tuple[str, ...]
    strategy_top_n: int
    require_above_ma200: bool
    allow_short: bool
    space_col: str
    space_min: float
    space_max: float
    short_space_col: str
    short_space_min: float
    short_space_max: float
    min_atr_pct: float
    max_hold: int

    # ---------- backtest ----------
    bt_start: str
    bt_end: str | None
    cost_bps: float
    bt_exec: str
    initial_cash: float


def load_settings() -> Settings:
    raw = _load_yaml()
    futu = raw.get("futu", {}) or {}
    data = raw.get("data", {}) or {}
    universe = raw.get("universe", {}) or {}
    strategy = raw.get("strategy", {}) or {}
    backtest = raw.get("backtest", {}) or {}

    data_root = Path(os.getenv("DATA_DIR", data.get("root", "data")))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    kline_end = _optional_str(data.get("kline_end"))
    if "KLINE_END" in os.environ:
        kline_end = _optional_str(os.environ.get("KLINE_END"))

    bt_end = _optional_str(backtest.get("end"))

    return Settings(
        futu_host=os.getenv("FUTU_HOST", str(futu.get("host", "127.0.0.1"))),
        futu_port=int(os.getenv("FUTU_PORT", futu.get("port", 11111))),
        data_root=data_root,
        kline_start=os.getenv("KLINE_START", str(data.get("kline_start", "2016-01-01"))),
        kline_end=kline_end,
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
        short_space_col=str(strategy.get("short_space_col", "my_dist_from_low_21")),
        short_space_min=float(strategy.get("short_space_min", 0.01)),
        short_space_max=float(strategy.get("short_space_max", 0.12)),
        min_atr_pct=float(strategy.get("min_atr_pct", 0.01)),
        max_hold=int(strategy.get("max_hold", 4)),
        bt_start=str(backtest.get("start", "2018-01-01")),
        bt_end=bt_end,
        cost_bps=float(backtest.get("cost_bps", 10)),
        bt_exec=str(backtest.get("exec", "next_close")),
        initial_cash=float(backtest.get("initial_cash", 20_000)),
    )
