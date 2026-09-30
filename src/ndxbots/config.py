from __future__ import annotations

"""
统一读取项目配置。

优先级（后者覆盖前者，仅对标了环境变量的字段生效）：
    1. 代码里的默认值（下面 Settings / load_settings 的 fallback）
    2. 项目根目录 config.yaml
    3. 环境变量 或 .env（python-dotenv 在 import 时已 load_dotenv）

本文件只负责「读出来变成 Settings」，不要在这里写业务逻辑。
改日期区间、选股參數、回测费用，优先改 config.yaml，不要改代码默认值。
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


def _optional_float(value: object | None) -> float | None:
    text = _optional_str(value)
    if text is None:
        return None
    return float(text)


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
    min_hold_days: int
    keep_rank: int
    ma200_buffer: float
    slope_atr_strong: float
    slope_atr_flat: float
    slope_pct_floor: float
    slope_weak_no_new: bool
    flat_keep_prev: bool
    macd_gate: bool
    rsi_gate: bool
    rsi_long_max: float
    rsi_short_min: float
    bb_width_gate: bool
    bb_width_min: float | None
    bb_width_pct_min: float
    qqq_regime_gate: bool
    qqq_ret_col: str
    qqq_rally_ret: float
    qqq_dump_ret: float
    qqq_chop_ret: float
    qqq_chop_width: float
    qqq_chop_no_new: bool
    qqq_chop_max_hold: int
    qqq_rally_short_top_n: int
    qqq_rally_short_ma200_buffer: float
    qqq_dump_ma200_buffer: float

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
        min_hold_days=int(strategy.get("min_hold_days", 10)),
        keep_rank=int(strategy.get("keep_rank", 8)),
        ma200_buffer=float(strategy.get("ma200_buffer", 0.015)),
        slope_atr_strong=float(strategy.get("slope_atr_strong", 1.5)),
        slope_atr_flat=float(strategy.get("slope_atr_flat", 0.5)),
        slope_pct_floor=float(strategy.get("slope_pct_floor", 0.0025)),
        slope_weak_no_new=bool(strategy.get("slope_weak_no_new", True)),
        flat_keep_prev=bool(strategy.get("flat_keep_prev", True)),
        macd_gate=bool(strategy.get("macd_gate", False)),
        rsi_gate=bool(strategy.get("rsi_gate", False)),
        rsi_long_max=float(strategy.get("rsi_long_max", 80)),
        rsi_short_min=float(strategy.get("rsi_short_min", 20)),
        bb_width_gate=bool(strategy.get("bb_width_gate", False)),
        bb_width_min=_optional_float(strategy.get("bb_width_min")),
        bb_width_pct_min=float(strategy.get("bb_width_pct_min", 0.20)),
        qqq_regime_gate=bool(strategy.get("qqq_regime_gate", False)),
        qqq_ret_col=str(strategy.get("qqq_ret_col", "ret_21")),
        qqq_rally_ret=float(strategy.get("qqq_rally_ret", 0.06)),
        qqq_dump_ret=float(strategy.get("qqq_dump_ret", -0.06)),
        qqq_chop_ret=float(strategy.get("qqq_chop_ret", 0.03)),
        qqq_chop_width=float(strategy.get("qqq_chop_width", 0.08)),
        qqq_chop_no_new=bool(strategy.get("qqq_chop_no_new", True)),
        qqq_chop_max_hold=int(strategy.get("qqq_chop_max_hold", 2)),
        qqq_rally_short_top_n=int(strategy.get("qqq_rally_short_top_n", 3)),
        qqq_rally_short_ma200_buffer=float(strategy.get("qqq_rally_short_ma200_buffer", 0.04)),
        qqq_dump_ma200_buffer=float(strategy.get("qqq_dump_ma200_buffer", 0.04)),
        bt_start=str(backtest.get("start", "2018-01-01")),
        bt_end=bt_end,
        cost_bps=float(backtest.get("cost_bps", 10)),
        bt_exec=str(backtest.get("exec", "next_close")),
        initial_cash=float(backtest.get("initial_cash", 20_000)),
    )
