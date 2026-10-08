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
import math
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


def _optional_str(value: object | None) -> str | None:
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
    futu_host: str
    futu_port: int
    data_root: Path
    kline_start: str
    kline_end: str | None
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
    regime_window: int
    regime_min_periods: int
    regime_ma_window: int
    regime_nhnl_window: int
    regime_rsi_window: int
    regime_ew_window: int
    regime_top_amount: int
    regime_min_names: int
    regime_breadth_weight: float
    regime_leverage_weight: float
    regime_hot: float
    regime_cold: float
    regime_split: float
    regime_leverage_lag_days: int
    use_gross_ladder: bool
    gross_smooth: int
    gross_ladder_start: float
    gross_ladder_step: float
    gross_step: float
    gross_floor: float
    dual_hot: float
    bt_start: str
    bt_end: str | None
    cost_bps: float
    bt_exec: str
    initial_cash: float
    rebalance_weekdays: tuple[int, ...] = (1, 4)
    replacement_score_gap: float = 0.05


def load_settings() -> Settings:
    raw = _load_yaml()
    futu = raw.get("futu", {}) or {}
    data = raw.get("data", {}) or {}
    universe = raw.get("universe", {}) or {}
    strategy = raw.get("strategy", {}) or {}
    regime = raw.get("regime", {}) or {}
    backtest = raw.get("backtest", {}) or {}

    data_root = Path(os.getenv("DATA_DIR", data.get("root", "data")))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    kline_end = _optional_str(data.get("kline_end"))
    if "KLINE_END" in os.environ:
        kline_end = _optional_str(os.environ.get("KLINE_END"))

    bt_end = _optional_str(backtest.get("end"))
    weekday_values = strategy.get("rebalance_weekdays", [1, 4])
    if (
        not isinstance(weekday_values, (list, tuple))
        or not weekday_values
        or any(isinstance(day, bool) or not isinstance(day, int) or day not in range(7)
               for day in weekday_values)
    ):
        raise ValueError("strategy.rebalance_weekdays 必须是 0~6 的整数列表（周一=0）")
    rebalance_weekdays = tuple(dict.fromkeys(weekday_values))
    replacement_score_gap = float(strategy.get("replacement_score_gap", 0.05))
    if not math.isfinite(replacement_score_gap) or not 0 <= replacement_score_gap <= 1:
        raise ValueError("strategy.replacement_score_gap 必须是 0~1 的有限数值")

    return Settings(
        futu_host=os.getenv("FUTU_HOST", str(futu.get("host", "127.0.0.1"))),
        futu_port=int(os.getenv("FUTU_PORT", futu.get("port", 11111))),
        data_root=data_root,
        kline_start=os.getenv("KLINE_START", str(data.get("kline_start", "2016-01-01"))),
        kline_end=kline_end,
        request_sleep_sec=float(os.getenv("REQUEST_SLEEP", data.get("request_sleep_sec", 0.35))),
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
        regime_window=int(regime.get("window", 252)),
        regime_min_periods=int(regime.get("min_periods", 252)),
        regime_ma_window=int(regime.get("ma_window", 50)),
        regime_nhnl_window=int(regime.get("nhnl_window", 20)),
        regime_rsi_window=int(regime.get("rsi_window", 14)),
        regime_ew_window=int(regime.get("ew_window", 60)),
        regime_top_amount=int(regime.get("top_amount", 10)),
        regime_min_names=int(regime.get("min_names", 40)),
        regime_breadth_weight=float(regime.get("breadth_weight", 0.7)),
        regime_leverage_weight=float(regime.get("leverage_weight", 0.3)),
        regime_hot=float(regime.get("hot", 0.8)),
        regime_cold=float(regime.get("cold", 0.2)),
        regime_split=float(regime.get("split", 0.5)),
        regime_leverage_lag_days=int(regime.get("leverage_lag_days", 21)),
        use_gross_ladder=bool(regime.get("use_gross_ladder", False)),
        gross_smooth=int(regime.get("gross_smooth", 5)),
        gross_ladder_start=float(regime.get("gross_ladder_start", 0.35)),
        gross_ladder_step=float(regime.get("gross_ladder_step", 0.05)),
        gross_step=float(regime.get("gross_step", 0.10)),
        gross_floor=float(regime.get("gross_floor", 0.60)),
        dual_hot=float(regime.get("dual_hot", 0.80)),
        bt_start=str(backtest.get("start", "2018-01-01")),
        bt_end=bt_end,
        cost_bps=float(backtest.get("cost_bps", 10)),
        bt_exec=str(backtest.get("exec", "next_close")),
        initial_cash=float(backtest.get("initial_cash", 20_000)),
        rebalance_weekdays=rebalance_weekdays,
        replacement_score_gap=replacement_score_gap,
    )
