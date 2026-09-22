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
    # 本机 OpenD 默认 127.0.0.1:11111。远程或改过端口时用环境变量 FUTU_HOST / FUTU_PORT。
    futu_host: str
    futu_port: int

    # ---------- data：本地行情 ----------
    # 相对路径会拼到 PROJECT_ROOT 下面，例如 data → <项目根>/data
    data_root: Path
    # 下载日 K 的起点（含当天）。已有 parquet 且非 --full-refresh 时，会从本地最后一天的次日续拉。
    kline_start: str
    # 下载日 K 的终点（含当天）。None = 拉到 OpenD 当前最新交易日。
    # 要「2016-01-01 到 2026-09-01」时在 yaml 写 kline_end: "2026-09-01"。
    kline_end: str | None
    # 每只股票、每一页请求之间的间隔，避免触发富途限频。
    request_sleep_sec: float
    # request_history_kline 单页最大根数。日 K 十年大约 2500 根，必须分页，这个值保持 1000 即可。
    max_count: int

    # ---------- universe：股票池 ----------
    # True：先在富途美股板块里搜 plate_keywords，找到纳指 100 就用官方成分。
    # False 或搜索失败：退回 src/ndxbots/data/universe.py 里的兜底名单。
    prefer_futu_plate: bool
    plate_keywords: tuple[str, ...]
    # 基准代码，必须带市场前缀。面板、交易日历、回测超额收益都用它。
    benchmark: str
    # 成分之外额外下载的代码，至少要包含 benchmark。
    extra_codes: tuple[str, ...]

    # ---------- strategy：选股层（日线观察池）----------
    # 参与打分的因子列名，必须能在 factors 表里找到。
    strategy_factors: tuple[str, ...]
    # 历史 IC 为负的因子放这里，打分时会乘 -1（越大越好统一成多头方向）。
    strategy_invert: tuple[str, ...]
    # 观察池大小。执行层只许盯这 top_n 只，不是最终持仓数。
    strategy_top_n: int
    # True：收盘价必须在日线 MA200 上方才允许进多头池。
    require_above_ma200: bool
    # 第一版只做多。True 时才生成空头候选（策略层还要自己接）。
    allow_short: bool
    # 「离高点距离」那一列。默认 my_dd_from_high_21，值是负数（-0.05 = 离 21 日高点跌了 5%）。
    space_col: str
    # 离高点跌太多视为结构坏了，踢出池子。-0.12 = 最多允许回撤 12%。
    space_min: float
    # 离高点太近视为追顶。-0.01 = 至少要离开高点 1%。
    space_max: float
    # 日线 ATR / 收盘价下限。波动太小的票不做。
    min_atr_pct: float
    # 组合最多同时持有的只数。观察池仍是 top_n，真正下权重的是 max_hold。
    max_hold: int

    # ---------- backtest：回测切片（和下载区间独立）----------
    # 回测起始日。面板里更早的数据仍可留给均线预热（MA200 大约要 200 根）。
    bt_start: str
    # 回测结束日。None = 用到面板最后一天。
    bt_end: str | None
    # 单边成本，单位 bp。10 = 买卖各 0.10%。换手时按成交权重差的绝对值扣。
    cost_bps: float
    # 成交假设。next_close = T 日收盘定池，权重作用在 T+1 的收益上，避免用当根收盘价成交。
    bt_exec: str
    # 回测初始资金，只影响权益曲线绝对金额，不影响收益率。
    initial_cash: float


def load_settings() -> Settings:
    """
    从 yaml + 环境变量组装一份不可变 Settings。

    日期相关环境变量：
        DATA_DIR      覆盖 data.root
        KLINE_START   覆盖 data.kline_start
        KLINE_END     覆盖 data.kline_end；设为空字符串表示拉到最新
        FUTU_HOST / FUTU_PORT / REQUEST_SLEEP 覆盖连接和限频
    """
    raw = _load_yaml()
    futu = raw.get("futu", {}) or {}
    data = raw.get("data", {}) or {}
    universe = raw.get("universe", {}) or {}
    strategy = raw.get("strategy", {}) or {}
    backtest = raw.get("backtest", {}) or {}

    data_root = Path(os.getenv("DATA_DIR", data.get("root", "data")))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    # yaml 优先，环境变量可临时覆盖。未写 kline_end 时保持 None = 不截断。
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
        min_atr_pct=float(strategy.get("min_atr_pct", 0.01)),
        max_hold=int(strategy.get("max_hold", 4)),
        bt_start=str(backtest.get("start", "2018-01-01")),
        bt_end=bt_end,
        cost_bps=float(backtest.get("cost_bps", 10)),
        bt_exec=str(backtest.get("exec", "next_close")),
        initial_cash=float(backtest.get("initial_cash", 20_000)),
    )