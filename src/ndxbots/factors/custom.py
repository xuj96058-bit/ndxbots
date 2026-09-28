"""
在这里写你自己的因子。

规则只有三条：
1. 因子必须是数字
2. 每一天、每一只股票都要有一个值（没有就空着）
3. 只能用「当天及以前」的数据，不能用未来的价格

close / volume / high / low / opn 都是表格：
  行 = 日期
  列 = 股票代码，例如 US.AAPL
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndxbots.factors.price import rsi

# 与 config.yaml 的 slope 视窗对齐。改视窗时两边一起改。
MA200_SLOPE_N = 20
MA200_SLOPE_N_FAST = 5
ATR_SLOPE_N = 20

BB_N = 20
BB_K = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def _true_range(
    close: pd.DataFrame,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
) -> pd.DataFrame:
    if high is not None and low is not None:
        prev = close.shift(1)
        tr = np.maximum((high - low).abs(), (high - prev).abs())
        tr = np.maximum(tr, (low - prev).abs())
        return pd.DataFrame(tr, index=close.index, columns=close.columns)
    ret = close.pct_change(fill_method=None).abs()
    return ret * close


def _macd_parts(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def my_factors(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
    opn: pd.DataFrame | None,
    pe: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    high21 = close.rolling(21, min_periods=21).max()
    low21 = close.rolling(21, min_periods=21).min()

    # 日线政权距离：对应旧策略 close vs MA50
    out["my_ma50_gap"] = close / ma50 - 1  # 均线偏离因子

    # 日线结构：对应旧策略 MA5 与 MA20 同向
    out["my_struct_gap"] = (ma5 - ma20) / close.replace(0, pd.NA)

    # 离 21 日高：多头用来量「有没有回踩空间」
    out["my_dd_from_high_21"] = close / high21 - 1

    # 离 21 日低：空头用来量「有没有反弹可卖」（正数 = 离低点涨了多少）
    out["my_dist_from_low_21"] = close / low21.replace(0, pd.NA) - 1

    # 长线位置：价格相对 MA200。闸门用，不参与打分。
    out["my_ma200_gap"] = close / ma200.replace(0, pd.NA) - 1

    # 长线斜率：百分比 + ATR 标准化。闸门用，不参与打分。
    ma200_prev = ma200.shift(MA200_SLOPE_N)
    out["my_ma200_slope"] = ma200 / ma200_prev.replace(0, pd.NA) - 1
    out["my_ma200_slope_5"] = ma200 / ma200.shift(MA200_SLOPE_N_FAST).replace(0, pd.NA) - 1

    tr = _true_range(close, high, low)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr20 = tr.rolling(ATR_SLOPE_N, min_periods=ATR_SLOPE_N).mean()
    out["my_atr_pct"] = atr14 / close.replace(0, pd.NA)
    out["my_atr_pct_20"] = atr20 / close.replace(0, pd.NA)
    out["my_ma200_slope_atr"] = (ma200 - ma200_prev) / atr20.replace(0, pd.NA)

    # ---------- RSI ----------
    # 0–100，跨股可直接比。正 IC = 越强势越好；负 IC 可以 invert 当超买反转。
    out["my_rsi_6"] = rsi(close, 6)
    out["my_rsi_14"] = rsi(close, 14)
    out["my_rsi_21"] = rsi(close, 21)
    out["my_rsi_14_gap"] = out["my_rsi_14"] - 50  # 距离中轴，正数偏多

    # ---------- 布林带 20, 2 ----------
    bb_mid = close.rolling(BB_N, min_periods=BB_N).mean()
    bb_std = close.rolling(BB_N, min_periods=BB_N).std()
    bb_upper = bb_mid + BB_K * bb_std
    bb_lower = bb_mid - BB_K * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, pd.NA)
    out["my_bb_pctb"] = (close - bb_lower) / bb_range  # 0=下轨, 0.5=中轨, 1=上轨
    out["my_bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, pd.NA)
    out["my_bb_gap"] = close / bb_mid.replace(0, pd.NA) - 1

    # ---------- MACD 12/26/9，除以收盘价才能跨股排名 ----------
    macd, signal, hist = _macd_parts(close)
    px = close.replace(0, pd.NA)
    out["my_macd"] = macd / px
    out["my_macd_signal"] = signal / px
    out["my_macd_hist"] = hist / px
    out["my_macd_hist_chg"] = hist.diff() / px  # 柱状图是否正在放大

    return out
