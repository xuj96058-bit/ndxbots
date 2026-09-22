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

    # 日线政权距离：对应旧策略 close vs MA50
    out["my_ma50_gap"] = close / ma50 - 1

    # 日线结构：对应旧策略 MA5 与 MA20 同向
    out["my_struct_gap"] = (ma5 - ma20) / close.replace(0, pd.NA)

    # 离 21 日高：用来量「有没有空间」，不是单独开仓条件
    out["my_dd_from_high_21"] = close / high21 - 1

    # 长线政权：方案 B 硬过滤
    out["my_ma200_gap"] = close / ma200 - 1

    if high is not None and low is not None:
        prev = close.shift(1)
        tr = np.maximum((high - low).abs(), (high - prev).abs())
        tr = np.maximum(tr, (low - prev).abs())
        atr = pd.DataFrame(tr, index=close.index, columns=close.columns)
        atr = atr.rolling(14, min_periods=14).mean()
        out["my_atr_pct"] = atr / close.replace(0, pd.NA)
    else:
        ret = close.pct_change(fill_method=None)
        out["my_atr_pct"] = ret.rolling(14, min_periods=14).std()

    return out
