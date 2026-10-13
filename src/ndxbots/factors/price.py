from __future__ import annotations

import numpy as np
import pandas as pd


def pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    df = panel.pivot(index="date", columns="code", values=col).sort_index()
    return df


def daily_return(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def trailing_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close / close.shift(window) - 1


def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window, min_periods=window).std()


def dist_from_high(close: pd.DataFrame, window: int) -> pd.DataFrame:
    high = close.rolling(window, min_periods=window).max()
    return close / high - 1


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # A complete rising path has RSI 100; a complete flat path is neutral.
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def volume_ratio(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    avg = volume.rolling(window, min_periods=window).mean()
    return volume / avg.replace(0, np.nan)


def excess_return(stock_ret: pd.DataFrame, benchmark_ret: pd.Series) -> pd.DataFrame:
    aligned = stock_ret.sub(benchmark_ret, axis=0)
    return aligned


def melt_factor(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    out = wide.stack(future_stack=True).rename(name).reset_index()
    out.columns = ["date", "code", name]
    return out

