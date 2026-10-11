from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ndxbots.config import Settings
from ndxbots.data.ingest import panel_path
from ndxbots.factors.compute import factor_path
from ndxbots.strategy.scores import build_score_table


def bt_dir(settings: Settings) -> Path:
    return settings.data_root / "backtest"


def run_backtest(
    settings: Settings,
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    *,
    scored: pd.DataFrame | None = None,
    minute_store=None,
) -> dict[str, pd.DataFrame | str | dict]:
    """After-close signals with mandatory next-session funded-budget execution.

    Selection warms up on all supplied factor history; the evaluation account
    starts in cash. Short proceeds are cash, with no short-collateral or borrow
    model. Quantities use input price units, not broker shares for QFQ inputs.
    """
    if settings.bt_exec != "next_intraday":
        raise ValueError("backtest.exec=next_intraday replaces the old next_close model")
    if not np.isfinite(settings.initial_cash) or settings.initial_cash <= 0:
        raise ValueError("Initial cash must be finite and positive")
    if not np.isfinite(settings.cost_bps) or settings.cost_bps < 0:
        raise ValueError("Cost must be finite and nonnegative")
    if settings.max_hold < 1:
        raise ValueError("Maximum holdings must be positive")
    if panel is None:
        path = panel_path(settings)
        if not path.exists():
            raise SystemExit(f"还没有行情面板: {path}")
        panel = pd.read_parquet(path)
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("Price panel contains duplicate date/code rows")
    if scored is None:
        if factors is None:
            path = factor_path(settings)
            if not path.exists():
                raise SystemExit(f"还没有因子表: {path}\n请先运行 python -m ndxbots.factors.compute")
            factors = pd.read_parquet(path)
        factors = factors.copy()
        factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
        scored = build_score_table(factors, settings)
    else:
        scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    if scored.duplicated(["date", "code"]).any():
        raise ValueError("Scores contain duplicate date/code rows")

    from ndxbots.backtest.intraday import run_intraday_backtest
    return run_intraday_backtest(settings, panel, factors, scored, minute_store=minute_store)
