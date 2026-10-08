from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ndxbots.backtest.execution import EPS, Order, execute_orders
from ndxbots.config import Settings
from ndxbots.data.ingest import panel_path
from ndxbots.factors.compute import factor_path
from ndxbots.regime.exposure import gross_by_date, load_regime_table
from ndxbots.strategy.scores import build_score_table
from ndxbots.strategy.targets import decide_quantity_targets


def bt_dir(settings: Settings) -> Path:
    return settings.data_root / "backtest"


def _slice_dates(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = df[df["date"] >= pd.Timestamp(settings.bt_start)]
    if settings.bt_end:
        out = out[out["date"] <= pd.Timestamp(settings.bt_end)]
    return out


def run_backtest(
    settings: Settings,
    panel: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    *,
    scored: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | str | dict]:
    """After-close signals and next-close fills using fixed requested quantities.

    Selection warms up on all supplied factor history; the evaluation account
    starts in cash. Short proceeds are cash, with no short-collateral or borrow
    model. Quantities use input price units, not broker shares for QFQ inputs.
    """
    if settings.bt_exec != "next_close":
        raise ValueError("Only backtest.exec=next_close is supported by the quantity ledger")
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

    benchmark = settings.benchmark
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    if benchmark not in close.columns:
        raise SystemExit(f"面板里没有基准 {benchmark}")
    calendar = pd.DatetimeIndex(panel.loc[panel.code.eq(benchmark), "date"].unique()).sort_values()
    dates = calendar[calendar >= pd.Timestamp(settings.bt_start)]
    if settings.bt_end:
        dates = dates[dates <= pd.Timestamp(settings.bt_end)]
    if not len(dates):
        raise ValueError("No benchmark sessions in the requested backtest interval")
    codes = np.array(sorted(set(scored.code) | (set(close.columns) - {benchmark})), dtype=str)
    raw_frame = close.reindex(index=calendar, columns=codes).where(lambda frame: frame > 0)
    raw = raw_frame.reindex(dates).to_numpy(float)
    valuation = raw_frame.ffill().reindex(dates).to_numpy(float)
    selected_rows = scored.loc[scored["in_hold"], ["date", "code", "side"]].copy()
    selected_rows["sign"] = selected_rows.side.map({"long": 1, "short": -1})
    if selected_rows["sign"].isna().any():
        raise ValueError("Selected positions must have a long/short side")
    selected = selected_rows.pivot(index="date", columns="code", values="sign")
    selected = selected.reindex(index=dates, columns=codes).fillna(0).to_numpy(np.int8)
    if np.count_nonzero(selected, axis=1).max(initial=0) > settings.max_hold:
        raise ValueError("Selection exceeds the configured position count")
    pool = scored.pivot(index="date", columns="code", values="in_pool")
    pool = pool.reindex(index=dates, columns=codes).fillna(False).astype(bool).to_numpy()
    gross = pd.Series(1.0, index=calendar)
    if settings.use_gross_ladder:
        regime = load_regime_table(settings)
        if regime is not None:
            gross = gross_by_date(regime, settings).reindex(calendar).fillna(1.0)
    gross = gross.reindex(dates).to_numpy(float)
    if not np.isfinite(gross).all() or (gross < 0).any():
        raise ValueError("Sentiment gross targets must be finite and nonnegative")
    weekdays = tuple(getattr(settings, "rebalance_weekdays", (1, 4)))
    if not weekdays or any(day not in range(7) for day in weekdays):
        raise ValueError("Rebalance weekdays must be integers in 0..6")

    quantity = np.zeros(len(codes), dtype=float)
    cash = float(settings.initial_cash)
    previous_nav = cash
    previous_prices = valuation[0]
    effective_gross = 1.0
    reductions: dict[int, float] = {}
    pending: list[Order] = []
    curve_rows, holding_rows, fill_rows, target_rows, log_rows = [], [], [], [], []
    audit = {"signal_rows": 0, "scheduled_reviews": 0, "order_rows": 0,
             "signal_missing_quotes": 0, "execution_missing_quotes": 0,
             "execution_capacity_blocks": 0, "cash_guard_partial_fills": 0,
             "stale_position_marks": 0, "last_unexecuted_order_rows": 0,
             "max_pnl_conservation_error": 0.0, "max_fill_nav_conservation_error": 0.0}
    for t, day in enumerate(dates):
        prices = valuation[t]
        held = np.abs(quantity) > EPS
        if not np.isfinite(prices[held]).all():
            raise ValueError(f"Cannot value an existing position on {day.date()}")
        mark_pnl = float(np.dot(quantity[held], prices[held] - previous_prices[held])) if t else 0.0
        before = float(cash + np.dot(quantity[held], prices[held]))
        error = abs(before - previous_nav - mark_pnl)
        audit["max_pnl_conservation_error"] = max(audit["max_pnl_conservation_error"], error)
        if error > max(1e-7, abs(previous_nav) * 1e-10):
            raise AssertionError("Mark-to-market accounting failed")
        for j in np.flatnonzero(held & ~np.isfinite(raw[t])):
            audit["stale_position_marks"] += 1
            log_rows.append({"date": day, "code": codes[j], "event": "stale_valuation",
                             "detail": "carried latest known past adjusted close"})
        if pending:
            if t == 0 or any(order.signal_date != dates[t - 1] for order in pending):
                raise AssertionError("Fill is not on the next benchmark session")
            cash, quantity, fills, blocked = execute_orders(
                cash, quantity, pending, raw[t], settings.cost_bps, codes, settings.max_hold
            )
        else:
            fills, blocked = [], []
        for item in blocked:
            audit["execution_missing_quotes"] += int(item["event"] == "missing_execution_quote")
            audit["execution_capacity_blocks"] += int(item["event"] == "slot_cap_after_blocked_exit")
            log_rows.append({"date": day, "code": codes[item["index"]], "event": item["event"],
                             "detail": f"{item['leg']}: {item['reason']}; unfilled order discarded"})
        traded = sum(abs(fill["signed_notional"]) for fill in fills)
        cost = sum(fill["cost"] for fill in fills)
        for fill in fills:
            audit["cash_guard_partial_fills"] += int(fill["cash_constrained_partial"])
            fill["date"], fill["code"] = day, codes[fill.pop("index")]
            fill_rows.append(fill)
        held = np.abs(quantity) > EPS
        signed_value = float(np.dot(quantity[held], prices[held]))
        nav = float(cash + signed_value)
        error = abs(nav - (before - cost))
        audit["max_fill_nav_conservation_error"] = max(audit["max_fill_nav_conservation_error"], error)
        if error > max(1e-7, abs(before) * 1e-10):
            raise AssertionError("Fill accounting failed")
        if nav <= 0:
            raise ValueError(f"Nonpositive account NAV on {day.date()}")
        if cash < -1e-7:
            raise ValueError(f"Risk-reducing short cover requires financing on {day.date()}; margin model unsupported")
        actual_gross = float(np.dot(np.abs(quantity[held]), prices[held]) / nav)
        decision = decide_quantity_targets(day, selected[t], quantity, raw[t], nav,
                                           gross[t], effective_gross, reductions, weekdays)
        effective_gross, reductions = decision.effective_gross, decision.reduction_targets
        audit["scheduled_reviews"] += int(day.weekday() in weekdays)
        pending = []
        union = (selected[t] != 0) | held | (np.abs(decision.quantities) > EPS)
        for j in np.flatnonzero(union):
            target = float(decision.quantities[j])
            delta = target - quantity[j]
            valid = bool(np.isfinite(raw[t, j]) and raw[t, j] > 0)
            reason = decision.reasons.get(int(j), "hold_quantity")
            status = "hold"
            if abs(delta) > EPS:
                if valid:
                    pending.append(Order(int(j), target, reason, day, nav))
                    audit["order_rows"] += 1
                    status = "queued_next_close"
                else:
                    audit["signal_missing_quotes"] += 1
                    status = "blocked_missing_signal_quote"
                    log_rows.append({"date": day, "code": codes[j], "event": "missing_signal_quote",
                                     "detail": "quantity order blocked; risk reduction reconsidered daily"})
            elif day.weekday() in weekdays and selected[t, j] and not valid:
                audit["signal_missing_quotes"] += 1
                status = "blocked_missing_signal_quote"
            side_sign = (np.sign(target) if abs(target) > EPS else
                         np.sign(quantity[j]) if abs(quantity[j]) > EPS else selected[t, j])
            mark = prices[j]
            target_rows.append({"date": day, "code": codes[j], "side": "long" if side_sign > 0 else "short",
                                "current_quantity": float(quantity[j]), "target_quantity": target,
                                "delta_quantity": float(delta), "current_weight": float(quantity[j] * mark / nav) if np.isfinite(mark) else 0.0,
                                "target_weight": float(target * mark / nav) if np.isfinite(mark) else 0.0,
                                "signal_price": float(raw[t, j]), "signal_nav": nav, "reason": reason,
                                "scheduled_review": day.weekday() in weekdays, "in_pool": bool(pool[t, j]),
                                "in_selected": bool(selected[t, j]), "order_status": status})
        for j in np.flatnonzero(held):
            holding_rows.append({"date": day, "code": codes[j], "side": "long" if quantity[j] > 0 else "short",
                                 "quantity": float(quantity[j]), "close": float(prices[j]), "w": float(quantity[j] * prices[j] / nav),
                                 "gross_target": float(gross[t]), "value": float(quantity[j] * prices[j])})
        curve_rows.append({"date": day, "port_ret": nav / previous_nav - 1.0,
                           "port_ret_gross": mark_pnl / previous_nav, "turnover": 0.5 * traded / before,
                           "n_hold": int(held.sum()), "n_long": int((quantity > EPS).sum()), "n_short": int((quantity < -EPS).sum()),
                           "net_exp": signed_value / nav, "gross_exp": actual_gross, "gross_target": float(gross[t]),
                           "equity": nav, "cash": cash, "signed_position_value": signed_value,
                           "mark_pnl": mark_pnl, "cost": cost, "abs_traded_notional": traded,
                           "effective_gross_budget": effective_gross, "fills": len(fills), "pending_order_rows": len(pending)})
        audit["signal_rows"] += 1
        previous_nav, previous_prices = nav, prices

    curve = pd.DataFrame(curve_rows)
    benchmark_close = close[benchmark].reindex(calendar).ffill().reindex(dates)
    if benchmark_close.isna().any() or (benchmark_close <= 0).any():
        raise ValueError("Benchmark cannot be valued in the evaluation interval")
    curve["bench_ret"] = benchmark_close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    curve["bench_equity"] = settings.initial_cash * benchmark_close.to_numpy() / benchmark_close.iloc[0]
    curve["excess"] = curve["port_ret"] - curve["bench_ret"]
    holdings = pd.DataFrame(holding_rows, columns=["date", "code", "side", "quantity", "close", "w", "gross_target", "value"])
    metadata_cols = [column for column in scored if column not in holdings.columns and column not in {"in_hold", "in_hold_long", "in_hold_short"}]
    holdings = holdings.merge(scored[["date", "code", *metadata_cols]], on=["date", "code"], how="left")
    audit["last_unexecuted_order_rows"] = len(pending)
    target_columns = ["date", "code", "side", "current_quantity", "target_quantity", "delta_quantity", "current_weight",
                      "target_weight", "signal_price", "signal_nav", "reason", "scheduled_review", "in_pool", "in_selected", "order_status"]
    fill_columns = ["quantity", "requested_quantity", "price", "signed_notional", "cost", "cash_constrained_partial", "leg", "reason", "signal_date", "signal_nav", "date", "code"]
    return {"curve": curve, "holdings": holdings, "stats": _summarize(curve, settings),
            "scored": _slice_dates(scored, settings), "fills": pd.DataFrame(fill_rows, columns=fill_columns),
            "targets": pd.DataFrame(target_rows, columns=target_columns),
            "logs": pd.DataFrame(log_rows, columns=["date", "code", "event", "detail"]), "audit": audit}


def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min()) if len(equity) else 0.0


def _summarize(curve: pd.DataFrame, settings: Settings) -> str:
    if curve.empty:
        return "回测无有效收益行"
    p = curve.port_ret.fillna(0)
    years = max((curve.date.iloc[-1] - curve.date.iloc[0]).days / 365.25, 1 / 365.25)
    end, bench_end = float(curve.equity.iloc[-1]), float(curve.bench_equity.iloc[-1])
    ann = (end / settings.initial_cash) ** (1 / years) - 1
    bench_ann = (bench_end / settings.initial_cash) ** (1 / years) - 1
    vol = float(p.std(ddof=1) * np.sqrt(252)) if len(p) > 1 else float("nan")
    sharpe = float(p.mean() / p.std(ddof=1) * np.sqrt(252)) if p.std(ddof=1) else float("nan")
    return "\n".join([
        "引擎 ndxbots.backtest  W11 完整持仓 / 固定数量研究账本",
        f"区间 {curve.date.iloc[0].date()} ~ {curve.date.iloc[-1].date()}  交易日 {len(curve)}",
        f"期初 {settings.initial_cash:,.2f}  期末 {end:,.2f}  年化 {ann*100:.2f}%",
        f"QQQ 期末 {bench_end:,.2f}  年化 {bench_ann*100:.2f}%",
        f"波动 {vol*100:.2f}%  Sharpe {sharpe:.3f}  最大回撤 {_max_dd(curve.equity)*100:.2f}%",
        f"年换手（买卖各半）{curve.turnover.sum()/years:.2f}  日均持仓 {curve.n_hold.mean():.2f}",
        f"成本单边 {settings.cost_bps:.1f}bp  持仓上限 {settings.max_hold}  情绪阶梯={settings.use_gross_ladder}",
        "成交假设: T 日收盘后根据当日净值和价格锁定数量；T+1 基准交易日收盘成交。",
        "普通调仓按设定工作日评估；每日只退出失效方向或降低情绪仓位，增加仓位等待普通调仓。",
        "买入资金不足按比例部分成交；先平旧方向再开反向。目标仓位不是实际毛曝险硬上限。",
        "价格为输入数据单位；QFQ 数量并非券商实际股数。未模拟借券、短仓保证金、融资利息、股息现金流或手数约束。",
    ])
