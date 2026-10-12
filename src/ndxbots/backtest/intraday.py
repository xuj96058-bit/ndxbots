"""First-group T+1 execution, with cash reservations and causal fill accounting.

Daily selection stays in strategy.targets. This account uses adjusted, possibly
fractional research units and never sends orders to a broker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ndxbots.backtest.execution import EPS, Order
from ndxbots.backtest.intraday_prices import IntradayPriceStore
from ndxbots.backtest.intraday_schedule import Policy, execution_schedule
from ndxbots.regime.exposure import gross_by_date, load_regime_table
from ndxbots.strategy.targets import decide_quantity_targets

ENTRY_POLICY = Policy("entry_d30_f50_gap1_atr", delay=30, first_fraction=.5,
                      gap_threshold=.01, remainder="atr")
EXIT_POLICY = Policy("exit_d15_immediate", delay=15, exit_mode="immediate")

FILL_COLUMNS = [
    "date", "code", "quantity", "requested_quantity", "price", "signed_notional", "cost",
    "cash_constrained_partial", "leg", "reason", "signal_date", "signal_nav", "time", "time_et",
    "forced", "trigger", "action", "side", "planned_notional", "reserved_budget", "deferred_minutes",
]
TARGET_COLUMNS = [
    "date", "code", "side", "current_quantity", "target_quantity", "delta_quantity", "current_weight",
    "target_weight", "signal_price", "signal_nav", "reason", "scheduled_review", "in_pool", "in_selected",
    "order_status", "urgent", "sizing_mode", "planned_addition_notional", "execution_mode", "entry_time_et",
    "exit_time_et", "deadline_rule_et",
]
COMPLETION_COLUMNS = [
    "date", "signal_date", "code", "addition", "complete", "planned_target_quantity", "actual_quantity",
    "quantity_shortfall", "planned_notional", "planned_addition_notional", "funded_budget", "spent_notional",
    "cash_capped", "cash_deferred", "slot_deferred", "reverse_deferred", "discarded_unfunded_limit",
    "late_entry", "first_due", "first_fill", "last_fill", "max_delay_minutes",
]


class IncompleteExecutionError(ValueError):
    """A required T+1 budget or reduction could not be completed legally."""


@dataclass
class Plan:
    order: Order
    delta: float
    action: str
    side: int
    urgent: bool
    reversal: bool
    schedule: list[dict]
    planned_notional: float
    due_fraction: float = 0.0
    filled_quantity: float = 0.0
    spent_notional: float = 0.0
    reserved_budget: float | None = None
    reserved_time: int | None = None
    schedule_index: int = 0
    first_due: int | None = None
    first_fill: int | None = None
    last_fill: int | None = None
    cash_deferred: bool = False
    slot_deferred: bool = False
    reverse_deferred: bool = False
    capped: bool = False
    forced_fraction: float = 0.0
    last_trigger: str = ""
    discarded_unfunded_limit: bool = False


def _atr20(panel, factors, dates, codes, close):
    if factors is not None and "my_atr_pct_20" in factors:
        factors = factors.copy()
        factors["date"] = pd.to_datetime(factors.date).dt.normalize()
        if factors.duplicated(["date", "code"]).any():
            raise ValueError("Factors contain duplicate date/code rows")
        return (factors.pivot(index="date", columns="code", values="my_atr_pct_20")
                .reindex(index=dates, columns=codes).to_numpy(float) * close)
    if not {"high", "low"}.issubset(panel.columns):
        raise ValueError("T-day ATR20 needs my_atr_pct_20 factors or full daily OHLC history")
    pieces = []
    for code, block in panel.groupby("code", sort=False):
        block = block.sort_values("date")
        previous = block.close.shift(1)
        tr = pd.concat([block.high - block.low, (block.high - previous).abs(),
                        (block.low - previous).abs()], axis=1).max(axis=1)
        pieces.append(pd.DataFrame({"date": block.date, "code": code,
                                    "atr": tr.rolling(20, min_periods=20).mean()}))
    return (pd.concat(pieces).pivot(index="date", columns="code", values="atr")
            .reindex(index=dates, columns=codes).to_numpy(float))


class IntradaySimulator:
    def __init__(self, settings, panel, factors, scored, *, minute_store=None):
        self.settings = settings
        self.initial_cash, self.max_hold = float(settings.initial_cash), int(settings.max_hold)
        self.weekdays = tuple(settings.rebalance_weekdays)
        if not self.weekdays or any(day not in range(7) for day in self.weekdays):
            raise ValueError("Rebalance weekdays must be integers in 0..6")
        self.panel, self.scored = panel.copy(), scored.copy()
        benchmark = settings.benchmark
        if "open" not in panel:
            raise ValueError("Intraday execution requires daily open and close columns")
        close = panel.pivot(index="date", columns="code", values="close").sort_index()
        if benchmark not in close:
            raise ValueError(f"Price panel has no benchmark {benchmark}")
        calendar = pd.DatetimeIndex(panel.loc[panel.code.eq(benchmark), "date"].unique()).sort_values()
        self.dates = calendar[calendar >= pd.Timestamp(settings.bt_start)]
        if settings.bt_end:
            self.dates = self.dates[self.dates <= pd.Timestamp(settings.bt_end)]
        if not len(self.dates):
            raise ValueError("No benchmark sessions in the requested backtest interval")
        self.codes = np.array(sorted(set(scored.code) | (set(panel.code) - {benchmark})), dtype=str)
        self.code_index = {code: i for i, code in enumerate(self.codes)}
        raw_frame = close.reindex(index=calendar, columns=self.codes).where(lambda x: x > 0)
        self.raw = raw_frame.reindex(self.dates).to_numpy(float)
        self.valuation = raw_frame.ffill().reindex(self.dates).to_numpy(float)
        self.open = (panel.pivot(index="date", columns="code", values="open")
                     .reindex(index=self.dates, columns=self.codes).to_numpy(float))
        selected = scored.loc[scored.in_hold, ["date", "code", "side"]].copy()
        selected["sign"] = selected.side.map({"long": 1, "short": -1})
        if selected.sign.isna().any():
            raise ValueError("Selected positions must have a long/short side")
        self.selected = (selected.pivot(index="date", columns="code", values="sign")
                         .reindex(index=self.dates, columns=self.codes).fillna(0).to_numpy(np.int8))
        directional = scored[["date", "code", "side"]].copy()
        directional["sign"] = directional.side.map({"long": 1, "short": -1}).fillna(0)
        self.directional = (directional.pivot(index="date", columns="code", values="sign")
                            .reindex(index=self.dates, columns=self.codes).fillna(0).to_numpy(np.int8))
        self.pool = (scored.pivot(index="date", columns="code", values="in_pool")
                     .reindex(index=self.dates, columns=self.codes).fillna(False).astype(bool).to_numpy())
        if np.count_nonzero(self.selected, axis=1).max(initial=0) > self.max_hold:
            raise ValueError("Selection exceeds the configured holding count")
        gross = pd.Series(1.0, index=calendar)
        if settings.use_gross_ladder:
            regime = load_regime_table(settings)
            if regime is not None:
                gross = gross_by_date(regime, settings).reindex(calendar).fillna(1.0)
        self.gross = gross.reindex(self.dates).to_numpy(float)
        if not np.isfinite(self.gross).all() or (self.gross < 0).any():
            raise ValueError("Sentiment gross targets must be finite and nonnegative")
        # Benchmark Hurst is known at the T-day close; these gross targets feed
        # the existing T+1 order queue, with no change to selection or cash limits.
        self.long_entry_allowed = np.ones(self.raw.shape, dtype=bool)
        if factors is not None and "my_hurst_126" in factors.columns:
            hurst_factors = factors[["date", "code", "my_hurst_126"]].copy()
            hurst_factors["date"] = pd.to_datetime(hurst_factors.date).dt.normalize()
            if hurst_factors.duplicated(["date", "code"]).any():
                raise ValueError("Hurst factors contain duplicate date/code rows")
            hurst_factors["my_hurst_126"] = pd.to_numeric(hurst_factors.my_hurst_126)
            stock_hurst = (hurst_factors.pivot(index="date", columns="code", values="my_hurst_126")
                           .reindex(index=self.dates, columns=self.codes)
                           .to_numpy(dtype=float, na_value=np.nan))
            # Legacy mode reproduces the score table's regime-only Hurst gate;
            # strict mode additionally checks actual new account positions.
            if getattr(settings, "hurst_gate_mode", "strict_entry") == "strict_entry":
                self.long_entry_allowed = np.isfinite(stock_hurst) & (stock_hurst > 0.67)
            benchmark_hurst = hurst_factors.loc[hurst_factors.code.eq(benchmark)]
            if not benchmark_hurst.empty:
                hurst = benchmark_hurst.set_index("date")["my_hurst_126"]
                hurst = hurst.reindex(self.dates).to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(hurst).any():
                    raise ValueError("Benchmark Hurst factors must be finite or missing")
                scale = np.ones(len(self.dates))
                scale[hurst > 0.75] = 1.3
                scale[hurst < 0.55] = 0.5
                self.gross = np.clip(self.gross * scale, 0.0, 1.5)
        self.atr = _atr20(panel, factors, self.dates, self.codes, self.raw)
        benchmark_close = close[benchmark].reindex(calendar).ffill().reindex(self.dates)
        if benchmark_close.isna().any() or (benchmark_close <= 0).any():
            raise ValueError("Benchmark cannot be valued in the evaluation interval")
        self.bench_ret = benchmark_close.pct_change(fill_method=None).fillna(0.0).to_numpy()
        self.bench_equity = self.initial_cash * benchmark_close.to_numpy() / benchmark_close.iloc[0]
        self.minute_root = getattr(settings, "intraday_root", None) or settings.data_root / "intraday" / "kline"
        self.minute_store = minute_store
        self._bars, self._prepared_bars, self._schedules = {}, {}, {}
        self._case_bar_keys = set()

    def bars(self, code, day, t):
        day = pd.Timestamp(day).normalize()
        key = (code, day)
        self._case_bar_keys.add(key)
        if key not in self._bars:
            if self.minute_store is None:
                self.minute_store = IntradayPriceStore(self.minute_root)
            prepared = self.minute_store.bars(code, day, float(self.open[t, self.code_index[code]]))
            self._prepared_bars[key] = prepared
            self._bars[key] = pd.DataFrame({"bar_start": prepared.starts, "bar_end": prepared.ends,
                                           "open": prepared.opens, "high": prepared.highs,
                                           "low": prepared.lows, "close": prepared.closes})
        return self._bars[key]

    def _schedule(self, code, t, side, action, policy, urgent):
        j = self.code_index[code]
        if urgent and action == "exit":
            price = float(self.open[t, j])
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f"{code} {self.dates[t].date()}: missing urgent-exit open")
            return [{"time": 570, "fraction": 1.0, "price": price,
                     "forced": False, "trigger": "urgent_open"}]
        self.bars(code, self.dates[t], t)
        cache_key = (code, t, side, action, urgent)
        if cache_key not in self._schedules:
            rows = execution_schedule(self._prepared_bars[(code, self.dates[t])], side, policy,
                                      float(self.raw[t - 1, j]), float(self.atr[t - 1, j]),
                                      action=action, urgent=urgent)
            if not rows or abs(sum(row["fraction"] for row in rows) - 1) > 1e-8:
                raise ValueError("An execution leg must schedule its full budget or reduction")
            self._schedules[cache_key] = rows
        return self._schedules[cache_key]

    def _deferred_price(self, code, t, minute, scheduled):
        at_time = [row for row in scheduled if int(row["time"]) == minute]
        if at_time:
            return float(at_time[-1]["price"])
        bars = self.bars(code, self.dates[t], t)
        if minute == int(bars.bar_end.iloc[-1]):
            return float(bars.close.iloc[-1])
        exact = bars.loc[bars.bar_start.eq(minute)]
        if not exact.empty:
            return float(exact.open.iloc[0])
        raise ValueError(f"{code} {self.dates[t].date()}: cannot price deferred fill at {minute}")

    def simulate(self, cost_bps=10.0) -> dict:
        entry_policy = ENTRY_POLICY
        exit_policy = EXIT_POLICY
        holding_rows = []
        log_rows = []
        if not np.isfinite(cost_bps) or cost_bps < 0:
            raise ValueError('cost_bps must be finite and nonnegative')
        fee = float(cost_bps) / 10000.0
        self._case_bar_keys = set()
        quantity = np.zeros(len(self.codes), float)
        cash = previous_nav = self.initial_cash
        previous_prices = self.valuation[0]
        effective_gross = 1.0
        reductions: dict[int, float] = {}
        pending: list[Order] = []
        urgent_by_index: dict[int, bool] = {}
        curve_rows, fill_rows, completion_rows, target_rows = ([], [], [], [])
        audit = {
            'sizing_mode': 'budget',
            'signal_rows': 0,
            'scheduled_reviews': 0,
            'order_rows': 0,
            'executed_order_rows': 0,
            'fully_completed_order_rows': 0,
            'incomplete_order_rows': 0,
            'addition_order_rows': 0,
            'completed_addition_order_rows': 0,
            'cash_guard_partial_fills': 0,
            'cash_deferral_legs': 0,
            'slot_deferral_legs': 0,
            'reverse_deferral_legs': 0,
            'cash_capped_budget_legs': 0,
            'zero_funded_addition_legs': 0,
            'planned_addition_notional': 0.0,
            'reserved_addition_notional': 0.0,
            'cash_cap_budget_reduction': 0.0,
            'forced_fill_rows': 0,
            'discarded_unfunded_limit_legs': 0,
            'late_entry_orders': 0,
            'deadline_missed_order_rows': 0,
            'signal_missing_quotes': 0,
            'execution_missing_quotes': 0,
            'execution_capacity_blocks': 0,
            'stale_position_marks': 0,
            'last_unexecuted_order_rows': 0,
            'max_hold': 0,
            'max_pnl_conservation_error': 0.0,
            'max_fill_nav_conservation_error': 0.0,
        }
        for t, day in enumerate(self.dates):
            prices = self.valuation[t]
            held_start = np.abs(quantity) > EPS
            if not np.isfinite(prices[held_start]).all():
                raise ValueError(f'Cannot value holdings on {day.date()}')
            mark_pnl = float(np.dot(quantity[held_start], prices[held_start] - previous_prices[held_start])) if t else 0.0
            before = float(cash + np.dot(quantity[held_start], prices[held_start]))
            pnl_error = abs(before - previous_nav - mark_pnl)
            audit['max_pnl_conservation_error'] = max(audit['max_pnl_conservation_error'], pnl_error)
            if pnl_error > max(1e-07, abs(previous_nav) * 1e-10):
                raise AssertionError('Previous-close mark accounting failed')
            audit['stale_position_marks'] += int(np.count_nonzero(held_start & ~np.isfinite(self.raw[t])))
            dayfills, daycompletions = ([], [])
            if pending:
                if t == 0 or any((order.signal_date != self.dates[t - 1] for order in pending)):
                    raise AssertionError('Order not executed on the next benchmark session')
                cash, quantity, dayfills, daycompletions = self._execute_day(t, cash, quantity, pending, urgent_by_index, entry_policy, exit_policy, fee, 'budget')
            traded = float(sum((abs(fill['signed_notional']) for fill in dayfills)))
            cost = float(sum((fill['cost'] for fill in dayfills)))
            execution_pnl = float(sum((fill['quantity'] * (prices[fill['index']] - fill['price']) for fill in dayfills)))
            held = np.abs(quantity) > EPS
            signed_value = float(np.dot(quantity[held], prices[held]))
            nav = float(cash + signed_value)
            fill_error = abs(nav - (before + execution_pnl - cost))
            audit['max_fill_nav_conservation_error'] = max(audit['max_fill_nav_conservation_error'], fill_error)
            if fill_error > max(1e-07, abs(before) * 1e-10):
                raise AssertionError('Intraday-fill-to-close conservation failed')
            if cash < -1e-07 or nav <= 0:
                raise ValueError(f'Financing or nonpositive NAV on {day.date()}; unsupported')
            actual_gross = float(np.dot(np.abs(quantity[held]), prices[held]) / nav)
            old_effective_gross = effective_gross
            decision = decide_quantity_targets(day, self.selected[t], quantity, self.raw[t], nav, self.gross[t], effective_gross, reductions, self.weekdays)
            effective_gross, reductions = (decision.effective_gross, decision.reduction_targets)
            pending, urgent_by_index = ([], {})
            union = (self.selected[t] != 0) | held | (np.abs(decision.quantities) > EPS)
            for j in np.flatnonzero(union):
                target = float(decision.quantities[j])
                reason = decision.reasons.get(int(j), 'hold_quantity')
                # Selection can predate bt_start; a selected name is not
                # necessarily already held by this cash account.
                hurst_blocked = target > EPS and quantity[j] <= EPS and not self.long_entry_allowed[t, j]
                if hurst_blocked:
                    target = 0.0  # A blocked reversal still closes its old short.
                    reason = 'new_long_hurst_blocked'
                delta = target - quantity[j]
                valid = bool(np.isfinite(self.raw[t, j]) and self.raw[t, j] > 0)
                if abs(delta) > EPS:
                    if valid:
                        pending.append(Order(int(j), target, reason, day, nav))
                        urgent_by_index[int(j)] = bool(held[j] and (self.directional[t, j] == 0 or self.directional[t, j] != np.sign(quantity[j]) or self.gross[t] < old_effective_gross - EPS or (reason == 'daily_sentiment_reduction')))
                        audit['order_rows'] += 1
                    else:
                        audit['signal_missing_quotes'] += 1
                        log_rows.append({
                            'date': day, 'code': self.codes[j], 'event': 'missing_signal_quote',
                            'detail': 'order blocked; no future quote substituted',
                        })
                target_rows.append({
                    'date': day,
                    'code': self.codes[j],
                    'side': 'long' if (target if abs(target) > EPS else quantity[j] if abs(quantity[j]) > EPS else self.selected[t, j]) > 0 else 'short',
                    'current_quantity': float(quantity[j]),
                    'target_quantity': target,
                    'delta_quantity': delta,
                    'current_weight': float(quantity[j] * prices[j] / nav) if np.isfinite(prices[j]) else 0.0,
                    'target_weight': float(target * prices[j] / nav) if np.isfinite(prices[j]) else 0.0,
                    'signal_price': float(self.raw[t, j]),
                    'signal_nav': nav,
                    'reason': reason,
                    'scheduled_review': day.weekday() in self.weekdays,
                    'in_pool': bool(self.pool[t, j]),
                    'in_selected': bool(self.selected[t, j]),
                    'order_status': 'queued_next_intraday' if abs(delta) > EPS and valid else 'blocked_missing_signal_quote' if abs(delta) > EPS else 'blocked_hurst_entry' if hurst_blocked else 'hold',
                    'urgent': urgent_by_index.get(int(j), False),
                    'sizing_mode': 'notional_budget',
                    'planned_addition_notional': (abs(target) if target * quantity[j] < 0 else max(0.0, abs(target) - abs(quantity[j]))) * float(self.raw[t, j]),
                    'execution_mode': 'next_intraday',
                    'entry_time_et': '10:00',
                    'exit_time_et': '09:30' if urgent_by_index.get(int(j), False) else '09:45',
                    'deadline_rule_et': '15:55 / half-day 12:55',
                })
            for item in daycompletions:
                audit['executed_order_rows'] += 1
                audit['fully_completed_order_rows'] += int(item['complete'])
                audit['incomplete_order_rows'] += int(not item['complete'])
                audit['addition_order_rows'] += int(item['addition'])
                audit['completed_addition_order_rows'] += int(item['addition'] and item['complete'])
                completion_rows.append(item)
            for fill in dayfills:
                audit['cash_guard_partial_fills'] += int(fill.get('cash_constrained_partial', False))
                audit['forced_fill_rows'] += int(fill.get('forced', False))
                item = dict(fill)
                item['date'], item['code'] = (day, self.codes[item.pop('index')])
                item['time_et'] = f"{item['time'] // 60:02d}:{item['time'] % 60:02d}"
                fill_rows.append(item)
            for item in daycompletions:
                audit['cash_deferral_legs'] += int(item.get('cash_deferred', False))
                audit['slot_deferral_legs'] += int(item.get('slot_deferred', False))
                audit['reverse_deferral_legs'] += int(item.get('reverse_deferred', False))
                audit['discarded_unfunded_limit_legs'] += int(item.get('discarded_unfunded_limit', False))
                audit['late_entry_orders'] += int(item.get('late_entry', False))
                audit['deadline_missed_order_rows'] += int(item.get('late_entry', False))
                if item['addition']:
                    audit['planned_addition_notional'] += float(item['planned_addition_notional'])
                    audit['reserved_addition_notional'] += float(item['funded_budget'])
                    audit['cash_cap_budget_reduction'] += max(0.0, float(item['planned_addition_notional']) - float(item['funded_budget']))
                    audit['cash_capped_budget_legs'] += int(item['cash_capped'])
                    audit['zero_funded_addition_legs'] += int(item['funded_budget'] <= EPS)
            audit['max_hold'] = max(audit['max_hold'], int(held.sum()))
            audit['signal_rows'] += 1
            audit['scheduled_reviews'] += int(day.weekday() in self.weekdays)
            curve_rows.append({
                'date': day,
                'port_ret': nav / previous_nav - 1,
                'port_ret_gross': (mark_pnl + execution_pnl) / previous_nav,
                'turnover': 0.5 * traded / before,
                'n_hold': int(held.sum()),
                'n_long': int((quantity > EPS).sum()),
                'n_short': int((quantity < -EPS).sum()),
                'net_exp': signed_value / nav,
                'gross_exp': actual_gross,
                'gross_target': float(self.gross[t]),
                'equity': nav,
                'cash': cash,
                'signed_position_value': signed_value,
                'mark_pnl': mark_pnl,
                'execution_pnl': execution_pnl,
                'cost': cost,
                'abs_traded_notional': traded,
                'effective_gross_budget': effective_gross,
                'fills': len(dayfills),
                'pending_order_rows': len(pending),
            })
            for j in np.flatnonzero(held):
                holding_rows.append({
                    'date': day,
                    'code': self.codes[j],
                    'side': 'long' if quantity[j] > 0 else 'short',
                    'quantity': float(quantity[j]),
                    'close': float(prices[j]),
                    'w': float(quantity[j] * prices[j] / nav),
                    'gross_target': float(self.gross[t]),
                    'value': float(quantity[j] * prices[j]),
                })
            for j in np.flatnonzero(held_start & ~np.isfinite(self.raw[t])):
                log_rows.append({
                    'date': day,
                    'code': self.codes[j],
                    'event': 'stale_valuation',
                    'detail': 'carried latest known past adjusted close',
                })
            for item in daycompletions:
                if not item['complete']:
                    log_rows.append({
                        'date': day,
                        'code': item['code'],
                        'event': 'incomplete_t1_execution',
                        'detail': 'required funded budget or reduction not completed',
                    })
            previous_nav, previous_prices = (nav, prices)
        curve = pd.DataFrame(curve_rows)
        curve['bench_ret'], curve['bench_equity'] = (self.bench_ret, self.bench_equity)
        curve['excess'] = curve.port_ret - curve.bench_ret
        audit['last_unexecuted_order_rows'] = len(pending)
        audit['completion_rate'] = audit['fully_completed_order_rows'] / max(1, audit['executed_order_rows'])
        audit['addition_completion_rate'] = audit['completed_addition_order_rows'] / max(1, audit['addition_order_rows'])
        audit['intraday_sessions_used'] = len(self._case_bar_keys)
        return {
            'curve': curve,
            'audit': audit,
            'fills': pd.DataFrame(fill_rows, columns=FILL_COLUMNS),
            'completion': pd.DataFrame(completion_rows, columns=COMPLETION_COLUMNS),
            'targets': pd.DataFrame(target_rows, columns=TARGET_COLUMNS),
            'holdings': pd.DataFrame(holding_rows, columns=['date', 'code', 'side', 'quantity', 'close', 'w', 'gross_target', 'value']),
            'logs': pd.DataFrame(log_rows, columns=['date', 'code', 'event', 'detail']),
        }

    def _execute_day(self, t: int, cash: float, quantities: np.ndarray, orders: list[Order], urgent_by_index: dict[int, bool], entry_policy: Any, exit_policy: Any, fee: float, sizing_mode: str='budget') -> tuple[float, np.ndarray, list[dict], list[dict]]:
        if sizing_mode != 'budget':
            raise ValueError('Only funded-budget execution is supported')
        if t <= 0 or t >= len(self.dates):
            raise ValueError('Execution requires a preceding signal session')
        quantities = quantities.copy()
        plans: list[Plan] = []
        for order in orders:
            j, current, target = (order.index, quantities[order.index], order.target_quantity)
            reverse = current * target < 0 and abs(current) > EPS and (abs(target) > EPS)
            if reverse:
                legs = [(-current, 'exit', int(np.sign(current))), (target, 'entry', int(np.sign(target)))]
            else:
                delta = target - current
                action = 'exit' if abs(target) < abs(current) or abs(target) <= EPS else 'entry'
                side = int(np.sign(current) if action == 'exit' else np.sign(target))
                legs = [(delta, action, side)]
            for delta, action, side in legs:
                if abs(delta) <= EPS:
                    continue
                urgent = urgent_by_index.get(j, False) and action == 'exit'
                policy = exit_policy if action == 'exit' else entry_policy
                schedule = self._schedule(self.codes[j], t, side, action, policy, urgent)
                plans.append(Plan(order, float(delta), action, side, urgent, reverse, schedule, abs(delta) * float(self.raw[t - 1, j])))
        times = {int(row['time']) for plan in plans for row in plan.schedule}
        if not plans:
            return float(cash), quantities, [], []
        # Reservations and vacant slots change only at actual fill events.
        # Include the session end to detect any missed completion deadline.
        for p in plans:
            daybars = self.bars(self.codes[p.order.index], self.dates[t], t)
            times.add(int(daybars.bar_end.iloc[-1]))
        times = sorted(times)
        fills: list[dict] = []
        snapshots: list[tuple[int, float, float, np.ndarray]] = [(min(times) - 1, cash, cash, quantities.copy())]
        price_cache: dict[tuple[int, int], float] = {}

        def plan_price(p: Plan, minute: int) -> float:
            key = (id(p), minute)
            if key in price_cache:
                return price_cache[key]
            current_rows = [row for row in p.schedule if int(row['time']) == minute]
            force_current_open = False
            for row in current_rows:
                if not str(row.get('trigger', '')).endswith('_limit'):
                    continue
                active_from = int(row.get('active_from', minute))
                if active_from >= minute:
                    continue
                past = [snapshot for snapshot in snapshots if snapshot[0] <= active_from]
                _, past_cash, past_free_cash, past_q = past[-1]
                j = p.order.index
                if p.action == 'exit':
                    slot_ok = np.sign(past_q[j]) == p.side and abs(past_q[j]) >= due_amount(p) - EPS
                    reverse_ok = True
                else:
                    slot_ok = abs(past_q[j]) > EPS or np.count_nonzero(np.abs(past_q) > EPS) < self.max_hold
                    reverse_ok = not p.reversal or abs(past_q[j]) <= EPS or np.sign(past_q[j]) == p.side
                if p.delta < 0:
                    funded = True
                elif p.action == 'entry':
                    funded = p.reserved_budget is not None and p.reserved_time is not None and (p.reserved_time <= active_from)
                else:
                    funded = past_free_cash >= due_amount(p) * float(row['price']) * (1 + fee) - 1e-07
                if not (slot_ok and reverse_ok and funded):
                    # A low inside a bar cannot fill a limit funded only after that bar started.
                    p.discarded_unfunded_limit = True
                    p.slot_deferred |= not slot_ok
                    p.reverse_deferred |= not reverse_ok
                    p.cash_deferred |= not funded
                    p.last_trigger = 'unfunded_limit_repriced_at_current_open'
                    force_current_open = True
            price = self._deferred_price(self.codes[p.order.index], t, minute, [] if force_current_open else p.schedule)
            price_cache[key] = price
            return price

        def reserved_cash() -> float:
            # Future slices retain their cash, including fees, until filled or released.
            return sum((max(0.0, float(p.reserved_budget) - p.spent_notional) * (1 + fee) for p in plans if p.action == 'entry' and p.delta > 0 and (p.reserved_budget is not None)))

        def legal(p: Plan) -> bool:
            j = p.order.index
            if p.action == 'entry' and p.reversal and (abs(quantities[j]) > EPS) and (np.sign(quantities[j]) != p.side):
                p.reverse_deferred = True
                return False
            if p.action == 'entry' and abs(quantities[j]) <= EPS and (np.count_nonzero(np.abs(quantities) > EPS) >= self.max_hold):
                p.slot_deferred = True
                return False
            return True

        def due_amount(p: Plan) -> float:
            if p.action == 'entry':
                if p.reserved_budget is None:
                    return 0.0
                return max(0.0, p.reserved_budget * p.due_fraction - p.spent_notional)
            return max(0.0, abs(p.delta) * p.due_fraction - abs(p.filled_quantity))

        def add_fill(p: Plan, signed_delta: float, price: float, minute: int,
                     partial: bool = False, requested_delta: float | None = None) -> None:
            nonlocal cash
            if abs(signed_delta) <= EPS:
                return
            j = p.order.index
            notional = signed_delta * price
            cost = abs(notional) * fee
            cash -= notional + cost
            if abs(cash) < 1e-08:
                cash = 0.0
            quantities[j] += signed_delta
            if abs(quantities[j]) <= EPS:
                quantities[j] = 0.0
            p.filled_quantity += signed_delta
            p.spent_notional += abs(notional)
            p.first_fill = minute if p.first_fill is None else p.first_fill
            p.last_fill = minute
            forced = bool(p.forced_fraction > 0)
            fills.append({
                'index': j,
                'quantity': float(signed_delta),
                'requested_quantity': float(signed_delta if requested_delta is None else requested_delta),
                'price': price,
                'signed_notional': notional,
                'cost': cost,
                'cash_constrained_partial': partial,
                'leg': 'close_before_reverse' if p.reversal and p.action == 'exit' else 'open_after_reverse' if p.reversal else 'rebalance',
                'reason': p.order.reason,
                'signal_date': p.order.signal_date,
                'signal_nav': p.order.signal_nav,
                'time': minute,
                'forced': forced,
                'trigger': p.last_trigger,
                'action': p.action,
                'side': p.side,
                'planned_notional': p.planned_notional,
                'reserved_budget': p.reserved_budget,
                'deferred_minutes': minute - (p.first_due or minute),
            })
            if cash < -1e-07 or np.count_nonzero(np.abs(quantities) > EPS) > self.max_hold:
                raise AssertionError('Cash borrowing or holding-count violation')
        for minute in times:
            for p in plans:
                while p.schedule_index < len(p.schedule) and int(p.schedule[p.schedule_index]['time']) <= minute:
                    item = p.schedule[p.schedule_index]
                    p.due_fraction = min(1.0, p.due_fraction + float(item['fraction']))
                    p.forced_fraction += float(item['fraction']) if item.get('forced', False) else 0.0
                    p.first_due = int(item['time']) if p.first_due is None else p.first_due
                    p.last_trigger = str(item.get('trigger', ''))
                    p.schedule_index += 1
            ready = [p for p in plans if p.action == 'exit' and p.due_fraction > EPS]
            # Credit executed sales first; exits remain fixed quantity risk reductions.
            for p in sorted([p for p in ready if p.action == 'exit' and p.delta < 0], key=lambda p: self.codes[p.order.index]):
                amount = due_amount(p)
                if amount > EPS:
                    price = plan_price(p, minute)
                    add_fill(p, -amount, price, minute)
            for p in sorted([p for p in ready if p.action == 'exit' and p.delta > 0], key=lambda p: self.codes[p.order.index]):
                amount = due_amount(p)
                if amount <= EPS:
                    continue
                price = plan_price(p, minute)
                affordable = max(0.0, cash - reserved_cash()) / (price * (1 + fee))
                filled = min(amount, affordable)
                p.cash_deferred |= filled < amount - EPS
                add_fill(p, filled, price, minute, filled < amount - EPS, requested_delta=amount)
            for p in sorted([p for p in plans if p.action == 'entry' and p.delta < 0 and (p.due_fraction > EPS)], key=lambda p: self.codes[p.order.index]):
                # A short addition is legal only after its former direction and slot clear.
                if not legal(p):
                    continue
                if p.reserved_budget is None:
                    p.reserved_budget = p.planned_notional
                    p.reserved_time = minute
                amount = due_amount(p)
                if amount <= EPS:
                    continue
                price = plan_price(p, minute)
                q = amount / price
                add_fill(p, -q, price, minute)
            for p in sorted([p for p in plans if p.action == 'exit' and p.delta > 0 and (p.due_fraction > EPS)], key=lambda p: self.codes[p.order.index]):
                amount = due_amount(p)
                if amount <= EPS:
                    continue
                price = plan_price(p, minute)
                filled = min(amount, max(0.0, cash - reserved_cash()) / (price * (1 + fee)))
                add_fill(p, filled, price, minute, filled < amount - EPS, requested_delta=amount)
            buys = [p for p in plans if p.action == 'entry' and p.delta > 0 and (p.due_fraction > EPS) and legal(p)]
            buys.sort(key=lambda p: self.codes[p.order.index])
            # Allocate vacant slots before reserving cash for simultaneous new names.
            free_slots = self.max_hold - int(np.count_nonzero(np.abs(quantities) > EPS))
            legal_buys = []
            for p in buys:
                if abs(quantities[p.order.index]) <= EPS:
                    if free_slots <= 0:
                        p.slot_deferred = True
                        continue
                    free_slots -= 1
                legal_buys.append(p)
            buys = legal_buys
            unreserved = [p for p in buys if p.reserved_budget is None]
            # Reserve the whole entry budget once; each slice buys its fraction at its own price.
            available = max(0.0, cash - reserved_cash()) / (1 + fee)
            needed = sum((p.planned_notional for p in unreserved))
            if unreserved and available > EPS:
                scale = min(1.0, available / needed) if needed > 0 else 1.0
                for p in unreserved:
                    p.reserved_budget = p.planned_notional * scale
                    p.reserved_time = minute
                    p.capped = scale < 1 - 1e-10
            else:
                for p in unreserved:
                    p.cash_deferred = True
            for p in buys:
                amount = due_amount(p)
                if amount <= EPS:
                    continue
                price = plan_price(p, minute)
                add_fill(p, amount / price, price, minute)
            snapshots.append((minute, cash, max(0.0, cash - reserved_cash()), quantities.copy()))
        completions = []
        for order in orders:
            j = order.index
            legs = [p for p in plans if p.order is order]
            additions = [p for p in legs if p.action == 'entry']
            legs_complete = [p.reserved_budget is not None and p.reserved_budget > EPS and (abs(p.spent_notional - p.reserved_budget) <= max(1e-07, p.reserved_budget * 1e-09)) if p.action == 'entry' else abs(p.filled_quantity - p.delta) <= max(EPS, abs(p.delta) * 1e-09) for p in legs]
            complete = all(legs_complete)
            late_entry = False
            if additions:
                for p in additions:
                    if p.last_fill is not None and p.last_fill >= 780:
                        daybars = self.bars(self.codes[j], self.dates[t], t)
                        deadline = int(daybars.bar_start.iloc[-1])
                        late_entry |= p.last_fill > deadline
            if late_entry:
                complete = False
            first_due = min((p.first_due for p in legs if p.first_due is not None))
            first_fills = [p.first_fill for p in legs if p.first_fill is not None]
            last_fills = [p.last_fill for p in legs if p.last_fill is not None]
            completions.append({
                'date': self.dates[t],
                'signal_date': order.signal_date,
                'code': self.codes[j],
                'addition': bool(additions),
                'complete': complete,
                'planned_target_quantity': order.target_quantity,
                'actual_quantity': quantities[j],
                'quantity_shortfall': np.nan,
                'planned_notional': sum((p.planned_notional for p in legs)),
                'planned_addition_notional': sum((p.planned_notional for p in additions)),
                'funded_budget': sum((float(p.reserved_budget or 0.0) for p in additions)),
                'spent_notional': sum((p.spent_notional for p in additions)),
                'cash_capped': any((p.capped for p in legs)),
                'cash_deferred': any((p.cash_deferred for p in legs)),
                'slot_deferred': any((p.slot_deferred for p in legs)),
                'reverse_deferred': any((p.reverse_deferred for p in legs)),
                'discarded_unfunded_limit': any((p.discarded_unfunded_limit for p in legs)),
                'late_entry': late_entry,
                'first_due': first_due,
                'first_fill': min(first_fills) if first_fills else np.nan,
                'last_fill': max(last_fills) if last_fills else np.nan,
                'max_delay_minutes': max((p.last_fill - p.first_due for p in legs if p.last_fill is not None), default=0),
            })
        return (float(cash), quantities, fills, completions)


def _summary(curve, settings):
    returns = curve.port_ret.fillna(0.)
    end = float(curve.equity.iloc[-1])
    total = end / settings.initial_cash - 1.
    cagr = (1. + total) ** (252. / len(curve)) - 1.
    dd = float((curve.equity / curve.equity.cummax() - 1.).min())
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std else float('nan')
    return "\n".join([
        "引擎 ndxbots.backtest / 第一組 T+1 預算成交帳本",
        f"區間 {curve.date.iloc[0].date()} ~ {curve.date.iloc[-1].date()}  交易日 {len(curve)}",
        f"期初 {settings.initial_cash:,.2f}  期末 {end:,.2f}  累計收益 {total*100:.2f}%",
        f"年化(252交易日) {cagr*100:.4f}%  Sharpe {sharpe:.3f}  最大回撤 {dd*100:.2f}%",
        f"成本單邊 {settings.cost_bps:.1f}bp  持倉上限 {settings.max_hold}",
        "T 日收盤發出信號；T+1 10:00 入場，方向不利跳空>1%先用50%預算；餘款限價=開盤-方向*0.2*T日ATR20。",
        "剩餘預算15:55完成，半日市12:55；普通減倉09:45、方向/情緒風險減倉09:30。",
        "股數隨實際成交价變化；預算受可用現金及預留成本限制，沒有借款。",
        "5分鐘K線模擬，QFQ/碎股研究單位；未模擬券商手數、排隊、借券、保證金、利息或稅。",
    ])


def run_intraday_backtest(settings, panel, factors, scored, *, minute_store=None):
    simulator = IntradaySimulator(settings, panel, factors, scored, minute_store=minute_store)
    result = simulator.simulate(cost_bps=settings.cost_bps)
    # Failing the daily completion contract must not look like a valid run.
    if result['audit']['incomplete_order_rows']:
        failures = result['completion'].loc[~result['completion'].complete]
        first = failures.iloc[0]
        raise IncompleteExecutionError(
            f"{first['code']} {first['date'].date()}: T+1 execution incomplete "
            f"(funded={first['funded_budget']:.2f}, spent={first['spent_notional']:.2f}); "
            "cash/holding limits prohibit completion")
    holdings = result['holdings']
    metadata_cols = [column for column in scored if column not in holdings.columns and
                     column not in {'in_hold','in_hold_long','in_hold_short'}]
    result['holdings'] = holdings.merge(scored[['date','code',*metadata_cols]],
                                       on=['date','code'], how='left')
    sliced = scored[scored.date.ge(pd.Timestamp(settings.bt_start))]
    if settings.bt_end:
        sliced = sliced[sliced.date.le(pd.Timestamp(settings.bt_end))]
    result['scored'] = sliced
    result['stats'] = _summary(result['curve'], settings)
    return result
