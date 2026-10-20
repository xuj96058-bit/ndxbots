"""Futu's US stock paper account, with explicit routing and validated data.

The adapter has no REAL environment option, never unlocks trading, and never
retries a mutation. Its caller must durably record an intent before ``place``.
All methods return plain dictionaries so recorded responses can be replayed.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import re

import numpy as np
import pandas as pd

from ndxbots.data.intraday import normalize_calendar
from ndxbots.order import FutuOrderClient, OrderError, OrderRequest


ET = "America/New_York"
_STOCK = re.compile(r"US\.[A-Z]+(?:[.-][A-Z]+)*")


def _code(value) -> str:
    if not isinstance(value, str) or not _STOCK.fullmatch(value):
        raise OrderError(f"Paper execution only supports US stock symbols: {value!r}")
    return value


def _number(value, name: str, *, positive=False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise OrderError(f"{name} must be finite") from None
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(number):
        raise OrderError(f"{name} must be finite")
    if positive and number <= 0:
        raise OrderError(f"{name} must be positive")
    return number


def _whole(value, name: str, *, nonnegative=True) -> int:
    number = _number(value, name)
    if number != math.trunc(number) or abs(number) > 2**53 - 1:
        raise OrderError(f"{name} must be an exact whole-share quantity")
    if nonnegative and number < 0:
        raise OrderError(f"{name} must be nonnegative")
    return int(number)


def _frame(value, columns: set[str], action: str, *, empty=False) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise OrderError(f"{action} did not return a DataFrame")
    if empty and value.empty:
        return value
    missing = columns - set(value.columns)
    if missing:
        raise OrderError(f"{action} is missing fields: {sorted(missing)}")
    return value


def _identifier(value, name="order_id") -> str:
    # Broker IDs can exceed 2**53; never pass through float.
    text = str(value)
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        raise OrderError(f"{name} must be a positive numeric identifier")
    return text


class FutuPaperBroker:
    """Only the explicitly selected active US SIMULATE stock account."""

    env = "SIMULATE"

    def __init__(self, settings, acc_id, security_firm="FUTUSECURITIES", api=None):
        self._client = FutuOrderClient(
            settings, env="SIMULATE", acc_id=acc_id,
            security_firm=security_firm, api=api,
        )
        self.acc_id = self._client._require_account()
        self.settings = settings
        self.security_firm = self._client.security_firm

    def _guard(self):
        if (self.env != "SIMULATE" or self._client.env != "SIMULATE"
                or self._client.acc_id != self.acc_id):
            raise OrderError("Paper broker requires its original SIMULATE account")

    @contextmanager
    def _trade(self):
        self._guard()
        with self._client._session() as (ctx, api):
            accounts = self._client._accounts(ctx, api)
            matches = accounts.loc[accounts.acc_id.map(str).eq(str(self.acc_id))]
            if len(matches) != 1:
                raise OrderError(f"Account {self.acc_id} is not an active US SIMULATE stock account")
            yield ctx, api, matches.iloc[0]

    @contextmanager
    def _quotes(self):
        self._guard()
        api = self._client._sdk()
        try:
            ctx = api.OpenQuoteContext(host=self.settings.futu_host, port=self.settings.futu_port)
        except Exception as exc:
            raise OrderError("Cannot connect to Futu OpenD quote context") from exc
        try:
            yield ctx, api
        finally:
            ctx.close()

    def _identity(self, frame, action):
        """Validate identity when present; the SDK otherwise routes via arguments."""
        if "trd_env" in frame and not frame.trd_env.eq("SIMULATE").all():
            raise OrderError(f"{action} returned a non-SIMULATE environment")
        if "acc_id" in frame and not frame.acc_id.map(str).eq(str(self.acc_id)).all():
            raise OrderError(f"{action} returned a different account")

    def _normalize_orders(self, frame) -> list[dict]:
        frame = _frame(frame, {"order_id", "remark", "code", "trd_side", "qty",
                              "dealt_qty", "dealt_avg_price", "order_status"},
                       "order_list_query", empty=True)
        self._identity(frame, "order_list_query")
        results = []
        ids = set()
        for row in frame.to_dict("records"):
            identifier = _identifier(row["order_id"])
            if identifier in ids:
                raise OrderError("order_list_query returned duplicate order IDs")
            ids.add(identifier)
            side = str(row["trd_side"])
            if side not in {"BUY", "SELL"}:
                raise OrderError("Unsupported broker order side")
            qty = _whole(row["qty"], "order qty")
            dealt = _whole(row["dealt_qty"], "dealt qty")
            average = _number(row["dealt_avg_price"], "dealt average")
            if qty <= 0 or dealt > qty or average < 0 or (dealt > 0 and average <= 0):
                raise OrderError("Invalid broker order fill totals")
            if "currency" in row and str(row["currency"]) != "USD":
                raise OrderError("US order currency is not USD")
            remark, status = row["remark"], row["order_status"]
            if not isinstance(remark, str) or not isinstance(status, str) or not status:
                raise OrderError("Invalid order remark or status")
            results.append({
                "order_id": identifier, "remark": remark, "code": _code(row["code"]),
                "side": side, "qty": qty, "dealt_qty": dealt,
                "dealt_avg_price": average, "status": status,
            })
        return results

    def _orders(self, ctx, api):
        return self._normalize_orders(self._client._result(
            ctx.order_list_query(trd_env=api.TrdEnv.SIMULATE, acc_id=self.acc_id,
                                 order_market=api.TrdMarket.US, refresh_cache=True),
            api, "order_list_query",
        ))

    def orders(self):
        """Open orders plus orders completed/cancelled in the past 24 hours."""
        with self._trade() as (ctx, api, _):
            return self._orders(ctx, api)

    def snapshot(self) -> dict:
        with self._trade() as (ctx, api, account):
            funds = self._client._result(ctx.accinfo_query(
                trd_env=api.TrdEnv.SIMULATE, acc_id=self.acc_id,
                refresh_cache=True, currency=api.Currency.USD,
            ), api, "accinfo_query")
            funds = _frame(funds, {"total_assets", "frozen_cash"}, "accinfo_query")
            self._identity(funds, "accinfo_query")
            if len(funds) != 1:
                raise OrderError("Expected exactly one paper account funds row")
            row = funds.iloc[0]
            currency = str(row.get("currency", "N/A"))
            # Older single-market paper accounts return N/A and ignore currency.
            # The US-only account authorization is the required currency evidence.
            single_us = set(account["trdmarket_auth"]) == {api.TrdMarket.US}
            if currency != "USD" and not (single_us and currency in {"N/A", "NONE", ""}):
                raise OrderError("Cannot establish USD denomination of paper account funds")
            nav = _number(row["total_assets"], "total_assets", positive=True)
            frozen = _number(row["frozen_cash"], "frozen_cash")
            if frozen < 0:
                raise OrderError("frozen_cash must be nonnegative")
            # Prefer the documented per-currency balance. The deprecated cash
            # field may be a placeholder on newer accounts; use it only for
            # older paper accounts that have no per-currency value.
            value = row.get("us_cash")
            explicit_usd = (value is not None and not pd.isna(value)
                            and str(value) not in {"", "N/A"})
            cash = _number(value if explicit_usd else row.get("cash"),
                           "us_cash" if explicit_usd else "cash") - frozen
            # Cash buying power can be absent/N/A for older paper accounts.
            # When provided, it further restricts cash; never use margin power.
            value = row.get("usd_net_cash_power")
            if value is not None and not pd.isna(value) and str(value) not in {"", "N/A"}:
                cash = min(cash, _number(value, "usd_net_cash_power"))
            positions = self._client._result(ctx.position_list_query(
                trd_env=api.TrdEnv.SIMULATE, acc_id=self.acc_id,
                position_market=api.TrdMarket.US, refresh_cache=True,
            ), api, "position_list_query")
            positions = _frame(positions, {"code", "qty", "can_sell_qty", "position_side"},
                               "position_list_query", empty=True)
            self._identity(positions, "position_list_query")
            holdings, sellable = {}, {}
            for position in positions.to_dict("records"):
                code = _code(position["code"])
                if code in holdings:
                    raise OrderError(f"Duplicate broker position: {code}")
                if "currency" in position and str(position["currency"]) != "USD":
                    raise OrderError("US position currency is not USD")
                qty = _whole(position["qty"], f"{code} qty", nonnegative=False)
                side = str(position["position_side"])
                if side not in {"LONG", "SHORT"} or (side == "LONG" and qty < 0):
                    raise OrderError(f"Invalid position direction for {code}")
                available = abs(_whole(position["can_sell_qty"], f"{code} can_sell_qty",
                                       nonnegative=False))
                if available > abs(qty):
                    raise OrderError(f"Available quantity exceeds position for {code}")
                if qty:
                    holdings[code] = abs(qty) * (1 if side == "LONG" else -1)
                    sellable[code] = available
            orders = self._orders(ctx, api)
        return {
            "acc_id": self.acc_id, "env": "SIMULATE", "cash": max(0.0, cash),
            "nav": nav, "positions": holdings, "sellable": sellable, "orders": orders,
        }

    def quote(self, codes) -> dict:
        codes = list(dict.fromkeys(_code(code) for code in codes))
        if not codes:
            return {}
        if len(codes) > 400:
            raise OrderError("At most 400 symbols may be quoted at once")
        with self._quotes() as (ctx, api):
            frame = self._client._result(ctx.get_market_snapshot(codes), api, "get_market_snapshot")
        frame = _frame(frame, {"code", "last_price", "open_price", "prev_close_price",
                              "update_time", "lot_size", "suspension"}, "get_market_snapshot")
        if frame.code.duplicated().any() or set(frame.code) != set(codes):
            raise OrderError("Quote response does not match requested symbols")
        result = {}
        for row in frame.to_dict("records"):
            code = _code(row["code"])
            try:
                stamp = pd.Timestamp(row["update_time"])
                if pd.isna(stamp):
                    raise ValueError("missing quote timestamp")
                stamp = stamp.tz_localize(ET) if stamp.tzinfo is None else stamp.tz_convert(ET)
            except (TypeError, ValueError) as exc:
                raise OrderError(f"{code}: invalid quote timestamp") from exc
            suspended = row["suspension"]
            if not isinstance(suspended, bool):
                raise OrderError(f"{code}: invalid suspension flag")
            lot = _whole(row["lot_size"], f"{code} lot_size")
            if lot != 1:
                raise OrderError(f"{code}: only whole US stock shares (lot_size=1) are supported")
            result[code] = {
                "price": _number(row["last_price"], f"{code} last_price", positive=True),
                "open": _number(row["open_price"], f"{code} open_price", positive=True),
                "prev_close": _number(row["prev_close_price"], f"{code} prev_close_price", positive=True),
                "updated_at": stamp.isoformat(), "lot_size": lot, "suspended": suspended,
            }
        return result

    def calendar(self, start, end) -> list[dict]:
        start, end = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
        if start > end:
            raise OrderError("Calendar start must not follow end")
        with self._quotes() as (ctx, api):
            rows = self._client._result(ctx.request_trading_days(
                market=api.TradeDateMarket.US, start=start, end=end,
            ), api, "request_trading_days")
        if len(rows) == 0:
            return []
        frame = normalize_calendar(rows)
        if not frame.date.between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise OrderError("Calendar returned sessions outside requested dates")
        return [{"date": str(row.date.date()), "close_minute": int(row.close_minute)}
                for row in frame.itertuples(index=False)]

    def raw_closes(self, codes, signal_date) -> dict:
        """Exact-date, unadjusted daily closes; missing days are never backfilled."""
        day = str(pd.Timestamp(signal_date).date())
        codes = list(dict.fromkeys(_code(code) for code in codes))
        if not codes:
            return {}
        result = {}
        with self._quotes() as (ctx, api):
            for code in codes:
                ret, frame, page = ctx.request_history_kline(
                    code, start=day, end=day, ktype=api.KLType.K_DAY,
                    autype=api.AuType.NONE, max_count=2, session=api.Session.RTH,
                )
                frame = self._client._result((ret, frame), api, "request_history_kline")
                frame = _frame(frame, {"time_key", "close"}, "request_history_kline")
                if len(frame) != 1 or page is not None:
                    raise OrderError(f"{code}: expected exactly one unadjusted daily candle on {day}")
                row = frame.iloc[0]
                if pd.isna(row["time_key"]) or str(pd.Timestamp(row["time_key"]).date()) != day:
                    raise OrderError(f"{code}: daily close is not from signal date {day}")
                if "code" in frame and str(row["code"]) != code:
                    raise OrderError(f"{code}: daily candle returned a different symbol")
                result[code] = _number(row["close"], f"{code} raw close", positive=True)
        return result

    def max_quantity(self, code, side, price, order_type, *, action="entry") -> int:
        """Broker limit for a separate entry or exit leg, never a reversal."""
        _code(code)
        if side not in {"BUY", "SELL"} or action not in {"entry", "exit"}:
            raise OrderError("Invalid side/action for tradable quantity")
        if order_type not in {"LIMIT", "MARKET"}:
            raise OrderError("Only LIMIT and MARKET orders are supported")
        price = _number(price, "quantity estimate price", positive=True)
        field = {
            ("BUY", "entry"): "max_cash_buy", ("SELL", "entry"): "max_sell_short",
            ("BUY", "exit"): "max_buy_back", ("SELL", "exit"): "max_position_sell",
        }[(side, action)]
        with self._trade() as (ctx, api, _):
            data = self._client._result(ctx.acctradinginfo_query(
                order_type=api.OrderType.NORMAL if order_type == "LIMIT" else api.OrderType.MARKET,
                code=code, price=price, adjust_limit=0,
                trd_env=api.TrdEnv.SIMULATE, acc_id=self.acc_id, session=api.Session.RTH,
            ), api, "acctradinginfo_query")
        data = _frame(data, {field}, "acctradinginfo_query")
        self._identity(data, "acctradinginfo_query")
        if len(data) != 1:
            raise OrderError("Expected exactly one tradable quantity row")
        number = _number(data.iloc[0][field], field)
        if number < 0 or number > 2**53 - 1:
            raise OrderError(f"{field} must be nonnegative and within the exact integer range")
        return math.floor(number)

    def place(self, request: OrderRequest) -> str:
        self._guard()
        result = self._client.place_order(request, submit=True)
        try:
            frame = _frame(result, {"order_id"}, "place_order")
            self._identity(frame, "place_order")
            if len(frame) != 1:
                raise OrderError("Expected exactly one order acknowledgement")
            row = frame.iloc[0]
            for field, expected in (("code", request.code), ("trd_side", request.side),
                                    ("remark", request.remark)):
                if field in frame and row[field] != expected:
                    raise OrderError(f"Order acknowledgement mismatches {field}")
            if "qty" in frame and _whole(row["qty"], "acknowledged qty") != request.qty:
                raise OrderError("Order acknowledgement mismatches quantity")
            return _identifier(row["order_id"])
        except (OrderError, ValueError, TypeError) as exc:
            raise OrderError("Order outcome unknown: invalid acknowledgement; reconcile before retry") from exc

    def cancel(self, order_id) -> None:
        self._guard()
        order_id = _identifier(order_id)
        result = self._client.cancel_order(order_id, submit=True)
        try:
            frame = _frame(result, {"order_id"}, "modify_order")
            self._identity(frame, "modify_order")
            if len(frame) != 1 or _identifier(frame.iloc[0]["order_id"]) != order_id:
                raise OrderError("Cancellation acknowledgement mismatches order ID")
        except (OrderError, ValueError, TypeError) as exc:
            raise OrderError("Cancel outcome unknown: invalid acknowledgement; reconcile before retry") from exc
