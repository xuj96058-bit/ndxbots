"""Explicit, single-order US stock trading through Futu OpenD.

SIMULATE is the default environment. Mutations preview unless submit=True;
REAL mutations additionally require confirm_real=True. Research targets are
deliberately not converted to broker orders by this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import getpass
import importlib
import json
import math
import os
import re
from typing import Iterator

from ndxbots.config import Settings, load_settings


class OrderError(RuntimeError):
    """Invalid request, broker rejection, or an uncertain submission outcome."""


def _positive_integer(value: object, name: str) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise OrderError(f"{name} must be a positive integer") from None
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise OrderError(f"{name} must be a positive integer")
    return int(number)


@dataclass(frozen=True)
class OrderRequest:
    code: str
    side: str
    qty: int
    order_type: str = "LIMIT"
    price: float | None = None
    remark: str = "ndxbots"

    def __post_init__(self) -> None:
        code = str(self.code).strip().upper()
        # Futu option codes contain expiry/strike digits; never treat contracts
        # as shares, including on combined STOCK_AND_OPTION paper accounts.
        if not re.fullmatch(r"US\.[A-Z]+(?:[.-][A-Z]+)*", code):
            raise OrderError("code must be a US stock code such as US.AAPL")
        side = str(self.side).upper()
        order_type = str(self.order_type).upper()
        if side not in {"BUY", "SELL"}:
            raise OrderError("side must be BUY or SELL")
        if order_type not in {"LIMIT", "MARKET"}:
            raise OrderError("order_type must be LIMIT or MARKET")
        qty = _positive_integer(self.qty, "qty")
        if qty > 2**53 - 1:
            raise OrderError("qty exceeds the SDK's exact integer range")
        if order_type == "MARKET":
            if self.price is not None:
                raise OrderError("omit price for MARKET orders")
            price = None
        else:
            try:
                if isinstance(self.price, bool):
                    raise ValueError
                decimal_price = Decimal(str(self.price))
                price = float(decimal_price)
            except (InvalidOperation, TypeError, ValueError, OverflowError):
                raise OrderError("LIMIT orders require a finite positive price") from None
            if not math.isfinite(price) or price <= 0:
                raise OrderError("LIMIT orders require a finite positive price")
            # Reject unsupported ticks instead of letting OpenD truncate a limit.
            tick = Decimal("0.0001") if decimal_price < 1 else Decimal("0.01")
            try:
                on_tick = decimal_price == decimal_price.quantize(tick)
            except InvalidOperation:
                on_tick = False
            if not on_tick:
                raise OrderError("US limit price allows 4 decimals below $1, otherwise 2")
        if not isinstance(self.remark, str) or len(self.remark.encode("utf-8")) > 64:
            raise OrderError("remark must contain at most 64 UTF-8 bytes")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "qty", qty)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "price", price)


class FutuOrderClient:
    """Separate trading gateway; importing it never opens an OpenD connection."""

    def __init__(
        self, settings: Settings | None = None, *, env: str = "SIMULATE",
        acc_id: int | None = None, security_firm: str = "FUTUSECURITIES", api=None,
    ) -> None:
        self.env = str(env).upper()
        if self.env not in {"SIMULATE", "REAL"}:
            raise OrderError("env must be SIMULATE or REAL")
        self.acc_id = None if acc_id is None else _positive_integer(acc_id, "acc_id")
        self.security_firm = str(security_firm).upper()
        self.settings = settings or load_settings()
        self._api = api

    def _sdk(self):
        if self._api is None:
            try:
                self._api = importlib.import_module("futu")
            except ImportError as exc:
                raise OrderError("Install dependencies first: pip install -e .") from exc
        return self._api

    @contextmanager
    def _session(self) -> Iterator:
        api = self._sdk()
        firm = getattr(api.SecurityFirm, self.security_firm, None)
        if firm is None or self.security_firm.startswith("_") or callable(firm):
            raise OrderError(f"Unsupported security firm: {self.security_firm}")
        try:
            ctx = api.OpenSecTradeContext(
                filter_trdmarket=api.TrdMarket.US,
                host=self.settings.futu_host, port=self.settings.futu_port,
                security_firm=firm,
            )
        except Exception as exc:
            raise OrderError("Cannot connect to Futu OpenD trading context") from exc
        try:
            yield ctx, api
        finally:
            ctx.close()

    @staticmethod
    def _result(result, api, action: str):
        ret, data = result
        if ret != api.RET_OK:
            raise OrderError(f"{action} failed: {data}")
        return data

    def _accounts(self, ctx, api):
        data = self._result(ctx.get_acc_list(), api, "get_acc_list")
        required = {"acc_id", "trd_env", "trdmarket_auth", "acc_status"}
        if self.env == "SIMULATE":
            required.add("sim_acc_type")
        if not required.issubset(data.columns):
            raise OrderError("Account response is missing environment or market fields")
        matching = data["trd_env"].eq(getattr(api.TrdEnv, self.env))
        matching &= data["trdmarket_auth"].map(
            lambda markets: isinstance(markets, (list, tuple, set)) and api.TrdMarket.US in markets
        )
        matching &= data["acc_status"].eq("ACTIVE")
        if self.env == "SIMULATE":
            matching &= data["sim_acc_type"].isin(["STOCK", "STOCK_AND_OPTION"])
        return data.loc[matching].copy()

    def accounts(self):
        """List active US accounts in the selected environment (read only)."""
        with self._session() as (ctx, api):
            return self._accounts(ctx, api)

    def _require_account(self) -> int:
        if self.acc_id is None:
            raise OrderError("Specify --acc-id from the accounts command")
        return self.acc_id

    def _check_account(self, ctx, api) -> None:
        acc_id = self._require_account()
        accounts = self._accounts(ctx, api)
        matches = accounts["acc_id"].map(lambda value: str(value) == str(acc_id))
        if int(matches.sum()) != 1:
            raise OrderError(f"Account {acc_id} is not an active US {self.env} stock account")

    def _check_write(self, submit: bool, confirm_real: bool) -> None:
        if not isinstance(submit, bool):
            raise OrderError("submit must be a boolean")
        if submit:
            self._require_account()
            if self.env == "REAL" and confirm_real is not True:
                raise OrderError("REAL submission requires --confirm-real")

    def _unlock(self, ctx, api, password: str | None) -> None:
        # GUI OpenD is unlocked in its UI; a password is optional for console OpenD.
        if self.env == "REAL" and password is not None:
            if not password:
                raise OrderError("Trade password must not be empty")
            try:
                ret, _ = ctx.unlock_trade(password=password)
            except Exception:
                raise OrderError("unlock_trade failed; check OpenD unlock state") from None
            if ret != api.RET_OK:
                raise OrderError("unlock_trade failed; check password or unlock in OpenD GUI")

    def _preview(self, action: str, **fields) -> dict:
        return {"action": action, "submitted": False, "env": self.env,
                "acc_id": self.acc_id, "security_firm": self.security_firm, **fields}

    def place_order(
        self, request: OrderRequest, *, submit: bool = False,
        confirm_real: bool = False, unlock_password: str | None = None,
    ):
        """Preview or submit exactly once. An acknowledgement is not a fill."""
        if not isinstance(request, OrderRequest):
            raise OrderError("request must be an OrderRequest")
        self._check_write(submit, confirm_real)
        if not submit:
            return self._preview("place", **asdict(request), time_in_force="DAY", session="RTH")
        with self._session() as (ctx, api):
            self._check_account(ctx, api)
            self._unlock(ctx, api, unlock_password)
            kwargs = dict(
                price=request.price if request.price is not None else 0,
                qty=request.qty, code=request.code,
                trd_side=getattr(api.TrdSide, request.side),
                order_type=api.OrderType.NORMAL if request.order_type == "LIMIT" else api.OrderType.MARKET,
                trd_env=getattr(api.TrdEnv, self.env), acc_id=self.acc_id,
                adjust_limit=0, remark=request.remark,
                time_in_force=api.TimeInForce.DAY, fill_outside_rth=False,
                session=api.Session.RTH,
            )
            try:
                result = ctx.place_order(**kwargs)
            except Exception as exc:
                raise OrderError(
                    "Order submission outcome unknown. Query orders before any retry; "
                    "this command did not retry."
                ) from exc
            return self._result(result, api, "place_order (query orders before retrying)")

    def orders(self, order_id: str | None = None):
        """Query open orders and recent completed orders; use the UI for history."""
        self._require_account()
        if order_id is not None and not str(order_id).strip():
            raise OrderError("order_id must not be empty")
        with self._session() as (ctx, api):
            self._check_account(ctx, api)
            return self._result(ctx.order_list_query(
                order_id="" if order_id is None else str(order_id).strip(),
                trd_env=getattr(api.TrdEnv, self.env), acc_id=self.acc_id,
                order_market=api.TrdMarket.US, refresh_cache=True,
            ), api, "order_list_query")

    def cancel_order(
        self, order_id: str, *, submit: bool = False,
        confirm_real: bool = False, unlock_password: str | None = None,
    ):
        order_id = str(order_id).strip()
        if not re.fullmatch(r"[0-9]+", order_id) or int(order_id) <= 0:
            raise OrderError("order_id must be a positive numeric broker order ID")
        self._check_write(submit, confirm_real)
        if not submit:
            return self._preview("cancel", order_id=order_id)
        with self._session() as (ctx, api):
            self._check_account(ctx, api)
            self._unlock(ctx, api, unlock_password)
            try:
                result = ctx.modify_order(
                    modify_order_op=api.ModifyOrderOp.CANCEL, order_id=order_id,
                    qty=0, price=0, trd_env=getattr(api.TrdEnv, self.env), acc_id=self.acc_id,
                )
            except Exception as exc:
                raise OrderError("Cancel outcome unknown. Query orders before any retry.") from exc
            return self._result(result, api, "cancel_order (query orders before retrying)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env", choices=["SIMULATE", "REAL"], default="SIMULATE")
    common.add_argument("--acc-id", type=int, help="Explicit broker account ID")
    common.add_argument("--security-firm", default="FUTUSECURITIES")
    commands.add_parser("accounts", parents=[common], help="List active US accounts")
    orders = commands.add_parser("orders", parents=[common], help="Query open and recent orders")
    orders.add_argument("--order-id")
    write = argparse.ArgumentParser(add_help=False, parents=[common])
    write.add_argument("--submit", action="store_true", help="Send to OpenD; default is offline preview")
    write.add_argument("--confirm-real", action="store_true", help="Required for REAL mutations")
    write.add_argument("--unlock", action="store_true", help="Prompt for console OpenD trade password")
    place = commands.add_parser("place", parents=[write], help="Preview or place one order")
    place.add_argument("--code", required=True)
    place.add_argument("--side", choices=["BUY", "SELL"], required=True)
    place.add_argument("--qty", required=True, help="Positive whole number of shares")
    place.add_argument("--order-type", choices=["LIMIT", "MARKET"], default="LIMIT")
    place.add_argument("--price", help="Required for LIMIT; omit for MARKET")
    place.add_argument("--remark", default="ndxbots")
    cancel = commands.add_parser("cancel", parents=[write], help="Preview or cancel one order")
    cancel.add_argument("--order-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        client = FutuOrderClient(env=args.env, acc_id=args.acc_id, security_firm=args.security_firm)
        request = None
        if args.command == "place":
            request = OrderRequest(args.code, args.side, args.qty, args.order_type, args.price, args.remark)
        if args.command in {"place", "cancel"}:
            client._check_write(args.submit, args.confirm_real)
            password = None
            if args.submit and args.env == "REAL":
                password = os.getenv("FUTU_TRADE_PASSWORD") or None
                if args.unlock and password is None:
                    password = getpass.getpass("Futu trade password: ")
            options = dict(submit=args.submit, confirm_real=args.confirm_real, unlock_password=password)
            result = (client.place_order(request, **options) if request is not None
                      else client.cancel_order(args.order_id, **options))
        elif args.command == "accounts":
            result = client.accounts()
        else:
            result = client.orders(args.order_id)
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result.to_json(orient="records", force_ascii=False, indent=2))
        return 0
    except (OrderError, OSError, ValueError, EOFError) as exc:
        parser.exit(1, f"order: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
