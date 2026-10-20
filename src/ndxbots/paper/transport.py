"""Bound Futu SDK connection/query time by isolating each operation in a process.

Futu's synchronous context constructor can keep reconnecting indefinitely.
Killing its child process bounds that constructor as well as SDK calls/cleanup;
threads cannot provide this guarantee. A timed-out mutation remains uncertain
and must be recovered using the engine's already persisted order intention.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import get_args, get_origin, get_type_hints
from uuid import uuid4

from ndxbots.config import Settings
from ndxbots.order import FutuOrderClient, OrderError, OrderRequest
from .validation import _code, _day, _integer, _mapping, _number, _timestamp


RESULT_PREFIX = "NDXBOTS_PAPER_RESULT_V1:"
OPERATIONS = frozenset({"accounts", "snapshot", "orders", "quote", "calendar",
                        "raw_closes", "max_quantity", "place", "cancel"})
_WRITES = frozenset({"place", "cancel"})
_SETTINGS_TYPES = get_type_hints(Settings)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _setting(value, annotation, name):
    if get_origin(annotation) is types.UnionType:
        allowed = get_args(annotation)
        if value is None and type(None) in allowed:
            return None
        return _setting(value, next(t for t in allowed if t is not type(None)), name)
    if annotation is Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a path string")
        return Path(value)
    if get_origin(annotation) is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a JSON array")
        item_type = get_args(annotation)[0]
        return tuple(_setting(item, item_type, name) for item in value)
    if annotation is float:
        return float(_number(value, name))
    if annotation is int:
        return _integer(value, name)
    if annotation in (str, bool) and type(value) is not annotation:
        raise ValueError(f"{name} has an invalid type")
    return value


def deserialize_settings(payload):
    """Rebuild Settings from the caller's values, without loading config again."""
    _mapping(payload, "settings")
    names = {field.name for field in fields(Settings)}
    if set(payload) != names:
        raise ValueError("Settings fields differ from the installed Settings schema")
    return Settings(**{name: _setting(payload[name], _SETTINGS_TYPES[name], name)
                       for name in names})


def serialize_settings(settings):
    if not isinstance(settings, Settings):
        raise OrderError("Isolated paper operations require Settings")
    payload = asdict(settings)
    for name, value in payload.items():
        if isinstance(value, Path):
            payload[name] = str(value)
        elif isinstance(value, tuple):
            payload[name] = list(value)
    try:
        deserialize_settings(payload)
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise OrderError(f"Settings cannot be serialized: {exc}") from exc
    return payload


def _order_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value) or int(value) <= 0:
        raise ValueError("Expected a positive numeric order ID string")


def _validate_orders(rows):
    if not isinstance(rows, list):
        raise ValueError("Expected an order list")
    identifiers = set()
    for row in rows:
        _mapping(row, "order")
        _order_id(row.get("order_id"))
        if row["order_id"] in identifiers:
            raise ValueError("Duplicate broker order ID")
        identifiers.add(row["order_id"])
        _code(row.get("code"))
        if row.get("side") not in ("BUY", "SELL"):
            raise ValueError("Invalid order side")
        qty = _integer(row.get("qty"), "qty", minimum=1)
        filled = _integer(row.get("dealt_qty"), "dealt_qty", minimum=0)
        average = _number(row.get("dealt_avg_price"), "dealt_avg_price", minimum=0)
        if filled > qty or (filled and average <= 0):
            raise ValueError("Invalid cumulative fills")
        if not isinstance(row.get("remark"), str) or not isinstance(row.get("status"), str) or not row["status"]:
            raise ValueError("Invalid order remark/status")


def _validate_result(operation, result, account, args):
    if operation == "accounts":
        if not isinstance(result, list):
            raise ValueError("Expected an account list")
        seen = set()
        for row in result:
            _mapping(row, "account")
            ident = _integer(row.get("acc_id"), "account ID", minimum=1)
            markets = row.get("trdmarket_auth")
            if (ident in seen or row.get("trd_env") != "SIMULATE"
                    or row.get("acc_status") != "ACTIVE"
                    or row.get("sim_acc_type") not in ("STOCK", "STOCK_AND_OPTION")
                    or not isinstance(markets, list) or "US" not in markets):
                raise ValueError("Account result is not an active US SIMULATE stock account")
            seen.add(ident)
    elif operation == "snapshot":
        _mapping(result, "snapshot")
        if result.get("env") != "SIMULATE" or type(result.get("acc_id")) is not int or result["acc_id"] != account:
            raise ValueError("Snapshot identity differs from requested SIMULATE account")
        _number(result.get("cash"), "cash", minimum=0)
        _number(result.get("nav"), "nav", positive=True)
        for name in ("positions", "sellable"):
            for code, qty in _mapping(result.get(name), name).items():
                _code(code)
                _integer(qty, name, minimum=0 if name == "sellable" else None)
        _validate_orders(result.get("orders"))
    elif operation == "orders":
        _validate_orders(result)
    elif operation in ("quote", "raw_closes"):
        _mapping(result, operation)
        if set(result) != set(args[0]):
            raise ValueError("Quote symbols differ from requested symbols")
        for code, value in result.items():
            _code(code)
            if operation == "raw_closes":
                _number(value, "raw close", positive=True)
            else:
                _mapping(value, "quote")
                for name in ("price", "open", "prev_close"):
                    _number(value.get(name), name, positive=True)
                if value.get("lot_size") != 1 or type(value.get("lot_size")) is not int or type(value.get("suspended")) is not bool:
                    raise ValueError("Unsupported quote lot size/suspension")
                if not isinstance(value.get("updated_at"), str):
                    raise ValueError("Quote timestamp is missing")
                _timestamp(value["updated_at"], "quote timestamp")
    elif operation == "calendar":
        if not isinstance(result, list):
            raise ValueError("Expected a session list")
        seen = set()
        for row in result:
            _mapping(row, "session")
            day = _day(row.get("date"), "session date")
            close = _integer(row.get("close_minute"), "close_minute")
            if close not in (780, 960) or day in seen:
                raise ValueError("Invalid session calendar")
            seen.add(day)
    elif operation == "max_quantity":
        _integer(result, "maximum quantity", minimum=0)
    elif operation == "place":
        _order_id(result)
    elif operation == "cancel" and result is not None:
        raise ValueError("Invalid cancellation acknowledgement")


def _validate_request(request):
    _mapping(request, "worker request")
    expected = {"version", "request_id", "env", "acc_id", "security_firm",
                "settings", "operation", "args", "kwargs"}
    if set(request) != expected or type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("Invalid worker request schema")
    if request["env"] != "SIMULATE":
        raise ValueError("Worker accepts SIMULATE only")
    if not isinstance(request["request_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", request["request_id"]):
        raise ValueError("Invalid request identity")
    op, args, kwargs = request["operation"], request["args"], request["kwargs"]
    if not isinstance(op, str) or op not in OPERATIONS:
        raise ValueError("Operation is not on the paper worker whitelist")
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise ValueError("Invalid worker arguments")
    if op == "accounts":
        if request["acc_id"] is not None:
            raise ValueError("Account listing does not select an account")
    else:
        _integer(request["acc_id"], "acc_id", minimum=1)
    if not isinstance(request["security_firm"], str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", request["security_firm"]):
        raise ValueError("Invalid security firm")
    counts = {"accounts": 0, "snapshot": 0, "orders": 0, "quote": 1, "calendar": 2,
              "raw_closes": 2, "max_quantity": 4, "place": 1, "cancel": 1}
    if len(args) != counts[op] or set(kwargs) != ({"action"} if op == "max_quantity" else set()):
        raise ValueError("Unexpected operation arguments")
    if op in ("quote", "raw_closes"):
        if not isinstance(args[0], list):
            raise ValueError("Symbols must be a list")
        for code in args[0]:
            _code(code)
    if op in ("calendar", "raw_closes"):
        dates = args if op == "calendar" else args[1:]
        if not all(isinstance(value, str) for value in dates):
            raise ValueError("Dates must be strings")
    if op == "max_quantity":
        _code(args[0])
        _number(args[2], "estimate price", positive=True)
        if args[1] not in ("BUY", "SELL") or args[3] not in ("MARKET", "LIMIT") or kwargs["action"] not in ("entry", "exit"):
            raise ValueError("Invalid quantity query")
    if op == "place":
        _mapping(args[0], "order request")
        if set(args[0]) != {field.name for field in fields(OrderRequest)}:
            raise ValueError("Invalid OrderRequest fields")
        OrderRequest(**args[0])
    if op == "cancel":
        _order_id(args[0])


def _worker_dispatch(request):
    """Only validated, explicit operations can reach the broker constructor."""
    _validate_request(request)
    settings = deserialize_settings(request["settings"])
    op, args, kwargs = request["operation"], request["args"], request["kwargs"]
    if op == "accounts":
        client = FutuOrderClient(settings, env="SIMULATE", security_firm=request["security_firm"])
        result = []
        for row in client.accounts().to_dict("records"):
            result.append({"acc_id": int(str(row["acc_id"])), "trd_env": str(row["trd_env"]),
                           "trdmarket_auth": list(row["trdmarket_auth"]),
                           "acc_status": str(row["acc_status"]), "sim_acc_type": str(row["sim_acc_type"])})
    else:
        from .broker import FutuPaperBroker
        broker = FutuPaperBroker(settings, request["acc_id"], security_firm=request["security_firm"])
        if op == "place":
            result = broker.place(OrderRequest(**args[0]))
        else:
            result = getattr(broker, op)(*args, **kwargs)
    _validate_result(op, result, request["acc_id"], args)
    return result


def _failure(operation, reason):
    message = f"Paper broker {operation}: {reason}"
    if operation in _WRITES:
        message += "; outcome unknown; do not retry; reconcile the saved intention with broker orders"
    return OrderError(message)


def _invoke(settings, account, security_firm, operation, args, kwargs, timeout):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise OrderError("Broker timeout must be finite, positive, and at most 300 seconds")
    request = {"version": 1, "request_id": uuid4().hex, "env": "SIMULATE",
               "acc_id": account, "security_firm": security_firm,
               "settings": serialize_settings(settings), "operation": operation,
               "args": args, "kwargs": kwargs}
    try:
        _validate_request(request)
        serialized = json.dumps(request, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise OrderError(f"Invalid paper broker request: {exc}") from exc
    options = dict(input=serialized, capture_output=True, text=True, encoding="utf-8",
                   errors="replace", timeout=timeout, check=False)
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        # subprocess.run kills and waits for the direct child on TimeoutExpired.
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ndxbots.paper.transport", "--worker"], **options)
    except subprocess.TimeoutExpired as exc:
        raise _failure(operation, f"timed out after {timeout:g}s; worker terminated") from exc
    except OSError as exc:
        raise _failure(operation, f"worker failed to start ({type(exc).__name__})") from exc
    try:
        lines = [line[len(RESULT_PREFIX):] for line in process.stdout.splitlines()
                 if line.startswith(RESULT_PREFIX)]
        if len(lines) != 1:
            raise ValueError("missing or duplicate result envelope")
        envelope = _loads(lines[0])
        _mapping(envelope, "result envelope")
        for key in ("version", "request_id", "env", "acc_id", "operation"):
            if type(envelope.get(key)) is not type(request[key]) or envelope[key] != request[key]:
                raise ValueError(f"result {key} does not match request")
        if type(envelope.get("ok")) is not bool:
            raise ValueError("result lacks a boolean success flag")
        if not envelope["ok"]:
            error = envelope.get("error")
            if not isinstance(error, str) or not error:
                raise ValueError("result lacks an error description")
            raise _failure(operation, error[:1000])
        if process.returncode != 0:
            raise ValueError("worker exited unsuccessfully after a result")
        if "result" not in envelope:
            raise ValueError("missing operation result")
        _validate_result(operation, envelope["result"], account, args)
        return envelope["result"]
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise _failure(operation, f"invalid worker response ({exc})") from exc


class IsolatedPaperBroker:
    """FutuPaperBroker interface with one bounded SIMULATE subprocess per call."""
    env = "SIMULATE"

    def __init__(self, settings, acc_id, security_firm="FUTUSECURITIES", *, timeout=45.0):
        try:
            _integer(acc_id, "acc_id", minimum=1)
        except ValueError as exc:
            raise OrderError(str(exc)) from exc
        self.settings, self.acc_id = settings, acc_id
        self.security_firm, self.timeout = str(security_firm).upper(), timeout
        self._account_id = acc_id
        serialize_settings(settings)  # Validate locally without launching a worker.

    def _call(self, operation, *args, **kwargs):
        if self.env != "SIMULATE" or self.acc_id != self._account_id or type(self.acc_id) is not int:
            raise OrderError("Isolated broker requires its original SIMULATE account")
        return _invoke(self.settings, self.acc_id, self.security_firm, operation,
                       list(args), kwargs, self.timeout)

    def snapshot(self):
        return self._call("snapshot")

    def orders(self):
        return self._call("orders")

    def quote(self, codes):
        return self._call("quote", list(codes))

    def calendar(self, start, end):
        return self._call("calendar", str(start), str(end))

    def raw_closes(self, codes, signal_date):
        return self._call("raw_closes", list(codes), str(signal_date))

    def max_quantity(self, code, side, price, order_type, *, action="entry"):
        return self._call("max_quantity", code, side, price, order_type, action=action)

    def place(self, request):
        if not isinstance(request, OrderRequest):
            raise OrderError("place requires an OrderRequest")
        return self._call("place", asdict(request))

    def cancel(self, order_id):
        return self._call("cancel", str(order_id))


def paper_accounts(settings, security_firm="FUTUSECURITIES", *, timeout=45.0):
    """Return validated active US SIMULATE stock account records, with a timeout."""
    return _invoke(settings, None, str(security_firm).upper(), "accounts", [], {}, timeout)


def _worker_main():
    request = {}
    try:
        # The parent sends one small JSON request and closes stdin.
        raw = sys.stdin.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Worker request is too large")
        request = _loads(raw)
        result = _worker_dispatch(request)
        envelope = {"ok": True, "result": result}
        exit_code = 0
    except Exception as exc:
        envelope = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
        exit_code = 1
    identity = request if isinstance(request, dict) else {}
    envelope.update({key: identity.get(key) for key in
                     ("version", "request_id", "acc_id", "operation")})
    envelope["env"] = "SIMULATE"
    # A preceding newline keeps this distinguishable from partial SDK log lines.
    print("\n" + RESULT_PREFIX + json.dumps(envelope, ensure_ascii=False, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Private paper broker worker; use python -m ndxbots.paper")
    raise SystemExit(_worker_main())
