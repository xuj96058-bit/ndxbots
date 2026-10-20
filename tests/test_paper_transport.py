"""Offline subprocess/serialization tests; no broker connection is opened."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.order import OrderError, OrderRequest
from ndxbots.paper import transport
from ndxbots.paper.transport import (
    IsolatedPaperBroker, RESULT_PREFIX, deserialize_settings, paper_accounts,
    serialize_settings,
)


def snapshot():
    return {"env": "SIMULATE", "acc_id": 123, "cash": 1000., "nav": 1000.,
            "positions": {}, "sellable": {}, "orders": []}


def request(settings, operation="snapshot", args=None, kwargs=None):
    return {"version": 1, "request_id": "a" * 32, "env": "SIMULATE",
            "acc_id": 123, "security_firm": "FUTUSECURITIES",
            "settings": serialize_settings(settings), "operation": operation,
            "args": [] if args is None else args, "kwargs": {} if kwargs is None else kwargs}


def response(request_doc, result, *, ok=True, error=None):
    envelope = {key: request_doc[key] for key in
                ("version", "request_id", "env", "acc_id", "operation")}
    envelope.update(ok=ok)
    if ok:
        envelope["result"] = result
    else:
        envelope["error"] = error
    return RESULT_PREFIX + json.dumps(envelope, ensure_ascii=False)


class PaperTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = replace(load_settings(), data_root=Path("測試資料"),
                               intraday_root=Path("測試資料") / "分鐘",
                               plate_keywords=("納斯達克", "NASDAQ"))

    def setUp(self):
        self.broker = IsolatedPaperBroker(self.settings, 123, timeout=.1)

    def run_result(self, value, *, log=True):
        def run(command, **options):
            request_doc = json.loads(options["input"])
            output = response(request_doc, value)
            if log:
                output = "SDK log line\n2024-01-08 connection initialized\n" + output + "\n"
            return subprocess.CompletedProcess(command, 0, output, "unrelated SDK stderr")
        return run

    def test_settings_roundtrip_preserves_paths_tuples_unicode_and_optional_values(self):
        serialized = serialize_settings(self.settings)
        payload = json.loads(json.dumps(serialized, ensure_ascii=False))
        rebuilt = deserialize_settings(payload)
        self.assertEqual(rebuilt, self.settings)
        self.assertIsInstance(rebuilt.data_root, Path)
        self.assertIsInstance(rebuilt.plate_keywords, tuple)
        nullable = replace(self.settings, intraday_root=None, kline_end=None)
        self.assertEqual(deserialize_settings(serialize_settings(nullable)), nullable)

    def test_settings_reject_unknown_fields_wrong_types_and_nonfinite_values(self):
        for key, value in (("unknown", 1), ("futu_port", True), ("data_root", 123),
                           ("plate_keywords", "NASDAQ"), ("cost_bps", float("nan"))):
            with self.subTest(key=key):
                doc = serialize_settings(self.settings)
                doc[key] = value
                with self.assertRaises(ValueError):
                    deserialize_settings(doc)
        with self.assertRaises(OrderError):
            IsolatedPaperBroker({"futu_host": "localhost"}, 123)

    def test_constructor_never_launches_worker_and_rejects_ambiguous_account(self):
        with patch.object(transport.subprocess, "run") as run:
            IsolatedPaperBroker(self.settings, 123)
            for account in (True, "123", 123.0, 0):
                with self.subTest(account=account), self.assertRaises(OrderError):
                    IsolatedPaperBroker(self.settings, account)
            run.assert_not_called()

    def test_snapshot_ignores_sdk_logs_and_preserves_exact_identity(self):
        with patch.object(transport.subprocess, "run", side_effect=self.run_result(snapshot())) as run:
            self.assertEqual(self.broker.snapshot(), snapshot())
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command, [sys.executable, "-X", "utf8", "-m",
                                   "ndxbots.paper.transport", "--worker"])
        options = run.call_args.kwargs
        self.assertEqual(options["timeout"], .1)
        self.assertEqual(options["encoding"], "utf-8")
        self.assertTrue(options["capture_output"])
        self.assertFalse(options["check"])
        if os.name == "nt":
            self.assertEqual(options["creationflags"], subprocess.CREATE_NO_WINDOW)
        request_doc = json.loads(options["input"])
        self.assertEqual((request_doc["env"], request_doc["acc_id"]), ("SIMULATE", 123))
        self.assertEqual(request_doc["settings"]["data_root"], "測試資料")

    def test_every_public_operation_serializes_only_its_whitelisted_arguments(self):
        quote = {"US.A": {"price": 100., "open": 100., "prev_close": 100.,
                         "updated_at": "2024-01-08T10:00:00-05:00",
                         "lot_size": 1, "suspended": False}}
        cases = [
            ("orders", (), {}, [], [], {}),
            ("quote", (["US.A"],), {}, quote, [["US.A"]], {}),
            ("calendar", ("2024-01-05", "2024-01-08"), {},
             [{"date": "2024-01-08", "close_minute": 960}],
             ["2024-01-05", "2024-01-08"], {}),
            ("raw_closes", (["US.A"], "2024-01-05"), {}, {"US.A": 100.},
             [["US.A"], "2024-01-05"], {}),
            ("max_quantity", ("US.A", "BUY", 100., "MARKET"), {"action": "exit"},
             10, ["US.A", "BUY", 100., "MARKET"], {"action": "exit"}),
            ("cancel", ("90071992547410001",), {}, None, ["90071992547410001"], {}),
        ]
        for name, args, kwargs, result, expected_args, expected_kwargs in cases:
            with self.subTest(operation=name):
                with patch.object(transport.subprocess, "run", side_effect=self.run_result(result)) as run:
                    self.assertEqual(getattr(self.broker, name)(*args, **kwargs), result)
                doc = json.loads(run.call_args.kwargs["input"])
                self.assertEqual((doc["operation"], doc["args"], doc["kwargs"]),
                                 (name, expected_args, expected_kwargs))

    def test_place_roundtrips_orderrequest_and_large_numeric_id_without_float(self):
        order = OrderRequest("US.AAPL", "BUY", 1, "LIMIT", .0001, "模擬")
        with patch.object(transport.subprocess, "run", side_effect=self.run_result("90071992547410001")) as run:
            self.assertEqual(self.broker.place(order), "90071992547410001")
        doc = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(OrderRequest(**doc["args"][0]), order)
        self.assertNotIn("confirm_real", doc["kwargs"])
        self.assertNotIn("unlock_password", doc["kwargs"])
        with self.assertRaises(OrderError):
            self.broker.place({"code": "US.AAPL"})

    def test_timeout_never_retries_and_marks_mutations_uncertain(self):
        for method, args in (("snapshot", ()), ("place", (OrderRequest("US.A", "BUY", 1, "MARKET"),)),
                             ("cancel", ("123",))):
            with self.subTest(method=method):
                with patch.object(transport.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired("worker", .1)) as run:
                    with self.assertRaisesRegex(OrderError, "timed out") as error:
                        getattr(self.broker, method)(*args)
                self.assertEqual(run.call_count, 1)
                if method != "snapshot":
                    self.assertIn("outcome unknown", str(error.exception))
                    self.assertIn("do not retry", str(error.exception))

    def test_actual_sleeping_child_is_terminated_and_reaped_within_timeout(self):
        real_run, real_popen = subprocess.run, subprocess.Popen
        children = []
        def create_child(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        def sleeping_worker(command, **options):
            # Exercise stdlib timeout handling without importing/contacting Futu.
            return real_run([sys.executable, "-X", "utf8", "-c", "import time; time.sleep(30)"], **options)
        started = time.monotonic()
        with patch.object(transport.subprocess, "run", side_effect=sleeping_worker), \
                patch.object(transport.subprocess, "Popen", side_effect=create_child):
            with self.assertRaisesRegex(OrderError, "worker terminated"):
                self.broker.snapshot()
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)

    def test_invalid_timeout_does_not_launch_child(self):
        for timeout in (0, -1, True, float("nan"), float("inf"), 301):
            with self.subTest(timeout=timeout):
                broker = IsolatedPaperBroker(self.settings, 123, timeout=timeout)
                with patch.object(transport.subprocess, "run") as run, self.assertRaises(OrderError):
                    broker.snapshot()
                run.assert_not_called()

    def test_changed_environment_or_account_never_launches_child(self):
        for name, value in (("env", "REAL"), ("acc_id", 456), ("acc_id", 123.0)):
            with self.subTest(name=name):
                broker = IsolatedPaperBroker(self.settings, 123)
                setattr(broker, name, value)
                with patch.object(transport.subprocess, "run") as run, self.assertRaises(OrderError):
                    broker.snapshot()
                run.assert_not_called()

    def test_missing_duplicate_or_malformed_envelopes_are_rejected(self):
        for output in ("SDK logs only", RESULT_PREFIX + "not-json",
                       RESULT_PREFIX + "{}\n" + RESULT_PREFIX + "{}",
                       RESULT_PREFIX + '{"version":1,"version":1}',
                       RESULT_PREFIX + '{"result":NaN}'):
            with self.subTest(output=output):
                with patch.object(transport.subprocess, "run",
                                  return_value=subprocess.CompletedProcess("worker", 0, output, "")):
                    with self.assertRaisesRegex(OrderError, "invalid worker response"):
                        self.broker.snapshot()

    def test_envelope_identity_and_types_must_match_request(self):
        for name, value in (("env", "REAL"), ("acc_id", 456), ("acc_id", 123.0),
                            ("request_id", "other"), ("operation", "cancel"),
                            ("ok", "true"), ("version", True)):
            with self.subTest(name=name):
                def run(command, **options):
                    doc = json.loads(options["input"])
                    envelope = json.loads(response(doc, snapshot())[len(RESULT_PREFIX):])
                    envelope[name] = value
                    return subprocess.CompletedProcess(command, 0, RESULT_PREFIX + json.dumps(envelope), "")
                with patch.object(transport.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(OrderError, "invalid worker response"):
                        self.broker.snapshot()

    def test_invalid_operation_results_are_rejected(self):
        cases = [("snapshot", (), {**snapshot(), "env": "REAL"}),
                 ("snapshot", (), {**snapshot(), "cash": -1}),
                 ("snapshot", (), {**snapshot(), "positions": {"US.A": 1.5}}),
                 ("orders", (), {"orders": []}),
                 ("quote", (["US.A"],), {}),
                 ("max_quantity", ("US.A", "BUY", 100., "MARKET"), True),
                 ("cancel", ("123",), {"ok": True}),
                 ("place", (OrderRequest("US.A", "BUY", 1, "MARKET"),), 123)]
        for method, args, result in cases:
            with self.subTest(method=method, result=result):
                with patch.object(transport.subprocess, "run", side_effect=self.run_result(result)):
                    with self.assertRaisesRegex(OrderError, "invalid worker response"):
                        getattr(self.broker, method)(*args)

    def test_worker_error_is_reported_without_replaying_sdk_logs(self):
        def run(command, **options):
            doc = json.loads(options["input"])
            return subprocess.CompletedProcess(command, 1,
                                               response(doc, None, ok=False, error="OrderError: disconnected"),
                                               "sensitive SDK stderr")
        with patch.object(transport.subprocess, "run", side_effect=run) as run:
            with self.assertRaisesRegex(OrderError, "disconnected") as error:
                self.broker.cancel("123")
        self.assertEqual(run.call_count, 1)
        self.assertIn("outcome unknown", str(error.exception))
        self.assertNotIn("sensitive", str(error.exception))

    def test_worker_refuses_nonwhitelisted_ops_real_and_injected_options_before_broker(self):
        bad = []
        for name, value in (("operation", "__dict__"), ("env", "REAL"), ("acc_id", True),
                            ("kwargs", {"env": "REAL"}), ("args", ["unexpected"])):
            doc = request(self.settings)
            doc[name] = value
            bad.append(doc)
        for doc in bad:
            with self.subTest(doc=doc["operation"]):
                with patch("ndxbots.paper.broker.FutuPaperBroker") as broker:
                    with self.assertRaises(ValueError):
                        transport._worker_dispatch(doc)
                broker.assert_not_called()

    def test_worker_passes_simulate_account_and_orderrequest_once(self):
        doc = request(self.settings, "place",
                      [{"code": "US.A", "side": "BUY", "qty": 1, "order_type": "MARKET",
                        "price": None, "remark": "intent-1"}])
        with patch("ndxbots.paper.broker.FutuPaperBroker") as factory:
            factory.return_value.place.return_value = "123"
            self.assertEqual(transport._worker_dispatch(doc), "123")
        factory.assert_called_once_with(self.settings, 123, security_firm="FUTUSECURITIES")
        self.assertIsInstance(factory.return_value.place.call_args.args[0], OrderRequest)
        factory.return_value.place.assert_called_once()

    def test_bounded_account_listing_and_worker_filter_validation(self):
        account = {"acc_id": 123, "trd_env": "SIMULATE", "trdmarket_auth": ["US"],
                   "acc_status": "ACTIVE", "sim_acc_type": "STOCK_AND_OPTION"}
        with patch.object(transport.subprocess, "run", side_effect=self.run_result([account])) as run:
            self.assertEqual(paper_accounts(self.settings, timeout=.1), [account])
        doc = json.loads(run.call_args.kwargs["input"])
        self.assertEqual((doc["operation"], doc["acc_id"]), ("accounts", None))
        with patch.object(transport, "FutuOrderClient") as client:
            client.return_value.accounts.return_value = pd.DataFrame([account])
            self.assertEqual(transport._worker_dispatch(doc), [account])
        client.assert_called_once_with(self.settings, env="SIMULATE", security_firm="FUTUSECURITIES")
        with patch.object(transport.subprocess, "run",
                          side_effect=self.run_result([{**account, "trd_env": "REAL"}])):
            with self.assertRaises(OrderError):
                paper_accounts(self.settings)

    def test_actual_worker_rejects_forbidden_method_without_importing_sdk(self):
        doc = request(self.settings, "__getattribute__")
        options = dict(input=json.dumps(doc, ensure_ascii=False), text=True,
                       encoding="utf-8", capture_output=True, timeout=10, check=False)
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run([sys.executable, "-X", "utf8", "-m",
                                 "ndxbots.paper.transport", "--worker"], **options)
        self.assertEqual(result.returncode, 1, result.stderr)
        rows = [line[len(RESULT_PREFIX):] for line in result.stdout.splitlines()
                if line.startswith(RESULT_PREFIX)]
        self.assertEqual(len(rows), 1)
        envelope = json.loads(rows[0])
        self.assertFalse(envelope["ok"])
        self.assertIn("whitelist", envelope["error"])
        self.assertEqual(envelope["env"], "SIMULATE")


if __name__ == "__main__":
    unittest.main()
