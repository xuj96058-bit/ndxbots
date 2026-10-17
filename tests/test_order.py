"""Broker routing and mutation guards; all contexts are fake and offline."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ndxbots.order import FutuOrderClient, OrderError, OrderRequest, main


def account(acc_id=101, env="SIMULATE", markets=None, status="ACTIVE",
            sim_type="STOCK_AND_OPTION"):
    return {"acc_id": acc_id, "trd_env": env,
            "trdmarket_auth": ["US"] if markets is None else markets,
            "acc_status": status, "sim_acc_type": sim_type}


def gateway(*, env="SIMULATE", acc_id=101, accounts=None):
    result = pd.DataFrame([{"order_id": "987654321", "order_status": "SUBMITTING"}])
    rows = [account(), account(202, "REAL", sim_type="N/A")] if accounts is None else accounts
    ctx = SimpleNamespace(
        get_acc_list=Mock(return_value=(0, pd.DataFrame(rows))),
        unlock_trade=Mock(return_value=(0, "unlocked")),
        place_order=Mock(return_value=(0, result)),
        order_list_query=Mock(return_value=(0, result)),
        modify_order=Mock(return_value=(0, result)),
        close=Mock(),
    )
    api = SimpleNamespace(
        RET_OK=0, OpenSecTradeContext=Mock(return_value=ctx),
        SecurityFirm=SimpleNamespace(FUTUSECURITIES="FUTUSECURITIES", FUTUSG="FUTUSG"),
        TrdEnv=SimpleNamespace(SIMULATE="SIMULATE", REAL="REAL"),
        TrdMarket=SimpleNamespace(US="US"),
        TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=SimpleNamespace(NORMAL="NORMAL", MARKET="MARKET"),
        TimeInForce=SimpleNamespace(DAY="DAY"),
        Session=SimpleNamespace(RTH="RTH"),
        ModifyOrderOp=SimpleNamespace(CANCEL="CANCEL"),
    )
    settings = SimpleNamespace(futu_host="127.0.0.2", futu_port=11112)
    client = FutuOrderClient(settings, env=env, acc_id=acc_id, api=api)
    return client, ctx, api


class RequestTests(unittest.TestCase):
    def test_request_normalizes_and_preserves_whole_share_quantity(self):
        request = OrderRequest(" us.aapl ", "buy", "2.0", price="100.25")
        self.assertEqual((request.code, request.side, request.qty, request.price),
                         ("US.AAPL", "BUY", 2, 100.25))

    def test_invalid_quantities_rejected(self):
        for quantity in [True, False, None, "", 0, -1, 0.5, "1.2", float("nan"),
                         float("inf"), "-Infinity", 2**53]:
            with self.subTest(quantity=quantity), self.assertRaises(OrderError):
                OrderRequest("US.AAPL", "BUY", quantity, price=100)

    def test_invalid_limit_prices_rejected(self):
        for price in [None, "", True, False, 0, -1, float("nan"), float("inf"),
                      "-Infinity", "not a price"]:
            with self.subTest(price=price), self.assertRaises(OrderError):
                OrderRequest("US.AAPL", "BUY", 1, price=price)

    def test_market_order_rejects_any_supplied_price(self):
        self.assertIsNone(OrderRequest("US.AAPL", "SELL", 1, "MARKET").price)
        for price in [0, 100, float("nan")]:
            with self.subTest(price=price), self.assertRaises(OrderError):
                OrderRequest("US.AAPL", "SELL", 1, "MARKET", price)

    def test_limit_price_precision_is_validated_without_silent_rounding(self):
        for price in ["0.0001", "0.9999", "1.00", "100.25", "100.2500"]:
            with self.subTest(price=price):
                self.assertEqual(OrderRequest("US.AAPL", "BUY", 1, price=price).price,
                                 float(price))
        for price in ["0.00001", "0.99999", "1.001", "100.251"]:
            with self.subTest(price=price), self.assertRaises(OrderError):
                OrderRequest("US.AAPL", "BUY", 1, price=price)

    def test_invalid_code_direction_type_and_utf8_remark_rejected(self):
        for code in ["AAPL", "HK.00700", "US.", "US.AAPL\nSELL", "US.AAPL261120C200000"]:
            with self.subTest(code=code), self.assertRaises(OrderError):
                OrderRequest(code, "BUY", 1, price=100)
        for side in ["LONG", "SELL_SHORT", "BUY_BACK"]:
            with self.subTest(side=side), self.assertRaises(OrderError):
                OrderRequest("US.AAPL", side, 1, price=100)
        with self.assertRaises(OrderError):
            OrderRequest("US.AAPL", "BUY", 1, "STOP", 100)
        OrderRequest("US.AAPL", "BUY", 1, price=100, remark="股" * 21)
        for remark in ["股" * 22, "x" * 65, None]:
            with self.subTest(remark=remark), self.assertRaises(OrderError):
                OrderRequest("US.AAPL", "BUY", 1, price=100, remark=remark)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.request = OrderRequest("US.AAPL", "BUY", 2, price=100.25)

    def test_default_simulated_preview_needs_no_sdk_account_or_connection(self):
        client = FutuOrderClient(SimpleNamespace(futu_host="unused", futu_port=1))
        with patch.object(client, "_sdk", side_effect=AssertionError("preview imported SDK")):
            preview = client.place_order(self.request, unlock_password="unused")
            cancel = client.cancel_order("987654321")
        self.assertEqual(preview["env"], "SIMULATE")
        self.assertFalse(preview["submitted"])
        self.assertIsNone(preview["acc_id"])
        self.assertEqual((preview["qty"], preview["session"], preview["time_in_force"]),
                         (2, "RTH", "DAY"))
        self.assertEqual((cancel["action"], cancel["order_id"]), ("cancel", "987654321"))

    def test_simulated_limit_order_routes_exact_account_and_never_unlocks(self):
        client, ctx, api = gateway()
        result = client.place_order(self.request, submit=True, unlock_password="unused")
        self.assertEqual(result.order_id.tolist(), ["987654321"])
        api.OpenSecTradeContext.assert_called_once_with(
            filter_trdmarket="US", host="127.0.0.2", port=11112,
            security_firm="FUTUSECURITIES")
        ctx.place_order.assert_called_once_with(
            price=100.25, qty=2, code="US.AAPL", trd_side="BUY", order_type="NORMAL",
            trd_env="SIMULATE", acc_id=101, adjust_limit=0, remark="ndxbots",
            time_in_force="DAY", fill_outside_rth=False, session="RTH")
        ctx.unlock_trade.assert_not_called()
        ctx.close.assert_called_once()

    def test_market_sell_uses_market_enum_and_zero_sdk_price(self):
        client, ctx, _ = gateway()
        client.place_order(OrderRequest("US.MSFT", "SELL", 3, "MARKET"), submit=True)
        request = ctx.place_order.call_args.kwargs
        self.assertEqual((request["code"], request["trd_side"], request["qty"],
                          request["order_type"], request["price"]),
                         ("US.MSFT", "SELL", 3, "MARKET", 0))
        ctx.close.assert_called_once()

    def test_real_confirmation_guard_runs_before_sdk_loading(self):
        client, ctx, _ = gateway(env="REAL", acc_id=202)
        with patch.object(client, "_sdk", side_effect=AssertionError("guard opened SDK")):
            for action in [lambda: client.place_order(self.request, submit=True),
                           lambda: client.cancel_order("987654321", submit=True)]:
                with self.subTest(action=action), self.assertRaisesRegex(OrderError, "confirm-real"):
                    action()
            self.assertFalse(client.place_order(self.request)["submitted"])
        ctx.unlock_trade.assert_not_called()

    def test_missing_account_rejected_before_connect_for_reads_and_writes(self):
        client, _, _ = gateway(acc_id=None)
        with patch.object(client, "_sdk", side_effect=AssertionError("guard opened SDK")):
            for action in [lambda: client.place_order(self.request, submit=True),
                           lambda: client.cancel_order("987654321", submit=True), client.orders]:
                with self.subTest(action=action), self.assertRaisesRegex(OrderError, "acc-id"):
                    action()

    def test_submit_must_be_boolean_before_connecting(self):
        client, _, api = gateway()
        for submit in ["False", "True", 0, 1, None]:
            with self.subTest(submit=submit):
                with self.assertRaisesRegex(OrderError, "boolean"):
                    client.place_order(self.request, submit=submit)
                with self.assertRaisesRegex(OrderError, "boolean"):
                    client.cancel_order("987654321", submit=submit)
        api.OpenSecTradeContext.assert_not_called()

    def test_invalid_account_ids_and_environment_rejected(self):
        for acc_id in [True, False, 0, -1, 1.2, float("nan"), float("inf")]:
            with self.subTest(acc_id=acc_id), self.assertRaises(OrderError):
                gateway(acc_id=acc_id)
        with self.assertRaises(OrderError):
            gateway(env="PAPER")

    def test_real_account_checked_before_unlock_and_submission(self):
        client, ctx, _ = gateway(env="REAL", acc_id=202)
        events = []
        ctx.get_acc_list.side_effect = lambda: (events.append("accounts") or 0,
                                              pd.DataFrame([account(202, "REAL")]))
        ctx.unlock_trade.side_effect = lambda **kwargs: (events.append("unlock") or 0, "ok")
        ctx.place_order.side_effect = lambda **kwargs: (events.append("place") or 0, pd.DataFrame())
        client.place_order(self.request, submit=True, confirm_real=True, unlock_password="secret")
        self.assertEqual(events, ["accounts", "unlock", "place"])
        ctx.unlock_trade.assert_called_once_with(password="secret")
        self.assertEqual(ctx.place_order.call_args.kwargs["acc_id"], 202)
        self.assertEqual(ctx.place_order.call_args.kwargs["trd_env"], "REAL")
        ctx.close.assert_called_once()

    def test_real_gui_session_does_not_call_unlock_without_password(self):
        client, ctx, _ = gateway(env="REAL", acc_id=202)
        client.place_order(self.request, submit=True, confirm_real=True)
        ctx.unlock_trade.assert_not_called()
        ctx.place_order.assert_called_once()
        ctx.close.assert_called_once()

    def test_account_mismatch_cannot_unlock_or_send(self):
        for rows in [[account(202)], [account(202, "REAL", markets=["HK"])],
                     [account(202, "REAL", status="DISABLED")],
                     [account(203, "REAL")],
                     [account(202, "REAL"), account(202, "REAL")]]:
            with self.subTest(rows=rows):
                client, ctx, _ = gateway(env="REAL", acc_id=202, accounts=rows)
                with self.assertRaisesRegex(OrderError, "not an active US REAL"):
                    client.place_order(self.request, submit=True, confirm_real=True,
                                       unlock_password="secret")
                ctx.unlock_trade.assert_not_called()
                ctx.place_order.assert_not_called()
                ctx.close.assert_called_once()

    def test_accounts_filters_environment_market_status_and_simulated_account_type(self):
        rows = [account(1, sim_type="STOCK"), account(2, sim_type="STOCK_AND_OPTION"),
                account(3, sim_type="OPTION"), account(4, sim_type="FUTURES"),
                account(5, status="DISABLED"), account(6, markets=["HK"]),
                account(7, "REAL"), account(8, markets="US")]
        client, ctx, _ = gateway(accounts=rows)
        self.assertEqual(client.accounts().acc_id.tolist(), [1, 2])
        ctx.place_order.assert_not_called()
        ctx.unlock_trade.assert_not_called()
        ctx.close.assert_called_once()

    def test_both_supported_simulated_stock_account_types_can_submit(self):
        for sim_type in ["STOCK", "STOCK_AND_OPTION"]:
            with self.subTest(sim_type=sim_type):
                client, ctx, _ = gateway(accounts=[account(sim_type=sim_type)])
                client.place_order(self.request, submit=True)
                ctx.place_order.assert_called_once()
                ctx.close.assert_called_once()
        client, ctx, _ = gateway(accounts=[account(sim_type="OPTION")])
        with self.assertRaises(OrderError):
            client.place_order(self.request, submit=True)
        ctx.place_order.assert_not_called()
        ctx.close.assert_called_once()

    def test_missing_account_routing_fields_fail_closed(self):
        for field in ["acc_id", "trd_env", "trdmarket_auth", "acc_status", "sim_acc_type"]:
            with self.subTest(field=field):
                row = account()
                del row[field]
                client, ctx, _ = gateway(accounts=[row])
                with self.assertRaisesRegex(OrderError, "missing"):
                    client.place_order(self.request, submit=True)
                ctx.place_order.assert_not_called()
                ctx.close.assert_called_once()

    def test_submission_rejection_or_timeout_never_retried_and_context_closed(self):
        for response in [(-1, "rejected"), TimeoutError("transport timed out")]:
            with self.subTest(response=response):
                client, ctx, _ = gateway()
                if isinstance(response, Exception):
                    ctx.place_order.side_effect = response
                else:
                    ctx.place_order.return_value = response
                with self.assertRaisesRegex(OrderError, "[Qq]uery orders"):
                    client.place_order(self.request, submit=True)
                ctx.place_order.assert_called_once()
                ctx.close.assert_called_once()

    def test_unlock_error_is_sanitized_and_cannot_send(self):
        for response in [(-1, "secret"), RuntimeError("secret")]:
            with self.subTest(response=response):
                client, ctx, _ = gateway(env="REAL", acc_id=202)
                if isinstance(response, Exception):
                    ctx.unlock_trade.side_effect = response
                else:
                    ctx.unlock_trade.return_value = response
                with self.assertRaises(OrderError) as caught:
                    client.place_order(self.request, submit=True, confirm_real=True,
                                       unlock_password="secret")
                self.assertNotIn("secret", str(caught.exception))
                ctx.unlock_trade.assert_called_once()
                ctx.place_order.assert_not_called()
                ctx.close.assert_called_once()

    def test_account_query_error_closes_without_send(self):
        client, ctx, _ = gateway()
        ctx.get_acc_list.return_value = (-1, "account access denied")
        with self.assertRaisesRegex(OrderError, "account access denied"):
            client.place_order(self.request, submit=True)
        ctx.place_order.assert_not_called()
        ctx.close.assert_called_once()

    def test_orders_refreshes_explicit_account_and_order_id_without_unlocking(self):
        client, ctx, _ = gateway(env="REAL", acc_id=202)
        client.orders(" 987654321 ")
        ctx.order_list_query.assert_called_once_with(
            order_id="987654321", trd_env="REAL", acc_id=202,
            order_market="US", refresh_cache=True)
        ctx.unlock_trade.assert_not_called()
        ctx.close.assert_called_once()

    def test_orders_query_error_closes_context(self):
        client, ctx, _ = gateway()
        ctx.order_list_query.return_value = (-1, "query rejected")
        with self.assertRaisesRegex(OrderError, "query rejected"):
            client.orders()
        ctx.order_list_query.assert_called_once()
        ctx.close.assert_called_once()

    def test_cancel_routes_exact_order_account_and_environment(self):
        client, ctx, _ = gateway(env="REAL", acc_id=202)
        client.cancel_order("987654321", submit=True, confirm_real=True)
        ctx.modify_order.assert_called_once_with(
            modify_order_op="CANCEL", order_id="987654321", qty=0, price=0,
            trd_env="REAL", acc_id=202)
        ctx.place_order.assert_not_called()
        ctx.unlock_trade.assert_not_called()
        ctx.close.assert_called_once()

    def test_invalid_cancel_ids_fail_before_connection(self):
        client, _, api = gateway()
        for order_id in [None, "", " ", "0", "-1", "1.5", "abc"]:
            with self.subTest(order_id=order_id), self.assertRaises(OrderError):
                client.cancel_order(order_id, submit=True)
        api.OpenSecTradeContext.assert_not_called()

    def test_cancel_error_or_timeout_never_retried_and_context_closed(self):
        for response in [(-1, "cannot cancel"), TimeoutError("transport timed out")]:
            with self.subTest(response=response):
                client, ctx, _ = gateway()
                if isinstance(response, Exception):
                    ctx.modify_order.side_effect = response
                else:
                    ctx.modify_order.return_value = response
                with self.assertRaisesRegex(OrderError, "[Qq]uery orders"):
                    client.cancel_order("987654321", submit=True)
                ctx.modify_order.assert_called_once()
                ctx.close.assert_called_once()


class CommandLineTests(unittest.TestCase):
    PLACE = ["place", "--code", "US.AAPL", "--side", "BUY", "--qty", "1", "--price", "100"]

    def test_preview_prints_json_without_import_or_password_prompt(self):
        for argv in [self.PLACE, ["cancel", "--order-id", "987654321"],
                     self.PLACE + ["--env", "REAL", "--unlock"]]:
            with self.subTest(argv=argv), \
                 patch.object(FutuOrderClient, "_sdk", side_effect=AssertionError("preview connected")), \
                 patch("ndxbots.order.getpass.getpass") as prompt, \
                 redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(argv), 0)
                result = json.loads(output.getvalue())
                self.assertFalse(result["submitted"])
                prompt.assert_not_called()

    def test_real_guard_fails_before_password_prompt_or_connection(self):
        with patch.object(FutuOrderClient, "_sdk", side_effect=AssertionError("guard connected")), \
             patch("ndxbots.order.getpass.getpass") as prompt, \
             redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as caught:
                main(self.PLACE + ["--env", "REAL", "--acc-id", "202", "--submit", "--unlock"])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("confirm-real", error.getvalue())
        prompt.assert_not_called()

    def test_invalid_cli_quantity_cannot_connect(self):
        with patch.object(FutuOrderClient, "_sdk", side_effect=AssertionError("invalid request connected")), \
             redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as caught:
                main(["place", "--code", "US.AAPL", "--side", "BUY", "--qty", "NaN",
                      "--price", "100", "--acc-id", "101", "--submit"])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("positive integer", error.getvalue())

    def test_real_cli_env_password_and_secure_prompt_reach_gateway_only_on_submit(self):
        argv = self.PLACE + ["--env", "REAL", "--acc-id", "202", "--submit", "--confirm-real"]
        for environment, flags, expected, prompts in [
            ({"FUTU_TRADE_PASSWORD": "from-env"}, [], "from-env", 0),
            ({}, ["--unlock"], "from-prompt", 1),
            ({}, [], None, 0),
        ]:
            with self.subTest(flags=flags, environment=bool(environment)), \
                 patch.dict("os.environ", environment, clear=True), \
                 patch("ndxbots.order.getpass.getpass", return_value="from-prompt") as prompt, \
                 patch.object(FutuOrderClient, "place_order", return_value={"submitted": True}) as place, \
                 redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(argv + flags), 0)
                self.assertEqual(place.call_args.kwargs["unlock_password"], expected)
                self.assertTrue(place.call_args.kwargs["submit"])
                self.assertTrue(place.call_args.kwargs["confirm_real"])
                self.assertEqual(prompt.call_count, prompts)
                self.assertNotIn("from-env", output.getvalue())
                self.assertNotIn("from-prompt", output.getvalue())


if __name__ == "__main__":
    unittest.main()
