"""Offline tests for paper-account routing, validation and SDK conventions."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd

from ndxbots.order import OrderError, OrderRequest
from ndxbots.paper.broker import FutuPaperBroker


def order_row(**changes):
    row = dict(order_id="90071992547410001", remark="ndxp-abc", code="US.AAPL",
               trd_side="BUY", qty=10.0, dealt_qty=4.0, dealt_avg_price=100.25,
               order_status="FILLED_PART", currency="USD")
    row.update(changes)
    return row


def position_row(**changes):
    row = dict(code="US.AAPL", qty=10.0, can_sell_qty=8.0,
               position_side="LONG", currency="USD")
    row.update(changes)
    return row


def gateway():
    accounts = pd.DataFrame([
        dict(acc_id=101, trd_env="SIMULATE", trdmarket_auth=["US"],
             acc_status="ACTIVE", sim_acc_type="STOCK_AND_OPTION"),
        dict(acc_id=202, trd_env="REAL", trdmarket_auth=["US"],
             acc_status="ACTIVE", sim_acc_type="N/A"),
    ])
    trade = SimpleNamespace(
        get_acc_list=Mock(return_value=(0, accounts)),
        accinfo_query=Mock(return_value=(0, pd.DataFrame([dict(
            total_assets=25000., cash=15000., frozen_cash=1000.,
            currency="USD", us_cash=15000., usd_net_cash_power=13500.,
        )]))),
        position_list_query=Mock(return_value=(0, pd.DataFrame([position_row()]))),
        order_list_query=Mock(return_value=(0, pd.DataFrame([order_row()]))),
        acctradinginfo_query=Mock(return_value=(0, pd.DataFrame([dict(
            max_cash_buy=8.9, max_position_sell=7., max_sell_short=6., max_buy_back=5.,
        )]))),
        place_order=Mock(return_value=(0, pd.DataFrame([{"order_id": "90071992547410001"}]))),
        modify_order=Mock(return_value=(0, pd.DataFrame([{"order_id": "90071992547410001"}]))),
        unlock_trade=Mock(), close=Mock(),
    )
    quotes = SimpleNamespace(
        get_market_snapshot=Mock(return_value=(0, pd.DataFrame([dict(
            code="US.AAPL", last_price=102., open_price=101., prev_close_price=100.,
            update_time="2026-10-20 10:00:01", lot_size=1, suspension=False,
        )]))),
        request_trading_days=Mock(return_value=(0, [
            {"time": "2026-11-25", "trade_date_type": "WHOLE"},
            {"time": "2026-11-27", "trade_date_type": "MORNING"},
        ])),
        request_history_kline=Mock(return_value=(0, pd.DataFrame([
            {"time_key": "2026-10-19 00:00:00", "close": 100., "code": "US.AAPL"},
        ]), None)),
        close=Mock(),
    )
    api = SimpleNamespace(
        RET_OK=0, OpenSecTradeContext=Mock(return_value=trade),
        OpenQuoteContext=Mock(return_value=quotes),
        SecurityFirm=SimpleNamespace(FUTUSECURITIES="FUTUSECURITIES"),
        TrdEnv=SimpleNamespace(SIMULATE="SIMULATE", REAL="REAL"),
        TrdMarket=SimpleNamespace(US="US"), TradeDateMarket=SimpleNamespace(US="US"),
        Currency=SimpleNamespace(USD="USD"),
        TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=SimpleNamespace(NORMAL="NORMAL", MARKET="MARKET"),
        Session=SimpleNamespace(RTH="RTH"), TimeInForce=SimpleNamespace(DAY="DAY"),
        ModifyOrderOp=SimpleNamespace(CANCEL="CANCEL"),
        KLType=SimpleNamespace(K_DAY="K_DAY"), AuType=SimpleNamespace(NONE="NONE"),
    )
    settings = SimpleNamespace(futu_host="127.0.0.2", futu_port=11112)
    return FutuPaperBroker(settings, 101, api=api), trade, quotes, api


class PaperBrokerTests(unittest.TestCase):
    def test_construction_does_not_connect(self):
        _, _, _, api = gateway()
        api.OpenSecTradeContext.assert_not_called()
        api.OpenQuoteContext.assert_not_called()

    def test_real_account_cannot_be_selected(self):
        broker, trade, _, _ = gateway()
        broker.acc_id = broker._client.acc_id = 202
        with self.assertRaisesRegex(OrderError, "SIMULATE"):
            broker.snapshot()
        trade.accinfo_query.assert_not_called()
        trade.place_order.assert_not_called()
        trade.close.assert_called_once()

    def test_environment_guard_runs_before_connection(self):
        for target in ["broker", "client"]:
            for method in ["snapshot", "orders", "quote", "calendar", "raw_closes",
                           "place", "cancel", "max_quantity"]:
                with self.subTest(target=target, method=method):
                    broker, _, _, api = gateway()
                    setattr(broker if target == "broker" else broker._client, "env", "REAL")
                    args = {
                        "snapshot": (), "orders": (), "quote": (["US.AAPL"],),
                        "calendar": ("2026-10-19", "2026-10-20"),
                        "raw_closes": (["US.AAPL"], "2026-10-19"),
                        "place": (OrderRequest("US.AAPL", "BUY", 1, price=100),),
                        "cancel": ("1",),
                        "max_quantity": ("US.AAPL", "BUY", 100, "LIMIT"),
                    }[method]
                    with self.assertRaisesRegex(OrderError, "original SIMULATE"):
                        getattr(broker, method)(*args)
                    api.OpenSecTradeContext.assert_not_called()
                    api.OpenQuoteContext.assert_not_called()

    def test_snapshot_routes_fresh_simulated_account_in_usd_and_normalizes_orders(self):
        broker, trade, _, _ = gateway()
        snapshot = broker.snapshot()
        self.assertEqual(snapshot["env"], "SIMULATE")
        self.assertEqual(snapshot["acc_id"], 101)
        self.assertEqual(snapshot["cash"], 13500.)
        self.assertEqual(snapshot["nav"], 25000.)
        self.assertEqual(snapshot["positions"], {"US.AAPL": 10})
        self.assertEqual(snapshot["sellable"], {"US.AAPL": 8})
        self.assertEqual(snapshot["orders"][0]["order_id"], "90071992547410001")
        self.assertEqual(snapshot["orders"][0]["status"], "FILLED_PART")
        self.assertEqual(snapshot["orders"][0]["dealt_qty"], 4)
        trade.accinfo_query.assert_called_once_with(
            trd_env="SIMULATE", acc_id=101, refresh_cache=True, currency="USD")
        trade.position_list_query.assert_called_once_with(
            trd_env="SIMULATE", acc_id=101, position_market="US", refresh_cache=True)
        trade.order_list_query.assert_called_once_with(
            trd_env="SIMULATE", acc_id=101, order_market="US", refresh_cache=True)
        trade.place_order.assert_not_called()
        trade.modify_order.assert_not_called()
        trade.unlock_trade.assert_not_called()
        trade.close.assert_called_once()

    def test_short_position_direction_is_not_inferred_from_quantity_sign(self):
        for quantity, available in [(4., 3.), (-4., -3.)]:
            with self.subTest(quantity=quantity):
                broker, trade, _, _ = gateway()
                trade.position_list_query.return_value = (0, pd.DataFrame([
                    position_row(position_side="SHORT", qty=quantity, can_sell_qty=available)
                ]))
                snapshot = broker.snapshot()
                self.assertEqual(snapshot["positions"], {"US.AAPL": -4})
                self.assertEqual(snapshot["sellable"], {"US.AAPL": 3})

    def test_empty_positions_and_orders_are_valid(self):
        broker, trade, _, _ = gateway()
        trade.position_list_query.return_value = (0, pd.DataFrame())
        trade.order_list_query.return_value = (0, pd.DataFrame())
        self.assertEqual(broker.snapshot()["positions"], {})
        self.assertEqual(broker.orders(), [])

    def test_negative_cash_is_zero_and_margin_power_does_not_supply_cash(self):
        broker, trade, _, _ = gateway()
        trade.accinfo_query.return_value = (0, pd.DataFrame([dict(
            total_assets=1000., cash=-100., frozen_cash=0., power=100000., currency="USD",
        )]))
        self.assertEqual(broker.snapshot()["cash"], 0.)

    def test_per_currency_cash_takes_precedence_over_deprecated_placeholder(self):
        broker, trade, _, _ = gateway()
        trade.accinfo_query.return_value = (0, pd.DataFrame([dict(
            total_assets=2000., cash=0., us_cash=1000., frozen_cash=100.,
            usd_net_cash_power=800., currency="USD",
        )]))
        self.assertEqual(broker.snapshot()["cash"], 800.)

    def test_old_single_us_account_can_establish_usd_without_currency_field(self):
        broker, trade, _, _ = gateway()
        trade.accinfo_query.return_value = (0, pd.DataFrame([dict(
            total_assets=1000., cash=700., frozen_cash=100.,
        )]))
        self.assertEqual(broker.snapshot()["cash"], 600.)
        accounts = trade.get_acc_list.return_value[1].copy()
        accounts.at[0, "trdmarket_auth"] = ["US", "HK"]
        trade.get_acc_list.return_value = (0, accounts)
        with self.assertRaisesRegex(OrderError, "USD denomination"):
            broker.snapshot()

    def test_invalid_funds_and_wrong_identity_fail_closed(self):
        for updates in [
            {"currency": "HKD"}, {"total_assets": 0}, {"us_cash": "invalid"},
            {"frozen_cash": -1}, {"usd_net_cash_power": "unknown"},
            {"trd_env": "REAL"}, {"acc_id": 202},
        ]:
            with self.subTest(updates=updates):
                broker, trade, _, _ = gateway()
                row = trade.accinfo_query.return_value[1].iloc[0].to_dict()
                row.update(updates)
                trade.accinfo_query.return_value = (0, pd.DataFrame([row]))
                with self.assertRaises(OrderError):
                    broker.snapshot()
                trade.place_order.assert_not_called()
                trade.close.assert_called_once()

    def test_invalid_or_non_stock_holdings_are_not_silently_ignored(self):
        for updates in [
            {"qty": 0.5}, {"qty": -1, "position_side": "LONG"},
            {"can_sell_qty": 11}, {"position_side": "NONE"},
            {"code": "US.AAPL261120C200000"}, {"code": "HK.00700"},
            {"currency": "HKD"}, {"acc_id": 202}, {"trd_env": "REAL"},
        ]:
            with self.subTest(updates=updates):
                broker, trade, _, _ = gateway()
                trade.position_list_query.return_value = (0, pd.DataFrame([position_row(**updates)]))
                with self.assertRaises(OrderError):
                    broker.snapshot()
                trade.close.assert_called_once()

    def test_duplicate_positions_or_orders_rejected(self):
        broker, trade, _, _ = gateway()
        trade.position_list_query.return_value = (0, pd.DataFrame([position_row(), position_row()]))
        with self.assertRaisesRegex(OrderError, "Duplicate"):
            broker.snapshot()
        trade.order_list_query.return_value = (0, pd.DataFrame([order_row(), order_row()]))
        with self.assertRaisesRegex(OrderError, "duplicate"):
            broker.orders()

    def test_invalid_order_totals_identity_and_instruments_rejected(self):
        for updates in [
            {"qty": 0}, {"qty": 2.5}, {"dealt_qty": 11},
            {"dealt_qty": -1}, {"dealt_avg_price": 0},
            {"trd_side": "BUY_BACK"}, {"trd_env": "REAL"},
            {"acc_id": 202}, {"currency": "HKD"},
            {"order_id": 1.5}, {"remark": None}, {"order_status": ""},
            {"code": "US.AAPL261120C200000"},
        ]:
            with self.subTest(updates=updates):
                broker, trade, _, _ = gateway()
                trade.order_list_query.return_value = (0, pd.DataFrame([order_row(**updates)]))
                with self.assertRaises(OrderError):
                    broker.orders()
                trade.close.assert_called_once()

    def test_quotes_have_exchange_timezone_with_dst(self):
        broker, _, quotes, _ = gateway()
        quote = broker.quote(["US.AAPL"])["US.AAPL"]
        self.assertEqual(quote["updated_at"], "2026-10-20T10:00:01-04:00")
        self.assertEqual((quote["price"], quote["open"], quote["prev_close"]), (102., 101., 100.))
        self.assertFalse(quote["suspended"])
        quotes.close.assert_called_once()
        frame = quotes.get_market_snapshot.return_value[1].copy()
        frame.loc[0, "update_time"] = "2026-12-01 10:00:01"
        quotes.get_market_snapshot.return_value = (0, frame)
        self.assertTrue(broker.quote(["US.AAPL"])["US.AAPL"]["updated_at"].endswith("-05:00"))

    def test_invalid_missing_or_duplicate_quotes_fail_closed(self):
        for updates in [
            {"last_price": 0}, {"open_price": float("nan")},
            {"prev_close_price": -1}, {"update_time": "not a date"},
            {"lot_size": 100}, {"suspension": "False"}, {"code": "US.MSFT"},
        ]:
            with self.subTest(updates=updates):
                broker, _, quotes, _ = gateway()
                row = quotes.get_market_snapshot.return_value[1].iloc[0].to_dict()
                row.update(updates)
                quotes.get_market_snapshot.return_value = (0, pd.DataFrame([row]))
                with self.assertRaises(OrderError):
                    broker.quote(["US.AAPL"])
                quotes.close.assert_called_once()
        broker, _, quotes, _ = gateway()
        frame = quotes.get_market_snapshot.return_value[1]
        quotes.get_market_snapshot.return_value = (0, pd.concat([frame, frame]))
        with self.assertRaisesRegex(OrderError, "match requested"):
            broker.quote(["US.AAPL"])

    def test_calendar_uses_broker_half_day_without_weekday_guessing(self):
        broker, _, quotes, _ = gateway()
        self.assertEqual(broker.calendar("2026-11-25", "2026-11-27"), [
            {"date": "2026-11-25", "close_minute": 960},
            {"date": "2026-11-27", "close_minute": 780},
        ])
        quotes.request_trading_days.assert_called_once_with(
            market="US", start="2026-11-25", end="2026-11-27")
        quotes.close.assert_called_once()
        quotes.request_trading_days.return_value = (0, [])
        self.assertEqual(broker.calendar("2026-11-26", "2026-11-26"), [])

    def test_calendar_out_of_range_rejected(self):
        broker, _, _, _ = gateway()
        with self.assertRaisesRegex(OrderError, "outside"):
            broker.calendar("2026-11-27", "2026-11-27")

    def test_raw_closes_are_exact_date_unadjusted_rth(self):
        broker, _, quotes, _ = gateway()
        self.assertEqual(broker.raw_closes(["US.AAPL"], "2026-10-19"), {"US.AAPL": 100.})
        quotes.request_history_kline.assert_called_once_with(
            "US.AAPL", start="2026-10-19", end="2026-10-19",
            ktype="K_DAY", autype="NONE", max_count=2, session="RTH")
        quotes.close.assert_called_once()

    def test_raw_closes_never_backfill_previous_day_or_accept_pagination(self):
        for update, page in [
            ({"time_key": "2026-10-16 00:00:00"}, None),
            ({"code": "US.MSFT"}, None), ({"close": 0}, None), ({}, "more"),
        ]:
            with self.subTest(update=update, page=page):
                broker, _, quotes, _ = gateway()
                row = quotes.request_history_kline.return_value[1].iloc[0].to_dict()
                row.update(update)
                quotes.request_history_kline.return_value = (0, pd.DataFrame([row]), page)
                with self.assertRaises(OrderError):
                    broker.raw_closes(["US.AAPL"], "2026-10-19")
                quotes.close.assert_called_once()

    def test_quantity_uses_cash_and_distinguishes_separate_exit_legs(self):
        for side, action, expected in [
            ("BUY", "entry", 8), ("SELL", "entry", 6),
            ("BUY", "exit", 5), ("SELL", "exit", 7),
        ]:
            with self.subTest(side=side, action=action):
                broker, trade, _, _ = gateway()
                self.assertEqual(broker.max_quantity("US.AAPL", side, 100, "MARKET", action=action),
                                 expected)
                trade.acctradinginfo_query.assert_called_once_with(
                    order_type="MARKET", code="US.AAPL", price=100., adjust_limit=0,
                    trd_env="SIMULATE", acc_id=101, session="RTH")
                trade.close.assert_called_once()
                trade.place_order.assert_not_called()

    def test_invalid_tradable_quantity_or_identity_is_rejected(self):
        for row in [
            {"max_cash_buy": -1}, {"max_cash_buy": float("nan")},
            {"max_cash_buy": 1, "trd_env": "REAL"},
            {"max_cash_buy": 1, "acc_id": 202}, {"max_cash_and_margin_buy": 100},
        ]:
            with self.subTest(row=row):
                broker, trade, _, _ = gateway()
                trade.acctradinginfo_query.return_value = (0, pd.DataFrame([row]))
                with self.assertRaises(OrderError):
                    broker.max_quantity("US.AAPL", "BUY", 100, "LIMIT")

    def test_place_routes_paper_only_and_preserves_large_id(self):
        broker, trade, _, _ = gateway()
        request = OrderRequest("US.AAPL", "BUY", 2, price=100., remark="ndxp-abc")
        self.assertEqual(broker.place(request), "90071992547410001")
        args = trade.place_order.call_args.kwargs
        self.assertEqual((args["trd_env"], args["acc_id"], args["session"]),
                         ("SIMULATE", 101, "RTH"))
        self.assertEqual((args["qty"], args["remark"], args["time_in_force"]),
                         (2, "ndxp-abc", "DAY"))
        trade.unlock_trade.assert_not_called()
        trade.place_order.assert_called_once()

    def test_place_timeout_or_malformed_ack_is_never_retried(self):
        request = OrderRequest("US.AAPL", "BUY", 2, price=100.)
        for response in [
            TimeoutError("transport"), pd.DataFrame(), pd.DataFrame([{"order_id": "bad"}]),
            pd.DataFrame([{"order_id": "1", "trd_env": "REAL"}]),
            pd.DataFrame([{"order_id": "1", "acc_id": 202}]),
            pd.DataFrame([{"order_id": "1", "code": "US.MSFT"}]),
            pd.DataFrame([{"order_id": "1", "qty": 3}]),
        ]:
            with self.subTest(response=type(response).__name__):
                broker, trade, _, _ = gateway()
                if isinstance(response, Exception):
                    trade.place_order.side_effect = response
                else:
                    trade.place_order.return_value = (0, response)
                with self.assertRaisesRegex(OrderError, "[Uu]nknown"):
                    broker.place(request)
                trade.place_order.assert_called_once()
                trade.close.assert_called_once()

    def test_cancel_routes_paper_and_requires_subsequent_query(self):
        broker, trade, _, _ = gateway()
        self.assertIsNone(broker.cancel("90071992547410001"))
        trade.modify_order.assert_called_once_with(
            modify_order_op="CANCEL", order_id="90071992547410001",
            qty=0, price=0, trd_env="SIMULATE", acc_id=101)
        trade.order_list_query.assert_not_called()
        trade.unlock_trade.assert_not_called()

    def test_cancel_timeout_or_wrong_ack_is_never_retried(self):
        for response in [TimeoutError("transport"), pd.DataFrame([{"order_id": "1"}])]:
            broker, trade, _, _ = gateway()
            if isinstance(response, Exception):
                trade.modify_order.side_effect = response
            else:
                trade.modify_order.return_value = (0, response)
            with self.assertRaisesRegex(OrderError, "[Uu]nknown"):
                broker.cancel("90071992547410001")
            trade.modify_order.assert_called_once()
            trade.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
