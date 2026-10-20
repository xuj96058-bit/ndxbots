"""Offline malformed-state cases that must fail before broker mutations."""
from copy import deepcopy
import unittest

from ndxbots.paper.validation import validate_plan, validate_state


def entry(code="US.A", side=1):
    return {"key": f"{code}:entry", "code": code, "action": "entry", "side": side,
            "quantity": 0, "notional": 1000., "prev_close": 100., "atr": 10.,
            "urgent": False}


def exit_leg(code="US.A", side=1, quantity=10):
    return {"key": f"{code}:exit", "code": code, "action": "exit", "side": side,
            "quantity": quantity, "notional": 0., "prev_close": 100., "atr": 0.,
            "urgent": True}


def plan():
    return {"version": 1, "env": "SIMULATE", "acc_id": 123,
            "signal_date": "2024-01-05", "execution_date": "2024-01-08",
            "close_minute": 960, "max_hold": 2, "cost_bps": 0.,
            "positions": {}, "legs": [entry()]}


def state():
    return {"version": 1, "plan": plan(), "plan_id": "engine-owned-hash",
            "status": "working", "last_tick": "2024-01-08T10:00:00-05:00",
            "legs": {"US.A:entry": {"budget": 1000., "observed_due": True,
                                    "initial_done": False, "done": False}},
            "orders": [{"leg": "US.A:entry", "code": "US.A", "side": "BUY",
                        "qty": 10, "dealt_qty": 0, "dealt_avg_price": 0.,
                        "order_id": None, "remark": "intent-1", "status": "INTENT",
                        "order_type": "MARKET", "price": None,
                        "cancel_requested": False, "attempt": 0,
                        "created_at": "2024-01-08T10:00:00-05:00"}]}


class PlanValidationTests(unittest.TestCase):
    def test_valid_plan_preserves_optional_metadata_without_mutation(self):
        doc = plan()
        doc.update(signal_nav=1000., signal_cash=0., gross=1.3, effective_gross=1.,
                   targets={"US.A": 10.5}, reduction_targets={"US.B": -2.},
                   strategy_config_hash="optional", diagnostic={"reason": "example"})
        before = deepcopy(doc)
        self.assertIsNone(validate_plan(doc))
        self.assertEqual(doc, before)
        del doc["version"]  # Generic caller plans need not duplicate state version.
        validate_plan(doc)

    def test_environment_and_exact_account_identity(self):
        for name, value in (("env", "REAL"), ("acc_id", True), ("acc_id", "123"),
                            ("acc_id", 123.0), ("acc_id", 0), ("version", 2)):
            with self.subTest(name=name, value=value):
                doc = plan()
                doc[name] = value
                with self.assertRaises(ValueError):
                    validate_plan(doc)

    def test_canonical_dates_must_be_forward_weekdays(self):
        for name, value in (("signal_date", "2024-1-5"), ("signal_date", "2024-02-30"),
                            ("signal_date", "2024-01-06"), ("execution_date", "2024-01-07"),
                            ("execution_date", "2024-01-05"),
                            ("execution_date", "2024-01-04"),
                            ("execution_date", "2024-01-08T00:00:00")):
            with self.subTest(name=name, value=value):
                doc = plan()
                doc[name] = value
                with self.assertRaises(ValueError):
                    validate_plan(doc)

    def test_session_and_limits_reject_boolean_or_float_integer_values(self):
        for name, value in (("close_minute", 780.0), ("close_minute", 900),
                            ("max_hold", True), ("max_hold", 2.5), ("max_hold", 0)):
            with self.subTest(name=name, value=value):
                doc = plan()
                doc[name] = value
                with self.assertRaises(ValueError):
                    validate_plan(doc)
        doc = plan()
        doc["close_minute"] = 780
        validate_plan(doc)

    def test_cost_and_optional_financial_values_are_finite_numbers(self):
        for name, value in (("cost_bps", "5"), ("cost_bps", True), ("cost_bps", -1),
                            ("cost_bps", 10000), ("cost_bps", float("nan")),
                            ("gross", float("inf")), ("signal_nav", 0),
                            ("signal_cash", -1), ("effective_gross", -0.1)):
            with self.subTest(name=name, value=value):
                doc = plan()
                doc[name] = value
                with self.assertRaises(ValueError):
                    validate_plan(doc)

    def test_positions_require_whole_supported_us_stocks(self):
        for positions in ({"HK.00700": 1}, {"US.A260101C100000": 1}, {"US.A": True},
                          {"US.A": 1.0}, {"US.A": "1"}, {"US.A": 1.5}, []):
            with self.subTest(positions=positions):
                doc = plan()
                doc["positions"] = positions
                with self.assertRaises(ValueError):
                    validate_plan(doc)
        doc = plan()
        doc["positions"] = {"US.BRK.B": 5, "US.BF-B": -2}
        validate_plan(doc)

    def test_leg_keys_must_be_unique_and_canonical(self):
        doc = plan()
        doc["legs"][0]["key"] = "custom"
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_plan(doc)
        doc = plan()
        doc["legs"].append(deepcopy(doc["legs"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_plan(doc)

    def test_entry_requires_positive_prices_budget_atr_and_exact_direction(self):
        for name, value in (("notional", 0), ("atr", 0), ("prev_close", float("inf")),
                            ("notional", "1000"), ("side", True), ("side", 1.0),
                            ("quantity", 5), ("urgent", "false"), ("urgent", True)):
            with self.subTest(name=name, value=value):
                doc = plan()
                doc["legs"][0][name] = value
                with self.assertRaises(ValueError):
                    validate_plan(doc)

    def test_exits_reduce_existing_position_without_exceeding_it(self):
        for side, qty, positions in ((1, 11, {"US.A": 10}), (-1, 5, {"US.A": 10}),
                                     (1, 1, {}), (1, 1.0, {"US.A": 10}),
                                     (1, True, {"US.A": 10})):
            with self.subTest(side=side, qty=qty, positions=positions):
                doc = plan()
                doc.update(positions=positions, legs=[exit_leg(side=side, quantity=qty)])
                with self.assertRaises(ValueError):
                    validate_plan(doc)
        doc = plan()
        doc.update(positions={"US.A": -10}, legs=[exit_leg(side=-1)])
        validate_plan(doc)

    def test_reversal_requires_full_opposite_exit_with_same_reference(self):
        doc = plan()
        doc.update(positions={"US.A": 10}, legs=[entry(side=-1)])
        with self.assertRaisesRegex(ValueError, "full exit"):
            validate_plan(doc)
        doc["legs"].insert(0, exit_leg(quantity=5))
        with self.assertRaisesRegex(ValueError, "full exit"):
            validate_plan(doc)
        doc["legs"][0]["quantity"] = 10
        validate_plan(doc)
        doc["legs"][1]["prev_close"] = 101.
        with self.assertRaisesRegex(ValueError, "previous close"):
            validate_plan(doc)
        doc["legs"][1]["prev_close"] = 100.
        doc["legs"][1]["side"] = 1
        with self.assertRaisesRegex(ValueError, "same direction"):
            validate_plan(doc)

    def test_existing_excess_holdings_can_all_be_reduced(self):
        doc = plan()
        doc.update(max_hold=1, positions={"US.A": 10, "US.B": 10, "US.C": 10},
                   legs=[exit_leg(code) for code in ("US.A", "US.B", "US.C")])
        validate_plan(doc)
        doc["legs"].extend([entry("US.D"), entry("US.E")])
        with self.assertRaisesRegex(ValueError, "Entry legs exceed"):
            validate_plan(doc)

    def test_empty_plan_is_valid_and_missing_shapes_raise_valueerror(self):
        doc = plan()
        doc["legs"] = []
        validate_plan(doc)
        for change in (None, [], {}, {"env": "SIMULATE"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    validate_plan(change)


class StateValidationTests(unittest.TestCase):
    def test_valid_write_ahead_and_partial_fill_states_do_not_mutate(self):
        doc = state()
        before = deepcopy(doc)
        validate_state(doc)
        self.assertEqual(doc, before)
        doc["orders"][0].update(order_id="1234567890123456789", status="FILLED_PART",
                                dealt_qty=3, dealt_avg_price=100.)
        doc["legs"]["US.A:entry"].update(open=102., limit=100., split=True, dust=0.)
        validate_state(doc)

    def test_state_version_and_plan_leg_mapping_are_required(self):
        for name, value in (("version", True), ("version", 2), ("orders", {}),
                            ("legs", {}), ("plan_id", ""), ("status", None)):
            with self.subTest(name=name, value=value):
                doc = state()
                doc[name] = value
                with self.assertRaises(ValueError):
                    validate_state(doc)

    def test_runtime_flags_and_budget_cannot_be_coerced(self):
        for name, value in (("budget", "1000"), ("budget", 1001.), ("budget", float("nan")),
                            ("observed_due", "false"), ("initial_done", 0),
                            ("done", None), ("dust", -1), ("split", 1)):
            with self.subTest(name=name, value=value):
                doc = state()
                doc["legs"]["US.A:entry"][name] = value
                with self.assertRaises(ValueError):
                    validate_state(doc)

    def test_saved_order_must_match_its_leg_identity(self):
        for name, value in (("leg", "US.B:entry"), ("code", "US.B"), ("side", "SELL")):
            with self.subTest(name=name, value=value):
                doc = state()
                doc["orders"][0][name] = value
                with self.assertRaises(ValueError):
                    validate_state(doc)

    def test_saved_quantities_and_cumulative_fill_prices_are_exact(self):
        for name, value in (("qty", True), ("qty", 10.0), ("qty", 0), ("dealt_qty", -1),
                            ("dealt_qty", 11), ("dealt_qty", 1.0),
                            ("dealt_avg_price", float("nan")), ("dealt_avg_price", 100.)):
            with self.subTest(name=name, value=value):
                doc = state()
                doc["orders"][0][name] = value
                with self.assertRaises(ValueError):
                    validate_state(doc)
        doc = state()
        doc["orders"][0]["dealt_qty"] = 1
        with self.assertRaisesRegex(ValueError, "positive average"):
            validate_state(doc)

    def test_saved_remarks_and_order_ids_are_unique(self):
        for change in ({}, {"remark": "different", "order_id": "123"}):
            with self.subTest(change=change):
                doc = state()
                doc["orders"][0]["order_id"] = "123"
                duplicate = deepcopy(doc["orders"][0])
                duplicate.update(change)
                doc["orders"].append(duplicate)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    validate_state(doc)
        for bad_id in (123, "", "0", "123.0", "-1"):
            with self.subTest(order_id=bad_id):
                doc = state()
                doc["orders"][0]["order_id"] = bad_id
                with self.assertRaises(ValueError):
                    validate_state(doc)

    def test_limit_orders_require_positive_numeric_price(self):
        for price in (None, 0, -1, "100", True, float("inf")):
            with self.subTest(price=price):
                doc = state()
                doc["orders"][0].update(order_type="LIMIT", price=price)
                with self.assertRaises(ValueError):
                    validate_state(doc)

    def test_unsupported_status_is_left_for_engine_to_reconcile_safely(self):
        doc = state()
        doc["orders"][0]["status"] = "NEW_BROKER_STATUS"
        validate_state(doc)

    def test_unfunded_entry_and_excess_exit_fills_are_rejected(self):
        doc = state()
        doc["legs"]["US.A:entry"]["budget"] = None
        with self.assertRaisesRegex(ValueError, "funded budget"):
            validate_state(doc)
        doc = state()
        doc["plan"].update(positions={"US.A": 10}, legs=[exit_leg()])
        values = doc["legs"].pop("US.A:entry")
        values["budget"] = None
        doc["legs"]["US.A:exit"] = values
        doc["orders"][0].update(leg="US.A:exit", side="SELL", dealt_qty=6,
                                dealt_avg_price=100.)
        validate_state(doc)
        extra = deepcopy(doc["orders"][0])
        extra.update(remark="intent-2", dealt_qty=5)
        doc["orders"].append(extra)
        with self.assertRaisesRegex(ValueError, "frozen reduction"):
            validate_state(doc)

    def test_saved_clock_values_must_include_timezone(self):
        for timestamp in ("2024-01-08T10:00:00", "bad", 123):
            with self.subTest(timestamp=timestamp):
                doc = state()
                doc["last_tick"] = timestamp
                with self.assertRaisesRegex(ValueError, "aware ISO"):
                    validate_state(doc)


if __name__ == "__main__":
    unittest.main()
