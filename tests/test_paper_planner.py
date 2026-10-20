"""Paper targets use live-account quantities and the existing strategy rules."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ndxbots.config import load_settings
from ndxbots.paper.planner import PlanError, build_plan, config_fingerprint
from ndxbots.strategy.scores import build_score_table
from ndxbots.strategy.targets import decide_quantity_targets


class PaperPlannerTests(unittest.TestCase):
    def setUp(self):
        self.settings = replace(
            load_settings(), use_gross_ladder=False, hurst_gate_mode="legacy_regime",
            cost_bps=0., initial_cash=99_000., max_hold=2,
        )
        self.day = pd.Timestamp("2024-01-05")  # Friday signal; Monday execution.
        self.dates = pd.bdate_range(end=self.day, periods=25)
        self.panel = pd.DataFrame([
            {"date": day, "code": code, "open": price, "high": price + 1,
             "low": price - 1, "close": price}
            for day in self.dates
            for code, price in (("US.A", 50.), ("US.B", 100.), ("US.QQQ", 400.))
        ])
        self.factors = self.panel[["date", "code"]].assign(
            my_atr_pct_20=.04, my_hurst_126=.7,
            my_ma50_gap=.1, my_struct_gap=.1, my_efficiency_21=.4,
        )
        self.scored = pd.DataFrame([
            {"date": day, "code": code, "side": "long",
             "in_hold": code == "US.A", "in_pool": True}
            for day in self.dates for code in ("US.A", "US.B")
        ])
        self.snapshot = {
            "acc_id": 123, "env": "SIMULATE", "cash": 1_000., "nav": 1_000.,
            "positions": {}, "sellable": {}, "orders": [],
        }
        self.raw = {"US.A": 100., "US.B": 200.}

    def plan(self, **changes):
        options = dict(
            acc_id=123, signal_date=str(self.day.date()), execution_date="2024-01-08",
            close_minute=960, snapshot=self.snapshot, raw_closes=self.raw,
            panel=self.panel, factors=self.factors, scored=self.scored,
        )
        settings = changes.pop("settings", self.settings)
        options.update(changes)
        return build_plan(settings, **options)

    def hold(self, code, qty):
        self.snapshot["positions"][code] = qty
        self.snapshot["sellable"][code] = max(0, qty)

    def select(self, code, side="long"):
        rows = self.scored["code"].eq(code)
        self.scored.loc[rows, "in_hold"] = True
        self.scored.loc[rows, "side"] = side

    def midweek(self):
        self.day = pd.Timestamp("2024-01-03")
        # Future local rows deliberately remain; planner must slice them away.

    def test_new_account_uses_actual_nav_and_raw_prices_not_research_cash(self):
        plan = self.plan()
        self.assertEqual(plan["signal_nav"], 1_000.)
        self.assertEqual(plan["targets"]["US.A"], 10.)
        leg = plan["legs"][0]
        self.assertEqual(leg["notional"], 1_000.)
        self.assertEqual(leg["prev_close"], 100.)
        self.assertEqual(leg["atr"], 4.)  # Adjusted ATR=2, raw/adjusted=2.
        self.assertEqual(leg["quantity"], 0)
        json.dumps(plan, allow_nan=False)

    def test_existing_position_only_adds_incremental_notional(self):
        self.hold("US.A", 8)
        plan = self.plan()
        self.assertEqual(plan["legs"][0]["notional"], 200.)
        self.assertEqual(plan["positions"], {"US.A": 8})

    def test_equal_weights_apply_to_selected_set_including_outside_watch_pool(self):
        self.select("US.B")
        self.scored.loc[self.scored["code"].eq("US.B"), "in_pool"] = False
        plan = self.plan()
        self.assertEqual([leg["notional"] for leg in plan["legs"]], [500., 500.])
        self.assertEqual(plan["targets"], {"US.A": 5., "US.B": 2.5})

    def test_ordinary_reduction_rounds_toward_zero_in_actual_shares(self):
        self.hold("US.A", 13)
        self.snapshot["nav"] = 1_050.
        plan = self.plan()
        self.assertEqual(plan["targets"]["US.A"], 10.)
        self.assertEqual(plan["legs"][0]["quantity"], 3)
        self.assertFalse(plan["legs"][0]["urgent"])
        self.assertEqual(plan["legs"][0]["action"], "exit")

    def test_floating_point_noise_does_not_sell_an_extra_share(self):
        self.hold("US.A", 10)
        self.snapshot["nav"] = 999.9999999999999
        self.assertEqual(self.plan()["legs"], [])

    def test_ordinary_rotation_is_0945_exit_when_direction_still_valid(self):
        self.hold("US.A", 10)
        self.scored.loc[self.scored["code"].eq("US.A"), "in_hold"] = False
        self.select("US.B")
        exit_leg = self.plan()["legs"][0]
        self.assertEqual(exit_leg["quantity"], 10)
        self.assertFalse(exit_leg["urgent"])

    def test_direction_reversal_has_separate_urgent_exit_and_entry(self):
        self.hold("US.A", -10)
        plan = self.plan()
        self.assertEqual([leg["action"] for leg in plan["legs"]], ["exit", "entry"])
        self.assertEqual([leg["side"] for leg in plan["legs"]], [-1, 1])
        self.assertEqual(plan["legs"][0]["quantity"], 10)
        self.assertTrue(plan["legs"][0]["urgent"])
        self.assertEqual(plan["legs"][1]["notional"], 1_000.)

    def test_strict_hurst_blocks_new_long_but_still_covers_old_short(self):
        self.hold("US.A", -10)
        self.factors.loc[self.factors["code"].eq("US.A"), "my_hurst_126"] = .67
        settings = replace(self.settings, hurst_gate_mode="strict_entry")
        plan = self.plan(settings=settings)
        self.assertEqual(len(plan["legs"]), 1)
        self.assertEqual(plan["legs"][0]["action"], "exit")
        self.assertEqual(plan["legs"][0]["reason"], "new_long_hurst_blocked")
        self.assertEqual(plan["targets"]["US.A"], 0)

    def test_legacy_hurst_does_not_add_an_account_entry_gate(self):
        self.factors.loc[self.factors["code"].eq("US.A"), "my_hurst_126"] = .2
        self.assertEqual(self.plan()["legs"][0]["notional"], 1_000.)

    def test_strict_hurst_does_not_block_addition_to_existing_long(self):
        self.hold("US.A", 5)
        self.factors.loc[self.factors["code"].eq("US.A"), "my_hurst_126"] = np.nan
        plan = self.plan(settings=replace(self.settings, hurst_gate_mode="strict_entry"))
        self.assertEqual(plan["legs"][0]["notional"], 500.)

    def test_benchmark_hurst_uses_existing_scale_and_ceiling(self):
        for hurst, scale, gross in [(.8, 1.3, 1.3), (.5, .5, .5), (.55, 1., 1.), (.75, 1., 1.)]:
            with self.subTest(hurst=hurst):
                self.factors.loc[self.factors["code"].eq("US.QQQ"), "my_hurst_126"] = hurst
                plan = self.plan()
                self.assertEqual(plan["gross"], gross)
                self.assertEqual(plan["hurst_scale"], scale)
                self.assertEqual(plan["legs"][0]["notional"], 1_000. * gross)

    def test_midweek_gross_drop_and_retry_keep_whole_share_ceiling(self):
        self.midweek()
        self.hold("US.A", 11)
        self.factors.loc[self.factors["code"].eq("US.QQQ"), "my_hurst_126"] = .5
        plan = self.plan(execution_date="2024-01-04")
        self.assertEqual(plan["targets"]["US.A"], 5.)
        self.assertEqual(plan["legs"][0]["quantity"], 6)
        self.assertTrue(plan["legs"][0]["urgent"])
        self.assertEqual(plan["reduction_targets"], {"US.A": 5.})
        self.day = pd.Timestamp("2024-01-04")
        retry = self.plan(execution_date="2024-01-05", previous_state=plan)
        self.assertEqual(retry["legs"][0]["quantity"], 6)
        self.assertEqual(retry["legs"][0]["reason"], "retry_risk_reduction")

    def test_midweek_gross_rise_does_not_open_or_increase(self):
        self.midweek()
        self.factors.loc[self.factors["code"].eq("US.QQQ"), "my_hurst_126"] = .8
        self.assertEqual(self.plan(execution_date="2024-01-04")["legs"], [])
        self.hold("US.A", 3)
        self.assertEqual(self.plan(execution_date="2024-01-04")["legs"], [])

    def test_missing_current_score_for_held_name_produces_urgent_exit(self):
        self.hold("US.A", 10)
        self.scored = self.scored.loc[
            ~(self.scored["date"].eq(self.day) & self.scored["code"].eq("US.A"))
        ]
        plan = self.plan()
        self.assertEqual(plan["legs"][0]["quantity"], 10)
        self.assertTrue(plan["legs"][0]["urgent"])

    def test_missing_raw_price_blocks_held_name_even_for_exit(self):
        self.hold("US.B", 3)
        with self.assertRaisesRegex(PlanError, "raw T close"):
            self.plan(raw_closes={"US.A": 100.})

    def test_selected_short_has_short_entry_budget(self):
        self.select("US.A", "short")
        plan = self.plan()
        self.assertEqual(plan["targets"]["US.A"], -10.)
        self.assertEqual(plan["legs"][0]["side"], -1)
        self.assertEqual(plan["legs"][0]["notional"], 1_000.)

    def test_shared_quantity_target_parity_before_integer_execution(self):
        self.hold("US.A", 4)
        self.hold("US.B", -1)
        self.select("US.B", "short")
        expected = decide_quantity_targets(
            self.day, np.array([1, -1]), np.array([4., -1.]), np.array([100., 200.]),
            1_000., 1., 1., {}, self.settings.rebalance_weekdays,
        )
        plan = self.plan()
        self.assertEqual(plan["targets"], dict(zip(["US.A", "US.B"], expected.quantities)))

    def test_future_rows_cannot_change_signal_plan(self):
        original = self.plan()
        future = pd.Timestamp("2024-01-08")
        for name in ("panel", "factors", "scored"):
            frame = getattr(self, name)
            tail = frame.loc[frame["date"].eq(self.day)].copy()
            tail["date"] = future
            if name == "factors":
                tail["my_hurst_126"] = .1
            if name == "scored":
                tail["side"] = "short"
                tail["in_hold"] = True
            setattr(self, name, pd.concat([frame, tail], ignore_index=True))
        self.assertEqual(self.plan(), original)

    def test_stale_daily_factors_and_score_tables_are_rejected(self):
        for name in ("panel", "factors", "scored"):
            with self.subTest(name=name):
                frame = getattr(self, name)
                with self.assertRaises(PlanError):
                    self.plan(**{name: frame.loc[frame["date"].lt(self.day)]})

    def test_duplicate_day_code_rows_are_rejected(self):
        for name in ("panel", "factors", "scored"):
            with self.subTest(name=name):
                frame = getattr(self, name)
                with self.assertRaisesRegex(PlanError, "unique"):
                    self.plan(**{name: pd.concat([frame, frame.iloc[[-1]]])})

    def test_infinite_or_missing_atr_blocks_entry(self):
        for invalid in (0., np.nan, np.inf, -1.):
            with self.subTest(value=invalid):
                frame = self.factors.copy()
                frame.loc[frame["code"].eq("US.A"), "my_atr_pct_20"] = invalid
                with self.assertRaisesRegex(PlanError, "ATR20"):
                    self.plan(factors=frame)

    def test_ohlc_atr_fallback_uses_same_raw_conversion(self):
        factors = self.factors.drop(columns="my_atr_pct_20")
        plan = self.plan(factors=factors)
        self.assertEqual(plan["legs"][0]["atr"], 4.)  # TR=2, ratio=2.

    def test_atr_not_required_for_exit_only(self):
        self.hold("US.A", 13)
        frame = self.factors.copy()
        frame["my_atr_pct_20"] = np.nan
        plan = self.plan(factors=frame)
        self.assertEqual(plan["legs"][0]["action"], "exit")
        self.assertEqual(plan["legs"][0]["atr"], 0.)

    def test_regime_missing_policy_is_shared_with_research(self):
        settings = replace(self.settings, use_gross_ladder=True, regime_missing_policy="floor")
        missing = pd.DataFrame(columns=["date"])
        plan = self.plan(settings=settings, regime=missing)
        self.assertEqual(plan["gross"], settings.gross_floor)
        self.assertTrue(plan["gross_reason"].startswith("missing_data_floor:"))
        with self.assertRaisesRegex(ValueError, "Regime exposure unavailable"):
            self.plan(settings=replace(settings, regime_missing_policy="error"), regime=missing)

    def test_real_wrong_account_and_invalid_account_numbers_rejected(self):
        for change in ({"env": "REAL"}, {"acc_id": 456}, {"acc_id": True}):
            with self.subTest(change=change), self.assertRaises(PlanError):
                self.plan(snapshot={**self.snapshot, **change})
        for value in (0, -1, True, 1.5):
            with self.subTest(acc_id=value), self.assertRaises(PlanError):
                self.plan(acc_id=value)

    def test_nonterminal_or_unknown_orders_block_new_plan(self):
        for status in ("SUBMITTED", "FILLED_PART", "CANCELLING_ALL", "unknown", ""):
            with self.subTest(status=status), self.assertRaisesRegex(PlanError, "open or uncertain"):
                self.plan(snapshot={**self.snapshot, "orders": [{"status": status}]})
        for status in ("FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED"):
            with self.subTest(status=status):
                self.plan(snapshot={**self.snapshot, "orders": [{"status": status}]})

    def test_invalid_balance_or_fractional_positions_are_rejected(self):
        for change in ({"nav": 0}, {"nav": np.inf}, {"cash": -1},
                       {"positions": {"US.A": .5}}, {"positions": {"US.A": True}},
                       {"positions": {"US.AAPL260101C100000": 1}}):
            with self.subTest(change=change), self.assertRaises(PlanError):
                self.plan(snapshot={**self.snapshot, **change})

    def test_dates_and_session_close_are_validated(self):
        for change in ({"execution_date": "2024-01-05"},
                       {"execution_date": "2024-01-06"}, {"close_minute": 900},
                       {"signal_date": "2024-01-05 12:00"},
                       {"signal_date": "2024-01-05T00:00:00Z"}):
            with self.subTest(change=change), self.assertRaises(PlanError):
                self.plan(**change)
        self.assertEqual(self.plan(close_minute=780)["close_minute"], 780)

    def test_invalid_previous_state_is_rejected(self):
        for state in ({"env": "REAL"}, {"acc_id": 456}, {"effective_gross": np.nan},
                      {"signal_date": "2024-01-05"}, {"reduction_targets": {"US.A": np.inf}}):
            with self.subTest(state=state), self.assertRaises(PlanError):
                self.plan(previous_state=state)

    def test_selection_max_hold_and_disabled_short_are_enforced(self):
        self.select("US.B")
        with self.assertRaisesRegex(PlanError, "max_hold"):
            self.plan(settings=replace(self.settings, max_hold=1))
        self.select("US.A", "short")
        with self.assertRaisesRegex(PlanError, "allow_short"):
            self.plan(settings=replace(self.settings, allow_short=False))

    def test_real_score_builder_receives_complete_signal_history(self):
        settings = replace(
            self.settings, strategy_top_n=2, qqq_regime_gate=False,
            require_above_ma200=False, space_col="", short_space_col="",
            min_atr_pct=0., rsi_gate=False, macd_gate=False, bb_width_gate=False,
        )
        expected = build_score_table(self.factors, settings)
        expected_codes = set(expected.loc[expected["date"].eq(self.day) & expected["in_hold"], "code"])
        with patch("ndxbots.paper.planner.build_score_table", wraps=build_score_table) as builder:
            plan = self.plan(settings=settings, scored=None)
        self.assertEqual(len(builder.call_args.args[0]), len(self.factors))
        self.assertEqual(set(plan["targets"]), expected_codes)

    def test_loads_only_daily_sources_not_backtest_execution_pool(self):
        with tempfile.TemporaryDirectory(prefix=".paper-plan-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            for name, frame in (("panel", self.panel), ("factors", self.factors)):
                (root / name).mkdir()
                frame.to_parquet(root / name / "daily.parquet", index=False)
            settings = replace(self.settings, data_root=root)
            plan = self.plan(settings=settings, panel=None, factors=None)
            self.assertEqual(plan["legs"][0]["notional"], 1_000.)

    def test_configuration_fingerprint_tracks_strategy_not_machine(self):
        original = config_fingerprint(self.settings)
        self.assertEqual(original, config_fingerprint(replace(
            self.settings, data_root=Path("elsewhere"), futu_port=12345,
        )))
        self.assertNotEqual(original, config_fingerprint(replace(
            self.settings, strategy_quality_weight=.5,
        )))


if __name__ == "__main__":
    unittest.main()
