from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.paper import cli
from ndxbots.paper.engine import ET, install_plan
from ndxbots.paper.planner import config_fingerprint
from ndxbots.paper.store import PaperError, StateStore


class CalendarTests(unittest.TestCase):
    def test_official_holiday_gap_and_half_day(self):
        rows = [{"date": "2024-11-27", "close_minute": 960},
                {"date": "2024-11-29", "close_minute": 780},
                {"date": "2024-12-02", "close_minute": 960}]
        signal, following = cli.session_pair(rows, datetime(2024, 11, 28, 12, tzinfo=ET))
        self.assertEqual(signal["date"], "2024-11-27")
        self.assertEqual(following, {"date": "2024-11-29", "close_minute": 780})
        signal, following = cli.session_pair(rows, datetime(2024, 11, 29, 13, 20, tzinfo=ET))
        self.assertEqual(signal["date"], "2024-11-29")
        self.assertEqual(following["date"], "2024-12-02")

    def test_dst_next_open_uses_new_york_not_fixed_utc_offset(self):
        rows = [{"date": "2024-03-08", "close_minute": 960},
                {"date": "2024-03-11", "close_minute": 960},
                {"date": "2024-03-12", "close_minute": 960}]
        signal, _ = cli.session_pair(rows, datetime(2024, 3, 11, 13, 29, tzinfo=timezone.utc))
        self.assertEqual(signal["date"], "2024-03-08")
        with self.assertRaises(cli.PreparationWindow):
            cli.session_pair(rows, datetime(2024, 3, 11, 13, 30, tzinfo=timezone.utc))

    def test_close_delay_never_replays_previous_signal(self):
        rows = [{"date": "2024-01-04", "close_minute": 960},
                {"date": "2024-01-05", "close_minute": 960},
                {"date": "2024-01-08", "close_minute": 960}]
        with self.assertRaises(cli.PreparationWindow):
            cli.session_pair(rows, datetime(2024, 1, 5, 16, 19, 59, tzinfo=ET))
        signal, _ = cli.session_pair(rows, datetime(2024, 1, 5, 16, 20, tzinfo=ET))
        self.assertEqual(signal["date"], "2024-01-05")

    def test_duplicate_unknown_close_and_naive_clock_fail(self):
        row = {"date": "2024-01-05", "close_minute": 960}
        for rows in ([row, row], [{**row, "close_minute": 600}], []):
            with self.assertRaises(PaperError):
                cli.session_pair(rows, datetime(2024, 1, 5, 17, tzinfo=ET))
        with self.assertRaises(PaperError):
            cli.session_pair([row], datetime(2024, 1, 5, 17))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = replace(load_settings(), data_root=self.root, use_gross_ladder=False)
        self.store = StateStore(cli.state_path(self.settings, 123))
        self.now = datetime(2024, 1, 5, 16, 20, tzinfo=ET)
        self.clock = lambda: self.now
        self.calendar = [{"date": "2024-01-04", "close_minute": 960},
                         {"date": "2024-01-05", "close_minute": 960},
                         {"date": "2024-01-08", "close_minute": 960},
                         {"date": "2024-01-09", "close_minute": 960}]
        self.snapshot = {"env": "SIMULATE", "acc_id": 123, "positions": {},
                         "orders": [], "cash": 1000., "nav": 1000.}
        self.broker = MagicMock()
        self.broker.calendar.return_value = self.calendar
        self.broker.snapshot.side_effect = lambda: deepcopy(self.snapshot)
        self.broker.raw_closes.return_value = {"US.A": 100.}
        self.refresher = MagicMock()
        self.controller = cli.PaperController(self.settings, 123, self.broker, self.store,
                                              clock=self.clock, refresher=self.refresher)
        self.write_inputs()
        self.score_patch = patch.object(cli, "build_score_table", side_effect=lambda factors, _: pd.DataFrame({
            "date": factors["date"], "code": factors["code"],
            "in_hold": factors["code"].eq("US.A"), "side": "long",
        }))
        self.score = self.score_patch.start()
        self.addCleanup(self.score_patch.stop)
        self.planner_patch = patch.object(cli, "build_plan", side_effect=self.fake_plan)
        self.planner = self.planner_patch.start()
        self.addCleanup(self.planner_patch.stop)

    def write_inputs(self):
        for subdirectory in ("meta", "panel", "factors"):
            (self.root / subdirectory).mkdir(exist_ok=True)
        pd.DataFrame({"code": ["US.A", "US.QQQ"]}).to_csv(self.root / "meta" / "universe.csv", index=False)
        frame = pd.DataFrame([
            {"date": pd.Timestamp(day), "code": code, "close": 100., "my_atr_pct_20": .02}
            for day in ("2024-01-04", "2024-01-05", "2024-01-08")
            for code in ("US.A", "US.QQQ")
        ])
        frame.to_parquet(self.root / "panel" / "daily.parquet", index=False)
        frame.to_parquet(self.root / "factors" / "daily.parquet", index=False)

    def fake_plan(self, settings, **kwargs):
        return {"env": "SIMULATE", "acc_id": kwargs["acc_id"], "security_firm": "FUTUSECURITIES",
                "signal_date": kwargs["signal_date"], "execution_date": kwargs["execution_date"],
                "close_minute": kwargs["close_minute"], "cost_bps": 10., "max_hold": 2,
                "positions": dict(kwargs["snapshot"]["positions"]), "legs": [],
                "effective_gross": .6, "reduction_targets": {},
                "strategy_config_hash": config_fingerprint(settings)}

    def saved(self, signal="2024-01-04", execution="2024-01-05", status="complete"):
        plan = self.fake_plan(self.settings, acc_id=123, signal_date=signal,
                              execution_date=execution, close_minute=960, snapshot=self.snapshot)
        state = install_plan(self.store, plan)
        state["status"] = status
        self.store.save(state)
        return state

    def test_prepare_slices_future_rows_and_queries_only_relevant_raw_codes(self):
        self.snapshot["positions"] = {"US.B": 2}
        state = self.controller.prepare()
        self.assertEqual(state["plan"]["signal_date"], "2024-01-05")
        self.assertEqual(state["plan"]["execution_date"], "2024-01-08")
        factors = self.score.call_args.args[0]
        self.assertEqual(factors["date"].max(), pd.Timestamp("2024-01-05"))
        self.broker.raw_closes.assert_called_once_with(["US.A", "US.B"], "2024-01-05")
        self.assertEqual(self.broker.snapshot.call_count, 2)
        self.broker.place.assert_not_called()
        self.broker.cancel.assert_not_called()

    def test_same_signal_is_idempotent_and_never_refreshes_again(self):
        first = self.controller.prepare(refresh=True)
        second = self.controller.prepare(refresh=True)
        self.assertEqual(first["plan_id"], second["plan_id"])
        self.refresher.assert_called_once()
        self.assertEqual(self.refresher.call_args.args[0].kline_end, "2024-01-05")

    def test_successful_refresh_marker_survives_planner_failure_and_restart(self):
        self.planner.side_effect = PaperError("temporary planner error")
        with self.assertRaises(PaperError):
            self.controller.prepare(refresh=True)
        self.planner.side_effect = self.fake_plan
        new = cli.PaperController(self.settings, 123, self.broker, self.store,
                                 clock=self.clock, refresher=self.refresher)
        new.prepare(refresh=True)
        self.refresher.assert_called_once()

    def test_unresolved_previous_plan_blocks_before_refresh(self):
        self.saved(status="working")
        with self.assertRaisesRegex(PaperError, "unresolved"):
            self.controller.prepare(refresh=True)
        self.refresher.assert_not_called()
        self.broker.snapshot.assert_not_called()

    def test_previous_risk_ceilings_are_passed_to_next_plan(self):
        old = self.saved()
        self.controller.prepare()
        self.assertEqual(self.planner.call_args.kwargs["previous_state"], old["plan"])

    def test_missing_universe_symbol_fails_even_when_benchmark_is_current(self):
        pd.DataFrame({"code": ["US.A", "US.B", "US.QQQ"]}).to_csv(
            self.root / "meta" / "universe.csv", index=False,
        )
        with self.assertRaisesRegex(PaperError, "missing symbols: US.B"):
            self.controller.prepare()
        self.assertIsNone(self.store.load())
        self.broker.raw_closes.assert_not_called()

    def test_failed_refresh_is_not_marked_successful(self):
        self.refresher.side_effect = RuntimeError("download failed")
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            self.controller.prepare(refresh=True)
        self.assertFalse(self.store.path.with_suffix(".refresh.json").exists())
        self.assertIsNone(self.store.load())

    def test_timed_out_refresh_does_not_mark_success_or_install_plan(self):
        self.controller.refresher = cli.refresh_inputs
        for failed_index in range(3):
            with self.subTest(failed_module=failed_index):
                outputs = [MagicMock(stdout="")] * failed_index
                outputs.append(subprocess.TimeoutExpired("daily refresh", 1800 if failed_index == 0 else 600))
                with patch.object(cli.subprocess, "run", side_effect=outputs) as run:
                    with self.assertRaisesRegex(PaperError, "timed out; refresh was not marked successful"):
                        self.controller.prepare(refresh=True)
                self.assertEqual(run.call_count, failed_index + 1)
                self.assertFalse(self.store.path.with_suffix(".refresh.json").exists())
                self.assertIsNone(self.store.load())
                self.planner.assert_not_called()

    def test_position_cash_or_nav_change_during_build_refuses_install(self):
        for change in ({"positions": {"US.A": 1}}, {"cash": 1100.}, {"nav": 1100.}):
            with self.subTest(change=change):
                self.broker.snapshot.side_effect = [deepcopy(self.snapshot), {**self.snapshot, **change}]
                with self.assertRaisesRegex(PaperError, "Account changed"):
                    self.controller.prepare()
                self.assertIsNone(self.store.load())

    def test_long_build_crossing_open_does_not_install_stale_plan(self):
        self.now = datetime(2024, 1, 8, 9, 29, tzinfo=ET)
        def build(settings, **kwargs):
            self.now = datetime(2024, 1, 8, 9, 30, tzinfo=ET)
            return self.fake_plan(settings, **kwargs)
        self.planner.side_effect = build
        with self.assertRaises(cli.PreparationWindow):
            self.controller.prepare()
        self.assertIsNone(self.store.load())

    def test_stop_requested_during_prepare_preserves_empty_ledger(self):
        def build(settings, **kwargs):
            self.store.stop_path.write_text("STOP", encoding="utf-8")
            return self.fake_plan(settings, **kwargs)
        self.planner.side_effect = build
        with self.assertRaisesRegex(PaperError, "STOP"):
            self.controller.prepare()
        self.assertIsNone(self.store.load())

    def test_new_worker_mid_session_waits_without_refresh_or_snapshot(self):
        self.now = datetime(2024, 1, 8, 10, tzinfo=ET)
        result = self.controller.run_once(submit=True, refresh=True)
        self.assertEqual(result["status"], "waiting")
        self.assertIn("no startup catch-up", result["last_message"])
        self.refresher.assert_not_called()
        self.broker.snapshot.assert_not_called()

    def test_tick_validates_configuration_and_calendar_before_broker_write(self):
        self.saved(signal="2024-01-05", execution="2024-01-08", status="prepared")
        self.controller.settings = replace(self.settings, cost_bps=self.settings.cost_bps + 1)
        with self.assertRaisesRegex(PaperError, "configuration changed"):
            self.controller.tick(submit=True)
        self.broker.snapshot.assert_not_called()
        self.controller.settings = self.settings
        self.broker.calendar.return_value = [
            row for row in self.calendar if row["date"] != "2024-01-08"
        ]
        with self.assertRaisesRegex(PaperError, "session changed"):
            self.controller.tick(submit=True)
        self.broker.snapshot.assert_not_called()

    def test_security_firm_cannot_change_for_saved_account_plan(self):
        self.saved(signal="2024-01-05", execution="2024-01-08", status="prepared")
        self.controller.security_firm = "FUTUINC"
        with self.assertRaisesRegex(PaperError, "security firm"):
            self.controller.tick(submit=True)
        self.broker.snapshot.assert_not_called()

    def test_stop_bypasses_strategy_hash_and_calendar_but_reconciles(self):
        self.saved(signal="2024-01-05", execution="2024-01-08", status="prepared")
        self.store.stop_path.write_text("STOP", encoding="utf-8")
        self.controller.settings = replace(self.settings, cost_bps=self.settings.cost_bps + 1)
        self.broker.calendar.side_effect = RuntimeError("calendar temporarily unavailable")
        result = self.controller.tick(submit=True)
        self.assertEqual(result["status"], "stopped")
        self.broker.calendar.assert_not_called()
        self.broker.snapshot.assert_called_once()
        self.broker.place.assert_not_called()
        self.broker.cancel.assert_not_called()

    def test_stop_still_rejects_wrong_security_firm_before_broker_access(self):
        self.saved(signal="2024-01-05", execution="2024-01-08", status="prepared")
        self.store.stop_path.write_text("STOP", encoding="utf-8")
        self.controller.security_firm = "FUTUINC"
        with self.assertRaisesRegex(PaperError, "security firm"):
            self.controller.tick(submit=True)
        self.broker.calendar.assert_not_called()
        self.broker.snapshot.assert_not_called()
        self.broker.cancel.assert_not_called()

    def test_calendar_cached_for_thirty_minutes_and_reset_each_day(self):
        self.controller.calendar(self.now)
        self.controller.calendar(self.now + cli.timedelta(minutes=29))
        self.assertEqual(self.broker.calendar.call_count, 1)
        self.controller.calendar(self.now + cli.timedelta(minutes=30))
        self.assertEqual(self.broker.calendar.call_count, 2)
        self.controller.calendar(self.now + cli.timedelta(days=1))
        self.assertEqual(self.broker.calendar.call_count, 3)

    def test_run_reconciles_existing_plan_before_new_preparation(self):
        self.saved()
        with patch.object(cli, "PaperEngine") as engine:
            engine.return_value.tick.return_value = {"status": "blocked", "last_message": "uncertain"}
            result = self.controller.run_once(submit=True, refresh=True)
            engine.assert_called_once_with(self.store, self.broker, clock=self.clock)
            engine.return_value.tick.assert_called_once_with(self.now, submit=True)
            self.assertEqual(result["status"], "blocked")
            self.refresher.assert_not_called()
            self.planner.assert_not_called()

    def test_refresh_runs_only_three_daily_modules_with_signal_day(self):
        with patch.object(cli.subprocess, "run") as run:
            run.return_value.stdout = "[ 1/2] OK US.A\n"
            cli.refresh_inputs(self.settings, "2024-01-05")
        calls = run.call_args_list
        self.assertEqual([c.args[0][2] for c in calls],
                         ["ndxbots.data.ingest", "ndxbots.factors.compute", "ndxbots.regime"])
        self.assertEqual(calls[0].args[0][-1], "--full-refresh")
        self.assertEqual([call.kwargs["timeout"] for call in calls], [1800, 600, 600])
        for call in calls:
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["env"]["KLINE_END"], "2024-01-05")
            self.assertEqual(call.kwargs["cwd"], cli.PROJECT_ROOT)
            self.assertEqual(call.kwargs["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def test_ingest_zero_exit_with_failed_download_does_not_run_factors(self):
        with patch.object(cli.subprocess, "run") as run:
            run.return_value.stdout = "[  2/102] FAIL US.AAPL rows=0 error\n"
            with self.assertRaisesRegex(PaperError, "failed downloads"):
                cli.refresh_inputs(self.settings, "2024-01-05")
            self.assertEqual(run.call_count, 1)

    def test_status_stop_resume_do_not_create_broker_or_modify_ledger(self):
        old = self.saved(status="blocked")
        with patch.object(cli, "load_settings", return_value=self.settings), \
                patch.object(cli, "IsolatedPaperBroker") as broker, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["stop", "--acc-id", "123"]), 0)
            self.assertTrue(self.store.stop_path.exists())
            self.assertEqual(cli.main(["status", "--acc-id", "123"]), 0)
            self.assertEqual(cli.main(["resume", "--acc-id", "123"]), 0)
            self.assertFalse(self.store.stop_path.exists())
            broker.assert_not_called()
        self.assertEqual(self.store.load(), old)


class CommandTests(unittest.TestCase):
    def test_preview_default_and_no_real_or_alternate_state_options(self):
        self.assertFalse(cli.parser().parse_args(["tick", "--acc-id", "123"]).submit)
        options = cli.parser().parse_args(["run", "--acc-id", "123"])
        self.assertFalse(options.submit)
        self.assertFalse(options.refresh)
        self.assertEqual(options.poll_seconds, 10)
        for args in (["tick", "--acc-id", "123", "--env", "REAL"],
                     ["tick", "--acc-id", "123", "--state", "other.sqlite3"],
                     ["run", "--acc-id", "123", "--poll-seconds", "4"],
                     ["run", "--acc-id", "123", "--poll-seconds", "nan"],
                     ["prepare"], ["tick", "--acc-id", "0"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(args)

    def test_worker_lock_is_independent_of_cwd_and_data_directory(self):
        first = cli.worker_lock_path("FUTUSECURITIES", 123)
        with patch.dict(cli.os.environ, {"DATA_DIR": "some-other-data"}):
            second = cli.worker_lock_path("FUTUSECURITIES", 123)
        self.assertEqual(first, second)
        self.assertNotEqual(first, cli.worker_lock_path("FUTUSECURITIES", 456))
        self.assertNotEqual(first, cli.worker_lock_path("FUTUINC", 123))

    def test_os_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "worker.lock"
            with cli.file_lock(path):
                with self.assertRaises(PaperError):
                    with cli.file_lock(path):
                        self.fail("second owner acquired worker lock")
            with cli.file_lock(path):
                pass

    def test_accounts_are_read_only_and_explicit_simulate(self):
        settings = load_settings()
        with patch.object(cli, "load_settings", return_value=settings), \
                patch.object(cli, "paper_accounts", return_value=[{"acc_id": 123}]) as accounts, \
                patch.object(cli, "IsolatedPaperBroker") as broker, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["accounts", "--security-firm", "FUTUINC"]), 0)
            accounts.assert_called_once_with(settings, security_firm="FUTUINC")
            self.assertEqual(json.loads(output.getvalue()),
                             {"env": "SIMULATE", "accounts": [{"acc_id": 123}]})
            broker.assert_not_called()

    def test_run_retries_read_failure_without_recreating_controller(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            settings = replace(load_settings(), data_root=Path(directory))
            with patch.object(cli, "load_settings", return_value=settings), \
                    patch.object(cli, "worker_lock_path", return_value=Path(directory) / "global.lock"), \
                    patch.object(cli, "IsolatedPaperBroker"), patch.object(cli, "PaperController") as factory, \
                    patch.object(cli.clock_time, "sleep", side_effect=[None, KeyboardInterrupt]), \
                    redirect_stdout(io.StringIO()) as output:
                factory.return_value.run_once.side_effect = [
                    RuntimeError("read disconnected"), {"status": "waiting"},
                ]
                self.assertEqual(cli.main(["run", "--acc-id", "123"]), 0)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(factory.return_value.run_once.call_count, 2)
                for call in factory.return_value.run_once.call_args_list:
                    self.assertEqual(call.kwargs, {"submit": False, "refresh": False})
                self.assertIn("read disconnected", output.getvalue())


if __name__ == "__main__":
    unittest.main()
