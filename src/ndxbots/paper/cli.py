"""A persistent, explicit-account SIMULATE worker; preview is the default."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, time, timedelta
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time as clock_time

import pandas as pd

from ndxbots.config import PROJECT_ROOT, load_settings
from ndxbots.strategy.scores import build_score_table
from .transport import IsolatedPaperBroker, paper_accounts
from .engine import ET, PaperEngine, TERMINAL, aware_et, install_plan
from .planner import build_plan, config_fingerprint
from .store import PaperError, StateStore
from .validation import validate_state


class PreparationWindow(PaperError):
    """A valid future execution plan cannot be created at this clock time."""


def _now():
    return datetime.now(ET)


@contextmanager
def file_lock(path: Path):
    """Use an OS-owned lock, released automatically when a process dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PaperError(f"Another paper process holds {path.name}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def state_path(settings, acc_id):
    return settings.data_root.resolve() / "paper" / f"{acc_id}.sqlite3"


def worker_lock_path(security_firm, acc_id):
    firm = str(security_firm).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", firm):
        raise PaperError("Invalid security firm")
    user = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
    user_key = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "ndxbots-paper-locks" / user_key / f"{firm}-{acc_id}.lock"


def _calendar_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise PaperError("Official US session calendar is empty")
    normalized = []
    for row in rows:
        try:
            day = datetime.strptime(row["date"], "%Y-%m-%d").date()
            close = row["close_minute"]
        except (KeyError, TypeError, ValueError):
            raise PaperError("Invalid official calendar row") from None
        if isinstance(close, bool) or close not in (780, 960):
            raise PaperError("Unsupported official session close")
        normalized.append({"date": day.isoformat(), "close_minute": close})
    normalized.sort(key=lambda row: row["date"])
    if len({row["date"] for row in normalized}) != len(normalized):
        raise PaperError("Duplicate official calendar dates")
    return normalized


def session_pair(rows, now):
    """Find the latest completed T and its actual next session, never weekdays."""
    now = aware_et(now)
    rows = _calendar_rows(rows)
    completed = []
    for index, row in enumerate(rows):
        day = datetime.strptime(row["date"], "%Y-%m-%d").date()
        ready = datetime.combine(day, time(), ET) + timedelta(minutes=row["close_minute"] + 20)
        if now >= ready:
            completed.append(index)
    if not completed:
        raise PreparationWindow("No official session has completed with the 20-minute data delay")
    index = completed[-1]
    if index + 1 >= len(rows):
        raise PaperError("Official calendar lacks the next session")
    signal, execution = rows[index], rows[index + 1]
    next_open = datetime.combine(
        datetime.strptime(execution["date"], "%Y-%m-%d").date(), time(9, 30), ET,
    )
    if now >= next_open:
        raise PreparationWindow(
            "Preparation window closed: no startup catch-up during this session; "
            "wait until the official close plus 20 minutes"
        )
    return signal, execution


def validate_plan_calendar(plan, rows):
    rows = _calendar_rows(rows)
    index = next((i for i, row in enumerate(rows) if row["date"] == plan["signal_date"]), None)
    if index is None or index + 1 >= len(rows):
        raise PaperError("Cannot verify the plan against the official calendar")
    following = rows[index + 1]
    if (following["date"] != plan["execution_date"]
            or following["close_minute"] != plan["close_minute"]):
        raise PaperError("Official execution session changed; review the saved plan")


def _read_inputs(settings, signal_date):
    """An ingest exit code of zero does not establish complete daily coverage."""
    try:
        universe = pd.read_csv(settings.data_root / "meta" / "universe.csv")
        panel = pd.read_parquet(settings.data_root / "panel" / "daily.parquet")
        factors = pd.read_parquet(settings.data_root / "factors" / "daily.parquet")
    except (OSError, ValueError) as exc:
        raise PaperError(f"Cannot load complete daily inputs: {exc}") from exc
    if "code" not in universe or universe.empty or universe["code"].isna().any():
        raise PaperError("Expected universe.csv with explicit nonempty stock codes")
    expected = set(universe["code"]) | {settings.benchmark}
    if not all(isinstance(code, str) and re.fullmatch(r"US\.[A-Z]+(?:[.-][A-Z]+)*", code)
               for code in expected):
        raise PaperError("Expected universe contains unsupported US stock symbols")
    day = pd.Timestamp(signal_date)
    sliced = []
    for label, frame in (("daily panel", panel), ("factors", factors)):
        if not {"date", "code"}.issubset(frame.columns):
            raise PaperError(f"{label} is missing date/code columns")
        frame = frame.copy()
        stamps = pd.to_datetime(frame["date"], errors="raise")
        if stamps.dt.tz is not None or stamps.isna().any():
            raise PaperError(f"{label} has invalid or timezone-aware dates")
        frame["date"] = stamps.dt.normalize()
        frame = frame.loc[frame["date"].le(day)].copy()
        if frame.duplicated(["date", "code"]).any():
            raise PaperError(f"{label} contains duplicate date/code rows")
        today = frame.loc[frame["date"].eq(day)]
        missing = sorted(expected - set(today["code"]))
        if missing:
            raise PaperError(f"Incomplete {label} on {signal_date}; missing symbols: {', '.join(missing)}")
        if label == "daily panel":
            if "close" not in today:
                raise PaperError("Daily panel is missing closes")
            closes = pd.to_numeric(today["close"], errors="coerce")
            invalid = today.loc[~closes.map(lambda value: math.isfinite(value) and value > 0), "code"]
            if not invalid.empty:
                raise PaperError(f"Invalid signal-day closes: {', '.join(invalid)}")
        sliced.append(frame)
    return sliced[0], sliced[1]


def refresh_inputs(settings, signal_date):
    """Refresh QFQ history as one unit; do not stitch incompatible split scales."""
    env = dict(os.environ)
    env.update(KLINE_END=signal_date, DATA_DIR=str(settings.data_root.resolve()),
               FUTU_HOST=str(settings.futu_host), FUTU_PORT=str(settings.futu_port))
    commands = [
        [sys.executable, "-m", "ndxbots.data.ingest", "--full-refresh"],
        [sys.executable, "-m", "ndxbots.factors.compute"],
        [sys.executable, "-m", "ndxbots.regime"],
    ]
    for command in commands:
        timeout = 1800 if command[2] == "ndxbots.data.ingest" else 600
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True,
                                    text=True, capture_output=True, timeout=timeout,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise PaperError(f"{command[2]} timed out; refresh was not marked successful") from exc
        if command[2] == "ndxbots.data.ingest" and re.search(
                r"(?m)^\[\s*\d+/\s*\d+\]\s+FAIL\b", result.stdout or ""):
            raise PaperError("Daily ingest reported failed downloads; no plan was created")
    _read_inputs(settings, signal_date)


class PaperController:
    """One testable orchestration step, with broker/calendar reads injectable."""

    def __init__(self, settings, acc_id, broker, store, *, clock=_now,
                 settings_loader=None, refresher=refresh_inputs,
                 security_firm="FUTUSECURITIES"):
        self.settings, self.acc_id, self.broker, self.store = settings, acc_id, broker, store
        self.clock, self.settings_loader, self.refresher = clock, settings_loader, refresher
        self.security_firm = security_firm.upper()
        self._calendar_cache, self._calendar_at = None, None

    def _settings(self):
        if self.settings_loader is not None:
            latest = self.settings_loader()
            if (state_path(latest, self.acc_id) != self.store.path.resolve()
                    or (latest.futu_host, latest.futu_port) !=
                    (self.settings.futu_host, self.settings.futu_port)):
                raise PaperError("State directory or OpenD endpoint changed; stop and review before restarting")
            self.settings = latest
        return self.settings

    def calendar(self, now):
        now = aware_et(now)
        if (self._calendar_cache is None or self._calendar_at is None
                or now.date() != self._calendar_at.date()
                or (now - self._calendar_at).total_seconds() >= 1800
                or now < self._calendar_at):
            start = now.date() - timedelta(days=30)
            old = self.store.load()
            if old:
                start = min(start, datetime.strptime(old["plan"]["signal_date"], "%Y-%m-%d").date())
            self._calendar_cache = _calendar_rows(self.broker.calendar(
                start.isoformat(), (now.date() + timedelta(days=14)).isoformat(),
            ))
            self._calendar_at = now
        return self._calendar_cache

    def _validate_current_plan(self, state, now, *, stopping=False):
        validate_state(state)
        plan = state["plan"]
        if plan.get("env") != "SIMULATE" or plan.get("acc_id") != self.acc_id:
            raise PaperError("Saved plan belongs to a different account/environment")
        if plan.get("security_firm") != self.security_firm:
            raise PaperError("Saved plan belongs to a different or unverified security firm")
        if stopping:
            return
        if plan.get("strategy_config_hash") != config_fingerprint(self.settings):
            raise PaperError("Strategy configuration changed; saved plan must not execute")
        validate_plan_calendar(plan, self.calendar(now))

    def prepare(self, *, refresh=False):
        settings = self._settings()
        if self.store.stop_path.exists():
            raise PaperError("STOP file is set; review status before resuming")
        # Distinct from the per-tick ledger lock: install_plan acquires that itself.
        with file_lock(settings.data_root.resolve() / "paper" / ".prepare.lock"):
            now = aware_et(self.clock())
            signal, execution = session_pair(self.calendar(now), now)
            old = self.store.load()
            if old:
                self._validate_current_plan(old, now)
                if old["plan"]["signal_date"] == signal["date"]:
                    return old
                if (old["status"] != "complete"
                        or any(order["status"] not in TERMINAL for order in old["orders"])):
                    raise PaperError("Previous plan is unresolved; no new plan or input refresh")
                if old["plan"]["signal_date"] >= signal["date"]:
                    raise PaperError("Refusing an old/replayed signal date")
            snapshot = self.broker.snapshot()
            signal_settings = replace(settings, kline_end=signal["date"])
            if refresh:
                marker_path = self.store.path.with_suffix(".refresh.json")
                marker = {}
                if marker_path.exists():
                    try:
                        marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        marker = {}
                expected = {"signal_date": signal["date"],
                            "config_hash": config_fingerprint(signal_settings)}
                if marker != expected:
                    self.refresher(signal_settings, signal["date"])
                    # Validate independently even when the refresher is substituted.
                    _read_inputs(signal_settings, signal["date"])
                    temporary = marker_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
                    os.replace(temporary, marker_path)
            panel, factors = _read_inputs(signal_settings, signal["date"])
            scored = build_score_table(factors, signal_settings)
            dates = pd.to_datetime(scored["date"]).dt.normalize()
            selected = scored.loc[dates.eq(pd.Timestamp(signal["date"])) & scored["in_hold"], "code"]
            codes = sorted(set(snapshot["positions"]) | set(selected))
            raw = self.broker.raw_closes(codes, signal["date"])
            plan = build_plan(
                signal_settings, acc_id=self.acc_id, signal_date=signal["date"],
                execution_date=execution["date"], close_minute=execution["close_minute"],
                snapshot=snapshot, raw_closes=raw, previous_state=old["plan"] if old else None,
                panel=panel, factors=factors, scored=scored,
            )
            plan["security_firm"] = self.security_firm
            # A long full refresh/build must not cross the execution open or a new signal.
            after = aware_et(self.clock())
            new_signal, new_execution = session_pair(self.calendar(after), after)
            if (new_signal, new_execution) != (signal, execution):
                raise PaperError("Session changed during preparation; no plan was installed")
            current = self.broker.snapshot()
            if (current.get("env") != "SIMULATE" or current.get("acc_id") != self.acc_id
                    or current["positions"] != snapshot["positions"]
                    or not math.isclose(float(current["cash"]), float(snapshot["cash"]),
                                        rel_tol=0, abs_tol=.005)
                    or not math.isclose(float(current["nav"]), float(snapshot["nav"]),
                                        rel_tol=0, abs_tol=.005)
                    or any(order["status"] not in TERMINAL for order in current["orders"])):
                raise PaperError("Account changed during preparation; no plan was installed")
            if self.store.stop_path.exists():
                raise PaperError("STOP requested during preparation; no plan was installed")
            if self.settings_loader and config_fingerprint(self.settings_loader()) != config_fingerprint(settings):
                raise PaperError("Strategy configuration changed during preparation")
            return install_plan(self.store, plan)

    def tick(self, *, submit=False):
        self._settings()
        now = aware_et(self.clock())
        state = self.store.load()
        if not state:
            raise PaperError("Prepare a paper plan first")
        self._validate_current_plan(state, now, stopping=self.store.stop_path.exists())
        return PaperEngine(self.store, self.broker, clock=self.clock).tick(now, submit=submit)

    def run_once(self, *, submit=False, refresh=False):
        """Reconcile the existing plan before considering a new signal."""
        self._settings()
        state = self.store.load()
        if state:
            result = self.tick(submit=submit)
            if result["status"] != "complete":
                return result
            # A preview must never make a nonterminal durable plan replaceable.
            if self.store.load()["status"] != "complete":
                return result
        if self.store.stop_path.exists():
            return {"status": "stopped", "last_message": "STOP file is set"}
        try:
            return self.prepare(refresh=refresh)
        except PreparationWindow as exc:
            return {"status": "waiting", "last_message": str(exc)}


def _account(value):
    if not re.fullmatch(r"[1-9][0-9]*", value) or int(value) > 2**53 - 1:
        raise argparse.ArgumentTypeError("acc-id must be an explicit positive exact integer")
    return int(value)


def _poll_seconds(value):
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("poll-seconds must be a number") from None
    if not math.isfinite(result) or result < 5:
        raise argparse.ArgumentTypeError("poll-seconds must be finite and at least 5")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    accounts = subcommands.add_parser("accounts", help="List US SIMULATE accounts; never select one automatically")
    accounts.add_argument("--security-firm", default="FUTUSECURITIES")
    for name in ("prepare", "tick", "run", "status", "stop", "resume"):
        command = subcommands.add_parser(name)
        command.add_argument("--acc-id", required=True, type=_account)
        command.add_argument("--security-firm", default="FUTUSECURITIES")
        if name in ("prepare", "run"):
            command.add_argument("--refresh", action="store_true",
                                 help="Refresh complete daily inputs once for each new signal")
        if name in ("tick", "run"):
            command.add_argument("--submit", action="store_true",
                                 help="Permit actual SIMULATE order/cancel requests (default: preview)")
        if name == "run":
            command.add_argument("--poll-seconds", type=_poll_seconds, default=10)
    return result


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str), flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = load_settings()
        if args.command == "accounts":
            accounts = paper_accounts(settings, security_firm=args.security_firm)
            emit({"env": "SIMULATE", "accounts": accounts})
            return 0
        store = StateStore(state_path(settings, args.acc_id))
        if args.command in {"status", "stop", "resume"}:
            if args.command == "stop":
                store.stop_path.write_text("STOP requested locally\n", encoding="utf-8")
            elif args.command == "resume":
                store.stop_path.unlink(missing_ok=True)
            emit({"acc_id": args.acc_id, "env": "SIMULATE",
                  "state_path": str(store.path), "stopped": store.stop_path.exists(),
                  "state": store.load()})
            return 0
        # Independent of checkout/DATA_DIR so two local clones cannot submit together.
        with file_lock(worker_lock_path(args.security_firm, args.acc_id)):
            broker = IsolatedPaperBroker(settings, args.acc_id, security_firm=args.security_firm)
            controller = PaperController(settings, args.acc_id, broker, store,
                                         settings_loader=load_settings,
                                         security_firm=args.security_firm)
            if args.command == "prepare":
                emit(controller.prepare(refresh=args.refresh))
                return 0
            if args.command == "tick":
                state = controller.tick(submit=args.submit)
                emit(state)
                return 2 if state["status"] == "blocked" else 0
            while True:
                try:
                    emit(controller.run_once(submit=args.submit, refresh=args.refresh))
                except Exception as exc:
                    # Read failures do not discard ledger intents or trigger resend.
                    emit({"status": "blocked", "error": type(exc).__name__, "last_message": str(exc)})
                clock_time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        emit({"status": "blocked", "error": type(exc).__name__, "last_message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
