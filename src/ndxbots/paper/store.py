"""Durable execution state and an OS lock shared by all local workers."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3


class PaperError(RuntimeError):
    pass


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_path = self.path.with_suffix(".STOP")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS archive (plan_id TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def locked(self):
        # A SQLite transaction cannot be kept open across an uncertain network
        # write. This separate OS lock covers the entire read/reconcile/write.
        with self.path.with_suffix(".lock").open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
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
                raise PaperError("Another paper worker owns this account state") from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load(self):
        with self._connect() as db:
            row = db.execute("SELECT value FROM state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else None

    def save(self, state):
        value = json.dumps(state, ensure_ascii=False, allow_nan=False, sort_keys=True)
        with self._connect() as db:
            db.execute("INSERT INTO state VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                       (value,))

    def archive(self, state):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO archive VALUES (?, ?)",
                       (state["plan_id"], json.dumps(state, ensure_ascii=False, allow_nan=False)))
