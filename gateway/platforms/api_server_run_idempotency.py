"""Durable idempotency reservations for API server runs."""

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from hermes_cli.sqlite_util import add_column_if_missing


# Keep the extracted store's log records on the API server logger.
logger = logging.getLogger("gateway.platforms.api_server")

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
EVIDENCE_ACK_RETENTION_SECONDS = 30 * 24 * 60 * 60
EVIDENCE_MAX_RETENTION_SECONDS = 90 * 24 * 60 * 60

_SELECT_BY_KEY = (
    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at "
    "FROM run_idempotency WHERE scope=? AND idempotency_key=?")
_EXTEND_RETENTION_BY_KEY = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND idempotency_key=? AND fingerprint=?")
_EXTEND_RETENTION_BY_RUN = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND run_id=?")
# Columns added after the first schema shipped; applied when missing.
_MIGRATIONS = {
    "owner_pid": "INTEGER NOT NULL DEFAULT 0",
    "owner_started": "INTEGER NOT NULL DEFAULT 0",
    "retention_until": "REAL NOT NULL DEFAULT 0",
    "acknowledged_at": "REAL",
    # Linked approved actions are durable single-execution facts.  Once reserved they must
    # survive ordinary terminal/ack retention, rather than becoming executable again.
    "durable_replay": "INTEGER NOT NULL DEFAULT 0"}


def _encode_status(status: Dict[str, Any]) -> str:
    return json.dumps(status, sort_keys=True, separators=(",", ":"))


def _record(run_id, status_json, owner_pid, owner_started, updated_at) -> dict[str, Any]:
    return {
        "run_id": run_id, "status": json.loads(status_json), "owner_pid": int(owner_pid or 0),
        "owner_started": int(owner_started or 0), "updated_at": float(updated_at or 0)}


def _outcome(row, fingerprint):
    """Classify a stored ``(scope, key)`` row against the caller's fingerprint."""
    return ("reused" if hmac.compare_digest(row[0], fingerprint) else "conflict"), _record(*row[1:])


class RunIdempotencyStore:
    """Durable, tenant-scoped reservations for ``POST /v1/runs``: a unique ``(scope, key)`` row
    inserted inside ``BEGIN IMMEDIATE`` so separate workers cannot both admit one request. Only
    fingerprints and public run status are stored — never request bodies or credentials."""

    RETENTION_SECONDS = 24 * 60 * 60
    ACKNOWLEDGED_RETENTION_SECONDS = 24 * 60 * 60

    @property
    def durable(self) -> bool:
        """Whether reservations survive this process."""
        return self._db_path is not None
    def __init__(self, db_path: str = None):
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "runs_idempotency.db")
            except Exception:
                db_path = ":memory:"
        self._db_path = None if db_path == ":memory:" else db_path
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        except Exception as exc:
            # Docker may create the container object before `docker run` fails to start it (e.g. exit code
            # 125 when the daemon isn't ready, or a timeout mid-pull). That orphan is left in "Created"
            # state — which the exited-only orphan reaper (reap_orphan_containers, status=exited) never
            # catches, so it leaks permanently. Remove it by its known name before re-raising. See #7439.
            logger.warning(
                "Run idempotency storage is unavailable; falling back to "
                "process memory, so replay will not survive a restart: %s", exc)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="runs_idempotency.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_idempotency (
                scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status_json TEXT NOT NULL,
                owner_pid INTEGER NOT NULL DEFAULT 0,
                owner_started INTEGER NOT NULL DEFAULT 0,
                retention_until REAL NOT NULL DEFAULT 0,
                acknowledged_at REAL,
                durable_replay INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            )"""
        )
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(run_idempotency)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in columns:
                add_column_if_missing(self._conn, "run_idempotency", column, f"{column} {ddl}")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS run_idempotency_run_id ON run_idempotency(run_id)")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_evidence_packages (
                scope TEXT NOT NULL,
                run_id TEXT NOT NULL,
                package_id TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                item_count INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                omitted_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                sealed_at REAL NOT NULL,
                acknowledged_at REAL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (scope, run_id)
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_evidence_items (
                scope TEXT NOT NULL,
                run_id TEXT NOT NULL,
                package_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                content BLOB NOT NULL,
                text_utf8 INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                PRIMARY KEY (scope, run_id, item_id),
                UNIQUE (scope, run_id, ordinal)
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_evidence_tombstones (
                scope TEXT NOT NULL,
                run_id TEXT NOT NULL,
                package_id TEXT NOT NULL,
                expired_at REAL NOT NULL,
                PRIMARY KEY (scope, run_id)
            )"""
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self._tighten_permissions()

    def _tighten_permissions(self) -> None:
        for suffix in ("", "-wal", "-shm") if self._db_path else ():
            candidate = Path(self._db_path + suffix)
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict run idempotency store permissions", exc_info=True)

    @contextmanager
    def _immediate_txn(self):
        """Hold the lock inside ``BEGIN IMMEDIATE``; the body commits, errors roll back."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._conn.rollback()
                raise

    def reserve(self, scope: str, key: str, fingerprint: str, run_id: str, status: Dict[str, Any], *,
                owner_pid: int = 0, owner_started: int = 0, retention_until: float = 0,
                durable_replay: bool = False):
        """Atomically reserve a key; return ``(outcome, stored_record)``."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = _encode_status(status)
        with self._immediate_txn():
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            if row is not None:
                if retention_until:
                    self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
                if durable_replay and hmac.compare_digest(row[0], fingerprint):
                    self._conn.execute(
                        "UPDATE run_idempotency SET durable_replay=1 "
                        "WHERE scope=? AND idempotency_key=?",
                        (scope, key))
                self._conn.commit()
                return _outcome(row, fingerprint)
            self._conn.execute(
                "INSERT INTO run_idempotency("
                "scope,idempotency_key,fingerprint,run_id,status_json,"
                "owner_pid,owner_started,retention_until,durable_replay,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0), int(owner_started or 0),
                 retention_until, int(bool(durable_replay)), now, now))
            self._conn.commit()
            return "created", _record(run_id, encoded, owner_pid, owner_started, now) | {"status": status}

    def lookup(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._immediate_txn():
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            self._conn.commit()
        return ("missing", None) if row is None else _outcome(row, fingerprint)

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal (caller holds the
        lock + transaction): a long or disconnected room turn may outlive the retention window."""
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, status_json
                FROM run_idempotency
                WHERE durable_replay=0 AND (
                       acknowledged_at <= ?
                    OR (retention_until > 0 AND retention_until <= ?)
                    OR (retention_until <= 0 AND updated_at < ?))""",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS),
        ).fetchall()
        for stale_scope, stale_key, stale_status in stale:
            try:
                terminal = json.loads(stale_status).get("status") in TERMINAL_STATUSES
            except Exception:
                terminal = False
            if terminal:
                self._conn.execute(
                    "DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (stale_scope, stale_key))

    def status_for_run(self, scope: str, run_id: str, *, retention_until: float = 0) -> dict[str, Any] | None:
        """Load one durable run status inside its authenticated scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_RUN, (retention_until, scope, run_id))
                self._conn.commit()
            row = self._conn.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at "
                "FROM run_idempotency WHERE scope=? AND run_id=?",
                (scope, run_id)).fetchone()
        if row is None:
            return None
        return {k: v for k, v in _record(None, *row).items() if k != "run_id"}

    def extend_retention(self, scope: str, run_id: str, until: float) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._lock:
            changed = self._conn.execute(_EXTEND_RETENTION_BY_RUN, (checked_until, scope, run_id)).rowcount
            self._conn.commit()
        return changed == 1

    def owns_run(self, scope: str, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?", (scope, run_id)).fetchone()
        return row is not None

    def update_status(self, run_id: str, status: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (_encode_status(status), time.time(), run_id))
            self._conn.commit()

    @staticmethod
    def _evidence_descriptor(row) -> dict[str, Any]:
        return {
            "package_id": str(row[0]),
            "state": str(row[1]),
            "manifest_sha256": str(row[2]),
            "item_count": int(row[3]),
            "total_bytes": int(row[4]),
            "omitted_count": int(row[5]),
        }

    def _prune_evidence_locked(self, now: float) -> None:
        expired = self._conn.execute(
            "SELECT scope,run_id,package_id FROM run_evidence_packages WHERE expires_at<=?",
            (now,),
        ).fetchall()
        for scope, run_id, package_id in expired:
            self._conn.execute(
                "INSERT OR IGNORE INTO run_evidence_tombstones(scope,run_id,package_id,expired_at) "
                "VALUES(?,?,?,?)",
                (scope, run_id, package_id, now),
            )
            self._conn.execute(
                "DELETE FROM run_evidence_items WHERE scope=? AND run_id=?",
                (scope, run_id),
            )
            self._conn.execute(
                "DELETE FROM run_evidence_packages WHERE scope=? AND run_id=?",
                (scope, run_id),
            )

    def seal_evidence(
        self,
        scope: str,
        run_id: str,
        *,
        state: str,
        items: list[dict[str, Any]],
        omitted_count: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Atomically persist one immutable owner/run evidence package."""
        if state not in {"complete", "partial", "unavailable"}:
            raise ValueError("invalid evidence state")
        if type(omitted_count) is not int or omitted_count < 0 or len(items) > 8:
            raise ValueError("invalid evidence counts")
        now = time.time() if now is None else float(now)
        with self._immediate_txn():
            self._prune_evidence_locked(now)
            existing = self._conn.execute(
                "SELECT package_id,state,manifest_sha256,item_count,total_bytes,omitted_count "
                "FROM run_evidence_packages WHERE scope=? AND run_id=?",
                (scope, run_id),
            ).fetchone()
            if existing is not None:
                self._conn.commit()
                return self._evidence_descriptor(existing)
            run = self._conn.execute(
                "SELECT created_at FROM run_idempotency WHERE scope=? AND run_id=?",
                (scope, run_id),
            ).fetchone()
            if run is None:
                raise ValueError("run evidence owner is unavailable")
            package_id = "evidence_" + secrets.token_hex(12)
            public_items: list[dict[str, Any]] = []
            total_bytes = 0
            for ordinal, item in enumerate(items, 1):
                item_id = item.get("item_id")
                content = item.get("bytes")
                size = item.get("size")
                sha256 = item.get("sha256")
                text_utf8 = item.get("text_utf8")
                if (
                    not isinstance(item_id, str)
                    or not item_id.startswith("item_")
                    or len(item_id) != 29
                    or any(char not in "0123456789abcdef" for char in item_id[5:])
                    or not isinstance(content, bytes)
                    or type(size) is not int
                    or size != len(content)
                    or size > 1024 * 1024
                    or not isinstance(sha256, str)
                    or hashlib.sha256(content).hexdigest() != sha256
                    or type(text_utf8) is not bool
                ):
                    raise ValueError("invalid evidence item")
                total_bytes += size
                if total_bytes > 4 * 1024 * 1024:
                    raise ValueError("evidence package too large")
                display_name = f"file-{ordinal:02d}.{'txt' if text_utf8 else 'bin'}"
                public_items.append({
                    "item_id": item_id,
                    "ordinal": ordinal,
                    "size": size,
                    "sha256": sha256,
                    "kind": "file",
                    "display_name": display_name,
                    "text_utf8": text_utf8,
                })
            manifest_source = {
                "state": state,
                "item_count": len(public_items),
                "total_bytes": total_bytes,
                "omitted_count": omitted_count,
                "items": public_items,
            }
            manifest_sha256 = hashlib.sha256(json.dumps(
                manifest_source, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            created_at = float(run[0])
            expires_at = created_at + EVIDENCE_MAX_RETENTION_SECONDS
            self._conn.execute(
                "INSERT INTO run_evidence_packages VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope, run_id, package_id, state, manifest_sha256,
                    len(public_items), total_bytes, omitted_count,
                    created_at, now, None, expires_at,
                ),
            )
            for item, public in zip(items, public_items):
                self._conn.execute(
                    "INSERT INTO run_evidence_items VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        scope, run_id, package_id, public["item_id"], public["ordinal"],
                        public["size"], public["sha256"], item["bytes"],
                        int(public["text_utf8"]), public["display_name"],
                    ),
                )
            self._conn.commit()
            return {
                "package_id": package_id,
                "state": state,
                "manifest_sha256": manifest_sha256,
                "item_count": len(public_items),
                "total_bytes": total_bytes,
                "omitted_count": omitted_count,
            }

    def evidence_tombstone(self, scope: str, run_id: str, package_id: str | None = None) -> bool:
        with self._immediate_txn():
            self._prune_evidence_locked(time.time())
            query = "SELECT package_id FROM run_evidence_tombstones WHERE scope=? AND run_id=?"
            row = self._conn.execute(query, (scope, run_id)).fetchone()
            self._conn.commit()
        return row is not None and (package_id is None or str(row[0]) == package_id)

    def evidence_manifest(self, scope: str, run_id: str) -> dict[str, Any] | None:
        with self._immediate_txn():
            self._prune_evidence_locked(time.time())
            package = self._conn.execute(
                "SELECT package_id,state,manifest_sha256,item_count,total_bytes,omitted_count "
                "FROM run_evidence_packages WHERE scope=? AND run_id=?",
                (scope, run_id),
            ).fetchone()
            if package is None:
                self._conn.commit()
                return None
            rows = self._conn.execute(
                "SELECT item_id,ordinal,size,sha256,text_utf8,display_name "
                "FROM run_evidence_items WHERE scope=? AND run_id=? ORDER BY ordinal",
                (scope, run_id),
            ).fetchall()
            self._conn.commit()
        return {
            "object": "hermes.run.evidence",
            "run_id": run_id,
            **self._evidence_descriptor(package),
            "items": [
                {
                    "item_id": str(row[0]), "ordinal": int(row[1]),
                    "size": int(row[2]), "sha256": str(row[3]), "kind": "file",
                    "display_name": str(row[5]), "text_utf8": bool(row[4]),
                }
                for row in rows
            ],
        }

    def evidence_item(
        self, scope: str, run_id: str, package_id: str, item_id: str
    ) -> dict[str, Any] | None:
        with self._immediate_txn():
            self._prune_evidence_locked(time.time())
            row = self._conn.execute(
                "SELECT content,size,sha256,text_utf8,display_name FROM run_evidence_items "
                "WHERE scope=? AND run_id=? AND package_id=? AND item_id=?",
                (scope, run_id, package_id, item_id),
            ).fetchone()
            self._conn.commit()
        if row is None:
            return None
        return {
            "bytes": bytes(row[0]), "size": int(row[1]), "sha256": str(row[2]),
            "text_utf8": bool(row[3]), "display_name": str(row[4]),
        }

    def acknowledge_evidence(
        self, scope: str, run_id: str, package_id: str, manifest_sha256: str,
        *, now: float | None = None,
    ) -> dict[str, Any] | None:
        now = time.time() if now is None else float(now)
        with self._immediate_txn():
            self._prune_evidence_locked(now)
            row = self._conn.execute(
                "SELECT manifest_sha256,created_at,acknowledged_at FROM run_evidence_packages "
                "WHERE scope=? AND run_id=? AND package_id=?",
                (scope, run_id, package_id),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            if not hmac.compare_digest(str(row[0]), manifest_sha256):
                raise ValueError("evidence manifest conflict")
            acknowledged_at = float(row[2]) if row[2] is not None else now
            expires_at = min(
                acknowledged_at + EVIDENCE_ACK_RETENTION_SECONDS,
                float(row[1]) + EVIDENCE_MAX_RETENTION_SECONDS,
            )
            self._conn.execute(
                "UPDATE run_evidence_packages SET acknowledged_at=?,expires_at=? "
                "WHERE scope=? AND run_id=? AND package_id=?",
                (acknowledged_at, expires_at, scope, run_id, package_id),
            )
            self._conn.commit()
        return {"acknowledged": True, "expires_at": expires_at}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
