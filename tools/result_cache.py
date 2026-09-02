"""C13 (Gap B+A): standalone stdlib-only result cache + compact-verbs query.

Companion to the spillover layer in ``tools/tool_result_storage.py``: when a
tool result is too large for context it is spilled to disk with a preview —
this module makes those spilled results *addressable and queryable* instead of
write-only (design doc tool-output-compaction-design.md L48, L57, L62, L89).

Pieces:
- ``ResultStore``: SQLite at ``$HERMES_HOME/state/result_cache.sqlite``.
  Retention: keep last 200 results / 72h, vacuumed on open.
- ``compact(full_text, tool, budget_chars)`` — extractive head+tail
  compactor with a leading marker (PARTIAL-contract slot).
- ``query(store, result_id, verb, pattern, n)`` — compact verbs over a
  stored result: ``grep`` ``errors`` ``head`` ``tail`` ``count`` ``lines``
  ``json`` ``summary``. The output of any verb is itself budget-capped
  (design L72) and is never re-spilled into the store — it is already the
  filtered view.
- CLI entry: ``python -m tools.result_cache ingest <file>`` /
  ``... query <id|last> <verb> [pattern]`` so tests and non-model tooling
  can drive the same surface.

Stdlib-only; no spine contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Union

__all__ = [
    "RETENTION_MAX_RESULTS",
    "RETENTION_MAX_AGE_S",
    "RESULT_QUERY_BUDGET_CHARS",
    "VERBS",
    "ResultStore",
    "compact",
    "query",
]

RETENTION_MAX_RESULTS = 200
RETENTION_MAX_AGE_S = 72 * 3600
# Design L72: result_query output is itself budget-capped.
RESULT_QUERY_BUDGET_CHARS = 4000

VERBS = ("grep", "errors", "head", "tail", "count", "lines", "json", "summary")

_ERROR_RE = re.compile(
    r"error|traceback|fatal|exception|failed|failure|assertionerror|syntaxerror",
    re.IGNORECASE,
)


def _default_db_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "result_cache.sqlite"


class ResultStore:
    """SQLite-backed store of large tool results, addressable by id or 'last'."""

    # ------------------------------------------------------------------ write
    def ingest(
        self,
        full_text: str,
        tool: str,
        args_json: Optional[dict] = None,
        session_id: str = "",
        meta_json: Optional[dict] = None,
    ) -> int:
        """Persist a tool result; write the full text to a spill file.

        Returns the result id. The spill file lives in
        ``$HERMES_HOME/cache/spillover`` — the same canonical home the
        existing truncate+spill layer uses, so the same housekeeping prunes it.
        """
        from tools.tool_result_storage import get_spillover_dir

        now = int(time.time())
        digest = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        spill_dir = get_spillover_dir()
        spill_dir.mkdir(parents=True, exist_ok=True)
        fd, full_path = tempfile.mkstemp(
            prefix=f"result_{digest[:12]}_", suffix=".txt", dir=str(spill_dir)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(full_text)

        cur = self._conn.execute(
            "INSERT INTO results (session_id, tool, args_json, created_at,"
            " full_path, meta_json, size, sha256) VALUES (?,?,?,?,?,?,?,?)",
            (
                session_id,
                tool,
                json.dumps(args_json or {}),
                now,
                full_path,
                json.dumps(meta_json or {}),
                len(full_text.encode("utf-8")),
                digest,
            ),
        )
        self._conn.commit()
        result_id = cur.lastrowid
        self._vacuum_expired()
        return result_id

    _vacuum_interval_s = 300  # throttled re-check on the read path

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = str(db_path) if db_path else str(_default_db_path())
        self._last_vacuum_check = 0.0
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                tool TEXT,
                args_json TEXT,
                created_at INTEGER,
                full_path TEXT,
                meta_json TEXT,
                size INTEGER,
                sha256 TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_created ON results(created_at)"
        )
        self._conn.commit()
        self._vacuum_expired()

    # ------------------------------------------------------------------- read
    def get_record(self, result_id: Union[int, str]) -> Optional[dict]:
        self._vacuum_if_due()
        row = self._row(result_id)
        if row is None:
            return None
        keys = (
            "id", "session_id", "tool", "args_json", "created_at",
            "full_path", "meta_json", "size", "sha256",
        )
        return dict(zip(keys, row))

    def fetch(self, result_id: Union[int, str]) -> Optional[str]:
        """Return the full text of a result; ``'last'`` = most recent."""
        rec = self.get_record(result_id)
        if rec is None or not rec["full_path"]:
            return None
        try:
            with open(rec["full_path"], "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None

    # -------------------------------------------------------------- retention
    def _row(self, result_id: Union[int, str]):
        if result_id == "last":
            cur = self._conn.execute(
                "SELECT id, session_id, tool, args_json, created_at, full_path,"
                " meta_json, size, sha256 FROM results ORDER BY id DESC LIMIT 1"
            )
        else:
            try:
                rid_int = int(result_id)
            except (TypeError, ValueError):
                return None
            cur = self._conn.execute(
                "SELECT id, session_id, tool, args_json, created_at, full_path,"
                " meta_json, size, sha256 FROM results WHERE id = ?",
                (rid_int,),
            )
        return cur.fetchone()

    def _vacuum_if_due(self) -> None:
        """Re-check expiry at most every _vacuum_interval_s (read path)."""
        import time as _time

        now = _time.monotonic()
        if now - self._last_vacuum_check < self._vacuum_interval_s:
            return
        self._last_vacuum_check = now
        self._vacuum_expired()

    def _vacuum_expired(self) -> None:
        """Keep last 200 results / 72h; delete spills of vacuumed rows."""
        cutoff = int(time.time()) - RETENTION_MAX_AGE_S
        rows = self._conn.execute(
            "SELECT id, full_path FROM results WHERE id NOT IN"
            " (SELECT id FROM results ORDER BY id DESC LIMIT ?)"
            " OR created_at < ?",
            (RETENTION_MAX_RESULTS, cutoff),
        ).fetchall()
        if not rows:
            return
        for _rid, full_path in rows:
            try:
                if full_path and os.path.exists(full_path):
                    os.unlink(full_path)
            except OSError:
                pass
        self._conn.execute(
            "DELETE FROM results WHERE id NOT IN"
            " (SELECT id FROM results ORDER BY id DESC LIMIT ?)"
            " OR created_at < ?",
            (RETENTION_MAX_RESULTS, cutoff),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# compact() — extractive head+tail with a leading marker (design L60)
# --------------------------------------------------------------------------

def compact(full_text: str, budget_chars: int = RESULT_QUERY_BUDGET_CHARS) -> Tuple[str, bool]:
    """Cap *full_text* to *budget_chars*; head+tail with middle elision.

    Returns ``(text, truncated)``. The marker sits in the leading slot per
    the PARTIAL-contract evidence (design L28/L60).
    """
    if len(full_text) <= budget_chars:
        return full_text, False
    marker = (
        f"[COMPACTED — showing first/last of {len(full_text):,} chars."
        f" Use result_query (grep/errors/head/tail/lines) to search.]\n"
    )
    keep = max(budget_chars - len(marker) - 40, 0)
    head_len = keep // 2
    tail_len = keep - head_len
    head = full_text[:head_len]
    tail = full_text[-tail_len:] if tail_len else ""
    return f"{marker}{head}\n[...elided...]\n{tail}", True


def _cap(text: str) -> str:
    out, _trunc = compact(text, RESULT_QUERY_BUDGET_CHARS)
    return out


# --------------------------------------------------------------------------
# query() — the compact verbs (design L62)
# --------------------------------------------------------------------------

def _read(store: ResultStore, result_id: Union[int, str]) -> str:
    text = store.fetch(result_id)
    if text is None:
        raise LookupError(f"result {result_id!r} not found")
    return text


def query(
    store: ResultStore,
    result_id: Union[int, str],
    verb: str,
    pattern: Optional[str] = None,
    n: Optional[Union[int, Tuple[int, int]]] = None,
) -> str:
    """Run one compact verb over a stored result. Budget-capped output (L72)."""
    if verb not in VERBS:
        raise ValueError(
            f"unknown verb {verb!r}; expected one of {', '.join(VERBS)}"
        )
    text = _read(store, result_id)
    lines = text.splitlines()

    if verb == "grep":
        if not pattern:
            raise ValueError("grep requires pattern")
        import fnmatch
        rx = re.compile(fnmatch.translate(f"*{pattern}*") if not _looks_regex(pattern) else pattern)
        matches = [ln for ln in lines if rx.search(ln)]
        if not matches:
            return f"no match for {pattern!r}"
        return _cap("\n".join(matches))

    if verb == "errors":
        matches = [ln for ln in lines if _ERROR_RE.match(ln)]
        if not matches:
            return "no error lines"
        return _cap("\n".join(matches))

    if verb == "head":
        count = n if isinstance(n, int) and n > 0 else 50
        return _cap("\n".join(lines[:count]))

    if verb == "tail":
        count = n if isinstance(n, int) and n > 0 else 50
        return _cap("\n".join(lines[-count:]))

    if verb == "count":
        if pattern:
            import fnmatch
            rx = re.compile(fnmatch.translate(f"*{pattern}*") if not _looks_regex(pattern) else pattern)
            hits = sum(1 for ln in lines if rx.search(ln))
            return f"{hits} lines match {pattern!r} ({len(lines)} total)"
        return f"{len(lines)} lines"

    if verb == "lines":
        if isinstance(n, tuple):
            start, end = n
        elif isinstance(n, int):
            start, end = n, n + 1
        else:
            raise ValueError("lines requires a range, e.g. n=(2, 4)")
        return _cap("\n".join(lines[max(start - 1, 0):end]))

    if verb == "json":
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return "stored result is not valid JSON"
        if pattern:
            for part in pattern.split("."):
                if isinstance(data, list):
                    try:
                        data = data[int(part)]
                    except (ValueError, IndexError):
                        return f"json path {pattern!r} not found"
                elif isinstance(data, dict):
                    if part not in data:
                        return f"json path {pattern!r} not found"
                    data = data[part]
                else:
                    return f"json path {pattern!r} not found"
        return _cap(f"json: {json.dumps(data)}")

    # verb == "summary": extractive — counts + error/traceback lines
    n_errors = sum(1 for ln in lines if _ERROR_RE.match(ln))
    parts = [f"summary: {len(lines)} lines, {n_errors} error-ish"]
    if n_errors:
        err_lines = [ln for ln in lines if _ERROR_RE.match(ln)][:10]
        parts.append("\n".join(err_lines))
    parts.append("head: " + (lines[0] if lines else "(empty)"))
    parts.append("tail: " + (lines[-1] if lines else "(empty)"))
    return _cap("\n".join(parts))


def _looks_regex(pattern: str) -> bool:
    return bool(re.search(r"[\\^$.|?*+\[\]()]", pattern))


# --------------------------------------------------------------------------
# CLI (design L63: tests and non-model tooling drive the same surface)
# --------------------------------------------------------------------------

def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="result_cache: ingest/query tool results")
    p.add_argument("--db", default=None, help="sqlite path (default $HERMES_HOME/state/result_cache.sqlite)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("file", help="file whose contents become the result text ('-' = stdin)")
    pi.add_argument("--tool", default="terminal")
    pq = sub.add_parser("query")
    pq.add_argument("id", help="result id or 'last'")
    pq.add_argument("verb", choices=VERBS)
    pq.add_argument("pattern", nargs="?")
    pq.add_argument("--n", type=int, default=None)
    args = p.parse_args(argv)

    store = ResultStore(args.db)
    try:
        if args.cmd == "ingest":
            if args.file == "-":
                text = sys.stdin.read()
            else:
                with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            print(store.ingest(text, tool=args.tool))
            return 0
        print(query(store, args.id, args.verb, pattern=args.pattern, n=args.n))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(_main())
