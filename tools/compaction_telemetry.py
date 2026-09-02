"""Retention telemetry for tool-output compaction.

Measures the *retention* half of the truncate + spill pattern
(tools/tool_result_storage.py): when an oversized tool result is spilled to
disk, only a preview enters context — the full text lives on disk.  If the
model later re-reads that spill file, the compaction "held": the preview was
enough to work from until the full text was genuinely needed.  This module
records that event as a compact JSONL line so hit-rates can be measured over
time.

Design source: tool-output-compaction-design.md (2026-09-01) — Gap B notes
"spilled files are write-only"; the retention metric closes the measurement
half of that gap without changing compaction behavior itself (v1 scope:
"one module + one seam call").

Contract:
- ``record_persisted()`` registers a spill at the compaction point
  (maybe_persist_tool_result).
- ``record_retention_event()`` fires at the re-read seam (read_file_tool)
  when the path being read is a registered spill file.
- Append is best-effort and never raises — telemetry must never break a
  tool result or a read.
- ``compaction_stats()`` returns hit-rate totals over the JSONL log.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_RELATIVE_PATH = Path("state") / "compaction_retention.jsonl"

# Bound the in-memory registry: a long-lived gateway process could otherwise
# accumulate entries for every spill in the process lifetime.  Oldest entries
# are evicted; a spill evicted before re-read is simply unmeasured.
MAX_REGISTRY_ENTRIES = 512

_lock = threading.Lock()
# spill_path -> {"tool", "original_chars", "compacted_chars",
#                "session", "ts"}
_registry: dict[str, dict] = {}

# Session id for events recorded when record_persisted() had none threaded
# through (e.g. budget-enforcement spills where the caller has no session).
_current_session: str = ""


def get_jsonl_path() -> Path:
    """Return ``$HERMES_HOME/state/compaction_retention.jsonl``."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / STATE_RELATIVE_PATH


def set_current_session(session_id: str) -> None:
    """Record the active session id (used as a fallback for events)."""
    global _current_session
    with _lock:
        _current_session = str(session_id or "")


def _reset_for_tests() -> None:
    """Clear module state (tests only)."""
    global _current_session
    try:
        with _lock:
            _registry.clear()
            _current_session = ""
    except Exception:
        pass


def record_persisted(
    spill_path: str | None,
    tool: str,
    original_chars: int,
    compacted_chars: int,
    session: str | None = None,
) -> None:
    """Register a spill produced by tool-result compaction.

    Called at the compaction point. Best-effort: never raises.
    """
    if not spill_path:
        return
    try:
        with _lock:
            if len(_registry) >= MAX_REGISTRY_ENTRIES:
                _registry.pop(next(iter(_registry)), None)
            _registry[str(spill_path)] = {
                "tool": str(tool or "unknown"),
                "original_chars": int(original_chars),
                "compacted_chars": int(compacted_chars),
                "session": str(session or _current_session or ""),
                "ts": time.time(),
            }
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("compaction telemetry register failed: %s", exc)


def _append_event(event: dict) -> None:
    """Append one event line to the JSONL log. Best-effort, never raises."""
    try:
        path = get_jsonl_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.debug("compaction telemetry append failed: %s", exc)


def record_retention_event(spill_path: str | None) -> None:
    """Record that a compacted tool output was re-read from disk.

    Called at the re-read seam when the path being read matches a
    registered spill. Consumes the registration (one event per spill) and
    is best-effort: never raises, never blocks the read.
    """
    if not spill_path:
        return
    key = str(spill_path)
    try:
        with _lock:
            meta = _registry.pop(key, None)
        if meta is None:
            return
        event = {
            "ts": time.time(),
            "tool": meta.get("tool", "unknown"),
            "original_chars": meta.get("original_chars", 0),
            "compacted_chars": meta.get("compacted_chars", 0),
            "age_s": round(max(0.0, time.time() - meta.get("ts", time.time())), 3),
            "session": meta.get("session", ""),
        }
        _append_event(event)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("compaction telemetry event failed: %s", exc)


def compaction_stats() -> dict:
    """Return hit-rate totals over the retention JSONL log.

    Best-effort: a missing or unreadable log yields zeroed totals.
    """
    totals = {
        "retention_events": 0,
        "total_original_chars": 0,
        "total_compacted_chars": 0,
        "avg_compaction_ratio": 0.0,
    }
    try:
        path = get_jsonl_path()
        if not path.exists():
            return totals
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                totals["retention_events"] += 1
                totals["total_original_chars"] += int(
                    event.get("original_chars", 0) or 0
                )
                totals["total_compacted_chars"] += int(
                    event.get("compacted_chars", 0) or 0
                )
    except Exception as exc:
        logger.debug("compaction stats read failed: %s", exc)
        return totals

    original = totals["total_original_chars"]
    if original > 0:
        totals["avg_compaction_ratio"] = round(
            totals["total_compacted_chars"] / original, 6
        )
    return totals
