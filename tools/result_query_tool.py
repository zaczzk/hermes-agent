"""C13 (Gap A): ``result_query`` — the model-facing compact-verbs tool.

Design ref: tool-output-compaction-design.md L67-70. Registered like any
Hermes tool via ``tools.registry``. When a spilled result footer shows a
``result_id``, the agent can query it instead of re-running the command.

The output of this tool is itself budget-capped (design L72) and is never
re-spilled into the store — ``tools.result_cache.query`` guarantees both.
"""

from __future__ import annotations

import threading
from typing import Optional, Union

from tools.registry import registry
from tools.result_cache import RESULT_QUERY_BUDGET_CHARS, VERBS, ResultStore, query

__all__ = ["result_query_tool"]

_lock = threading.Lock()
_store: Optional[ResultStore] = None


def _get_store() -> ResultStore:
    """Lazily open the process-wide store (one connection, lock-guarded)."""
    global _store
    with _lock:
        if _store is None:
            _store = ResultStore()
        return _store


def _current_session_id() -> str:
    """Best-effort session scoping (C14), contextvars only.

    Resolution: approval observability contextvar, then the gateway
    session-context ContextVar. No os.environ fallback: the env var is a
    durable session id (CLI sessions) that never matches ingest-time
    scoping and would leak across test processes. No contextvar set (CLI,
    cron, tests) → returns "" → the store-global legacy path.
    """
    try:
        from tools.approval import _approval_session_id

        sid = _approval_session_id.get()
        if sid:
            return sid
    except Exception:
        pass
    try:
        from gateway.session_context import _SESSION_ID, _UNSET

        value = _SESSION_ID.get()
        if value is not _UNSET and value:
            return value
    except Exception:
        pass
    return ""


def result_query_tool(
    id: Union[int, str] = "last",
    verb: str = "summary",
    pattern: Optional[str] = None,
    n: Optional[int] = None,
    _lines_range: Optional[tuple] = None,
) -> str:
    """Query a stored tool result with a compact verb.

    Args:
        id: result id (from a spill footer) or ``"last"`` for the most
            recent result of this session (session-scoped, C14).
        verb: one of grep|errors|head|tail|count|lines|json|summary.
        pattern: for grep/count (substring or regex) or json (dot path).
        n: for head/tail (line count) or lines (1-based line number; pass
            ``_lines_range`` programmatically for a (start, end) window).
    """
    try:
        store = _get_store()
        n_arg = _lines_range if _lines_range is not None else n
        session = _current_session_id() or None
        # No session context (CLI/tests) → session_id=None preserves the
        # legacy store-global behavior exactly.
        return query(store, id, verb, pattern=pattern, n=n_arg,
                     session_id=session)
    except LookupError as exc:
        return f"result_query error: {exc}"
    except ValueError as exc:
        return f"result_query error: {exc}"


_RESULT_QUERY_SCHEMA = {
    "name": "result_query",
    "description": (
        "Query a previously spilled (oversized) tool result without re-running "
        "the command that produced it. Verbs: "
        "grep (pattern search), errors (error lines only), head, tail "
        "(first/last n lines), count (line counts), lines (line range), "
        "json (project a dot-path from JSON output), summary (extractive "
        "overview). id='last' targets the most recent stored result. Output "
        f"is capped at ~{RESULT_QUERY_BUDGET_CHARS} chars."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": ["integer", "string"],
                "description": "Result id from a spill footer, or 'last'.",
            },
            "verb": {
                "type": "string",
                "enum": list(VERBS),
                "description": "Compact verb to run over the stored result.",
            },
            "pattern": {
                "type": "string",
                "description": "grep/count: substring or regex. json: dot path like 'items.0.status'.",
            },
            "n": {
                "type": "integer",
                "description": "head/tail: number of lines (default 50). lines: start line (1-based).",
            },
        },
        "required": ["verb"],
    },
}


registry.register(
    name="result_query",
    toolset="terminal",
    schema=_RESULT_QUERY_SCHEMA,
    handler=lambda args, **kw: result_query_tool(
        id=args.get("id", "last"),
        verb=args.get("verb", "summary"),
        pattern=args.get("pattern"),
        n=args.get("n"),
    ),
    emoji="🔎",
)
