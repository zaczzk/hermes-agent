"""Opaque, durable identity for one canonical Hermes state database."""
from __future__ import annotations

import re

from hermes_state_errors import _STATE_DB_GENERATION_KEY

_GENERATION_RE = re.compile(r"[0-9a-f]{32}")
_STORE_ID_PREFIX = "hermes-state-v1:"


def get_store_id(db) -> str:
    """Return the stable identity minted by ``SessionDB`` for its current file generation.

    The generation changes when ``state.db`` is replaced or recreated.  Missing or malformed
    metadata fails closed; callers must not substitute a path, profile name, or session ID.
    """
    if db is None or not callable(getattr(db, "_read_one", None)):
        raise TypeError("a SessionDB handle is required")
    halt = getattr(db, "_halt_if_db_generation_changed", None)
    if not callable(halt):
        raise TypeError("a SessionDB handle is required")
    halt()
    row = db._read_one(
        "SELECT value FROM state_meta WHERE key = ?", (_STATE_DB_GENERATION_KEY,)
    )
    generation = str(row[0]) if row and row[0] is not None else ""
    if _GENERATION_RE.fullmatch(generation) is None:
        raise RuntimeError("canonical state store identity is unavailable")
    halt()
    return _STORE_ID_PREFIX + generation
