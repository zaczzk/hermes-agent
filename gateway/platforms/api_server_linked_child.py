"""Strict linked-child admission layered onto the existing ``/v1/runs`` executor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass

from gateway.platforms.api_server_room_grants import _json_error
from hermes_state_linked_child import LinkedChildAdmission, LinkedChildConflictError
from hermes_state_passive_history import (
    PassiveHistoryBusyError,
    PassiveHistoryConflictError,
    PassiveHistoryRetiredError,
    PassiveHistoryTargetError,
)
from passive_history_ingress import PRODUCER

_TOP_LEVEL_KEYS = frozenset({"input", "session_id", "origin", "child"})
_ORIGIN_KEYS = frozenset({"event_id", "origin_turn_id"})
_CHILD_KEYS = frozenset({"goal", "context", "correlation_id"})


@dataclass(frozen=True, slots=True)
class PreparedLinkedChild:
    admission: LinkedChildAdmission
    idempotency_scope: str
    goal: str
    labelled_context: str


def has_linked_child_fields(body) -> bool:
    return isinstance(body, dict) and bool({"origin", "child"}.intersection(body))


def capabilities(adapter, db) -> dict | None:
    """Advertise only when both durable authorities are actually available."""
    if db is None or not getattr(adapter._run_idempotency_store, "durable", False):
        return None
    from hermes_state_store_identity import get_store_id

    get_store_id(db)
    return {
        "version": 1,
        "supported": True,
        "separate_child_goal": True,
        "origin_sources": ["fresh", "passive_receipt"],
    }


def _invalid(_openai_error, message: str, *, code: str = "invalid_linked_child_request"):
    return _json_error(_openai_error, message, code=code, status=400)


async def prepare(
    adapter, request, body, *, idempotency_key: str, profile: str, _openai_error,
    gateway_session_key: str,
) -> tuple[PreparedLinkedChild | None, object | None]:
    """Validate and durably bind a linked child before generic run parsing can act."""
    if not isinstance(body, dict) or set(body) != _TOP_LEVEL_KEYS:
        return None, _invalid(
            _openai_error,
            "Linked-child runs require exactly input, session_id, origin, and child",
        )
    if not idempotency_key:
        return None, _invalid(
            _openai_error,
            "Linked-child runs require an Idempotency-Key",
            code="missing_idempotency_key",
        )
    origin, child = body.get("origin"), body.get("child")
    if not isinstance(origin, dict) or set(origin) not in (
        _ORIGIN_KEYS,
        _ORIGIN_KEYS | {"receipt_id"},
    ):
        return None, _invalid(_openai_error, "Linked-child origin is invalid")
    if not isinstance(child, dict) or set(child) != _CHILD_KEYS:
        return None, _invalid(
            _openai_error,
            "Linked-child goal, context, and correlation_id are required; workers and attachments "
            "are unsupported",
        )
    if not getattr(adapter._run_idempotency_store, "durable", False):
        return None, _json_error(
            _openai_error,
            "Linked-child dispatch requires durable run idempotency",
            code="child_dispatch_unsupported",
            status=503,
        )
    db = await adapter._ensure_session_db_async()
    if db is None:
        return None, _json_error(
            _openai_error, "Canonical session store is unavailable",
            code="store_unavailable", status=503)
    try:
        from hermes_state_store_identity import get_store_id

        store_id = await asyncio.to_thread(get_store_id, db)
        base_scope = adapter._run_idempotency_scope(request)
        authority_scope = hashlib.sha256(json.dumps(
            ["linked-child-v1", base_scope, profile, store_id],
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
        admission = await asyncio.to_thread(
            db.admit_linked_child,
            body["session_id"],
            authority_scope=authority_scope,
            idempotency_key_hash=key_hash,
            producer=PRODUCER,
            event_id=origin.get("event_id"),
            origin_turn_id=origin.get("origin_turn_id"),
            receipt_revision=origin.get("receipt_id"),
            input_text=body["input"],
            goal=child.get("goal"),
            context=child.get("context"),
            correlation_id=child.get("correlation_id"),
            gateway_session_key=gateway_session_key,
        )
    except PassiveHistoryBusyError:
        return None, _json_error(
            _openai_error, "Parent conversation is busy", code="parent_busy", status=409)
    except PassiveHistoryRetiredError:
        return None, _json_error(
            _openai_error, "Parent input receipt is retired", code="origin_retired", status=410)
    except (LinkedChildConflictError, PassiveHistoryConflictError):
        return None, _json_error(
            _openai_error, "Linked-child request conflicts with its canonical receipt",
            code="linked_child_conflict", status=409)
    except PassiveHistoryTargetError:
        return None, _json_error(
            _openai_error, "Parent conversation is unavailable",
            code="target_unavailable", status=409)
    except (ValueError, TypeError, UnicodeError):
        return None, _invalid(_openai_error, "Linked-child request is invalid")
    except (RuntimeError, sqlite3.Error):
        return None, _json_error(
            _openai_error, "Canonical session store is unavailable",
            code="store_unavailable", status=503)

    context = child["context"]
    labelled = "Linked child context (not parent transcript):\n" + context
    return PreparedLinkedChild(
        admission=admission,
        # The runs store keeps its existing credential/profile scope so GET/status can recover
        # after restart without a request body.  The fingerprint and canonical admission above
        # remain store-bound, so a replacement store conflicts rather than adopting old work.
        idempotency_scope=base_scope,
        goal=child["goal"],
        labelled_context=labelled,
    ), None
