"""Authenticated hosts' shared passive-history service; no execution dependencies.

Hosts own authentication and profile-scoped DB handles. Attachment authority is scoped to a
host epoch and principal; durable event receipts remain readable after transport authority expires.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import asdict

from hermes_state_passive_history import (
    PassiveHistoryBusyError, PassiveHistoryConflictError, PassiveHistoryRetiredError,
    PassiveHistoryTargetError, _validated_identifier,
)

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 160 * 1024
MAX_SNAPSHOT_BYTES = 32 * 1024
MAX_SNAPSHOT_MESSAGES = 20
PRODUCER = "passive.ingress.v1"
_ATTACHMENT_KEYS = {"tab_id", "attachment_id", "generation", "session_id"}
_BODY_KEYS = {
    "attach": {"tab_id", "session_id"},
    "snapshot": _ATTACHMENT_KEYS,
    "commit": _ATTACHMENT_KEYS | {"event_id", "origin_turn_id", "messages"},
    "reconcile": {"session_id", "event_id"},
    "detach": _ATTACHMENT_KEYS,
}


class IngressError(ValueError):
    def __init__(self, code: str, status: int = 409):
        super().__init__(code)
        self.code, self.status = code, status


def principal_key(identity: str) -> str:
    """Only an irreversible principal key goes into the attachment table."""
    return hashlib.sha256(identity.encode()).hexdigest()


def capabilities() -> dict:
    return {
        "version": PROTOCOL_VERSION, "passive_only": True, "origin_adoption": False,
        "operations": ["attach", "snapshot", "commit", "reconcile", "detach"],
        "max_request_bytes": MAX_REQUEST_BYTES, "max_message_bytes": 64 * 1024,
        "max_messages": 2, "max_snapshot_messages": MAX_SNAPSHOT_MESSAGES,
        "max_snapshot_bytes": MAX_SNAPSHOT_BYTES,
        "restart_requires_reattach": True,
    }


def error_response(exc: Exception) -> tuple[dict, int]:
    """Shared, content-free diagnostics; never render database paths or rejected text."""
    if isinstance(exc, IngressError):
        return {"error": exc.code, "retryable": False}, exc.status
    if isinstance(exc, PassiveHistoryBusyError):
        return {"error": "busy", "retryable": True}, 409
    if isinstance(exc, PassiveHistoryRetiredError):
        return {"error": "retired", "retryable": False}, 410
    if isinstance(exc, PassiveHistoryConflictError):
        return {"error": "event_conflict", "retryable": False}, 409
    if isinstance(exc, PassiveHistoryTargetError):
        return {"error": "target_unavailable", "retryable": False}, 409
    if isinstance(exc, (ValueError, TypeError)):
        return {"error": "invalid_request", "retryable": False}, 400
    if isinstance(exc, sqlite3.Error):
        return {"error": "store_unavailable", "retryable": True}, 503
    raise exc


class PassiveHistoryIngress:
    def __init__(self):
        self.epoch = uuid.uuid4().hex

    def dispatch(self, db, *, profile: str, principal: str, operation: str, body: dict) -> dict:
        if operation not in _BODY_KEYS:
            raise IngressError("unsupported_operation", 404)
        required = _BODY_KEYS[operation]
        if not isinstance(body, dict) or set(body) != required:
            raise IngressError("invalid_request", 400)
        if not profile or not principal:
            raise IngressError("denied", 403)
        if not isinstance(body["session_id"], str) or not 1 <= len(body["session_id"]) <= 256:
            raise IngressError("invalid_session_id", 400)
        for key in required - {"messages", "generation", "session_id"}:
            _validated_identifier(body[key], key, 128)
        if "generation" in required and (
            type(body["generation"]) is not int or body["generation"] < 1
        ):
            raise IngressError("invalid_generation", 400)
        scope = (profile, principal_key(principal), body.get("tab_id"))
        handler = getattr(self, f"_{operation}")
        return handler(db, scope, body)

    def _check_attachment(self, db, conn, scope, body):
        row = conn.execute(
            "SELECT * FROM passive_history_attachments "
            "WHERE profile_id=? AND principal_id=? AND tab_id=?", scope).fetchone()
        if row is None or any((
            row["attachment_id"] != body["attachment_id"],
            row["generation"] != body["generation"], row["host_epoch"] != self.epoch,
            row["session_id"] != body["session_id"],
        )):
            raise IngressError("stale_attachment")
        owner = db._passive_conversation_id(conn, body["session_id"])
        if owner != row["conversation_id"]:
            raise IngressError("stale_attachment")
        return row

    def _snapshot_data(self, db, conn, session_id, conversation_id):
        from agent.compaction_display import project_compaction_message_for_display
        from agent.prompt_builder import STEER_DISPLAY_KIND
        from hermes_state_passive_history import PASSIVE_HISTORY_DISPLAY_KIND

        tip = db._resolve_passive_history_tip(
            conn, conversation_id, requested_session_id=session_id)
        rows = conn.execute(
            "SELECT id, role, content, display_kind, _compressed_summary FROM messages "
            "WHERE session_id=? AND active=1 AND role IN ('user','assistant') "
            "ORDER BY id DESC LIMIT ?",
            (tip, MAX_SNAPSHOT_MESSAGES + 1)).fetchall()
        truncated = len(rows) > MAX_SNAPSHOT_MESSAGES
        messages, remaining = [], MAX_SNAPSHOT_BYTES
        for row in rows[:MAX_SNAPSHOT_MESSAGES]:
            message = project_compaction_message_for_display(
                {**dict(row), "content": db._decode_content(row["content"])})
            if message is None or message.get("display_kind") not in (
                None, "", STEER_DISPLAY_KIND, PASSIVE_HISTORY_DISPLAY_KIND,
            ):
                truncated = True
                continue
            content = message.get("content")
            if not isinstance(content, str):
                truncated = True
                continue
            encoded = content.encode("utf-8")
            clipped = len(encoded) > remaining
            if clipped:
                encoded = encoded[:remaining]
                truncated = True
            text = encoded.decode("utf-8", errors="ignore")
            if text:
                messages.append({"id": row["id"], "role": row["role"], "content": text})
            else:
                truncated = True
            remaining -= len(text.encode("utf-8"))
            if clipped or remaining <= 0:
                truncated = True
                break
        return {
            "conversation_id": conversation_id, "session_id": tip,
            "messages": list(reversed(messages)), "truncated": truncated,
            "capabilities": capabilities(),
        }

    def _attach(self, db, scope, body):
        def write(conn):
            if db._passive_lineage_row(conn, body["session_id"]) is None:
                raise IngressError("target_missing", 404)
            owner = db._passive_conversation_id(conn, body["session_id"])
            snapshot = self._snapshot_data(db, conn, body["session_id"], owner)
            attachment_id = uuid.uuid4().hex
            row = conn.execute(
                "INSERT INTO passive_history_attachments "
                "(profile_id,principal_id,tab_id,attachment_id,generation,host_epoch,"
                "session_id,conversation_id,snapshot_session_id) VALUES (?,?,?,?,1,?,?,?,?) "
                "ON CONFLICT(profile_id,principal_id,tab_id) DO UPDATE SET "
                "attachment_id=excluded.attachment_id,generation=generation+1,"
                "host_epoch=excluded.host_epoch,session_id=excluded.session_id,"
                "conversation_id=excluded.conversation_id,snapshot_session_id=excluded.snapshot_session_id "
                "RETURNING generation",
                (*scope, attachment_id, self.epoch, body["session_id"], owner, snapshot["session_id"]),
            ).fetchone()
            return {"profile": scope[0], "tab_id": body["tab_id"],
                    "attachment_id": attachment_id, "generation": row[0],
                    "session_id": body["session_id"], "snapshot": snapshot}
        return db._execute_write(write)

    def _snapshot(self, db, scope, body):
        with db._read_ctx() as conn:
            row = self._check_attachment(db, conn, scope, body)
            return {"profile": scope[0], **self._snapshot_data(
                db, conn, body["session_id"], row["conversation_id"])}

    def _commit(self, db, scope, body):
        receipt = db.append_passive_messages(
            body["session_id"], producer=PRODUCER, event_id=body["event_id"],
            origin_turn_id=body["origin_turn_id"], messages=body["messages"],
            _commit_guard=lambda conn: self._check_attachment(db, conn, scope, body))
        return {"status": "already_saved" if receipt.replayed else "saved",
                "profile": scope[0], "receipt": asdict(receipt)}

    def _reconcile(self, db, scope, body):
        try:
            receipt = db.get_passive_history_receipt(
                body["session_id"], producer=PRODUCER, event_id=body["event_id"])
        except PassiveHistoryTargetError:
            if db.get_session(body["session_id"]) is None:
                raise IngressError("target_missing", 404) from None
            raise
        return {"status": "saved" if receipt else "unknown", "profile": scope[0],
                "receipt": asdict(receipt) if receipt else None}

    def _detach(self, db, scope, body):
        def write(conn):
            self._check_attachment(db, conn, scope, body)
            conn.execute("UPDATE passive_history_attachments SET host_epoch='',generation=generation+1 "
                         "WHERE profile_id=? AND principal_id=? AND tab_id=?", scope)
            return {"status": "detached"}
        return db._execute_write(write)
