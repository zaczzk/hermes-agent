"""Atomic canonical-input receipts for authenticated linked-child run admission."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass

from hermes_state_passive_history import (
    _PRODUCER_MAX_CHARS,
    _validated_content,
    _validated_identifier,
    _validated_messages,
    _payload_fingerprint,
)

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


class LinkedChildConflictError(ValueError):
    """An idempotency key or action correlation was reused with different bytes."""


@dataclass(frozen=True, slots=True)
class LinkedChildAdmission:
    run_id: str
    child_session_id: str
    selected_parent_session_id: str
    canonical_parent_session_id: str
    parent_message_id: int
    passive_revision: int
    event_id: str
    origin_turn_id: str
    correlation_id: str
    request_sha256: str
    replayed: bool


def _bounded_text(value, field: str, maximum: int, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{field} must be text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} is not valid Unicode") from exc
    if len(encoded) > maximum:
        raise ValueError(f"{field} is too large")
    return value


def _sha256_field(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise ValueError(f"{field} must be an opaque SHA-256 value")
    return value


def linked_child_request_fingerprint(
    *, authority_scope: str, idempotency_key_hash: str, input_text: str,
    selected_parent_session_id: str, producer: str, event_id: str,
    origin_turn_id: str, receipt_revision: int | None, goal: str, context: str,
    correlation_id: str, gateway_session_key: str,
) -> str:
    """Hash every authority- or content-bearing byte in the version-one request."""
    payload = {
        "version": 1,
        "authority_scope": authority_scope,
        "idempotency_key_hash": idempotency_key_hash,
        "input": input_text,
        "selected_parent_session_id": selected_parent_session_id,
        "origin": {
            "producer": producer,
            "event_id": event_id,
            "origin_turn_id": origin_turn_id,
            "receipt_revision": receipt_revision,
        },
        "child": {"goal": goal, "context": context, "correlation_id": correlation_id},
        "gateway_session_key": gateway_session_key,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


class SessionLinkedChildMixin:
    """Bind one passive parent input to one stable child-run receipt."""

    def admit_linked_child(
        self, selected_parent_session_id: str, *, authority_scope: str,
        idempotency_key_hash: str, producer: str, event_id: str,
        origin_turn_id: str, receipt_revision: int | None, input_text: str,
        goal: str, context: str, correlation_id: str, gateway_session_key: str,
    ) -> LinkedChildAdmission:
        if not isinstance(selected_parent_session_id, str) or not (
            1 <= len(selected_parent_session_id) <= 256
        ):
            raise ValueError("selected parent session is invalid")
        authority_scope = _sha256_field(authority_scope, "authority_scope")
        idempotency_key_hash = _sha256_field(idempotency_key_hash, "idempotency_key_hash")
        producer = _validated_identifier(producer, "producer", _PRODUCER_MAX_CHARS)
        event_id = _validated_identifier(event_id, "event_id", 128)
        origin_turn_id = _validated_identifier(origin_turn_id, "origin_turn_id", 128)
        correlation_id = _validated_identifier(correlation_id, "correlation_id", 128)
        if receipt_revision is not None and (
            type(receipt_revision) is not int or receipt_revision < 1
        ):
            raise ValueError("receipt revision is invalid")
        input_text = _validated_content(input_text)
        goal = _bounded_text(goal, "child goal", 16_000)
        context = _bounded_text(context, "child context", 32_000, nonempty=False)
        gateway_session_key = _bounded_text(
            gateway_session_key, "gateway session key", 512, nonempty=False)
        rows = _validated_messages([{"role": "user", "content": input_text}])
        passive_fingerprint = _payload_fingerprint(origin_turn_id, rows)
        request_sha256 = linked_child_request_fingerprint(
            authority_scope=authority_scope, idempotency_key_hash=idempotency_key_hash,
            input_text=input_text, selected_parent_session_id=selected_parent_session_id,
            producer=producer, event_id=event_id, origin_turn_id=origin_turn_id,
            receipt_revision=receipt_revision, goal=goal, context=context,
            correlation_id=correlation_id, gateway_session_key=gateway_session_key)
        candidate_run_id = "run_" + uuid.uuid4().hex
        candidate_child_id = "child_" + hashlib.sha256(
            candidate_run_id.encode("ascii")
        ).hexdigest()[:32]

        def _write(conn):
            existing = conn.execute(
                "SELECT * FROM linked_child_admissions "
                "WHERE authority_scope=? AND idempotency_key_hash=?",
                (authority_scope, idempotency_key_hash),
            ).fetchone()
            if existing is not None and str(existing["request_sha256"]) != request_sha256:
                raise LinkedChildConflictError("linked-child idempotency key conflict")
            correlated = conn.execute(
                "SELECT request_sha256 FROM linked_child_admissions "
                "WHERE authority_scope=? AND correlation_id=?",
                (authority_scope, correlation_id),
            ).fetchone()
            if correlated is not None and str(correlated[0]) != request_sha256:
                raise LinkedChildConflictError("linked-child correlation conflict")

            if receipt_revision is not None:
                receipt_row = conn.execute(
                    "SELECT 1 FROM passive_history_commits WHERE producer=? AND event_id=?",
                    (producer, event_id),
                ).fetchone()
                if receipt_row is None:
                    raise LinkedChildConflictError("passive receipt is missing")
            receipt = self._append_passive_messages_on_conn(
                conn, selected_parent_session_id, producer=producer, event_id=event_id,
                origin_turn_id=origin_turn_id, rows=rows, fingerprint=passive_fingerprint)
            if receipt_revision is not None and receipt.revision != receipt_revision:
                raise LinkedChildConflictError("passive receipt revision conflict")
            if len(receipt.message_ids) != 1:
                raise LinkedChildConflictError("passive receipt is not one parent input")

            if existing is not None:
                if any((
                    str(existing["selected_parent_session_id"]) != selected_parent_session_id,
                    str(existing["canonical_parent_session_id"]) != receipt.session_id,
                    int(existing["passive_revision"]) != receipt.revision,
                    int(existing["parent_message_id"]) != receipt.message_ids[0],
                    str(existing["event_id"]) != event_id,
                    str(existing["origin_turn_id"]) != origin_turn_id,
                    str(existing["correlation_id"]) != correlation_id,
                )):
                    raise LinkedChildConflictError("linked-child admission receipt conflict")
                return LinkedChildAdmission(
                    run_id=str(existing["run_id"]),
                    child_session_id=str(existing["child_session_id"]),
                    selected_parent_session_id=selected_parent_session_id,
                    canonical_parent_session_id=receipt.session_id,
                    parent_message_id=receipt.message_ids[0],
                    passive_revision=receipt.revision, event_id=event_id,
                    origin_turn_id=origin_turn_id, correlation_id=correlation_id,
                    request_sha256=request_sha256, replayed=True)

            conn.execute(
                "INSERT INTO linked_child_admissions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (authority_scope, idempotency_key_hash, request_sha256,
                 candidate_run_id, candidate_child_id, selected_parent_session_id,
                 receipt.session_id, producer, event_id, origin_turn_id, receipt.revision,
                 receipt.message_ids[0], correlation_id, time.time()),
            )
            return LinkedChildAdmission(
                run_id=candidate_run_id, child_session_id=candidate_child_id,
                selected_parent_session_id=selected_parent_session_id,
                canonical_parent_session_id=receipt.session_id,
                parent_message_id=receipt.message_ids[0],
                passive_revision=receipt.revision, event_id=event_id,
                origin_turn_id=origin_turn_id, correlation_id=correlation_id,
                request_sha256=request_sha256, replayed=False)

        return self._execute_write(_write, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
