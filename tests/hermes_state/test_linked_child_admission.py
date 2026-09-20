"""Canonical parent-input and linked-child admission share one durable transaction."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_linked_child import LinkedChildConflictError
from passive_history_ingress import PRODUCER


def _args(**changes):
    values = {
        "authority_scope": "a" * 64,
        "idempotency_key_hash": "b" * 64,
        "producer": PRODUCER,
        "event_id": "event-1",
        "origin_turn_id": "origin-1",
        "receipt_revision": None,
        "input_text": "Exact parent words",
        "goal": "Execute the approved child goal",
        "context": "Verified child context",
        "correlation_id": "action-1",
        "gateway_session_key": "",
    }
    values.update(changes)
    return values


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("parent", source="test")
    try:
        yield db, path
    finally:
        db.close()


def test_fresh_admission_replays_same_run_child_and_content_free_receipt(store):
    db, path = store
    first = db.admit_linked_child("parent", **_args())
    assert not first.replayed
    assert first.run_id.startswith("run_") and first.child_session_id.startswith("child_")
    assert first.canonical_parent_session_id == "parent"
    assert [row["content"] for row in db.get_messages("parent")] == ["Exact parent words"]

    reopened = SessionDB(path)
    try:
        replay = reopened.admit_linked_child("parent", **_args())
        assert replay.replayed
        assert (replay.run_id, replay.child_session_id, replay.parent_message_id) == (
            first.run_id, first.child_session_id, first.parent_message_id)
        with reopened._read_ctx() as conn:
            row = dict(conn.execute("SELECT * FROM linked_child_admissions").fetchone())
        blob = repr(row)
        assert "Exact parent words" not in blob
        assert "Execute the approved child goal" not in blob
        assert "Verified child context" not in blob
        assert "raw-idempotency-key" not in blob
    finally:
        reopened.close()


def test_changed_retry_and_reused_correlation_fail_before_another_parent_commit(store):
    db, _ = store
    db.admit_linked_child("parent", **_args())
    with pytest.raises(LinkedChildConflictError):
        db.admit_linked_child("parent", **_args(context="Changed context"))
    with pytest.raises(LinkedChildConflictError):
        db.admit_linked_child(
            "parent",
            **_args(
                idempotency_key_hash="c" * 64,
                event_id="event-2",
                origin_turn_id="origin-2",
                input_text="Second parent input",
            ),
        )
    assert [row["content"] for row in db.get_messages("parent")] == ["Exact parent words"]


def test_supplied_receipt_rechecks_exact_revision_origin_and_input(store):
    db, _ = store
    passive = db.append_passive_messages(
        "parent", producer=PRODUCER, event_id="event-1", origin_turn_id="origin-1",
        messages=[{"role": "user", "content": "Exact parent words"}],
    )
    admitted = db.admit_linked_child(
        "parent", **_args(receipt_revision=passive.revision))
    assert admitted.parent_message_id == passive.message_ids[0]
    assert len(db.get_messages("parent")) == 1

    for changed in (
        {"receipt_revision": passive.revision + 1},
        {"input_text": "Changed parent words"},
        {"origin_turn_id": "origin-other"},
    ):
        args = _args(
            idempotency_key_hash=hashlib.sha256(repr(changed).encode()).hexdigest(),
            receipt_revision=passive.revision,
        )
        args.update(changed)
        with pytest.raises(LinkedChildConflictError):
            db.admit_linked_child("parent", **args)
    assert len(db.get_messages("parent")) == 1


def test_concurrent_callers_create_one_admission(store):
    db, path = store
    peers = [SessionDB(path) for _ in range(8)]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = list(pool.map(
                lambda peer: peer.admit_linked_child("parent", **_args()), peers))
        assert len({item.run_id for item in receipts}) == 1
        assert len({item.child_session_id for item in receipts}) == 1
        assert sum(not item.replayed for item in receipts) == 1
        with db._read_ctx() as conn:
            assert conn.execute("SELECT count(*) FROM linked_child_admissions").fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM passive_history_commits").fetchone()[0] == 1
        assert len(db.get_messages("parent")) == 1
    finally:
        for peer in peers:
            peer.close()


def test_admission_insert_abort_rolls_back_parent_input_and_receipt(store):
    db, _ = store
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_linked_child BEFORE INSERT ON linked_child_admissions "
        "BEGIN SELECT RAISE(ABORT,'simulated admission crash'); END"))
    with pytest.raises(sqlite3.IntegrityError):
        db.admit_linked_child("parent", **_args())
    assert db.get_messages("parent") == []
    with db._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM passive_history_commits").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM linked_child_admissions").fetchone()[0] == 0
    db._execute_write(lambda conn: conn.execute("DROP TRIGGER reject_linked_child"))
    assert db.admit_linked_child("parent", **_args()).parent_message_id > 0
