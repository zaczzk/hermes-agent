"""Real SQLite attachment fencing and response-loss reconciliation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hermes_state import SessionDB
from hermes_state_passive_history import (
    PassiveHistoryBusyError, PassiveHistoryConflictError, PassiveHistoryRetiredError,
    PassiveHistoryTargetError,
)
from passive_history_ingress import IngressError, PassiveHistoryIngress


@pytest.fixture
def store(tmp_path):
    first, second = SessionDB(tmp_path / "state.db"), SessionDB(tmp_path / "state.db")
    first.create_session("original", source="test")
    first.create_session("other", source="test")
    try:
        yield first, second, PassiveHistoryIngress()
    finally:
        first.close()
        second.close()


def call(service, db, operation, body, **scope):
    return service.dispatch(db, profile=scope.get("profile", "test"),
                            principal=scope.get("principal", "owner"), operation=operation, body=body)


def attach(service, db, tab="tab-a", session="original"):
    response = call(service, db, "attach", {"tab_id": tab, "session_id": session})
    return {key: response[key] for key in ("tab_id", "session_id", "generation", "attachment_id")}


def event(attachment, event_id="event-1"):
    return {**attachment, "event_id": event_id, "origin_turn_id": "utterance-1",
            "messages": [{"role": "user", "content": "saved speech"}]}


def test_independent_tabs_detach_and_restart_reconcile(store):
    db, _, service = store
    first, second = attach(service, db), attach(service, db, "tab-b")
    saved = call(service, db, "commit", event(first))
    replay = call(service, db, "commit", event(first))
    assert saved["status"] == "saved" and replay["status"] == "already_saved"
    assert saved["receipt"]["message_ids"] == replay["receipt"]["message_ids"]
    replacement = attach(service, db, session="other")
    with pytest.raises(IngressError, match="stale_attachment"):
        call(service, db, "detach", first)
    assert call(service, db, "snapshot", replacement)["session_id"] == "other"
    assert call(service, db, "snapshot", second)["session_id"] == "original"
    call(service, db, "detach", second)
    with pytest.raises(IngressError, match="stale_attachment"):
        call(service, db, "commit", event(second, "event-2"))
    restarted = PassiveHistoryIngress()
    with pytest.raises(IngressError, match="stale_attachment"):
        call(restarted, db, "commit", event(replacement, "event-2"))
    reconciled = call(restarted, db, "reconcile", {"session_id": "original", "event_id": "event-1"})
    assert reconciled["receipt"]["message_ids"] == saved["receipt"]["message_ids"]
    assert call(restarted, db, "reconcile", {
        "session_id": "original", "event_id": "event-2"})["status"] == "unknown"
    assert len(db.get_messages("original")) == 1
    assert db.get_messages("other") == []


def test_switch_wins_while_commit_waits_for_writer_admission(store, monkeypatch):
    db, switch_db, service = store
    old = attach(service, db)
    waiting, release = Event(), Event()
    execute = db._execute_write

    def delayed_execute(fn, **kwargs):
        waiting.set()
        assert release.wait(5)
        return execute(fn, **kwargs)

    monkeypatch.setattr(db, "_execute_write", delayed_execute)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(call, service, db, "commit", event(old))
        try:
            assert waiting.wait(5)
            attach(service, switch_db, session="other")
        finally:
            release.set()
        with pytest.raises(IngressError, match="stale_attachment"):
            pending.result(timeout=5)
    assert db.get_messages("original") == db.get_messages("other") == []
    assert db.get_passive_history_watermark("original").revision == 0


def test_guard_is_inside_writer_transaction_and_scope_bound(store, monkeypatch):
    db, _, service = store
    attached = attach(service, db)
    check = service._check_attachment
    observed = []

    def check_in_transaction(db, conn, scope, body):
        observed.append(conn.in_transaction)
        return check(db, conn, scope, body)

    monkeypatch.setattr(service, "_check_attachment", check_in_transaction)
    for scope in ({"principal": "foreign"}, {"profile": "foreign"}):
        with pytest.raises(IngressError, match="stale_attachment"):
            call(service, db, "commit", event(attached), **scope)
    call(service, db, "commit", event(attached))
    assert observed == [True, True, True]
    with pytest.raises(PassiveHistoryConflictError):
        changed = event(attached)
        changed["messages"][0]["content"] = "changed speech"
        call(service, db, "commit", changed)
    assert [m["content"] for m in db.get_messages("original")] == ["saved speech"]


@pytest.mark.parametrize("delete_session", [False, True])
def test_retention_invalidates_authority_and_receipt(store, delete_session):
    db, _, service = store
    attached = attach(service, db)
    call(service, db, "commit", event(attached))
    if delete_session:
        db.delete_session("original")
        db.create_session("original", source="test")
    else:
        db.clear_messages("original")
    with pytest.raises(IngressError, match="stale_attachment"):
        call(service, db, "commit", event(attached, "new-event"))
    with pytest.raises(PassiveHistoryRetiredError):
        call(service, db, "reconcile", {"session_id": "original", "event_id": "event-1"})
    assert db.get_messages("original") == []


def test_compression_follows_only_canonical_owner_and_busy_refuses(store):
    db, _, service = store
    attached = attach(service, db)
    assert db.acquire_session_turn_lease("original", "worker", wait_seconds=0)
    with pytest.raises(PassiveHistoryBusyError):
        call(service, db, "commit", event(attached))
    db.release_session_turn_lease("original", "worker")
    db.end_session("original", "compression")
    db.create_session("tip", source="test", parent_session_id="original")
    db.create_session("branch", source="test", parent_session_id="original",
                      model_config={"_branched_from": "original"})
    saved = call(service, db, "commit", event(attached))
    assert saved["receipt"]["session_id"] == "tip"
    assert db.get_messages("branch") == []
    with pytest.raises(PassiveHistoryConflictError):
        call(service, db, "reconcile", {"session_id": "branch", "event_id": "event-1"})
    db.end_session("tip", "closed")
    with pytest.raises(PassiveHistoryTargetError):
        attach(service, db)


def test_snapshot_is_bounded_display_context(store):
    db, _, service = store
    db.append_message("original", "system", "must not be disclosed")
    for index in range(25):
        db.append_message("original", "user", f"{index}:" + "😀" * 3000)
    snapshot = call(service, db, "attach", {"tab_id": "tab-a", "session_id": "original"})["snapshot"]
    assert snapshot["truncated"] is True
    assert len(snapshot["messages"]) <= 20
    assert sum(len(m["content"].encode()) for m in snapshot["messages"]) <= 32 * 1024
    assert all(m["role"] == "user" for m in snapshot["messages"])
    assert snapshot["capabilities"]["origin_adoption"] is False
