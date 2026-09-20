"""Authenticated linked children extend /v1/runs without a second executor."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms import api_server_run_idempotency, api_server_runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB
from passive_history_ingress import PRODUCER


AUTH = {"Authorization": "Bearer sk-secret", "Idempotency-Key": "approved-action-1"}


def _body(**changes):
    value = {
        "input": "Exact words discussed with the owner",
        "session_id": "parent",
        "origin": {"event_id": "event-1", "origin_turn_id": "origin-1"},
        "child": {
            "goal": "Execute the approved child goal",
            "context": "Verified parent context",
            "correlation_id": "action-1",
        },
    }
    value.update(changes)
    return value


def _app(adapter):
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


@pytest.fixture
def host(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    db = SessionDB(home / "state.db")
    db.create_session("parent", source="test")
    adapter._session_db = db
    try:
        yield adapter, db, home
    finally:
        api_server_runs._close_run_state(adapter)
        db.close()


class _Agent:
    def __init__(self):
        self.calls = []
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_id = None

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.calls.append((user_message, conversation_history, task_id, kwargs))
        self.session_id = task_id
        return {"final_response": "child complete"}


async def _settled(cli, run_id):
    for _ in range(100):
        response = await cli.get(f"/v1/runs/{run_id}", headers={"Authorization": "Bearer sk-secret"})
        data = await response.json()
        if data.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return data
        await asyncio.sleep(0.01)
    raise AssertionError("run did not settle")


@pytest.mark.asyncio
async def test_strict_branch_rejects_partial_or_extended_shapes_before_legacy_execution(host):
    adapter, db, _ = host
    create = MagicMock()
    with patch.object(adapter, "_create_agent", create):
        async with TestClient(TestServer(_app(adapter))) as cli:
            for body in (
                {"input": "words", "origin": {"event_id": "e", "origin_turn_id": "o"}},
                {**_body(), "model": "ignored-if-legacy"},
                {**_body(), "child": {**_body()["child"], "worker": "codex"}},
            ):
                response = await cli.post("/v1/runs", json=body, headers=AUTH)
                assert response.status == 400
    create.assert_not_called()
    assert db.get_messages("parent") == []


@pytest.mark.asyncio
async def test_fresh_parent_input_runs_exact_goal_in_separate_child_and_replays(host):
    adapter, db, _ = host
    agent = _Agent()
    with patch.object(adapter, "_create_agent", return_value=agent) as create:
        async with TestClient(TestServer(_app(adapter))) as cli:
            accepted = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            assert accepted.status == 202
            first = await accepted.json()
            status = await _settled(cli, first["run_id"])
            replayed = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            replay = await replayed.json()

    assert replayed.status == 202 and replay == {
        "run_id": first["run_id"], "status": "completed", "replayed": True}
    assert create.call_count == 1
    kwargs = create.call_args.kwargs
    assert kwargs["session_id"] == status["child_session_id"]
    assert kwargs["parent_session_id"] == "parent"
    assert kwargs["ephemeral_system_prompt"] == (
        "Linked child context (not parent transcript):\nVerified parent context")
    assert agent.calls == [(
        "Execute the approved child goal", [], status["child_session_id"], {})]
    assert status["session_id"] == "parent"
    assert status["parent_message_id"] > 0
    assert [row["content"] for row in db.get_messages("parent")] == [
        "Exact words discussed with the owner"]


@pytest.mark.asyncio
async def test_execution_session_key_is_bound_into_canonical_request(host):
    adapter, _db, _ = host
    agent = _Agent()
    first_headers = {**AUTH, "X-Hermes-Session-Key": "scope-a"}
    changed_headers = {**AUTH, "X-Hermes-Session-Key": "scope-b"}
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_app(adapter))) as cli:
            accepted = await cli.post("/v1/runs", json=_body(), headers=first_headers)
            await _settled(cli, (await accepted.json())["run_id"])
            conflict = await cli.post("/v1/runs", json=_body(), headers=changed_headers)
    assert accepted.status == 202
    assert conflict.status == 409
    assert len(agent.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_exact_retry_preserves_first_run_maps_and_launches_once(host):
    adapter, _db, _ = host
    arrived = 0
    both_arrived = asyncio.Event()
    launches = []
    hold_launch = asyncio.Event()

    async def resolve_barrier(_adapter, session_id):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_arrived.set()
        await both_arrived.wait()
        return session_id

    async def held_execute(_adapter, launch, **_kwargs):
        launches.append(launch)
        await hold_launch.wait()

    with (
        patch("gateway.platforms.api_server_runs._resolve_live_session_id", resolve_barrier),
        patch("gateway.platforms.api_server_runs._execute_run", held_execute),
    ):
        async with TestClient(TestServer(_app(adapter))) as cli:
            responses = await asyncio.gather(*(
                cli.post("/v1/runs", json=_body(), headers=AUTH) for _ in range(2)))
            payloads = [await response.json() for response in responses]
            await asyncio.sleep(0)

    assert [response.status for response in responses] == [202, 202]
    assert len({payload["run_id"] for payload in payloads}) == 1
    assert sorted(payload["replayed"] for payload in payloads) == [False, True]
    run_id = payloads[0]["run_id"]
    assert len(launches) == 1
    assert adapter._active_run_tasks[run_id] is not None
    assert launches[0].queue is adapter._run_streams[run_id]
    assert adapter._run_statuses[run_id]["run_id"] == run_id
    assert adapter._run_approval_sessions[run_id] == run_id
    assert len(adapter._run_owners[run_id]) == 64
    task = adapter._active_run_tasks.pop(run_id)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_supplied_receipt_must_match_revision_event_origin_and_input(host):
    adapter, db, _ = host
    receipt = db.append_passive_messages(
        "parent", producer=PRODUCER, event_id="event-1", origin_turn_id="origin-1",
        messages=[{"role": "user", "content": "Exact words discussed with the owner"}],
    )
    origin = {**_body()["origin"], "receipt_id": receipt.revision}
    good = _body(origin=origin)
    agent = _Agent()
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_app(adapter))) as cli:
            bad_revision = await cli.post(
                "/v1/runs", json=_body(origin={**origin, "receipt_id": receipt.revision + 1}),
                headers={**AUTH, "Idempotency-Key": "bad-revision"})
            bad_input = await cli.post(
                "/v1/runs", json={**good, "input": "different words"},
                headers={**AUTH, "Idempotency-Key": "bad-input"})
            accepted = await cli.post("/v1/runs", json=good, headers=AUTH)
            assert bad_revision.status == bad_input.status == 409
            assert accepted.status == 202
            await _settled(cli, (await accepted.json())["run_id"])
    assert len(db.get_messages("parent")) == 1


@pytest.mark.asyncio
async def test_crash_before_run_reservation_reuses_canonical_run_and_child(host):
    adapter, db, _ = host
    reserve = adapter._run_idempotency_store.reserve
    adapter._run_idempotency_store.reserve = MagicMock(side_effect=RuntimeError("crash boundary"))
    async with TestClient(TestServer(_app(adapter))) as cli:
        failed = await cli.post("/v1/runs", json=_body(), headers=AUTH)
        assert failed.status == 500
    with db._read_ctx() as conn:
        stored = dict(conn.execute("SELECT * FROM linked_child_admissions").fetchone())

    adapter._run_idempotency_store.reserve = reserve
    agent = _Agent()
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_app(adapter))) as cli:
            accepted = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            payload = await accepted.json()
            assert accepted.status == 202 and payload["run_id"] == stored["run_id"]
            status = await _settled(cli, payload["run_id"])
    assert status["child_session_id"] == stored["child_session_id"]
    assert len(agent.calls) == 1 and len(db.get_messages("parent")) == 1


@pytest.mark.asyncio
async def test_reserved_unsettled_retry_never_launches_again(host):
    adapter, db, home = host
    gate = asyncio.Event()

    async def held_execute(*_args, **_kwargs):
        await gate.wait()

    with patch("gateway.platforms.api_server_runs._execute_run", held_execute):
        async with TestClient(TestServer(_app(adapter))) as cli:
            accepted = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            run_id = (await accepted.json())["run_id"]
            child_session_id = adapter._run_statuses[run_id]["child_session_id"]
            task = adapter._active_run_tasks.pop(run_id)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    adapter._run_idempotency_store._conn.execute(
        "UPDATE run_idempotency SET owner_pid=99999999 WHERE run_id=?", (run_id,))
    adapter._run_idempotency_store._conn.commit()
    api_server_runs._close_run_state(adapter)
    db.close()
    reopened_db = SessionDB(home / "state.db")
    adapter._session_db = reopened_db
    adapter._run_idempotency_store = RunIdempotencyStore(str(home / "runs_idempotency.db"))
    adapter._run_statuses.clear()
    adapter._run_idempotency_ids.clear()
    adapter._run_owners.clear()
    create = MagicMock()
    with patch.object(adapter, "_create_agent", create):
        async with TestClient(TestServer(_app(adapter))) as cli:
            replayed = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            payload = await replayed.json()
            status_response = await cli.get(
                f"/v1/runs/{run_id}", headers={"Authorization": "Bearer sk-secret"})
            status = await status_response.json()
    reopened_db.close()
    assert replayed.status == 202
    assert payload == {"run_id": run_id, "status": "interrupted", "replayed": True}
    assert status["run_id"] == run_id
    assert status["child_session_id"] == child_session_id
    create.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_ack_expiry_cannot_make_linked_action_executable_again(host, monkeypatch):
    adapter, _db, _ = host
    first_agent = _Agent()
    with patch.object(adapter, "_create_agent", return_value=first_agent):
        async with TestClient(TestServer(_app(adapter))) as cli:
            accepted = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            run_id = (await accepted.json())["run_id"]
            await _settled(cli, run_id)

    store = adapter._run_idempotency_store
    store._conn.execute(
        "UPDATE run_idempotency SET acknowledged_at=100, updated_at=100, retention_until=1 "
        "WHERE run_id=?",
        (run_id,),
    )
    store._conn.commit()
    expired = 100 + store.ACKNOWLEDGED_RETENTION_SECONDS + 1
    monkeypatch.setattr(api_server_run_idempotency.time, "time", lambda: expired)
    adapter._run_statuses.clear()
    adapter._run_owners.clear()
    create = MagicMock()
    with patch.object(adapter, "_create_agent", create):
        async with TestClient(TestServer(_app(adapter))) as cli:
            replayed = await cli.post("/v1/runs", json=_body(), headers=AUTH)
            payload = await replayed.json()
    row = store._conn.execute(
        "SELECT durable_replay FROM run_idempotency WHERE run_id=?", (run_id,)
    ).fetchone()
    assert replayed.status == 202
    assert payload == {"run_id": run_id, "status": "completed", "replayed": True}
    assert row[0] == 1
    create.assert_not_called()


@pytest.mark.asyncio
async def test_capability_requires_durable_store_and_advertises_no_extra_workers(host):
    adapter, _db, _ = host
    async with TestClient(TestServer(_app(adapter))) as cli:
        durable = await cli.get("/v1/capabilities", headers={"Authorization": "Bearer sk-secret"})
        child = (await durable.json())["features"]["linked_child_dispatch"]
        assert child == {
            "version": 1, "supported": True, "separate_child_goal": True,
            "origin_sources": ["fresh", "passive_receipt"]}

    api_server_runs._close_run_state(adapter)
    adapter._run_idempotency_store = RunIdempotencyStore(":memory:")
    async with TestClient(TestServer(_app(adapter))) as cli:
        transient = await cli.get("/v1/capabilities", headers={"Authorization": "Bearer sk-secret"})
        assert "linked_child_dispatch" not in (await transient.json())["features"]
