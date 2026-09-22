import hashlib
import json
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_run_evidence, api_server_run_idempotency as store_module
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB
from tools.run_evidence import RunEvidenceScope, RunEvidenceSession, _stable_read
from tools.file_tools import patch_tool, write_file_tool


def _scope(run_id: str, owner: str = "a" * 64) -> RunEvidenceScope:
    return RunEvidenceScope(
        owner_scope=owner,
        api_run_id=run_id,
        canonical_parent_session_id="parent",
        child_session_id="child",
        origin_turn_id="turn",
        correlation_id="action",
        request_sha256="b" * 64,
    )


def _reserve(store, owner: str, run_id: str, *, created_status="running") -> None:
    outcome, _ = store.reserve(
        owner,
        f"key-{run_id}",
        hashlib.sha256(run_id.encode()).hexdigest(),
        run_id,
        {"object": "hermes.run", "run_id": run_id, "status": created_status},
        durable_replay=True,
    )
    assert outcome == "created"


def _local_ops():
    return SimpleNamespace(env=SimpleNamespace(is_local=True))


def test_scoped_write_seals_exact_bytes_without_private_path(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    run_id, owner = "run_evidence", "a" * 64
    _reserve(store, owner, run_id)
    target = tmp_path / "owner secret" / "result.txt"
    target.parent.mkdir()
    target.write_bytes(b"verified output\n")
    capture = RunEvidenceSession(_scope(run_id, owner))
    capture.note_write(str(target), tool_call_id="call-1", file_ops=_local_ops())

    descriptor = capture.seal(store)
    manifest = store.evidence_manifest(owner, run_id)
    assert descriptor["state"] == "complete"
    assert manifest["items"] == [{
        "item_id": manifest["items"][0]["item_id"],
        "ordinal": 1,
        "size": 16,
        "sha256": hashlib.sha256(b"verified output\n").hexdigest(),
        "kind": "file",
        "display_name": "file-01.txt",
        "text_utf8": True,
    }]
    assert str(target) not in json.dumps(manifest)
    stored = store.evidence_item(
        owner, run_id, descriptor["package_id"], manifest["items"][0]["item_id"]
    )
    assert stored["bytes"] == b"verified output\n"
    assert store.evidence_manifest("b" * 64, run_id) is None
    store.close()
    reopened = RunIdempotencyStore(str(tmp_path / "runs.db"))
    assert reopened.evidence_manifest(owner, run_id) == manifest
    reopened.close()


def test_real_write_and_patch_tools_capture_final_bytes(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    owner, run_id = "a" * 64, "run_file_tools"
    _reserve(store, owner, run_id)
    capture = RunEvidenceSession(_scope(run_id, owner))
    target = tmp_path / "edited.txt"
    written = json.loads(write_file_tool(
        str(target),
        "before\n",
        task_id="evidence-tool-task",
        run_evidence_session=capture,
        tool_call_id="write-call",
    ))
    assert "error" not in written
    patched = json.loads(patch_tool(
        path=str(target),
        old_string="before",
        new_string="after",
        task_id="evidence-tool-task",
        run_evidence_session=capture,
        tool_call_id="patch-call",
    ))
    assert "error" not in patched
    descriptor = capture.seal(store)
    manifest = store.evidence_manifest(owner, run_id)
    item = manifest["items"][0]
    stored = store.evidence_item(owner, run_id, descriptor["package_id"], item["item_id"])
    assert stored["bytes"].replace(b"\r\n", b"\n") == b"after\n"
    assert descriptor["item_count"] == 1 and descriptor["state"] == "complete"
    store.close()


def test_replacement_unbound_tool_and_item_cap_are_partial(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    owner = "a" * 64

    _reserve(store, owner, "run_replaced")
    changed = tmp_path / "changed.txt"
    changed.write_text("first", encoding="utf-8")
    replaced = RunEvidenceSession(_scope("run_replaced", owner))
    replaced.note_write(str(changed), tool_call_id="write", file_ops=_local_ops())
    changed.write_text("second", encoding="utf-8")
    replaced.note_tool("terminal", "shell-call")
    descriptor = replaced.seal(store)
    assert descriptor == {
        "package_id": descriptor["package_id"],
        "state": "unavailable",
        "manifest_sha256": descriptor["manifest_sha256"],
        "item_count": 0,
        "total_bytes": 0,
        "omitted_count": 2,
    }

    _reserve(store, owner, "run_capped")
    capped = RunEvidenceSession(_scope("run_capped", owner))
    for index in range(9):
        path = tmp_path / f"file-{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        capped.note_write(str(path), tool_call_id=f"call-{index}", file_ops=_local_ops())
    descriptor = capped.seal(store)
    assert descriptor["state"] == "partial"
    assert descriptor["item_count"] == 8
    assert descriptor["omitted_count"] == 1
    store.close()


def test_nonlocal_and_symlink_writes_never_capture_bytes(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    owner, run_id = "a" * 64, "run_unsafe"
    _reserve(store, owner, run_id)
    target = tmp_path / "target.txt"
    target.write_text("private", encoding="utf-8")
    capture = RunEvidenceSession(_scope(run_id, owner))
    capture.note_write(
        str(target),
        tool_call_id="remote",
        file_ops=SimpleNamespace(env=SimpleNamespace(is_local=False)),
    )
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pass
    else:
        capture.note_write(str(link), tool_call_id="link", file_ops=_local_ops())
    descriptor = capture.seal(store)
    assert descriptor["state"] == "unavailable"
    assert descriptor["item_count"] == 0
    assert descriptor["omitted_count"] >= 1
    store.close()


def test_nonfollowing_reader_refuses_symlink_mode(monkeypatch, tmp_path):
    target = tmp_path / "ordinary.txt"
    target.write_text("content", encoding="utf-8")
    actual = target.lstat()
    fake = SimpleNamespace(
        st_mode=0o120777,
        st_dev=actual.st_dev,
        st_ino=actual.st_ino,
        st_mtime=actual.st_mtime,
        st_mtime_ns=actual.st_mtime_ns,
        st_file_attributes=0,
        st_size=actual.st_size,
    )
    monkeypatch.setattr("tools.run_evidence.os.lstat", lambda _path: fake)
    assert _stable_read(str(target)) is None


def test_ack_expiry_leaves_tombstone_and_run_replay(monkeypatch, tmp_path):
    clock = {"now": 100.0}
    monkeypatch.setattr(store_module.time, "time", lambda: clock["now"])
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    owner, run_id = "a" * 64, "run_retained"
    _reserve(store, owner, run_id)
    capture = RunEvidenceSession(_scope(run_id, owner))
    target = tmp_path / "result.txt"
    target.write_text("ok", encoding="utf-8")
    capture.note_write(str(target), tool_call_id="write", file_ops=_local_ops())
    descriptor = capture.seal(store)
    clock["now"] = 200.0
    first = store.acknowledge_evidence(
        owner, run_id, descriptor["package_id"], descriptor["manifest_sha256"]
    )
    second = store.acknowledge_evidence(
        owner, run_id, descriptor["package_id"], descriptor["manifest_sha256"]
    )
    assert first == second
    with pytest.raises(ValueError):
        store.acknowledge_evidence(owner, run_id, descriptor["package_id"], "0" * 64)

    clock["now"] = first["expires_at"] + 1
    assert store.evidence_manifest(owner, run_id) is None
    assert store.evidence_tombstone(owner, run_id, descriptor["package_id"])
    assert store.owns_run(owner, run_id)

    clock["now"] = 1_000.0
    _reserve(store, owner, "run_hard_expiry")
    empty = RunEvidenceSession(_scope("run_hard_expiry", owner)).seal(store)
    clock["now"] = 1_000.0 + 90 * 24 * 60 * 60 + 1
    assert store.evidence_manifest(owner, "run_hard_expiry") is None
    assert store.evidence_tombstone(
        owner, "run_hard_expiry", empty["package_id"]
    )
    store.close()


@pytest.mark.asyncio
async def test_authenticated_endpoints_are_owner_fenced_and_safe(tmp_path, monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True,
        extra={"key": "owner-secret-key-123456", "runs_idempotency_path": str(tmp_path / "runs.db")},
    ))
    request = SimpleNamespace(headers={}, path="/", method="GET")
    owner = adapter._run_idempotency_scope(request)
    monkeypatch.setattr(
        adapter,
        "_run_idempotency_scope",
        lambda incoming: incoming.headers.get("X-Test-Owner", ""),
    )
    run_id = "run_http_evidence"
    _reserve(adapter._run_idempotency_store, owner, run_id)
    adapter._run_owners[run_id] = owner
    target = tmp_path / "private-name.txt"
    target.write_text("hello", encoding="utf-8")
    capture = RunEvidenceSession(_scope(run_id, owner))
    capture.note_write(str(target), tool_call_id="write", file_ops=_local_ops())
    descriptor = capture.seal(adapter._run_idempotency_store)

    app = web.Application()
    app.router.add_get("/v1/runs/{run_id}/evidence", adapter._handle_run_evidence)
    app.router.add_get(
        "/v1/runs/{run_id}/evidence/{package_id}/{item_id}",
        adapter._handle_run_evidence_item,
    )
    app.router.add_post(
        "/v1/runs/{run_id}/evidence/{package_id}/ack",
        adapter._handle_run_evidence_ack,
    )
    headers = {
        "Authorization": "Bearer owner-secret-key-123456",
        "X-Test-Owner": owner,
    }
    async with TestClient(TestServer(app)) as client:
        response = await client.get(f"/v1/runs/{run_id}/evidence", headers=headers)
        assert response.status == 200
        manifest = await response.json()
        item = manifest["items"][0]
        assert "private-name" not in json.dumps(manifest)

        response = await client.get(
            f"/v1/runs/{run_id}/evidence/{descriptor['package_id']}/{item['item_id']}",
            headers=headers,
        )
        assert response.status == 200 and await response.read() == b"hello"
        assert response.headers["Content-Type"] == "application/octet-stream"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Content-Security-Policy"] == "sandbox; default-src 'none'"
        assert response.headers["Content-Disposition"] == 'attachment; filename="file-01.txt"'

        response = await client.post(
            f"/v1/runs/{run_id}/evidence/{descriptor['package_id']}/ack",
            headers=headers,
            json={"manifest_sha256": descriptor["manifest_sha256"]},
        )
        assert response.status == 200 and (await response.json())["acknowledged"] is True
        response = await client.get(f"/v1/runs/{run_id}/evidence")
        assert response.status == 401
        response = await client.get(
            f"/v1/runs/{run_id}/evidence",
            headers={**headers, "X-Test-Owner": "other-owner"},
        )
        assert response.status == 404
    await adapter.disconnect()


def test_capability_requires_durable_store_and_local_backend(monkeypatch):
    adapter = SimpleNamespace(_run_idempotency_store=SimpleNamespace(durable=False))
    assert api_server_run_evidence.capabilities(adapter) is None
    adapter._run_idempotency_store.durable = True
    monkeypatch.setattr(
        "tools.terminal_tool._session_scope",
        lambda: SimpleNamespace(env_type="ssh"),
    )
    assert api_server_run_evidence.capabilities(adapter) is None


@pytest.mark.asyncio
async def test_owned_run_before_seal_is_not_ready(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True,
        extra={"key": "owner-secret-key-123456", "runs_idempotency_path": str(tmp_path / "runs.db")},
    ))
    request = SimpleNamespace(headers={}, path="/", method="GET")
    owner = adapter._run_idempotency_scope(request)
    run_id = "run_not_ready"
    _reserve(adapter._run_idempotency_store, owner, run_id)
    adapter._run_owners[run_id] = owner
    app = web.Application()
    app.router.add_get("/v1/runs/{run_id}/evidence", adapter._handle_run_evidence)
    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            f"/v1/runs/{run_id}/evidence",
            headers={"Authorization": "Bearer owner-secret-key-123456"},
        )
        assert response.status == 409
        assert (await response.json())["error"]["code"] == "evidence_not_ready"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_linked_run_binds_scope_through_real_file_tool_and_status(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True,
        extra={"key": "owner-secret-key-123456", "runs_idempotency_path": str(home / "runs.db")},
    ))
    db = SessionDB(home / "state.db")
    db.create_session("parent", source="test")
    adapter._session_db = db
    target = tmp_path / "linked-output.txt"

    class Agent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = None

        def run_conversation(self, *, user_message, conversation_history, task_id, **_kwargs):
            scope = self._run_evidence_session.scope
            assert scope.api_run_id.startswith("run_")
            assert scope.child_session_id == task_id
            assert scope.canonical_parent_session_id == "parent"
            result = json.loads(write_file_tool(
                str(target),
                "linked bytes",
                task_id=task_id,
                run_evidence_session=self._run_evidence_session,
                tool_call_id="linked-write",
            ))
            assert "error" not in result
            self.session_id = task_id
            return {"final_response": "complete"}

    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/evidence", adapter._handle_run_evidence)
    headers = {
        "Authorization": "Bearer owner-secret-key-123456",
        "Idempotency-Key": "reviewed-action-1",
    }
    body = {
        "input": "Owner approved this exact action",
        "session_id": "parent",
        "origin": {"event_id": "event-1", "origin_turn_id": "turn-1"},
        "child": {
            "goal": "Write the accepted result",
            "context": "Exact reviewed scope",
            "correlation_id": "action-1",
        },
    }
    with patch.object(adapter, "_create_agent", return_value=Agent()):
        async with TestClient(TestServer(app)) as client:
            accepted = await client.post("/v1/runs", headers=headers, json=body)
            run_id = (await accepted.json())["run_id"]
            await asyncio.wait_for(
                asyncio.shield(adapter._active_run_tasks[run_id]), timeout=10
            )
            response = await client.get(
                f"/v1/runs/{run_id}",
                headers={"Authorization": "Bearer owner-secret-key-123456"},
            )
            status = await response.json()
            assert status.get("status") == "completed", status
            assert status["evidence"]["state"] == "complete"
            evidence = await client.get(
                f"/v1/runs/{run_id}/evidence",
                headers={"Authorization": "Bearer owner-secret-key-123456"},
            )
            manifest = await evidence.json()
            assert evidence.status == 200 and manifest["item_count"] == 1
            assert "linked-output" not in json.dumps(manifest)
    await adapter.disconnect()
    db.close()
