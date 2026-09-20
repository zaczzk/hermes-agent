from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_talk_test_host.py"
SPEC = importlib.util.spec_from_file_location("run_talk_test_host", SCRIPT)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _lease(tmp_path: Path, **overrides):
    values = {
        "path": tmp_path / "host.json",
        "profile": "default",
        "profile_home": tmp_path / "profile",
        "store_id": "hermes-state-v1:" + "a" * 32,
    }
    values.update(overrides)
    return launcher.ProcessLease(**values)


def test_constants_pin_isolated_loopback_listener():
    assert (launcher.HOST, launcher.PORT) == ("127.0.0.1", 8643)


def test_source_root_precedes_imports_and_wrong_checkout_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", ["elsewhere"])
    launcher._install_source_root()
    assert Path(sys.path[0]).resolve() == launcher.SOURCE_ROOT
    fake = SimpleNamespace(__file__=tmp_path / "installed.py")
    fake.__file__.write_text("", encoding="utf-8")
    with pytest.raises(launcher.LaunchRefused, match="isolated host checkout"):
        launcher._require_source_module(fake, "fake")


def test_process_lease_is_exclusive_and_owner_fenced(tmp_path):
    first = _lease(tmp_path)
    first.acquire()
    record = json.loads(first.path.read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid()
    assert record["host"] == "127.0.0.1"
    assert record["port"] == 8643
    assert "key" not in record

    with pytest.raises(launcher.LaunchRefused, match="owned"):
        _lease(tmp_path).acquire()

    record["owner_token"] = "another-owner"
    first.path.write_text(json.dumps(record), encoding="utf-8")
    assert first.release() is False
    assert first.path.exists()


def test_process_lease_recovers_only_proved_stale_owner(tmp_path, monkeypatch):
    lease = _lease(tmp_path)
    lease.path.write_text(json.dumps({
        "pid": 999_999_999,
        "process_created_at": 1.0,
        "owner_token": "stale",
    }), encoding="utf-8")
    monkeypatch.setattr(launcher, "_same_live_process", lambda *_: False)
    lease.acquire()
    assert json.loads(lease.path.read_text(encoding="utf-8"))["owner_token"] == lease.token
    assert lease.release() is True
    assert not lease.path.exists()


def test_process_lease_refuses_unverifiable_owner(tmp_path, monkeypatch):
    lease = _lease(tmp_path)
    lease.path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "_same_live_process", lambda *_: None)
    with pytest.raises(launcher.LaunchRefused, match="cannot be verified"):
        lease.acquire()
    assert lease.path.exists()


def test_resolve_key_prefers_shared_talk_process_secret(tmp_path, monkeypatch):
    monkeypatch.setenv(launcher.KEY_ENV, "t" * 32)
    monkeypatch.setenv("API_SERVER_KEY", "p" * 32)
    assert launcher._resolve_key(tmp_path) == "t" * 32


def test_blank_shared_talk_secret_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(launcher.KEY_ENV, "   ")
    monkeypatch.setenv("API_SERVER_KEY", "p" * 32)
    with pytest.raises(launcher.LaunchRefused, match=launcher.KEY_ENV):
        launcher._resolve_key(tmp_path)


def test_real_adapter_factory_has_no_gateway_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    run_store_path = launcher._run_store_path(tmp_path)
    run_store_path.parent.mkdir()
    adapter = launcher._build_adapter("k" * 32, run_store_path)
    try:
        assert (adapter._host, adapter._port) == (launcher.HOST, launcher.PORT)
        assert adapter.gateway_runner is None
        assert adapter._api_key == "k" * 32
        assert Path(adapter._run_idempotency_store._db_path) == run_store_path
        assert run_store_path != tmp_path / "runs_idempotency.db"
    finally:
        asyncio.run(adapter.disconnect())


def test_isolated_run_store_reopens_same_durable_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = launcher._run_store_path(tmp_path)
    path.parent.mkdir()
    first = launcher._build_adapter("k" * 32, path)
    try:
        outcome, _ = first._run_idempotency_store.reserve(
            "scope", "stable-key", "fingerprint", "run_stable",
            {"status": "queued"}, durable_replay=True,
        )
        assert outcome == "created"
    finally:
        asyncio.run(first.disconnect())

    second = launcher._build_adapter("k" * 32, path)
    try:
        outcome, record = second._run_idempotency_store.lookup(
            "scope", "stable-key", "fingerprint"
        )
        assert outcome == "reused"
        assert record["run_id"] == "run_stable"
        assert not (tmp_path / "runs_idempotency.db").exists()
    finally:
        asyncio.run(second.disconnect())


def test_serve_always_stops_adapter_and_releases_own_lease(tmp_path, capsys):
    lease = _lease(tmp_path)
    lease.acquire()
    adapter = SimpleNamespace(
        connect=AsyncMock(return_value=True),
        interrupt_active_runs=lambda _reason: 0,
        active_agent_work_count=lambda: 0,
        cancel_background_tasks=AsyncMock(),
        disconnect=AsyncMock(),
    )

    async def immediate(_stop):
        return None

    asyncio.run(launcher._serve(adapter, lease, wait=immediate))
    ready = json.loads(capsys.readouterr().out)
    assert ready == {
        "event": "ready",
        "host": launcher.HOST,
        "pid": os.getpid(),
        "port": launcher.PORT,
        "profile": "default",
        "store_id": "hermes-state-v1:" + "a" * 32,
    }
    adapter.cancel_background_tasks.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()
    assert not lease.path.exists()


def test_failed_connect_disconnects_and_releases_lease(tmp_path):
    lease = _lease(tmp_path)
    lease.acquire()
    adapter = SimpleNamespace(connect=AsyncMock(return_value=False), disconnect=AsyncMock())

    with pytest.raises(launcher.LaunchRefused, match="refused startup"):
        asyncio.run(launcher._serve(adapter, lease))
    adapter.disconnect.assert_awaited_once()
    assert not lease.path.exists()


def test_check_validates_without_adapter_construction_or_lease(tmp_path, monkeypatch, capsys):
    profile_home = tmp_path / "profile"
    imports_and_store = []
    monkeypatch.setattr(launcher, "_resolve_profile", lambda _name: ("default", profile_home))
    monkeypatch.setattr(launcher, "_resolve_key", lambda _home: "k" * 32)
    monkeypatch.setattr(
        launcher,
        "_canonical_store_id",
        lambda _home: imports_and_store.append("store") or "hermes-state-v1:" + "b" * 32,
    )
    monkeypatch.setattr(
        launcher, "_adapter_class", lambda: imports_and_store.append("adapter") or object
    )
    monkeypatch.setattr(launcher, "_port_is_unused", lambda: True)
    monkeypatch.setattr(
        launcher, "_build_adapter", lambda *_args: pytest.fail("check constructed adapter")
    )
    monkeypatch.setattr(
        launcher.ProcessLease, "acquire", lambda _self: pytest.fail("check acquired lease")
    )

    assert launcher.main(["--check"]) == 0
    assert imports_and_store == ["adapter", "store"]
    assert json.loads(capsys.readouterr().out) == {
        "event": "checked",
        "host": launcher.HOST,
        "port": launcher.PORT,
        "profile": "default",
        "store_id": "hermes-state-v1:" + "b" * 32,
    }
