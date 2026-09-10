"""Authenticated real HTTP ingress, profile separation and zero execution."""
import secrets
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from passive_history_ingress import MAX_REQUEST_BYTES


@pytest.mark.asyncio
async def test_authenticated_profile_ingress_and_canonical_readback(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    homes = {name: root / "profiles" / name for name in ("alpha", "beta")}
    keys = {name: secrets.token_hex(24) for name in homes}
    for name, home in homes.items():
        home.mkdir(parents=True)
        (home / ".env").write_text(f"API_SERVER_KEY={keys[name]}\n", encoding="utf-8")
        (home / "config.yaml").write_text("model:\n  default: test-model\n", encoding="utf-8")
    stores = {name: SessionDB(home / "state.db") for name, home in homes.items()}
    for db in stores.values():
        db.create_session("same-id", source="test")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": keys["alpha"]}))
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))

    def forbidden(*args, **kwargs):
        raise AssertionError("passive ingress must not execute")

    monkeypatch.setattr(adapter, "_create_agent", forbidden)
    monkeypatch.setattr("run_agent.AIAgent.__init__", forbidden)
    monkeypatch.setattr("model_tools.handle_function_call", forbidden)
    monkeypatch.setattr("tools.approval.submit_pending", forbidden)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, "/p/{profile}" + path, handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    base = "/p/alpha/v1/passive-history"
    auth = {"Authorization": "Bearer " + keys["alpha"]}
    try:
        assert (await client.post(base + "/attach", json={})).status == 401
        for headers in ({"X-Api-Key": keys["alpha"]}, {"Authorization": "Basic " + keys["alpha"]},
                        {"Authorization": "bearer " + keys["alpha"]}):
            assert (await client.get(base + "/capabilities", headers=headers)).status == 401
        response = await client.get("/p/alpha/v1/capabilities", headers=auth)
        assert (await response.json())["features"]["passive_history"]["origin_adoption"] is False
        response = await client.post(base + "/attach", headers=auth,
                                     json={"tab_id": "tab", "session_id": "same-id"})
        assert response.status == 200
        attachment = await response.json()
        identity = {key: attachment[key] for key in ("tab_id", "session_id", "generation", "attachment_id")}
        payload = {**identity, "event_id": "evt", "origin_turn_id": "origin",
                   "messages": [{"role": "user", "content": "the finalized speech"}]}
        assert stores["alpha"].acquire_session_turn_lease("same-id", "test-worker", wait_seconds=0)
        try:
            response = await client.post(base + "/commit", headers=auth, json=payload)
            assert response.status == 409
            assert await response.json() == {"error": "busy", "retryable": True}
        finally:
            stores["alpha"].release_session_turn_lease("same-id", "test-worker")
        response = await client.post(base + "/commit", headers=auth, json=payload)
        assert response.status == 200
        saved = await response.json()
        assert saved["status"] == "saved"
        response = await client.post(base + "/commit", headers=auth, json={
            **payload, "messages": [{"role": "user", "content": "changed payload"}]})
        assert response.status == 409
        assert (await response.json())["error"] == "event_conflict"
        response = await client.post(base + "/attach", headers=auth,
                                     json={"tab_id": "missing", "session_id": "missing"})
        assert response.status == 404
        stores["alpha"].create_session("closed", source="test")
        stores["alpha"].end_session("closed", "closed")
        response = await client.post(base + "/attach", headers=auth,
                                     json={"tab_id": "closed", "session_id": "closed"})
        assert response.status == 409
        assert (await response.json())["error"] == "target_unavailable"
        # Lost response: a read-only retry finds exactly the original receipt.
        response = await client.post(base + "/reconcile", headers=auth,
                                     json={"session_id": "same-id", "event_id": "evt"})
        assert (await response.json())["receipt"]["message_ids"] == saved["receipt"]["message_ids"]
        response = await client.get("/p/alpha/api/sessions/same-id/messages", headers=auth)
        assert [m["content"] for m in (await response.json())["data"]] == ["the finalized speech"]
        foreign = base.replace("alpha", "beta")
        assert (await client.post(foreign + "/commit", headers=auth, json=payload)).status == 401
        beta_auth = {"Authorization": "Bearer " + keys["beta"]}
        assert (await client.post(foreign + "/commit", headers=beta_auth, json=payload)).status == 409
        assert stores["beta"].get_messages("same-id") == []
        for bad in ({**payload, "callback_url": "https://example.invalid"},
                    {**payload, "messages": [{"role": "system", "content": "forbidden"}]}):
            assert (await client.post(base + "/commit", headers=auth, json=bad)).status == 400
        assert (await client.post(base + "/commit", headers=auth,
                                 data=b"x" * (MAX_REQUEST_BYTES + 1))).status == 413
        stores["alpha"].delete_session("same-id")
        response = await client.post(base + "/reconcile", headers=auth,
                                     json={"session_id": "same-id", "event_id": "evt"})
        assert response.status == 410
    finally:
        await client.close()
        adapter._close_cached_session_dbs()
        for db in stores.values():
            db.close()
