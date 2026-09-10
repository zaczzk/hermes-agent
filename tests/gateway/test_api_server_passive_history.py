"""Authenticated real HTTP ingress, profile separation and zero execution."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from hermes_state import SessionDB
from passive_history_ingress import MAX_REQUEST_BYTES


@pytest.mark.asyncio
async def test_authenticated_profile_ingress_and_canonical_readback(tmp_path, monkeypatch):
    stores = {name: SessionDB(tmp_path / f"{name}.db") for name in ("alpha", "beta")}
    keys = {name: f"{name}-test-key-with-at-least-sixteen-characters" for name in stores}
    for db in stores.values():
        db.create_session("same-id", source="test")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": keys["alpha"]}))
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve",
                        lambda **_: [(name, tmp_path / name) for name in stores])
    monkeypatch.setattr(adapter, "_profile_scope", lambda _: nullcontext())
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: keys[_api_request_profile.get() or "alpha"])

    async def profile_db():
        return stores[_api_request_profile.get() or "alpha"]

    monkeypatch.setattr(adapter, "_ensure_session_db_async", profile_db)

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
        response = await client.get("/p/alpha/v1/capabilities", headers=auth)
        assert (await response.json())["features"]["passive_history"]["origin_adoption"] is False
        response = await client.post(base + "/attach", headers=auth,
                                     json={"tab_id": "tab", "session_id": "same-id"})
        assert response.status == 200
        attachment = await response.json()
        identity = {key: attachment[key] for key in ("tab_id", "session_id", "generation", "attachment_id")}
        payload = {**identity, "event_id": "evt", "origin_turn_id": "origin",
                   "messages": [{"role": "user", "content": "the finalized speech"}]}
        response = await client.post(base + "/commit", headers=auth, json=payload)
        assert response.status == 200
        saved = await response.json()
        assert saved["status"] == "saved"
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
        for db in stores.values():
            db.close()
