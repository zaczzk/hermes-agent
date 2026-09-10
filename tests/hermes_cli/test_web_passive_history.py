"""Dashboard's real session-token middleware and profile-scoped ingress routes."""
from starlette.testclient import TestClient

from hermes_state import SessionDB
from passive_history_ingress import PassiveHistoryIngress


def test_dashboard_auth_restart_and_profile_isolation(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.web_routers import passive_history as routes

    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    paths = {name: root / "profiles" / name / "state.db" for name in ("alpha", "beta")}
    for path in paths.values():
        path.parent.mkdir(parents=True)
        db = SessionDB(path)
        db.create_session("same-id", source="test")
        db.close()
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "dashboard-test-token")
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(routes, "service", PassiveHistoryIngress())

    def forbidden(*args, **kwargs):
        raise AssertionError("passive ingress must not execute")

    monkeypatch.setattr("run_agent.AIAgent.__init__", forbidden)
    monkeypatch.setattr("model_tools.handle_function_call", forbidden)
    monkeypatch.setattr("tools.approval.submit_pending", forbidden)
    client = TestClient(web_server.app)
    auth = {web_server._SESSION_HEADER_NAME: "dashboard-test-token"}
    prefix = "/api/passive-history"
    assert client.post(prefix + "/attach?profile=alpha", json={}).status_code == 401
    assert client.get(prefix + "/capabilities", headers=auth).json()["passive_only"] is True
    attached = client.post(prefix + "/attach?profile=alpha", headers=auth,
                           json={"tab_id": "tab", "session_id": "same-id"})
    assert attached.status_code == 200
    identity = {key: attached.json()[key] for key in ("tab_id", "session_id", "generation", "attachment_id")}
    payload = {**identity, "event_id": "event", "origin_turn_id": "origin",
               "messages": [{"role": "user", "content": "speech saved by dashboard"}]}
    saved = client.post(prefix + "/commit?profile=alpha", headers=auth, json=payload)
    assert saved.status_code == 200
    assert client.post(prefix + "/commit?profile=beta", headers=auth, json=payload).status_code == 409
    monkeypatch.setattr(routes, "service", PassiveHistoryIngress())
    # A reconnect can reconcile saved content, but cannot revive the old attachment.
    assert client.post(prefix + "/commit?profile=alpha", headers=auth, json=payload).status_code == 409
    response = client.post(prefix + "/reconcile?profile=alpha", headers=auth,
                           json={"session_id": "same-id", "event_id": "event"})
    assert response.json()["receipt"]["message_ids"] == saved.json()["receipt"]["message_ids"]
    for name, path in paths.items():
        db = SessionDB(path)
        try:
            contents = [row["content"] for row in db.get_messages("same-id")]
            assert contents == (["speech saved by dashboard"] if name == "alpha" else [])
            assert db.get_session("same-id")["api_call_count"] == 0
        finally:
            db.close()
