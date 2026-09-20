from types import SimpleNamespace

import pytest
from starlette.requests import Request

from hermes_cli.dashboard_auth.base import Session, TokenPrincipal
from hermes_cli.dashboard_task_context import resolve_dashboard_task_context
from hermes_state import SessionDB
from hermes_state_store_identity import get_store_id


def _request(*, headers=(), auth_required=True):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/plugins/hermes-talk/tasks",
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers],
            "app": SimpleNamespace(state=SimpleNamespace(auth_required=auth_required)),
        }
    )
    return request


def _profiles(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    homes = {name: root / "profiles" / name for name in ("alpha", "beta")}
    expected = {}
    for name, home in homes.items():
        home.mkdir(parents=True)
        db = SessionDB(home / "state.db")
        try:
            expected[name] = get_store_id(db)
        finally:
            db.close()
    return homes, expected


def test_verified_session_resolves_exact_profile_and_store(tmp_path, monkeypatch):
    homes, expected = _profiles(tmp_path, monkeypatch)
    request = _request()
    request.state.session = Session(
        user_id="owner", email="owner@example.test", display_name="Owner",
        org_id="org", provider="fixture", expires_at=2**31,
        access_token="access", refresh_token="refresh",
    )

    alpha = resolve_dashboard_task_context(request, "alpha")
    beta = resolve_dashboard_task_context(request, "beta")
    alpha_again = resolve_dashboard_task_context(request, "alpha")

    assert alpha.principal_id == beta.principal_id == alpha_again.principal_id
    assert alpha.principal_kind == "verified_subject"
    assert (alpha.profile_name, alpha.profile_home, alpha.store_id) == (
        "alpha", homes["alpha"].resolve(), expected["alpha"]
    )
    assert (beta.profile_name, beta.profile_home, beta.store_id) == (
        "beta", homes["beta"].resolve(), expected["beta"]
    )
    assert alpha.store_id != beta.store_id
    assert alpha_again.store_id == alpha.store_id


def test_context_requires_middleware_verified_identity(tmp_path, monkeypatch):
    _profiles(tmp_path, monkeypatch)
    request = _request()
    with pytest.raises(PermissionError):
        resolve_dashboard_task_context(request, "alpha")

    request.state.token_principal = TokenPrincipal("service", "fixture", ("profile:alpha",))
    with pytest.raises(PermissionError):
        resolve_dashboard_task_context(request, "alpha")
    request.state.token_authenticated = True
    context = resolve_dashboard_task_context(request, "alpha")
    assert context.principal_kind == "verified_service"


def test_legacy_session_token_is_one_shared_operator(tmp_path, monkeypatch):
    from hermes_cli import web_server

    _homes, expected = _profiles(tmp_path, monkeypatch)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "first-dashboard-secret")
    header = (web_server._SESSION_HEADER_NAME, "first-dashboard-secret")
    first = resolve_dashboard_task_context(
        _request(headers=[header], auth_required=False), "alpha"
    )
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "rotated-dashboard-secret")
    rotated_header = (web_server._SESSION_HEADER_NAME, "rotated-dashboard-secret")
    second = resolve_dashboard_task_context(
        _request(headers=[rotated_header], auth_required=False), "alpha"
    )
    with pytest.raises(PermissionError):
        resolve_dashboard_task_context(
            _request(headers=[header], auth_required=False), "alpha"
        )
    beta = resolve_dashboard_task_context(
        _request(headers=[rotated_header], auth_required=False), "beta"
    )
    assert first.principal_kind == "shared_operator"
    assert first.principal_id == second.principal_id
    assert beta.principal_id != first.principal_id
    assert first.store_id == expected["alpha"]
    assert beta.store_id == expected["beta"]
    assert "dashboard-secret" not in first.principal_id

    alpha_path = _homes["alpha"] / "state.db"
    for candidate in (
        alpha_path,
        alpha_path.with_name(alpha_path.name + "-wal"),
        alpha_path.with_name(alpha_path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)
    replacement = SessionDB(alpha_path)
    replacement.close()
    replaced = resolve_dashboard_task_context(
        _request(headers=[rotated_header], auth_required=False), "alpha"
    )
    assert replaced.store_id != first.store_id
    assert replaced.principal_id != first.principal_id


def test_service_token_requires_explicit_profile_grant(tmp_path, monkeypatch):
    _profiles(tmp_path, monkeypatch)
    request = _request()
    request.state.token_authenticated = True
    request.state.token_principal = TokenPrincipal("service", "fixture", ("profile:alpha",))
    assert resolve_dashboard_task_context(request, "alpha").profile_name == "alpha"
    with pytest.raises(PermissionError):
        resolve_dashboard_task_context(request, "beta")


def test_unknown_profile_is_refused_without_creating_it(tmp_path, monkeypatch):
    homes, _expected = _profiles(tmp_path, monkeypatch)
    request = _request()
    request.state.session = Session(
        user_id="owner", email="owner@example.test", display_name="Owner",
        org_id="org", provider="fixture", expires_at=2**31,
        access_token="access", refresh_token="refresh",
    )
    missing = homes["alpha"].parent / "missing"
    with pytest.raises(Exception) as exc:
        resolve_dashboard_task_context(request, "missing")
    assert getattr(exc.value, "status_code", None) == 404
    assert not missing.exists()
