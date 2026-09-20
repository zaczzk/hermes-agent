"""Verified dashboard actor and canonical profile/store context for server plugins."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DashboardTaskContext:
    principal_id: str
    principal_kind: str
    profile_name: str
    profile_home: Path
    store_id: str


def _principal_digest(kind: str, *parts: str) -> str:
    if not kind or any(not isinstance(part, str) for part in parts):
        raise PermissionError("dashboard identity is unavailable")
    encoded = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":")).encode()
    return "dashboard:" + hashlib.sha256(encoded).hexdigest()


def _verified_principal(request) -> tuple[str, str]:
    session = getattr(request.state, "session", None)
    if session is not None:
        if not session.provider or not session.user_id:
            raise PermissionError("dashboard identity is unavailable")
        return (
            _principal_digest("session", session.provider, session.org_id, session.user_id),
            "verified_subject",
        )
    token = getattr(request.state, "token_principal", None)
    if getattr(request.state, "token_authenticated", False) and token is not None:
        if not token.provider or not token.principal:
            raise PermissionError("dashboard identity is unavailable")
        return (
            _principal_digest("token", token.provider, token.principal),
            "verified_service",
        )
    from hermes_cli import web_server

    if not getattr(request.app.state, "auth_required", False) and web_server._has_valid_session_token(
        request
    ):
        return "", "shared_operator"
    raise PermissionError("dashboard identity is unavailable")


def resolve_dashboard_task_context(request, profile=None) -> DashboardTaskContext:
    """Bind a verified request to one served profile and its canonical ``SessionDB``."""
    principal_id, principal_kind = _verified_principal(request)
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_cli.web_server_sessions import _open_session_db_for_profile
    from hermes_state_registry import release_or_close
    from hermes_state_store_identity import get_store_id

    profile_name, profile_home = _cron_profile_home(profile)
    if principal_kind == "verified_service":
        token = request.state.token_principal
        allowed_profiles = {f"profile:{profile_name}", "profile:*"}
        if (
            not isinstance(token.scopes, tuple)
            or any(not isinstance(scope, str) for scope in token.scopes)
            or not allowed_profiles.intersection(token.scopes)
        ):
            raise PermissionError("dashboard token does not grant this profile")
    db = _open_session_db_for_profile(profile_name, read_only=False)
    try:
        store_id = get_store_id(db)
    finally:
        release_or_close(db)
    if principal_kind == "shared_operator":
        principal_id = _principal_digest("shared_operator", profile_name, store_id)
    return DashboardTaskContext(
        principal_id=principal_id,
        principal_kind=principal_kind,
        profile_name=profile_name,
        profile_home=Path(profile_home).resolve(),
        store_id=store_id,
    )
