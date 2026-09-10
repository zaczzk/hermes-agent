"""Dashboard session-auth/profile adapter over the shared passive-history service."""
from __future__ import annotations

import asyncio
import json
import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hermes_cli.web_server_sessions import _open_session_db_for_profile
from hermes_cli.web_server_cron import _cron_default_profile, _cron_profile_home
from hermes_state_registry import release_or_close
from passive_history_ingress import (
    IngressError, MAX_REQUEST_BYTES, PassiveHistoryIngress, capabilities, error_response,
)

router = APIRouter()
service = PassiveHistoryIngress()


def _principal(request):
    # The existing middleware verifies OAuth identity or the process's session token.
    session = getattr(request.state, "session", None)
    if session is not None:
        return f"dashboard:{session.provider}:{session.org_id}:{session.user_id}"
    from hermes_cli import web_server
    if web_server._has_valid_session_token(request):
        return "dashboard:" + web_server._SESSION_TOKEN
    raise IngressError("denied", 403)


@router.get("/api/passive-history/capabilities")
async def get_capabilities(request: Request):
    try:
        _principal(request)
        return capabilities()
    except IngressError as exc:
        payload, status = error_response(exc)
        return JSONResponse(payload, status_code=status)


def _dispatch(profile, principal, operation, body):
    resolved = _cron_profile_home(profile)[0] if profile else _cron_default_profile()
    db = _open_session_db_for_profile(profile, read_only=operation in {"snapshot", "reconcile"})
    try:
        return service.dispatch(db, profile=resolved, principal=principal, operation=operation, body=body)
    finally:
        release_or_close(db)


@router.post("/api/passive-history/{operation}")
async def passive_operation(operation: str, request: Request, profile: str | None = None):
    try:
        principal = _principal(request)
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_REQUEST_BYTES:
                raise IngressError("payload_too_large", 413)
            raw.extend(chunk)
        result = await asyncio.to_thread(_dispatch, profile, principal, operation, json.loads(raw))
        return JSONResponse(result)
    except (ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
        payload, status = error_response(exc)
        return JSONResponse(payload, status_code=status)
