"""Dashboard session-auth/profile adapter over the shared passive-history service."""
from __future__ import annotations

import asyncio
import json
import sqlite3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hermes_cli.dashboard_task_context import resolve_dashboard_task_context
from hermes_cli.web_server_sessions import _open_session_db_for_profile
from hermes_state_registry import release_or_close
from passive_history_ingress import (
    IngressError, MAX_REQUEST_BYTES, PassiveHistoryIngress, capabilities, error_response,
)

router = APIRouter()
service = PassiveHistoryIngress()


@router.get("/api/passive-history/capabilities")
async def get_capabilities(request: Request, profile: str | None = None):
    try:
        context = await asyncio.to_thread(
            resolve_dashboard_task_context, request, profile=profile
        )
        return {**capabilities(), "store_id": context.store_id}
    except (PermissionError, RuntimeError, sqlite3.Error) as exc:
        if isinstance(exc, PermissionError):
            exc = IngressError("denied", 403)
        elif isinstance(exc, RuntimeError):
            exc = IngressError("store_unavailable", 503)
        payload, status = error_response(exc)
        return JSONResponse(payload, status_code=status)


def _dispatch(context, operation, body):
    db = _open_session_db_for_profile(
        context.profile_name, read_only=operation in {"snapshot", "reconcile"}
    )
    try:
        from hermes_state_store_identity import get_store_id

        if get_store_id(db) != context.store_id:
            raise IngressError("store_unavailable", 503)
        return service.dispatch(
            db,
            profile=context.profile_name,
            principal=context.principal_id,
            operation=operation,
            body=body,
        )
    finally:
        release_or_close(db)


@router.post("/api/passive-history/{operation}")
async def passive_operation(operation: str, request: Request, profile: str | None = None):
    try:
        context = await asyncio.to_thread(
            resolve_dashboard_task_context, request, profile=profile
        )
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_REQUEST_BYTES:
                raise IngressError("payload_too_large", 413)
            raw.extend(chunk)
        result = await asyncio.to_thread(_dispatch, context, operation, json.loads(raw))
        return JSONResponse(result)
    except (PermissionError, ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
        if isinstance(exc, PermissionError):
            exc = IngressError("denied", 403)
        elif isinstance(exc, RuntimeError):
            exc = IngressError("store_unavailable", 503)
        payload, status = error_response(exc)
        return JSONResponse(payload, status_code=status)
