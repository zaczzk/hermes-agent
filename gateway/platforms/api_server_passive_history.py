"""Thin bearer-authenticated HTTP adapter over the passive-history service."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from functools import partial

from aiohttp import web

from passive_history_ingress import (
    IngressError, MAX_REQUEST_BYTES, PassiveHistoryIngress, capabilities, error_response,
)


def http_routes(adapter):
    return [("GET", "/v1/passive-history/capabilities", partial(handle, adapter, "capabilities")),
            *(("POST", f"/v1/passive-history/{operation}", partial(handle, adapter, operation))
              for operation in capabilities()["operations"])]


def initialize(adapter):
    adapter._passive_history_ingress = PassiveHistoryIngress()


async def handle(adapter, operation, request):
    # Unlike legacy manually-wired test routes, passive authority never has a no-key mode.
    if not adapter._expected_api_key():
        return adapter._auth_failed_response()
    auth_error = adapter._check_auth(request)
    if auth_error is not None:
        return auth_error
    if operation == "capabilities":
        return web.json_response(capabilities())
    try:
        raw = bytearray()
        async for chunk in request.content.iter_chunked(4096):
            if len(raw) + len(chunk) > MAX_REQUEST_BYTES:
                raise IngressError("payload_too_large", 413)
            raw.extend(chunk)
        body = json.loads(raw)
        db = await adapter._ensure_session_db_async()
        if db is None:
            raise IngressError("store_unavailable", 503)
        from hermes_cli.profiles import get_active_profile_name
        from gateway.platforms.api_server import _api_request_profile
        profile = _api_request_profile.get() or get_active_profile_name()
        result = await asyncio.to_thread(
            adapter._passive_history_ingress.dispatch, db, profile=profile,
            principal="gateway:" + request.headers["Authorization"][7:].strip(),
            operation=operation, body=body)
        return web.json_response(result)
    except (ValueError, TypeError, RuntimeError, sqlite3.Error) as exc:
        payload, status = error_response(exc)
        return web.json_response(payload, status=status)
