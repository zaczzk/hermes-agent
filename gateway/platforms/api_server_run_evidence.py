"""Authenticated retrieval for immutable linked-run file evidence."""

from __future__ import annotations

import re

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.platforms.api_server_room_grants import _json_error


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def http_routes(self) -> list[tuple[str, str, object]]:
    return [
        ("GET", "/v1/runs/{run_id}/evidence", self._handle_run_evidence),
        (
            "GET",
            "/v1/runs/{run_id}/evidence/{package_id}/{item_id}",
            self._handle_run_evidence_item,
        ),
        (
            "POST",
            "/v1/runs/{run_id}/evidence/{package_id}/ack",
            self._handle_run_evidence_ack,
        ),
    ]


def capabilities(self) -> dict | None:
    store = getattr(self, "_run_idempotency_store", None)
    if store is None or not getattr(store, "durable", False):
        return None
    try:
        from tools.terminal_tool import _session_scope

        if _session_scope().env_type != "local":
            return None
    except Exception:
        return None
    return {
        "version": 1,
        "linked_child_only": True,
        "max_items": 8,
        "max_item_bytes": 1024 * 1024,
        "max_total_bytes": 4 * 1024 * 1024,
        "ack_retention_days": 30,
        "max_retention_days": 90,
    }


def _not_found(_api_server, run_id: str):
    return _json_error(
        _api_server._openai_error,
        f"Run evidence not found: {run_id}",
        code="evidence_not_found",
        status=404,
    )


def _gone(_api_server, run_id: str):
    return _json_error(
        _api_server._openai_error,
        f"Run evidence expired: {run_id}",
        code="evidence_expired",
        status=410,
    )


def _owned_scope(self, request, run_id: str, package_id: str | None, *, _api_server):
    auth = self._check_auth(request)
    if auth:
        return None, auth
    scope = self._run_idempotency_scope(request)
    store = self._run_idempotency_store
    if store.evidence_tombstone(scope, run_id, package_id):
        return None, _gone(_api_server, run_id)
    if not self._request_owns_run(request, run_id):
        return None, _not_found(_api_server, run_id)
    return scope, None


async def handle_manifest(self, request, *, _api_server):
    run_id = request.match_info["run_id"]
    scope, error = _owned_scope(self, request, run_id, None, _api_server=_api_server)
    if error is not None:
        return error
    manifest = self._run_idempotency_store.evidence_manifest(scope, run_id)
    if manifest is None:
        return _json_error(
            _api_server._openai_error,
            "Run evidence is not ready",
            code="evidence_not_ready",
            status=409,
        )
    return web.json_response(manifest)


async def handle_item(self, request, *, _api_server):
    run_id = request.match_info["run_id"]
    package_id = request.match_info["package_id"]
    scope, error = _owned_scope(
        self, request, run_id, package_id, _api_server=_api_server
    )
    if error is not None:
        return error
    item = self._run_idempotency_store.evidence_item(
        scope, run_id, package_id, request.match_info["item_id"]
    )
    if item is None:
        return _not_found(_api_server, run_id)
    return web.Response(
        body=item["bytes"],
        content_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{item["display_name"]}"',
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        },
    )


async def handle_ack(self, request, *, _api_server):
    run_id = request.match_info["run_id"]
    package_id = request.match_info["package_id"]
    scope, error = _owned_scope(
        self, request, run_id, package_id, _api_server=_api_server
    )
    if error is not None:
        return error
    try:
        body = await request.json()
    except Exception:
        body = None
    digest = body.get("manifest_sha256") if isinstance(body, dict) and set(body) == {"manifest_sha256"} else None
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        return _json_error(
            _api_server._openai_error,
            "Evidence acknowledgement is invalid",
            code="invalid_evidence_ack",
            status=400,
        )
    try:
        result = self._run_idempotency_store.acknowledge_evidence(
            scope, run_id, package_id, digest
        )
    except ValueError:
        return _json_error(
            _api_server._openai_error,
            "Evidence manifest changed",
            code="evidence_manifest_conflict",
            status=409,
        )
    return _not_found(_api_server, run_id) if result is None else web.json_response({
        "object": "hermes.run.evidence_ack",
        "run_id": run_id,
        "package_id": package_id,
        **result,
    })

