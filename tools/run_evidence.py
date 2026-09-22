"""Run-scoped capture of immutable bytes produced by local file tools."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_ITEMS = 8
MAX_ITEM_BYTES = 1024 * 1024
MAX_RUN_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RunEvidenceScope:
    owner_scope: str
    api_run_id: str
    canonical_parent_session_id: str
    child_session_id: str
    origin_turn_id: str
    correlation_id: str
    request_sha256: str

    def __post_init__(self) -> None:
        hashes = (self.owner_scope, self.request_sha256)
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ValueError("run evidence scope digest is invalid")
        bounded = (
            (self.api_run_id, 128),
            (self.canonical_parent_session_id, 256),
            (self.child_session_id, 256),
            (self.origin_turn_id, 128),
            (self.correlation_id, 128),
        )
        if any(
            not isinstance(value, str)
            or not value
            or len(value) > limit
            or any(ord(char) < 33 or ord(char) > 126 for char in value)
            for value, limit in bounded
        ):
            raise ValueError("run evidence scope identity is invalid")


@dataclass(frozen=True, slots=True)
class _WriteReceipt:
    item_id: str
    path: str
    identity: tuple[int, int, int, int]
    size: int
    sha256: str
    tool_call_id: str


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(getattr(value, "st_mtime_ns", int(value.st_mtime * 1_000_000_000))),
    )


def _is_reparse(value: os.stat_result) -> bool:
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(int(getattr(value, "st_file_attributes", 0)) & flag)


def _stable_read(path: str) -> tuple[tuple[int, int, int, int], bytes] | None:
    """Read one regular file while refusing symlink/reparse and identity changes."""
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or _is_reparse(before):
            return None
        if before.st_size > MAX_ITEM_BYTES:
            return None
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                return None
            chunks: list[bytes] = []
            remaining = MAX_ITEM_BYTES + 1
            while remaining:
                chunk = os.read(fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        final_path = os.lstat(path)
        if (
            len(data) > MAX_ITEM_BYTES
            or _identity(opened) != _identity(after)
            or opened.st_size != len(data)
            or _identity(after) != _identity(final_path)
            or _is_reparse(final_path)
        ):
            return None
        return _identity(after), data
    except (OSError, ValueError):
        return None


def _safe_text(data: bytes) -> bool:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return "\x00" not in text and all(
        char in "\r\n\t" or not unicodedata.category(char).startswith("C")
        for char in text
    )


class RunEvidenceSession:
    """One explicitly propagated linked-run capture; thread-safe and single-seal."""

    def __init__(self, scope: RunEvidenceScope):
        self.scope = scope
        self._receipts: dict[str, _WriteReceipt] = {}
        self._omitted = 0
        self._untrusted_calls: set[str] = set()
        self._sealed: dict[str, Any] | None = None
        self._sealing = False
        self._late_omitted = 0
        self._lock = threading.Lock()

    def note_tool(self, tool_name: str, tool_call_id: str | None) -> None:
        if tool_name not in {"terminal", "execute_code"}:
            return
        key = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else uuid.uuid4().hex
        with self._lock:
            if self._sealing:
                self._late_omitted += 1
            elif self._sealed is None and key not in self._untrusted_calls:
                self._untrusted_calls.add(key)
                self._omitted += 1

    def note_write(self, resolved: str, *, tool_call_id: str | None, file_ops: Any) -> None:
        """Capture a post-write proof only for the local backend."""
        with self._lock:
            if self._sealed is not None:
                return
            if self._sealing:
                self._late_omitted += 1
                return
        if getattr(getattr(file_ops, "env", None), "is_local", False) is not True:
            with self._lock:
                if self._sealed is None:
                    self._omitted += 1
            return
        try:
            canonical = os.path.abspath(os.fspath(Path(resolved)))
        except (OSError, TypeError, ValueError):
            canonical = ""
        captured = _stable_read(canonical) if canonical else None
        if captured is None:
            with self._lock:
                if self._sealed is None:
                    self._omitted += 1
            return
        identity, data = captured
        receipt = _WriteReceipt(
            item_id="item_" + uuid.uuid4().hex[:24],
            path=canonical,
            identity=identity,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else "",
        )
        with self._lock:
            if self._sealed is not None:
                return
            if self._sealing:
                self._late_omitted += 1
                return
            if canonical not in self._receipts and len(self._receipts) >= MAX_ITEMS:
                self._omitted += 1
                return
            self._receipts[canonical] = receipt

    def seal(self, store) -> dict[str, Any]:
        with self._lock:
            if self._sealed is not None:
                return dict(self._sealed)
            if self._sealing:
                raise RuntimeError("run evidence seal is already in progress")
            self._sealing = True
            receipts = list(self._receipts.values())
            omitted = self._omitted
        items: list[dict[str, Any]] = []
        total = 0
        for receipt in receipts:
            captured = _stable_read(receipt.path)
            if captured is None:
                omitted += 1
                continue
            identity, data = captured
            digest = hashlib.sha256(data).hexdigest()
            if (
                identity != receipt.identity
                or len(data) != receipt.size
                or digest != receipt.sha256
                or total + len(data) > MAX_RUN_BYTES
            ):
                omitted += 1
                continue
            total += len(data)
            items.append({
                "item_id": receipt.item_id,
                "size": len(data),
                "sha256": digest,
                "bytes": data,
                "text_utf8": _safe_text(data),
            })
        with self._lock:
            omitted += self._late_omitted
            state = "complete" if items and omitted == 0 else "partial" if items else "unavailable"
            try:
                descriptor = store.seal_evidence(
                    self.scope.owner_scope,
                    self.scope.api_run_id,
                    state=state,
                    items=items,
                    omitted_count=omitted,
                )
            except Exception:
                self._sealing = False
                raise
            self._sealed = dict(descriptor)
            self._receipts.clear()
        return dict(descriptor)
