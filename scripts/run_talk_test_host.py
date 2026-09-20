#!/usr/bin/env python3
"""Run one isolated Talk API listener against an existing Hermes profile.

This is deliberately not a gateway launcher: it owns one real
``APIServerAdapter`` on 127.0.0.1:8643 and therefore starts no messaging
adapters, cron scheduler, or gateway runner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import socket
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import psutil


HOST = "127.0.0.1"
PORT = 8643
KEY_ENV = "TALK_API_SERVER_KEY"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
_LEASE_VERSION = 1
_SHUTDOWN_GRACE_SECONDS = 15.0
_RUNTIME_DIR = "talk-test-host"


class LaunchRefused(RuntimeError):
    """A fail-closed precondition prevented the test host from starting."""


def _install_source_root() -> None:
    root = str(SOURCE_ROOT)
    if not sys.path or sys.path[0] != root:
        sys.path.insert(0, root)


def _require_source_module(module: Any, label: str) -> None:
    raw = getattr(module, "__file__", None)
    try:
        Path(raw).resolve(strict=True).relative_to(SOURCE_ROOT)
    except (TypeError, OSError, RuntimeError, ValueError) as exc:
        raise LaunchRefused(f"{label} did not load from the isolated host checkout") from exc


def _current_process_identity() -> tuple[int, float]:
    process = psutil.Process(os.getpid())
    return process.pid, process.create_time()


def _same_live_process(pid: object, created_at: object) -> bool | None:
    """True for the same live process, False for a proved stale PID, None if uncertain."""
    if type(pid) is not int or not isinstance(created_at, (int, float)):
        return None
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - float(created_at)) < 0.01
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, OSError):
        return None


@dataclass(slots=True)
class ProcessLease:
    path: Path
    profile: str
    profile_home: Path
    store_id: str
    token: str = ""
    pid: int = 0
    created_at: float = 0.0

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pid, self.created_at = _current_process_identity()
        self.token = secrets.token_hex(16)
        record = {
            "version": _LEASE_VERSION,
            "owner_token": self.token,
            "pid": self.pid,
            "process_created_at": self.created_at,
            "host": HOST,
            "port": PORT,
            "profile": self.profile,
            "profile_home": str(self.profile_home),
            "store_id": self.store_id,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                existing = self._read()
                same = _same_live_process(
                    existing.get("pid"), existing.get("process_created_at")
                )
                if same is not False:
                    raise LaunchRefused("test-host lease is owned or cannot be verified")
                with suppress(FileNotFoundError):
                    self.path.unlink()
                continue
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                with suppress(OSError):
                    self.path.chmod(0o600)
                return
            except BaseException:
                with suppress(OSError):
                    os.close(fd)
                with suppress(FileNotFoundError):
                    self.path.unlink()
                raise
        raise LaunchRefused("test-host lease changed during stale-owner recovery")

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LaunchRefused("test-host lease is unreadable; refusing unsafe takeover") from exc
        if not isinstance(value, dict):
            raise LaunchRefused("test-host lease is invalid; refusing unsafe takeover")
        return value

    def release(self) -> bool:
        if not self.token:
            return False
        try:
            existing = self._read()
        except LaunchRefused:
            return False
        if (
            existing.get("owner_token") != self.token
            or existing.get("pid") != self.pid
            or existing.get("process_created_at") != self.created_at
        ):
            return False
        try:
            self.path.unlink()
            self.token = ""
            return True
        except FileNotFoundError:
            self.token = ""
            return True
        except OSError:
            return False


def _default_lease_path(profile_home: Path) -> Path:
    digest = hashlib.sha256(str(profile_home.resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"hermes-talk-test-host-{digest}.json"


def _run_store_path(profile_home: Path) -> Path:
    return profile_home / _RUNTIME_DIR / "runs_idempotency.db"


def _port_is_unused() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((HOST, PORT))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _resolve_profile(profile: str) -> tuple[str, Path]:
    normalized = profile.strip().lower()
    if not normalized:
        raise LaunchRefused("profile is required")
    from hermes_cli import profiles

    _require_source_module(profiles, "profile resolver")

    try:
        home = Path(profiles.resolve_profile_env(normalized)).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LaunchRefused("profile is unavailable") from exc
    if not home.is_dir() or not (home / "state.db").is_file():
        raise LaunchRefused("profile has no existing canonical state store")
    return normalized, home


def _resolve_key(profile_home: Path) -> str:
    """Resolve the shared process/profile secret without writing or displaying it."""
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.auth import has_usable_secret

    if KEY_ENV in os.environ:
        key = os.environ.get(KEY_ENV, "").strip()
    else:
        key = str(build_profile_secret_scope(profile_home).get("API_SERVER_KEY") or "").strip()
        if not key:
            key = os.environ.get("API_SERVER_KEY", "").strip()
    if not has_usable_secret(key, min_length=16):
        raise LaunchRefused(f"{KEY_ENV} or the profile API_SERVER_KEY must be a strong secret")
    return key


def _canonical_store_id(profile_home: Path) -> str:
    import hermes_state
    from hermes_state_registry import acquire, release_or_close
    from hermes_state_store_identity import get_store_id

    _require_source_module(hermes_state, "SessionDB")
    db = acquire(profile_home / "state.db")
    try:
        return get_store_id(db)
    finally:
        release_or_close(db)


def _adapter_class():
    from gateway.platforms import api_server

    _require_source_module(api_server, "APIServerAdapter")
    return api_server.APIServerAdapter


def _build_adapter(key: str, run_store_path: Path):
    from gateway.config import PlatformConfig

    return _adapter_class()(
        PlatformConfig(enabled=True, extra={
            "host": HOST,
            "port": PORT,
            "key": key,
            "runs_idempotency_path": str(run_store_path),
        })
    )


async def _stop_adapter(adapter: Any) -> None:
    try:
        adapter.interrupt_active_runs("isolated Talk test host shutdown")
        deadline = asyncio.get_running_loop().time() + _SHUTDOWN_GRACE_SECONDS
        while adapter.active_agent_work_count() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
        await adapter.cancel_background_tasks()
    finally:
        await adapter.disconnect()


async def _serve(
    adapter: Any,
    lease: ProcessLease,
    *,
    wait: Callable[[asyncio.Event], Any] | None = None,
) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass
    connected = False
    try:
        connected = bool(await adapter.connect())
        if not connected:
            raise LaunchRefused("API adapter refused startup")
        print(json.dumps({
            "event": "ready",
            "pid": lease.pid,
            "profile": lease.profile,
            "host": HOST,
            "port": PORT,
            "store_id": lease.store_id,
        }, sort_keys=True), flush=True)
        if wait is None:
            await stop.wait()
        else:
            await wait(stop)
    finally:
        try:
            if connected:
                await _stop_adapter(adapter)
            else:
                await adapter.disconnect()
        finally:
            lease.release()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated Talk host on 127.0.0.1:8643")
    parser.add_argument("--profile", default="default", help="existing canonical Hermes profile")
    parser.add_argument("--lease-file", type=Path, help="override the temporary ownership lease")
    parser.add_argument(
        "--check", action="store_true", help="verify source, profile, key, store, and port without binding"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    adapter = None
    lease = None
    try:
        _install_source_root()
        profile, profile_home = _resolve_profile(args.profile)
        os.environ["HERMES_HOME"] = str(profile_home)
        os.environ["GATEWAY_MULTIPLEX_PROFILES"] = "false"
        key = _resolve_key(profile_home)
        _adapter_class()
        store_id = _canonical_store_id(profile_home)
        if not _port_is_unused():
            raise LaunchRefused(f"{HOST}:{PORT} is already in use")
        if args.check:
            print(json.dumps({
                "event": "checked",
                "profile": profile,
                "host": HOST,
                "port": PORT,
                "store_id": store_id,
            }, sort_keys=True), flush=True)
            return 0
        run_store_path = _run_store_path(profile_home)
        run_store_path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            run_store_path.parent.chmod(0o700)
        lease = ProcessLease(
            args.lease_file or _default_lease_path(profile_home),
            profile,
            profile_home,
            store_id,
        )
        lease.acquire()
        adapter = _build_adapter(key, run_store_path)
        asyncio.run(_serve(adapter, lease))
        return 0
    except (LaunchRefused, KeyboardInterrupt) as exc:
        if lease is not None:
            lease.release()
        message = str(exc) if isinstance(exc, LaunchRefused) else "interrupted"
        print(json.dumps({"event": "refused", "reason": message}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        if adapter is not None:
            with suppress(Exception):
                asyncio.run(adapter.disconnect())
        if lease is not None:
            lease.release()
        print(json.dumps({
            "event": "failed", "reason": f"unexpected {type(exc).__name__}"
        }, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
