"""Detached Python workers on a notebook host (pidfile + log).

Lives under repo ``molab/`` (synced to ``/marimo/molab/``); not part of ``lpap``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_STEP_RE = re.compile(r"(?:^|\s)step=(\d+)\b")


class BackgroundWorkerError(RuntimeError):
    """Raised when a background worker cannot be spawned or inspected."""


def read_pid(pid_path: str | Path) -> int | None:
    path = Path(pid_path)
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return int(raw)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Linux zombies still satisfy kill(pid, 0); treat them as dead so relaunch
    # is not blocked after a crashed worker.
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            # Format: pid (comm) state ... — comm may contain spaces/parens.
            state = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[-1].strip()
            if state[:1] == "Z":
                return False
        except OSError:
            pass
    return True


def bg_worker_status(pid_path: str | Path) -> dict[str, Any]:
    path = Path(pid_path)
    pid = read_pid(path)
    alive = False if pid is None else process_alive(pid)
    return {
        "pid_path": str(path),
        "pid": pid,
        "alive": alive,
    }


def refuse_if_alive(pid_path: str | Path) -> None:
    status = bg_worker_status(pid_path)
    if status["alive"]:
        raise BackgroundWorkerError(
            f"background worker still alive (pid={status['pid']}) "
            f"for {status['pid_path']}"
        )


def parse_bg_log_steps(text: str) -> list[int]:
    """Extract ``step=N`` values in log order (duplicates kept)."""
    return [int(match.group(1)) for match in _STEP_RE.finditer(text)]


def last_bg_log_step(log_path: str | Path) -> int | None:
    path = Path(log_path)
    if not path.is_file():
        return None
    steps = parse_bg_log_steps(path.read_text(encoding="utf-8", errors="replace"))
    return steps[-1] if steps else None


def bg_log_tail(log_path: str | Path, *, lines: int = 12) -> list[str]:
    path = Path(log_path)
    if not path.is_file() or lines <= 0:
        return []
    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return all_lines[-lines:]


def spawn_detached_python(
    script_path: str | Path,
    *,
    cwd: str | Path,
    pid_path: str | Path,
    log_path: str | Path,
    env: Mapping[str, str] | None = None,
    python: str | None = None,
    append_log: bool = False,
) -> dict[str, Any]:
    """Spawn ``python script`` detached; write pidfile; tee stdout/stderr to log.

    Raises ``BackgroundWorkerError`` if a prior pidfile process is still alive.
    """
    script = Path(script_path)
    if not script.is_file():
        raise FileNotFoundError(script)
    pid_file = Path(pid_path)
    log_file = Path(log_path)
    refuse_if_alive(pid_file)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_env = dict(os.environ if env is None else env)
    mode = "a" if append_log else "w"
    # Parent closes after Popen; the child keeps a duplicated fd on Unix.
    with log_file.open(mode, encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [python or sys.executable, str(script)],
            cwd=str(cwd),
            env=resolved_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    return {
        "pid": proc.pid,
        "script_path": str(script),
        "cwd": str(cwd),
        "pid_path": str(pid_file),
        "log_path": str(log_file),
    }


def require_env_keys(
    keys: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a copy of ``env`` (default ``os.environ``) after checking keys."""
    source = os.environ if env is None else env
    missing = [key for key in keys if not str(source.get(key, "")).strip()]
    if missing:
        raise BackgroundWorkerError(
            "missing required env for detached worker: " + ", ".join(missing)
        )
    return dict(source)


__all__ = [
    "BackgroundWorkerError",
    "bg_log_tail",
    "bg_worker_status",
    "last_bg_log_step",
    "parse_bg_log_steps",
    "process_alive",
    "read_pid",
    "refuse_if_alive",
    "require_env_keys",
    "spawn_detached_python",
]
