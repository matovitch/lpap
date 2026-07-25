"""Push notifications for training lifecycle events (Pushover)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class NotifyError(RuntimeError):
    """Raised when a notification backend rejects or cannot send a message."""


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def notify_on_finished_enabled() -> bool:
    """True when ``LPAP_NOTIFY_ON_FINISHED`` is set (via secrets inject)."""
    return env_flag("LPAP_NOTIFY_ON_FINISHED")


def pushover_credentials(
    *,
    token: str | None = None,
    user: str | None = None,
) -> tuple[str, str]:
    resolved_token = (token or os.environ.get("PUSHOVER_TOKEN", "")).strip()
    resolved_user = (user or os.environ.get("PUSHOVER_USER", "")).strip()
    if not resolved_token or not resolved_user:
        raise NotifyError(
            "Pushover credentials missing; set PUSHOVER_TOKEN and PUSHOVER_USER "
            "via configs/secrets.toml + molab-inject-secrets.sh (or export locally)"
        )
    return resolved_token, resolved_user


def send_pushover(
    message: str,
    *,
    title: str | None = None,
    priority: int = 1,
    sound: str = "pushover",
    token: str | None = None,
    user: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Send a Pushover message. Returns the parsed API JSON on success."""
    resolved_token, resolved_user = pushover_credentials(token=token, user=user)
    text = message.strip()
    if not text:
        raise ValueError("message must be non-empty")
    payload: dict[str, str] = {
        "token": resolved_token,
        "user": resolved_user,
        "message": text,
        "priority": str(priority),
        "sound": sound,
    }
    if title is not None and title.strip():
        payload["title"] = title.strip()
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.pushover.net/1/messages.json",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise NotifyError(f"Pushover HTTP {exc.code}: {raw[:300]}") from exc
    except urllib.error.URLError as exc:
        raise NotifyError(f"Pushover request failed: {exc}") from exc

    data = json.loads(raw)
    if status != 200 or data.get("status") != 1:
        raise NotifyError(f"Pushover rejected message: {data}")
    return data


def notify_training_finished(
    *,
    run_id: str,
    step: int | None = None,
    total_steps: int | None = None,
    best_metric: float | None = None,
    status: str = "finished",
    title: str | None = None,
    priority: int = 1,
) -> dict[str, Any]:
    """Format and send a training-finished Pushover notification."""
    parts = [f"run={run_id}", f"status={status}"]
    if step is not None and total_steps is not None:
        parts.append(f"step={step}/{total_steps}")
    elif step is not None:
        parts.append(f"step={step}")
    if best_metric is not None:
        parts.append(f"best={best_metric:.5g}")
    resolved_title = title or f"LPAP {status}"
    return send_pushover(
        " · ".join(parts),
        title=resolved_title,
        priority=priority,
    )


__all__ = [
    "NotifyError",
    "env_flag",
    "notify_on_finished_enabled",
    "notify_training_finished",
    "pushover_credentials",
    "send_pushover",
]
