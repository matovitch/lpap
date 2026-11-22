"""Marimo server file API helpers (kernel-safe; uses curl, not the scratch exec).

``urllib`` against this API can return 403; the other molab skill scripts
already go through curl with a Bearer token, so this module does the same.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class MolabFileError(RuntimeError):
    """A file-API request failed or returned a truncated body."""


def molab_base_url() -> str:
    url = (os.environ.get("MOLAB_URL") or "").strip()
    if not url:
        raise MolabFileError("MOLAB_URL is not set")
    return url.rstrip("/")


def molab_token() -> str:
    token = (os.environ.get("MOLAB_TOKEN") or os.environ.get("MARIMO_TOKEN") or "").strip()
    if not token:
        raise MolabFileError("MOLAB_TOKEN or MARIMO_TOKEN is not set")
    return token


def curl_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int = 60,
    dest: Path | None = None,
) -> dict[str, Any] | Path:
    """POST JSON to a molab API endpoint.

    If ``dest`` is set, the raw body is written there and the path is returned
    (for large file_details responses). Otherwise the body is parsed as JSON.
    """
    base = molab_base_url()
    token = molab_token()
    url = f"{base}{endpoint}"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        payload_path = handle.name
    out_path = dest if dest is not None else Path(tempfile.mkstemp(suffix=".json")[1])
    try:
        result = subprocess.run(
            [
                "curl",
                "-sS",
                "-o",
                str(out_path),
                "-w",
                "%{http_code}",
                "-m",
                str(timeout),
                "-X",
                "POST",
                url,
                "-H",
                f"Authorization: Bearer {token}",
                "-H",
                "Content-Type: application/json",
                "--data",
                f"@{payload_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        code_text = (result.stdout or "").strip()
        try:
            status = int(code_text)
        except ValueError:
            status = 0
        if result.returncode != 0 or status < 200 or status >= 300:
            snippet = ""
            try:
                snippet = out_path.read_text(encoding="utf-8")[:240]
            except OSError:
                snippet = (result.stderr or "")[:240]
            raise MolabFileError(
                f"POST {endpoint} failed (curl={result.returncode} http={status or '?'}): {snippet}"
            )
        if dest is not None:
            return dest
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        Path(payload_path).unlink(missing_ok=True)
        if dest is None:
            out_path.unlink(missing_ok=True)


def list_files(path: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    data = curl_json("/api/files/list_files", {"path": path}, timeout=timeout)
    if not isinstance(data, dict):
        raise MolabFileError(f"list_files returned {type(data).__name__}")
    files = data.get("files") or []
    if not isinstance(files, list):
        raise MolabFileError(f"list_files.files is {type(files).__name__}")
    return files


def decode_file_details(response: dict[str, Any]) -> bytes:
    """Return file bytes from a ``file_details`` JSON object.

    Raises ``FileNotFoundError`` when the server reports a missing path, and
    ``MolabFileError`` when the payload is too large or truncated.
    """
    if "contents" not in response:
        detail = response.get("detail")
        text = str(detail) if detail is not None else json.dumps(response)[:240]
        if "No such file" in text or "Errno 2" in text:
            raise FileNotFoundError(text)
        raise MolabFileError(f"file_details missing contents: {text}")
    if response.get("isTooLarge"):
        info = response.get("file") or {}
        raise MolabFileError(
            f"file_details isTooLarge path={info.get('path')} size={info.get('size')}"
        )
    contents = response["contents"]
    if response.get("isBase64"):
        if not isinstance(contents, str):
            raise MolabFileError("base64 contents must be a string")
        data = base64.b64decode(contents, validate=False)
    elif isinstance(contents, str):
        data = contents.encode("utf-8")
    elif isinstance(contents, bytes):
        data = contents
    else:
        raise MolabFileError(f"unsupported contents type {type(contents).__name__}")
    declared = (response.get("file") or {}).get("size")
    if declared is not None and len(data) != int(declared):
        raise MolabFileError(
            f"truncated file_details payload: got {len(data)} bytes, "
            f"server declared {declared}"
        )
    return data


def download_file(remote_path: str, dest: Path, *, timeout: int = 300) -> Path:
    """Download ``remote_path`` through file_details into ``dest``.

    Suitable for logs and other modest files. Do not use this for 100+ MB
    checkpoints: the API returns the whole file as base64 JSON, which is
    slow and has already been truncated by jq/timeouts.
    """
    raw = Path(tempfile.mkstemp(suffix=".file_details.json")[1])
    try:
        curl_json(
            "/api/files/file_details",
            {"path": remote_path},
            timeout=timeout,
            dest=raw,
        )
        response = json.loads(raw.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise MolabFileError("file_details response is not a JSON object")
        data = decode_file_details(response)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
    finally:
        raw.unlink(missing_ok=True)
