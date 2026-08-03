"""Sync training artifacts to/from a Hugging Face Storage Bucket.

Designed for the molab summer workflow: upload from the paired kernel with
``HF_TOKEN`` (from ``configs/secrets.toml`` via inject, or export locally),
download locally without auth when the bucket is public. Bucket settings live
in ``configs/storage.toml`` (required under the project root).

Training checkpoints on HF use a dual-slot layout (no in-place overwrite of the
live object):

- ``checkpoints/<stem>.slot0.pt`` / ``.slot1.pt``
- ``checkpoints/<stem>.current.json`` pointer (required for ensure/resume)

Local code still uses ``checkpoints/<stem>.pt``. SQLite logs remain a single
best-effort key under ``training_logs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from lpap.storage import (
    infer_project_root_from_checkpoint,
    load_storage_config,
)


def default_artifacts_bucket(project_root: str | Path) -> str:
    return load_storage_config(project_root).artifacts.bucket


def _require_project_root_for_defaults(
    project_root: str | Path | None,
    *,
    what: str,
) -> str | Path:
    if project_root is None:
        raise ValueError(
            f"{what} requires project_root with configs/storage.toml, "
            "or pass an explicit bucket="
        )
    return project_root


def resolve_hf_token(*, token: str | None = None) -> str | None:
    """Return an HF token from ``token`` or ``HF_TOKEN`` / hub env."""
    if token:
        return token.strip() or None
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def apply_hf_token(*, token: str | None = None) -> str | None:
    """Load a token into ``HF_TOKEN`` if available; return the resolved value."""
    resolved = resolve_hf_token(token=token)
    if resolved:
        os.environ["HF_TOKEN"] = resolved
    return resolved


def bucket_uri(bucket: str, remote_path: str) -> str:
    remote = remote_path.lstrip("/")
    return f"hf://buckets/{bucket}/{remote}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_stem(name: str | Path) -> str:
    """Return stem for ``foo.pt`` → ``foo`` (basename only)."""
    filename = Path(name).name
    if filename.endswith(".pt"):
        return filename[: -len(".pt")]
    return filename


def checkpoint_slot_remote_path(stem: str, slot: int) -> str:
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    return f"checkpoints/{stem}.slot{slot}.pt"


def checkpoint_pointer_remote_path(stem: str) -> str:
    return f"checkpoints/{stem}.current.json"


def is_canonical_checkpoint_remote(remote_path: str) -> bool:
    """True for ``checkpoints/<stem>.pt``, not slots or pointer files."""
    remote = remote_path.lstrip("/")
    if not remote.startswith("checkpoints/"):
        return False
    name = remote[len("checkpoints/") :]
    if not name.endswith(".pt"):
        return False
    if name.endswith(".slot0.pt") or name.endswith(".slot1.pt"):
        return False
    return True


def _resolve_bucket(
    *,
    bucket: str | None,
    project_root: str | Path | None,
    what: str,
) -> str:
    if bucket is not None:
        return bucket
    root = _require_project_root_for_defaults(project_root, what=what)
    return default_artifacts_bucket(root)


def _require_write_token(*, token: str | None = None) -> str:
    resolved = apply_hf_token(token=token)
    if not resolved:
        raise RuntimeError(
            "HF write token not found; set HF_TOKEN "
            "(configs/secrets.toml + molab-inject-secrets.sh, or export locally)"
        )
    return resolved


def _bucket_object_exists(
    bucket: str,
    remote_path: str,
    *,
    token: str | None = None,
) -> bool:
    """Return whether a bucket object exists (``HfFileSystem.exists`` is unreliable)."""
    from huggingface_hub import get_bucket_paths_info

    try:
        infos = list(
            get_bucket_paths_info(
                bucket, [remote_path.lstrip("/")], token=token or True
            )
        )
    except Exception:
        return False
    return bool(infos)


def read_checkpoint_pointer(
    stem: str,
    *,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load ``checkpoints/<stem>.current.json`` from HF.

    Raises ``FileNotFoundError`` if the pointer is missing.
    """
    from huggingface_hub import HfFileSystem

    resolved_bucket = _resolve_bucket(
        bucket=bucket,
        project_root=project_root,
        what="checkpoint pointer",
    )
    resolved = apply_hf_token(token=token)
    remote = checkpoint_pointer_remote_path(stem)
    uri = bucket_uri(resolved_bucket, remote)
    if not _bucket_object_exists(
        resolved_bucket, remote, token=resolved
    ):
        raise FileNotFoundError(
            f"checkpoint pointer not found: {uri} "
            f"(expected dual-slot layout for stem={stem!r})"
        )
    fs = HfFileSystem(token=resolved if resolved else False)
    with fs.open(uri, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint pointer must be a JSON object: {uri}")
    return payload


def upload_files(
    pairs: Sequence[tuple[Path | str, str]],
    *,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
) -> list[str]:
    """Upload ``(local_path, remote_path)`` pairs to a storage bucket.

    Requires a write-capable token (``token=`` or ``HF_TOKEN`` env).
    """
    from huggingface_hub import HfApi

    resolved_bucket = _resolve_bucket(
        bucket=bucket,
        project_root=project_root,
        what="default artifacts bucket",
    )
    _require_write_token(token=token)

    add: list[tuple[str | bytes, str]] = []
    for local, remote in pairs:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError(path)
        add.append((str(path), remote.lstrip("/")))

    HfApi().batch_bucket_files(resolved_bucket, add=add)
    return [remote for _, remote in add]


def download_files(
    pairs: Sequence[tuple[str, Path | str]],
    *,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
) -> list[Path]:
    """Download ``(remote_path, local_path)`` pairs from a storage bucket.

    Public buckets can be read anonymously (``token=False``). If a token is
    available it is used; otherwise downloads run without authentication.
    Writes via a ``.partial`` sibling then renames into place.
    """
    from huggingface_hub import HfFileSystem

    resolved_bucket = _resolve_bucket(
        bucket=bucket,
        project_root=project_root,
        what="default artifacts bucket",
    )
    resolved = apply_hf_token(token=token)
    fs = HfFileSystem(token=resolved if resolved else False)
    written: list[Path] = []
    for remote, local in pairs:
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        uri = bucket_uri(resolved_bucket, remote)
        if not _bucket_object_exists(
            resolved_bucket, remote, token=resolved
        ):
            raise FileNotFoundError(uri)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            fs.get(uri, str(tmp))
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        written.append(dest)
    return written


def _checkpoint_step_from_file(path: Path) -> int | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(payload, dict) or "step" not in payload:
        return None
    try:
        return int(payload["step"])
    except (TypeError, ValueError):
        return None


def _bucket_file_size(
    api: Any,
    bucket: str,
    remote_path: str,
    *,
    token: str | None = None,
) -> int:
    """Return object size via ``get_bucket_paths_info`` (metadata.size is unreliable)."""
    from huggingface_hub import get_bucket_paths_info

    infos = list(
        get_bucket_paths_info(
            bucket, [remote_path.lstrip("/")], token=token or True
        )
    )
    if not infos:
        raise FileNotFoundError(
            f"bucket object not found after upload: {bucket}/{remote_path}"
        )
    info = infos[0]
    size = getattr(info, "size", None)
    if size is None:
        raise RuntimeError(f"bucket object missing size: {bucket}/{remote_path}")
    return int(size)


def _bucket_file_xet_hash(
    api: Any,
    bucket: str,
    remote_path: str,
    *,
    token: str | None = None,
) -> str:
    from huggingface_hub import get_bucket_paths_info

    infos = list(
        get_bucket_paths_info(
            bucket, [remote_path.lstrip("/")], token=token or True
        )
    )
    if not infos:
        raise FileNotFoundError(
            f"bucket object not found: {bucket}/{remote_path}"
        )
    xet_hash = getattr(infos[0], "xet_hash", None)
    if not xet_hash:
        raise RuntimeError(f"bucket object missing xet_hash: {bucket}/{remote_path}")
    return str(xet_hash)


def upload_checkpoint_to_bucket(
    local_path: str | Path,
    *,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
    canonical_name: str | None = None,
) -> list[str]:
    """Upload a local ``.pt`` via dual-slot layout and flip ``current.json``.

    Writes the inactive slot, verifies remote size, then updates the pointer.
    Optionally deletes the previous slot afterward (best-effort).
    """
    from huggingface_hub import HfApi

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stem = checkpoint_stem(canonical_name or path.name)
    resolved_bucket = _resolve_bucket(
        bucket=bucket,
        project_root=project_root,
        what="checkpoint upload",
    )
    write_token = _require_write_token(token=token)
    api = HfApi()

    active: int | None = None
    pointer_remote = checkpoint_pointer_remote_path(stem)
    if _bucket_object_exists(
        resolved_bucket, pointer_remote, token=write_token
    ):
        pointer = read_checkpoint_pointer(
            stem,
            bucket=resolved_bucket,
            token=token,
            project_root=project_root,
        )
        active = int(pointer["slot"])
        if active not in (0, 1):
            raise ValueError(
                f"invalid pointer slot {active} in "
                f"{bucket_uri(resolved_bucket, pointer_remote)}"
            )

    inactive = 0 if active is None else 1 - active
    slot_remote = checkpoint_slot_remote_path(stem, inactive)
    digest = sha256_file(path)
    size = path.stat().st_size
    step = _checkpoint_step_from_file(path)

    api.batch_bucket_files(
        resolved_bucket, add=[(str(path), slot_remote)]
    )
    remote_size = _bucket_file_size(
        api, resolved_bucket, slot_remote, token=write_token
    )
    if remote_size != size:
        raise RuntimeError(
            f"checkpoint slot size mismatch after upload: "
            f"local={size} remote={remote_size} key={slot_remote}"
        )

    pointer_payload: dict[str, Any] = {
        "slot": inactive,
        "sha256": digest,
        "size": size,
        "name": f"{stem}.pt",
    }
    if step is not None:
        pointer_payload["step"] = step
    api.batch_bucket_files(
        resolved_bucket,
        add=[
            (
                json.dumps(pointer_payload, sort_keys=True).encode("utf-8"),
                pointer_remote,
            )
        ],
    )

    uploaded = [slot_remote, pointer_remote]
    if active is not None:
        old_slot = checkpoint_slot_remote_path(stem, active)
        try:
            api.batch_bucket_files(resolved_bucket, delete=[old_slot])
        except Exception as exc:
            warnings.warn(
                f"failed to delete previous checkpoint slot {old_slot}: {exc}",
                stacklevel=2,
            )
    return uploaded


def download_checkpoint_from_bucket(
    local_path: str | Path,
    *,
    canonical_name: str | None = None,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Download the pointed dual-slot checkpoint into ``local_path``.

    Raises ``FileNotFoundError`` if ``current.json`` (or the pointed slot) is
    missing — no legacy bare-``.pt`` fallback.
    """
    dest = Path(local_path)
    stem = checkpoint_stem(canonical_name or dest.name)
    pointer = read_checkpoint_pointer(
        stem,
        bucket=bucket,
        token=token,
        project_root=project_root,
    )
    try:
        slot = int(pointer["slot"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint pointer for stem={stem!r} missing valid 'slot'"
        ) from exc
    if slot not in (0, 1):
        raise ValueError(f"checkpoint pointer slot must be 0 or 1, got {slot}")
    remote = checkpoint_slot_remote_path(stem, slot)
    download_files(
        [(remote, dest)],
        bucket=bucket,
        token=token,
        project_root=project_root,
    )
    return dest


def ensure_project_artifact(
    local_path: str | Path,
    *,
    project_root: str | Path,
    remote_path: str | None = None,
    bucket: str | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> Path:
    """Return a local artifact path, downloading from the artifacts bucket if missing.

    ``local_path`` may be absolute or relative to ``project_root``. The remote key
    defaults to the path relative to ``project_root`` (e.g. ``checkpoints/a.pt``,
    ``data/encoded_energies_ae_best.pt``). Canonical checkpoint remotes
    (``checkpoints/<stem>.pt``) resolve via the dual-slot pointer.
    """
    root = Path(project_root).resolve()
    path = Path(local_path)
    dest = path.resolve() if path.is_absolute() else (root / path).resolve()
    if dest.is_file() and not force_download:
        return dest
    if remote_path is None:
        try:
            remote = dest.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"local path {dest} is outside project_root {root}; "
                "pass remote_path= explicitly"
            ) from exc
    else:
        remote = remote_path.lstrip("/")
    if is_canonical_checkpoint_remote(remote):
        return download_checkpoint_from_bucket(
            dest,
            canonical_name=Path(remote).name,
            bucket=bucket,
            token=token,
            project_root=root,
        )
    download_files(
        [(remote, dest)],
        bucket=bucket,
        token=token,
        project_root=root,
    )
    return dest


def ensure_checkpoint(
    project_root: str | Path,
    name: str,
    *,
    force_download: bool = False,
    token: str | None = None,
) -> Path:
    """Ensure ``checkpoints/<name>`` exists locally (lazy HF pull if needed)."""
    root = Path(project_root)
    path = Path(name)
    local = path if path.is_absolute() else root / "checkpoints" / path
    return ensure_project_artifact(
        local,
        project_root=root,
        force_download=force_download,
        token=token,
    )


def default_artifact_pairs(
    project_root: Path | str,
    *,
    checkpoint_names: Iterable[str] = ("surrogate_synthetic.pt",),
    log_names: Iterable[str] = ("surrogate.sqlite",),
) -> tuple[list[tuple[Path, str]], list[tuple[str, Path]]]:
    """Build upload/download pairs under ``checkpoints/`` and ``training_logs/``.

    Checkpoint remotes listed here are canonical names; dual-slot resolution
    happens in ``download_training_artifacts`` / ``upload_training_artifacts``.
    """
    root = Path(project_root)
    upload: list[tuple[Path, str]] = []
    download: list[tuple[str, Path]] = []
    for name in checkpoint_names:
        local = root / "checkpoints" / name
        remote = f"checkpoints/{name}"
        upload.append((local, remote))
        download.append((remote, local))
    for name in log_names:
        local = root / "training_logs" / name
        remote = f"training_logs/{name}"
        upload.append((local, remote))
        download.append((remote, local))
    return upload, download


def upload_checkpoint_artifacts(
    *,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
) -> list[str]:
    """Upload checkpoint (dual-slot) and/or SQLite log (best-effort).

    The ``.pt`` upload is required to succeed when ``checkpoint_path`` is set.
    SQLite failures are warned and skipped so a log race cannot block the
    durable checkpoint promote.
    """
    root = project_root
    if root is None and checkpoint_path is not None:
        root = infer_project_root_from_checkpoint(checkpoint_path)

    uploaded: list[str] = []
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    log_file = Path(log_path) if log_path is not None else None
    has_ckpt = ckpt_path is not None and ckpt_path.is_file()
    has_log = log_file is not None and log_file.is_file()
    if not has_ckpt and not has_log:
        raise FileNotFoundError("no checkpoint/log files found to upload")

    if has_ckpt:
        assert ckpt_path is not None
        uploaded.extend(
            upload_checkpoint_to_bucket(
                ckpt_path,
                bucket=bucket,
                token=token,
                project_root=root,
            )
        )

    if has_log:
        assert log_file is not None
        try:
            uploaded.extend(
                upload_files(
                    [(log_file, f"training_logs/{log_file.name}")],
                    bucket=bucket,
                    token=token,
                    project_root=root,
                )
            )
        except Exception as exc:
            warnings.warn(
                f"sqlite artifact upload failed for {log_file.name}: {exc}",
                stacklevel=2,
            )
    return uploaded


def upload_training_artifacts(
    project_root: Path | str,
    *,
    bucket: str | None = None,
    checkpoint_names: Sequence[str] = ("surrogate_synthetic.pt",),
    log_names: Sequence[str] = ("surrogate.sqlite",),
    token: str | None = None,
) -> list[str]:
    """Upload selected checkpoint/log files that exist under ``project_root``."""
    root = Path(project_root)
    uploaded: list[str] = []
    found = False
    for name in checkpoint_names:
        local = root / "checkpoints" / name
        if not local.is_file():
            continue
        found = True
        uploaded.extend(
            upload_checkpoint_to_bucket(
                local,
                bucket=bucket,
                token=token,
                project_root=root,
                canonical_name=name,
            )
        )
    log_pairs: list[tuple[Path, str]] = []
    for name in log_names:
        local = root / "training_logs" / name
        if local.is_file():
            found = True
            log_pairs.append((local, f"training_logs/{name}"))
    if log_pairs:
        try:
            uploaded.extend(
                upload_files(
                    log_pairs, bucket=bucket, token=token, project_root=root
                )
            )
        except Exception as exc:
            warnings.warn(
                f"sqlite artifact upload failed: {exc}",
                stacklevel=2,
            )
    if not found:
        raise FileNotFoundError(
            f"no artifacts found under {Path(project_root)} for upload"
        )
    return uploaded


def download_training_artifacts(
    project_root: Path | str,
    *,
    bucket: str | None = None,
    checkpoint_names: Sequence[str] = ("surrogate_synthetic.pt",),
    log_names: Sequence[str] = ("surrogate.sqlite",),
    token: str | None = None,
) -> list[Path]:
    """Download selected checkpoint/log files into ``project_root``."""
    root = Path(project_root)
    written: list[Path] = []
    for name in checkpoint_names:
        written.append(
            ensure_checkpoint(
                root, name, force_download=True, token=token
            )
        )
    log_pairs: list[tuple[str, Path]] = []
    for name in log_names:
        log_pairs.append(
            (f"training_logs/{name}", root / "training_logs" / name)
        )
    if log_pairs:
        written.extend(
            download_files(
                log_pairs, bucket=bucket, token=token, project_root=root
            )
        )
    return written


def list_bare_checkpoint_names(
    *,
    bucket: str | None = None,
    token: str | None = None,
    project_root: str | Path | None = None,
) -> list[str]:
    """Return basenames of bare ``checkpoints/*.pt`` objects (not slots)."""
    from huggingface_hub import HfApi

    resolved_bucket = _resolve_bucket(
        bucket=bucket,
        project_root=project_root,
        what="list bare checkpoints",
    )
    apply_hf_token(token=token)
    api = HfApi()
    names: list[str] = []
    for entry in api.list_bucket_tree(
        resolved_bucket, prefix="checkpoints/", recursive=True
    ):
        path = getattr(entry, "path", None) or getattr(entry, "name", None)
        if path is None:
            continue
        remote = str(path).lstrip("/")
        if is_canonical_checkpoint_remote(remote):
            names.append(Path(remote).name)
    return sorted(set(names))


def migrate_bare_checkpoints_to_dual_slot(
    project_root: str | Path,
    *,
    bucket: str | None = None,
    token: str | None = None,
    delete_bare: bool = True,
    checkpoint_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Wrap bare HF ``checkpoints/<stem>.pt`` files as slot0 + ``current.json``.

    Prefers a server-side xet copy of the bare object into ``slot0`` (no
    re-upload). Falls back to local download + ``upload_checkpoint_to_bucket``
    when needed.

    Returns a list of result dicts for logging.
    """
    from huggingface_hub import HfApi

    root = Path(project_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    resolved_bucket = _resolve_bucket(
        bucket=bucket,
        project_root=root,
        what="migrate checkpoints",
    )
    write_token = _require_write_token(token=token)
    api = HfApi()

    if checkpoint_names is None:
        names = list_bare_checkpoint_names(
            bucket=resolved_bucket, token=token, project_root=root
        )
    else:
        names = [Path(name).name for name in checkpoint_names]

    results: list[dict[str, Any]] = []
    for name in names:
        stem = checkpoint_stem(name)
        bare_remote = f"checkpoints/{name}"
        pointer_remote = checkpoint_pointer_remote_path(stem)
        slot_remote = checkpoint_slot_remote_path(stem, 0)
        local = root / "checkpoints" / name
        result: dict[str, Any] = {"name": name, "stem": stem}

        bare_exists = _bucket_object_exists(
            resolved_bucket, bare_remote, token=write_token
        )
        if _bucket_object_exists(
            resolved_bucket, pointer_remote, token=write_token
        ):
            result["status"] = "already_migrated"
            if delete_bare and bare_exists:
                api.batch_bucket_files(resolved_bucket, delete=[bare_remote])
                result["deleted_bare"] = bare_remote
            results.append(result)
            continue

        if not bare_exists and not local.is_file():
            result["status"] = "missing"
            results.append(result)
            continue

        if bare_exists:
            xet_hash = _bucket_file_xet_hash(
                api, resolved_bucket, bare_remote, token=write_token
            )
            size = _bucket_file_size(
                api, resolved_bucket, bare_remote, token=write_token
            )
            api.batch_bucket_files(
                resolved_bucket,
                copy=[
                    ("bucket", resolved_bucket, xet_hash, slot_remote),
                ],
            )
            remote_size = _bucket_file_size(
                api, resolved_bucket, slot_remote, token=write_token
            )
            if remote_size != size:
                raise RuntimeError(
                    f"migrate copy size mismatch for {name}: "
                    f"bare={size} slot={remote_size}"
                )
            pointer_payload: dict[str, Any] = {
                "slot": 0,
                "size": size,
                "name": name,
                "xet_hash": xet_hash,
            }
            if local.is_file():
                pointer_payload["sha256"] = sha256_file(local)
                step = _checkpoint_step_from_file(local)
                if step is not None:
                    pointer_payload["step"] = step
            api.batch_bucket_files(
                resolved_bucket,
                add=[
                    (
                        json.dumps(pointer_payload, sort_keys=True).encode(
                            "utf-8"
                        ),
                        pointer_remote,
                    )
                ],
            )
            result["uploaded"] = [slot_remote, pointer_remote]
            result["status"] = "migrated"
            result["method"] = "xet_copy"
        else:
            uploaded = upload_checkpoint_to_bucket(
                local,
                bucket=resolved_bucket,
                token=token,
                project_root=root,
                canonical_name=name,
            )
            result["uploaded"] = uploaded
            result["status"] = "migrated"
            result["method"] = "local_upload"

        if delete_bare and _bucket_object_exists(
            resolved_bucket, bare_remote, token=write_token
        ):
            api.batch_bucket_files(resolved_bucket, delete=[bare_remote])
            result["deleted_bare"] = bare_remote
        results.append(result)
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("upload", "download", "migrate-checkpoints"),
        help=(
            "upload/download training artifacts, or migrate bare HF "
            "checkpoints to dual-slot layout"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="project root with configs/storage.toml, checkpoints/, training_logs/",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="HF storage bucket id (default: configs/storage.toml artifacts.bucket)",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="checkpoint filename under checkpoints/ (repeatable)",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=None,
        help="sqlite filename under training_logs/ (repeatable)",
    )
    parser.add_argument(
        "--keep-bare",
        action="store_true",
        help="with migrate-checkpoints: keep bare checkpoints/*.pt after migrate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoints = tuple(args.checkpoint or ("surrogate_synthetic.pt",))
    logs = tuple(args.log or ("surrogate.sqlite",))
    bucket = args.bucket or default_artifacts_bucket(args.project_root)
    if args.action == "upload":
        remotes = upload_training_artifacts(
            args.project_root,
            bucket=bucket,
            checkpoint_names=checkpoints,
            log_names=logs,
        )
        for remote in remotes:
            print(f"uploaded {remote}")
    elif args.action == "download":
        paths = download_training_artifacts(
            args.project_root,
            bucket=bucket,
            checkpoint_names=checkpoints,
            log_names=logs,
        )
        for path in paths:
            print(f"downloaded {path} ({path.stat().st_size} bytes)")
    else:
        results = migrate_bare_checkpoints_to_dual_slot(
            args.project_root,
            bucket=bucket,
            delete_bare=not args.keep_bare,
            checkpoint_names=args.checkpoint,
        )
        for row in results:
            print(json.dumps(row, sort_keys=True))
        migrated = sum(1 for row in results if row.get("status") == "migrated")
        print(f"migrate-checkpoints done: {migrated}/{len(results)} migrated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
