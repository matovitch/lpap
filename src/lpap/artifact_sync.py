"""Sync training artifacts to/from a Hugging Face Storage Bucket.

Designed for the molab summer workflow: upload from the paired kernel with
``HF_TOKEN`` (from ``configs/secrets.toml`` via inject, or export locally),
download locally without auth when the bucket is public. Bucket settings live
in ``configs/storage.toml`` (required under the project root).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

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

    if bucket is None:
        root = _require_project_root_for_defaults(
            project_root, what="default artifacts bucket"
        )
        resolved_bucket = default_artifacts_bucket(root)
    else:
        resolved_bucket = bucket
    resolved = apply_hf_token(token=token)
    if not resolved:
        raise RuntimeError(
            "HF write token not found; set HF_TOKEN "
            "(configs/secrets.toml + molab-inject-secrets.sh, or export locally)"
        )

    add: list[tuple[str, str]] = []
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

    if bucket is None:
        root = _require_project_root_for_defaults(
            project_root, what="default artifacts bucket"
        )
        resolved_bucket = default_artifacts_bucket(root)
    else:
        resolved_bucket = bucket
    resolved = apply_hf_token(token=token)
    fs = HfFileSystem(token=resolved if resolved else False)
    written: list[Path] = []
    for remote, local in pairs:
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        uri = bucket_uri(resolved_bucket, remote)
        if not fs.exists(uri):
            raise FileNotFoundError(uri)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            fs.get(uri, str(tmp))
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        written.append(dest)
    return written


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
    ``data/encoded_energies_ae_best.pt``). Session-local caches under molab are
    ephemeral; HF remains the source of truth.
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
    """Build upload/download pairs under ``checkpoints/`` and ``training_logs/``."""
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
    """Upload a checkpoint and/or SQLite log into the standard bucket layout.

    Remote keys are ``checkpoints/<name>`` and ``training_logs/<name>``. Missing
    files are skipped; raises ``FileNotFoundError`` if nothing exists to upload.
    When ``bucket`` is omitted, storage config is loaded from ``project_root`` or
    inferred from ``…/checkpoints/<file>``.
    """
    pairs: list[tuple[Path, str]] = []
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.is_file():
            pairs.append((path, f"checkpoints/{path.name}"))
    if log_path is not None:
        path = Path(log_path)
        if path.is_file():
            pairs.append((path, f"training_logs/{path.name}"))
    if not pairs:
        raise FileNotFoundError("no checkpoint/log files found to upload")
    root = project_root
    if root is None and checkpoint_path is not None:
        root = infer_project_root_from_checkpoint(checkpoint_path)
    return upload_files(
        pairs, bucket=bucket, token=token, project_root=root
    )


def upload_training_artifacts(
    project_root: Path | str,
    *,
    bucket: str | None = None,
    checkpoint_names: Sequence[str] = ("surrogate_synthetic.pt",),
    log_names: Sequence[str] = ("surrogate.sqlite",),
    token: str | None = None,
) -> list[str]:
    """Upload selected checkpoint/log files that exist under ``project_root``."""
    upload_pairs, _ = default_artifact_pairs(
        project_root,
        checkpoint_names=checkpoint_names,
        log_names=log_names,
    )
    existing = [(path, remote) for path, remote in upload_pairs if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"no artifacts found under {Path(project_root)} for upload"
        )
    return upload_files(
        existing, bucket=bucket, token=token, project_root=project_root
    )


def download_training_artifacts(
    project_root: Path | str,
    *,
    bucket: str | None = None,
    checkpoint_names: Sequence[str] = ("surrogate_synthetic.pt",),
    log_names: Sequence[str] = ("surrogate.sqlite",),
    token: str | None = None,
) -> list[Path]:
    """Download selected checkpoint/log files into ``project_root``."""
    _, download_pairs = default_artifact_pairs(
        project_root,
        checkpoint_names=checkpoint_names,
        log_names=log_names,
    )
    return download_files(
        download_pairs, bucket=bucket, token=token, project_root=project_root
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("upload", "download"),
        help="upload local artifacts or download from the bucket",
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
    else:
        paths = download_training_artifacts(
            args.project_root,
            bucket=bucket,
            checkpoint_names=checkpoints,
            log_names=logs,
        )
        for path in paths:
            print(f"downloaded {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
