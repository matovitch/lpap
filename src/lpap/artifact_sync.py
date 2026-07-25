"""Sync training artifacts to/from a Hugging Face Storage Bucket.

Designed for the molab summer workflow: upload from the paired kernel with
``HF_TOKEN`` / ``/marimo/.hf_token``, download locally without auth when the
bucket is public.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_BUCKET = "matovitch/lpap-molab-artifacts"
DEFAULT_TOKEN_FILES = (
    Path("/marimo/.hf_token"),
    Path(".hf_token"),
)


def resolve_hf_token(*, token: str | None = None) -> str | None:
    """Return an HF token from ``token``, env, or a local token file."""
    if token:
        return token.strip() or None
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    for path in DEFAULT_TOKEN_FILES:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
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
    bucket: str = DEFAULT_BUCKET,
    token: str | None = None,
) -> list[str]:
    """Upload ``(local_path, remote_path)`` pairs to a storage bucket.

    Requires a write-capable token (env, ``token=``, or ``.hf_token``).
    """
    from huggingface_hub import HfApi

    resolved = apply_hf_token(token=token)
    if not resolved:
        raise RuntimeError(
            "HF write token not found; set HF_TOKEN or create .hf_token"
        )

    add: list[tuple[str, str]] = []
    for local, remote in pairs:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError(path)
        add.append((str(path), remote.lstrip("/")))

    HfApi().batch_bucket_files(bucket, add=add)
    return [remote for _, remote in add]


def download_files(
    pairs: Sequence[tuple[str, Path | str]],
    *,
    bucket: str = DEFAULT_BUCKET,
    token: str | None = None,
) -> list[Path]:
    """Download ``(remote_path, local_path)`` pairs from a storage bucket.

    Public buckets can be read anonymously (``token=False``). If a token is
    available it is used; otherwise downloads run without authentication.
    """
    from huggingface_hub import HfFileSystem

    resolved = apply_hf_token(token=token)
    fs = HfFileSystem(token=resolved if resolved else False)
    written: list[Path] = []
    for remote, local in pairs:
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        uri = bucket_uri(bucket, remote)
        if not fs.exists(uri):
            raise FileNotFoundError(uri)
        fs.get(uri, str(dest))
        written.append(dest)
    return written


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
    bucket: str = DEFAULT_BUCKET,
    token: str | None = None,
) -> list[str]:
    """Upload a checkpoint and/or SQLite log into the standard bucket layout.

    Remote keys are ``checkpoints/<name>`` and ``training_logs/<name>``. Missing
    files are skipped; raises ``FileNotFoundError`` if nothing exists to upload.
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
    return upload_files(pairs, bucket=bucket, token=token)


def upload_training_artifacts(
    project_root: Path | str,
    *,
    bucket: str = DEFAULT_BUCKET,
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
    return upload_files(existing, bucket=bucket, token=token)


def download_training_artifacts(
    project_root: Path | str,
    *,
    bucket: str = DEFAULT_BUCKET,
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
    return download_files(download_pairs, bucket=bucket, token=token)


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
        help="project root containing checkpoints/ and training_logs/",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"HF storage bucket id (default: {DEFAULT_BUCKET})",
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
    if args.action == "upload":
        remotes = upload_training_artifacts(
            args.project_root,
            bucket=args.bucket,
            checkpoint_names=checkpoints,
            log_names=logs,
        )
        for remote in remotes:
            print(f"uploaded {remote}")
    else:
        paths = download_training_artifacts(
            args.project_root,
            bucket=args.bucket,
            checkpoint_names=checkpoints,
            log_names=logs,
        )
        for path in paths:
            print(f"downloaded {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
