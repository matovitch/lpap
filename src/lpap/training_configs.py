"""Training TOMLs pinned to the installed lpap commit.

``configs/training/*.toml`` is not package data, so a pip-installed ``lpap``
fetches its matching configs from the GitHub archive of the commit it was
installed from (``direct_url.json``). Code and configs then always agree.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import tarfile
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath

ARCHIVE_URL = "https://codeload.github.com/matovitch/lpap/tar.gz/{ref}"
TRAINING_DIR = PurePosixPath("configs/training")
PIPELINE_TOML = "retrain_image_pipeline.toml"
STAMP = ".lpap_commit"


def installed_lpap_commit() -> str:
    """Commit sha of the VCS-installed ``lpap`` distribution ('' if unknown)."""
    try:
        raw = importlib.metadata.distribution("lpap").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return ""
    info = json.loads(raw or "{}")
    return str((info.get("vcs_info") or {}).get("commit_id", ""))


def extract_training_configs(archive: bytes) -> dict[str, bytes]:
    """``{name: bytes}`` of top-level ``configs/training/*.toml`` in a repo tarball."""
    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 4:
                continue  # <repo>-<sha>/configs/training/<name>.toml only
            if PurePosixPath(*parts[1:3]) != TRAINING_DIR:
                continue
            if not parts[3].endswith(".toml"):
                continue
            handle = tar.extractfile(member)
            assert handle is not None
            found[parts[3]] = handle.read()
    return found


def required_config_names(pipeline_toml: bytes) -> set[str]:
    """Training TOMLs the retrain pipeline references, plus the pipeline itself."""
    pipeline = tomllib.loads(pipeline_toml.decode())["pipeline"]
    refs = [*pipeline["teacher_configs"], pipeline["autoencoder_config"]]
    names = {PIPELINE_TOML}
    for ref in refs:
        path = PurePosixPath(ref)
        if path.parent != TRAINING_DIR:
            raise ValueError(f"{PIPELINE_TOML} references {ref} outside {TRAINING_DIR}")
        names.add(path.name)
    return names


def write_training_configs(
    project_root: str | Path, configs: dict[str, bytes], commit: str
) -> list[Path]:
    """Replace ``configs/training/*.toml`` with ``configs``; subdirectories stay.

    Raises ``FileNotFoundError`` when a TOML the pipeline references is missing.
    """
    if PIPELINE_TOML not in configs:
        raise FileNotFoundError(f"{TRAINING_DIR / PIPELINE_TOML} not in lpap@{commit}")
    missing = required_config_names(configs[PIPELINE_TOML]) - configs.keys()
    if missing:
        raise FileNotFoundError(
            f"lpap@{commit} lacks {sorted(str(TRAINING_DIR / n) for n in missing)}"
        )
    dest = Path(project_root) / TRAINING_DIR
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.toml"):
        if stale.name not in configs:
            stale.unlink()
    written = []
    for name, data in sorted(configs.items()):
        (dest / name).write_bytes(data)
        written.append(dest / name)
    (dest / STAMP).write_text(commit + "\n")
    return written


def sync_training_configs(project_root: str | Path, commit: str = "") -> list[Path]:
    """Install the training TOMLs of ``commit`` (default: installed lpap commit)."""
    commit = commit or installed_lpap_commit()
    if not commit:
        raise RuntimeError("lpap is not VCS-installed: no commit to fetch configs from")
    with urllib.request.urlopen(ARCHIVE_URL.format(ref=commit), timeout=120) as resp:
        archive = resp.read()
    return write_training_configs(
        project_root, extract_training_configs(archive), commit
    )
