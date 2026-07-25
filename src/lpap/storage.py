"""Load Hugging Face storage settings from ``configs/storage.toml``."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactsStorageConfig:
    bucket: str

    def validate(self) -> None:
        if not self.bucket.strip():
            raise ValueError("artifacts.bucket must be non-empty")


@dataclass(frozen=True)
class ImagesStorageConfig:
    bucket: str
    remote_zst: str
    local_pt: str
    local_zst: str

    def validate(self) -> None:
        if not self.bucket.strip():
            raise ValueError("images.bucket must be non-empty")
        if not self.remote_zst.strip():
            raise ValueError("images.remote_zst must be non-empty")
        if not self.local_pt.strip():
            raise ValueError("images.local_pt must be non-empty")
        if not self.local_zst.strip():
            raise ValueError("images.local_zst must be non-empty")


@dataclass(frozen=True)
class StorageConfig:
    artifacts: ArtifactsStorageConfig
    images: ImagesStorageConfig

    def validate(self) -> None:
        self.artifacts.validate()
        self.images.validate()


def storage_config_path(project_root: str | Path) -> Path:
    return Path(project_root) / "configs" / "storage.toml"


def storage_config_from_dict(data: dict[str, Any]) -> StorageConfig:
    artifacts = data.get("artifacts", {})
    images = data.get("images", {})
    if not isinstance(artifacts, dict):
        raise TypeError("artifacts section must be a table")
    if not isinstance(images, dict):
        raise TypeError("images section must be a table")
    config = StorageConfig(
        artifacts=ArtifactsStorageConfig(bucket=str(artifacts["bucket"])),
        images=ImagesStorageConfig(
            bucket=str(images["bucket"]),
            remote_zst=str(images["remote_zst"]),
            local_pt=str(images["local_pt"]),
            local_zst=str(images["local_zst"]),
        ),
    )
    config.validate()
    return config


def storage_config_from_file(path: str | Path) -> StorageConfig:
    with Path(path).open("rb") as file:
        return storage_config_from_dict(tomllib.load(file))


def _apply_env_overrides(config: StorageConfig) -> StorageConfig:
    artifacts_bucket = os.environ.get("LPAP_ARTIFACTS_BUCKET", "").strip()
    images_bucket = os.environ.get("LPAP_IMAGES_BUCKET", "").strip()
    if not artifacts_bucket and not images_bucket:
        return config
    return StorageConfig(
        artifacts=ArtifactsStorageConfig(
            bucket=artifacts_bucket or config.artifacts.bucket
        ),
        images=ImagesStorageConfig(
            bucket=images_bucket or config.images.bucket,
            remote_zst=config.images.remote_zst,
            local_pt=config.images.local_pt,
            local_zst=config.images.local_zst,
        ),
    )


def load_storage_config(project_root: str | Path) -> StorageConfig:
    """Load ``configs/storage.toml`` under ``project_root``.

    Raises ``FileNotFoundError`` if the file is missing. After load,
    ``LPAP_ARTIFACTS_BUCKET`` / ``LPAP_IMAGES_BUCKET`` override bucket fields
    when set.
    """
    path = storage_config_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing storage config: {path}. "
            "Copy configs/storage.toml from the repo (required; no packaged "
            "default), or pass an explicit bucket= where the API allows it."
        )
    return _apply_env_overrides(storage_config_from_file(path))


def infer_project_root_from_checkpoint(
    checkpoint_path: str | Path,
) -> Path:
    """Treat ``…/checkpoints/<file>`` as living under the project root."""
    path = Path(checkpoint_path).resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


__all__ = [
    "ArtifactsStorageConfig",
    "ImagesStorageConfig",
    "StorageConfig",
    "infer_project_root_from_checkpoint",
    "load_storage_config",
    "storage_config_from_dict",
    "storage_config_from_file",
    "storage_config_path",
]
