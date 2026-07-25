"""Load Hugging Face storage defaults from TOML (not Python constants)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from importlib import resources
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
class AuthStorageConfig:
    token_files: tuple[str, ...]

    def validate(self) -> None:
        if not self.token_files:
            raise ValueError("auth.token_files must be non-empty")


@dataclass(frozen=True)
class StorageConfig:
    artifacts: ArtifactsStorageConfig
    images: ImagesStorageConfig
    auth: AuthStorageConfig

    def validate(self) -> None:
        self.artifacts.validate()
        self.images.validate()
        self.auth.validate()


def storage_config_path(project_root: str | Path) -> Path:
    return Path(project_root) / "configs" / "storage.toml"


def storage_config_from_dict(data: dict[str, Any]) -> StorageConfig:
    artifacts = data.get("artifacts", {})
    images = data.get("images", {})
    auth = data.get("auth", {})
    if not isinstance(artifacts, dict):
        raise TypeError("artifacts section must be a table")
    if not isinstance(images, dict):
        raise TypeError("images section must be a table")
    if not isinstance(auth, dict):
        raise TypeError("auth section must be a table")
    token_files = auth.get("token_files", ())
    if isinstance(token_files, str):
        token_files_tuple = (token_files,)
    else:
        token_files_tuple = tuple(str(item) for item in token_files)
    config = StorageConfig(
        artifacts=ArtifactsStorageConfig(bucket=str(artifacts["bucket"])),
        images=ImagesStorageConfig(
            bucket=str(images["bucket"]),
            remote_zst=str(images["remote_zst"]),
            local_pt=str(images["local_pt"]),
            local_zst=str(images["local_zst"]),
        ),
        auth=AuthStorageConfig(token_files=token_files_tuple),
    )
    config.validate()
    return config


def storage_config_from_file(path: str | Path) -> StorageConfig:
    with Path(path).open("rb") as file:
        return storage_config_from_dict(tomllib.load(file))


def packaged_default_storage_config() -> StorageConfig:
    resource = resources.files("lpap").joinpath("default_storage.toml")
    with resource.open("rb") as file:
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
        auth=config.auth,
    )


def load_storage_config(project_root: str | Path | None = None) -> StorageConfig:
    """Load storage defaults from project TOML, else packaged defaults.

    Resolution after load: ``LPAP_ARTIFACTS_BUCKET`` / ``LPAP_IMAGES_BUCKET``
    override bucket fields when set.
    """
    if project_root is not None:
        path = storage_config_path(project_root)
        if path.is_file():
            return _apply_env_overrides(storage_config_from_file(path))
    return _apply_env_overrides(packaged_default_storage_config())


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
    "AuthStorageConfig",
    "ImagesStorageConfig",
    "StorageConfig",
    "infer_project_root_from_checkpoint",
    "load_storage_config",
    "packaged_default_storage_config",
    "storage_config_from_dict",
    "storage_config_from_file",
    "storage_config_path",
]
