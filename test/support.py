"""Paths to checked-in training TOML fixtures (not live ``configs/training/``)."""

from __future__ import annotations

from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"
TRAINING_FIXTURES = RESOURCES / "training"


def training_fixture(name: str) -> Path:
    """Return ``test/resources/training/<name>`` (must exist)."""
    path = TRAINING_FIXTURES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
