"""Resume helpers for a training run: log fetch, resume mark, data seed.

The step budget stays outside the TOML. These helpers keep one SQLite history
across resumes and reseed the training loader so a resumed run does not replay
epoch 0.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lpap.storage import infer_project_root_from_checkpoint
from lpap.training_configs import installed_lpap_commit
from lpap.training_log import record_resume_event


def training_data_seed(seed: int, start_step: int) -> int:
    """Seed for the training data order at ``start_step`` (1 on a fresh run)."""
    return int(seed) + int(start_step)


def seed_dataloader(
    loader: DataLoader,
    *,
    seed: int,
    start_step: int,
) -> None:
    """Reseed a training DataLoader and its shuffle sampler."""
    combined = training_data_seed(seed, start_step)
    if loader.generator is None:
        loader.generator = torch.Generator()
    loader.generator.manual_seed(combined)
    sampler = getattr(loader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    if sampler_generator is not None and sampler_generator is not loader.generator:
        sampler_generator.manual_seed(combined)


def prepare_resume_log(
    *,
    checkpoint_path: Path,
    log_path: Path,
    resume: bool,
) -> None:
    """Fetch a missing SQLite log when a checkpoint is present.

    A local log is kept and appended. A checkpoint with no local log and no
    HF pointer raises ``FileNotFoundError``.
    """
    if not resume or log_path.is_file() or not checkpoint_path.is_file():
        return
    from lpap.artifact_sync import ensure_training_log

    root = infer_project_root_from_checkpoint(checkpoint_path)
    try:
        ensure_training_log(root, log_path.name)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise FileNotFoundError(
            f"checkpoint {checkpoint_path} exists but its training log "
            f"{log_path.name} was not found locally or in the HF repo"
        ) from exc
    if not log_path.is_file():
        raise FileNotFoundError(
            f"checkpoint {checkpoint_path} exists but its training log "
            f"{log_path.name} was not found locally or in the HF repo"
        )


def note_resume(*, log_path: Path, run_id: str, step: int) -> None:
    """Record one resume mark: step, timestamp, and the installed lpap commit."""
    record_resume_event(
        log_path,
        run_id=run_id,
        step=int(step),
        commit=installed_lpap_commit(),
    )


__all__ = [
    "note_resume",
    "prepare_resume_log",
    "seed_dataloader",
    "training_data_seed",
]
