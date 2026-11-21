"""Git-like branch of one training run: copy checkpoint + log, record parent.

The parent files stay read-only. The copy gets the child run id and a parent
record. This is a runtime choice; training TOML does not carry a parent.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from lpap.checkpoints import load_training_checkpoint, write_training_checkpoint_payload
from lpap.training_log import initialize_training_log


@dataclass(frozen=True)
class RunParent:
    """Parent run to branch from. ``checkpoint`` is a filename or path."""

    run_id: str
    checkpoint: str


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    try:
        shutil.copy2(src, temporary)
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_sqlite(src: Path, dest: Path) -> None:
    _copy_file(src, dest)
    for suffix in ("-wal", "-shm"):
        extra = Path(str(src) + suffix)
        target = Path(str(dest) + suffix)
        if extra.is_file():
            _copy_file(extra, target)


def _checkpoint_wal(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()


def _connect_readonly(path: Path) -> sqlite3.Connection:
    # ``immutable=1`` keeps SQLite from creating a WAL next to ``path``.
    return sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
    )


def resolve_parent_checkpoint(project_root: Path, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "checkpoints":
        return project_root / path
    return project_root / "checkpoints" / path.name


def resolve_parent_log(project_root: Path, checkpoint_path: Path) -> Path:
    return project_root / "training_logs" / f"{checkpoint_path.stem}.sqlite"


def _logged_run_id(
    log_path: Path,
    *,
    requested: str,
    checkpoint_run_id: str | None,
) -> str:
    with closing(_connect_readonly(log_path)) as connection:

        def exists(run_id: str) -> bool:
            row = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return row is not None

        if checkpoint_run_id and exists(checkpoint_run_id):
            return checkpoint_run_id
        if exists(requested):
            return requested
        row = connection.execute(
            """
            SELECT run_id
            FROM step_metrics
            WHERE run_id = ? OR run_id LIKE ?
            GROUP BY run_id
            ORDER BY MAX(step) DESC, run_id DESC
            LIMIT 1
            """,
            (requested, f"{requested}:%"),
        ).fetchone()
    if row is None or row[0] is None:
        raise FileNotFoundError(f"parent run {requested!r} is not in {log_path}")
    return str(row[0])


def _retarget_log(
    path: Path,
    *,
    parent_run_id: str,
    child_run_id: str,
    parent_checkpoint_name: str,
) -> None:
    initialize_training_log(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        row = connection.execute(
            "SELECT metadata_json FROM runs WHERE run_id = ?",
            (parent_run_id,),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(
                f"parent run {parent_run_id!r} is not in the copied log {path}"
            )
        metadata = json.loads(row[0])
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["parent_run_id"] = parent_run_id
        metadata["parent_checkpoint"] = parent_checkpoint_name
        metadata["display_name"] = child_run_id
        connection.execute(
            "UPDATE step_metrics SET run_id = ? WHERE run_id = ?",
            (child_run_id, parent_run_id),
        )
        connection.execute(
            "UPDATE run_attempts SET run_id = ? WHERE run_id = ?",
            (child_run_id, parent_run_id),
        )
        connection.execute(
            "UPDATE run_events SET run_id = ? WHERE run_id = ?",
            (child_run_id, parent_run_id),
        )
        connection.execute(
            """
            UPDATE runs
            SET run_id = ?, metadata_json = ?
            WHERE run_id = ?
            """,
            (child_run_id, json.dumps(metadata, sort_keys=True), parent_run_id),
        )
        connection.commit()


def _lineage_from_checkpoint(path: Path, parent: RunParent) -> dict[str, str]:
    payload = load_training_checkpoint(path, map_location="cpu")
    training_state = payload.get("training_state") or {}
    metadata = {}
    if isinstance(training_state, dict):
        raw = training_state.get("metadata") or {}
        if isinstance(raw, dict):
            metadata = raw
    parent_run_id = metadata.get("parent_run_id") or parent.run_id
    parent_checkpoint = (
        metadata.get("parent_checkpoint") or Path(parent.checkpoint).name
    )
    return {
        "parent_run_id": str(parent_run_id),
        "parent_checkpoint": str(parent_checkpoint),
    }


def _patch_checkpoint(
    path: Path,
    *,
    child_run_id: str,
    parent_run_id: str,
    parent_checkpoint_name: str,
) -> None:
    payload = load_training_checkpoint(path, map_location="cpu")
    training_state = payload.get("training_state") or {}
    if not isinstance(training_state, dict):
        training_state = {}
    metadata = training_state.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata["parent_run_id"] = parent_run_id
    metadata["parent_checkpoint"] = parent_checkpoint_name
    metadata["display_name"] = child_run_id
    training_state = dict(training_state)
    training_state["run_id"] = child_run_id
    training_state["metadata"] = metadata
    payload["training_state"] = training_state
    write_training_checkpoint_payload(path, payload)


def materialize_branch(
    *,
    project_root: str | Path,
    parent: RunParent,
    dest_checkpoint: str | Path,
    dest_log: str | Path,
    child_run_id: str,
) -> dict[str, str]:
    """Copy ``parent`` onto the child paths and record the parent.

    When the child checkpoint already exists the copy is skipped. Parent
    paths are never opened for write.
    """
    if not child_run_id.strip():
        raise ValueError("child run id must be non-empty")
    root = Path(project_root)
    parent_checkpoint = resolve_parent_checkpoint(root, parent.checkpoint)
    parent_log = resolve_parent_log(root, parent_checkpoint)
    dest_checkpoint_path = Path(dest_checkpoint)
    dest_log_path = Path(dest_log)
    if dest_checkpoint_path.is_file():
        return _lineage_from_checkpoint(dest_checkpoint_path, parent)
    if not parent_checkpoint.is_file():
        from lpap.artifact_sync import ensure_checkpoint

        ensure_checkpoint(root, parent_checkpoint.name)
    if not parent_checkpoint.is_file():
        raise FileNotFoundError(f"parent checkpoint not found: {parent_checkpoint}")
    if not parent_log.is_file():
        from lpap.artifact_sync import ensure_training_log

        try:
            ensure_training_log(root, parent_log.name)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise FileNotFoundError(
                f"parent checkpoint {parent_checkpoint.name} exists but its "
                f"training log {parent_log.name} was not found locally or in the HF repo"
            ) from exc
    if not parent_log.is_file():
        raise FileNotFoundError(
            f"parent checkpoint {parent_checkpoint.name} exists but its "
            f"training log {parent_log.name} was not found"
        )
    parent_payload = load_training_checkpoint(parent_checkpoint, map_location="cpu")
    training_state = parent_payload.get("training_state") or {}
    checkpoint_run_id = None
    if isinstance(training_state, dict) and isinstance(
        training_state.get("run_id"), str
    ):
        checkpoint_run_id = str(training_state["run_id"])
    parent_run_id = _logged_run_id(
        parent_log,
        requested=parent.run_id,
        checkpoint_run_id=checkpoint_run_id,
    )
    if child_run_id == parent_run_id:
        raise ValueError(
            f"branch run id {child_run_id!r} matches parent run {parent_run_id!r}"
        )
    _copy_file(parent_checkpoint, dest_checkpoint_path)
    _copy_sqlite(parent_log, dest_log_path)
    _retarget_log(
        dest_log_path,
        parent_run_id=parent_run_id,
        child_run_id=child_run_id,
        parent_checkpoint_name=parent_checkpoint.name,
    )
    _checkpoint_wal(dest_log_path)
    _patch_checkpoint(
        dest_checkpoint_path,
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        parent_checkpoint_name=parent_checkpoint.name,
    )
    return {
        "parent_run_id": parent_run_id,
        "parent_checkpoint": parent_checkpoint.name,
    }


__all__ = [
    "RunParent",
    "materialize_branch",
    "resolve_parent_checkpoint",
    "resolve_parent_log",
]
