from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any


_RUN_ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "clear",
    "cosmic",
    "daring",
    "eager",
    "frosty",
    "gentle",
    "golden",
    "lively",
    "lucid",
    "nimble",
    "quiet",
    "radiant",
    "rapid",
    "steady",
    "vivid",
)
_RUN_NOUNS = (
    "arc",
    "bloom",
    "brook",
    "cipher",
    "comet",
    "delta",
    "ember",
    "field",
    "harbor",
    "kernel",
    "lantern",
    "matrix",
    "orbit",
    "pulse",
    "signal",
    "summit",
    "vector",
    "wave",
)


def _connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_training_log(path: str | Path) -> None:
    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT,
                status TEXT NOT NULL,
                resumed INTEGER NOT NULL,
                start_step INTEGER NOT NULL,
                checkpoint_step INTEGER,
                message TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS step_metrics (
                run_id TEXT NOT NULL,
                attempt_id INTEGER,
                step INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                improved INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, step, metric_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (attempt_id) REFERENCES run_attempts(attempt_id)
            )
            """
        )


def make_run_display_name() -> str:
    adjective = secrets.choice(_RUN_ADJECTIVES)
    noun = secrets.choice(_RUN_NOUNS)
    suffix = secrets.token_hex(3)
    return f"{adjective}-{noun}-{suffix}"


def make_run_instance_id(run_id: str, *, display_name: str | None = None) -> str:
    return f"{run_id}:{display_name or make_run_display_name()}"


def prune_run_history(
    path: str | Path,
    *,
    base_run_id: str,
    keep_last: int,
) -> None:
    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    initialize_training_log(path)
    prefix = f"{base_run_id}:%"
    with closing(_connect(path)) as connection, connection:
        rows = connection.execute(
            """
            SELECT run_id, metadata_json
            FROM runs
            WHERE run_id = ? OR run_id LIKE ?
            ORDER BY created_at DESC, run_id DESC
            LIMIT -1 OFFSET ?
            """,
            (base_run_id, prefix, keep_last),
        ).fetchall()
        stale_run_ids = [
            row[0] for row in rows if not bool(json.loads(row[1]).get("pinned", False))
        ]
        if not stale_run_ids:
            return
        placeholders = ",".join("?" for _run_id in stale_run_ids)
        connection.execute(
            f"DELETE FROM step_metrics WHERE run_id IN ({placeholders})", stale_run_ids
        )
        connection.execute(
            f"DELETE FROM run_attempts WHERE run_id IN ({placeholders})", stale_run_ids
        )
        connection.execute(
            f"DELETE FROM runs WHERE run_id IN ({placeholders})", stale_run_ids
        )


def upsert_run(
    path: str | Path,
    *,
    run_id: str,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    status: str = "running",
) -> None:
    initialize_training_log(path)
    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO runs (run_id, status, checkpoint_path, config_json, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                status = excluded.status,
                checkpoint_path = excluded.checkpoint_path,
                config_json = excluded.config_json,
                metadata_json = excluded.metadata_json
            """,
            (
                run_id,
                status,
                str(checkpoint_path),
                json.dumps(dict(config), sort_keys=True),
                json.dumps({} if metadata is None else dict(metadata), sort_keys=True),
            ),
        )


def _base_run_id(run_id: str, metadata: Mapping[str, Any]) -> str:
    base = metadata.get("base_run_id")
    if isinstance(base, str):
        return base
    return run_id.split(":", 1)[0]


def _run_display_name(run_id: str, metadata: Mapping[str, Any]) -> str:
    display_name = metadata.get("display_name")
    if isinstance(display_name, str):
        return display_name
    if ":" in run_id:
        return run_id.split(":", 1)[1]
    return run_id


def _run_tags(metadata: Mapping[str, Any]) -> list[str]:
    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        return []
    return [str(tag) for tag in tags]


def load_run_record(path: str | Path, *, run_id: str) -> dict[str, Any]:
    initialize_training_log(path)
    with closing(_connect(path)) as connection:
        row = connection.execute(
            """
            SELECT run_id, created_at, updated_at, status, checkpoint_path,
                   config_json, metadata_json
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"run not found: {run_id}")
    metadata = json.loads(row[6])
    return {
        "run_id": row[0],
        "base_run_id": _base_run_id(row[0], metadata),
        "display_name": _run_display_name(row[0], metadata),
        "created_at": row[1],
        "updated_at": row[2],
        "status": row[3],
        "checkpoint_path": row[4],
        "config": json.loads(row[5]),
        "metadata": metadata,
        "note": str(metadata.get("note", "")),
        "tags": _run_tags(metadata),
        "pinned": bool(metadata.get("pinned", False)),
    }


def list_training_runs(
    path: str | Path,
    *,
    base_run_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    initialize_training_log(path)
    with closing(_connect(path)) as connection:
        if base_run_id is None:
            rows = connection.execute(
                """
                SELECT run_id, created_at, updated_at, status, checkpoint_path,
                       config_json, metadata_json
                FROM runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT run_id, created_at, updated_at, status, checkpoint_path,
                       config_json, metadata_json
                FROM runs
                WHERE run_id = ? OR run_id LIKE ?
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (base_run_id, f"{base_run_id}:%", limit),
            ).fetchall()

        records = []
        for row in rows:
            run_id = row[0]
            metadata = json.loads(row[6])
            last_step = connection.execute(
                "SELECT MAX(step) FROM step_metrics WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            best_validation_loss = connection.execute(
                """
                SELECT MIN(metric_value)
                FROM step_metrics
                WHERE run_id = ? AND metric_name = 'validation_loss'
                """,
                (run_id,),
            ).fetchone()[0]
            records.append(
                {
                    "run_id": run_id,
                    "base_run_id": _base_run_id(run_id, metadata),
                    "display_name": _run_display_name(run_id, metadata),
                    "created_at": row[1],
                    "updated_at": row[2],
                    "status": row[3],
                    "checkpoint_path": row[4],
                    "config": json.loads(row[5]),
                    "metadata": metadata,
                    "note": str(metadata.get("note", "")),
                    "tags": _run_tags(metadata),
                    "pinned": bool(metadata.get("pinned", False)),
                    "last_step": last_step,
                    "best_validation_loss": best_validation_loss,
                }
            )
    return records


def start_run_attempt(
    path: str | Path,
    *,
    run_id: str,
    resumed: bool,
    start_step: int,
    checkpoint_step: int | None,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> int:
    initialize_training_log(path)
    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO run_attempts (
                run_id, status, resumed, start_step, checkpoint_step, message,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                "running",
                int(resumed),
                start_step,
                checkpoint_step,
                message,
                json.dumps({} if metadata is None else dict(metadata), sort_keys=True),
            ),
        )
        return int(cursor.lastrowid)


def finish_run_attempt(path: str | Path, *, attempt_id: int, status: str) -> None:
    initialize_training_log(path)
    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            UPDATE run_attempts
            SET ended_at = CURRENT_TIMESTAMP, status = ?
            WHERE attempt_id = ?
            """,
            (status, attempt_id),
        )


def log_step_metrics(
    path: str | Path,
    *,
    run_id: str,
    attempt_id: int | None = None,
    step: int,
    epoch: int,
    metrics: Mapping[str, float],
    best_metric_name: str | None = None,
    best_metric: float | None = None,
    improved: bool,
) -> None:
    initialize_training_log(path)
    metric_values = dict(metrics)
    if best_metric_name is not None and best_metric is not None:
        metric_values[f"best_{best_metric_name}"] = best_metric
    with closing(_connect(path)) as connection, connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO step_metrics (
                run_id, attempt_id, step, epoch, metric_name, metric_value, improved
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, attempt_id, step, epoch, name, float(value), int(improved))
                for name, value in metric_values.items()
            ],
        )


def mark_run_status(path: str | Path, *, run_id: str, status: str) -> None:
    initialize_training_log(path)
    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            UPDATE runs
            SET updated_at = CURRENT_TIMESTAMP, status = ?
            WHERE run_id = ?
            """,
            (status, run_id),
        )


def load_recent_metrics(
    path: str | Path, *, run_id: str, limit: int = 12
) -> list[dict[str, Any]]:
    initialize_training_log(path)
    with closing(_connect(path)) as connection:
        step_rows = connection.execute(
            """
            SELECT step
            FROM step_metrics
            WHERE run_id = ?
            GROUP BY step
            ORDER BY step DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        steps = [row[0] for row in reversed(step_rows)]
        if not steps:
            return []
        placeholders = ",".join("?" for _step in steps)
        rows = connection.execute(
            f"""
            SELECT step, epoch, metric_name, metric_value, improved
            FROM step_metrics
            WHERE run_id = ? AND step IN ({placeholders})
            ORDER BY step ASC, metric_name ASC
            """,
            (run_id, *steps),
        ).fetchall()

    by_step: dict[int, dict[str, Any]] = {
        step: {"step": step, "best": False} for step in steps
    }
    for step, epoch, metric_name, metric_value, improved in rows:
        row = by_step[step]
        row["epoch"] = epoch
        row[metric_name] = metric_value
        row["best"] = row["best"] or bool(improved)
    return [by_step[step] for step in steps]


def load_metric_history(
    path: str | Path,
    *,
    run_id: str,
    metric_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    initialize_training_log(path)
    names = None if metric_names is None else tuple(metric_names)
    with closing(_connect(path)) as connection:
        if names:
            placeholders = ",".join("?" for _name in names)
            rows = connection.execute(
                f"""
                SELECT step, epoch, attempt_id, metric_name, metric_value, improved
                FROM step_metrics
                WHERE run_id = ? AND metric_name IN ({placeholders})
                ORDER BY step ASC, metric_name ASC
                """,
                (run_id, *names),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT step, epoch, attempt_id, metric_name, metric_value, improved
                FROM step_metrics
                WHERE run_id = ?
                ORDER BY step ASC, metric_name ASC
                """,
                (run_id,),
            ).fetchall()

    by_step: dict[int, dict[str, Any]] = {}
    for step, epoch, attempt_id, metric_name, metric_value, improved in rows:
        row = by_step.setdefault(
            step,
            {"step": step, "epoch": epoch, "attempt_id": attempt_id, "best": False},
        )
        row[metric_name] = metric_value
        row["best"] = row["best"] or bool(improved)
        if row.get("attempt_id") is None and attempt_id is not None:
            row["attempt_id"] = attempt_id
    return [by_step[step] for step in sorted(by_step)]


def load_best_metric_row(
    path: str | Path,
    *,
    run_id: str,
    metric_name: str = "validation_loss",
    mode: str = "min",
) -> dict[str, Any] | None:
    rows = load_metric_history(path, run_id=run_id)
    candidates = [row for row in rows if metric_name in row]
    if not candidates:
        return None
    if mode == "min":
        return min(candidates, key=lambda row: float(row[metric_name]))
    if mode == "max":
        return max(candidates, key=lambda row: float(row[metric_name]))
    raise ValueError("mode must be 'min' or 'max'")
