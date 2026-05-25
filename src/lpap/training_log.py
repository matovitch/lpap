from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_training_log(path: str | Path) -> None:
    with _connect(path) as connection:
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
            CREATE TABLE IF NOT EXISTS step_metrics (
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                improved INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, step, metric_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
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
    with _connect(path) as connection:
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


def log_step_metrics(
    path: str | Path,
    *,
    run_id: str,
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
    with _connect(path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO step_metrics (
                run_id, step, epoch, metric_name, metric_value, improved
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, step, epoch, name, float(value), int(improved))
                for name, value in metric_values.items()
            ],
        )


def mark_run_status(path: str | Path, *, run_id: str, status: str) -> None:
    initialize_training_log(path)
    with _connect(path) as connection:
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
    with _connect(path) as connection:
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
