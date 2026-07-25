"""Summarize one training run from checkpoint + SQLite log.

Usable locally (``pixi run train-status``) or on a remote kernel after
``lpap`` is installed (e.g. inside ``molab-exec``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lpap.checkpoints import load_training_checkpoint
from lpap.training_log import (
    list_training_runs,
    load_best_metric_row,
    load_recent_metrics,
    load_run_record,
)


def resolve_artifact_path(
    project_root: str | Path,
    *,
    name: str | None,
    subdirectory: str,
) -> Path | None:
    if name is None:
        return None
    path = Path(name)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return Path(project_root) / path
    return Path(project_root) / subdirectory / path


def summarize_training_status(
    *,
    project_root: str | Path = ".",
    checkpoint_name: str | None = None,
    log_name: str | None = None,
    run_id: str | None = None,
    metric_name: str = "validation_loss",
    metric_mode: str = "min",
    recent_limit: int = 5,
) -> dict[str, Any]:
    """Return a JSON-serializable status dict for one training run.

    Paths may be bare filenames under ``checkpoints/`` / ``training_logs/``,
    relative paths under ``project_root``, or absolute paths.
    """
    root = Path(project_root)
    checkpoint_path = resolve_artifact_path(
        root, name=checkpoint_name, subdirectory="checkpoints"
    )
    log_path = resolve_artifact_path(
        root, name=log_name, subdirectory="training_logs"
    )
    summary: dict[str, Any] = {
        "project_root": str(root.resolve()),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "log_path": None if log_path is None else str(log_path),
        "run_id": run_id,
        "checkpoint": None,
        "run": None,
        "best_metric_row": None,
        "recent_metrics": [],
    }

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
        summary["checkpoint"] = {
            "step": int(payload["step"]),
            "best_metric": payload.get("best_metric"),
            "keys": sorted(str(key) for key in payload),
        }

    if log_path is not None:
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        resolved_run_id = run_id
        if resolved_run_id is None:
            runs = list_training_runs(log_path, limit=1)
            if not runs:
                raise KeyError(f"no runs in training log: {log_path}")
            resolved_run_id = str(runs[0]["run_id"])
            summary["run_id"] = resolved_run_id
        record = load_run_record(log_path, run_id=resolved_run_id)
        summary["run"] = {
            "run_id": record["run_id"],
            "status": record["status"],
            "updated_at": record["updated_at"],
            "checkpoint_path": record["checkpoint_path"],
            "note": record.get("note", ""),
            "tags": list(record.get("tags", [])),
        }
        summary["best_metric_row"] = load_best_metric_row(
            log_path,
            run_id=resolved_run_id,
            metric_name=metric_name,
            mode=metric_mode,
        )
        summary["recent_metrics"] = load_recent_metrics(
            log_path, run_id=resolved_run_id, limit=recent_limit
        )

    if summary["checkpoint"] is None and summary["run"] is None:
        raise ValueError("provide --checkpoint and/or --log")

    return summary


def format_training_status(summary: Mapping[str, Any]) -> str:
    lines: list[str] = [
        f"project_root: {summary.get('project_root')}",
        f"run_id: {summary.get('run_id')}",
    ]
    checkpoint = summary.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        lines.append(
            "checkpoint: "
            f"step={checkpoint.get('step')} best_metric={checkpoint.get('best_metric')}"
        )
        lines.append(f"checkpoint_path: {summary.get('checkpoint_path')}")
    run = summary.get("run")
    if isinstance(run, Mapping):
        lines.append(
            "run: "
            f"status={run.get('status')} updated_at={run.get('updated_at')}"
        )
        lines.append(f"log_path: {summary.get('log_path')}")
        if run.get("note"):
            lines.append(f"note: {run.get('note')}")
        tags = run.get("tags") or []
        if tags:
            lines.append(f"tags: {', '.join(str(tag) for tag in tags)}")
    best = summary.get("best_metric_row")
    if isinstance(best, Mapping):
        metric_items = [
            f"{key}={best[key]}"
            for key in sorted(best)
            if key not in {"attempt_id"} and best[key] is not None
        ]
        lines.append("best_metric_row: " + " ".join(metric_items))
    recent = summary.get("recent_metrics") or []
    if recent:
        lines.append("recent_metrics:")
        for row in recent:
            if not isinstance(row, Mapping):
                continue
            step = row.get("step")
            loss = row.get("loss", row.get("validation_loss"))
            lines.append(f"  step={step} loss={loss} best={row.get('best')}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="project root containing checkpoints/ and training_logs/",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint filename under checkpoints/ (or path)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="sqlite filename under training_logs/ (or path)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="training log run_id (default: most recent run in --log)",
    )
    parser.add_argument(
        "--metric-name",
        default="validation_loss",
        help="metric used for best-row lookup (default: validation_loss)",
    )
    parser.add_argument(
        "--metric-mode",
        choices=("min", "max"),
        default="min",
        help="best-row mode (default: min)",
    )
    parser.add_argument(
        "--recent-limit",
        type=int,
        default=5,
        help="how many recent metric steps to include",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of text",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = summarize_training_status(
        project_root=args.project_root,
        checkpoint_name=args.checkpoint,
        log_name=args.log,
        run_id=args.run_id,
        metric_name=args.metric_name,
        metric_mode=args.metric_mode,
        recent_limit=args.recent_limit,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(format_training_status(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
