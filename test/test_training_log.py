from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from lpap.training_log import (
    finish_run_attempt,
    list_training_runs,
    load_best_metric_row,
    load_metric_history,
    load_recent_metrics,
    load_run_record,
    log_step_metrics,
    prune_run_history,
    mark_run_status,
    start_run_attempt,
    upsert_run,
)


class TrainingLogTest(unittest.TestCase):
    def test_logs_run_config_and_step_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training.sqlite"

            upsert_run(
                path,
                run_id="run-a",
                checkpoint_path="checkpoints/model.pt",
                config={"bucket_count": 64, "permutation_seed": 123},
                metadata={"device": "cuda"},
            )
            attempt_id = start_run_attempt(
                path,
                run_id="run-a",
                resumed=False,
                start_step=1,
                checkpoint_step=None,
                message="starting fresh",
            )
            log_step_metrics(
                path,
                run_id="run-a",
                attempt_id=attempt_id,
                step=1,
                epoch=1,
                metrics={
                    "loss": 1.5,
                    "accuracy": 0.25,
                    "weighted_accuracy": 0.5,
                    "mean_weight": 0.75,
                    "custom_energy": 3.0,
                },
                best_metric_name="loss",
                best_metric=1.5,
                improved=True,
            )
            mark_run_status(path, run_id="run-a", status="finished")
            finish_run_attempt(path, attempt_id=attempt_id, status="finished")

            recent = load_recent_metrics(path, run_id="run-a")
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["step"], 1)
            self.assertEqual(recent[0]["best"], True)
            self.assertEqual(recent[0]["custom_energy"], 3.0)
            self.assertEqual(recent[0]["best_loss"], 1.5)

            with sqlite3.connect(path) as connection:
                status, config_json = connection.execute(
                    "SELECT status, config_json FROM runs WHERE run_id = ?", ("run-a",)
                ).fetchone()

            self.assertEqual(status, "finished")
            self.assertIn('"permutation_seed": 123', config_json)

            history = load_metric_history(
                path, run_id="run-a", metric_names=("loss", "custom_energy")
            )
            self.assertEqual(history[0]["attempt_id"], attempt_id)
            self.assertEqual(history[0]["loss"], 1.5)
            best_row = load_best_metric_row(path, run_id="run-a", metric_name="loss")
            self.assertIsNotNone(best_row)
            assert best_row is not None
            self.assertEqual(best_row["weighted_accuracy"], 0.5)

            with sqlite3.connect(path) as connection:
                attempt = connection.execute(
                    """
                    SELECT status, resumed, start_step, checkpoint_step, message
                    FROM run_attempts
                    WHERE attempt_id = ?
                    """,
                    (attempt_id,),
                ).fetchone()

            self.assertEqual(attempt, ("finished", 0, 1, None, "starting fresh"))

            runs = list_training_runs(path, base_run_id="run-a")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], "run-a")
            self.assertEqual(runs[0]["last_step"], 1)
            self.assertEqual(runs[0]["best_validation_loss"], None)
            record = load_run_record(path, run_id="run-a")
            self.assertEqual(record["config"]["bucket_count"], 64)

    def test_prunes_old_run_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training.sqlite"
            for index in range(12):
                run_id = f"run-a:instance-{index:02d}"
                upsert_run(
                    path,
                    run_id=run_id,
                    checkpoint_path="checkpoints/model.pt",
                    config={},
                )
                attempt_id = start_run_attempt(
                    path,
                    run_id=run_id,
                    resumed=False,
                    start_step=1,
                    checkpoint_step=None,
                    message="starting fresh",
                )
                log_step_metrics(
                    path,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    step=1,
                    epoch=1,
                    metrics={"loss": float(index)},
                    improved=True,
                )

            prune_run_history(path, base_run_id="run-a", keep_last=10)

            with sqlite3.connect(path) as connection:
                run_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT run_id FROM runs ORDER BY run_id"
                    ).fetchall()
                ]

            self.assertEqual(len(run_ids), 10)
            self.assertNotIn("run-a:instance-00", run_ids)
            self.assertNotIn("run-a:instance-01", run_ids)


if __name__ == "__main__":
    unittest.main()
