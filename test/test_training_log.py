from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from lpap.training_log import (
    load_recent_metrics,
    log_step_metrics,
    mark_run_status,
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
            log_step_metrics(
                path,
                run_id="run-a",
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


if __name__ == "__main__":
    unittest.main()
