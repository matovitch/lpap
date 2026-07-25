from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from lpap.training_log import (
    log_step_metrics,
    mark_run_status,
    start_run_attempt,
    upsert_run,
)
from lpap.training_status import (
    format_training_status,
    main,
    summarize_training_status,
)


class TrainingStatusTest(unittest.TestCase):
    def test_summarize_checkpoint_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / "checkpoints"
            logs = root / "training_logs"
            checkpoints.mkdir()
            logs.mkdir()

            ckpt = checkpoints / "model.pt"
            torch.save(
                {
                    "step": 40,
                    "best_metric": 0.12,
                    "model_state": {},
                    "best_model_state": {},
                },
                ckpt,
            )

            log_path = logs / "model.sqlite"
            upsert_run(
                log_path,
                run_id="model",
                checkpoint_path=str(ckpt),
                config={"steps": 100},
                metadata={"note": "unit", "tags": ["test"]},
            )
            attempt_id = start_run_attempt(
                log_path,
                run_id="model",
                resumed=False,
                start_step=1,
                checkpoint_step=None,
                message="starting fresh",
            )
            log_step_metrics(
                log_path,
                run_id="model",
                attempt_id=attempt_id,
                step=20,
                epoch=1,
                metrics={"loss": 0.5, "validation_loss": 0.4},
                best_metric_name="validation_loss",
                best_metric=0.4,
                improved=True,
            )
            log_step_metrics(
                log_path,
                run_id="model",
                attempt_id=attempt_id,
                step=40,
                epoch=1,
                metrics={"loss": 0.2, "validation_loss": 0.12},
                best_metric_name="validation_loss",
                best_metric=0.12,
                improved=True,
            )
            mark_run_status(log_path, run_id="model", status="running")

            summary = summarize_training_status(
                project_root=root,
                checkpoint_name="model.pt",
                log_name="model.sqlite",
                run_id="model",
            )
            self.assertEqual(summary["checkpoint"]["step"], 40)
            self.assertEqual(summary["checkpoint"]["best_metric"], 0.12)
            self.assertEqual(summary["run"]["status"], "running")
            self.assertEqual(summary["best_metric_row"]["step"], 40)
            self.assertEqual(summary["best_metric_row"]["validation_loss"], 0.12)
            self.assertGreaterEqual(len(summary["recent_metrics"]), 1)

            text = format_training_status(summary)
            self.assertIn("step=40", text)
            self.assertIn("best_metric=0.12", text)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(
                    [
                        "--project-root",
                        str(root),
                        "--checkpoint",
                        "model.pt",
                        "--log",
                        "model.sqlite",
                        "--run-id",
                        "model",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["checkpoint"]["step"], 40)

    def test_bg_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / "checkpoints"
            logs = root / "training_logs"
            checkpoints.mkdir()
            logs.mkdir()
            ckpt = checkpoints / "model.pt"
            torch.save(
                {"step": 40, "best_metric": 0.12, "model_state": {}},
                ckpt,
            )
            log_path = logs / "model.sqlite"
            upsert_run(
                log_path,
                run_id="model",
                checkpoint_path=str(ckpt),
                config={"steps": 100},
                metadata={"note": "bg", "tags": []},
            )
            attempt_id = start_run_attempt(
                log_path,
                run_id="model",
                resumed=False,
                start_step=1,
                checkpoint_step=None,
                message="fresh",
            )
            log_step_metrics(
                log_path,
                run_id="model",
                attempt_id=attempt_id,
                step=40,
                epoch=1,
                metrics={"loss": 0.2, "validation_loss": 0.12},
                best_metric_name="validation_loss",
                best_metric=0.12,
                improved=True,
            )
            (logs / "model_bg.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
            (logs / "model_bg.log").write_text(
                "step=35\nstep=40 loss=0.2\n", encoding="utf-8"
            )

            summary = summarize_training_status(
                project_root=root,
                checkpoint_name="model.pt",
                log_name="model.sqlite",
                run_id="model",
                bg_stem="model_bg",
            )
            self.assertEqual(summary["log_max_step"], 40)
            self.assertTrue(summary["bg"]["alive"])
            self.assertEqual(summary["bg"]["log_last_step"], 40)
            text = format_training_status(summary)
            self.assertIn("bg:", text)
            self.assertNotIn("progress:", text)

    def test_requires_checkpoint_or_log(self) -> None:
        with self.assertRaises(ValueError):
            summarize_training_status(project_root=".")

    def test_json_round_trip_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt = root / "x.pt"
            torch.save({"step": 1, "best_metric": 1.0, "model_state": {}}, ckpt)
            summary = summarize_training_status(
                project_root=root, checkpoint_name=str(ckpt)
            )
            encoded = json.dumps(summary, default=str)
            self.assertIn('"step": 1', encoded)


if __name__ == "__main__":
    unittest.main()
