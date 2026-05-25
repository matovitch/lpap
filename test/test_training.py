from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lpap.checkpoints import load_training_checkpoint
from lpap.training import TrainingRun, TrainingRunConfig
from lpap.training_log import load_recent_metrics


class TrainingRunTest(unittest.TestCase):
    def test_records_generic_metrics_and_resumes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoints" / "model.pt"
            log_path = root / "training_logs" / "run.sqlite"
            model = nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint_path,
                    log_path=log_path,
                    total_steps=2,
                    monitor="loss",
                    mode="min",
                    checkpoint_every=2,
                    log_every=1,
                    display_every=2,
                ),
                model=model,
                optimizer=optimizer,
                run_config={"kind": "unit-test"},
                model_config={"width": 2},
            )

            resume = run.resume_or_initialize()
            self.assertFalse(resume.resumed)
            first = run.record_step(
                step=1,
                metrics={"loss": 2.0, "accuracy": 0.25, "aux": 7.0},
            )
            best_weight = model.weight.detach().clone()
            best_bias = model.bias.detach().clone()
            with torch.no_grad():
                model.weight.add_(1.0)
                model.bias.add_(1.0)
            second = run.record_step(
                step=2,
                metrics={"loss": 3.0, "accuracy": 0.5, "aux": 8.0},
            )
            run.mark_finished()

            self.assertTrue(first.improved)
            self.assertFalse(first.checkpointed)
            self.assertFalse(second.improved)
            self.assertTrue(second.checkpointed)
            self.assertTrue(second.should_display)
            payload = load_training_checkpoint(checkpoint_path)
            self.assertEqual(payload["best_metric"], 2.0)
            self.assertTrue(
                torch.equal(payload["best_model_state"]["weight"], best_weight)
            )
            self.assertTrue(torch.equal(payload["best_model_state"]["bias"], best_bias))

            recent = load_recent_metrics(log_path, run_id="run-a")
            self.assertEqual([row["step"] for row in recent], [1, 2])
            self.assertEqual(recent[0]["aux"], 7.0)

            resumed_model = nn.Linear(2, 1)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1.0e-3)
            resumed_run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint_path,
                    log_path=log_path,
                    total_steps=3,
                    monitor="loss",
                    mode="min",
                ),
                model=resumed_model,
                optimizer=resumed_optimizer,
                model_config={"width": 2},
            )

            resumed = resumed_run.resume_or_initialize()
            self.assertTrue(resumed.resumed)
            self.assertEqual(resumed.start_step, 3)

    def test_missing_monitor_metric_does_not_update_best(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoints" / "model.pt"
            log_path = root / "training_logs" / "run.sqlite"
            model = nn.Linear(2, 1)
            run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint_path,
                    log_path=log_path,
                    total_steps=2,
                    monitor="validation_loss",
                    checkpoint_every=None,
                    checkpoint_on_improvement=True,
                    checkpoint_at_end=False,
                ),
                model=model,
            )

            run.resume_or_initialize()
            training_only = run.record_step(step=1, metrics={"loss": 2.0})
            validation = run.record_step(
                step=2, metrics={"loss": 1.0, "validation_loss": 0.75}
            )

            self.assertFalse(training_only.improved)
            self.assertFalse(training_only.checkpointed)
            self.assertTrue(validation.improved)
            self.assertTrue(validation.checkpointed)
            payload = load_training_checkpoint(checkpoint_path)
            self.assertEqual(payload["metric_name"], "validation_loss")
            self.assertEqual(payload["best_metric"], 0.75)


if __name__ == "__main__":
    unittest.main()
