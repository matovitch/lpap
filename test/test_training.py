from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

import torch
from torch import nn

from lpap.checkpoints import load_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder_training import (
    LPAPDecoderModelConfig,
    LPAPDecoderRunConfig,
    LPAPDecoderTeacherConfig,
    LPAPDecoderTrainingConfig,
    rerun_lpap_decoder_training_config_from_log,
)
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateOptimizerConfig,
    LPAPSurrogateRunConfig,
    LPAPSurrogateTrainingConfig,
    LPAPSurrogateValidationConfig,
    rerun_lpap_surrogate_training_config_from_log,
)
from lpap.training import TrainingRun, TrainingRunConfig
from lpap.training_log import load_metric_history, load_recent_metrics, upsert_run


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
            self.assertEqual(resume.base_run_id, "run-a")
            self.assertTrue(resume.run_id.startswith("run-a:"))
            self.assertEqual(resume.attempt_id, 1)
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
            self.assertEqual(payload["training_state"]["run_config"], {"kind": "unit-test"})
            self.assertTrue(
                torch.equal(payload["best_model_state"]["weight"], best_weight)
            )
            self.assertTrue(torch.equal(payload["best_model_state"]["bias"], best_bias))

            recent = load_recent_metrics(log_path, run_id=resume.run_id)
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
            self.assertEqual(resumed.run_id, resume.run_id)
            self.assertEqual(resumed.start_step, 3)
            self.assertEqual(resumed.attempt_id, 2)
            resumed_step = resumed_run.record_step(
                step=3, metrics={"loss": 1.5, "accuracy": 0.75, "aux": 9.0}
            )
            resumed_run.mark_finished()

            self.assertTrue(resumed_step.improved)
            history = load_metric_history(
                log_path, run_id=resume.run_id, metric_names=("loss",)
            )
            self.assertEqual([row["attempt_id"] for row in history], [1, 1, 2])

            with sqlite3.connect(log_path) as connection:
                attempts = connection.execute(
                    """
                    SELECT attempt_id, status, resumed, start_step, checkpoint_step
                    FROM run_attempts
                    WHERE run_id = ?
                    ORDER BY attempt_id
                    """,
                    (resume.run_id,),
                ).fetchall()

            self.assertEqual(
                attempts,
                [(1, "finished", 0, 1, None), (2, "finished", 1, 3, 2)],
            )

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

    def test_fresh_start_after_deleted_checkpoint_uses_new_run_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoints" / "model.pt"
            log_path = root / "training_logs" / "run.sqlite"

            first_run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint_path,
                    log_path=log_path,
                    total_steps=1,
                    checkpoint_every=1,
                ),
                model=nn.Linear(2, 1),
            )
            first = first_run.resume_or_initialize()
            first_run.record_step(step=1, metrics={"loss": 2.0})
            first_run.mark_finished()
            checkpoint_path.unlink()

            second_run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint_path,
                    log_path=log_path,
                    total_steps=1,
                    checkpoint_every=1,
                ),
                model=nn.Linear(2, 1),
            )
            second = second_run.resume_or_initialize()
            second_run.record_step(step=1, metrics={"loss": 1.0})
            second_run.mark_finished()

            self.assertNotEqual(first.run_id, second.run_id)
            first_history = load_metric_history(log_path, run_id=first.run_id)
            second_history = load_metric_history(log_path, run_id=second.run_id)
            self.assertEqual(first_history[0]["loss"], 2.0)
            self.assertEqual(second_history[0]["loss"], 1.0)

    def test_reconstructs_training_configs_from_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "training_logs" / "runs.sqlite"
            harmonics = SyntheticHarmonicConfig(
                harmonic_count=7,
                gain_variance=0.5,
                gain_half_life=3.0,
                spikiness_range=(2.0, 5.0),
            )
            surrogate_config = LPAPSurrogateTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=4,
                    bucket_count=8,
                    probe_count=4,
                    harmonics=harmonics,
                ),
                model=LPAPSurrogateModelConfig(
                    k_max=3, hidden_dim=32, layer_count=2, head_count=4
                ),
                optimizer=LPAPSurrogateOptimizerConfig(learning_rate=2.0e-3),
                validation=LPAPSurrogateValidationConfig(every=5, batch_size=6),
                run=LPAPSurrogateRunConfig(run_id="surrogate-a", steps=10),
            )
            decoder_config = LPAPDecoderTrainingConfig(
                data=surrogate_config.data,
                decoder=LPAPDecoderModelConfig(
                    frontend_initial_temperature=0.75,
                    hidden_dim=32,
                    layer_count=2,
                    head_count=4,
                ),
                optimizer=surrogate_config.optimizer,
                validation=surrogate_config.validation,
                teacher=LPAPDecoderTeacherConfig(
                    checkpoint_name="teacher.pt", require_checkpoint=True
                ),
                run=LPAPDecoderRunConfig(run_id="decoder-a", steps=11),
            )
            upsert_run(
                log_path,
                run_id="surrogate-a:001",
                checkpoint_path="checkpoints/surrogate.pt",
                config=surrogate_config.as_run_config(),
            )
            upsert_run(
                log_path,
                run_id="decoder-a:001",
                checkpoint_path="checkpoints/decoder.pt",
                config=decoder_config.as_run_config(),
            )

            loaded_surrogate = rerun_lpap_surrogate_training_config_from_log(
                log_path, run_id="surrogate-a:001"
            )
            loaded_decoder = rerun_lpap_decoder_training_config_from_log(
                log_path, run_id="decoder-a:001"
            )

            self.assertFalse(loaded_surrogate.run.resume_from_checkpoint)
            self.assertEqual(loaded_surrogate.model.hidden_dim, 32)
            self.assertEqual(loaded_surrogate.data.harmonics.harmonic_count, 7)
            self.assertFalse(loaded_decoder.run.resume_from_checkpoint)
            self.assertEqual(loaded_decoder.decoder.frontend_initial_temperature, 0.75)
            self.assertEqual(loaded_decoder.teacher.checkpoint_name, "teacher.pt")


if __name__ == "__main__":
    unittest.main()
