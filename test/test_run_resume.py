"""Resume, log continuity, and branching for the image retrain pipeline."""

from __future__ import annotations

import shutil
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from lpap.artifact_sync import sha256_file
from lpap.checkpoints import load_training_checkpoint
from lpap.energy_bank import EnergyBankConfig
from lpap.retrain_image_pipeline import step_overrides_from_total
from lpap.run_branch import RunParent, materialize_branch
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateRunConfig,
    LPAPSurrogateTrainingConfig,
    LPAPSurrogateValidationConfig,
    create_lpap_surrogate_training_session,
    iter_lpap_surrogate_training,
)
from lpap.training import TrainingRun, TrainingRunConfig
from lpap.training_log import (
    list_run_events,
    load_metric_history,
    log_step_metrics,
    upsert_run,
)


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return sha256_file(path)


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def _assert_equal_state(test: unittest.TestCase, left, right) -> None:
    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertTrue(torch.equal(left[key], right[key]), key)


def _tiny_surrogate_config(
    *,
    steps: int,
    resume: bool,
) -> LPAPSurrogateTrainingConfig:
    return LPAPSurrogateTrainingConfig(
        data=LPAPSurrogateDataConfig(
            batch_size=4,
            bucket_count=4,
            probe_count=4,
            energy_bank=EnergyBankConfig(path="data/bank.pt"),
        ),
        model=LPAPSurrogateModelConfig(
            k_max=2,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        ),
        validation=LPAPSurrogateValidationConfig(every=1, batch_size=4, seed=7),
        run=LPAPSurrogateRunConfig(
            steps=steps,
            seed=11,
            display_every=1,
            log_every=1,
            run_id="tiny-surrogate",
            checkpoint_name="tiny_surrogate.pt",
            log_name="tiny_surrogate.sqlite",
            resume_from_checkpoint=resume,
        ),
    )


def _write_bank(root: Path) -> None:
    path = root / "data" / "bank.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"energies": torch.randn(16, 16, generator=torch.Generator().manual_seed(3))},
        path,
    )


def _parent_hashes(root: Path, checkpoint: str, log: str) -> dict[str, str]:
    return {
        "checkpoint": _sha256(root / "checkpoints" / checkpoint),
        "log": _sha256(root / "training_logs" / log),
        "wal": _sha256(root / "training_logs" / f"{log}-wal"),
    }


class ResumeBudgetTest(unittest.TestCase):
    def test_resume_bit_equal_after_raising_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_bank(root)
            session = create_lpap_surrogate_training_session(
                project_root=root,
                config=_tiny_surrogate_config(steps=1, resume=False),
                device="cpu",
            )
            first = list(iter_lpap_surrogate_training(session))
            self.assertEqual([item.step for item in first], [1])
            payload = load_training_checkpoint(session.checkpoint_path)
            saved = {
                key: value.detach().cpu().clone()
                for key, value in payload["model_state"].items()
            }

            snap_a = Path(temp_dir) / "a"
            snap_b = Path(temp_dir) / "b"
            for snap in (snap_a, snap_b):
                shutil.copytree(root / "checkpoints", snap / "checkpoints")
                shutil.copytree(root / "training_logs", snap / "training_logs")
                shutil.copytree(root / "data", snap / "data")

            def continue_run(snap: Path):
                resumed = create_lpap_surrogate_training_session(
                    project_root=snap,
                    config=_tiny_surrogate_config(steps=3, resume=True),
                    device="cpu",
                )
                self.assertTrue(resumed.resume_info.resumed)
                self.assertEqual(resumed.resume_info.start_step, 2)
                expected = torch.Generator(device="cpu").manual_seed(
                    resumed.config.run.seed + resumed.resume_info.start_step
                )
                self.assertTrue(
                    torch.equal(resumed.generator.get_state(), expected.get_state())
                )
                loaded = _state(resumed.model)
                results = list(iter_lpap_surrogate_training(resumed))
                return loaded, _state(resumed.model), [item.step for item in results]

            loaded_a, final_a, steps_a = continue_run(snap_a)
            loaded_b, final_b, steps_b = continue_run(snap_b)
            _assert_equal_state(self, loaded_a, saved)
            _assert_equal_state(self, loaded_b, saved)
            _assert_equal_state(self, final_a, final_b)
            self.assertEqual(steps_a, [2, 3])
            self.assertEqual(steps_b, [2, 3])
            self.assertFalse(
                all(torch.equal(final_a[key], saved[key]) for key in saved)
            )


class ResumeLogTest(unittest.TestCase):
    def test_fetches_appends_and_records_one_resume_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoints" / "model.pt"
            log_path = root / "training_logs" / "model.sqlite"
            remote_log = root / "remote.sqlite"
            model = nn.Linear(2, 1)
            run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=1,
                    checkpoint_every=1,
                    log_every=1,
                ),
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-3),
                model_config={"width": 2},
            )
            info = run.resume_or_initialize()
            run.record_step(step=1, metrics={"loss": 1.0})
            run.mark_finished()
            shutil.copy2(log_path, remote_log)
            log_path.unlink()
            for suffix in ("-wal", "-shm"):
                Path(str(log_path) + suffix).unlink(missing_ok=True)

            def fake_ensure(project_root, name, **kwargs):
                dest = Path(project_root) / "training_logs" / Path(name).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(remote_log, dest)
                return dest

            resumed_model = nn.Linear(2, 1)
            resumed = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=2,
                    log_every=1,
                ),
                model=resumed_model,
                optimizer=torch.optim.AdamW(resumed_model.parameters(), lr=1.0e-3),
                model_config={"width": 2},
            )
            with (
                patch(
                    "lpap.artifact_sync.ensure_training_log",
                    side_effect=fake_ensure,
                ),
                patch(
                    "lpap.run_resume.installed_lpap_commit",
                    return_value="abc123",
                ),
            ):
                resumed_info = resumed.resume_or_initialize()
            self.assertTrue(resumed_info.resumed)
            self.assertEqual(resumed_info.run_id, info.run_id)
            resumed.record_step(step=2, metrics={"loss": 0.5})
            history = load_metric_history(
                log_path, run_id=info.run_id, metric_names=("loss",)
            )
            self.assertEqual([row["step"] for row in history], [1, 2])
            self.assertTrue(history[0].get("resume"))
            events = list_run_events(log_path, run_id=info.run_id)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["kind"], "resume")
            self.assertEqual(events[0]["step"], 1)
            self.assertEqual(events[0]["commit"], "abc123")
            self.assertTrue(events[0]["created_at"])

    def test_checkpoint_without_log_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoints" / "model.pt"
            log_path = root / "training_logs" / "model.sqlite"
            model = nn.Linear(2, 1)
            run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=1,
                    checkpoint_every=1,
                ),
                model=model,
                model_config={"width": 2},
            )
            run.resume_or_initialize()
            run.record_step(step=1, metrics={"loss": 1.0})
            log_path.unlink()
            for suffix in ("-wal", "-shm"):
                Path(str(log_path) + suffix).unlink(missing_ok=True)
            again = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=2,
                    resume=True,
                ),
                model=nn.Linear(2, 1),
                model_config={"width": 2},
            )
            with self.assertRaisesRegex(FileNotFoundError, "training log"):
                again.resume_or_initialize()

    def test_upload_after_resume_keeps_prior_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "training_logs" / "model.sqlite"
            upsert_run(
                log_path,
                run_id="run-a",
                checkpoint_path=root / "checkpoints" / "model.pt",
                config={"kind": "unit"},
                metadata={"comment": ""},
            )
            log_step_metrics(
                log_path,
                run_id="run-a",
                step=1,
                epoch=1,
                metrics={"loss": 1.0},
                improved=False,
            )
            log_step_metrics(
                log_path,
                run_id="run-a",
                step=2,
                epoch=2,
                metrics={"loss": 0.5},
                improved=True,
            )
            from lpap.artifact_sync import upload_training_log_to_bucket

            captured: dict[str, bytes] = {}

            def batch(bucket, add=None, delete=None):
                for item, remote in add or []:
                    if isinstance(item, bytes):
                        captured[remote] = item
                    else:
                        captured[remote] = Path(item).read_bytes()

            with (
                patch.dict("os.environ", {"HF_TOKEN": "tok"}),
                patch("lpap.artifact_sync.apply_hf_token", return_value="tok"),
                patch("lpap.artifact_sync._require_write_token", return_value="tok"),
                patch("lpap.artifact_sync._resolve_bucket", return_value="user/bucket"),
                patch(
                    "lpap.artifact_sync._bucket_object_exists",
                    return_value=True,
                ),
                patch(
                    "lpap.artifact_sync.read_log_pointer",
                    return_value={"slot": 0, "sha256": "x", "size": 1},
                ),
                patch(
                    "lpap.artifact_sync._bucket_file_size",
                    side_effect=lambda *a, **k: log_path.stat().st_size,
                ),
                patch("huggingface_hub.HfApi") as api_cls,
            ):
                api_cls.return_value.batch_bucket_files.side_effect = batch
                remotes = upload_training_log_to_bucket(
                    log_path,
                    bucket="user/bucket",
                    project_root=root,
                )
            self.assertEqual(
                remotes[0],
                "training_logs/model.slot1.sqlite",
            )
            self.assertEqual(remotes[1], "training_logs/model.current.json")
            uploaded = root / "uploaded.sqlite"
            uploaded.write_bytes(captured["training_logs/model.slot1.sqlite"])
            with closing(sqlite3.connect(uploaded)) as connection:
                steps = [
                    row[0]
                    for row in connection.execute(
                        "SELECT step FROM step_metrics ORDER BY step"
                    )
                ]
            self.assertEqual(steps, [1, 2])
            pointer = captured["training_logs/model.current.json"]
            self.assertIn(b'"slot": 1', pointer)


class BranchTest(unittest.TestCase):
    def test_branch_copy_leaves_parent_intact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoints" / "parent.pt"
            log_path = root / "training_logs" / "parent.sqlite"
            model = nn.Linear(2, 1)
            run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="parent-run",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=1,
                    checkpoint_every=1,
                    log_every=1,
                ),
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-3),
                model_config={"width": 2},
            )
            info = run.resume_or_initialize()
            run.record_step(step=1, metrics={"loss": 1.25})
            before = {
                "checkpoint": _sha256(checkpoint),
                "log": _sha256(log_path),
                "wal": _sha256(Path(str(log_path) + "-wal")),
            }
            child_checkpoint = root / "checkpoints" / "child.pt"
            child_log = root / "training_logs" / "child.sqlite"
            lineage = materialize_branch(
                project_root=root,
                parent=RunParent(run_id="parent-run", checkpoint="parent.pt"),
                dest_checkpoint=child_checkpoint,
                dest_log=child_log,
                child_run_id="child-run",
            )
            child_model = nn.Linear(2, 1)
            child = TrainingRun(
                config=TrainingRunConfig(
                    run_id="child-run",
                    checkpoint_path=child_checkpoint,
                    log_path=child_log,
                    total_steps=2,
                    checkpoint_every=1,
                    log_every=1,
                ),
                model=child_model,
                optimizer=torch.optim.AdamW(child_model.parameters(), lr=1.0e-3),
                model_config={"width": 2},
            )
            resumed = child.resume_or_initialize()
            child.record_step(step=2, metrics={"loss": 0.5})
            after = {
                "checkpoint": _sha256(checkpoint),
                "log": _sha256(log_path),
                "wal": _sha256(Path(str(log_path) + "-wal")),
            }
            self.assertEqual(before, after)
            self.assertEqual(lineage["parent_checkpoint"], "parent.pt")
            self.assertEqual(resumed.run_id, "child-run")
            self.assertTrue(resumed.resumed)
            parent_history = load_metric_history(log_path, run_id=info.run_id)
            self.assertEqual([row["step"] for row in parent_history], [1])
            child_history = load_metric_history(child_log, run_id="child-run")
            self.assertEqual([row["step"] for row in child_history], [1, 2])
            with closing(
                sqlite3.connect(
                    f"{log_path.resolve().as_uri()}?mode=ro&immutable=1",
                    uri=True,
                )
            ) as connection:
                parent_ids = [
                    row[0] for row in connection.execute("SELECT run_id FROM runs")
                ]
            self.assertNotIn("child-run", parent_ids)


class ModelConfigMismatchTest(unittest.TestCase):
    def test_mismatch_on_resume_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoints" / "model.pt"
            log_path = root / "training_logs" / "model.sqlite"
            model = nn.Linear(2, 1)
            run = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=1,
                    checkpoint_every=1,
                ),
                model=model,
                model_config={"width": 2},
            )
            run.resume_or_initialize()
            run.record_step(step=1, metrics={"loss": 1.0})
            mismatched = TrainingRun(
                config=TrainingRunConfig(
                    run_id="run-a",
                    checkpoint_path=checkpoint,
                    log_path=log_path,
                    total_steps=3,
                    resume=True,
                ),
                model=nn.Linear(2, 1),
                model_config={"width": 3},
            )
            with self.assertRaisesRegex(ValueError, "model config differs"):
                mismatched.resume_or_initialize()
            payload = load_training_checkpoint(checkpoint)
            self.assertEqual(payload["step"], 1)


class StepOverrideTest(unittest.TestCase):
    def test_step_overrides_from_total(self) -> None:
        self.assertIsNone(step_overrides_from_total(None))
        self.assertEqual(
            step_overrides_from_total(4, ("autoencoder",)),
            {"autoencoder": 4},
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            step_overrides_from_total(0)


if __name__ == "__main__":
    unittest.main()
