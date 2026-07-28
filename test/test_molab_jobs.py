from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from molab.jobs import (
    AE_ENERGY_BANK_BG_STEM,
    AE_ENERGY_BANK_SCRIPT,
    ae_energy_bank_worker_source,
    flow_energy_bank_worker_source,
    launch_ae_energy_bank_bg,
    launch_flow_energy_bank_bg,
    lpap_teacher_worker_source,
)


class MolabJobsTest(unittest.TestCase):
    def test_ae_worker_source_uses_bank_flows(self) -> None:
        source = ae_energy_bank_worker_source(
            target_steps=20000,
            project_root="/marimo",
            upload_artifacts_on_checkpoint=True,
            notify_on_finished=True,
            comment="unit",
        )
        self.assertIn("TARGET = 20000", source)
        self.assertIn("Path('/marimo')", source)
        self.assertIn("upload_artifacts_on_checkpoint=True", source)
        self.assertIn("notify_on_finished=True", source)
        self.assertIn("comment='unit'", source)
        self.assertIn('name="c128"', source)
        self.assertIn('name="c256"', source)
        self.assertIn("surrogate_c256_k4.pt", source)
        self.assertIn("decoder_c256_k4.pt", source)
        self.assertIn("image_to_energy_energy_bank.pt", source)
        self.assertIn("energy_to_image_energy_bank.pt", source)
        self.assertIn("resume_from_checkpoint=False", source)
        self.assertNotIn('image_to_energy_checkpoint_name="image_to_energy.pt"', source)
        compile(source, "<ae_worker>", "exec")

    def test_ae_worker_source_resume(self) -> None:
        source = ae_energy_bank_worker_source(
            target_steps=36550,
            resume_from_checkpoint=True,
        )
        self.assertIn("resume_from_checkpoint=True", source)
        self.assertIn("TARGET = 36550", source)
        compile(source, "<ae_resume>", "exec")

    def test_lpap_teacher_worker_source(self) -> None:
        source = lpap_teacher_worker_source(
            backend_kind="surrogate",
            config_relpath="configs/training/surrogate_c512.toml",
            target_steps=10000,
        )
        self.assertIn("TARGET = 10000", source)
        self.assertIn("surrogate_c512.toml", source)
        self.assertIn("SURROGATE_DONE", source)
        self.assertIn("KIND = 'surrogate'", source)
        compile(source, "<surr_worker>", "exec")

    def test_flow_worker_source_i2e(self) -> None:
        source = flow_energy_bank_worker_source(
            "image_to_energy_energy_bank",
            target_steps=10000,
            project_root="/marimo",
            energy_bank_path="data/encoded_energies_ae_best.pt",
        )
        self.assertIn("TARGET = 10000", source)
        self.assertIn("image_to_energy_energy_bank", source)
        self.assertIn("encoded_energies_ae_best.pt", source)
        self.assertIn("I2E_ENERGY_BANK_DONE", source)
        self.assertIn("upload_artifacts_on_checkpoint=True", source)
        compile(source, "<i2e_worker>", "exec")

    def test_flow_worker_source_e2i(self) -> None:
        source = flow_energy_bank_worker_source(
            "energy_to_image_energy_bank",
            target_steps=10000,
            project_root="/marimo",
        )
        self.assertIn("E2I_ENERGY_BANK_DONE", source)
        self.assertIn("energy_to_image_energy_bank", source)
        compile(source, "<e2i_worker>", "exec")

    def test_launch_ae_writes_script_and_spawns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.dict(
                    os.environ,
                    {
                        "HF_TOKEN": "tok",
                        "PUSHOVER_TOKEN": "app",
                        "PUSHOVER_USER": "user",
                    },
                    clear=False,
                ),
                patch(
                    "molab.jobs.spawn_detached_python",
                    return_value={
                        "pid": 99,
                        "script_path": str(root / "training_logs" / AE_ENERGY_BANK_SCRIPT),
                        "cwd": str(root),
                        "pid_path": str(
                            root / "training_logs" / f"{AE_ENERGY_BANK_BG_STEM}.pid"
                        ),
                        "log_path": str(
                            root / "training_logs" / f"{AE_ENERGY_BANK_BG_STEM}.log"
                        ),
                    },
                ) as spawn,
            ):
                result = launch_ae_energy_bank_bg(
                    root, target_steps=1000, require_secrets=True
                )
            script = root / "training_logs" / AE_ENERGY_BANK_SCRIPT
            self.assertTrue(script.is_file())
            self.assertIn("TARGET = 1000", script.read_text(encoding="utf-8"))
            spawn.assert_called_once()
            self.assertEqual(result["pid"], 99)
            self.assertEqual(result["target_steps"], 1000)
            self.assertEqual(result["bg_stem"], AE_ENERGY_BANK_BG_STEM)

    def test_launch_flow_writes_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False),
                patch(
                    "molab.jobs.spawn_detached_python",
                    return_value={
                        "pid": 7,
                        "script_path": "x",
                        "cwd": str(root),
                        "pid_path": "y",
                        "log_path": "z",
                    },
                ),
            ):
                result = launch_flow_energy_bank_bg(
                    "image_to_energy_energy_bank",
                    root,
                    target_steps=10000,
                    require_secrets=True,
                    notify_on_finished=False,
                )
            script = (
                root
                / "training_logs"
                / "train_image_to_energy_energy_bank_bg.py"
            )
            self.assertTrue(script.is_file())
            self.assertEqual(result["pid"], 7)
            self.assertEqual(result["bg_stem"], "image_to_energy_energy_bank_bg")


if __name__ == "__main__":
    unittest.main()
