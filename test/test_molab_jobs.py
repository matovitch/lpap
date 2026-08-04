from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from molab.jobs import (
    AE_BIDIR_FLOW_BG_STEM,
    AE_BIDIR_FLOW_SCRIPT,
    FLOW_BG_STEM,
    FLOW_SCRIPT,
    ae_bidirectional_flow_worker_source,
    image_energy_flow_worker_source,
    launch_ae_bidirectional_flow_bg,
    launch_image_energy_flow_bg,
    lpap_teacher_worker_source,
)


class MolabJobsTest(unittest.TestCase):
    def test_ae_bidirectional_flow_worker_source(self) -> None:
        source = ae_bidirectional_flow_worker_source(
            target_steps=20000,
            project_root="/marimo",
            comment="unit-bidir",
        )
        self.assertIn("TARGET = 20000", source)
        self.assertIn('flow_checkpoint_name="image_energy_flow.pt"', source)
        self.assertIn("surrogate_c128_k16.pt", source)
        self.assertIn("decoder_c256_k24.pt", source)
        self.assertIn("surrogate_c512_k32.pt", source)
        self.assertIn("decoder_c512_k32.pt", source)
        self.assertIn('name="c128_k16"', source)
        self.assertIn('name="c256_k24"', source)
        self.assertIn('name="c512_k32"', source)
        self.assertIn("comment='unit-bidir'", source)
        self.assertIn("AE_TRI_BIDIR_FLOW_DONE", source)
        self.assertIn("image_autoencoder_tri_flow", source)
        compile(source, "<ae_bidir_worker>", "exec")

    def test_ae_worker_source_resume(self) -> None:
        source = ae_bidirectional_flow_worker_source(
            target_steps=36550,
            resume_from_checkpoint=True,
        )
        self.assertIn("resume_from_checkpoint=True", source)
        self.assertIn("TARGET = 36550", source)
        compile(source, "<ae_resume>", "exec")

    def test_lpap_teacher_worker_source(self) -> None:
        source = lpap_teacher_worker_source(
            backend_kind="surrogate",
            config_relpath="configs/training/teacher_c512_k32.toml",
            target_steps=10000,
        )
        self.assertIn("TARGET = 10000", source)
        self.assertIn("teacher_c512_k32.toml", source)
        self.assertIn("project_teacher_config", source)
        self.assertIn("SURROGATE_DONE", source)
        self.assertIn("KIND = 'surrogate'", source)
        compile(source, "<surr_worker>", "exec")

    def test_flow_worker_source_signed_lognormal_prior(self) -> None:
        source = image_energy_flow_worker_source(
            target_steps=10000,
            project_root="/marimo",
        )
        self.assertIn("TARGET = 10000", source)
        self.assertIn("image_energy_flow", source)
        self.assertIn("default_image_energy_flow_training_config", source)
        self.assertIn("IMAGE_ENERGY_FLOW_DONE", source)
        self.assertIn("upload_artifacts_on_checkpoint=True", source)
        self.assertIn("resume_from_checkpoint=False", source)
        compile(source, "<flow_worker>", "exec")

    def test_flow_worker_source_resume(self) -> None:
        source = image_energy_flow_worker_source(
            target_steps=20000,
            resume_from_checkpoint=True,
        )
        self.assertIn("TARGET = 20000", source)
        self.assertIn("resume_from_checkpoint=True", source)
        compile(source, "<flow_resume>", "exec")

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
                        "script_path": str(
                            root / "training_logs" / AE_BIDIR_FLOW_SCRIPT
                        ),
                        "cwd": str(root),
                        "pid_path": str(
                            root / "training_logs" / f"{AE_BIDIR_FLOW_BG_STEM}.pid"
                        ),
                        "log_path": str(
                            root / "training_logs" / f"{AE_BIDIR_FLOW_BG_STEM}.log"
                        ),
                    },
                ) as spawn,
            ):
                result = launch_ae_bidirectional_flow_bg(
                    root, target_steps=1000, require_secrets=True
                )
            script = root / "training_logs" / AE_BIDIR_FLOW_SCRIPT
            self.assertTrue(script.is_file())
            self.assertIn("TARGET = 1000", script.read_text(encoding="utf-8"))
            spawn.assert_called_once()
            self.assertEqual(result["pid"], 99)
            self.assertEqual(result["target_steps"], 1000)
            self.assertEqual(result["bg_stem"], AE_BIDIR_FLOW_BG_STEM)

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
                result = launch_image_energy_flow_bg(
                    root,
                    target_steps=10000,
                    require_secrets=True,
                    notify_on_finished=False,
                )
            script = root / "training_logs" / FLOW_SCRIPT
            self.assertTrue(script.is_file())
            self.assertEqual(result["pid"], 7)
            self.assertEqual(result["bg_stem"], FLOW_BG_STEM)


if __name__ == "__main__":
    unittest.main()
