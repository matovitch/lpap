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
    launch_ae_energy_bank_bg,
)


class MolabJobsTest(unittest.TestCase):
    def test_worker_source_embeds_target(self) -> None:
        source = ae_energy_bank_worker_source(
            target_steps=58200,
            project_root="/marimo",
            upload_artifacts_on_checkpoint=True,
            notify_on_finished=True,
            comment="unit",
        )
        self.assertIn("TARGET = 58200", source)
        self.assertIn("Path('/marimo')", source)
        self.assertIn("upload_artifacts_on_checkpoint=True", source)
        self.assertIn("notify_on_finished=True", source)
        self.assertIn("comment='unit'", source)
        self.assertIn('name="c128"', source)
        self.assertIn('name="c256"', source)
        self.assertIn("surrogate_c256.pt", source)
        self.assertIn("decoder_c256.pt", source)
        compile(source, "<ae_worker>", "exec")

    def test_launch_writes_script_and_spawns(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
