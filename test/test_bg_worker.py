from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from molab.bg_worker import (
    BackgroundWorkerError,
    bg_worker_status,
    last_bg_log_step,
    parse_bg_log_steps,
    refuse_if_alive,
    require_env_keys,
    spawn_detached_python,
)


class BgWorkerTest(unittest.TestCase):
    def test_parse_bg_log_steps(self) -> None:
        text = "device=cuda\nstep=10 loss=1\nstep=20 val=0.1\nother\n"
        self.assertEqual(parse_bg_log_steps(text), [10, 20])

    def test_last_bg_log_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "w.log"
            path.write_text("step=3\nstep=9\n", encoding="utf-8")
            self.assertEqual(last_bg_log_step(path), 9)
            self.assertIsNone(last_bg_log_step(Path(temp_dir) / "missing.log"))

    def test_refuse_if_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = Path(temp_dir) / "w.pid"
            pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
            with self.assertRaises(BackgroundWorkerError):
                refuse_if_alive(pid_path)
            status = bg_worker_status(pid_path)
            self.assertTrue(status["alive"])
            self.assertEqual(status["pid"], os.getpid())

    def test_require_env_keys(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False):
            env = require_env_keys(("HF_TOKEN",))
            self.assertEqual(env["HF_TOKEN"], "tok")
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
            with self.assertRaises(BackgroundWorkerError):
                require_env_keys(("HF_TOKEN",))

    def test_spawn_detached_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "job.py"
            script.write_text("print('hi')\n", encoding="utf-8")
            pid_path = root / "job.pid"
            log_path = root / "job.log"
            mock_proc = MagicMock()
            mock_proc.pid = 4242
            with patch("molab.bg_worker.subprocess.Popen", return_value=mock_proc) as popen:
                result = spawn_detached_python(
                    script,
                    cwd=root,
                    pid_path=pid_path,
                    log_path=log_path,
                    env={"HF_TOKEN": "x"},
                )
            self.assertEqual(result["pid"], 4242)
            self.assertEqual(pid_path.read_text(encoding="utf-8").strip(), "4242")
            popen.assert_called_once()
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["cwd"], str(root))
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["HF_TOKEN"], "x")


if __name__ == "__main__":
    unittest.main()
