from __future__ import annotations

import base64
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "skills"
    / "molab-workflow"
    / "scripts"
)


class MolabExportPicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SCRIPTS))
        from molab_export_pics import (  # noqa: WPS433
            latest_run,
            load_hf_token,
            normalize_suffix,
            parse_args,
            resume_steps,
            series,
        )
        from molab_file_api import decode_file_details  # noqa: WPS433

        cls.normalize_suffix = staticmethod(normalize_suffix)
        cls.parse_args = staticmethod(parse_args)
        cls.load_hf_token = staticmethod(load_hf_token)
        cls.decode_file_details = staticmethod(decode_file_details)
        cls.latest_run = staticmethod(latest_run)
        cls.series = staticmethod(series)
        cls.resume_steps = staticmethod(resume_steps)

    def test_normalize_suffix(self) -> None:
        self.assertEqual(self.normalize_suffix("v3"), "-v3")
        self.assertEqual(self.normalize_suffix("-v3"), "-v3")
        self.assertEqual(self.normalize_suffix(""), "")
        self.assertEqual(self.normalize_suffix(None), "")
        self.assertEqual(self.normalize_suffix("-"), "")

    def test_parse_args_run_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.parse_args(
                [
                    "--run",
                    "v3",
                    "--output-dir",
                    str(root / "out"),
                    "--project-root",
                    str(root),
                ]
            )
        self.assertEqual(args.suffix, "-v3")
        self.assertEqual(args.pairs, ("c128_k16", "c256_k24", "c512_k32"))

    def test_parse_args_dashed_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.parse_args(
                [
                    "--suffix",
                    "-v3",
                    "--output-dir",
                    str(root / "out"),
                    "--project-root",
                    str(root),
                ]
            )
        self.assertEqual(args.suffix, "-v3")

    def test_decode_file_details_checks_size(self) -> None:
        payload = b"sqlite-bytes"
        response = {
            "isBase64": True,
            "isTooLarge": False,
            "contents": base64.b64encode(payload).decode(),
            "file": {"size": len(payload), "path": "/marimo/training_logs/x.sqlite"},
        }
        self.assertEqual(self.decode_file_details(response), payload)

        truncated = dict(response)
        truncated["file"] = {"size": len(payload) + 10}
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            self.decode_file_details(truncated)

        missing = {"detail": "[Errno 2] No such file or directory: '/marimo/x'"}
        with self.assertRaises(FileNotFoundError):
            self.decode_file_details(missing)

        huge = {"isTooLarge": True, "file": {"path": "/marimo/x.pt", "size": 612877183}}
        with self.assertRaisesRegex(RuntimeError, "isTooLarge"):
            self.decode_file_details(huge)

    def test_load_hf_token_from_flattened_env(self) -> None:
        import os

        previous = {
            key: os.environ.pop(key, None)
            for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SECRETS_TOML", "LPAP_SECRETS_TOML")
        }
        try:
            os.environ["SECRETS_TOML"] = (
                "# comment  [huggingface] token = \"hf_testtoken123\"  "
                "[pushover] user = \"u\" token = \"p\""
            )
            token = self.load_hf_token()
            self.assertEqual(token, "hf_testtoken123")
            self.assertEqual(os.environ.get("HF_TOKEN"), "hf_testtoken123")
        finally:
            for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SECRETS_TOML", "LPAP_SECRETS_TOML"):
                os.environ.pop(key, None)
            for key, value in previous.items():
                if value is not None:
                    os.environ[key] = value

    def test_latest_run_and_resume_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.sqlite"
            con = sqlite3.connect(path)
            con.executescript(
                """
                create table runs (
                    run_id text primary key,
                    updated_at text,
                    status text,
                    config_json text
                );
                create table step_metrics (
                    run_id text,
                    step integer,
                    metric_name text,
                    metric_value real
                );
                create table run_events (
                    event_id integer primary key,
                    run_id text,
                    kind text,
                    step integer
                );
                insert into runs values ('empty', '2026-01-02', 'done', '{}');
                insert into runs values ('live', '2026-01-01', 'running', '{}');
                insert into step_metrics values ('live', 10, 'loss', 0.2);
                insert into step_metrics values ('live', 20, 'loss', 0.1);
                insert into run_events values (1, 'live', 'resume', 10);
                """
            )
            con.commit()
            # updated_at desc would pick 'empty' first; latest_run skips it.
            self.assertEqual(self.latest_run(con), "live")
            steps, values = self.series(con, "live", "loss")
            self.assertEqual(list(steps), [10, 20])
            self.assertEqual(list(values), [0.2, 0.1])
            self.assertEqual(self.resume_steps(con, "live"), [10])
            con.close()


if __name__ == "__main__":
    unittest.main()
