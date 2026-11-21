from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from lpap.training_configs import (
    STAMP,
    extract_training_configs,
    write_training_configs,
)

_PIPELINE = b"""\
[pipeline]
teacher_configs = ["configs/training/teacher_a.toml"]
autoencoder_config = "configs/training/ae.toml"
"""


def _tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TrainingConfigsTest(unittest.TestCase):
    def test_extract_keeps_only_top_level_training_tomls(self) -> None:
        archive = _tarball(
            {
                "lpap-abc/configs/training/retrain_image_pipeline.toml": _PIPELINE,
                "lpap-abc/configs/training/ae.toml": b"x=1",
                "lpap-abc/configs/training/generated/ae-v3.toml": b"x=2",
                "lpap-abc/configs/storage.toml": b"x=3",
                "lpap-abc/test/resources/configs/training/ae.toml": b"x=4",
            }
        )
        self.assertEqual(
            extract_training_configs(archive),
            {"retrain_image_pipeline.toml": _PIPELINE, "ae.toml": b"x=1"},
        )

    def test_write_replaces_stale_tomls_and_keeps_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dest = root / "configs" / "training"
            (dest / "generated").mkdir(parents=True)
            (dest / "generated" / "ae-v3.toml").write_text("keep")
            (dest / "old.toml").write_text("stale")
            configs = {
                "retrain_image_pipeline.toml": _PIPELINE,
                "teacher_a.toml": b"t=1",
                "ae.toml": b"a=1",
            }
            write_training_configs(root, configs, "abc123")
            self.assertEqual(
                sorted(p.name for p in dest.glob("*.toml")), sorted(configs)
            )
            self.assertEqual((dest / "generated" / "ae-v3.toml").read_text(), "keep")
            self.assertEqual((dest / STAMP).read_text(), "abc123\n")

    def test_write_fails_loudly_on_missing_referenced_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                write_training_configs(
                    temp_dir,
                    {"retrain_image_pipeline.toml": _PIPELINE, "ae.toml": b""},
                    "abc123",
                )
            self.assertIn("teacher_a.toml", str(ctx.exception))
            self.assertFalse((Path(temp_dir) / "configs").exists())

    def test_write_fails_without_pipeline_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                write_training_configs(temp_dir, {"ae.toml": b""}, "abc123")


if __name__ == "__main__":
    unittest.main()
