from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from lpap.storage import (
    infer_project_root_from_checkpoint,
    load_storage_config,
    storage_config_from_file,
    storage_config_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_SAMPLE_TOML = textwrap.dedent(
    """\
    [artifacts]
    bucket = "org/custom-artifacts"

    [images]
    bucket = "org/custom-images"
    remote_zst = "remote.pt.zst"
    local_pt = "data/local.pt"
    local_zst = "data/local.pt.zst"

    [auth]
    token_files = ["tok.txt"]
    """
)


def _write_storage_toml(root: Path, contents: str = _SAMPLE_TOML) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "storage.toml"
    path.write_text(contents, encoding="utf-8")
    return path


class StorageConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patch = patch.dict(
            os.environ,
            {"LPAP_ARTIFACTS_BUCKET": "", "LPAP_IMAGES_BUCKET": ""},
            clear=False,
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_repo_storage_toml(self) -> None:
        config = load_storage_config(REPO_ROOT)
        self.assertEqual(config.artifacts.bucket, "matovitch/lpap-molab-artifacts")
        self.assertEqual(config.images.bucket, "matovitch/lpap-images")
        self.assertEqual(config.images.remote_zst, "images_32x32_gray.pt.zst")
        self.assertEqual(config.images.local_pt, "data/images_32x32_gray.pt")
        self.assertIn(".hf_token", config.auth.token_files)

    def test_load_project_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_storage_toml(root)
            config = load_storage_config(root)
            self.assertEqual(config.artifacts.bucket, "org/custom-artifacts")
            self.assertEqual(config.images.bucket, "org/custom-images")
            self.assertEqual(config.auth.token_files, ("tok.txt",))

    def test_env_overrides_buckets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LPAP_ARTIFACTS_BUCKET": "env/artifacts",
                "LPAP_IMAGES_BUCKET": "env/images",
            },
            clear=False,
        ):
            config = load_storage_config(REPO_ROOT)
            self.assertEqual(config.artifacts.bucket, "env/artifacts")
            self.assertEqual(config.images.bucket, "env/images")

    def test_missing_project_toml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError) as ctx:
                load_storage_config(root)
            self.assertIn(str(storage_config_path(root)), str(ctx.exception))

    def test_infer_project_root_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt = root / "checkpoints" / "model.pt"
            ckpt.parent.mkdir()
            ckpt.write_bytes(b"x")
            self.assertEqual(infer_project_root_from_checkpoint(ckpt), root)

    def test_storage_config_from_file_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.toml"
            path.write_text(
                "[artifacts]\nbucket = \"\"\n"
                "[images]\nbucket = \"b\"\nremote_zst = \"r\"\n"
                "local_pt = \"p\"\nlocal_zst = \"z\"\n"
                "[auth]\ntoken_files = [\".hf_token\"]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                storage_config_from_file(path)


if __name__ == "__main__":
    unittest.main()
