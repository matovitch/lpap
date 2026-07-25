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
    packaged_default_storage_config,
    storage_config_from_file,
)


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

    def test_packaged_defaults(self) -> None:
        config = packaged_default_storage_config()
        self.assertEqual(config.artifacts.bucket, "matovitch/lpap-molab-artifacts")
        self.assertEqual(config.images.bucket, "matovitch/lpap-images")
        self.assertEqual(config.images.remote_zst, "images_32x32_gray.pt.zst")
        self.assertEqual(config.images.local_pt, "data/images_32x32_gray.pt")
        self.assertIn(".hf_token", config.auth.token_files)

    def test_load_project_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "storage.toml").write_text(
                textwrap.dedent(
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
                ),
                encoding="utf-8",
            )
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
            config = load_storage_config()
            self.assertEqual(config.artifacts.bucket, "env/artifacts")
            self.assertEqual(config.images.bucket, "env/images")

    def test_missing_project_toml_falls_back_to_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_storage_config(temp_dir)
            self.assertEqual(
                config.artifacts.bucket,
                packaged_default_storage_config().artifacts.bucket,
            )

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
