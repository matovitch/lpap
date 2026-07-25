from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lpap.artifact_sync import (
    apply_hf_token,
    bucket_uri,
    default_artifact_pairs,
    default_artifacts_bucket,
    download_files,
    download_training_artifacts,
    resolve_hf_token,
    upload_checkpoint_artifacts,
    upload_files,
    upload_training_artifacts,
)
from lpap.storage import load_storage_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class ArtifactSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patch = patch.dict(
            os.environ,
            {"LPAP_ARTIFACTS_BUCKET": "", "LPAP_IMAGES_BUCKET": ""},
            clear=False,
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_bucket_uri(self) -> None:
        self.assertEqual(
            bucket_uri("org/bucket", "/checkpoints/a.pt"),
            "hf://buckets/org/bucket/checkpoints/a.pt",
        )

    def test_default_artifacts_bucket_from_repo_config(self) -> None:
        self.assertEqual(
            default_artifacts_bucket(REPO_ROOT),
            load_storage_config(REPO_ROOT).artifacts.bucket,
        )
        self.assertEqual(
            default_artifacts_bucket(REPO_ROOT),
            "matovitch/lpap-molab-artifacts",
        )

    def test_default_artifacts_bucket_requires_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                default_artifacts_bucket(temp_dir)
        with self.assertRaises(ValueError):
            upload_files([], bucket=None, project_root=None)

    def test_resolve_hf_token_prefers_explicit_then_env(self) -> None:
        self.assertEqual(resolve_hf_token(token=" explicit "), "explicit")
        with patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=False):
            self.assertEqual(resolve_hf_token(), "env-token")
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            self.assertIsNone(resolve_hf_token())

    def test_apply_hf_token_sets_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            apply_hf_token(token="abc")
            self.assertEqual(os.environ["HF_TOKEN"], "abc")

    def test_default_artifact_pairs(self) -> None:
        root = Path("/tmp/project")
        upload, download = default_artifact_pairs(
            root,
            checkpoint_names=("surrogate_synthetic.pt",),
            log_names=("surrogate.sqlite",),
        )
        self.assertEqual(
            upload,
            [
                (
                    root / "checkpoints" / "surrogate_synthetic.pt",
                    "checkpoints/surrogate_synthetic.pt",
                ),
                (
                    root / "training_logs" / "surrogate.sqlite",
                    "training_logs/surrogate.sqlite",
                ),
            ],
        )
        self.assertEqual(
            download,
            [
                (
                    "checkpoints/surrogate_synthetic.pt",
                    root / "checkpoints" / "surrogate_synthetic.pt",
                ),
                (
                    "training_logs/surrogate.sqlite",
                    root / "training_logs" / "surrogate.sqlite",
                ),
            ],
        )

    def test_upload_files_requires_token_and_calls_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local = Path(temp_dir) / "a.pt"
            local.write_bytes(b"ckpt")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                with self.assertRaises(RuntimeError):
                    upload_files(
                        [(local, "checkpoints/a.pt")],
                        bucket="user/bucket",
                    )

            mock_api = MagicMock()
            with (
                patch("lpap.artifact_sync.apply_hf_token", return_value="tok"),
                patch("huggingface_hub.HfApi", return_value=mock_api),
            ):
                remotes = upload_files(
                    [(local, "checkpoints/a.pt")], bucket="user/bucket"
                )
            self.assertEqual(remotes, ["checkpoints/a.pt"])
            mock_api.batch_bucket_files.assert_called_once_with(
                "user/bucket",
                add=[(str(local), "checkpoints/a.pt")],
            )

    def test_download_files_uses_hf_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dest = Path(temp_dir) / "checkpoints" / "a.pt"
            mock_fs = MagicMock()
            mock_fs.exists.return_value = True

            def _get(uri: str, local: str) -> None:
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                Path(local).write_bytes(b"data")

            mock_fs.get.side_effect = _get
            bucket = default_artifacts_bucket(REPO_ROOT)
            with (
                patch("lpap.artifact_sync.apply_hf_token", return_value=None),
                patch("huggingface_hub.HfFileSystem", return_value=mock_fs),
            ):
                paths = download_files(
                    [("checkpoints/a.pt", dest)],
                    bucket=bucket,
                )
            self.assertEqual(paths, [dest])
            self.assertEqual(dest.read_bytes(), b"data")
            mock_fs.exists.assert_called_once_with(
                bucket_uri(bucket, "checkpoints/a.pt")
            )

    def test_upload_and_download_training_artifacts_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt = root / "checkpoints" / "surrogate_synthetic.pt"
            log = root / "training_logs" / "surrogate.sqlite"
            ckpt.parent.mkdir(parents=True)
            log.parent.mkdir(parents=True)
            ckpt.write_bytes(b"ckpt")
            log.write_bytes(b"log")

            with patch(
                "lpap.artifact_sync.upload_files", return_value=["checkpoints/x"]
            ) as upload_mock:
                remotes = upload_training_artifacts(root)
            self.assertEqual(remotes, ["checkpoints/x"])
            upload_mock.assert_called_once()
            pairs = upload_mock.call_args.args[0]
            self.assertEqual(
                [path.name for path, _ in pairs],
                ["surrogate_synthetic.pt", "surrogate.sqlite"],
            )

            with patch(
                "lpap.artifact_sync.download_files",
                return_value=[ckpt, log],
            ) as download_mock:
                paths = download_training_artifacts(root)
            self.assertEqual(paths, [ckpt, log])
            download_mock.assert_called_once()

    def test_upload_checkpoint_artifacts_uses_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt = root / "checkpoints" / "model.pt"
            log = root / "training_logs" / "model.sqlite"
            ckpt.parent.mkdir(parents=True)
            log.parent.mkdir(parents=True)
            ckpt.write_bytes(b"ckpt")
            log.write_bytes(b"log")
            with patch(
                "lpap.artifact_sync.upload_files",
                return_value=["checkpoints/model.pt", "training_logs/model.sqlite"],
            ) as upload_mock:
                remotes = upload_checkpoint_artifacts(
                    checkpoint_path=ckpt, log_path=log
                )
            self.assertEqual(
                remotes, ["checkpoints/model.pt", "training_logs/model.sqlite"]
            )
            pairs = upload_mock.call_args.args[0]
            self.assertEqual(
                [(path.name, remote) for path, remote in pairs],
                [
                    ("model.pt", "checkpoints/model.pt"),
                    ("model.sqlite", "training_logs/model.sqlite"),
                ],
            )
            self.assertEqual(
                upload_mock.call_args.kwargs["project_root"],
                root,
            )


if __name__ == "__main__":
    unittest.main()
