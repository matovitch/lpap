from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lpap.artifact_sync import (
    apply_hf_token,
    bucket_uri,
    checkpoint_pointer_remote_path,
    checkpoint_slot_remote_path,
    checkpoint_stem,
    default_artifact_pairs,
    default_artifacts_bucket,
    download_files,
    download_training_artifacts,
    ensure_checkpoint,
    ensure_project_artifact,
    is_canonical_checkpoint_remote,
    resolve_hf_token,
    sha256_file,
    upload_checkpoint_artifacts,
    upload_checkpoint_to_bucket,
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

    def test_checkpoint_path_helpers(self) -> None:
        self.assertEqual(checkpoint_stem("image_energy_flow.pt"), "image_energy_flow")
        self.assertEqual(
            checkpoint_slot_remote_path("image_energy_flow", 0),
            "checkpoints/image_energy_flow.slot0.pt",
        )
        self.assertEqual(
            checkpoint_pointer_remote_path("image_energy_flow"),
            "checkpoints/image_energy_flow.current.json",
        )
        self.assertTrue(is_canonical_checkpoint_remote("checkpoints/foo.pt"))
        self.assertFalse(
            is_canonical_checkpoint_remote("checkpoints/foo.slot0.pt")
        )
        self.assertFalse(is_canonical_checkpoint_remote("data/bank.pt"))

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "a.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
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

            def _get(uri: str, local: str) -> None:
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                Path(local).write_bytes(b"data")

            mock_fs.get.side_effect = _get
            bucket = default_artifacts_bucket(REPO_ROOT)
            with (
                patch("lpap.artifact_sync.apply_hf_token", return_value=None),
                patch(
                    "lpap.artifact_sync._bucket_object_exists",
                    return_value=True,
                ) as exists_mock,
                patch("huggingface_hub.HfFileSystem", return_value=mock_fs),
            ):
                paths = download_files(
                    [("checkpoints/a.pt", dest)],
                    bucket=bucket,
                )
            self.assertEqual(paths, [dest])
            self.assertEqual(dest.read_bytes(), b"data")
            exists_mock.assert_called_once_with(
                bucket, "checkpoints/a.pt", token=None
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

            with (
                patch(
                    "lpap.artifact_sync.upload_checkpoint_to_bucket",
                    return_value=[
                        "checkpoints/surrogate_synthetic.slot0.pt",
                        "checkpoints/surrogate_synthetic.current.json",
                    ],
                ) as ckpt_mock,
                patch(
                    "lpap.artifact_sync.upload_files",
                    return_value=["training_logs/surrogate.sqlite"],
                ) as upload_mock,
            ):
                remotes = upload_training_artifacts(root)
            self.assertEqual(
                remotes,
                [
                    "checkpoints/surrogate_synthetic.slot0.pt",
                    "checkpoints/surrogate_synthetic.current.json",
                    "training_logs/surrogate.sqlite",
                ],
            )
            ckpt_mock.assert_called_once()
            upload_mock.assert_called_once()

            with patch(
                "lpap.artifact_sync.ensure_checkpoint",
                return_value=ckpt,
            ) as ensure_mock:
                with patch(
                    "lpap.artifact_sync.download_files",
                    return_value=[log],
                ) as download_mock:
                    paths = download_training_artifacts(root)
            self.assertEqual(paths, [ckpt, log])
            ensure_mock.assert_called_once()
            download_mock.assert_called_once()

    def test_upload_checkpoint_to_bucket_first_slot_and_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local = Path(temp_dir) / "model.pt"
            local.write_bytes(b"ckpt-bytes")
            mock_api = MagicMock()
            with (
                patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False),
                patch("lpap.artifact_sync.apply_hf_token", return_value="tok"),
                patch("huggingface_hub.HfApi", return_value=mock_api),
                patch(
                    "lpap.artifact_sync._bucket_object_exists",
                    return_value=False,
                ),
                patch(
                    "lpap.artifact_sync._bucket_file_size",
                    return_value=local.stat().st_size,
                ),
            ):
                remotes = upload_checkpoint_to_bucket(
                    local, bucket="user/bucket", canonical_name="model.pt"
                )
            self.assertEqual(
                remotes,
                [
                    "checkpoints/model.slot0.pt",
                    "checkpoints/model.current.json",
                ],
            )
            # First call: slot upload
            first = mock_api.batch_bucket_files.call_args_list[0]
            self.assertEqual(first.args[0], "user/bucket")
            self.assertEqual(
                first.kwargs["add"],
                [(str(local), "checkpoints/model.slot0.pt")],
            )
            # Second call: pointer JSON
            second = mock_api.batch_bucket_files.call_args_list[1]
            pointer_bytes, pointer_remote = second.kwargs["add"][0]
            self.assertEqual(pointer_remote, "checkpoints/model.current.json")
            pointer = json.loads(pointer_bytes.decode())
            self.assertEqual(pointer["slot"], 0)
            self.assertEqual(pointer["size"], local.stat().st_size)
            self.assertEqual(pointer["sha256"], sha256_file(local))

    def test_upload_checkpoint_to_bucket_alternates_and_deletes_old(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local = Path(temp_dir) / "model.pt"
            local.write_bytes(b"ckpt-bytes-2")
            mock_api = MagicMock()
            pointer = {"slot": 0, "sha256": "x", "size": 1, "name": "model.pt"}
            with (
                patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False),
                patch("lpap.artifact_sync.apply_hf_token", return_value="tok"),
                patch("huggingface_hub.HfApi", return_value=mock_api),
                patch(
                    "lpap.artifact_sync._bucket_object_exists",
                    return_value=True,
                ),
                patch(
                    "lpap.artifact_sync.read_checkpoint_pointer",
                    return_value=pointer,
                ),
                patch(
                    "lpap.artifact_sync._bucket_file_size",
                    return_value=local.stat().st_size,
                ),
            ):
                remotes = upload_checkpoint_to_bucket(
                    local, bucket="user/bucket", canonical_name="model.pt"
                )
            self.assertEqual(remotes[0], "checkpoints/model.slot1.pt")
            # Last batch call should delete slot0
            delete_calls = [
                c
                for c in mock_api.batch_bucket_files.call_args_list
                if c.kwargs.get("delete")
            ]
            self.assertTrue(delete_calls)
            self.assertEqual(
                delete_calls[-1].kwargs["delete"],
                ["checkpoints/model.slot0.pt"],
            )

    def test_upload_checkpoint_to_bucket_size_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local = Path(temp_dir) / "model.pt"
            local.write_bytes(b"ckpt")
            mock_api = MagicMock()
            with (
                patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False),
                patch("lpap.artifact_sync.apply_hf_token", return_value="tok"),
                patch("huggingface_hub.HfApi", return_value=mock_api),
                patch(
                    "lpap.artifact_sync._bucket_object_exists",
                    return_value=False,
                ),
                patch("lpap.artifact_sync._bucket_file_size", return_value=999),
            ):
                with self.assertRaises(RuntimeError):
                    upload_checkpoint_to_bucket(
                        local, bucket="user/bucket", canonical_name="model.pt"
                    )
            # Pointer must not be written on verify failure
            self.assertEqual(mock_api.batch_bucket_files.call_count, 1)

    def test_upload_checkpoint_artifacts_sqlite_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt = root / "checkpoints" / "model.pt"
            log = root / "training_logs" / "model.sqlite"
            ckpt.parent.mkdir(parents=True)
            log.parent.mkdir(parents=True)
            ckpt.write_bytes(b"ckpt")
            log.write_bytes(b"log")
            with (
                patch(
                    "lpap.artifact_sync.upload_checkpoint_to_bucket",
                    return_value=[
                        "checkpoints/model.slot0.pt",
                        "checkpoints/model.current.json",
                    ],
                ),
                patch(
                    "lpap.artifact_sync.upload_files",
                    side_effect=RuntimeError("sqlite boom"),
                ),
            ):
                with self.assertWarns(UserWarning):
                    remotes = upload_checkpoint_artifacts(
                        checkpoint_path=ckpt, log_path=log
                    )
            self.assertEqual(
                remotes,
                [
                    "checkpoints/model.slot0.pt",
                    "checkpoints/model.current.json",
                ],
            )

    def test_ensure_project_artifact_returns_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "checkpoints" / "model.pt"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"local")
            with patch("lpap.artifact_sync.download_files") as download_mock:
                path = ensure_project_artifact(local, project_root=root)
            self.assertEqual(path, local.resolve())
            download_mock.assert_not_called()

    def test_ensure_project_artifact_downloads_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "data" / "bank.pt"

            def _fake_download(pairs, **kwargs):
                dest = Path(pairs[0][1])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"remote")
                return [dest]

            with patch(
                "lpap.artifact_sync.download_files", side_effect=_fake_download
            ) as download_mock:
                path = ensure_project_artifact(
                    "data/bank.pt", project_root=root
                )
            self.assertEqual(path, local.resolve())
            self.assertEqual(path.read_bytes(), b"remote")
            remotes = [remote for remote, _ in download_mock.call_args.args[0]]
            self.assertEqual(remotes, ["data/bank.pt"])

    def test_ensure_checkpoint_requires_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "lpap.artifact_sync.read_checkpoint_pointer",
                side_effect=FileNotFoundError("missing pointer"),
            ):
                with self.assertRaises(FileNotFoundError):
                    ensure_checkpoint(root, "image_energy_flow.pt")

    def test_ensure_checkpoint_downloads_pointed_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def _fake_download(pairs, **kwargs):
                dest = Path(pairs[0][1])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"ckpt")
                return [dest]

            with (
                patch(
                    "lpap.artifact_sync.read_checkpoint_pointer",
                    return_value={"slot": 1, "sha256": "x", "size": 4},
                ),
                patch(
                    "lpap.artifact_sync.download_files",
                    side_effect=_fake_download,
                ) as download_mock,
            ):
                path = ensure_checkpoint(root, "image_energy_flow.pt")
            self.assertEqual(path.name, "image_energy_flow.pt")
            self.assertTrue(path.is_file())
            remotes = [remote for remote, _ in download_mock.call_args.args[0]]
            self.assertEqual(
                remotes, ["checkpoints/image_energy_flow.slot1.pt"]
            )

    def test_migrate_bare_checkpoints_skips_when_pointer_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mock_api = MagicMock()

            def _exists(bucket: str, remote: str, *, token=None) -> bool:
                return remote.endswith(".current.json") or remote.endswith(
                    "/model.pt"
                ) or remote == "checkpoints/model.pt"

            with (
                patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False),
                patch("lpap.artifact_sync.apply_hf_token", return_value="tok"),
                patch(
                    "lpap.artifact_sync._resolve_bucket",
                    return_value="user/bucket",
                ),
                patch("huggingface_hub.HfApi", return_value=mock_api),
                patch(
                    "lpap.artifact_sync._bucket_object_exists",
                    side_effect=_exists,
                ),
                patch(
                    "lpap.artifact_sync.list_bare_checkpoint_names",
                    return_value=["model.pt"],
                ),
            ):
                from lpap.artifact_sync import migrate_bare_checkpoints_to_dual_slot

                results = migrate_bare_checkpoints_to_dual_slot(root)
            self.assertEqual(results[0]["status"], "already_migrated")
            mock_api.batch_bucket_files.assert_called_with(
                "user/bucket", delete=["checkpoints/model.pt"]
            )


if __name__ == "__main__":
    unittest.main()
