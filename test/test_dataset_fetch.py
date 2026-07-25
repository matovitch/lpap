from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from lpap.dataset_fetch import (
    decompress_zstd_file,
    default_image_pt_path,
    ensure_image_tensor_archive,
)

_STORAGE_TOML = textwrap.dedent(
    """\
    [artifacts]
    bucket = "org/artifacts"

    [images]
    bucket = "org/images"
    remote_zst = "images_32x32_gray.pt.zst"
    local_pt = "data/images_32x32_gray.pt"
    local_zst = "data/images_32x32_gray.pt.zst"
    """
)


def _write_storage_toml(root: Path) -> None:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "storage.toml").write_text(_STORAGE_TOML, encoding="utf-8")


class DatasetFetchTest(unittest.TestCase):
    def test_default_image_pt_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_storage_toml(root)
            self.assertEqual(
                default_image_pt_path(root),
                root / "data" / "images_32x32_gray.pt",
            )

    def test_decompress_zstd_file(self) -> None:
        import zstandard as zstd

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "file.pt"
            zst = root / "file.pt.zst"
            out = root / "out.pt"
            payload = b"lpap-image-bytes" * 100
            raw.write_bytes(payload)
            with zst.open("wb") as handle:
                handle.write(zstd.ZstdCompressor().compress(payload))
            decompress_zstd_file(zst, out)
            self.assertEqual(out.read_bytes(), payload)

    def test_ensure_reuses_existing_pt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_storage_toml(root)
            pt = root / "data" / "images_32x32_gray.pt"
            pt.parent.mkdir(parents=True)
            pt.write_bytes(b"cached")
            with patch("lpap.dataset_fetch.download_image_zst") as download:
                path = ensure_image_tensor_archive(root)
            self.assertEqual(path, pt)
            download.assert_not_called()

    def test_ensure_downloads_and_decompresses(self) -> None:
        import zstandard as zstd

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_storage_toml(root)
            payload = b"dataset-payload"
            zst_bytes = zstd.ZstdCompressor().compress(payload)

            def fake_download(destination: Path, **kwargs: object) -> Path:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(zst_bytes)
                return destination

            with patch(
                "lpap.dataset_fetch.download_image_zst", side_effect=fake_download
            ):
                path = ensure_image_tensor_archive(root, keep_zst=True)
            self.assertEqual(path.read_bytes(), payload)
            self.assertTrue((root / "data" / "images_32x32_gray.pt.zst").is_file())


if __name__ == "__main__":
    unittest.main()
