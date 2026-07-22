"""Fetch the public grayscale image tensor archive from Hugging Face.

The canonical training file is ``data/images_32x32_gray.pt``. Distribution uses
a zstd-compressed copy on the public storage bucket ``matovitch/lpap-images``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

DEFAULT_IMAGE_BUCKET = "matovitch/lpap-images"
DEFAULT_REMOTE_ZST = "images_32x32_gray.pt.zst"
DEFAULT_LOCAL_PT_NAME = "images_32x32_gray.pt"
DEFAULT_LOCAL_ZST_NAME = "images_32x32_gray.pt.zst"


def default_image_pt_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / DEFAULT_LOCAL_PT_NAME


def default_image_zst_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / DEFAULT_LOCAL_ZST_NAME


def decompress_zstd_file(source: Path, destination: Path) -> Path:
    """Stream-decompress ``source`` (.zst) into ``destination``."""
    import zstandard as zstd

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    decompressor = zstd.ZstdDecompressor()
    with source.open("rb") as compressed, tmp.open("wb") as out:
        decompressor.copy_stream(compressed, out)
    tmp.replace(destination)
    return destination


def download_image_zst(
    destination: Path,
    *,
    bucket: str = DEFAULT_IMAGE_BUCKET,
    remote_path: str = DEFAULT_REMOTE_ZST,
    token: str | bool | None = None,
) -> Path:
    """Download the compressed archive from the public HF storage bucket."""
    from huggingface_hub import HfFileSystem

    destination.parent.mkdir(parents=True, exist_ok=True)
    uri = f"hf://buckets/{bucket}/{remote_path.lstrip('/')}"
    fs = HfFileSystem(token=False if token is None else token)
    if not fs.exists(uri):
        raise FileNotFoundError(uri)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    fs.get(uri, str(tmp))
    tmp.replace(destination)
    return destination


def ensure_image_tensor_archive(
    project_root: str | Path = ".",
    *,
    force_download: bool = False,
    keep_zst: bool = True,
    bucket: str = DEFAULT_IMAGE_BUCKET,
    remote_path: str = DEFAULT_REMOTE_ZST,
    token: str | bool | None = None,
) -> Path:
    """Return ``data/images_32x32_gray.pt``, downloading/decompressing if needed.

    If the ``.pt`` already exists and ``force_download`` is false, it is reused.
    Otherwise the public ``.pt.zst`` is downloaded (unless a local ``.zst`` is
    already present) and decompressed beside it.
    """
    root = Path(project_root)
    pt_path = default_image_pt_path(root)
    zst_path = default_image_zst_path(root)

    if pt_path.is_file() and not force_download:
        return pt_path

    if force_download or not zst_path.is_file():
        download_image_zst(
            zst_path, bucket=bucket, remote_path=remote_path, token=token
        )

    decompress_zstd_file(zst_path, pt_path)
    if not keep_zst:
        zst_path.unlink(missing_ok=True)
    return pt_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing data/",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download the .zst even if local files exist",
    )
    parser.add_argument(
        "--delete-zst",
        action="store_true",
        help="remove the local .zst after successful decompress",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_IMAGE_BUCKET,
        help=f"HF storage bucket id (default: {DEFAULT_IMAGE_BUCKET})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = ensure_image_tensor_archive(
        args.project_root,
        force_download=args.force_download,
        keep_zst=not args.delete_zst,
        bucket=args.bucket,
    )
    size = path.stat().st_size
    print(f"ready {path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
