from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from jaxtyping import UInt8
from torch.utils.data import DataLoader, Dataset


class ImageTensorDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        images: UInt8[torch.Tensor, "batch channel height width"],  # noqa: F722
        names: list[str] | tuple[str, ...] | None = None,
        *,
        normalize: bool = False,
    ) -> None:
        if images.ndim != 4:
            raise ValueError("images must use NCHW layout")
        if images.shape[1] != 1:
            raise ValueError("images must have one grayscale channel")
        if images.dtype != torch.uint8:
            raise TypeError("images must be stored as torch.uint8")
        if names is not None and len(names) != images.shape[0]:
            raise ValueError("names length must match image count")

        self.images = images
        self.names = (
            list(names)
            if names is not None
            else [str(index) for index in range(images.shape[0])]
        )
        self.normalize = normalize

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image = self.images[index]
        if self.normalize:
            image = image.to(torch.float32).div(255.0)
        return image, self.names[index]


def load_image_tensor_dataset(
    path: str | Path,
    *,
    normalize: bool = False,
    map_location: str | torch.device | None = "cpu",
    mmap: bool = True,
    weights_only: bool = True,
) -> ImageTensorDataset:
    payload: dict[str, Any] = torch.load(
        Path(path),
        map_location=map_location,
        mmap=mmap,
        weights_only=weights_only,
    )
    return ImageTensorDataset(
        payload["images"], payload.get("names"), normalize=normalize
    )


def image_dataloader(
    path: str | Path,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    normalize: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    map_location: str | torch.device | None = "cpu",
    mmap: bool = True,
    persistent_workers: bool = False,
    multiprocessing_context: Literal["fork", "spawn", "forkserver"] | None = None,
) -> DataLoader[tuple[torch.Tensor, tuple[str, ...]]]:
    dataset = load_image_tensor_dataset(
        path,
        normalize=normalize,
        map_location=map_location,
        mmap=mmap,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        multiprocessing_context=multiprocessing_context,
    )
