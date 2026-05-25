from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from jaxtyping import Float, UInt8
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class SyntheticHarmonicBatch(TypedDict):
    values: Float[torch.Tensor, "batch n"]  # noqa: F722
    gains: Float[torch.Tensor, "batch harmonics"]  # noqa: F722
    phases: Float[torch.Tensor, "batch harmonics"]  # noqa: F722
    spikiness: Float[torch.Tensor, "batch harmonics"]  # noqa: F722
    frequencies: Float[torch.Tensor, "harmonics"]  # noqa: F722, F821


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


def sample_synthetic_harmonic_batch(
    *,
    batch_size: int,
    n: int,
    harmonic_count: int,
    gain_variance: float = 1.0,
    gain_half_life: float = 4.0,
    spikiness_range: tuple[float, float] = (4.0, 8.0),
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    return_parameters: bool = False,
) -> Float[torch.Tensor, "batch n"] | SyntheticHarmonicBatch:  # noqa: F722
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    if harmonic_count <= 0:
        raise ValueError("harmonic_count must be positive")
    if gain_variance < 0:
        raise ValueError("gain_variance must be non-negative")
    if gain_half_life <= 0:
        raise ValueError("gain_half_life must be positive")

    spikiness_min, spikiness_max = spikiness_range
    if spikiness_min > spikiness_max:
        raise ValueError("spikiness_range must be ordered as (min, max)")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    frequencies = torch.arange(1, harmonic_count + 1, device=target_device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, n, device=target_device, dtype=dtype)

    variances = gain_variance * torch.pow(
        torch.tensor(0.5, device=target_device, dtype=dtype),
        (frequencies - 1) / gain_half_life,
    )
    gain_std = torch.sqrt(variances)
    gains = (
        torch.randn(
            (batch_size, harmonic_count),
            generator=generator,
            device=target_device,
            dtype=dtype,
        )
        * gain_std
    )
    phases = torch.rand(
        (batch_size, harmonic_count),
        generator=generator,
        device=target_device,
        dtype=dtype,
    )
    spikiness = torch.empty(
        (batch_size, harmonic_count), device=target_device, dtype=dtype
    ).uniform_(spikiness_min, spikiness_max, generator=generator)

    angles = (
        torch.pi * frequencies[None, :, None] * (x[None, None, :] + phases[:, :, None])
    )
    basis = (1.0 - torch.abs(torch.sin(angles))).pow(torch.exp(spikiness)[:, :, None])
    values = (gains[:, :, None] * basis).sum(dim=1)

    if not return_parameters:
        return values

    return {
        "values": values,
        "gains": gains,
        "phases": phases,
        "spikiness": spikiness,
        "frequencies": frequencies,
    }


class SyntheticHarmonicDataset(
    IterableDataset[Float[torch.Tensor, "batch n"] | SyntheticHarmonicBatch]  # noqa: F722
):
    def __init__(
        self,
        *,
        n: int,
        harmonic_count: int,
        batch_size: int,
        batches_per_epoch: int | None = None,
        gain_variance: float = 1.0,
        gain_half_life: float = 4.0,
        spikiness_range: tuple[float, float] = (4.0, 8.0),
        seed: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        return_parameters: bool = False,
    ) -> None:
        self.n = n
        self.harmonic_count = harmonic_count
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.gain_variance = gain_variance
        self.gain_half_life = gain_half_life
        self.spikiness_range = spikiness_range
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.return_parameters = return_parameters

        sample_synthetic_harmonic_batch(
            batch_size=1,
            n=n,
            harmonic_count=harmonic_count,
            gain_variance=gain_variance,
            gain_half_life=gain_half_life,
            spikiness_range=spikiness_range,
            device=device,
            dtype=dtype,
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        target_device = (
            torch.device("cpu") if self.device is None else torch.device(self.device)
        )
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=target_device)
            generator.manual_seed(self.seed + worker_id)

        batch_index = 0
        while self.batches_per_epoch is None or batch_index < self.batches_per_epoch:
            yield sample_synthetic_harmonic_batch(
                batch_size=self.batch_size,
                n=self.n,
                harmonic_count=self.harmonic_count,
                gain_variance=self.gain_variance,
                gain_half_life=self.gain_half_life,
                spikiness_range=self.spikiness_range,
                generator=generator,
                device=target_device,
                dtype=self.dtype,
                return_parameters=self.return_parameters,
            )
            batch_index += 1


def synthetic_harmonic_dataloader(
    *,
    n: int,
    harmonic_count: int,
    batch_size: int,
    batches_per_epoch: int | None = None,
    gain_variance: float = 1.0,
    gain_half_life: float = 4.0,
    spikiness_range: tuple[float, float] = (4.0, 8.0),
    seed: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    return_parameters: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> DataLoader[Float[torch.Tensor, "batch n"] | SyntheticHarmonicBatch]:  # noqa: F722
    dataset = SyntheticHarmonicDataset(
        n=n,
        harmonic_count=harmonic_count,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        gain_variance=gain_variance,
        gain_half_life=gain_half_life,
        spikiness_range=spikiness_range,
        seed=seed,
        device=device,
        dtype=dtype,
        return_parameters=return_parameters,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
