from __future__ import annotations

import torch
from jaxtyping import Float, Int

from lpap.permutation import invert_permutation_indices


def _validate_side(side: int) -> None:
    if side <= 0:
        raise ValueError("side must be positive")
    if side & (side - 1):
        raise ValueError("side must be a power of two")


def _hilbert_index_to_xy(index: int, side: int) -> tuple[int, int]:
    x = 0
    y = 0
    distance = index
    scale = 1
    while scale < side:
        rx = 1 & (distance // 2)
        ry = 1 & (distance ^ rx)
        if ry == 0:
            if rx == 1:
                x = scale - 1 - x
                y = scale - 1 - y
            x, y = y, x
        x += scale * rx
        y += scale * ry
        distance //= 4
        scale *= 2
    return x, y


def hilbert_permutation(
    side: int = 32, *, device: str | torch.device | None = None
) -> Int[torch.Tensor, "n"]:
    _validate_side(side)
    target_device = torch.device("cpu") if device is None else torch.device(device)
    indices = [
        y * side + x
        for distance in range(side * side)
        for x, y in (_hilbert_index_to_xy(distance, side),)
    ]
    return torch.tensor(indices, device=target_device, dtype=torch.long)


def inverse_permutation(
    perm: Int[torch.Tensor, "n"],
) -> Int[torch.Tensor, "n"]:
    return invert_permutation_indices(perm)


def inverse_hilbert_permutation(
    side: int = 32, *, device: str | torch.device | None = None
) -> Int[torch.Tensor, "n"]:
    return inverse_permutation(hilbert_permutation(side=side, device=device))


def hilbert_flatten_images(
    images: Float[torch.Tensor, "batch 1 height width"],
    side: int = 32,
) -> Float[torch.Tensor, "batch 1 n"]:
    _validate_side(side)
    if images.ndim != 4:
        raise ValueError("images must have shape batch x 1 x side x side")
    if images.shape[1:] != (1, side, side):
        raise ValueError("images must have shape batch x 1 x side x side")
    perm = hilbert_permutation(side=side, device=images.device)
    return images.flatten(2).index_select(2, perm)


def hilbert_unflatten_images(
    sequence: Float[torch.Tensor, "batch 1 n"],
    side: int = 32,
) -> Float[torch.Tensor, "batch 1 height width"]:
    _validate_side(side)
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape batch x 1 x n")
    if sequence.shape[1] != 1 or sequence.shape[2] != side * side:
        raise ValueError("sequence must have shape batch x 1 x side*side")
    output = torch.empty_like(sequence)
    perm = hilbert_permutation(side=side, device=sequence.device)
    output.scatter_(2, perm.view(1, 1, -1).expand_as(sequence), sequence)
    return output.reshape(sequence.shape[0], 1, side, side)


def hilbert_metadata(side: int = 32) -> dict[str, int]:
    _validate_side(side)
    return {"side": side, "sequence_length": side * side}
