from __future__ import annotations

import torch
from jaxtyping import Float, Int


def make_grouped_permutation_indices(
    *,
    value_count: int,
    bucket_count: int,
    seed: int,
    device: str | torch.device | None = None,
) -> Int[torch.Tensor, "n"]:  # noqa: F722, F821
    if value_count <= 0:
        raise ValueError("value_count must be positive")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if value_count % bucket_count != 0:
        raise ValueError("value_count must be divisible by bucket_count")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    probe_count = value_count // bucket_count
    generator = torch.Generator(device=target_device).manual_seed(seed)
    by_bucket: list[list[int]] = [[] for _ in range(bucket_count)]

    for source_group in range(bucket_count):
        source_start = source_group * probe_count
        local_order = torch.randperm(
            probe_count, generator=generator, device=target_device
        ).tolist()
        for local_position, source_offset in enumerate(local_order):
            target_bucket = (local_position + source_group) % bucket_count
            by_bucket[target_bucket].append(source_start + int(source_offset))

    permutation = torch.empty(value_count, dtype=torch.long, device=target_device)
    for target_bucket, source_indices in enumerate(by_bucket):
        if len(source_indices) != probe_count:
            raise RuntimeError(
                "grouped permutation construction violated bucket balance"
            )
        row_order = torch.randperm(
            probe_count, generator=generator, device=target_device
        ).tolist()
        for target_row, source_list_index in enumerate(row_order):
            destination_index = target_row * bucket_count + target_bucket
            permutation[destination_index] = source_indices[int(source_list_index)]

    return permutation


def invert_permutation_indices(
    permutation: Int[torch.Tensor, "n"],  # noqa: F722, F821
) -> Int[torch.Tensor, "n"]:  # noqa: F722, F821
    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")

    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def apply_grouped_permutation(
    values: Float[torch.Tensor, "... n"],  # noqa: F722
    permutation: Int[torch.Tensor, "n"],  # noqa: F722, F821
) -> Float[torch.Tensor, "... n"]:  # noqa: F722
    if values.shape[-1] != permutation.numel():
        raise ValueError("values last dimension must match permutation length")
    return values.index_select(-1, permutation.to(device=values.device))


def reverse_grouped_permutation(
    values: Float[torch.Tensor, "... n"],  # noqa: F722
    permutation: Int[torch.Tensor, "n"],  # noqa: F722, F821
) -> Float[torch.Tensor, "... n"]:  # noqa: F722
    inverse = invert_permutation_indices(permutation.to(device=values.device))
    return values.index_select(-1, inverse)


def fold_grouped_permutation_tokens(
    values: Float[torch.Tensor, "... n"],  # noqa: F722
    *,
    bucket_count: int,
) -> Float[torch.Tensor, "... buckets probe"]:  # noqa: F722
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if values.shape[-1] % bucket_count != 0:
        raise ValueError("values last dimension must be divisible by bucket_count")

    probe_count = values.shape[-1] // bucket_count
    return values.reshape(*values.shape[:-1], probe_count, bucket_count).movedim(-1, -2)


def unfold_grouped_permutation_tokens(
    tokens: Float[torch.Tensor, "... buckets probe"],  # noqa: F722
) -> Float[torch.Tensor, "... n"]:  # noqa: F722
    if tokens.ndim < 2:
        raise ValueError("tokens must have bucket and probe dimensions")
    return tokens.movedim(-2, -1).reshape(*tokens.shape[:-2], -1)
