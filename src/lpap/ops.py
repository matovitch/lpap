from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int


def _validate_lpap_args(values: torch.Tensor, bucket_count: int, k_max: int) -> None:
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if k_max < 0:
        raise ValueError("k_max must be non-negative")
    if values.shape[-1] % bucket_count != 0:
        raise ValueError(
            "the last dimension of values must be divisible by bucket_count"
        )
    if not values.dtype.is_floating_point:
        raise TypeError("values must be a floating point tensor")


def lpap_torch(
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    *,
    bucket_count: int,
    k_max: int,
) -> tuple[
    Float[torch.Tensor, "batch buckets"],  # noqa: F722
    Int[torch.Tensor, "batch buckets"],  # noqa: F722
    Float[torch.Tensor, "batch n"],  # noqa: F722
]:
    _validate_lpap_args(values, bucket_count, k_max)

    input_shape = values.shape
    value_count = input_shape[-1]
    probe_count = value_count // bucket_count
    batch_count = math.prod(input_shape[:-1]) if len(input_shape) > 1 else 1

    work = values.reshape(batch_count, bucket_count, probe_count).clone()
    dibs_diff = torch.zeros(
        (batch_count, bucket_count, probe_count),
        device=values.device,
        dtype=torch.int64,
    )
    buckets = torch.zeros(
        (batch_count, bucket_count), device=values.device, dtype=values.dtype
    )
    dibs = torch.zeros(
        (batch_count, bucket_count), device=values.device, dtype=torch.int64
    )

    for roll_count in range(k_max):
        for bucket_index in range(bucket_count):
            source_lane = (bucket_index - roll_count) % bucket_count
            lane_values = work[:, source_lane, :]
            candidate_indices = lane_values.abs().argmax(dim=-1)
            batch_indices = torch.arange(batch_count, device=values.device)
            candidates = lane_values[batch_indices, candidate_indices]
            selected_diffs = dibs_diff[batch_indices, source_lane, candidate_indices]
            candidate_dibs = selected_diffs + roll_count
            update = candidates.abs() >= buckets[:, bucket_index].abs()

            old_bucket_values = buckets[:, bucket_index].clone()
            old_dibs = dibs[:, bucket_index].clone()

            swap_diffs = old_dibs - roll_count

            update_batches = batch_indices[update]
            update_candidate_indices = candidate_indices[update]
            work[update_batches, source_lane, update_candidate_indices] = (
                old_bucket_values[update]
            )
            dibs_diff[update_batches, source_lane, update_candidate_indices] = (
                swap_diffs[update]
            )
            buckets[update_batches, bucket_index] = candidates[update]
            dibs[update_batches, bucket_index] = candidate_dibs[update]

    output_batch_shape = input_shape[:-1]
    return (
        buckets.reshape(*output_batch_shape, bucket_count),
        dibs.reshape(*output_batch_shape, bucket_count),
        work.reshape(input_shape),
    )
