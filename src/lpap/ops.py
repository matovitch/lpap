from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int


def _validate_lpap_args(values: torch.Tensor, bucket_count: int, k_max: int) -> None:
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    if values.shape[-1] % bucket_count != 0:
        raise ValueError(
            "the last dimension of values must be divisible by bucket_count"
        )
    if not values.dtype.is_floating_point:
        raise TypeError("values must be a floating point tensor")


def lpap_torch(
    values: Float[torch.Tensor, "batch n"],
    *,
    bucket_count: int,
    k_max: int,
) -> tuple[
    Float[torch.Tensor, "batch buckets"],
    Int[torch.Tensor, "batch buckets"],
    Float[torch.Tensor, "batch n"],
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

    batch_indices = torch.arange(batch_count, device=values.device)[:, None]
    batch_grid = batch_indices.expand(batch_count, bucket_count)
    bucket_arange = torch.arange(bucket_count, device=values.device)
    bucket_indices_grid = bucket_arange[None, :]

    # For a fixed roll_count, bucket_index -> source_lane is a bijection over the
    # lanes, so every bucket reads and writes a distinct lane: the inner bucket
    # iterations are independent and can run in one vectorized step, leaving only
    # the roll_count loop in Python.
    for roll_count in range(k_max):
        source_lanes = (bucket_arange - roll_count) % bucket_count
        source_lane_grid = source_lanes[None, :].expand(batch_count, bucket_count)
        lane_values = work[:, source_lanes, :]
        lane_dibs_diff = dibs_diff[:, source_lanes, :]
        candidate_positions = lane_values.abs().argmax(dim=-1)
        candidates = lane_values[
            batch_indices, bucket_indices_grid, candidate_positions
        ]
        selected_diffs = lane_dibs_diff[
            batch_indices, bucket_indices_grid, candidate_positions
        ]
        candidate_dibs = selected_diffs + roll_count
        update = candidates.abs() >= buckets.abs()

        old_bucket_values = buckets.clone()
        old_dibs = dibs.clone()

        work[
            batch_grid[update], source_lane_grid[update], candidate_positions[update]
        ] = old_bucket_values[update]
        dibs_diff[
            batch_grid[update], source_lane_grid[update], candidate_positions[update]
        ] = old_dibs[update] - roll_count
        buckets[update] = candidates[update]
        dibs[update] = candidate_dibs[update]

    output_batch_shape = input_shape[:-1]
    return (
        buckets.reshape(*output_batch_shape, bucket_count),
        dibs.reshape(*output_batch_shape, bucket_count),
        work.reshape(input_shape),
    )
