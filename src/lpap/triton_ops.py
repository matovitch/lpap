"""
Triton compiles one LPAP kernel per constexpr shape tuple: value_count,
bucket_count, probe_count, k_max, and block_probe_count. The training configs keep
these fixed, which is the intended fast path. Shape or k_max sweeps will pay a
one-time JIT compile for each new combination before steady-state execution.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int

from lpap.ops import lpap_torch


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _lpap_kernel(
    values_ptr,
    work_ptr,
    diff_ptr,
    buckets_ptr,
    dibs_ptr,
    value_count: tl.constexpr,
    bucket_count: tl.constexpr,
    probe_count: tl.constexpr,
    k_max: tl.constexpr,
    block_probe_count: tl.constexpr,
):
    batch_index = tl.program_id(0)
    value_base = batch_index * value_count
    bucket_base = batch_index * bucket_count
    probe_offsets = tl.arange(0, block_probe_count)
    probe_mask = probe_offsets < probe_count

    for bucket_index in tl.static_range(0, bucket_count):
        value_offsets = value_base + bucket_index * probe_count + probe_offsets
        values = tl.load(values_ptr + value_offsets, mask=probe_mask, other=0.0)
        tl.store(work_ptr + value_offsets, values, mask=probe_mask)
        tl.store(diff_ptr + value_offsets, 0, mask=probe_mask)
        tl.store(buckets_ptr + bucket_base + bucket_index, 0.0)
        tl.store(dibs_ptr + bucket_base + bucket_index, 0)

    for roll_count in tl.static_range(0, k_max):
        for bucket_index in tl.static_range(0, bucket_count):
            source_lane = (bucket_index - roll_count) % bucket_count
            source_base = value_base + source_lane * probe_count
            source_offsets = source_base + probe_offsets
            lane_values = tl.load(work_ptr + source_offsets, mask=probe_mask, other=0.0)
            scores = tl.abs(lane_values)
            max_score = tl.max(scores, axis=0)
            index_candidates = tl.where(
                (scores == max_score) & probe_mask, probe_offsets, block_probe_count
            )
            candidate_index = tl.min(index_candidates, axis=0)
            candidate_offset = source_base + candidate_index
            candidate = tl.load(work_ptr + candidate_offset)
            candidate_diff = tl.load(diff_ptr + candidate_offset)

            bucket_offset = bucket_base + bucket_index
            old_bucket = tl.load(buckets_ptr + bucket_offset)
            old_dib = tl.load(dibs_ptr + bucket_offset)
            should_write = tl.abs(candidate) >= tl.abs(old_bucket)

            tl.store(work_ptr + candidate_offset, old_bucket, mask=should_write)
            tl.store(
                diff_ptr + candidate_offset, old_dib - roll_count, mask=should_write
            )
            tl.store(buckets_ptr + bucket_offset, candidate, mask=should_write)
            tl.store(
                dibs_ptr + bucket_offset, candidate_diff + roll_count, mask=should_write
            )


def lpap_triton(
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    *,
    bucket_count: int,
    k_max: int,
) -> tuple[
    Float[torch.Tensor, "batch buckets"],  # noqa: F722
    Int[torch.Tensor, "batch buckets"],  # noqa: F722
    Float[torch.Tensor, "batch n"],  # noqa: F722
]:
    if not values.is_cuda:
        return lpap_torch(values, bucket_count=bucket_count, k_max=k_max)
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

    input_shape = values.shape
    value_count = input_shape[-1]
    probe_count = value_count // bucket_count
    batch_count = math.prod(input_shape[:-1]) if len(input_shape) > 1 else 1
    block_probe_count = _next_power_of_2(probe_count)
    if block_probe_count > 131_072:
        raise ValueError("probe_count is too large for the current Triton LPAP kernel")

    contiguous_values = values.reshape(batch_count, value_count).contiguous()
    work = torch.empty_like(contiguous_values)
    dibs_diff = torch.empty(
        (batch_count, value_count), device=values.device, dtype=torch.int64
    )
    buckets = torch.empty(
        (batch_count, bucket_count), device=values.device, dtype=values.dtype
    )
    dibs = torch.empty(
        (batch_count, bucket_count), device=values.device, dtype=torch.int64
    )

    _lpap_kernel[(batch_count,)](
        contiguous_values,
        work,
        dibs_diff,
        buckets,
        dibs,
        value_count,
        bucket_count,
        probe_count,
        k_max,
        block_probe_count,
    )
    return (
        buckets.reshape(*input_shape[:-1], bucket_count),
        dibs.reshape(*input_shape[:-1], bucket_count),
        work.reshape(input_shape),
    )
