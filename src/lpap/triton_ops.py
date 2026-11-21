"""Triton CUDA LPAP kernel with lazy Triton import and PyTorch CPU fallback."""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int

from lpap.ops import (
    LPAPTokenResult,
    _validate_lpap_args,
    _validate_lpap_token_args,
    lpap_tokens_torch,
)
from lpap.permutation import (
    apply_permutation,
    fold_permutation_tokens,
    reverse_permutation,
    unfold_permutation_tokens,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU/non-Linux fallback
    triton = None
    tl = None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton is not None and tl is not None:

    @triton.jit
    def _lpap_tokens_kernel(
        values_ptr,
        work_ptr,
        work_valid_ptr,
        diff_ptr,
        work_probe_ptr,
        buckets_ptr,
        bucket_valid_ptr,
        dibs_ptr,
        bucket_probe_ptr,
        value_count,
        bucket_count,
        probe_count,
        k_max,
        block_probe_count: tl.constexpr,
    ):
        batch_index = tl.program_id(0).to(tl.int64)
        value_count_i64 = value_count.to(tl.int64)
        bucket_count_i64 = bucket_count.to(tl.int64)
        probe_count_i64 = probe_count.to(tl.int64)
        value_base = batch_index * value_count_i64
        bucket_base = batch_index * bucket_count_i64

        value_dtype = values_ptr.dtype.element_ty
        zero_val = tl.full([], 0.0, dtype=value_dtype)
        neg_one_vec = tl.full([block_probe_count], -1.0, dtype=value_dtype)
        zero_i64 = tl.full([], 0, dtype=tl.int64)
        zeros_i64_vec = tl.full([block_probe_count], 0, dtype=tl.int64)
        zero_i8 = tl.full([], 0, dtype=tl.int8)
        one_i8 = tl.full([], 1, dtype=tl.int8)
        ones_i8_vec = tl.full([block_probe_count], 1, dtype=tl.int8)

        probe_offsets = tl.arange(0, block_probe_count).to(tl.int64)
        probe_mask = probe_offsets < probe_count_i64
        sentinel_index = tl.full([block_probe_count], block_probe_count, dtype=tl.int64)

        for bucket_index in range(0, bucket_count):
            bucket_i64 = bucket_index.to(tl.int64)
            value_offsets = value_base + bucket_i64 * probe_count_i64 + probe_offsets
            lane_init = tl.load(values_ptr + value_offsets, mask=probe_mask, other=zero_val)
            tl.store(work_ptr + value_offsets, lane_init, mask=probe_mask)
            tl.store(work_valid_ptr + value_offsets, ones_i8_vec, mask=probe_mask)
            tl.store(diff_ptr + value_offsets, zeros_i64_vec, mask=probe_mask)
            tl.store(work_probe_ptr + value_offsets, probe_offsets, mask=probe_mask)

            bucket_offset = bucket_base + bucket_i64
            tl.store(buckets_ptr + bucket_offset, zero_val)
            tl.store(bucket_valid_ptr + bucket_offset, zero_i8)
            tl.store(dibs_ptr + bucket_offset, zero_i64)
            tl.store(bucket_probe_ptr + bucket_offset, zero_i64)

        for roll_count in range(0, k_max):
            roll_i64 = roll_count.to(tl.int64)
            for bucket_index in range(0, bucket_count):
                bucket_i64 = bucket_index.to(tl.int64)
                # Add bucket_count before modulo so numerator is always positive.
                source_lane = (bucket_index - roll_count + bucket_count) % bucket_count
                source_base = value_base + source_lane.to(tl.int64) * probe_count_i64
                source_offsets = source_base + probe_offsets

                lane_values = tl.load(
                    work_ptr + source_offsets, mask=probe_mask, other=zero_val
                )
                lane_valid = tl.load(
                    work_valid_ptr + source_offsets, mask=probe_mask, other=zero_i8
                )
                is_valid = probe_mask & (lane_valid != zero_i8) & (lane_values == lane_values)
                scores = tl.where(is_valid, tl.abs(lane_values), neg_one_vec)
                max_score = tl.max(scores, axis=0)
                has_valid = max_score >= zero_val

                index_candidates = tl.where(
                    (scores == max_score) & is_valid, probe_offsets, sentinel_index
                )
                raw_candidate_index = tl.min(index_candidates, axis=0)
                candidate_index = tl.where(has_valid, raw_candidate_index, zero_i64)
                candidate_offset = source_base + candidate_index

                candidate = tl.load(
                    work_ptr + candidate_offset, mask=has_valid, other=zero_val
                )
                candidate_diff = tl.load(
                    diff_ptr + candidate_offset, mask=has_valid, other=zero_i64
                )
                candidate_probe = tl.load(
                    work_probe_ptr + candidate_offset, mask=has_valid, other=zero_i64
                )

                bucket_offset = bucket_base + bucket_i64
                old_bucket = tl.load(buckets_ptr + bucket_offset)
                old_valid = tl.load(bucket_valid_ptr + bucket_offset)
                old_dib = tl.load(dibs_ptr + bucket_offset)
                old_probe = tl.load(bucket_probe_ptr + bucket_offset)

                old_is_valid = old_valid != zero_i8
                should_write = has_valid & (
                    (~old_is_valid) | (tl.abs(candidate) >= tl.abs(old_bucket))
                )

                swap_value = tl.where(old_is_valid, old_bucket, zero_val)
                swap_diff = tl.where(old_is_valid, old_dib - roll_i64, zero_i64)
                swap_probe = tl.where(old_is_valid, old_probe, zero_i64)

                tl.store(work_ptr + candidate_offset, swap_value, mask=should_write)
                tl.store(work_valid_ptr + candidate_offset, old_valid, mask=should_write)
                tl.store(diff_ptr + candidate_offset, swap_diff, mask=should_write)
                tl.store(work_probe_ptr + candidate_offset, swap_probe, mask=should_write)

                tl.store(buckets_ptr + bucket_offset, candidate, mask=should_write)
                tl.store(bucket_valid_ptr + bucket_offset, one_i8, mask=should_write)
                tl.store(
                    dibs_ptr + bucket_offset,
                    candidate_diff + roll_i64,
                    mask=should_write,
                )
                tl.store(
                    bucket_probe_ptr + bucket_offset,
                    candidate_probe,
                    mask=should_write,
                )


def lpap_tokens_triton(
    tokens: Float[torch.Tensor, "... buckets probe"],
    *,
    k_max: int,
) -> LPAPTokenResult:
    """Triton CUDA LPAP on folded ``(..., C, P)`` tokens (CPU fallback via PyTorch)."""
    if tokens.requires_grad:
        raise RuntimeError(
            "lpap_triton does not support autograd; detach inputs or use lpap_torch"
        )
    _validate_lpap_token_args(tokens, k_max)

    batch_shape = tokens.shape[:-2]
    bucket_count = tokens.shape[-2]
    probe_count = tokens.shape[-1]
    batch_count = math.prod(batch_shape) if batch_shape else 1

    if batch_count == 0:
        return LPAPTokenResult(
            buckets=torch.empty(
                (*batch_shape, bucket_count), device=tokens.device, dtype=tokens.dtype
            ),
            dibs=torch.empty(
                (*batch_shape, bucket_count), device=tokens.device, dtype=torch.int64
            ),
            bucket_probes=torch.empty(
                (*batch_shape, bucket_count), device=tokens.device, dtype=torch.int64
            ),
            remaining=torch.empty_like(tokens),
        )

    if not tokens.is_cuda or triton is None:
        return lpap_tokens_torch(tokens, k_max=k_max)

    value_count = bucket_count * probe_count
    block_probe_count = _next_power_of_2(probe_count)
    if block_probe_count > 131_072:
        raise ValueError("probe_count is too large for the current Triton LPAP kernel")

    contiguous_tokens = tokens.reshape(batch_count, value_count).contiguous()
    work = torch.empty_like(contiguous_tokens)
    work_valid = torch.empty(
        (batch_count, value_count), device=tokens.device, dtype=torch.int8
    )
    dibs_diff = torch.empty(
        (batch_count, value_count), device=tokens.device, dtype=torch.int64
    )
    work_probes = torch.empty(
        (batch_count, value_count), device=tokens.device, dtype=torch.int64
    )
    buckets = torch.empty(
        (batch_count, bucket_count), device=tokens.device, dtype=tokens.dtype
    )
    bucket_valid = torch.empty(
        (batch_count, bucket_count), device=tokens.device, dtype=torch.int8
    )
    dibs = torch.empty(
        (batch_count, bucket_count), device=tokens.device, dtype=torch.int64
    )
    bucket_probes = torch.empty(
        (batch_count, bucket_count), device=tokens.device, dtype=torch.int64
    )

    _lpap_tokens_kernel[(batch_count,)](
        contiguous_tokens,
        work,
        work_valid,
        dibs_diff,
        work_probes,
        buckets,
        bucket_valid,
        dibs,
        bucket_probes,
        value_count,
        bucket_count,
        probe_count,
        k_max,
        block_probe_count,
    )
    return LPAPTokenResult(
        buckets=buckets.reshape(*batch_shape, bucket_count),
        dibs=dibs.reshape(*batch_shape, bucket_count),
        bucket_probes=bucket_probes.reshape(*batch_shape, bucket_count),
        remaining=work.reshape(*batch_shape, bucket_count, probe_count),
    )


def lpap_triton(
    values: Float[torch.Tensor, "... n"],
    *,
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
) -> tuple[
    Float[torch.Tensor, "... buckets"],
    Int[torch.Tensor, "... buckets"],
    Float[torch.Tensor, "... n"],
]:
    """Triton CUDA LPAP operator using canonical column-major folding."""
    if values.requires_grad:
        raise RuntimeError(
            "lpap_triton does not support autograd; detach inputs or use lpap_torch"
        )
    _validate_lpap_args(values, bucket_count, k_max)

    ordered = (
        apply_permutation(values, permutation) if permutation is not None else values
    )
    tokens = fold_permutation_tokens(ordered, bucket_count=bucket_count)
    result = lpap_tokens_triton(tokens, k_max=k_max)
    remaining = unfold_permutation_tokens(result.remaining)
    if permutation is not None:
        remaining = reverse_permutation(remaining, permutation)
    return result.buckets, result.dibs, remaining


__all__ = ["lpap_tokens_triton", "lpap_triton"]
