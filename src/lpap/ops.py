from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int

from lpap.permutation import (
    LPAP_FOLD_ORDER,
    apply_permutation,
    fold_permutation_tokens,
    reverse_permutation,
    unfold_permutation_tokens,
)


@dataclass(frozen=True)
class LPAPTokenResult:
    """Output of exact LPAP on folded ``(..., buckets, probe)`` tokens."""

    buckets: Float[torch.Tensor, "... buckets"]
    dibs: Int[torch.Tensor, "... buckets"]
    bucket_probes: Int[torch.Tensor, "... buckets"]
    remaining: Float[torch.Tensor, "... buckets probe"]


def _validate_lpap_token_args(tokens: torch.Tensor, k_max: int) -> None:
    if tokens.ndim < 2:
        raise ValueError("tokens must have at least two dimensions")
    bucket_count = tokens.shape[-2]
    probe_count = tokens.shape[-1]
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    if k_max > bucket_count:
        raise ValueError("k_max must be at most bucket_count")
    if not tokens.dtype.is_floating_point:
        raise TypeError("tokens must be a floating point tensor")


def _validate_lpap_args(values: torch.Tensor, bucket_count: int, k_max: int) -> None:
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if values.shape[-1] <= 0:
        raise ValueError("the last dimension of values must be positive")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    if k_max > bucket_count:
        raise ValueError("k_max must be at most bucket_count")
    if values.shape[-1] % bucket_count != 0:
        raise ValueError(
            "the last dimension of values must be divisible by bucket_count"
        )
    if not values.dtype.is_floating_point:
        raise TypeError("values must be a floating point tensor")


def lpap_tokens_torch(
    tokens: Float[torch.Tensor, "... buckets probe"],
    *,
    k_max: int,
) -> LPAPTokenResult:
    """Reference PyTorch LPAP operator on folded ``(..., C, P)`` tokens.

    Tracks explicit valid flags for buckets and work slots: initially empty
    buckets accept any valid candidate and mark vacated work slots invalid so
    empties never win or re-enter. Occupied buckets use ``>=`` amplitude ties.
    Uses only fixed-shape ``torch.roll`` / ``torch.where`` updates (no boolean
    indexing host-device syncs).
    """
    _validate_lpap_token_args(tokens, k_max)

    batch_shape = tokens.shape[:-2]
    bucket_count = tokens.shape[-2]
    probe_count = tokens.shape[-1]
    batch_count = math.prod(batch_shape) if batch_shape else 1
    device = tokens.device
    dtype = tokens.dtype

    work = tokens.reshape(batch_count, bucket_count, probe_count).clone()
    work_valid = torch.ones(
        (batch_count, bucket_count, probe_count), device=device, dtype=torch.bool
    )
    work_probes = (
        torch.arange(probe_count, device=device, dtype=torch.int64)
        .view(1, 1, probe_count)
        .expand(batch_count, bucket_count, probe_count)
        .clone()
    )
    dibs_diff = torch.zeros(
        (batch_count, bucket_count, probe_count), device=device, dtype=torch.int64
    )

    buckets = torch.zeros((batch_count, bucket_count), device=device, dtype=dtype)
    bucket_valid = torch.zeros(
        (batch_count, bucket_count), device=device, dtype=torch.bool
    )
    dibs = torch.zeros((batch_count, bucket_count), device=device, dtype=torch.int64)
    bucket_probes = torch.zeros(
        (batch_count, bucket_count), device=device, dtype=torch.int64
    )

    probe_arange = torch.arange(probe_count, device=device, dtype=torch.int64).view(
        1, 1, probe_count
    )
    neg_one = torch.tensor(-1.0, device=device, dtype=dtype)
    zero_val = torch.zeros((batch_count, bucket_count), device=device, dtype=dtype)
    zero_i64 = torch.zeros(
        (batch_count, bucket_count), device=device, dtype=torch.int64
    )

    for roll_count in range(k_max):
        if roll_count == 0:
            lane_values = work
            lane_valid = work_valid
            lane_diff = dibs_diff
            lane_probes = work_probes
        else:
            lane_values = torch.roll(work, shifts=roll_count, dims=1)
            lane_valid = torch.roll(work_valid, shifts=roll_count, dims=1)
            lane_diff = torch.roll(dibs_diff, shifts=roll_count, dims=1)
            lane_probes = torch.roll(work_probes, shifts=roll_count, dims=1)

        # Exclude vacated slots and NaN values from candidate selection.
        finite_valid = lane_valid & ~torch.isnan(lane_values)
        scores = torch.where(finite_valid, lane_values.abs(), neg_one)
        candidate_positions = scores.argmax(dim=-1)
        gather_index = candidate_positions.unsqueeze(-1)

        candidates = lane_values.gather(-1, gather_index).squeeze(-1)
        candidate_valid = finite_valid.gather(-1, gather_index).squeeze(-1)
        selected_diffs = lane_diff.gather(-1, gather_index).squeeze(-1)
        selected_probes = lane_probes.gather(-1, gather_index).squeeze(-1)
        candidate_dibs = selected_diffs + roll_count

        update = candidate_valid & (~bucket_valid | (candidates.abs() >= buckets.abs()))
        slot_mask = (probe_arange == gather_index) & update.unsqueeze(-1)

        swap_values = torch.where(bucket_valid, buckets, zero_val).unsqueeze(-1)
        swap_valid = bucket_valid.unsqueeze(-1)
        swap_diffs = torch.where(
            bucket_valid, dibs - roll_count, zero_i64
        ).unsqueeze(-1)
        swap_probes = torch.where(bucket_valid, bucket_probes, zero_i64).unsqueeze(-1)

        new_lane_values = torch.where(slot_mask, swap_values, lane_values)
        new_lane_valid = torch.where(slot_mask, swap_valid, lane_valid)
        new_lane_diff = torch.where(slot_mask, swap_diffs, lane_diff)
        new_lane_probes = torch.where(slot_mask, swap_probes, lane_probes)

        if roll_count == 0:
            work = new_lane_values
            work_valid = new_lane_valid
            dibs_diff = new_lane_diff
            work_probes = new_lane_probes
        else:
            work = torch.roll(new_lane_values, shifts=-roll_count, dims=1)
            work_valid = torch.roll(new_lane_valid, shifts=-roll_count, dims=1)
            dibs_diff = torch.roll(new_lane_diff, shifts=-roll_count, dims=1)
            work_probes = torch.roll(new_lane_probes, shifts=-roll_count, dims=1)

        buckets = torch.where(update, candidates, buckets)
        bucket_valid = bucket_valid | update
        dibs = torch.where(update, candidate_dibs, dibs)
        bucket_probes = torch.where(update, selected_probes, bucket_probes)

    return LPAPTokenResult(
        buckets=buckets.reshape(*batch_shape, bucket_count),
        dibs=dibs.reshape(*batch_shape, bucket_count),
        bucket_probes=bucket_probes.reshape(*batch_shape, bucket_count),
        remaining=work.reshape(*batch_shape, bucket_count, probe_count),
    )


def lpap_torch(
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
    """Reference PyTorch LPAP operator using canonical column-major folding."""
    _validate_lpap_args(values, bucket_count, k_max)

    ordered = (
        apply_permutation(values, permutation) if permutation is not None else values
    )
    tokens = fold_permutation_tokens(ordered, bucket_count=bucket_count)
    result = lpap_tokens_torch(tokens, k_max=k_max)
    remaining = unfold_permutation_tokens(result.remaining)
    if permutation is not None:
        remaining = reverse_permutation(remaining, permutation)
    return result.buckets, result.dibs, remaining


__all__ = [
    "LPAP_FOLD_ORDER",
    "LPAPTokenResult",
    "lpap_tokens_torch",
    "lpap_torch",
]
