from __future__ import annotations

import torch
from jaxtyping import Float, Int

LPAP_FOLD_ORDER = "column_major"


def _validate_permutation_bijection(
    permutation: torch.Tensor,
    *,
    expected_length: int | None = None,
) -> Int[torch.Tensor, "n"]:
    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    if permutation.dtype.is_floating_point or permutation.dtype == torch.bool:
        raise TypeError("permutation must have an integer dtype")
    length = int(permutation.numel())
    if length <= 0:
        raise ValueError("permutation must be non-empty")
    if expected_length is not None and length != expected_length:
        raise ValueError(
            f"permutation length must be {expected_length}, got {length}"
        )
    perm_long = permutation.detach().long()
    if int(perm_long.min()) < 0 or int(perm_long.max()) >= length:
        raise ValueError("permutation indices must be in [0, n)")
    seen = torch.zeros(length, dtype=torch.bool, device=perm_long.device)
    seen.index_fill_(0, perm_long, True)
    if not bool(seen.all()):
        raise ValueError("permutation must be a bijection over 0..n-1")
    return perm_long


def make_permutation_indices(
    *,
    value_count: int,
    seed: int,
    device: str | torch.device | None = None,
) -> Int[torch.Tensor, "n"]:
    """Build a fixed seeded random LPAP permutation.

    RNG runs on CPU (then the tensor is moved to ``device``) so the same seed
    yields the same layout on CPU and CUDA. Checkpoints store the concrete
    ``permutation`` tensor; loaders read that tensor when reloading a run.
    """
    if value_count <= 0:
        raise ValueError("value_count must be positive")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(value_count, generator=generator, device="cpu")
    return permutation.to(device=target_device)


def as_long_permutation(
    permutation: torch.Tensor,
    *,
    value_count: int,
) -> Int[torch.Tensor, "n"]:
    """Validate and normalize a permutation tensor to contiguous CPU ``long``."""
    if value_count <= 0:
        raise ValueError("value_count must be positive")
    validated = _validate_permutation_bijection(permutation, expected_length=value_count)
    return validated.cpu().contiguous().clone()


def invert_permutation_indices(
    permutation: Int[torch.Tensor, "n"],
) -> Int[torch.Tensor, "n"]:
    validated = _validate_permutation_bijection(permutation)
    inverse = torch.empty_like(validated)
    inverse[validated] = torch.arange(validated.numel(), device=validated.device)
    return inverse


def apply_permutation(
    values: Float[torch.Tensor, "... n"],
    permutation: Int[torch.Tensor, "n"],
) -> Float[torch.Tensor, "... n"]:
    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    if permutation.dtype.is_floating_point or permutation.dtype == torch.bool:
        raise TypeError("permutation must have an integer dtype")
    if values.shape[-1] != permutation.numel():
        raise ValueError("values last dimension must match permutation length")
    return values.index_select(-1, permutation.to(device=values.device, dtype=torch.long))


def reverse_permutation(
    values: Float[torch.Tensor, "... n"],
    permutation: Int[torch.Tensor, "n"],
) -> Float[torch.Tensor, "... n"]:
    if values.shape[-1] != permutation.numel():
        raise ValueError("values last dimension must match permutation length")
    inverse = invert_permutation_indices(permutation.to(device=values.device))
    return values.index_select(-1, inverse)


def fold_permutation_tokens(
    values: Float[torch.Tensor, "... n"],
    *,
    bucket_count: int,
) -> Float[torch.Tensor, "... buckets probe"]:
    """Fold flat values column-major into ``(..., bucket_count, probe_count)``.

    Flat index ``pos = probe * bucket_count + bucket`` maps to lane ``bucket``
    at probe position ``probe``.
    """
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if values.shape[-1] <= 0:
        raise ValueError("values last dimension must be positive")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if values.shape[-1] % bucket_count != 0:
        raise ValueError("values last dimension must be divisible by bucket_count")

    probe_count = values.shape[-1] // bucket_count
    return values.reshape(*values.shape[:-1], probe_count, bucket_count).movedim(-1, -2)


def unfold_permutation_tokens(
    tokens: Float[torch.Tensor, "... buckets probe"],
) -> Float[torch.Tensor, "... n"]:
    """Inverse of :func:`fold_permutation_tokens` (column-major)."""
    if tokens.ndim < 2:
        raise ValueError("tokens must have bucket and probe dimensions")
    if tokens.shape[-2] <= 0 or tokens.shape[-1] <= 0:
        raise ValueError("tokens bucket and probe dimensions must be positive")
    bucket_count = tokens.shape[-2]
    probe_count = tokens.shape[-1]
    return tokens.movedim(-2, -1).reshape(
        *tokens.shape[:-2], bucket_count * probe_count
    )


def folded_positions_to_flat_indices(
    *,
    source_buckets: Int[torch.Tensor, "..."],
    probe_indices: Int[torch.Tensor, "..."],
    bucket_count: int,
) -> Int[torch.Tensor, "..."]:
    """Map column-major ``(source_bucket, probe)`` coordinates to flat ``0..N-1``."""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    return probe_indices * bucket_count + source_buckets
