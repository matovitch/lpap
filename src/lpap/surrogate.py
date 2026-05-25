from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as torch_functional

from lpap.permutation import apply_grouped_permutation, fold_grouped_permutation_tokens
from lpap.transformer import TransformerBlock


@dataclass(frozen=True)
class LPAPSurrogateTargets:
    indices: Int[torch.Tensor, "batch buckets"]  # noqa: F722
    weights: Float[torch.Tensor, "batch buckets"]  # noqa: F722
    buckets: Float[torch.Tensor, "batch buckets"]  # noqa: F722
    dibs: Int[torch.Tensor, "batch buckets"]  # noqa: F722


@dataclass(frozen=True)
class LPAPSurrogateMetrics:
    loss: float
    accuracy: float
    weighted_accuracy: float
    mean_weight: float


def _validate_token_values(tokens: torch.Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape batch x buckets x probe")
    if not tokens.dtype.is_floating_point:
        raise TypeError("tokens must be a floating point tensor")


def lpap_surrogate_targets(
    tokens: Float[torch.Tensor, "batch buckets probe"],  # noqa: F722
    *,
    k_max: int,
) -> LPAPSurrogateTargets:
    _validate_token_values(tokens)
    if k_max < 0:
        raise ValueError("k_max must be non-negative")

    batch_count, bucket_count, probe_count = tokens.shape
    work = tokens.clone()
    work_indices = (
        torch.arange(probe_count, device=tokens.device)
        .expand(batch_count, bucket_count, probe_count)
        .clone()
    )
    dibs_diff = torch.zeros(
        (batch_count, bucket_count, probe_count), device=tokens.device, dtype=torch.long
    )
    buckets = torch.zeros(
        (batch_count, bucket_count), device=tokens.device, dtype=tokens.dtype
    )
    dibs = torch.zeros(
        (batch_count, bucket_count), device=tokens.device, dtype=torch.long
    )
    bucket_indices = torch.zeros(
        (batch_count, bucket_count), device=tokens.device, dtype=torch.long
    )
    batch_indices = torch.arange(batch_count, device=tokens.device)[:, None]
    bucket_indices_grid = torch.arange(bucket_count, device=tokens.device)[None, :]

    for roll_count in range(k_max):
        source_lanes = (
            torch.arange(bucket_count, device=tokens.device) - roll_count
        ) % bucket_count
        lane_values = work[:, source_lanes, :]
        lane_dibs_diff = dibs_diff[:, source_lanes, :]
        lane_indices = work_indices[:, source_lanes, :]
        candidate_positions = lane_values.abs().argmax(dim=-1)
        candidates = lane_values[
            batch_indices, bucket_indices_grid, candidate_positions
        ]
        selected_diffs = lane_dibs_diff[
            batch_indices, bucket_indices_grid, candidate_positions
        ]
        selected_indices = lane_indices[
            batch_indices, bucket_indices_grid, candidate_positions
        ]
        candidate_dibs = selected_diffs + roll_count
        update = candidates.abs() >= buckets.abs()

        old_bucket_values = buckets.clone()
        old_dibs = dibs.clone()
        old_bucket_indices = bucket_indices.clone()
        source_lane_grid = source_lanes[None, :].expand(batch_count, bucket_count)
        batch_grid = batch_indices.expand(batch_count, bucket_count)

        work[
            batch_grid[update], source_lane_grid[update], candidate_positions[update]
        ] = old_bucket_values[update]
        dibs_diff[
            batch_grid[update], source_lane_grid[update], candidate_positions[update]
        ] = old_dibs[update] - roll_count
        work_indices[
            batch_grid[update], source_lane_grid[update], candidate_positions[update]
        ] = old_bucket_indices[update]
        buckets[update] = candidates[update]
        dibs[update] = candidate_dibs[update]
        bucket_indices[update] = selected_indices[update]

    return LPAPSurrogateTargets(
        indices=bucket_indices,
        weights=buckets.abs(),
        buckets=buckets,
        dibs=dibs,
    )


def circular_previous_attention_mask(
    *,
    bucket_count: int,
    k_max: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    query_positions = torch.arange(bucket_count, device=target_device)[:, None]
    key_positions = torch.arange(bucket_count, device=target_device)[None, :]
    backward_distance = (query_positions - key_positions) % bucket_count
    return backward_distance < min(k_max, bucket_count)


class LPAPSurrogateTransformer(nn.Module):
    def __init__(
        self,
        *,
        probe_count: int,
        k_max: int,
        hidden_dim: int = 128,
        layer_count: int = 2,
        head_count: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if probe_count <= 0:
            raise ValueError("probe_count must be positive")
        self.probe_count = probe_count
        self.k_max = k_max
        self.input = nn.Linear(probe_count, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    head_count=head_count,
                    dropout=dropout,
                )
                for _layer_index in range(layer_count)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, probe_count)

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch buckets probe"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch buckets probe"]:  # noqa: F722
        _validate_token_values(tokens)
        if tokens.shape[-1] != self.probe_count:
            raise ValueError("tokens probe dimension must match model probe_count")

        hidden = self.input(tokens)
        attention_mask = circular_previous_attention_mask(
            bucket_count=tokens.shape[1], k_max=self.k_max, device=tokens.device
        )
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.output(self.output_norm(hidden))


def lpap_surrogate_loss(
    logits: Float[torch.Tensor, "batch buckets probe"],  # noqa: F722
    targets: LPAPSurrogateTargets,
) -> tuple[torch.Tensor, LPAPSurrogateMetrics]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape batch x buckets x probe")
    batch_count, bucket_count, probe_count = logits.shape
    flat_loss = torch_functional.cross_entropy(
        logits.reshape(batch_count * bucket_count, probe_count),
        targets.indices.reshape(batch_count * bucket_count),
        reduction="none",
    ).reshape(batch_count, bucket_count)
    weights = targets.weights.to(dtype=logits.dtype)
    weight_total = weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    loss = (flat_loss * weights).sum() / weight_total

    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets.indices)
    accuracy = correct.to(torch.float32).mean()
    weighted_accuracy = (correct.to(logits.dtype) * weights).sum() / weight_total
    metrics = LPAPSurrogateMetrics(
        loss=float(loss.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        weighted_accuracy=float(weighted_accuracy.detach().cpu()),
        mean_weight=float(weights.mean().detach().cpu()),
    )
    return loss, metrics


def prepare_lpap_surrogate_batch(
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    *,
    bucket_count: int,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
) -> Float[torch.Tensor, "batch buckets probe"]:  # noqa: F722
    if values.ndim != 2:
        raise ValueError("values must have shape batch x n")
    if permutation is not None:
        values = apply_grouped_permutation(values, permutation)
    return fold_grouped_permutation_tokens(values, bucket_count=bucket_count)


def train_lpap_surrogate_step(
    *,
    model: LPAPSurrogateTransformer,
    optimizer: torch.optim.Optimizer,
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
) -> LPAPSurrogateMetrics:
    model.train()
    model_device = next(model.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    tokens = prepare_lpap_surrogate_batch(
        values, bucket_count=bucket_count, permutation=permutation
    )
    targets = lpap_surrogate_targets(tokens.detach(), k_max=k_max)
    logits = model(tokens)
    loss, metrics = lpap_surrogate_loss(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return metrics


def evaluate_lpap_surrogate_batch(
    *,
    model: LPAPSurrogateTransformer,
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
) -> LPAPSurrogateMetrics:
    was_training = model.training
    model.eval()
    model_device = next(model.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    with torch.no_grad():
        tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=bucket_count, permutation=permutation
        )
        targets = lpap_surrogate_targets(tokens, k_max=k_max)
        logits = model(tokens)
        _loss, metrics = lpap_surrogate_loss(logits, targets)
    if was_training:
        model.train()
    return metrics


def train_lpap_surrogate(
    *,
    model: LPAPSurrogateTransformer,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[torch.Tensor],
    bucket_count: int,
    k_max: int,
    steps: int,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
) -> list[LPAPSurrogateMetrics]:
    metrics: list[LPAPSurrogateMetrics] = []
    for step_index, values in enumerate(batches):
        if step_index >= steps:
            break
        metrics.append(
            train_lpap_surrogate_step(
                model=model,
                optimizer=optimizer,
                values=values,
                bucket_count=bucket_count,
                k_max=k_max,
                permutation=permutation,
            )
        )
    return metrics
