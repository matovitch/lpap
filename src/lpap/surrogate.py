from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as torch_functional

from lpap.ops import lpap_tokens_torch
from lpap.permutation import (
    apply_permutation,
    fold_permutation_tokens,
    folded_positions_to_flat_indices,
)
from lpap.transformer import TransformerBlock


@dataclass(frozen=True)
class LPAPSurrogateTargets:
    local_classes: Int[torch.Tensor, "batch buckets"]
    source_indices: Int[torch.Tensor, "batch buckets"]
    weights: Float[torch.Tensor, "batch buckets"]
    buckets: Float[torch.Tensor, "batch buckets"]
    dibs: Int[torch.Tensor, "batch buckets"]
    bucket_probes: Int[torch.Tensor, "batch buckets"]


@dataclass(frozen=True)
class LPAPSurrogateMetrics:
    loss: float
    accuracy: float
    weighted_accuracy: float
    mean_weight: float


def _validate_token_values(tokens: torch.Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape batch x buckets x probe")
    if tokens.shape[1] <= 0 or tokens.shape[2] <= 0:
        raise ValueError("tokens bucket and probe dimensions must be positive")
    if not tokens.dtype.is_floating_point:
        raise TypeError("tokens must be a floating point tensor")


def local_classes_to_source_indices(
    local_classes: Int[torch.Tensor, "batch buckets"],
    *,
    bucket_count: int,
    probe_count: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
) -> Int[torch.Tensor, "batch buckets"]:
    """Map local ``(dib, probe)`` class IDs to global unpermuted source indices."""
    if local_classes.ndim != 2:
        raise ValueError("local_classes must have shape batch x buckets")
    if bucket_count <= 0 or probe_count <= 0:
        raise ValueError("bucket_count and probe_count must be positive")
    if local_classes.shape[1] != bucket_count:
        raise ValueError("local_classes bucket dimension must match bucket_count")

    dibs = local_classes // probe_count
    probes = local_classes % probe_count
    bucket_indices = torch.arange(bucket_count, device=local_classes.device)[None, :]
    source_buckets = (bucket_indices - dibs) % bucket_count
    source_positions = folded_positions_to_flat_indices(
        source_buckets=source_buckets,
        probe_indices=probes,
        bucket_count=bucket_count,
    )
    if permutation is None:
        return source_positions
    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    if permutation.numel() != bucket_count * probe_count:
        raise ValueError("permutation length must match bucket_count * probe_count")
    return (
        permutation.to(device=local_classes.device, dtype=torch.long)
        .index_select(0, source_positions.reshape(-1))
        .reshape_as(source_positions)
    )


def gather_visible_candidates(
    tokens: Float[torch.Tensor, "batch buckets probe"],
    *,
    k_max: int,
) -> Float[torch.Tensor, "batch buckets k_max probe"]:
    """Gather the ``k_max x probe`` reachable candidate values for each bucket."""
    _validate_token_values(tokens)
    bucket_count = tokens.shape[1]
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    if k_max > bucket_count:
        raise ValueError("k_max must be at most bucket_count")

    bucket_arange = torch.arange(bucket_count, device=tokens.device)
    dib_arange = torch.arange(k_max, device=tokens.device)
    source_lanes = (bucket_arange[:, None] - dib_arange[None, :]) % bucket_count
    return tokens[:, source_lanes, :]


def lpap_surrogate_targets(
    tokens: Float[torch.Tensor, "batch buckets probe"],
    *,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
) -> LPAPSurrogateTargets:
    """Compute exact LPAP supervision targets using PyTorch or Triton LPAP."""
    _validate_token_values(tokens)
    bucket_count = tokens.shape[1]
    probe_count = tokens.shape[2]
    if use_triton and tokens.is_cuda:
        from lpap.triton_ops import lpap_tokens_triton

        result = lpap_tokens_triton(tokens.detach(), k_max=k_max)
    else:
        result = lpap_tokens_torch(tokens, k_max=k_max)
    local_classes = result.dibs * probe_count + result.bucket_probes
    source_indices = local_classes_to_source_indices(
        local_classes,
        bucket_count=bucket_count,
        probe_count=probe_count,
        permutation=permutation,
    )
    return LPAPSurrogateTargets(
        local_classes=local_classes,
        source_indices=source_indices,
        weights=result.buckets.abs(),
        buckets=result.buckets,
        dibs=result.dibs,
        bucket_probes=result.bucket_probes,
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
    if k_max > bucket_count:
        raise ValueError("k_max must be at most bucket_count")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    query_positions = torch.arange(bucket_count, device=target_device)[:, None]
    key_positions = torch.arange(bucket_count, device=target_device)[None, :]
    backward_distance = (query_positions - key_positions) % bucket_count
    return backward_distance < k_max


class LPAPSurrogateTransformer(nn.Module):
    """Permutation-invariant LPAP surrogate predicting local ``(dib, probe)`` logits."""

    def __init__(
        self,
        *,
        value_count: int,
        probe_count: int,
        k_max: int,
        hidden_dim: int = 128,
        layer_count: int = 2,
        head_count: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if value_count <= 0:
            raise ValueError("value_count must be positive")
        if probe_count <= 0:
            raise ValueError("probe_count must be positive")
        if value_count % probe_count != 0:
            raise ValueError("value_count must be divisible by probe_count")
        bucket_count = value_count // probe_count
        if k_max <= 0:
            raise ValueError("k_max must be positive")
        if k_max > bucket_count:
            raise ValueError("k_max must be at most bucket_count")
        self.value_count = value_count
        self.probe_count = probe_count
        self.k_max = k_max
        self.bucket_count = bucket_count
        self.local_class_count = k_max * probe_count
        self.input = nn.Linear(probe_count, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    head_count=head_count,
                    dropout=dropout,
                    circular=True,
                )
                for _layer_index in range(layer_count)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.local_class_count)
        self.register_buffer(
            "attention_mask",
            circular_previous_attention_mask(
                bucket_count=self.bucket_count, k_max=k_max
            ),
            persistent=False,
        )

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch buckets probe"],
    ) -> Float[torch.Tensor, "batch buckets local_class"]:
        _validate_token_values(tokens)
        if tokens.shape[-1] != self.probe_count:
            raise ValueError("tokens probe dimension must match model probe_count")
        if tokens.shape[1] != self.bucket_count:
            raise ValueError("tokens bucket dimension must match model bucket_count")

        hidden = self.input(tokens)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=self.attention_mask)
        return self.output(self.output_norm(hidden))


def lpap_surrogate_loss(
    logits: Float[torch.Tensor, "batch buckets local_class"],
    targets: LPAPSurrogateTargets,
) -> tuple[torch.Tensor, LPAPSurrogateMetrics]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape batch x buckets x local_class")
    batch_count, bucket_count, class_count = logits.shape
    if (batch_count, bucket_count) != tuple(targets.local_classes.shape):
        raise ValueError("logits batch/bucket dimensions must match targets")
    flat_loss = torch_functional.cross_entropy(
        logits.reshape(batch_count * bucket_count, class_count),
        targets.local_classes.reshape(batch_count * bucket_count),
        reduction="none",
    ).reshape(batch_count, bucket_count)
    weights = targets.weights.to(dtype=logits.dtype)
    weight_total = weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    loss = (flat_loss * weights).sum() / weight_total

    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets.local_classes)
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
    values: Float[torch.Tensor, "batch n"],
    *,
    bucket_count: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
) -> Float[torch.Tensor, "batch buckets probe"]:
    if values.ndim != 2:
        raise ValueError("values must have shape batch x n")
    if permutation is not None:
        values = apply_permutation(values, permutation)
    return fold_permutation_tokens(values, bucket_count=bucket_count)


def train_lpap_surrogate_step(
    *,
    model: LPAPSurrogateTransformer,
    optimizer: torch.optim.Optimizer,
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
) -> LPAPSurrogateMetrics:
    model.train()
    model_device = next(model.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    tokens = prepare_lpap_surrogate_batch(
        values, bucket_count=bucket_count, permutation=permutation
    )
    targets = lpap_surrogate_targets(
        tokens.detach(), k_max=k_max, permutation=permutation, use_triton=use_triton
    )
    logits = model(tokens)
    loss, metrics = lpap_surrogate_loss(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return metrics


def evaluate_lpap_surrogate_batch(
    *,
    model: LPAPSurrogateTransformer,
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
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
        targets = lpap_surrogate_targets(
            tokens, k_max=k_max, permutation=permutation, use_triton=use_triton
        )
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
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
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
                use_triton=use_triton,
            )
        )
    return metrics
