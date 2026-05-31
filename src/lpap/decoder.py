from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as torch_functional

from lpap.surrogate import (
    LPAPSurrogateTargets,
    LPAPSurrogateTransformer,
    lpap_surrogate_targets,
    prepare_lpap_surrogate_batch,
)
from lpap.transformer import TransformerBlock


@dataclass(frozen=True)
class LPAPDecoderBatch:
    tokens: Float[torch.Tensor, "batch buckets channel"]  # noqa: F722
    values: Float[torch.Tensor, "batch n"]  # noqa: F722
    targets: Int[torch.Tensor, "batch buckets"]  # noqa: F722
    weights: Float[torch.Tensor, "batch buckets"]  # noqa: F722
    amplitudes: Float[torch.Tensor, "batch buckets"]  # noqa: F722
    dibs: Float[torch.Tensor, "batch buckets"]  # noqa: F722
    entropy: Float[torch.Tensor, "batch buckets"]  # noqa: F722
    surrogate_targets: LPAPSurrogateTargets


@dataclass(frozen=True)
class LPAPDecoderMetrics:
    loss: float
    reconstruction_l1: float
    source_ce: float
    source_ce_regularizer: float
    source_ce_weight: float
    accuracy: float
    weighted_accuracy: float
    mean_weight: float
    mean_entropy: float


def _validate_surrogate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError("surrogate logits must have shape batch x buckets x classes")
    if not logits.dtype.is_floating_point:
        raise TypeError("surrogate logits must be a floating point tensor")


def decoder_dibs_from_source_logits(
    logits: Float[torch.Tensor, "batch buckets n"],  # noqa: F722
    *,
    bucket_count: int,
) -> Int[torch.Tensor, "batch buckets"]:  # noqa: F722
    _validate_surrogate_logits(logits)
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if logits.shape[-1] % bucket_count != 0:
        raise ValueError("source logits class count must be divisible by bucket_count")

    source_indices = logits.argmax(dim=-1)
    source_buckets = source_indices % bucket_count
    bucket_indices = torch.arange(bucket_count, device=logits.device)[None, :]
    return (bucket_indices - source_buckets) % bucket_count


def prepare_lpap_decoder_batch(
    *,
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    surrogate_logits: Float[torch.Tensor, "batch buckets classes"],  # noqa: F722
    bucket_count: int,
    k_max: int,
    temperature: float | torch.Tensor,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
) -> LPAPDecoderBatch:
    if values.ndim != 2:
        raise ValueError("values must have shape batch x n")
    _validate_surrogate_logits(surrogate_logits)
    if isinstance(temperature, Real):
        temperature_value = float(temperature)
        if temperature_value <= 0:
            raise ValueError("temperature must be positive")
        temperature_tensor = torch.tensor(
            temperature_value,
            device=surrogate_logits.device,
            dtype=surrogate_logits.dtype,
        )
    else:
        temperature_tensor = temperature.to(
            device=surrogate_logits.device, dtype=surrogate_logits.dtype
        )
    if bool((temperature_tensor <= 0).any().detach().cpu()):
        raise ValueError("temperature must be positive")

    tokens = prepare_lpap_surrogate_batch(
        values, bucket_count=bucket_count, permutation=permutation
    )
    batch_count, actual_bucket_count, _probe_count = tokens.shape
    if actual_bucket_count != bucket_count:
        raise ValueError("bucket_count does not match folded token shape")
    if surrogate_logits.shape[:2] != (batch_count, bucket_count):
        raise ValueError("surrogate logits batch/bucket dimensions must match values")

    targets = lpap_surrogate_targets(tokens.detach(), k_max=k_max)
    probabilities = torch.softmax(surrogate_logits / temperature_tensor, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    if surrogate_logits.shape[-1] != values.shape[-1]:
        raise ValueError("surrogate logits class count must equal value_count")

    amplitudes = (probabilities * values[:, None, :]).sum(dim=-1)
    dibs = decoder_dibs_from_source_logits(surrogate_logits, bucket_count=bucket_count)
    if permutation is not None:
        targets = lpap_surrogate_targets(
            tokens.detach(), k_max=k_max, permutation=permutation
        )

    normalizer = max(bucket_count - 1, 1)
    decoder_tokens = torch.stack(
        (
            amplitudes,
            dibs.to(dtype=amplitudes.dtype) / normalizer,
            entropy.to(dtype=amplitudes.dtype),
        ),
        dim=-1,
    )
    return LPAPDecoderBatch(
        tokens=decoder_tokens,
        values=values,
        targets=targets.source_indices,
        weights=amplitudes.abs(),
        amplitudes=amplitudes,
        dibs=dibs.to(dtype=amplitudes.dtype),
        entropy=entropy,
        surrogate_targets=targets,
    )


class LPAPDecoderTransformer(nn.Module):
    def __init__(
        self,
        *,
        value_count: int,
        input_dim: int = 3,
        frontend_initial_temperature: float = 0.25,
        hidden_dim: int = 128,
        layer_count: int = 2,
        head_count: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if value_count <= 0:
            raise ValueError("value_count must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if frontend_initial_temperature <= 0:
            raise ValueError("frontend_initial_temperature must be positive")
        self.value_count = value_count
        self.log_frontend_temperature = nn.Parameter(
            torch.tensor(math.log(frontend_initial_temperature), dtype=torch.float32)
        )
        self.input = nn.Linear(input_dim, hidden_dim)
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
        self.output = nn.Linear(hidden_dim, value_count)

    def frontend_temperature(self) -> torch.Tensor:
        return self.log_frontend_temperature.exp()

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch buckets channel"],  # noqa: F722
    ) -> Float[torch.Tensor, "batch buckets n"]:  # noqa: F722
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape batch x buckets x channel")
        hidden = self.input(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))


def reconstruct_lpap_decoder_values(
    logits: Float[torch.Tensor, "batch buckets n"],  # noqa: F722
    batch: LPAPDecoderBatch,
) -> Float[torch.Tensor, "batch n"]:  # noqa: F722
    if logits.ndim != 3:
        raise ValueError("logits must have shape batch x buckets x n")
    if logits.shape[:2] != batch.tokens.shape[:2]:
        raise ValueError("logits batch/bucket dimensions must match decoder batch")
    if logits.shape[-1] != batch.values.shape[-1]:
        raise ValueError("logits class count must match batch value_count")
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities * batch.amplitudes[..., None]).sum(dim=1)


def reconstruct_lpap_bucket_values(
    batch: LPAPDecoderBatch,
) -> Float[torch.Tensor, "batch n"]:  # noqa: F722
    reconstruction = torch.zeros_like(batch.values)
    return reconstruction.scatter_add(
        dim=1,
        index=batch.targets,
        src=batch.surrogate_targets.buckets.to(dtype=batch.values.dtype),
    )


def lpap_decoder_loss(
    logits: Float[torch.Tensor, "batch buckets n"],  # noqa: F722
    batch: LPAPDecoderBatch,
    *,
    source_ce_weight: float = 0.0,
    source_ce_l1_reference: float = 1.0,
    source_ce_power: float = 2.0,
) -> tuple[torch.Tensor, LPAPDecoderMetrics]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape batch x buckets x n")
    if source_ce_weight < 0:
        raise ValueError("source_ce_weight must be non-negative")
    if source_ce_l1_reference <= 0:
        raise ValueError("source_ce_l1_reference must be positive")
    if source_ce_power <= 0:
        raise ValueError("source_ce_power must be positive")
    reconstruction = reconstruct_lpap_decoder_values(logits, batch)
    reconstruction_l1 = torch_functional.l1_loss(
        reconstruction, batch.values, reduction="mean"
    )
    weights = batch.weights.to(dtype=logits.dtype)
    weight_total = weights.sum().clamp_min(torch.finfo(logits.dtype).eps)

    source_ce_per_bucket = torch_functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.targets.reshape(-1),
        reduction="none",
    ).reshape_as(batch.targets)
    source_ce = (source_ce_per_bucket * weights).sum() / weight_total
    adaptive_ce_weight = source_ce_weight * float(
        (reconstruction_l1.detach() / source_ce_l1_reference)
        .clamp(0.0, 1.0)
        .pow(source_ce_power)
        .cpu()
    )
    source_ce_regularizer = source_ce * adaptive_ce_weight
    loss = reconstruction_l1 + source_ce_regularizer

    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(batch.targets)
    accuracy = correct.to(torch.float32).mean()
    weighted_accuracy = (correct.to(logits.dtype) * weights).sum() / weight_total
    metrics = LPAPDecoderMetrics(
        loss=float(loss.detach().cpu()),
        reconstruction_l1=float(reconstruction_l1.detach().cpu()),
        source_ce=float(source_ce.detach().cpu()),
        source_ce_regularizer=float(source_ce_regularizer.detach().cpu()),
        source_ce_weight=adaptive_ce_weight,
        accuracy=float(accuracy.detach().cpu()),
        weighted_accuracy=float(weighted_accuracy.detach().cpu()),
        mean_weight=float(weights.mean().detach().cpu()),
        mean_entropy=float(batch.entropy.mean().detach().cpu()),
    )
    return loss, metrics


def train_lpap_decoder_step(
    *,
    decoder: LPAPDecoderTransformer,
    surrogate: LPAPSurrogateTransformer,
    optimizer: torch.optim.Optimizer,
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
    source_ce_weight: float = 0.0,
    source_ce_l1_reference: float = 1.0,
    source_ce_power: float = 2.0,
) -> LPAPDecoderMetrics:
    decoder.train()
    surrogate.eval()
    model_device = next(decoder.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    with torch.no_grad():
        surrogate_tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=bucket_count, permutation=permutation
        )
        surrogate_logits = surrogate(surrogate_tokens)
    decoder_batch = prepare_lpap_decoder_batch(
        values=values,
        surrogate_logits=surrogate_logits,
        bucket_count=bucket_count,
        k_max=k_max,
        temperature=decoder.frontend_temperature(),
        permutation=permutation,
    )
    logits = decoder(decoder_batch.tokens)
    loss, metrics = lpap_decoder_loss(
        logits,
        decoder_batch,
        source_ce_weight=source_ce_weight,
        source_ce_l1_reference=source_ce_l1_reference,
        source_ce_power=source_ce_power,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return metrics


def evaluate_lpap_decoder_batch(
    *,
    decoder: LPAPDecoderTransformer,
    surrogate: LPAPSurrogateTransformer,
    values: Float[torch.Tensor, "batch n"],  # noqa: F722
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,  # noqa: F722, F821
    source_ce_weight: float = 0.0,
    source_ce_l1_reference: float = 1.0,
    source_ce_power: float = 2.0,
) -> LPAPDecoderMetrics:
    decoder_was_training = decoder.training
    surrogate_was_training = surrogate.training
    decoder.eval()
    surrogate.eval()
    model_device = next(decoder.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    with torch.no_grad():
        surrogate_tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=bucket_count, permutation=permutation
        )
        surrogate_logits = surrogate(surrogate_tokens)
        decoder_batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=bucket_count,
            k_max=k_max,
            temperature=decoder.frontend_temperature(),
            permutation=permutation,
        )
        logits = decoder(decoder_batch.tokens)
        _loss, metrics = lpap_decoder_loss(
            logits,
            decoder_batch,
            source_ce_weight=source_ce_weight,
            source_ce_l1_reference=source_ce_l1_reference,
            source_ce_power=source_ce_power,
        )
    if decoder_was_training:
        decoder.train()
    if surrogate_was_training:
        surrogate.train()
    return metrics
