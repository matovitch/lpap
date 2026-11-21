from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as torch_functional

from lpap.ops import lpap_tokens_torch
from lpap.surrogate import (
    LPAPSurrogateTargets,
    LPAPSurrogateTransformer,
    gather_visible_candidates,
    local_classes_to_source_indices,
    lpap_surrogate_targets,
    prepare_lpap_surrogate_batch,
)
from lpap.transformer import TransformerBlock


@dataclass(frozen=True)
class LPAPDecoderBatch:
    tokens: Float[torch.Tensor, "batch buckets channel"]
    values: Float[torch.Tensor, "batch n"]
    targets: Int[torch.Tensor, "batch buckets"]
    weights: Float[torch.Tensor, "batch buckets"]
    amplitudes: Float[torch.Tensor, "batch buckets"]
    dibs: Float[torch.Tensor, "batch buckets"]
    entropy: Float[torch.Tensor, "batch buckets"]
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


def _validate_temperature(
    temperature: int | float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(temperature, int | float):
        temperature_value = float(temperature)
        if not math.isfinite(temperature_value) or temperature_value <= 0:
            raise ValueError("temperature must be positive")
        return torch.tensor(temperature_value, device=device, dtype=dtype)
    if temperature.numel() != 1:
        raise ValueError("temperature tensor must have exactly one element")
    if not bool(torch.isfinite(temperature).all()) or not bool((temperature > 0).all()):
        raise ValueError("temperature must be positive")
    return temperature.to(device=device, dtype=dtype)


def normalize_lpap_dibs(
    dibs: Int[torch.Tensor, "... buckets"],
    *,
    k_max: int,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "... buckets"]:
    """Scale integer DIBs in ``[0, k_max - 1]`` consistently onto ``[0, 1]``."""
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    normalizer = max(k_max - 1, 1)
    return dibs.to(dtype=dtype) / normalizer


def decoder_dibs_from_local_logits(
    logits: Float[torch.Tensor, "batch buckets local_class"],
    *,
    bucket_count: int,
    probe_count: int,
    k_max: int,
) -> Int[torch.Tensor, "batch buckets"]:
    """Extract per-bucket DIBs directly from local ``(dib, probe)`` logits."""
    _validate_surrogate_logits(logits)
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    if k_max > bucket_count:
        raise ValueError("k_max must be at most bucket_count")
    if logits.shape[1] != bucket_count:
        raise ValueError("surrogate logits bucket dimension must match bucket_count")
    expected_classes = k_max * probe_count
    if logits.shape[-1] != expected_classes:
        raise ValueError(
            f"local surrogate logits class count must equal k_max * probe_count ({expected_classes})"
        )
    return logits.argmax(dim=-1) // probe_count


def prepare_lpap_decoder_batch(
    *,
    values: Float[torch.Tensor, "batch n"],
    surrogate_logits: Float[torch.Tensor, "batch buckets local_class"],
    bucket_count: int,
    k_max: int,
    temperature: int | float | torch.Tensor,
    permutation: Int[torch.Tensor, "n"] | None = None,
    surrogate_tokens: Float[torch.Tensor, "batch buckets probe"] | None = None,
) -> LPAPDecoderBatch:
    if values.ndim != 2:
        raise ValueError("values must have shape batch x n")
    _validate_surrogate_logits(surrogate_logits)
    temperature_tensor = _validate_temperature(
        temperature, device=surrogate_logits.device, dtype=surrogate_logits.dtype
    )

    tokens = (
        prepare_lpap_surrogate_batch(
            values, bucket_count=bucket_count, permutation=permutation
        )
        if surrogate_tokens is None
        else surrogate_tokens
    )
    batch_count, actual_bucket_count, probe_count = tokens.shape
    if actual_bucket_count != bucket_count:
        raise ValueError("bucket_count does not match folded token shape")
    if surrogate_logits.shape[:2] != (batch_count, bucket_count):
        raise ValueError("surrogate logits batch/bucket dimensions must match values")

    dibs = decoder_dibs_from_local_logits(
        surrogate_logits,
        bucket_count=bucket_count,
        probe_count=probe_count,
        k_max=k_max,
    )
    targets = lpap_surrogate_targets(
        tokens.detach(), k_max=k_max, permutation=permutation
    )

    probabilities = torch.softmax(surrogate_logits / temperature_tensor, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    visible = gather_visible_candidates(tokens, k_max=k_max).reshape(
        batch_count, bucket_count, k_max * probe_count
    )
    amplitudes = (probabilities * visible).sum(dim=-1)
    normalized_dibs = normalize_lpap_dibs(dibs, k_max=k_max, dtype=amplitudes.dtype)
    decoder_tokens = torch.stack(
        (
            amplitudes,
            normalized_dibs,
            entropy.to(dtype=amplitudes.dtype),
        ),
        dim=-1,
    )
    return LPAPDecoderBatch(
        tokens=decoder_tokens,
        values=values,
        targets=targets.source_indices,
        weights=amplitudes.abs().detach(),
        amplitudes=amplitudes,
        dibs=dibs.to(dtype=amplitudes.dtype),
        entropy=entropy,
        surrogate_targets=targets,
    )


def prepare_exact_lpap_decoder_batch(
    *,
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
) -> LPAPDecoderBatch:
    """Build decoder tokens directly from exact LPAP ``(amplitude, dib, 0)``."""
    if values.ndim != 2:
        raise ValueError("values must have shape batch x n")
    tokens = prepare_lpap_surrogate_batch(
        values, bucket_count=bucket_count, permutation=permutation
    )
    _batch_count, actual_bucket_count, probe_count = tokens.shape
    if actual_bucket_count != bucket_count:
        raise ValueError("bucket_count does not match folded token shape")

    detached_tokens = tokens.detach()
    if use_triton and detached_tokens.is_cuda:
        from lpap.triton_ops import lpap_tokens_triton

        result = lpap_tokens_triton(detached_tokens, k_max=k_max)
    else:
        result = lpap_tokens_torch(detached_tokens, k_max=k_max)

    local_classes = result.dibs * probe_count + result.bucket_probes
    source_indices = local_classes_to_source_indices(
        local_classes,
        bucket_count=bucket_count,
        probe_count=probe_count,
        permutation=permutation,
    )
    targets = LPAPSurrogateTargets(
        local_classes=local_classes,
        source_indices=source_indices,
        weights=result.buckets.abs(),
        buckets=result.buckets,
        dibs=result.dibs,
        bucket_probes=result.bucket_probes,
    )
    amplitudes = result.buckets
    normalized_dibs = normalize_lpap_dibs(
        result.dibs, k_max=k_max, dtype=amplitudes.dtype
    )
    entropy = torch.zeros_like(amplitudes)
    decoder_tokens = torch.stack((amplitudes, normalized_dibs, entropy), dim=-1)
    return LPAPDecoderBatch(
        tokens=decoder_tokens,
        values=values,
        targets=targets.source_indices,
        weights=amplitudes.abs(),
        amplitudes=amplitudes,
        dibs=result.dibs.to(dtype=amplitudes.dtype),
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
        tokens: Float[torch.Tensor, "batch buckets channel"],
    ) -> Float[torch.Tensor, "batch buckets n"]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape batch x buckets x channel")
        hidden = self.input(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))


def reconstruct_lpap_decoder_values(
    logits: Float[torch.Tensor, "batch buckets n"],
    batch: LPAPDecoderBatch,
) -> Float[torch.Tensor, "batch n"]:
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
) -> Float[torch.Tensor, "batch n"]:
    reconstruction = torch.zeros_like(batch.values)
    return reconstruction.scatter_add(
        dim=1,
        index=batch.targets,
        src=batch.surrogate_targets.buckets.to(dtype=batch.values.dtype),
    )


def lpap_decoder_loss(
    logits: Float[torch.Tensor, "batch buckets n"],
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
    adaptive_ce_weight = source_ce_weight * (
        reconstruction_l1.detach() / source_ce_l1_reference
    ).clamp(0.0, 1.0).pow(source_ce_power)
    source_ce_regularizer = source_ce * adaptive_ce_weight

    source_ce_weight_metric = float(adaptive_ce_weight.detach().cpu())
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
        source_ce_weight=source_ce_weight_metric,
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
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
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
        surrogate_tokens=surrogate_tokens,
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
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
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
            surrogate_tokens=surrogate_tokens,
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


def train_lpap_decoder_exact_step(
    *,
    decoder: LPAPDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
    source_ce_weight: float = 0.0,
    source_ce_l1_reference: float = 1.0,
    source_ce_power: float = 2.0,
) -> LPAPDecoderMetrics:
    """Single optimizer step fine-tuning only the decoder on exact LPAP."""
    decoder.train()
    model_device = next(decoder.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    decoder_batch = prepare_exact_lpap_decoder_batch(
        values=values,
        bucket_count=bucket_count,
        k_max=k_max,
        permutation=permutation,
        use_triton=use_triton,
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


def evaluate_lpap_decoder_exact_batch(
    *,
    decoder: LPAPDecoderTransformer,
    values: Float[torch.Tensor, "batch n"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    use_triton: bool = False,
    source_ce_weight: float = 0.0,
    source_ce_l1_reference: float = 1.0,
    source_ce_power: float = 2.0,
) -> LPAPDecoderMetrics:
    """Evaluate a decoder directly on exact LPAP ``(amplitude, dib, 0)``."""
    decoder_was_training = decoder.training
    decoder.eval()
    model_device = next(decoder.parameters()).device
    values = values.to(model_device)
    if permutation is not None:
        permutation = permutation.to(model_device)
    with torch.no_grad():
        decoder_batch = prepare_exact_lpap_decoder_batch(
            values=values,
            bucket_count=bucket_count,
            k_max=k_max,
            permutation=permutation,
            use_triton=use_triton,
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
    return metrics
