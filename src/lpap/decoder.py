from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as torch_functional

from lpap.ops import lpap_tokens_torch
from lpap.permutation import folded_positions_to_flat_indices
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
    amplitudes: Float[torch.Tensor, "batch buckets"]
    dib_weights: Float[torch.Tensor, "batch buckets k_max"]
    dibs: Int[torch.Tensor, "batch buckets"]
    permutation: Int[torch.Tensor, "n"] | None
    bucket_count: int
    probe_count: int
    k_max: int
    values: Float[torch.Tensor, "batch n"]
    targets: Int[torch.Tensor, "batch buckets"]
    target_probes: Int[torch.Tensor, "batch buckets"]
    weights: Float[torch.Tensor, "batch buckets"]
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


# (C, P, k_max, permutation data_ptr, device) -> (permutation, indices).
# A hit also checks tensor equality so a recycled data_ptr cannot reuse a
# table built for a different permutation.
_CANDIDATE_INDEX_CACHE: dict[
    tuple[int, int, int, int | None, str, int | None],
    tuple[torch.Tensor | None, torch.Tensor],
] = {}


def _candidate_source_indices(
    *,
    bucket_count: int,
    probe_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None,
    device: torch.device,
) -> Int[torch.Tensor, "buckets k_max probe"]:
    cache_key = (
        bucket_count,
        probe_count,
        k_max,
        None if permutation is None else permutation.data_ptr(),
        device.type,
        device.index,
    )
    cached = _CANDIDATE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        cached_permutation, cached_indices = cached
        same_permutation = permutation is None or (
            cached_permutation is not None
            and cached_indices.device == device
            and torch.equal(cached_permutation, permutation)
        )
        if same_permutation and cached_indices.device == device:
            return cached_indices
    if len(_CANDIDATE_INDEX_CACHE) > 32:
        _CANDIDATE_INDEX_CACHE.clear()

    bucket_arange = torch.arange(bucket_count, device=device, dtype=torch.long)
    dib_arange = torch.arange(k_max, device=device, dtype=torch.long)
    probe_arange = torch.arange(probe_count, device=device, dtype=torch.long)
    source_buckets = (bucket_arange[:, None] - dib_arange[None, :]) % bucket_count
    flat_indices = folded_positions_to_flat_indices(
        source_buckets=source_buckets[:, :, None],
        probe_indices=probe_arange[None, None, :],
        bucket_count=bucket_count,
    )
    if permutation is not None:
        if permutation.ndim != 1:
            raise ValueError("permutation must be one-dimensional")
        if permutation.numel() != bucket_count * probe_count:
            raise ValueError("permutation length must match bucket_count * probe_count")
        flat_indices = (
            permutation.to(device=device, dtype=torch.long)
            .index_select(0, flat_indices.reshape(-1))
            .reshape_as(flat_indices)
        )
    _CANDIDATE_INDEX_CACHE[cache_key] = (permutation, flat_indices)
    return flat_indices


def prepare_lpap_decoder_batch(
    *,
    values: Float[torch.Tensor, "batch n"],
    surrogate_logits: Float[torch.Tensor, "batch buckets local_class"],
    bucket_count: int,
    k_max: int,
    permutation: Int[torch.Tensor, "n"] | None = None,
    surrogate_tokens: Float[torch.Tensor, "batch buckets probe"] | None = None,
    use_triton: bool = False,
) -> LPAPDecoderBatch:
    """Build decoder inputs via categorical straight-through over local surrogate classes."""
    if values.ndim != 2:
        raise ValueError("values must have shape batch x n")
    _validate_surrogate_logits(surrogate_logits)

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
        tokens.detach(),
        k_max=k_max,
        permutation=permutation,
        use_triton=use_triton,
    )

    probabilities = torch.softmax(surrogate_logits, dim=-1)
    hard_class = surrogate_logits.argmax(dim=-1)
    one_hot = torch_functional.one_hot(
        hard_class, num_classes=k_max * probe_count
    ).to(dtype=probabilities.dtype)
    # Parenthesised so BF16 forward stays bit-exact one-hot while gradients flow through p.
    p_st = one_hot + (probabilities - probabilities.detach())

    visible = gather_visible_candidates(tokens, k_max=k_max).reshape(
        batch_count, bucket_count, k_max * probe_count
    )
    amplitudes = (p_st * visible.to(dtype=p_st.dtype)).sum(dim=-1)
    dib_weights = p_st.reshape(batch_count, bucket_count, k_max, probe_count).sum(
        dim=-1
    )
    with torch.no_grad():
        probs_detached = probabilities.detach()
        entropy = -(
            probs_detached * probs_detached.clamp_min(1.0e-12).log()
        ).sum(dim=-1)

    return LPAPDecoderBatch(
        amplitudes=amplitudes,
        dib_weights=dib_weights,
        dibs=dibs,
        permutation=permutation,
        bucket_count=bucket_count,
        probe_count=probe_count,
        k_max=k_max,
        values=values,
        targets=targets.source_indices,
        target_probes=targets.bucket_probes,
        weights=amplitudes.abs().detach(),
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
    """Build decoder inputs directly from exact LPAP ``(amplitude, one-hot dib)``."""
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
    dib_weights = torch_functional.one_hot(result.dibs, num_classes=k_max).to(
        dtype=amplitudes.dtype
    )
    entropy = torch.zeros_like(amplitudes)
    return LPAPDecoderBatch(
        amplitudes=amplitudes,
        dib_weights=dib_weights,
        dibs=result.dibs,
        permutation=permutation,
        bucket_count=bucket_count,
        probe_count=probe_count,
        k_max=k_max,
        values=values,
        targets=targets.source_indices,
        target_probes=targets.bucket_probes,
        weights=amplitudes.abs(),
        entropy=entropy,
        surrogate_targets=targets,
    )


class LPAPDecoderTransformer(nn.Module):
    """Circular decoder predicting ``P``-way probe logits per bucket."""

    def __init__(
        self,
        *,
        bucket_count: int,
        probe_count: int,
        k_max: int,
        hidden_dim: int = 128,
        layer_count: int = 2,
        head_count: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if probe_count <= 0:
            raise ValueError("probe_count must be positive")
        if k_max <= 0:
            raise ValueError("k_max must be positive")
        if k_max > bucket_count:
            raise ValueError("k_max must be at most bucket_count")

        self.bucket_count = bucket_count
        self.probe_count = probe_count
        self.k_max = k_max
        self.value_count = bucket_count * probe_count

        self.amplitude_proj = nn.Linear(1, hidden_dim)
        self.source_bucket_embed = nn.Embedding(bucket_count, hidden_dim)
        self.dib_embed = nn.Linear(k_max, hidden_dim, bias=False)
        self.bucket_pos_embed = nn.Parameter(
            torch.empty(1, bucket_count, hidden_dim)
        )
        nn.init.normal_(self.bucket_pos_embed, std=0.02)

        bucket_arange = torch.arange(bucket_count, dtype=torch.long)
        dib_arange = torch.arange(k_max, dtype=torch.long)
        source_bucket_table = (
            bucket_arange[:, None] - dib_arange[None, :]
        ) % bucket_count
        self.register_buffer(
            "source_bucket_table", source_bucket_table, persistent=False
        )

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
        self.output = nn.Linear(hidden_dim, probe_count)

    def forward(
        self,
        amplitudes: Float[torch.Tensor, "batch buckets"],
        dib_weights: Float[torch.Tensor, "batch buckets k_max"],
    ) -> Float[torch.Tensor, "batch buckets probe"]:
        if amplitudes.ndim != 2 or amplitudes.shape[1] != self.bucket_count:
            raise ValueError(
                f"amplitudes must have shape batch x {self.bucket_count}"
            )
        if dib_weights.ndim != 3 or dib_weights.shape != (
            amplitudes.shape[0],
            self.bucket_count,
            self.k_max,
        ):
            raise ValueError(
                f"dib_weights must have shape batch x {self.bucket_count} x {self.k_max}"
            )

        amp_term = self.amplitude_proj(amplitudes.unsqueeze(-1))
        compute_dtype = amp_term.dtype
        dw = dib_weights.to(dtype=compute_dtype)
        source_embeds = self.source_bucket_embed(self.source_bucket_table).to(
            dtype=compute_dtype
        )
        source_term = torch.einsum("bck,ckd->bcd", dw, source_embeds)
        dib_term = self.dib_embed(dw)
        hidden = (
            amp_term
            + source_term
            + dib_term
            + self.bucket_pos_embed.to(dtype=compute_dtype)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))


def decoder_predicted_source_indices(
    probe_logits: Float[torch.Tensor, "batch buckets probe"],
    batch: LPAPDecoderBatch,
) -> Int[torch.Tensor, "batch buckets"]:
    """Map argmax probe predictions and bucket DIBs to global source indices."""
    if probe_logits.ndim != 3:
        raise ValueError("probe_logits must have shape batch x buckets x probe")
    batch_count = batch.amplitudes.shape[0]
    if probe_logits.shape != (batch_count, batch.bucket_count, batch.probe_count):
        raise ValueError(
            "probe_logits shape must match (batch, bucket_count, probe_count)"
        )
    predicted_probes = probe_logits.argmax(dim=-1)
    local_classes = batch.dibs * batch.probe_count + predicted_probes
    return local_classes_to_source_indices(
        local_classes,
        bucket_count=batch.bucket_count,
        probe_count=batch.probe_count,
        permutation=batch.permutation,
    )


def reconstruct_lpap_decoder_values(
    probe_logits: Float[torch.Tensor, "batch buckets probe"],
    batch: LPAPDecoderBatch,
) -> Float[torch.Tensor, "batch n"]:
    if probe_logits.ndim != 3:
        raise ValueError("probe_logits must have shape batch x buckets x probe")
    batch_count = batch.amplitudes.shape[0]
    if probe_logits.shape != (batch_count, batch.bucket_count, batch.probe_count):
        raise ValueError(
            "probe_logits shape must match (batch, bucket_count, probe_count)"
        )
    probe_probs = torch.softmax(probe_logits, dim=-1)
    dtype = probe_probs.dtype
    amplitudes = batch.amplitudes.to(dtype=dtype)
    dib_weights = batch.dib_weights.to(dtype=dtype)
    masses = (
        dib_weights.unsqueeze(-1)
        * probe_probs.unsqueeze(-2)
        * amplitudes[:, :, None, None]
    )
    candidate_indices = _candidate_source_indices(
        bucket_count=batch.bucket_count,
        probe_count=batch.probe_count,
        k_max=batch.k_max,
        permutation=batch.permutation,
        device=probe_logits.device,
    )
    flat_indices = candidate_indices.reshape(1, -1).expand(batch_count, -1)
    flat_masses = masses.reshape(batch_count, -1)
    reconstruction = torch.zeros(
        (batch_count, batch.values.shape[-1]),
        device=probe_logits.device,
        dtype=dtype,
    )
    return reconstruction.scatter_add(dim=1, index=flat_indices, src=flat_masses)


def reconstruct_lpap_bucket_values(
    batch: LPAPDecoderBatch,
) -> Float[torch.Tensor, "batch n"]:
    reconstruction = torch.zeros_like(batch.values)
    return reconstruction.scatter_add(
        dim=1,
        index=batch.targets,
        src=batch.surrogate_targets.buckets.to(dtype=batch.values.dtype),
    )


@dataclass(frozen=True)
class LPAPDecoderProbeStats:
    """Probe cross-entropy and accuracy shared by the decoder loss and the AE."""

    source_ce: torch.Tensor
    accuracy: torch.Tensor
    weighted_accuracy: torch.Tensor
    dib_accuracy: torch.Tensor


def lpap_decoder_probe_stats(
    probe_logits: Float[torch.Tensor, "batch buckets probe"],
    batch: LPAPDecoderBatch,
) -> LPAPDecoderProbeStats:
    """DIB-masked probe CE plus end-to-end and DIB-only accuracy."""
    if probe_logits.ndim != 3:
        raise ValueError("probe_logits must have shape batch x buckets x probe")
    weights = batch.weights.to(dtype=probe_logits.dtype)
    weight_total = weights.sum().clamp_min(torch.finfo(probe_logits.dtype).eps)
    dib_match = batch.dibs.eq(batch.surrogate_targets.dibs)
    matched_weights = weights * dib_match.to(dtype=probe_logits.dtype)
    matched_weight_total = matched_weights.sum().clamp_min(
        torch.finfo(probe_logits.dtype).eps
    )
    source_ce_per_bucket = torch_functional.cross_entropy(
        probe_logits.reshape(-1, probe_logits.shape[-1]),
        batch.target_probes.reshape(-1),
        reduction="none",
    ).reshape_as(batch.target_probes)
    source_ce = (source_ce_per_bucket * matched_weights).sum() / matched_weight_total
    predicted_probes = probe_logits.argmax(dim=-1)
    correct = dib_match & predicted_probes.eq(batch.target_probes)
    accuracy = correct.to(torch.float32).mean()
    weighted_accuracy = (correct.to(probe_logits.dtype) * weights).sum() / weight_total
    dib_accuracy = dib_match.to(torch.float32).mean()
    return LPAPDecoderProbeStats(
        source_ce=source_ce,
        accuracy=accuracy,
        weighted_accuracy=weighted_accuracy,
        dib_accuracy=dib_accuracy,
    )


def lpap_decoder_loss(
    probe_logits: Float[torch.Tensor, "batch buckets probe"],
    batch: LPAPDecoderBatch,
    *,
    source_ce_weight: float = 0.0,
    source_ce_l1_reference: float = 1.0,
    source_ce_power: float = 2.0,
) -> tuple[torch.Tensor, LPAPDecoderMetrics]:
    if probe_logits.ndim != 3:
        raise ValueError("probe_logits must have shape batch x buckets x probe")
    if source_ce_weight < 0:
        raise ValueError("source_ce_weight must be non-negative")
    if source_ce_l1_reference <= 0:
        raise ValueError("source_ce_l1_reference must be positive")
    if source_ce_power <= 0:
        raise ValueError("source_ce_power must be positive")

    reconstruction = reconstruct_lpap_decoder_values(probe_logits, batch)
    reconstruction_l1 = torch_functional.l1_loss(
        reconstruction, batch.values, reduction="mean"
    )
    probe_stats = lpap_decoder_probe_stats(probe_logits, batch)
    source_ce = probe_stats.source_ce
    adaptive_ce_weight = source_ce_weight * (
        reconstruction_l1.detach() / source_ce_l1_reference
    ).clamp(0.0, 1.0).pow(source_ce_power)
    source_ce_regularizer = source_ce * adaptive_ce_weight

    source_ce_weight_metric = float(adaptive_ce_weight.detach().cpu())
    loss = reconstruction_l1 + source_ce_regularizer
    weights = batch.weights.to(dtype=probe_logits.dtype)
    metrics = LPAPDecoderMetrics(
        loss=float(loss.detach().cpu()),
        reconstruction_l1=float(reconstruction_l1.detach().cpu()),
        source_ce=float(source_ce.detach().cpu()),
        source_ce_regularizer=float(source_ce_regularizer.detach().cpu()),
        source_ce_weight=source_ce_weight_metric,
        accuracy=float(probe_stats.accuracy.detach().cpu()),
        weighted_accuracy=float(probe_stats.weighted_accuracy.detach().cpu()),
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
    use_triton: bool = False,
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
        permutation=permutation,
        surrogate_tokens=surrogate_tokens,
        use_triton=use_triton,
    )
    probe_logits = decoder(decoder_batch.amplitudes, decoder_batch.dib_weights)
    loss, metrics = lpap_decoder_loss(
        probe_logits,
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
    use_triton: bool = False,
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
            permutation=permutation,
            surrogate_tokens=surrogate_tokens,
            use_triton=use_triton,
        )
        probe_logits = decoder(decoder_batch.amplitudes, decoder_batch.dib_weights)
        _loss, metrics = lpap_decoder_loss(
            probe_logits,
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
    """Single optimizer step training only the decoder on exact LPAP."""
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
    probe_logits = decoder(decoder_batch.amplitudes, decoder_batch.dib_weights)
    loss, metrics = lpap_decoder_loss(
        probe_logits,
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
    """Evaluate a decoder directly on exact LPAP ``(amplitude, one-hot dib)``."""
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
        probe_logits = decoder(decoder_batch.amplitudes, decoder_batch.dib_weights)
        _loss, metrics = lpap_decoder_loss(
            probe_logits,
            decoder_batch,
            source_ce_weight=source_ce_weight,
            source_ce_l1_reference=source_ce_l1_reference,
            source_ce_power=source_ce_power,
        )
    if decoder_was_training:
        decoder.train()
    return metrics
