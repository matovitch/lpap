from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as torch_functional


@dataclass(frozen=True)
class FlowMatchingMetrics:
    loss: float
    velocity_mse: float
    velocity_cosine: float
    velocity_rel_l2_percent: float
    image_rms: float
    target_rms: float
    image_mean: float
    target_mean: float


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(
        self,
        time: Float[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch dim"]:
        if time.ndim != 1:
            raise ValueError("time must have shape batch")
        half_dim = self.dim // 2
        if half_dim == 0:
            return time[:, None]
        frequencies = torch.exp(
            torch.arange(half_dim, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        embedding = time[:, None] * frequencies[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return embedding


class DilatedResidualBlock1d(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        time_dim: int,
        dilation: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if time_dim <= 0:
            raise ValueError("time_dim must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = dilation * (kernel_size // 2)
        self.norm = nn.GroupNorm(1, width)
        self.time = nn.Linear(time_dim, width * 2)
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(
        self,
        values: Float[torch.Tensor, "batch width n"],
        time_embedding: Float[torch.Tensor, "batch time"],
    ) -> Float[torch.Tensor, "batch width n"]:
        scale, shift = self.time(time_embedding).chunk(2, dim=-1)
        hidden = self.norm(values)
        hidden = hidden * (1.0 + scale[:, :, None]) + shift[:, :, None]
        hidden = torch_functional.silu(hidden)
        return values + self.conv(hidden)


class DilatedConvFlow1d(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        width: int = 128,
        time_dim: int = 128,
        dilation_cycles: int = 2,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size: int = 3,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if time_dim <= 0:
            raise ValueError("time_dim must be positive")
        if dilation_cycles <= 0:
            raise ValueError("dilation_cycles must be positive")
        if not dilations:
            raise ValueError("dilations must not be empty")
        self.sequence_length = sequence_length
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv1d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock1d(
                    width=width,
                    time_dim=time_dim,
                    dilation=dilation,
                    kernel_size=kernel_size,
                )
                for _cycle in range(dilation_cycles)
                for dilation in dilations
            ]
        )
        self.output_norm = nn.GroupNorm(1, width)
        self.output = nn.Conv1d(width, 1, kernel_size=1)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        values: Float[torch.Tensor, "batch 1 n"],
        time: Float[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch 1 n"]:
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("values must have shape batch x 1 x n")
        if values.shape[2] != self.sequence_length:
            raise ValueError("values sequence length must match model sequence_length")
        if time.ndim != 1 or time.shape[0] != values.shape[0]:
            raise ValueError("time must have shape batch")
        time_embedding = self.time_embedding(time.to(dtype=values.dtype))
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return self.output(torch_functional.silu(self.output_norm(hidden)))


def interpolate_linear(
    start: Float[torch.Tensor, "batch 1 n"],
    end: Float[torch.Tensor, "batch 1 n"],
    time: Float[torch.Tensor, "batch"],
) -> Float[torch.Tensor, "batch 1 n"]:
    if start.shape != end.shape:
        raise ValueError("start and end must have matching shapes")
    if time.ndim != 1 or time.shape[0] != start.shape[0]:
        raise ValueError("time must have shape batch")
    mix = time.to(device=start.device, dtype=start.dtype)[:, None, None]
    return (1.0 - mix) * start + mix * end


def flow_matching_loss(
    model: nn.Module,
    start: Float[torch.Tensor, "batch 1 n"],
    end: Float[torch.Tensor, "batch 1 n"],
    time: Float[torch.Tensor, "batch"],
) -> tuple[torch.Tensor, FlowMatchingMetrics]:
    values = interpolate_linear(start, end, time)
    target_velocity = end - start
    predicted_velocity = model(values, time)
    loss = torch_functional.mse_loss(predicted_velocity, target_velocity)
    return loss, flow_metrics(
        loss=loss,
        predicted_velocity=predicted_velocity,
        target_velocity=target_velocity,
        start=start,
        end=end,
    )


def flow_metrics(
    *,
    loss: torch.Tensor,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> FlowMatchingMetrics:
    flat_predicted = predicted_velocity.flatten(1)
    flat_target = target_velocity.flatten(1)
    target_norm = flat_target.norm(dim=1).clamp_min(torch.finfo(flat_target.dtype).eps)
    relative_l2 = (flat_predicted - flat_target).norm(dim=1) / target_norm
    cosine = torch_functional.cosine_similarity(flat_predicted, flat_target, dim=1)
    return FlowMatchingMetrics(
        loss=float(loss.detach().cpu()),
        velocity_mse=float(loss.detach().cpu()),
        velocity_cosine=float(cosine.mean().detach().cpu()),
        velocity_rel_l2_percent=float((relative_l2.mean() * 100.0).detach().cpu()),
        image_rms=float(start.square().mean().sqrt().detach().cpu()),
        target_rms=float(end.square().mean().sqrt().detach().cpu()),
        image_mean=float(start.mean().detach().cpu()),
        target_mean=float(end.mean().detach().cpu()),
    )


def integrate_euler_midpoint_time(
    vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    start: Float[torch.Tensor, "batch 1 n"],
    steps: int,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
) -> Float[torch.Tensor, "batch 1 n"]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    values = start
    delta = (t1 - t0) / steps
    for step in range(steps):
        midpoint = t0 + (step + 0.5) * delta
        time = torch.full(
            (values.shape[0],), midpoint, device=values.device, dtype=values.dtype
        )
        values = values + delta * vector_field(values, time)
    return values
