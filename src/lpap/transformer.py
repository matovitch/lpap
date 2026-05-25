from __future__ import annotations

import torch
from torch import nn


def apply_rope(tensor: torch.Tensor) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    rotary_dim = head_dim - (head_dim % 2)
    if rotary_dim == 0:
        return tensor

    positions = torch.arange(tensor.shape[-2], device=tensor.device, dtype=tensor.dtype)
    frequencies = torch.arange(
        0, rotary_dim, 2, device=tensor.device, dtype=tensor.dtype
    )
    inv_frequencies = 1.0 / (10000 ** (frequencies / rotary_dim))
    angles = positions[:, None] * inv_frequencies[None, :]
    cosines = angles.cos()[None, None, :, :]
    sines = angles.sin()[None, None, :, :]

    rotary = tensor[..., :rotary_dim]
    pass_through = tensor[..., rotary_dim:]
    even = rotary[..., 0::2]
    odd = rotary[..., 1::2]
    rotated = torch.stack(
        (even * cosines - odd * sines, even * sines + odd * cosines), dim=-1
    )
    rotated = rotated.flatten(-2)
    return torch.cat((rotated, pass_through), dim=-1)


class RotarySelfAttention(nn.Module):
    def __init__(self, *, hidden_dim: int, head_count: int) -> None:
        super().__init__()
        if hidden_dim % head_count != 0:
            raise ValueError("hidden_dim must be divisible by head_count")
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, tokens: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_count, token_count, _hidden_dim = tokens.shape
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = query.reshape(
            batch_count, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch_count, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch_count, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        query = apply_rope(query)
        key = apply_rope(key)

        scores = query @ key.transpose(-2, -1) / (self.head_dim**0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask[None, None, :, :].to(device=tokens.device),
                torch.finfo(scores.dtype).min,
            )
        attention = torch.softmax(scores, dim=-1)
        attended = attention @ value
        attended = attended.transpose(1, 2).reshape(
            batch_count, token_count, self.hidden_dim
        )
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        head_count: int,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = RotarySelfAttention(
            hidden_dim=hidden_dim, head_count=head_count
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_multiplier, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = tokens + self.dropout(
            self.attention(self.attention_norm(tokens), attention_mask=attention_mask)
        )
        tokens = tokens + self.dropout(self.mlp(self.mlp_norm(tokens)))
        return tokens
