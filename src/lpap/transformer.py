from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional
from torch import nn


def _rope_cos_sin(
    *,
    token_count: int,
    rotary_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    circular: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.promote_types(dtype, torch.float32)
    positions = torch.arange(token_count, device=device, dtype=compute_dtype)
    if circular:
        harmonics = torch.arange(
            1, (rotary_dim // 2) + 1, device=device, dtype=compute_dtype
        )
        inv_frequencies = (2.0 * math.pi / token_count) * harmonics
    else:
        frequencies = torch.arange(
            0, rotary_dim, 2, device=device, dtype=compute_dtype
        )
        inv_frequencies = 1.0 / (10000 ** (frequencies / rotary_dim))
    angles = positions[:, None] * inv_frequencies[None, :]
    cosines = angles.cos().to(dtype=dtype)[None, None, :, :]
    sines = angles.sin().to(dtype=dtype)[None, None, :, :]
    return cosines, sines


def apply_rope(
    tensor: torch.Tensor,
    *,
    cosines: torch.Tensor | None = None,
    sines: torch.Tensor | None = None,
    circular: bool = False,
) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    rotary_dim = head_dim - (head_dim % 2)
    if rotary_dim == 0:
        return tensor

    if cosines is None or sines is None:
        cosines, sines = _rope_cos_sin(
            token_count=tensor.shape[-2],
            rotary_dim=rotary_dim,
            device=tensor.device,
            dtype=tensor.dtype,
            circular=circular,
        )

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
    def __init__(
        self, *, hidden_dim: int, head_count: int, circular: bool = False
    ) -> None:
        super().__init__()
        if hidden_dim % head_count != 0:
            raise ValueError("hidden_dim must be divisible by head_count")
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        self.circular = circular
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("_rope_cosines", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sines", torch.empty(0), persistent=False)

    def _rope_tables(
        self, *, token_count: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rotary_dim = self.head_dim - (self.head_dim % 2)
        if rotary_dim == 0:
            return None, None
        table_shape = (1, 1, token_count, rotary_dim // 2)
        if (
            self._rope_cosines.shape != table_shape
            or self._rope_cosines.device != device
            or self._rope_cosines.dtype != dtype
        ):
            self._rope_cosines, self._rope_sines = _rope_cos_sin(
                token_count=token_count,
                rotary_dim=rotary_dim,
                device=device,
                dtype=dtype,
                circular=self.circular,
            )
        return self._rope_cosines, self._rope_sines

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
        cosines, sines = self._rope_tables(
            token_count=token_count, dtype=tokens.dtype, device=tokens.device
        )
        query = apply_rope(query, cosines=cosines, sines=sines)
        key = apply_rope(key, cosines=cosines, sines=sines)

        # Fused attention (flash / memory-efficient kernels): avoids materializing
        # the (batch, heads, tokens, tokens) score tensor. Boolean masks use the
        # same convention as before (True = may attend); every LPAP mask keeps
        # the diagonal, so no row is fully masked.
        if attention_mask is not None:
            if attention_mask.device != tokens.device:
                attention_mask = attention_mask.to(device=tokens.device)
            attention_mask = attention_mask[None, None, :, :]
        attended = torch_functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
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
        circular: bool = False,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = RotarySelfAttention(
            hidden_dim=hidden_dim, head_count=head_count, circular=circular
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
