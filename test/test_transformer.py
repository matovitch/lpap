from __future__ import annotations

import unittest

import torch

from lpap.transformer import RotarySelfAttention, TransformerBlock, apply_rope


class TransformerTest(unittest.TestCase):
    def test_apply_rope_preserves_shape_and_norm(self) -> None:
        values = torch.randn(2, 3, 5, 6, generator=torch.Generator().manual_seed(2))

        rotated = apply_rope(values)

        self.assertEqual(rotated.shape, values.shape)
        torch.testing.assert_close(rotated.norm(dim=-1), values.norm(dim=-1))

    def test_rotary_attention_respects_mask_shape(self) -> None:
        attention = RotarySelfAttention(hidden_dim=8, head_count=2)
        tokens = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(3))
        mask = torch.eye(4, dtype=torch.bool)

        output = attention(tokens, attention_mask=mask)

        self.assertEqual(output.shape, tokens.shape)

    def test_transformer_block_preserves_shape(self) -> None:
        block = TransformerBlock(hidden_dim=8, head_count=2)
        tokens = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(4))

        output = block(tokens)

        self.assertEqual(output.shape, tokens.shape)

    def test_circular_rope_preserves_norm_and_shift_invariant_scores(self) -> None:
        query = torch.randn(
            2, 3, 16, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(11)
        )
        key = torch.randn(
            2, 3, 16, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(12)
        )

        rotated_q = apply_rope(query, circular=True)
        rotated_k = apply_rope(key, circular=True)
        torch.testing.assert_close(rotated_q.norm(dim=-1), query.norm(dim=-1))

        base_scores = rotated_q @ rotated_k.transpose(-2, -1)
        for shift in (1, 5, -3):
            shifted_q = apply_rope(
                torch.roll(query, shifts=shift, dims=-2), circular=True
            )
            shifted_k = apply_rope(
                torch.roll(key, shifts=shift, dims=-2), circular=True
            )
            shifted_scores = shifted_q @ shifted_k.transpose(-2, -1)
            expected = torch.roll(
                torch.roll(base_scores, shifts=shift, dims=-2), shifts=shift, dims=-1
            )
            torch.testing.assert_close(shifted_scores, expected, atol=1e-12, rtol=1e-12)

    def test_circular_transformer_block_is_shift_equivariant(self) -> None:
        torch.manual_seed(19)
        block = TransformerBlock(hidden_dim=16, head_count=4, circular=True)
        block.eval()
        tokens = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(20))
        q_pos = torch.arange(16)[:, None]
        k_pos = torch.arange(16)[None, :]
        mask = ((q_pos - k_pos) % 16) < 4

        base_out = block(tokens, attention_mask=mask)
        for shift in (1, 3, 7, -5):
            rolled_out = block(
                torch.roll(tokens, shifts=shift, dims=1), attention_mask=mask
            )
            expected = torch.roll(base_out, shifts=shift, dims=1)
            torch.testing.assert_close(rolled_out, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
