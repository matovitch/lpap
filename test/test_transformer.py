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


if __name__ == "__main__":
    unittest.main()
