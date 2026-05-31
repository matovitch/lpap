from __future__ import annotations

import unittest

import torch

from lpap.ops import lpap_torch
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import (
    LPAPSurrogateTransformer,
    circular_previous_attention_mask,
    lpap_surrogate_loss,
    lpap_surrogate_targets,
    prepare_lpap_surrogate_batch,
    train_lpap_surrogate_step,
)


class LPAPSurrogateTest(unittest.TestCase):
    def test_targets_match_lpap_bucket_values(self) -> None:
        tokens = torch.tensor([[[1.0, 9.0], [3.0, 4.0], [2.0, 8.0]]])

        targets = lpap_surrogate_targets(tokens, k_max=2)
        expected_buckets, expected_dibs, _remaining = lpap_torch(
            tokens.reshape(1, 6), bucket_count=3, k_max=2
        )

        torch.testing.assert_close(targets.buckets, expected_buckets)
        torch.testing.assert_close(targets.dibs, expected_dibs)
        torch.testing.assert_close(targets.source_indices, torch.tensor([[3, 4, 5]]))
        torch.testing.assert_close(targets.weights, expected_buckets.abs())

    def test_circular_previous_attention_mask_matches_roll_window(self) -> None:
        mask = circular_previous_attention_mask(bucket_count=5, k_max=3)
        expected = torch.tensor(
            [
                [True, False, False, True, True],
                [True, True, False, False, True],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
            ]
        )

        torch.testing.assert_close(mask, expected)

    def test_model_loss_and_training_step_report_kpis(self) -> None:
        values = torch.randn(4, 16, generator=torch.Generator().manual_seed(7))
        permutation = make_grouped_permutation_indices(
            value_count=16, bucket_count=4, seed=5
        )
        tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=4, permutation=permutation
        )
        targets = lpap_surrogate_targets(tokens, k_max=2)
        model = LPAPSurrogateTransformer(
            value_count=16,
            probe_count=4,
            k_max=2,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

        logits = model(tokens)
        self.assertEqual(logits.shape, (4, 4, 16))
        loss, metrics = lpap_surrogate_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)

        step_metrics = train_lpap_surrogate_step(
            model=model,
            optimizer=optimizer,
            values=values,
            bucket_count=4,
            k_max=2,
            permutation=permutation,
        )
        self.assertGreaterEqual(step_metrics.weighted_accuracy, 0.0)
        self.assertLessEqual(step_metrics.weighted_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
