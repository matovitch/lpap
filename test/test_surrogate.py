from __future__ import annotations

import unittest

import torch

from lpap.ops import lpap_torch
from lpap.permutation import (
    make_permutation_indices,
    unfold_permutation_tokens,
)
from lpap.surrogate import (
    LPAPSurrogateTransformer,
    circular_previous_attention_mask,
    gather_visible_candidates,
    local_classes_to_source_indices,
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
            unfold_permutation_tokens(tokens), bucket_count=3, k_max=2
        )

        torch.testing.assert_close(targets.buckets, expected_buckets)
        torch.testing.assert_close(targets.dibs, expected_dibs)
        torch.testing.assert_close(targets.source_indices, torch.tensor([[3, 4, 5]]))
        torch.testing.assert_close(targets.local_classes, torch.tensor([[1, 1, 1]]))
        torch.testing.assert_close(targets.weights, expected_buckets.abs())

    def test_local_classes_round_trip_to_source_indices_with_permutation(self) -> None:
        values = torch.randn(3, 16, generator=torch.Generator().manual_seed(9))
        permutation = make_permutation_indices(value_count=16, seed=17)
        tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=4, permutation=permutation
        )
        targets = lpap_surrogate_targets(tokens, k_max=3, permutation=permutation)

        recovered = local_classes_to_source_indices(
            targets.local_classes,
            bucket_count=4,
            probe_count=4,
            permutation=permutation,
        )
        torch.testing.assert_close(recovered, targets.source_indices)

        visible = gather_visible_candidates(tokens, k_max=3).reshape(3, 4, 12)
        gathered_buckets = visible.gather(
            dim=-1, index=targets.local_classes.unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(gathered_buckets, targets.buckets)
        torch.testing.assert_close(
            values.gather(dim=1, index=targets.source_indices), targets.buckets
        )

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
        permutation = make_permutation_indices(value_count=16, seed=5)
        tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=4, permutation=permutation
        )
        targets = lpap_surrogate_targets(tokens, k_max=2, permutation=permutation)
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
        # Local classes = k_max * probe_count = 2 * 4 = 8.
        self.assertEqual(logits.shape, (4, 4, 8))
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

    def test_model_rejects_k_max_exceeding_bucket_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most bucket_count"):
            LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=5,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )

    def test_surrogate_is_circular_shift_equivariant(self) -> None:
        torch.manual_seed(23)
        model = LPAPSurrogateTransformer(
            value_count=1024,
            probe_count=8,
            k_max=16,
            hidden_dim=32,
            layer_count=2,
            head_count=4,
        )
        model.eval()
        tokens = torch.randn(2, 128, 8, generator=torch.Generator().manual_seed(24))

        base_logits = model(tokens)
        for shift in (1, 15, 64, -17):
            rolled_logits = model(torch.roll(tokens, shifts=shift, dims=1))
            expected = torch.roll(base_logits, shifts=shift, dims=1)
            torch.testing.assert_close(rolled_logits, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
