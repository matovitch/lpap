from __future__ import annotations

import unittest

import torch

from lpap.decoder import (
    LPAPDecoderTransformer,
    decoder_dibs_from_local_logits,
    evaluate_lpap_decoder_exact_batch,
    lpap_decoder_loss,
    normalize_lpap_dibs,
    prepare_exact_lpap_decoder_batch,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_bucket_values,
    reconstruct_lpap_decoder_values,
    train_lpap_decoder_exact_step,
    train_lpap_decoder_step,
)
from lpap.permutation import make_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer, prepare_lpap_surrogate_batch


class LPAPDecoderTest(unittest.TestCase):
    def test_prepare_batch_uses_local_surrogate_probabilities(self) -> None:
        # N=4, C=2, P=2, k_max=1 -> k_max * P = 2 local classes per bucket.
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        surrogate_logits = torch.tensor([[[0.0, 4.0], [4.0, 0.0]]])

        batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=2,
            k_max=1,
            temperature=0.5,
        )

        self.assertEqual(batch.tokens.shape, (1, 2, 3))
        self.assertEqual(batch.targets.shape, (1, 2))
        self.assertTrue(torch.all(batch.weights >= 0))
        self.assertTrue(torch.all(batch.entropy >= 0))

    def test_dibs_from_local_logits_divides_by_probe_count(self) -> None:
        # C=3, P=2, k_max=3 -> 6 local classes = dib * 2 + probe.
        logits = torch.zeros(1, 3, 6)
        logits[0, 0, 3] = 1.0  # dib=1, probe=1
        logits[0, 1, 1] = 1.0  # dib=0, probe=1
        logits[0, 2, 4] = 1.0  # dib=2, probe=0

        dibs = decoder_dibs_from_local_logits(
            logits, bucket_count=3, probe_count=2, k_max=3
        )

        torch.testing.assert_close(dibs, torch.tensor([[1, 0, 2]]))

    def test_normalize_lpap_dibs_is_consistent(self) -> None:
        dibs = torch.tensor([[0, 1, 2, 3]])
        torch.testing.assert_close(
            normalize_lpap_dibs(dibs, k_max=4, dtype=torch.float32),
            torch.tensor([[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]]),
        )
        torch.testing.assert_close(
            normalize_lpap_dibs(torch.tensor([[0, 0]]), k_max=1, dtype=torch.float32),
            torch.tensor([[0.0, 0.0]]),
        )

    def test_exact_lpap_decoder_batch_and_step(self) -> None:
        values = torch.randn(3, 16, generator=torch.Generator().manual_seed(21))
        permutation = make_permutation_indices(value_count=16, seed=9)
        decoder = LPAPDecoderTransformer(
            value_count=16,
            frontend_initial_temperature=0.5,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        )
        batch = prepare_exact_lpap_decoder_batch(
            values=values, bucket_count=4, k_max=3, permutation=permutation
        )
        self.assertEqual(batch.tokens.shape, (3, 4, 3))
        torch.testing.assert_close(batch.entropy, torch.zeros_like(batch.entropy))
        torch.testing.assert_close(
            batch.tokens[..., 1],
            normalize_lpap_dibs(batch.surrogate_targets.dibs, k_max=3, dtype=values.dtype),
        )

        optimizer = torch.optim.AdamW(decoder.parameters(), lr=1.0e-3)
        step_metrics = train_lpap_decoder_exact_step(
            decoder=decoder,
            optimizer=optimizer,
            values=values,
            bucket_count=4,
            k_max=3,
            permutation=permutation,
        )
        eval_metrics = evaluate_lpap_decoder_exact_batch(
            decoder=decoder,
            values=values,
            bucket_count=4,
            k_max=3,
            permutation=permutation,
        )
        self.assertEqual(step_metrics.mean_entropy, 0.0)
        self.assertEqual(eval_metrics.mean_entropy, 0.0)
        self.assertGreaterEqual(eval_metrics.reconstruction_l1, 0.0)

    def test_model_loss_and_training_step_report_kpis(self) -> None:
        values = torch.randn(3, 16, generator=torch.Generator().manual_seed(11))
        permutation = make_permutation_indices(value_count=16, seed=5)
        surrogate = LPAPSurrogateTransformer(
            value_count=16,
            probe_count=4,
            k_max=2,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        )
        decoder = LPAPDecoderTransformer(
            value_count=16,
            frontend_initial_temperature=0.5,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        )
        self.assertTrue(decoder.frontend_temperature().requires_grad)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=1.0e-3)
        tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=4, permutation=permutation
        )
        surrogate_logits = surrogate(tokens)
        batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=4,
            k_max=2,
            temperature=0.5,
            permutation=permutation,
        )

        logits = decoder(batch.tokens)
        self.assertEqual(logits.shape, (3, 4, 16))
        loss, metrics = lpap_decoder_loss(
            logits,
            batch,
            source_ce_weight=0.5,
            source_ce_l1_reference=0.1,
            source_ce_power=2.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics.reconstruction_l1, 0.0)
        self.assertGreater(metrics.source_ce, 0.0)
        self.assertGreaterEqual(metrics.source_ce_regularizer, 0.0)
        self.assertGreaterEqual(metrics.source_ce_weight, 0.0)
        self.assertLessEqual(metrics.source_ce_weight, 0.5)
        reconstruction = reconstruct_lpap_decoder_values(logits, batch)
        self.assertEqual(reconstruction.shape, values.shape)
        lpap_reconstruction = reconstruct_lpap_bucket_values(batch)
        self.assertEqual(lpap_reconstruction.shape, values.shape)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)

        step_metrics = train_lpap_decoder_step(
            decoder=decoder,
            surrogate=surrogate,
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
