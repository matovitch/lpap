from __future__ import annotations

import unittest

import torch

from lpap.decoder import (
    LPAPDecoderTransformer,
    decoder_dibs_from_source_logits,
    lpap_decoder_loss,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_bucket_values,
    reconstruct_lpap_decoder_values,
    train_lpap_decoder_step,
)
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer, prepare_lpap_surrogate_batch


class LPAPDecoderTest(unittest.TestCase):
    def test_prepare_batch_uses_surrogate_probabilities(self) -> None:
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        surrogate_logits = torch.tensor([[[0.0, 0.0, 0.0, 4.0], [0.0, 4.0, 0.0, 0.0]]])

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

    def test_dibs_from_source_logits_uses_bucket_modulo(self) -> None:
        logits = torch.zeros(1, 3, 6)
        logits[0, 0, 5] = 1.0
        logits[0, 1, 1] = 1.0
        logits[0, 2, 3] = 1.0

        dibs = decoder_dibs_from_source_logits(logits, bucket_count=3)

        torch.testing.assert_close(dibs, torch.tensor([[1, 0, 2]]))

    def test_model_loss_and_training_step_report_kpis(self) -> None:
        values = torch.randn(3, 16, generator=torch.Generator().manual_seed(11))
        permutation = make_grouped_permutation_indices(
            value_count=16, bucket_count=4, seed=5
        )
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
