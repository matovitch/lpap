from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from lpap.decoder import (
    _candidate_source_indices,
    LPAPDecoderTransformer,
    decoder_dibs_from_local_logits,
    decoder_predicted_source_indices,
    evaluate_lpap_decoder_exact_batch,
    lpap_decoder_loss,
    prepare_exact_lpap_decoder_batch,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_bucket_values,
    reconstruct_lpap_decoder_values,
    train_lpap_decoder_exact_step,
    train_lpap_decoder_step,
)
from lpap.permutation import make_permutation_indices
from lpap.surrogate import (
    LPAPSurrogateTransformer,
    lpap_surrogate_targets,
    prepare_lpap_surrogate_batch,
)


class LPAPDecoderTest(unittest.TestCase):
    def test_prepare_batch_uses_categorical_st_and_matches_exact_in_fp32_and_bf16(
        self,
    ) -> None:
        perm = make_permutation_indices(value_count=16, seed=17)
        for dtype in (torch.float32, torch.bfloat16):
            values = torch.randn(
                3, 16, generator=torch.Generator().manual_seed(19)
            ).to(dtype=dtype)
            tokens = prepare_lpap_surrogate_batch(
                values, bucket_count=4, permutation=perm
            )
            targets = lpap_surrogate_targets(tokens, k_max=3, permutation=perm)
            # Imperfect (non-infinite) oracle logits so softmax p is strictly < 1,
            # exercising BF16 bit-exactness of one_hot + (p - p.detach()).
            surrogate_logits = torch.zeros(3, 4, 3 * 4, dtype=dtype)
            surrogate_logits.scatter_(2, targets.local_classes.unsqueeze(-1), 2.5)

            st_batch = prepare_lpap_decoder_batch(
                values=values,
                surrogate_logits=surrogate_logits,
                bucket_count=4,
                k_max=3,
                permutation=perm,
                surrogate_tokens=tokens,
            )
            exact_batch = prepare_exact_lpap_decoder_batch(
                values=values,
                bucket_count=4,
                k_max=3,
                permutation=perm,
            )

            self.assertTrue(torch.equal(st_batch.amplitudes, exact_batch.amplitudes))
            self.assertTrue(torch.equal(st_batch.dib_weights, exact_batch.dib_weights))
            self.assertTrue(torch.equal(st_batch.dibs, exact_batch.dibs))
            self.assertTrue(torch.all(st_batch.weights >= 0))
            self.assertTrue(torch.all(st_batch.entropy >= 0))
            self.assertFalse(st_batch.entropy.requires_grad)

    def test_st_frontend_has_zero_smear_on_values_grad_and_live_surrogate_grad(
        self,
    ) -> None:
        # N=8, C=4, P=2, k_max=2 -> 4 local classes per bucket.
        values = torch.randn(
            1, 8, generator=torch.Generator().manual_seed(7), requires_grad=True
        )
        surrogate_logits = torch.randn(
            1, 4, 4, generator=torch.Generator().manual_seed(11), requires_grad=True
        )
        perm = make_permutation_indices(value_count=8, seed=3)

        batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=4,
            k_max=2,
            permutation=perm,
        )
        # Only bucket 0 contributes to the loss; gradient wrt values must be 1.0
        # at the single hard-selected source index and exact 0.0 everywhere else.
        batch.amplitudes[0, 0].backward()

        assert values.grad is not None
        assert surrogate_logits.grad is not None
        hard_class = int(surrogate_logits[0, 0].argmax().item())
        dib = hard_class // 2
        probe = hard_class % 2
        source_bucket = (0 - dib) % 4
        expected_source = int(perm[probe * 4 + source_bucket].item())
        expected_values_grad = torch.zeros_like(values)
        expected_values_grad[0, expected_source] = 1.0
        torch.testing.assert_close(values.grad, expected_values_grad, atol=0.0, rtol=0.0)
        self.assertGreater(float(surrogate_logits.grad[0, 0].abs().sum()), 0.0)

    def test_ce_weights_are_stop_gradient(self) -> None:
        values = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
        surrogate_logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)

        batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=2,
            k_max=1,
        )

        self.assertFalse(batch.weights.requires_grad)
        self.assertTrue(batch.amplitudes.requires_grad)
        torch.testing.assert_close(batch.weights, batch.amplitudes.detach().abs())

        # Cut amplitudes and dib_weights; CE weights are then the only possible route.
        ce_only = replace(
            batch,
            amplitudes=batch.amplitudes.detach(),
            dib_weights=batch.dib_weights.detach(),
        )
        probe_logits = torch.randn(1, 2, 2, requires_grad=True)
        loss, _metrics = lpap_decoder_loss(
            probe_logits,
            ce_only,
            source_ce_weight=1.0,
            source_ce_l1_reference=1.0,
            source_ce_power=2.0,
        )
        loss.backward()
        self.assertIsNone(surrogate_logits.grad)
        self.assertIsNotNone(probe_logits.grad)

    def test_dibs_from_local_logits_divides_by_probe_count(self) -> None:
        logits = torch.zeros(1, 3, 6)
        logits[0, 0, 3] = 1.0  # dib=1, probe=1
        logits[0, 1, 1] = 1.0  # dib=0, probe=1
        logits[0, 2, 4] = 1.0  # dib=2, probe=0

        dibs = decoder_dibs_from_local_logits(
            logits, bucket_count=3, probe_count=2, k_max=3
        )

        torch.testing.assert_close(dibs, torch.tensor([[1, 0, 2]]))

    def test_reconstruct_restricts_support_to_collision_set_and_propagates_dib_grads(
        self,
    ) -> None:
        # N=16, C=4, P=4, k_max=3.
        values = torch.randn(1, 16, generator=torch.Generator().manual_seed(31))
        perm = make_permutation_indices(value_count=16, seed=13)
        surrogate_logits = torch.randn(
            1, 4, 12, generator=torch.Generator().manual_seed(37), requires_grad=True
        )
        batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=4,
            k_max=3,
            permutation=perm,
        )
        # Zero out all buckets except bucket 1 to inspect bucket 1's exact support.
        masked_amps = torch.zeros_like(batch.amplitudes)
        masked_amps[0, 1] = 3.5
        single_bucket_batch = replace(batch, amplitudes=masked_amps)
        probe_logits = torch.randn(
            1, 4, 4, generator=torch.Generator().manual_seed(41), requires_grad=True
        )

        recon = reconstruct_lpap_decoder_values(probe_logits, single_bucket_batch)

        dib_b1 = int(batch.dibs[0, 1].item())
        source_bucket_b1 = (1 - dib_b1) % 4
        allowed_indices = {
            int(perm[p * 4 + source_bucket_b1].item()) for p in range(4)
        }
        for idx in range(16):
            if idx not in allowed_indices:
                self.assertEqual(float(recon[0, idx].item()), 0.0)
        self.assertAlmostEqual(
            float(recon[0, list(allowed_indices)].sum().item()), 3.5, places=5
        )

        # Verify gradients flow through dib_weights into surrogate_logits.
        (recon * values).sum().backward()
        assert probe_logits.grad is not None
        assert surrogate_logits.grad is not None
        self.assertGreater(float(probe_logits.grad[0, 1].abs().sum()), 0.0)
        self.assertGreater(float(surrogate_logits.grad[0, 1].abs().sum()), 0.0)

    def test_oracle_probe_logits_match_lpap_bucket_reconstruction_and_targets(
        self,
    ) -> None:
        values = torch.randn(2, 16, generator=torch.Generator().manual_seed(51))
        perm = make_permutation_indices(value_count=16, seed=53)
        batch = prepare_exact_lpap_decoder_batch(
            values=values, bucket_count=4, k_max=3, permutation=perm
        )
        oracle_probe_logits = torch.full((2, 4, 4), -100.0)
        oracle_probe_logits.scatter_(2, batch.target_probes.unsqueeze(-1), 100.0)

        recon = reconstruct_lpap_decoder_values(oracle_probe_logits, batch)
        oracle_recon = reconstruct_lpap_bucket_values(batch)
        torch.testing.assert_close(recon, oracle_recon, atol=1.0e-6, rtol=1.0e-6)
        torch.testing.assert_close(
            decoder_predicted_source_indices(oracle_probe_logits, batch),
            batch.targets,
        )

    def test_decoder_loss_masks_source_ce_to_matching_dibs(self) -> None:
        values = torch.randn(1, 16, generator=torch.Generator().manual_seed(61))
        perm = make_permutation_indices(value_count=16, seed=67)
        exact = prepare_exact_lpap_decoder_batch(
            values=values, bucket_count=4, k_max=3, permutation=perm
        )
        # Force all DIBs to mismatch oracle DIBs.
        wrong_dibs = (exact.dibs + 1) % 3
        mismatched = replace(
            exact,
            dibs=wrong_dibs,
            dib_weights=torch.nn.functional.one_hot(wrong_dibs, num_classes=3).to(
                dtype=values.dtype
            ),
        )
        oracle_probe_logits = torch.full((1, 4, 4), -20.0, requires_grad=True)
        with torch.no_grad():
            oracle_probe_logits.scatter_(2, exact.target_probes.unsqueeze(-1), 20.0)

        loss, metrics = lpap_decoder_loss(
            oracle_probe_logits,
            mismatched,
            source_ce_weight=1.0,
            source_ce_l1_reference=0.1,
            source_ce_power=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics.source_ce, 0.0)
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.weighted_accuracy, 0.0)

    def test_exact_lpap_decoder_batch_and_step(self) -> None:
        values = torch.randn(3, 16, generator=torch.Generator().manual_seed(21))
        permutation = make_permutation_indices(value_count=16, seed=9)
        decoder = LPAPDecoderTransformer(
            bucket_count=4,
            probe_count=4,
            k_max=3,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        )
        self.assertFalse(hasattr(decoder, "log_frontend_temperature"))
        self.assertTrue(all(block.attention.circular for block in decoder.blocks))

        batch = prepare_exact_lpap_decoder_batch(
            values=values, bucket_count=4, k_max=3, permutation=permutation
        )
        self.assertEqual(batch.amplitudes.shape, (3, 4))
        self.assertEqual(batch.dib_weights.shape, (3, 4, 3))
        torch.testing.assert_close(batch.entropy, torch.zeros_like(batch.entropy))
        torch.testing.assert_close(batch.dibs, batch.surrogate_targets.dibs)

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
            bucket_count=4,
            probe_count=4,
            k_max=2,
            hidden_dim=16,
            layer_count=1,
            head_count=4,
        )
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
            permutation=permutation,
        )

        probe_logits = decoder(batch.amplitudes, batch.dib_weights)
        self.assertEqual(probe_logits.shape, (3, 4, 4))
        loss, metrics = lpap_decoder_loss(
            probe_logits,
            batch,
            source_ce_weight=0.5,
            source_ce_l1_reference=0.1,
            source_ce_power=2.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics.reconstruction_l1, 0.0)
        self.assertGreaterEqual(metrics.source_ce, 0.0)
        self.assertGreaterEqual(metrics.source_ce_regularizer, 0.0)
        self.assertGreaterEqual(metrics.source_ce_weight, 0.0)
        self.assertLessEqual(metrics.source_ce_weight, 0.5)
        reconstruction = reconstruct_lpap_decoder_values(probe_logits, batch)
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

    def test_candidate_source_indices_are_cached_for_the_same_permutation(self) -> None:
        permutation = make_permutation_indices(value_count=16, seed=5)
        first = _candidate_source_indices(
            bucket_count=4,
            probe_count=4,
            k_max=2,
            permutation=permutation,
            device=permutation.device,
        )
        second = _candidate_source_indices(
            bucket_count=4,
            probe_count=4,
            k_max=2,
            permutation=permutation,
            device=permutation.device,
        )
        self.assertEqual(first.data_ptr(), second.data_ptr())


if __name__ == "__main__":
    unittest.main()
