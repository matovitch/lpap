"""LPAP operator correctness and edge-case invariant tests."""

from __future__ import annotations

import unittest

import torch

from lpap.decoder import (
    decoder_dibs_from_local_logits,
    prepare_lpap_decoder_batch,
)
from lpap.ops import lpap_torch
from lpap.permutation import (
    as_long_permutation,
    invert_permutation_indices,
    make_permutation_indices,
)
from lpap.surrogate import (
    lpap_surrogate_targets,
    prepare_lpap_surrogate_batch,
)
from lpap.triton_ops import lpap_triton


class LpapInvariantsTest(unittest.TestCase):
    """Core LPAP operator, permutation, surrogate, and decoder invariants."""

    def test_oracle_decoder_dibs_match_lpap_targets_with_permutation(self) -> None:
        """Oracle local surrogate logits yield decoder DIBs equal to LPAP DIBs."""
        perm = make_permutation_indices(value_count=16, seed=123)
        values = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
        tokens = prepare_lpap_surrogate_batch(values, bucket_count=4, permutation=perm)
        targets = lpap_surrogate_targets(tokens, k_max=2, permutation=perm)

        # Local logits have k_max * probe_count = 2 * 4 = 8 classes per bucket.
        oracle_logits = torch.full((4, 4, 8), -100.0)
        oracle_logits.scatter_(2, targets.local_classes.unsqueeze(-1), 100.0)
        decoder_batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=oracle_logits,
            bucket_count=4,
            k_max=2,
            permutation=perm,
        )

        torch.testing.assert_close(decoder_batch.dibs, targets.dibs)
        torch.testing.assert_close(decoder_batch.amplitudes, targets.buckets)

    def test_lpap_torch_matches_surrogate_targets_on_same_flat_input(self) -> None:
        """lpap_torch and prepare_lpap_surrogate_batch+targets share column-major fold."""
        values = torch.arange(1, 17, dtype=torch.float32).unsqueeze(0)

        op_buckets, op_dibs, _remaining = lpap_torch(values, bucket_count=4, k_max=2)
        tokens = prepare_lpap_surrogate_batch(values, bucket_count=4, permutation=None)
        targets = lpap_surrogate_targets(tokens, k_max=2, permutation=None)

        torch.testing.assert_close(op_buckets, targets.buckets)
        torch.testing.assert_close(op_dibs, targets.dibs)

    def test_sparse_input_source_indices_map_back_to_bucket_values(self) -> None:
        """Every target source_index gathers the exact bucket value without aliasing."""
        values = torch.zeros(1, 16)
        values[0, 0] = 10.0
        tokens = prepare_lpap_surrogate_batch(values, bucket_count=4, permutation=None)
        targets = lpap_surrogate_targets(tokens, k_max=4, permutation=None)

        gathered = values.gather(1, targets.source_indices)
        torch.testing.assert_close(gathered, targets.buckets)
        self.assertEqual(targets.source_indices[0].unique().numel(), 4)

    def test_c_equals_n_zero_element_keeps_zero_dib(self) -> None:
        """When C == N (probe_count == 1), zero elements experience no collisions."""
        values = torch.tensor([[1.0, -2.0, 0.0, 4.0]])

        _buckets, dibs, _remaining = lpap_torch(values, bucket_count=4, k_max=3)

        torch.testing.assert_close(dibs, torch.zeros_like(dibs))

    def test_c_equals_1_rejects_k_max_gt_1_and_keeps_zero_dib(self) -> None:
        """A single-bucket table (C == 1) allows only k_max == 1 and reports DIB == 0."""
        values = torch.tensor([[5.0, -5.0, 3.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "at most bucket_count"):
            lpap_torch(values, bucket_count=1, k_max=3)

        buckets, dibs, _remaining = lpap_torch(values, bucket_count=1, k_max=1)
        torch.testing.assert_close(buckets, torch.tensor([[5.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0]]))

    def test_k_max_gt_bucket_count_is_rejected(self) -> None:
        """k_max > bucket_count is rejected so DIBs stay strictly below bucket_count."""
        values = torch.tensor([[10.0, -10.0, 10.0, 1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "at most bucket_count"):
            lpap_torch(values, bucket_count=2, k_max=5)

        _buckets, dibs, _remaining = lpap_torch(values, bucket_count=2, k_max=2)
        self.assertTrue(bool((dibs < 2).all()))

    def test_all_zero_input_preserves_nonzero_autograd_grad_for_k_gt_1(self) -> None:
        """All-zero inputs with k_max > 1 route gradients cleanly to selected slots."""
        values = torch.zeros(1, 6, requires_grad=True)

        buckets, _dibs, _remaining = lpap_torch(values, bucket_count=3, k_max=2)
        buckets.sum().backward()

        assert values.grad is not None
        self.assertAlmostEqual(float(values.grad.sum()), 3.0)

    def test_empty_value_dimension_raises_value_error(self) -> None:
        """Zero-length last dimension (N == 0) fails validation with ValueError."""
        with self.assertRaises(ValueError):
            lpap_torch(torch.empty(2, 0), bucket_count=4, k_max=1)

    def test_nan_in_lane_does_not_mask_finite_values(self) -> None:
        """A NaN in one probe slot does not hide finite values while staying unpicked."""
        # Column-major (C=2, P=2): lane 0 = [nan, 10.0], lane 1 = [2.0, 3.0].
        values = torch.tensor([[float("nan"), 2.0, 10.0, 3.0]])

        buckets, _dibs, _remaining = lpap_torch(values, bucket_count=2, k_max=1)

        torch.testing.assert_close(buckets, torch.tensor([[10.0, 3.0]]))

    def test_as_long_permutation_rejects_non_bijection(self) -> None:
        """Non-bijective index tensors are rejected as permutations."""
        bad = torch.tensor([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            as_long_permutation(bad, value_count=4)

    def test_invert_permutation_rejects_non_bijection(self) -> None:
        """invert_permutation_indices rejects duplicate indices."""
        bad = torch.tensor([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            invert_permutation_indices(bad)

    def test_decoder_dibs_validates_bucket_count_matches_logits_dim1(self) -> None:
        """Mismatched bucket_count and logits.shape[1] raises ValueError."""
        logits = torch.zeros(1, 2, 4)
        with self.assertRaises(ValueError):
            decoder_dibs_from_local_logits(
                logits, bucket_count=4, probe_count=2, k_max=2
            )


class LpapEdgeCasesTest(unittest.TestCase):
    """Edge-case behavior satisfied by the unified LPAP implementation."""

    def test_k_exhausted_leaves_unplaced_peaks_in_residual(self) -> None:
        # Column-major (C=3, P=4): place 4 peaks in lane 0 (flat indices 0, 3, 6, 9).
        values = torch.tensor(
            [[100.0, 1.0, 1.0, 90.0, 1.0, 1.0, 80.0, 1.0, 1.0, 70.0, 1.0, 1.0]]
        )
        buckets, dibs, remaining = lpap_torch(values, bucket_count=3, k_max=2)

        torch.testing.assert_close(buckets, torch.tensor([[100.0, 90.0, 1.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 1, 1]]))
        self.assertIn(80.0, remaining.tolist()[0])
        self.assertIn(70.0, remaining.tolist()[0])
        self.assertTrue(bool(((dibs >= 0) & (dibs < 2)).all()))

    def test_k_equals_c_recovers_exact_top_c_on_distinct_values(self) -> None:
        generator = torch.Generator().manual_seed(7)
        for bucket_count, probe_count in ((2, 4), (4, 4), (8, 8)):
            values = torch.randn(64, bucket_count * probe_count, generator=generator)
            buckets, dibs, _remaining = lpap_torch(
                values, bucket_count=bucket_count, k_max=bucket_count
            )
            expected_topk = values.abs().topk(bucket_count, dim=-1).values.sort(-1).values
            actual_sorted = buckets.abs().sort(-1).values
            torch.testing.assert_close(actual_sorted, expected_topk)
            self.assertTrue(bool(((dibs >= 0) & (dibs < bucket_count)).all()))

    def test_negative_and_signed_values_preserve_sign_by_abs_amplitude(self) -> None:
        # Column-major (C=3, P=2): lane 0=[-1, -9], lane 1=[-3, -8], lane 2=[-2, -7].
        values = torch.tensor([[-1.0, -3.0, -2.0, -9.0, -8.0, -7.0]])
        buckets, dibs, _remaining = lpap_torch(values, bucket_count=3, k_max=2)

        torch.testing.assert_close(buckets, torch.tensor([[-9.0, -8.0, -7.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0, 0]]))

    def test_arbitrary_leading_batch_dims_and_empty_batch(self) -> None:
        flat = torch.randn(6, 8, generator=torch.Generator().manual_seed(13))
        ref_b, ref_d, ref_r = lpap_torch(flat, bucket_count=4, k_max=2)

        b1, d1, r1 = lpap_torch(flat[0], bucket_count=4, k_max=2)
        self.assertEqual(b1.shape, (4,))
        self.assertEqual(d1.shape, (4,))
        self.assertEqual(r1.shape, (8,))
        torch.testing.assert_close(b1, ref_b[0])
        torch.testing.assert_close(d1, ref_d[0])
        torch.testing.assert_close(r1, ref_r[0])

        b3, d3, r3 = lpap_torch(flat.reshape(2, 3, 8), bucket_count=4, k_max=2)
        self.assertEqual(b3.shape, (2, 3, 4))
        torch.testing.assert_close(b3.reshape(6, 4), ref_b)
        torch.testing.assert_close(d3.reshape(6, 4), ref_d)
        torch.testing.assert_close(r3.reshape(6, 8), ref_r)

        b0, d0, r0 = lpap_torch(torch.empty(0, 8), bucket_count=4, k_max=2)
        self.assertEqual(b0.shape, (0, 4))
        self.assertEqual(d0.shape, (0, 4))
        self.assertEqual(r0.shape, (0, 8))

        with self.assertRaises(ValueError):
            lpap_torch(torch.randn(1, 4), bucket_count=8, k_max=1)

    def test_floating_dtypes_and_non_contiguous_inputs(self) -> None:
        base = torch.randn(3, 8, generator=torch.Generator().manual_seed(17))
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            values = base.to(dtype)
            buckets, dibs, remaining = lpap_torch(values, bucket_count=4, k_max=2)
            self.assertEqual(buckets.dtype, dtype)
            self.assertEqual(dibs.dtype, torch.int64)
            self.assertEqual(remaining.dtype, dtype)

        non_contig = base.t().contiguous().t()
        self.assertFalse(non_contig.is_contiguous())
        b_nc, d_nc, r_nc = lpap_torch(non_contig, bucket_count=4, k_max=2)
        b_ref, d_ref, r_ref = lpap_torch(base, bucket_count=4, k_max=2)
        torch.testing.assert_close(b_nc, b_ref)
        torch.testing.assert_close(d_nc, d_ref)
        torch.testing.assert_close(r_nc, r_ref)

        with self.assertRaises(TypeError):
            lpap_torch(torch.ones(2, 8, dtype=torch.int32), bucket_count=4, k_max=2)

    def test_autograd_routes_gradients_through_multihop_swap_back(self) -> None:
        # Column-major (C=3, P=3):
        # lane 0=[100, 90, 1], lane 1=[80, 70, 1], lane 2=[2, 1, 1].
        # 80.0 starts in lane 1 -> bucket 1 (k=0), evicted to lane 0 (k=1),
        # re-picked into bucket 2 (k=2).
        values = torch.tensor(
            [[100.0, 80.0, 2.0, 90.0, 70.0, 1.0, 1.0, 1.0, 1.0]],
            requires_grad=True,
        )
        with torch.autograd.detect_anomaly():
            buckets, dibs, _remaining = lpap_torch(values, bucket_count=3, k_max=3)
            loss = (buckets * torch.tensor([[1.0, 2.0, 3.0]])).sum()
            loss.backward()

        torch.testing.assert_close(buckets.detach(), torch.tensor([[100.0, 90.0, 80.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 1, 1]]))
        assert values.grad is not None
        torch.testing.assert_close(
            values.grad,
            torch.tensor([[1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )

    def test_decoder_frontend_gradient_flow_components(self) -> None:
        values = torch.randn(2, 16, generator=torch.Generator().manual_seed(23))
        values.requires_grad_(True)
        # Local logits: (B=2, C=4, k_max * P = 2 * 4 = 8).
        logits = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(29))
        logits.requires_grad_(True)
        perm = make_permutation_indices(value_count=16, seed=42)

        batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=logits,
            bucket_count=4,
            k_max=2,
            permutation=perm,
        )
        self.assertFalse(batch.dibs.requires_grad)
        self.assertFalse(batch.entropy.requires_grad)
        self.assertTrue(batch.amplitudes.requires_grad)
        self.assertTrue(batch.dib_weights.requires_grad)

        (batch.amplitudes.sum() + (batch.dib_weights * torch.tensor([1.0, 2.0])).sum()).backward()
        assert values.grad is not None
        assert logits.grad is not None
        self.assertGreater(float(values.grad.abs().sum()), 0.0)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_determinism_under_deterministic_algorithms(self) -> None:
        torch.use_deterministic_algorithms(True)
        try:
            values = torch.randn(4, 16, generator=torch.Generator().manual_seed(31))
            perm_a = make_permutation_indices(value_count=16, seed=99)
            perm_b = make_permutation_indices(value_count=16, seed=99)
            torch.testing.assert_close(perm_a, perm_b)

            out_a = lpap_torch(values, bucket_count=4, k_max=3)
            out_b = lpap_torch(values, bucket_count=4, k_max=3)
            for left, right in zip(out_a, out_b, strict=True):
                torch.testing.assert_close(left, right)

            tok = prepare_lpap_surrogate_batch(
                values, bucket_count=4, permutation=perm_a
            )
            t_a = lpap_surrogate_targets(tok, k_max=3, permutation=perm_a)
            t_b = lpap_surrogate_targets(tok, k_max=3, permutation=perm_b)
            torch.testing.assert_close(t_a.source_indices, t_b.source_indices)
            torch.testing.assert_close(t_a.buckets, t_b.buckets)
            torch.testing.assert_close(t_a.dibs, t_b.dibs)

            # CPU fallback of lpap_triton matches lpap_torch deterministically.
            triton_cpu = lpap_triton(values, bucket_count=4, k_max=3)
            for left, right in zip(triton_cpu, out_a, strict=True):
                torch.testing.assert_close(left, right)
        finally:
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    unittest.main()
