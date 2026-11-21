from __future__ import annotations

import subprocess
import sys
import unittest

import torch

from lpap.ops import lpap_tokens_torch, lpap_torch
from lpap.permutation import fold_permutation_tokens, make_permutation_indices
from lpap.surrogate import lpap_surrogate_targets
from lpap.triton_ops import lpap_tokens_triton, lpap_triton


class LpapTorchTest(unittest.TestCase):
    def test_single_projection_selects_largest_amplitude_per_bucket_lane(self) -> None:
        # Column-major fold (C=2, P=2): lane 0 = [1, -2], lane 1 = [9, -8].
        values = torch.tensor([[1.0, 9.0, -2.0, -8.0]])

        buckets, dibs, remaining = lpap_torch(values, bucket_count=2, k_max=1)

        torch.testing.assert_close(buckets, torch.tensor([[-2.0, 9.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0]]))
        torch.testing.assert_close(remaining, torch.tensor([[1.0, 0.0, 0.0, -8.0]]))
        torch.testing.assert_close(values, torch.tensor([[1.0, 9.0, -2.0, -8.0]]))

    def test_repeated_projection_uses_rolls_and_swap_backs(self) -> None:
        # Column-major fold (C=3, P=2): lane 0=[1, 4], lane 1=[9, 2], lane 2=[3, 8].
        values = torch.tensor([[1.0, 9.0, 3.0, 4.0, 2.0, 8.0]])

        buckets, dibs, remaining = lpap_torch(values, bucket_count=3, k_max=2)

        torch.testing.assert_close(buckets, torch.tensor([[4.0, 9.0, 8.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0, 0]]))
        torch.testing.assert_close(
            remaining, torch.tensor([[1.0, 0.0, 3.0, 0.0, 2.0, 0.0]])
        )

    def test_reproduces_manim_rows_bit_exact(self) -> None:
        # Matches animations/lpap_operator.py _ROWS (C=4, P=3, K_MAX=3).
        rows = (
            (-1.0, 6.0, 3.0),
            (2.0, 5.0, 4.0),
            (-8.0, 7.0, -5.0),
            (9.0, -6.0, -4.0),
        )
        c, p = len(rows), len(rows[0])
        flat = torch.tensor(
            [[rows[r][col] for col in range(p) for r in range(c)]],
            dtype=torch.float32,
        )
        buckets, dibs, _remaining = lpap_torch(flat, bucket_count=c, k_max=3)
        targets = lpap_surrogate_targets(
            fold_permutation_tokens(flat, bucket_count=c), k_max=3
        )

        torch.testing.assert_close(buckets, torch.tensor([[7.0, 6.0, -8.0, 9.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[2, 1, 0, 0]]))
        torch.testing.assert_close(targets.buckets, buckets)
        torch.testing.assert_close(targets.dibs, dibs)
        torch.testing.assert_close(
            targets.source_indices, torch.tensor([[6, 4, 2, 3]])
        )

    def test_table_updates_by_amplitude_across_roll_step(self) -> None:
        # Column-major (C=3, P=2): lane 0=[1, 0], lane 1=[2, 0], lane 2=[100, 50].
        values = torch.tensor([[1.0, 2.0, 100.0, 0.0, 0.0, 50.0]])

        buckets, dibs, remaining = lpap_torch(values, bucket_count=3, k_max=2)

        torch.testing.assert_close(buckets, torch.tensor([[50.0, 2.0, 100.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[1, 0, 0]]))
        torch.testing.assert_close(
            remaining, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        )

    def test_batched_items_are_independent(self) -> None:
        # Column-major (C=2, P=2):
        # row 0: lane 0=[1, 2], lane 1=[9, 8] -> [2, 9]
        # row 1: lane 0=[-7, 6], lane 1=[3, 4] -> [-7, 4]
        values = torch.tensor(
            [
                [1.0, 9.0, 2.0, 8.0],
                [-7.0, 3.0, 6.0, 4.0],
            ]
        )

        buckets, dibs, remaining = lpap_torch(values, bucket_count=2, k_max=1)

        torch.testing.assert_close(buckets, torch.tensor([[2.0, 9.0], [-7.0, 4.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0], [0, 0]]))
        torch.testing.assert_close(
            remaining,
            torch.tensor([[1.0, 0.0, 0.0, 8.0], [0.0, 3.0, 6.0, 0.0]]),
        )

    def test_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            lpap_torch(torch.ones(1, 4), bucket_count=0, k_max=1)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            lpap_torch(torch.ones(1, 5), bucket_count=2, k_max=1)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            lpap_torch(torch.ones(1, 4), bucket_count=2, k_max=0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            lpap_torch(torch.ones(1, 4), bucket_count=2, k_max=-1)
        with self.assertRaisesRegex(ValueError, "at most bucket_count"):
            lpap_torch(torch.ones(1, 4), bucket_count=2, k_max=3)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            lpap_torch(torch.empty(1, 0), bucket_count=2, k_max=1)

    def test_lpap_triton_rejects_requires_grad_on_cpu_too(self) -> None:
        values = torch.randn(2, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "autograd"):
            lpap_triton(values, bucket_count=4, k_max=2)

    def test_import_lpap_does_not_eagerly_import_triton(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, lpap; assert 'triton' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class LpapTritonTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Triton LPAP requires CUDA")
    def test_triton_matches_torch_for_same_inputs(self) -> None:
        values = torch.tensor(
            [
                [1.0, 9.0, 3.0, 4.0, 2.0, 8.0],
                [-5.0, -1.0, 7.0, 6.0, 4.0, -3.0],
            ],
            device="cuda",
        )

        expected = lpap_torch(values.cpu(), bucket_count=3, k_max=2)
        actual = lpap_triton(values, bucket_count=3, k_max=2)

        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor.cpu(), expected_tensor)

    @unittest.skipUnless(torch.cuda.is_available(), "Triton LPAP requires CUDA")
    def test_triton_matches_torch_for_random_and_large_shapes(self) -> None:
        generator = torch.Generator().manual_seed(123)
        cases = (
            (1, 6, 3, 1),
            (2, 8, 4, 2),
            (3, 16, 4, 3),
            (4, 1024, 128, 16),
            (4, 1024, 256, 24),
            (4, 1024, 512, 32),
            (0, 16, 4, 2),
        )

        for batch_count, value_count, bucket_count, k_max in cases:
            with self.subTest(
                batch_count=batch_count,
                value_count=value_count,
                bucket_count=bucket_count,
                k_max=k_max,
            ):
                values = torch.randn(batch_count, value_count, generator=generator)
                perm = make_permutation_indices(value_count=value_count, seed=42)

                expected = lpap_torch(
                    values, bucket_count=bucket_count, k_max=k_max, permutation=perm
                )
                actual = lpap_triton(
                    values.cuda(),
                    bucket_count=bucket_count,
                    k_max=k_max,
                    permutation=perm.cuda(),
                )

                for actual_tensor, expected_tensor in zip(
                    actual, expected, strict=True
                ):
                    torch.testing.assert_close(actual_tensor.cpu(), expected_tensor)

    @unittest.skipUnless(torch.cuda.is_available(), "Triton LPAP requires CUDA")
    def test_triton_matches_torch_on_zeros_nans_and_dtypes(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                tokens = torch.tensor(
                    [
                        [[0.0, 0.0], [0.0, 0.0], [5.0, -3.0], [0.0, 0.0]],
                        [[float("nan"), 2.0], [float("nan"), float("nan")], [1.0, 4.0], [-6.0, 0.0]],
                    ],
                    dtype=dtype,
                )
                expected = lpap_tokens_torch(tokens, k_max=3)
                actual = lpap_tokens_triton(tokens.cuda(), k_max=3)
                torch.testing.assert_close(
                    actual.buckets.cpu(), expected.buckets, equal_nan=True
                )
                torch.testing.assert_close(actual.dibs.cpu(), expected.dibs)
                torch.testing.assert_close(
                    actual.bucket_probes.cpu(), expected.bucket_probes
                )
                torch.testing.assert_close(
                    actual.remaining.cpu(), expected.remaining, equal_nan=True
                )

    @unittest.skipUnless(torch.cuda.is_available(), "Triton LPAP requires CUDA")
    def test_triton_rejects_requires_grad_on_cuda(self) -> None:
        values = torch.randn(2, 8, device="cuda", requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "autograd"):
            lpap_triton(values, bucket_count=4, k_max=2)


if __name__ == "__main__":
    unittest.main()
