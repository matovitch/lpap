from __future__ import annotations

import unittest

import torch

from lpap.ops import lpap_torch
from lpap.triton_ops import lpap_triton


class LpapTorchTest(unittest.TestCase):
    def test_single_projection_selects_largest_amplitude_per_bucket_lane(self) -> None:
        values = torch.tensor([[1.0, 9.0, -2.0, -8.0]])

        buckets, dibs, remaining = lpap_torch(values, bucket_count=2, k_max=1)

        torch.testing.assert_close(buckets, torch.tensor([[9.0, -8.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0]]))
        torch.testing.assert_close(remaining, torch.tensor([[1.0, 0.0, -2.0, 0.0]]))
        torch.testing.assert_close(values, torch.tensor([[1.0, 9.0, -2.0, -8.0]]))

    def test_repeated_projection_uses_rolls_and_swap_backs(self) -> None:
        values = torch.tensor([[1.0, 9.0, 3.0, 4.0, 2.0, 8.0]])

        buckets, dibs, remaining = lpap_torch(values, bucket_count=3, k_max=2)

        torch.testing.assert_close(buckets, torch.tensor([[9.0, 4.0, 8.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0, 0]]))
        torch.testing.assert_close(
            remaining, torch.tensor([[1.0, 0.0, 3.0, 0.0, 2.0, 0.0]])
        )

    def test_table_is_always_full_and_updates_by_amplitude(self) -> None:
        values = torch.tensor([[1.0, 0.0, 2.0, 0.0, 100.0, 50.0]])

        buckets, dibs, remaining = lpap_torch(values, bucket_count=3, k_max=2)

        torch.testing.assert_close(buckets, torch.tensor([[50.0, 2.0, 100.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[1, 0, 0]]))
        torch.testing.assert_close(
            remaining, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        )

    def test_batched_items_are_independent(self) -> None:
        values = torch.tensor(
            [
                [1.0, 9.0, 2.0, 8.0],
                [-7.0, 3.0, 6.0, 4.0],
            ]
        )

        buckets, dibs, remaining = lpap_torch(values, bucket_count=2, k_max=1)

        torch.testing.assert_close(buckets, torch.tensor([[9.0, 8.0], [-7.0, 6.0]]))
        torch.testing.assert_close(dibs, torch.tensor([[0, 0], [0, 0]]))
        torch.testing.assert_close(
            remaining,
            torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]]),
        )

    def test_rejects_invalid_shapes(self) -> None:
        buckets, dibs, remaining = lpap_torch(torch.ones(1, 4), bucket_count=2, k_max=0)
        torch.testing.assert_close(buckets, torch.zeros(1, 2))
        torch.testing.assert_close(dibs, torch.zeros(1, 2, dtype=torch.int64))
        torch.testing.assert_close(remaining, torch.ones(1, 4))

        with self.assertRaisesRegex(ValueError, "must be positive"):
            lpap_torch(torch.ones(1, 4), bucket_count=0, k_max=1)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            lpap_torch(torch.ones(1, 5), bucket_count=2, k_max=1)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            lpap_torch(torch.ones(1, 4), bucket_count=2, k_max=-1)


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


if __name__ == "__main__":
    unittest.main()
