from __future__ import annotations

import unittest

import torch

from lpap.permutation import (
    apply_permutation,
    fold_permutation_tokens,
    make_permutation_indices,
    reverse_permutation,
    unfold_permutation_tokens,
)


class PermutationTest(unittest.TestCase):
    def test_permutation_is_seeded_and_invertible(self) -> None:
        permutation = make_permutation_indices(value_count=24, seed=123)
        same_permutation = make_permutation_indices(value_count=24, seed=123)
        other_permutation = make_permutation_indices(value_count=24, seed=124)

        torch.testing.assert_close(permutation, same_permutation)
        self.assertFalse(torch.equal(permutation, other_permutation))
        torch.testing.assert_close(permutation.sort().values, torch.arange(24))

        values = torch.arange(48, dtype=torch.float32).reshape(2, 24)
        restored = reverse_permutation(
            apply_permutation(values, permutation), permutation
        )
        torch.testing.assert_close(restored, values)

    def test_permutation_seed_is_device_stable(self) -> None:
        cpu = make_permutation_indices(value_count=24, seed=123, device="cpu")
        # Requesting CUDA still uses the CPU RNG stream; device only selects
        # the output placement.
        placed = make_permutation_indices(value_count=24, seed=123, device="cpu")
        torch.testing.assert_close(cpu, placed)
        if torch.cuda.is_available():
            cuda = make_permutation_indices(value_count=24, seed=123, device="cuda")
            torch.testing.assert_close(cpu, cuda.cpu())

    def test_fold_and_unfold_tokens_round_trip(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        tokens = fold_permutation_tokens(values, bucket_count=3)

        self.assertEqual(tokens.shape, (2, 3, 4))
        torch.testing.assert_close(unfold_permutation_tokens(tokens), values)


if __name__ == "__main__":
    unittest.main()
