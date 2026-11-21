from __future__ import annotations

import unittest

import torch

from lpap.permutation import (
    LPAP_FOLD_ORDER,
    apply_permutation,
    as_long_permutation,
    fold_permutation_tokens,
    folded_positions_to_flat_indices,
    invert_permutation_indices,
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
        placed = make_permutation_indices(value_count=24, seed=123, device="cpu")
        torch.testing.assert_close(cpu, placed)
        if torch.cuda.is_available():
            cuda = make_permutation_indices(value_count=24, seed=123, device="cuda")
            torch.testing.assert_close(cpu, cuda.cpu())

    def test_fold_and_unfold_tokens_round_trip(self) -> None:
        self.assertEqual(LPAP_FOLD_ORDER, "column_major")
        values = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        tokens = fold_permutation_tokens(values, bucket_count=3)

        self.assertEqual(tokens.shape, (2, 3, 4))
        # Column-major: token[b, p] == flat[p * C + b].
        torch.testing.assert_close(tokens[0, 1, 2], values[0, 2 * 3 + 1])
        torch.testing.assert_close(unfold_permutation_tokens(tokens), values)

        flat_idx = folded_positions_to_flat_indices(
            source_buckets=torch.tensor([0, 1, 2]),
            probe_indices=torch.tensor([3, 2, 1]),
            bucket_count=3,
        )
        torch.testing.assert_close(flat_idx, torch.tensor([9, 7, 5]))

    def test_permutation_validation_rejects_non_bijections_and_floats(self) -> None:
        with self.assertRaisesRegex(ValueError, "bijection"):
            as_long_permutation(torch.tensor([0, 0, 1, 2]), value_count=4)
        with self.assertRaisesRegex(ValueError, r"\[0, n\)"):
            as_long_permutation(torch.tensor([0, 1, 2, 4]), value_count=4)
        with self.assertRaisesRegex(ValueError, "bijection"):
            invert_permutation_indices(torch.tensor([1, 1, 2, 3]))
        with self.assertRaisesRegex(TypeError, "integer"):
            as_long_permutation(torch.tensor([0.0, 1.0, 2.0, 3.0]), value_count=4)
        with self.assertRaisesRegex(TypeError, "integer"):
            as_long_permutation(torch.tensor([True, False]), value_count=2)


if __name__ == "__main__":
    unittest.main()
