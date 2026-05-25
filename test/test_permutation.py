from __future__ import annotations

import unittest

import torch

from lpap.permutation import (
    apply_grouped_permutation,
    fold_grouped_permutation_tokens,
    make_grouped_permutation_indices,
    reverse_grouped_permutation,
    unfold_grouped_permutation_tokens,
)


class GroupedPermutationTest(unittest.TestCase):
    def test_permutation_is_seeded_and_invertible(self) -> None:
        permutation = make_grouped_permutation_indices(
            value_count=24, bucket_count=4, seed=123
        )
        same_permutation = make_grouped_permutation_indices(
            value_count=24, bucket_count=4, seed=123
        )
        other_permutation = make_grouped_permutation_indices(
            value_count=24, bucket_count=4, seed=124
        )

        torch.testing.assert_close(permutation, same_permutation)
        self.assertFalse(torch.equal(permutation, other_permutation))
        torch.testing.assert_close(permutation.sort().values, torch.arange(24))

        values = torch.arange(48, dtype=torch.float32).reshape(2, 24)
        restored = reverse_grouped_permutation(
            apply_grouped_permutation(values, permutation), permutation
        )
        torch.testing.assert_close(restored, values)

    def test_each_bucket_gets_balanced_source_groups(self) -> None:
        value_count = 30
        bucket_count = 5
        probe_count = value_count // bucket_count
        permutation = make_grouped_permutation_indices(
            value_count=value_count, bucket_count=bucket_count, seed=3
        )

        folded_sources = (permutation // probe_count).reshape(probe_count, bucket_count)
        for bucket_index in range(bucket_count):
            counts = torch.bincount(
                folded_sources[:, bucket_index], minlength=bucket_count
            )
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_fold_and_unfold_tokens_round_trip(self) -> None:
        values = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        tokens = fold_grouped_permutation_tokens(values, bucket_count=3)

        self.assertEqual(tokens.shape, (2, 3, 4))
        torch.testing.assert_close(unfold_grouped_permutation_tokens(tokens), values)


if __name__ == "__main__":
    unittest.main()
