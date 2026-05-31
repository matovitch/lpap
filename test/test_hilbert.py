from __future__ import annotations

import unittest

import torch

from lpap.hilbert import (
    hilbert_flatten_images,
    hilbert_permutation,
    hilbert_unflatten_images,
    inverse_permutation,
)


class HilbertTest(unittest.TestCase):
    def test_permutation_covers_raster_indices_once(self) -> None:
        perm = hilbert_permutation(side=4)

        self.assertEqual(perm.shape, (16,))
        torch.testing.assert_close(perm.sort().values, torch.arange(16))

    def test_inverse_permutation_round_trip(self) -> None:
        perm = hilbert_permutation(side=4)
        inverse = inverse_permutation(perm)

        torch.testing.assert_close(perm[inverse], torch.arange(16))
        torch.testing.assert_close(inverse[perm], torch.arange(16))

    def test_flatten_unflatten_round_trip(self) -> None:
        images = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)

        sequence = hilbert_flatten_images(images, side=4)
        restored = hilbert_unflatten_images(sequence, side=4)

        self.assertEqual(sequence.shape, (2, 1, 16))
        torch.testing.assert_close(restored, images)

    def test_non_power_of_two_side_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            hilbert_permutation(side=3)


if __name__ == "__main__":
    unittest.main()