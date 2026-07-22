from __future__ import annotations

import unittest

import torch

from lpap.image_autoencoder_loss import signed_mass_balance_loss


class SignedMassBalanceLossTest(unittest.TestCase):
    def test_balanced_at_floor_is_zero(self) -> None:
        # Two +0.02 and two -0.02 → m+ = m- = 0.01 == tau
        energy = torch.tensor([-0.02, -0.02, 0.02, 0.02])
        loss, imbalance, gap, floor = signed_mass_balance_loss(
            energy, floor_tau=0.01, floor_coef=1.0
        )
        self.assertAlmostEqual(float(gap), 0.0, places=6)
        self.assertAlmostEqual(float(floor), 0.0, places=6)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(imbalance), 0.0, places=6)

    def test_collapse_to_zero_pays_floor(self) -> None:
        energy = torch.zeros(8)
        loss, _imbalance, gap, floor = signed_mass_balance_loss(
            energy, floor_tau=0.01, floor_coef=1.0
        )
        self.assertAlmostEqual(float(gap), 0.0, places=6)
        self.assertAlmostEqual(float(floor), 2.0, places=6)
        self.assertAlmostEqual(float(loss), 2.0, places=6)

    def test_one_sided_pays_gap_and_floor(self) -> None:
        energy = torch.tensor([0.02, 0.02, 0.0, 0.0])
        loss, imbalance, gap, floor = signed_mass_balance_loss(
            energy, floor_tau=0.01, floor_coef=1.0
        )
        # m+=0.01, m-=0 → gap=((0.01)/0.01)^2=1, floor=0+1=1
        self.assertAlmostEqual(float(gap), 1.0, places=6)
        self.assertAlmostEqual(float(floor), 1.0, places=6)
        self.assertAlmostEqual(float(loss), 2.0, places=6)
        self.assertAlmostEqual(float(imbalance), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
