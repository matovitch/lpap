from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from lpap.data import ImageTensorDataset
from lpap.energy_bank_encode import encode_image_dataset_to_energy_bank


class _DummyFlow(nn.Module):
    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(values)


class EnergyBankEncodeTest(unittest.TestCase):
    def test_requires_normalize_true(self) -> None:
        images = torch.randint(0, 256, (8, 1, 4, 4), dtype=torch.uint8)
        dataset = ImageTensorDataset(images, normalize=False)
        with self.assertRaisesRegex(ValueError, "normalize=True"):
            encode_image_dataset_to_energy_bank(
                dataset,
                flow=_DummyFlow(),
                side=4,
                image_to_energy_steps=2,
                device=torch.device("cpu"),
                batch_size=4,
                probe_batches=1,
            )

    def test_encode_uses_float_batch_and_asserts(self) -> None:
        images = torch.randint(0, 256, (12, 1, 4, 4), dtype=torch.uint8)
        dataset = ImageTensorDataset(images, normalize=True)
        messages: list[str] = []

        def fake_integrate(flow, seq, steps, *, t0, t1):
            del flow, steps, t0, t1
            # Energy-scale output independent of image intensity.
            return 0.03 * torch.randn_like(seq)

        with patch(
            "lpap.energy_bank_encode.integrate_euler_midpoint_time",
            side_effect=fake_integrate,
        ):
            result = encode_image_dataset_to_energy_bank(
                dataset,
                flow=_DummyFlow(),
                side=4,
                image_to_energy_steps=4,
                device=torch.device("cpu"),
                batch_size=4,
                probe_batches=1,
                progress_every=1,
                progress=messages.append,
            )

        self.assertEqual(tuple(result.energies.shape), (12, 16))
        self.assertLess(abs(result.final_stats.mean), 0.05)
        self.assertGreater(result.probe_raw_image_rel_rmse, 0.5)
        self.assertEqual(result.metadata["normalize_applied"], "ImageTensorDataset.float_batch")
        self.assertTrue(any(m.startswith("probe ok") for m in messages))

    def test_probe_rejects_image_scale_output(self) -> None:
        images = torch.randint(0, 256, (8, 1, 4, 4), dtype=torch.uint8)
        dataset = ImageTensorDataset(images, normalize=True)

        def fake_integrate(flow, seq, steps, *, t0, t1):
            del flow, steps, t0, t1
            # Simulate forgot-/255 identity: stay near 0..255 Hilbert values.
            return seq * 255.0

        with patch(
            "lpap.energy_bank_encode.integrate_euler_midpoint_time",
            side_effect=fake_integrate,
        ):
            with self.assertRaisesRegex(ValueError, "skip /255"):
                encode_image_dataset_to_energy_bank(
                    dataset,
                    flow=_DummyFlow(),
                    side=4,
                    image_to_energy_steps=2,
                    device=torch.device("cpu"),
                    batch_size=4,
                    probe_batches=1,
                )


if __name__ == "__main__":
    unittest.main()
