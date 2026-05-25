from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.data import (
    ImageTensorDataset,
    SyntheticHarmonicConfig,
    image_dataloader,
    load_image_tensor_dataset,
    sample_synthetic_harmonic_batch,
    synthetic_harmonic_dataloader,
)


class ImageDatasetTest(unittest.TestCase):
    def test_load_and_batch_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pt_path = temp_path / "images.pt"
            torch.save(
                {
                    "images": torch.full((1, 1, 2, 2), 76, dtype=torch.uint8),
                    "names": ["train_32x32/sample.png"],
                    "shape": (2, 2),
                    "dtype": "uint8",
                    "layout": "NCHW",
                },
                pt_path,
            )
            dataset = load_image_tensor_dataset(pt_path)

            self.assertEqual(len(dataset), 1)
            image, name = dataset[0]
            self.assertEqual(name, "train_32x32/sample.png")
            self.assertEqual(image.shape, (1, 2, 2))
            self.assertEqual(image.dtype, torch.uint8)
            self.assertTrue(
                torch.equal(image, torch.full((1, 2, 2), 76, dtype=torch.uint8))
            )

            normalized = ImageTensorDataset(
                dataset.images, dataset.names, normalize=True
            )
            normalized_image, _name = normalized[0]
            self.assertEqual(normalized_image.dtype, torch.float32)
            self.assertAlmostEqual(float(normalized_image[0, 0, 0]), 76 / 255)

            loader = image_dataloader(pt_path, batch_size=1, shuffle=False)
            batch_images, batch_names = next(iter(loader))
            self.assertEqual(batch_images.shape, (1, 1, 2, 2))
            self.assertEqual(batch_names, ("train_32x32/sample.png",))


class SyntheticHarmonicDatasetTest(unittest.TestCase):
    def test_config_samples_batch(self) -> None:
        config = SyntheticHarmonicConfig(
            harmonic_count=3,
            gain_variance=0.5,
            gain_half_life=2.0,
            spikiness_range=(3.0, 5.0),
        )

        batch = config.sample_batch(
            batch_size=2,
            n=9,
            generator=torch.Generator().manual_seed(123),
            return_parameters=True,
        )

        self.assertEqual(batch["values"].shape, (2, 9))
        self.assertEqual(batch["gains"].shape, (2, 3))
        self.assertEqual(config.as_dict()["harmonic_count"], 3)

    def test_sample_batch_shape_and_reproducibility(self) -> None:
        generator_a = torch.Generator().manual_seed(123)
        generator_b = torch.Generator().manual_seed(123)

        batch_a = sample_synthetic_harmonic_batch(
            batch_size=4,
            n=16,
            harmonic_count=5,
            gain_half_life=2.0,
            spikiness_range=(4.0, 8.0),
            generator=generator_a,
        )
        batch_b = sample_synthetic_harmonic_batch(
            batch_size=4,
            n=16,
            harmonic_count=5,
            gain_half_life=2.0,
            spikiness_range=(4.0, 8.0),
            generator=generator_b,
        )

        self.assertEqual(batch_a.shape, (4, 16))
        self.assertEqual(batch_a.dtype, torch.float32)
        torch.testing.assert_close(batch_a, batch_b)

    def test_parameters_include_decayed_gain_variance(self) -> None:
        generator = torch.Generator().manual_seed(123)

        batch = sample_synthetic_harmonic_batch(
            batch_size=8192,
            n=8,
            harmonic_count=4,
            gain_variance=2.0,
            gain_half_life=1.0,
            spikiness_range=(4.0, 4.0),
            generator=generator,
            return_parameters=True,
        )

        self.assertEqual(batch["values"].shape, (8192, 8))
        self.assertEqual(batch["gains"].shape, (8192, 4))
        self.assertEqual(batch["phases"].shape, (8192, 4))
        self.assertEqual(batch["spikiness"].shape, (8192, 4))
        torch.testing.assert_close(
            batch["frequencies"], torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        torch.testing.assert_close(batch["spikiness"], torch.full((8192, 4), 4.0))

        empirical_variance = batch["gains"].var(dim=0, unbiased=False)
        expected_variance = torch.tensor([2.0, 1.0, 0.5, 0.25])
        torch.testing.assert_close(
            empirical_variance, expected_variance, rtol=0.08, atol=0.08
        )

    def test_spikiness_produces_sparse_harmonic_spikes(self) -> None:
        batch = sample_synthetic_harmonic_batch(
            batch_size=1,
            n=1024,
            harmonic_count=1,
            gain_variance=1.0,
            spikiness_range=(4.0, 4.0),
            generator=torch.Generator().manual_seed(1),
            return_parameters=True,
        )

        values = batch["values"][0]
        gain = batch["gains"][0, 0]
        normalized = values.abs() / gain.abs().clamp_min(1.0e-12)

        self.assertLess(float((normalized > 0.1).float().mean()), 0.1)
        self.assertGreater(float(normalized.max()), 0.9)

    def test_zero_gain_variance_produces_zero_signals(self) -> None:
        batch = sample_synthetic_harmonic_batch(
            batch_size=3,
            n=11,
            harmonic_count=7,
            gain_variance=0.0,
        )

        torch.testing.assert_close(batch, torch.zeros(3, 11))

    def test_synthetic_dataloader_yields_prebatched_tensors(self) -> None:
        loader = synthetic_harmonic_dataloader(
            batch_size=3,
            n=17,
            harmonic_count=4,
            batches_per_epoch=2,
            seed=123,
        )

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (3, 17))
        self.assertEqual(batches[1].shape, (3, 17))


if __name__ == "__main__":
    unittest.main()
