from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.data import ImageTensorDataset, image_dataloader, load_image_tensor_dataset


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


if __name__ == "__main__":
    unittest.main()
