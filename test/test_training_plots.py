from __future__ import annotations

import unittest

import torch

from lpap.image_to_energy_training import ImageToEnergyGalleryItem
from lpap.energy_to_image_reflow_training import EnergyToImageReflowGalleryItem
from lpap.image_autoencoder_training import ImageAutoencoderGalleryItem
from lpap.training_plots import (
    render_energy_to_image_reflow_gallery_html,
    render_image_autoencoder_gallery_html,
    render_image_to_energy_gallery_html,
    render_loss_history_svg,
)


class TrainingPlotsTest(unittest.TestCase):
    def test_renders_validation_regularizer_curve(self) -> None:
        svg = render_loss_history_svg(
            [
                {
                    "step": 1,
                    "loss": 2.0,
                    "validation_loss": 1.5,
                    "validation_source_ce_regularizer": 0.4,
                },
                {
                    "step": 2,
                    "loss": 1.8,
                    "validation_loss": 1.1,
                    "validation_source_ce_regularizer": 0.2,
                },
            ],
            validation_regularizer_metrics=("validation_source_ce_regularizer",),
        )

        self.assertIn("source ce regularizer", svg)
        self.assertIn("stroke-dasharray", svg)

    def test_renders_image_to_energy_gallery_with_image_then_steps(self) -> None:
        html = render_image_to_energy_gallery_html(
            [
                ImageToEnergyGalleryItem(
                    image=torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4),
                    generated={
                        64: torch.linspace(-1.0, 1.0, 16).reshape(1, 4, 4),
                        32: torch.linspace(1.0, -1.0, 16).reshape(1, 4, 4),
                        4: torch.zeros(1, 4, 4),
                    },
                )
            ],
            steps=(64, 32, 4),
            size=4,
        )

        self.assertLess(html.index("image"), html.index(">64 steps<"))
        self.assertLess(html.index(">64 steps<"), html.index(">32 steps<"))
        self.assertLess(html.index(">32 steps<"), html.index(">4 steps<"))
        self.assertIn("rgb(255, 255, 255)", html)
        self.assertIn("rgb(255, 0, 0)", html)
        self.assertIn("rgb(0, 0, 255)", html)

    def test_renders_energy_to_image_reflow_gallery(self) -> None:
        html = render_energy_to_image_reflow_gallery_html(
            [
                EnergyToImageReflowGalleryItem(
                    source=torch.linspace(-1.0, 1.0, 16),
                    target=torch.linspace(0.0, 1.0, 16),
                    teacher=torch.ones(16),
                    student=torch.zeros(16),
                    error=-torch.ones(16),
                )
            ],
            size=4,
        )

        self.assertLess(html.index("source energy"), html.index("image sample"))
        self.assertLess(html.index("image sample"), html.index("teacher"))
        self.assertLess(html.index("teacher"), html.index("student"))
        self.assertLess(html.index("student"), html.index("student - teacher"))
        self.assertIn("rgb(255, 255, 255)", html)
        self.assertIn("rgb(0, 0, 255)", html)

    def test_renders_image_autoencoder_gallery(self) -> None:
        html = render_image_autoencoder_gallery_html(
            [
                ImageAutoencoderGalleryItem(
                    image=torch.linspace(0.0, 1.0, 16),
                    reconstructed_image=torch.ones(16),
                    image_error=torch.linspace(-1.0, 1.0, 16),
                    encoded_energy=torch.linspace(-1.0, 1.0, 16),
                    decoded_energy=torch.linspace(1.0, -1.0, 16),
                    energy_error=-torch.ones(16),
                )
            ],
            size=4,
        )

        self.assertLess(html.index("image"), html.index("reconstruction"))
        self.assertLess(html.index("reconstruction"), html.index("image error"))
        self.assertLess(html.index("encoded energy"), html.index("decoded energy"))
        self.assertIn("energy error", html)
        self.assertIn("rgb(255, 255, 255)", html)
        self.assertIn("rgb(0, 0, 255)", html)


if __name__ == "__main__":
    unittest.main()
