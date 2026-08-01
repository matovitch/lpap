from __future__ import annotations

import unittest

import torch

from lpap.image_energy_flow_training import ImageEnergyFlowGalleryItem
from lpap.image_autoencoder_training import ImageAutoencoderGalleryItem
from lpap.training_plots import (
    render_image_autoencoder_gallery_html,
    render_image_energy_flow_gallery_html,
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

    def test_renders_image_energy_flow_gallery_with_both_directions(self) -> None:
        html = render_image_energy_flow_gallery_html(
            [
                ImageEnergyFlowGalleryItem(
                    image=torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4),
                    encoded={
                        64: torch.linspace(-1.0, 1.0, 16).reshape(1, 4, 4),
                        32: torch.linspace(1.0, -1.0, 16).reshape(1, 4, 4),
                        4: torch.zeros(1, 4, 4),
                    },
                    reconstructed={
                        64: torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4),
                        32: torch.linspace(1.0, 0.0, 16).reshape(1, 4, 4),
                        4: torch.zeros(1, 4, 4),
                    },
                    prior_energy=torch.linspace(-0.5, 0.5, 16).reshape(1, 4, 4),
                    from_prior={
                        64: torch.linspace(0.2, 0.8, 16).reshape(1, 4, 4),
                        32: torch.linspace(0.8, 0.2, 16).reshape(1, 4, 4),
                        4: torch.full((1, 4, 4), 0.5),
                    },
                )
            ],
            steps=(64, 32, 4),
            size=4,
        )

        self.assertLess(html.index("image"), html.index(">64 steps<"))
        self.assertLess(html.index(">64 steps<"), html.index(">32 steps<"))
        self.assertLess(html.index(">32 steps<"), html.index(">4 steps<"))
        self.assertIn("round-trip energy → image", html)
        self.assertIn("prior energy → image", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn('image-rendering: pixelated', html)

    def test_renders_image_autoencoder_gallery(self) -> None:
        from lpap.image_autoencoder_training import ImageAutoencoderGalleryPairItem

        html = render_image_autoencoder_gallery_html(
            [
                ImageAutoencoderGalleryItem(
                    image=torch.linspace(0.0, 1.0, 16),
                    encoded_energy=torch.linspace(-1.0, 1.0, 16),
                    pairs=(
                        ImageAutoencoderGalleryPairItem(
                            name="c256",
                            decoded_energy=torch.linspace(1.0, -1.0, 16),
                            reconstructed_image=torch.ones(16),
                            energy_error=-torch.ones(16),
                            image_error=torch.linspace(-1.0, 1.0, 16),
                        ),
                        ImageAutoencoderGalleryPairItem(
                            name="c128",
                            decoded_energy=torch.linspace(-0.5, 0.5, 16),
                            reconstructed_image=torch.zeros(16),
                            energy_error=torch.ones(16) * 0.25,
                            image_error=torch.linspace(1.0, -1.0, 16),
                        ),
                    ),
                )
            ],
            size=4,
        )

        self.assertLess(html.index("source energy"), html.index("c256 energy"))
        self.assertLess(html.index("c256 energy"), html.index("c128 energy"))
        self.assertLess(html.index("c128 energy"), html.index("Δ energy c256"))
        self.assertLess(html.index("Δ energy c256"), html.index("Δ energy c128"))
        self.assertLess(html.index("source image"), html.index("c256 image"))
        self.assertLess(html.index("c256 image"), html.index("c128 image"))
        self.assertLess(html.index("c128 image"), html.index("Δ image c256"))
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("energy then image", html)
        self.assertIn("display γ=1", html)

    def test_display_gamma_lifts_small_values_and_rejects_nonpositive(self) -> None:
        from lpap.training_plots import _apply_display_gamma

        self.assertAlmostEqual(_apply_display_gamma(0.25, gamma=1.0), 0.25)
        self.assertGreater(_apply_display_gamma(0.25, gamma=0.5), 0.25)
        self.assertLess(_apply_display_gamma(0.25, gamma=2.0), 0.25)
        self.assertAlmostEqual(_apply_display_gamma(-0.25, gamma=0.5), -0.5)
        with self.assertRaises(ValueError):
            _apply_display_gamma(0.5, gamma=0.0)

        from lpap.image_autoencoder_training import ImageAutoencoderGalleryPairItem

        html = render_image_autoencoder_gallery_html(
            [
                ImageAutoencoderGalleryItem(
                    image=torch.linspace(0.0, 1.0, 16),
                    encoded_energy=torch.linspace(-1.0, 1.0, 16),
                    pairs=(
                        ImageAutoencoderGalleryPairItem(
                            name="c256",
                            decoded_energy=torch.linspace(1.0, -1.0, 16),
                            reconstructed_image=torch.ones(16),
                            energy_error=-torch.ones(16),
                            image_error=torch.linspace(-1.0, 1.0, 16),
                        ),
                    ),
                )
            ],
            size=4,
            gamma=0.5,
        )
        self.assertIn("display γ=0.5", html)
        with self.assertRaises(ValueError):
            render_image_autoencoder_gallery_html(
                [
                    ImageAutoencoderGalleryItem(
                        image=torch.zeros(16),
                        encoded_energy=torch.zeros(16),
                        pairs=(),
                    )
                ],
                size=4,
                gamma=0.0,
            )

    def test_renders_signed_triplet_gallery_as_png(self) -> None:
        from types import SimpleNamespace

        from lpap.training_plots import render_signed_triplet_gallery_html

        html = render_signed_triplet_gallery_html(
            [
                SimpleNamespace(
                    energy=torch.linspace(-1.0, 1.0, 16),
                    lpap=torch.linspace(1.0, -1.0, 16),
                    surrogate_hard=torch.linspace(-0.5, 0.5, 16),
                    decoder=torch.zeros(16),
                )
            ],
            size=4,
            display_px=64,
            gamma=1.0,
        )
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("image-rendering: pixelated", html)
        self.assertIn("source energy", html)
        self.assertIn("oracle LPAP", html)
        self.assertIn("surrogate hard", html)
        self.assertIn("decoder soft", html)


if __name__ == "__main__":
    unittest.main()
