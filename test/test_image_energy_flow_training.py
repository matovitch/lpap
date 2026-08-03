from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.flow import DilatedConvFlow1d, bidirectional_image_energy_state
from lpap.flow_training import (
    FlowImageConfig,
    FlowModelConfig,
    FlowTimeConfig,
    FlowValidationConfig,
    sample_bidirectional_flow_time,
)
from lpap.image_energy_flow_training import (
    ENERGY_TO_IMAGE_T0,
    ENERGY_TO_IMAGE_T1,
    IMAGE_TO_ENERGY_T0,
    IMAGE_TO_ENERGY_T1,
    ImageEnergyFlowPriorConfig,
    ImageEnergyFlowRunConfig,
    ImageEnergyFlowTrainingConfig,
    collect_image_energy_flow_gallery,
    create_image_energy_flow_training_session,
    image_energy_flow_training_config_from_dict,
    iter_image_energy_flow_training,
)


class ImageEnergyFlowTrainingTest(unittest.TestCase):
    def test_bidirectional_state_endpoints(self) -> None:
        image = torch.zeros(2, 1, 4)
        energy = torch.ones(2, 1, 4)
        left_t = torch.tensor([-1.0, 0.0])
        values, velocity = bidirectional_image_energy_state(image, energy, left_t)
        self.assertTrue(torch.allclose(values[0], image[0]))
        self.assertTrue(torch.allclose(values[1], energy[1]))
        self.assertTrue(torch.allclose(velocity[0], energy[0] - image[0]))
        right_t = torch.tensor([0.0, 1.0])
        values_r, velocity_r = bidirectional_image_energy_state(image, energy, right_t)
        self.assertTrue(torch.allclose(values_r[0], energy[0]))
        self.assertTrue(torch.allclose(values_r[1], image[1]))
        self.assertTrue(torch.allclose(velocity_r[1], image[1] - energy[1]))

    def test_sample_bidirectional_time_range(self) -> None:
        gen = torch.Generator().manual_seed(0)
        times = sample_bidirectional_flow_time(
            batch_size=256,
            config=FlowTimeConfig(distribution="uniform"),
            generator=gen,
        )
        self.assertTrue(float(times.min()) >= -1.0)
        self.assertTrue(float(times.max()) <= 1.0)
        self.assertTrue(bool((times < 0).any()))
        self.assertTrue(bool((times > 0).any()))

    def test_session_trains_gaussian_prior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data" / "images.pt"
            dataset_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "images": torch.arange(8 * 1 * 4 * 4, dtype=torch.uint8).reshape(
                        8, 1, 4, 4
                    ),
                    "names": [str(index) for index in range(8)],
                },
                dataset_path,
            )
            config = ImageEnergyFlowTrainingConfig(
                image=FlowImageConfig(
                    dataset_path="data/images.pt",
                    batch_size=2,
                    side=4,
                    normalize=True,
                    shuffle=False,
                ),
                prior=ImageEnergyFlowPriorConfig(sigma=1.0),
                flow=FlowModelConfig(
                    sequence_length=16,
                    width=8,
                    time_dim=8,
                    dilation_cycles=1,
                    dilations=(1, 2),
                ),
                time=FlowTimeConfig(distribution="uniform"),
                validation=FlowValidationConfig(
                    every=1,
                    batch_size=4,
                    euler_steps=(1,),
                ),
                run=ImageEnergyFlowRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-image-energy-flow",
                ),
            )
            session = create_image_energy_flow_training_session(
                project_root=root, config=config, device="cpu"
            )
            results = list(iter_image_energy_flow_training(session))
            self.assertEqual(len(results), 2)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertIn(
                "validation_encoded_energy_rms_steps_1", results[-1].metrics
            )
            self.assertIn(
                "validation_reconstructed_image_rms_steps_1", results[-1].metrics
            )
            self.assertEqual(session.config.prior.sigma, 1.0)

    def test_gallery_both_directions(self) -> None:
        model = DilatedConvFlow1d(
            sequence_length=16,
            width=8,
            time_dim=8,
            dilation_cycles=1,
            dilations=(1,),
        )
        images = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
        energies = torch.randn(1, 1, 16)
        items = collect_image_energy_flow_gallery(
            model=model,
            images=images,
            energies=energies,
            side=4,
            steps=(4, 2),
            device=torch.device("cpu"),
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(tuple(items[0].encoded), (4, 2))
        self.assertEqual(tuple(items[0].reconstructed), (4, 2))
        self.assertEqual(tuple(items[0].from_prior), (4, 2))
        self.assertEqual(tuple(items[0].prior_energy.shape), (1, 4, 4))
        self.assertEqual(
            (IMAGE_TO_ENERGY_T0, IMAGE_TO_ENERGY_T1),
            (-1.0, 0.0),
        )
        self.assertEqual(
            (ENERGY_TO_IMAGE_T0, ENERGY_TO_IMAGE_T1),
            (0.0, 1.0),
        )

    def test_config_round_trip(self) -> None:
        raw = {
            "image": FlowImageConfig(dataset_path="data/images.pt", side=4).as_dict(),
            "prior": {"sigma": 0.5},
            "flow": FlowModelConfig(sequence_length=16, width=8, time_dim=8).as_dict(),
            "time": FlowTimeConfig().as_dict(),
            "optimizer": {"learning_rate": 1e-4, "max_grad_norm": 1.0},
            "validation": FlowValidationConfig(euler_steps=(1,)).as_dict(),
            "run": ImageEnergyFlowRunConfig(steps=3).as_dict(),
        }
        config = image_energy_flow_training_config_from_dict(raw)
        self.assertEqual(config.prior.sigma, 0.5)

    def test_legacy_prior_keys_rejected(self) -> None:
        with self.assertRaises(ValueError):
            image_energy_flow_training_config_from_dict(
                {
                    "image": FlowImageConfig(
                        dataset_path="data/images.pt", side=4
                    ).as_dict(),
                    "prior": {"kind": "harmonics", "sigma": 1.0},
                    "flow": FlowModelConfig(
                        sequence_length=16, width=8, time_dim=8
                    ).as_dict(),
                    "time": FlowTimeConfig().as_dict(),
                    "optimizer": {"learning_rate": 1e-4, "max_grad_norm": 1.0},
                    "validation": FlowValidationConfig(euler_steps=(1,)).as_dict(),
                    "run": ImageEnergyFlowRunConfig(steps=3).as_dict(),
                }
            )


if __name__ == "__main__":
    unittest.main()
