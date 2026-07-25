from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.data import SyntheticHarmonicConfig
from lpap.energy_bank import EnergyBankConfig
from lpap.flow import DilatedConvFlow1d
from lpap.image_to_energy_training import (
    ImageToEnergyFlowConfig,
    ImageToEnergyImageConfig,
    ImageToEnergyRunConfig,
    ImageToEnergyTargetConfig,
    ImageToEnergyTimeConfig,
    ImageToEnergyTrainingConfig,
    ImageToEnergyValidationConfig,
    collect_image_to_energy_gallery,
    create_image_to_energy_training_session,
    image_to_energy_training_config_from_dict,
    iter_image_to_energy_training,
)


class ImageToEnergyTrainingTest(unittest.TestCase):
    def test_collect_gallery_integrates_requested_steps(self) -> None:
        model = DilatedConvFlow1d(
            sequence_length=16,
            width=8,
            time_dim=8,
            dilation_cycles=1,
            dilations=(1,),
        )
        images = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)

        items = collect_image_to_energy_gallery(
            model=model,
            images=images,
            side=4,
            steps=(8, 4),
            device=torch.device("cpu"),
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].image.shape, (1, 4, 4))
        self.assertEqual(tuple(items[0].generated), (8, 4))
        self.assertEqual(items[0].generated[8].shape, (1, 4, 4))

    def test_session_trains_and_logs_small_run(self) -> None:
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
            config = ImageToEnergyTrainingConfig(
                image=ImageToEnergyImageConfig(
                    dataset_path="data/images.pt",
                    batch_size=2,
                    side=4,
                    normalize=True,
                    shuffle=False,
                ),
                target=ImageToEnergyTargetConfig(
                    harmonics=SyntheticHarmonicConfig(harmonic_count=3)
                ),
                flow=ImageToEnergyFlowConfig(
                    sequence_length=16,
                    width=8,
                    time_dim=8,
                    dilation_cycles=1,
                    dilations=(1, 2),
                ),
                time=ImageToEnergyTimeConfig(distribution="uniform"),
                validation=ImageToEnergyValidationConfig(
                    every=1,
                    batch_size=4,
                    euler_steps=(1,),
                ),
                run=ImageToEnergyRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-image-to-energy",
                ),
            )
            session = create_image_to_energy_training_session(
                project_root=root, config=config, device="cpu"
            )

            results = list(iter_image_to_energy_training(session))

            self.assertEqual(len(results), 2)
            self.assertEqual(session.validation_image_loader.batch_size, 4)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertIn(
                "validation_generated_energy_rms_steps_1", results[-1].metrics
            )
            self.assertIsNone(session.energy_bank)

    def test_target_config_round_trip_defaults_to_harmonics(self) -> None:
        legacy = {
            "harmonics": SyntheticHarmonicConfig(harmonic_count=3).as_dict(),
        }
        config = image_to_energy_training_config_from_dict(
            {
                "image": ImageToEnergyImageConfig(
                    dataset_path="data/images.pt", side=4, batch_size=2
                ).as_dict(),
                "target": legacy,
                "flow": ImageToEnergyFlowConfig(
                    sequence_length=16,
                    width=8,
                    time_dim=8,
                    dilation_cycles=1,
                    dilations=(1,),
                ).as_dict(),
                "time": ImageToEnergyTimeConfig(distribution="uniform").as_dict(),
                "optimizer": {"learning_rate": 1.0e-3, "max_grad_norm": 1.0},
                "validation": ImageToEnergyValidationConfig(
                    every=1, batch_size=2, euler_steps=(1,)
                ).as_dict(),
                "run": ImageToEnergyRunConfig(steps=1, run_id="legacy").as_dict(),
            }
        )
        self.assertEqual(config.target.kind, "harmonics")

    def test_session_trains_from_energy_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "data" / "images.pt"
            bank_path = root / "data" / "bank.pt"
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
            torch.save(
                {"energies": torch.randn(16, 16), "metadata": {"n": 16}},
                bank_path,
            )
            config = ImageToEnergyTrainingConfig(
                image=ImageToEnergyImageConfig(
                    dataset_path="data/images.pt",
                    batch_size=2,
                    side=4,
                    normalize=True,
                    shuffle=False,
                ),
                target=ImageToEnergyTargetConfig(
                    kind="energy_bank",
                    harmonics=SyntheticHarmonicConfig(harmonic_count=3),
                    energy_bank=EnergyBankConfig(path="data/bank.pt"),
                ),
                flow=ImageToEnergyFlowConfig(
                    sequence_length=16,
                    width=8,
                    time_dim=8,
                    dilation_cycles=1,
                    dilations=(1, 2),
                ),
                time=ImageToEnergyTimeConfig(distribution="uniform"),
                validation=ImageToEnergyValidationConfig(
                    every=1,
                    batch_size=4,
                    euler_steps=(1,),
                ),
                run=ImageToEnergyRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-image-to-energy-bank",
                ),
            )
            session = create_image_to_energy_training_session(
                project_root=root, config=config, device="cpu"
            )
            results = list(iter_image_to_energy_training(session))

            self.assertEqual(len(results), 2)
            self.assertIsNotNone(session.energy_bank)
            assert session.energy_bank is not None
            self.assertEqual(tuple(session.energy_bank.shape), (16, 16))
            self.assertIn("loss", results[-1].metrics)


if __name__ == "__main__":
    unittest.main()
