from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder import LPAPDecoderTransformer
from lpap.energy_to_image_training import (
    EnergyToImageRunConfig,
    EnergyToImageSourceConfig,
    EnergyToImageTrainingConfig,
    create_energy_to_image_training_session,
    iter_energy_to_image_training,
)
from lpap.image_to_energy_training import (
    ImageToEnergyFlowConfig,
    ImageToEnergyImageConfig,
    ImageToEnergyTimeConfig,
    ImageToEnergyValidationConfig,
)
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import LPAPSurrogateDataConfig


class EnergyToImageTrainingTest(unittest.TestCase):
    def test_session_trains_from_decoder_projected_harmonics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            data_dir = root / "data"
            data_dir.mkdir(parents=True)
            torch.save(
                {
                    "images": torch.arange(8 * 1 * 4 * 4, dtype=torch.uint8).reshape(8, 1, 4, 4),
                    "names": [str(index) for index in range(8)],
                },
                data_dir / "images.pt",
            )
            surrogate = LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            harmonics = SyntheticHarmonicConfig(harmonic_count=3)
            save_training_checkpoint(
                checkpoint_dir / "surrogate.pt",
                model=surrogate,
                step=1,
                training_state={
                    "run_config": {
                        "data": LPAPSurrogateDataConfig(
                            batch_size=2,
                            bucket_count=4,
                            probe_count=4,
                            harmonics=harmonics,
                        ).as_dict()
                    },
                    "model_config": {
                        "value_count": 16,
                        "bucket_count": 4,
                        "probe_count": 4,
                        "k_max": 2,
                        "hidden_dim": 16,
                        "layer_count": 1,
                        "head_count": 4,
                        "permutation_seed": 123,
                    },
                },
            )
            decoder = LPAPDecoderTransformer(
                value_count=16,
                frontend_initial_temperature=0.5,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            save_training_checkpoint(
                checkpoint_dir / "decoder.pt",
                model=decoder,
                step=1,
                training_state={
                    "model_config": {
                        "value_count": 16,
                        "bucket_count": 4,
                        "probe_count": 4,
                        "surrogate": {},
                        "frontend_initial_temperature": 0.5,
                        "hidden_dim": 16,
                        "layer_count": 1,
                        "head_count": 4,
                        "permutation_seed": 123,
                    }
                },
            )
            config = EnergyToImageTrainingConfig(
                image=ImageToEnergyImageConfig(
                    dataset_path="data/images.pt",
                    batch_size=2,
                    side=4,
                    normalize=True,
                    shuffle=False,
                ),
                source=EnergyToImageSourceConfig(
                    surrogate_checkpoint_name="surrogate.pt",
                    decoder_checkpoint_name="decoder.pt",
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
                    batch_size=2,
                    euler_steps=(1,),
                ),
                run=EnergyToImageRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-energy-to-image",
                    checkpoint_name="energy_to_image.pt",
                    log_name="energy_to_image.sqlite",
                ),
            )
            session = create_energy_to_image_training_session(
                project_root=root, config=config, device="cpu"
            )

            results = list(iter_energy_to_image_training(session))

            self.assertEqual(len(results), 2)
            self.assertEqual(session.harmonics.harmonic_count, 3)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertIn("validation_generated_image_rms_steps_1", results[-1].metrics)


if __name__ == "__main__":
    unittest.main()
