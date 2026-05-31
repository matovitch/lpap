from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder import LPAPDecoderTransformer
from lpap.flow import DilatedConvFlow1d
from lpap.image_autoencoder_training import (
    ImageAutoencoderIntegrationConfig,
    ImageAutoencoderLossConfig,
    ImageAutoencoderRunConfig,
    ImageAutoencoderSourceConfig,
    ImageAutoencoderTrainingConfig,
    collect_image_autoencoder_gallery,
    create_image_autoencoder_training_session,
    iter_image_autoencoder_training,
)
from lpap.image_to_energy_training import (
    ImageToEnergyFlowConfig,
    ImageToEnergyImageConfig,
    ImageToEnergyOptimizerConfig,
    ImageToEnergyValidationConfig,
)
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import LPAPSurrogateDataConfig


class ImageAutoencoderTrainingTest(unittest.TestCase):
    def test_session_trains_and_logs_total_autoencoder_metrics(self) -> None:
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
            flow_kwargs = {
                "sequence_length": 16,
                "width": 8,
                "time_dim": 8,
                "dilation_cycles": 1,
                "dilations": (1, 2),
            }
            image_to_energy = DilatedConvFlow1d(**flow_kwargs)
            energy_to_image = DilatedConvFlow1d(**flow_kwargs)
            save_training_checkpoint(
                checkpoint_dir / "image_to_energy.pt",
                model=image_to_energy,
                step=1,
                training_state={"model_config": {"sequence_length": 16}},
            )
            save_training_checkpoint(
                checkpoint_dir / "energy_to_image_reflow_8.pt",
                model=energy_to_image,
                step=1,
                training_state={"model_config": {"sequence_length": 16}},
            )
            flow_config = ImageToEnergyFlowConfig(**flow_kwargs)
            config = ImageAutoencoderTrainingConfig(
                image=ImageToEnergyImageConfig(
                    dataset_path="data/images.pt",
                    batch_size=2,
                    side=4,
                    normalize=True,
                    shuffle=False,
                ),
                source=ImageAutoencoderSourceConfig(
                    surrogate_checkpoint_name="surrogate.pt",
                    decoder_checkpoint_name="decoder.pt",
                    image_to_energy_checkpoint_name="image_to_energy.pt",
                    energy_to_image_checkpoint_name="energy_to_image_reflow_8.pt",
                    train_image_to_energy_flow=True,
                    train_surrogate=True,
                    train_decoder=True,
                    train_energy_to_image_flow=True,
                ),
                image_to_energy_flow=flow_config,
                energy_to_image_flow=flow_config,
                integration=ImageAutoencoderIntegrationConfig(
                    image_to_energy_steps=1,
                    energy_to_image_steps=1,
                ),
                loss=ImageAutoencoderLossConfig(
                    image_l2_weight=1.0,
                    energy_l2_weight=0.25,
                    energy_l1_weight=0.01,
                    energy_l1_reference=0.1,
                    surrogate_teacher_weight=0.1,
                ),
                optimizer=ImageToEnergyOptimizerConfig(
                    learning_rate=1.0e-4,
                    max_grad_norm=1.0,
                ),
                validation=ImageToEnergyValidationConfig(
                    every=1,
                    batch_size=2,
                    euler_steps=(1,),
                ),
                run=ImageAutoencoderRunConfig(
                    steps=1,
                    display_every=1,
                    run_id="tiny-image-autoencoder",
                    checkpoint_name="image_autoencoder.pt",
                    log_name="image_autoencoder.sqlite",
                ),
            )
            session = create_image_autoencoder_training_session(
                project_root=root, config=config, device="cpu"
            )

            results = list(iter_image_autoencoder_training(session))
            gallery = collect_image_autoencoder_gallery(session, sample_count=1)

            self.assertEqual(len(results), 1)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("image_reconstruction_l2", results[-1].metrics)
            self.assertIn("energy_reconstruction_l2", results[-1].metrics)
            self.assertIn("energy_l1_regularizer", results[-1].metrics)
            self.assertIn("surrogate_teacher_ce", results[-1].metrics)
            self.assertIn("validation_image_reconstruction_l2", results[-1].metrics)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(gallery[0].image.shape, (16,))
            self.assertEqual(gallery[0].encoded_energy.shape, (16,))
            self.assertEqual(gallery[0].reconstructed_image.shape, (16,))


if __name__ == "__main__":
    unittest.main()
