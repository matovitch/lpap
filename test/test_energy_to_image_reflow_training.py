from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder import LPAPDecoderTransformer
from lpap.energy_to_image_reflow_training import (
    EnergyToImageReflowConfig,
    EnergyToImageReflowRunConfig,
    EnergyToImageReflowTeacherConfig,
    EnergyToImageReflowTrainingConfig,
    collect_energy_to_image_reflow_gallery,
    create_energy_to_image_reflow_training_session,
    iter_energy_to_image_reflow_training,
)
from lpap.energy_to_image_training import EnergyToImageSourceConfig
from lpap.flow import DilatedConvFlow1d
from lpap.image_to_energy_training import (
    ImageToEnergyFlowConfig,
    ImageToEnergyImageConfig,
    ImageToEnergyOptimizerConfig,
    ImageToEnergyValidationConfig,
)
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import LPAPSurrogateDataConfig


class EnergyToImageReflowTrainingTest(unittest.TestCase):
    def test_session_distills_high_step_teacher_into_student(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            data_dir = root / "data"
            data_dir.mkdir(parents=True)
            torch.save(
                {
                    "images": torch.arange(8 * 1 * 4 * 4, dtype=torch.uint8).reshape(
                        8, 1, 4, 4
                    ),
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
            teacher = DilatedConvFlow1d(
                sequence_length=16,
                width=8,
                time_dim=8,
                dilation_cycles=1,
                dilations=(1, 2),
            )
            save_training_checkpoint(
                checkpoint_dir / "energy_to_image.pt",
                model=teacher,
                step=1,
                training_state={"model_config": {"sequence_length": 16}},
            )
            config = EnergyToImageReflowTrainingConfig(
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
                teacher=EnergyToImageReflowTeacherConfig(
                    checkpoint_name="energy_to_image.pt",
                    teacher_steps=2,
                    warm_start_student=True,
                ),
                reflow=EnergyToImageReflowConfig(
                    student_steps=1,
                    endpoint_l2_weight=1.0,
                    image_anchor_l2_weight=0.1,
                ),
                optimizer=ImageToEnergyOptimizerConfig(
                    learning_rate=1.0e-4,
                    max_grad_norm=1.0,
                ),
                validation=ImageToEnergyValidationConfig(
                    every=1,
                    batch_size=2,
                    euler_steps=(1, 2),
                ),
                run=EnergyToImageReflowRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-energy-to-image-reflow",
                    checkpoint_name="energy_to_image_reflow.pt",
                    log_name="energy_to_image_reflow.sqlite",
                ),
            )
            session = create_energy_to_image_reflow_training_session(
                project_root=root, config=config, device="cpu"
            )

            results = list(iter_energy_to_image_reflow_training(session))
            gallery = collect_energy_to_image_reflow_gallery(session, sample_count=1)

            self.assertEqual(len(results), 2)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("teacher_endpoint_l2", results[-1].metrics)
            self.assertIn("image_anchor_l2", results[-1].metrics)
            self.assertIn("validation_student_teacher_l2_steps_1", results[-1].metrics)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(gallery[0].source.shape, (16,))
            self.assertEqual(gallery[0].student.shape, (16,))


if __name__ == "__main__":
    unittest.main()
