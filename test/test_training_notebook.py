from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lpap.decoder_training import LPAPDecoderTrainingConfig
from lpap.energy_to_image_training import EnergyToImageTrainingConfig
from lpap.energy_to_image_reflow_training import EnergyToImageReflowTrainingConfig
from lpap.image_autoencoder_training import ImageAutoencoderTrainingConfig
from lpap.image_to_energy_training import ImageToEnergyTrainingConfig
from lpap.surrogate_training import LPAPSurrogateTrainingConfig
from lpap.training_log import upsert_run
from lpap.training_notebook import (
    restore_training_config_from_log,
    training_config_from_file,
    training_config_from_project_file,
    training_config_path,
    training_config_to_toml,
)
from lpap.visualization_notebook import render_decoder_run_gallery


class TrainingNotebookConfigTest(unittest.TestCase):
    def test_loads_project_training_toml_configs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        surrogate = training_config_from_project_file(project_root, "surrogate")
        decoder = training_config_from_project_file(project_root, "decoder")
        image_to_energy = training_config_from_project_file(
            project_root, "image_to_energy"
        )
        energy_to_image = training_config_from_project_file(
            project_root, "energy_to_image"
        )
        energy_to_image_reflow = training_config_from_project_file(
            project_root, "energy_to_image_reflow"
        )
        image_autoencoder = training_config_from_project_file(
            project_root, "image_autoencoder"
        )

        self.assertIsInstance(surrogate, LPAPSurrogateTrainingConfig)
        self.assertEqual(surrogate.run.run_id, "surrogate_synthetic")
        self.assertEqual(surrogate.run.tags, ("baseline",))
        self.assertIsInstance(decoder, LPAPDecoderTrainingConfig)
        self.assertEqual(decoder.run.run_id, "decoder_synthetic")
        self.assertTrue(decoder.teacher.require_checkpoint)
        self.assertEqual(decoder.regularization.source_ce_weight, 0.1)
        self.assertIsInstance(image_to_energy, ImageToEnergyTrainingConfig)
        self.assertEqual(image_to_energy.run.run_id, "image_to_energy")
        self.assertEqual(
            image_to_energy.image.dataset_path, "data/images_32x32_gray.pt"
        )
        self.assertEqual(image_to_energy.time.distribution, "beta")
        self.assertIsInstance(energy_to_image, EnergyToImageTrainingConfig)
        self.assertEqual(energy_to_image.run.run_id, "energy_to_image")
        self.assertEqual(
            energy_to_image.source.decoder_checkpoint_name, "decoder_synthetic.pt"
        )
        self.assertIsInstance(energy_to_image_reflow, EnergyToImageReflowTrainingConfig)
        self.assertEqual(energy_to_image_reflow.run.run_id, "energy_to_image_reflow")
        self.assertEqual(energy_to_image_reflow.reflow.student_steps, 8)
        self.assertEqual(
            energy_to_image_reflow.teacher.checkpoint_name, "energy_to_image.pt"
        )
        self.assertIsInstance(image_autoencoder, ImageAutoencoderTrainingConfig)
        self.assertEqual(image_autoencoder.run.run_id, "image_autoencoder")
        self.assertEqual(image_autoencoder.integration.image_to_energy_steps, 8)
        self.assertEqual(image_autoencoder.integration.energy_to_image_steps, 8)
        self.assertEqual(
            image_autoencoder.source.energy_to_image_checkpoint_name,
            "energy_to_image_reflow_8.pt",
        )

    def test_loads_custom_surrogate_toml_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "surrogate.toml"
            path.write_text(
                """
                [data]
                batch_size = 4
                bucket_count = 8
                probe_count = 2

                [data.harmonics]
                harmonic_count = 3
                gain_variance = 0.5
                gain_half_life = 2.0
                spikiness_range = [1.0, 3.0]
                dtype = "torch.float32"

                [model]
                k_max = 2
                hidden_dim = 16
                layer_count = 1
                head_count = 4

                [optimizer]
                learning_rate = 0.002

                [validation]
                enabled = true
                every = 2
                batch_size = 5
                seed = 42
                validate_at_end = true

                [run]
                run_training = false
                resume_from_checkpoint = false
                steps = 7
                seed = 9
                permutation_seed = 10
                display_every = 1
                log_every = 1
                run_id = "surrogate_custom"
                checkpoint_name = "surrogate_custom.pt"
                log_name = "surrogate_custom.sqlite"
                note = "custom config"
                tags = ["tiny", "test"]
                pinned = true
                """,
                encoding="utf-8",
            )

            config = training_config_from_file(path, "surrogate")

            self.assertEqual(config.data.batch_size, 4)
            self.assertEqual(config.model.hidden_dim, 16)
            self.assertEqual(config.run.run_id, "surrogate_custom")
            self.assertEqual(config.run.tags, ("tiny", "test"))
            self.assertTrue(config.run.pinned)

    def test_training_config_path_uses_model_kind_filename(self) -> None:
        path = training_config_path("/tmp/project", "energy_to_image")

        self.assertEqual(
            path, Path("/tmp/project/configs/training/energy_to_image.toml")
        )

    def test_serializes_training_config_to_toml(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = training_config_from_project_file(project_root, "surrogate")

        text = training_config_to_toml(config)

        self.assertIn("[data.harmonics]", text)
        self.assertIn('dtype = "torch.float32"', text)
        self.assertIn('tags = ["baseline"]', text)

    def test_decoder_toml_does_not_serialize_harmonics(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = training_config_from_project_file(project_root, "decoder")

        text = training_config_to_toml(config)

        self.assertNotIn("[data.harmonics]", text)
        self.assertIn("[teacher]", text)

    def test_restores_training_toml_from_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            log_path = project_root / "training_logs" / "surrogate.sqlite"
            source_config = training_config_from_project_file(
                Path(__file__).resolve().parents[1], "surrogate"
            )
            run_config = source_config.as_run_config()
            run_config["run"]["run_training"] = False
            run_config["run"]["resume_from_checkpoint"] = True
            run_config["run"]["steps"] = 17
            run_config["run"]["note"] = "restored from sqlite"
            upsert_run(
                log_path,
                run_id="surrogate_synthetic:restored-run",
                checkpoint_path="checkpoints/surrogate.pt",
                config=run_config,
            )

            restored_path = restore_training_config_from_log(
                "surrogate",
                project_root=project_root,
                run_id="surrogate_synthetic:restored-run",
            )
            restored_config = training_config_from_file(restored_path, "surrogate")

            self.assertEqual(
                restored_path, project_root / "configs" / "training" / "surrogate.toml"
            )
            self.assertFalse(restored_config.run.resume_from_checkpoint)
            self.assertEqual(restored_config.run.steps, 17)
            self.assertEqual(restored_config.run.note, "restored from sqlite")

    def test_decoder_run_gallery_requires_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            log_path = project_root / "training_logs" / "decoder.sqlite"
            source_config = training_config_from_project_file(
                Path(__file__).resolve().parents[1], "decoder"
            )
            upsert_run(
                log_path,
                run_id="decoder_synthetic:missing-checkpoint",
                checkpoint_path="checkpoints/missing_decoder.pt",
                config=source_config.as_run_config(),
            )

            with self.assertRaises(FileNotFoundError):
                render_decoder_run_gallery(
                    project_root=project_root,
                    log_path=log_path,
                    run_id="decoder_synthetic:missing-checkpoint",
                    sample_count=1,
                )


if __name__ == "__main__":
    unittest.main()
