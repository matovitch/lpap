from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lpap.decoder_training import LPAPDecoderTrainingConfig
from lpap.image_autoencoder_training import ImageAutoencoderTrainingConfig
from lpap.image_energy_flow_training import ImageEnergyFlowTrainingConfig
from lpap.surrogate_training import LPAPSurrogateTrainingConfig
from lpap.training_log import upsert_run
from lpap.training_notebook import (
    default_image_energy_flow_energy_bank_training_config,
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
        image_energy_flow = training_config_from_project_file(
            project_root, "image_energy_flow"
        )
        image_autoencoder = training_config_from_project_file(
            project_root, "image_autoencoder"
        )
        image_energy_flow_bank = training_config_from_file(
            project_root / "configs/training/image_energy_flow_energy_bank.toml",
            "image_energy_flow_energy_bank",
        )

        self.assertIsInstance(surrogate, LPAPSurrogateTrainingConfig)
        self.assertEqual(surrogate.run.run_id, "surrogate_synthetic")
        self.assertEqual(surrogate.run.checkpoint_name, "surrogate_c128_k4.pt")
        self.assertIn("legacy c128 k_max=4", surrogate.run.comment)
        self.assertIsInstance(decoder, LPAPDecoderTrainingConfig)
        self.assertEqual(decoder.run.run_id, "decoder_synthetic")
        self.assertEqual(decoder.run.checkpoint_name, "decoder_c128_k4.pt")
        self.assertEqual(decoder.teacher.checkpoint_name, "surrogate_c128_k4.pt")
        self.assertTrue(decoder.teacher.require_checkpoint)
        self.assertEqual(decoder.regularization.source_ce_weight, 0.1)
        self.assertIsInstance(image_energy_flow, ImageEnergyFlowTrainingConfig)
        self.assertEqual(image_energy_flow.run.run_id, "image_energy_flow")
        self.assertEqual(
            image_energy_flow.image.dataset_path, "data/images_32x32_gray.pt"
        )
        self.assertEqual(image_energy_flow.time.distribution, "beta")
        self.assertEqual(image_energy_flow.prior.kind, "harmonics")
        self.assertIsNone(image_energy_flow.prior.energy_bank)
        self.assertEqual(image_energy_flow_bank.prior.kind, "energy_bank")
        self.assertEqual(
            image_energy_flow_bank.prior.energy_bank.path,
            "data/encoded_energies_ae_best.pt",
        )
        self.assertEqual(image_energy_flow_bank.run.steps, 2000)
        self.assertEqual(image_energy_flow_bank.validation.every, 50)
        self.assertIsInstance(image_autoencoder, ImageAutoencoderTrainingConfig)
        self.assertEqual(image_autoencoder.run.run_id, "image_autoencoder")
        self.assertEqual(image_autoencoder.integration.image_to_energy_steps, 16)
        self.assertEqual(image_autoencoder.integration.energy_to_image_steps, 16)
        self.assertEqual(
            image_autoencoder.source.flow_checkpoint_name,
            "image_energy_flow.pt",
        )
        self.assertEqual(
            image_autoencoder.run.comment,
            "16-step e2e AE; bidirectional flow clone at init",
        )
        self.assertEqual(image_autoencoder.loss.signed_mass_balance_weight, 0.02)
        self.assertEqual(image_autoencoder.loss.energy_l1_weight, 0.5)
        self.assertEqual(image_autoencoder.loss.surrogate_teacher_weight, 0.05)
        self.assertEqual(image_autoencoder.loss.signed_mass_floor_tau, 0.01)
        self.assertEqual(image_autoencoder.loss.signed_mass_floor_coef, 1.0)
        self.assertEqual(image_autoencoder.run.steps, 20_000)
        self.assertEqual(image_autoencoder.image.batch_size, 32)
        self.assertEqual(image_autoencoder.validation.batch_size, 32)

    def test_default_energy_bank_helper(self) -> None:
        flow = default_image_energy_flow_energy_bank_training_config()
        self.assertEqual(flow.prior.kind, "energy_bank")
        self.assertEqual(flow.run.run_id, "image_energy_flow_energy_bank")
        self.assertEqual(flow.run.steps, 2000)

    def test_energy_bank_kinds_load_from_project_and_map_backend(self) -> None:
        from lpap.training_notebook import training_backend_kind

        project_root = Path(__file__).resolve().parents[1]
        flow = training_config_from_project_file(
            project_root, "image_energy_flow_energy_bank"
        )
        self.assertEqual(flow.prior.kind, "energy_bank")
        self.assertEqual(
            training_backend_kind("image_energy_flow_energy_bank"),
            "image_energy_flow",
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

                [data.energy_bank]
                path = "data/encoded_energies_ae_best.pt"
                energies_key = "energies"

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
                comment = "custom config"
                pinned = true
                """,
                encoding="utf-8",
            )

            config = training_config_from_file(path, "surrogate")

            self.assertEqual(config.data.batch_size, 4)
            self.assertEqual(config.model.hidden_dim, 16)
            self.assertEqual(config.run.run_id, "surrogate_custom")
            self.assertEqual(config.run.comment, "custom config")
            self.assertTrue(config.run.pinned)

    def test_training_config_path_uses_model_kind_filename(self) -> None:
        path = training_config_path("/tmp/project", "image_energy_flow")

        self.assertEqual(
            path, Path("/tmp/project/configs/training/image_energy_flow.toml")
        )

    def test_serializes_training_config_to_toml(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = training_config_from_project_file(project_root, "surrogate")

        text = training_config_to_toml(config)

        self.assertIn("[data.energy_bank]", text)
        self.assertIn("encoded_energies_ae_best.pt", text)
        self.assertIn("surrogate_c128_k4.pt", text)

    def test_decoder_toml_serializes_energy_bank(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = training_config_from_project_file(project_root, "decoder")

        text = training_config_to_toml(config)

        self.assertIn("[data.energy_bank]", text)
        self.assertNotIn("[data.harmonics]", text)
        self.assertIn("[teacher]", text)

    def test_image_energy_flow_serializes_prior(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = training_config_from_project_file(project_root, "image_energy_flow")
        text = training_config_to_toml(config)
        self.assertIn("[prior.harmonics]", text)
        self.assertIn('kind = "harmonics"', text)

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
            run_config["run"]["comment"] = "restored from sqlite"
            upsert_run(
                log_path,
                run_id="surrogate_synthetic:restored-run",
                checkpoint_path="checkpoints/surrogate.pt",
                config=run_config,
                metadata={"comment": "restored from sqlite"},
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
            self.assertEqual(restored_config.run.comment, "restored from sqlite")

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
