from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lpap.decoder_training import LPAPDecoderTrainingConfig
from lpap.image_autoencoder_training import ImageAutoencoderTrainingConfig
from lpap.image_energy_flow_training import ImageEnergyFlowTrainingConfig
from lpap.surrogate_training import LPAPSurrogateTrainingConfig
from lpap.teacher_config import project_teacher_config
from lpap.training_log import upsert_run
from lpap.training_notebook import (
    restore_training_config_from_log,
    training_config_from_file,
    training_config_from_project_file,
    training_config_path,
    training_config_to_toml,
)
from lpap.visualization_notebook import render_decoder_run_gallery
from support import training_fixture


class TrainingNotebookConfigTest(unittest.TestCase):
    def test_loads_fixture_training_toml_configs(self) -> None:
        teacher_path = training_fixture("teacher_c128_k16.toml")

        surrogate = project_teacher_config(teacher_path, "surrogate")
        decoder = project_teacher_config(teacher_path, "decoder")
        image_energy_flow = training_config_from_file(
            training_fixture("image_energy_flow.toml"), "image_energy_flow"
        )
        image_autoencoder = training_config_from_file(
            training_fixture("image_autoencoder.toml"), "image_autoencoder"
        )

        self.assertIsInstance(surrogate, LPAPSurrogateTrainingConfig)
        self.assertEqual(surrogate.run.run_id, "surrogate_c128_k16")
        self.assertEqual(surrogate.run.checkpoint_name, "surrogate_c128_k16.pt")
        self.assertEqual(surrogate.model.k_max, 16)
        self.assertIsInstance(decoder, LPAPDecoderTrainingConfig)
        self.assertEqual(decoder.run.run_id, "decoder_c128_k16")
        self.assertEqual(decoder.run.checkpoint_name, "decoder_c128_k16.pt")
        self.assertEqual(decoder.teacher.checkpoint_name, "surrogate_c128_k16.pt")
        self.assertTrue(decoder.teacher.require_checkpoint)
        self.assertEqual(decoder.regularization.source_ce_weight, 0.1)
        self.assertIsInstance(image_energy_flow, ImageEnergyFlowTrainingConfig)
        self.assertEqual(image_energy_flow.run.run_id, "image_energy_flow")
        self.assertEqual(
            image_energy_flow.image.dataset_path, "data/images_32x32_gray.pt"
        )
        self.assertEqual(image_energy_flow.time.distribution, "beta")
        self.assertGreater(image_energy_flow.prior.sigma, 0.0)
        self.assertGreater(image_energy_flow.prior.scale, 0.0)
        self.assertEqual(image_energy_flow.validation.num_batches, 4)
        self.assertIsInstance(image_autoencoder, ImageAutoencoderTrainingConfig)
        self.assertEqual(image_autoencoder.run.run_id, "image_autoencoder")
        self.assertEqual(image_autoencoder.integration.image_to_energy_steps, 16)
        self.assertEqual(image_autoencoder.integration.energy_to_image_steps, 16)
        self.assertEqual(
            image_autoencoder.source.flow_checkpoint_name,
            "image_energy_flow.pt",
        )
        self.assertGreater(image_autoencoder.loss.signed_mass_floor_tau, 0.0)
        self.assertGreaterEqual(image_autoencoder.loss.signed_mass_balance_weight, 0.0)
        self.assertGreater(image_autoencoder.run.steps, 0)
        self.assertGreater(image_autoencoder.image.batch_size, 0)

    def test_live_training_tomls_parse(self) -> None:
        """Operator configs under configs/training/ must still load; values free."""
        project_root = Path(__file__).resolve().parents[1]
        for kind in ("image_energy_flow", "image_autoencoder"):
            config = training_config_from_project_file(project_root, kind)
            self.assertIsNotNone(config)
        for name in (
            "teacher_c128_k16.toml",
            "teacher_c256_k24.toml",
            "teacher_c512_k32.toml",
        ):
            path = project_root / "configs" / "training" / name
            self.assertTrue(path.is_file(), name)
            project_teacher_config(path, "surrogate")
            project_teacher_config(path, "decoder")

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
        config = project_teacher_config(
            training_fixture("teacher_c128_k16.toml"), "surrogate"
        )

        text = training_config_to_toml(config)

        self.assertIn("[data.energy_bank]", text)
        self.assertIn("encoded_energies_bank_flow_best.pt", text)
        self.assertIn("surrogate_c128_k16.pt", text)

    def test_decoder_toml_serializes_energy_bank(self) -> None:
        config = project_teacher_config(
            training_fixture("teacher_c128_k16.toml"), "decoder"
        )

        text = training_config_to_toml(config)

        self.assertIn("[data.energy_bank]", text)
        self.assertIn("[teacher]", text)

    def test_image_energy_flow_serializes_prior(self) -> None:
        config = training_config_from_file(
            training_fixture("image_energy_flow.toml"), "image_energy_flow"
        )
        text = training_config_to_toml(config)
        self.assertIn("[prior]", text)
        self.assertRegex(text, r"(?m)^sigma = ")
        self.assertRegex(text, r"(?m)^scale = ")

    def test_restores_training_toml_from_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            # Restore falls back to default_training_config when kind TOML is absent;
            # that default uses log_name "surrogate.sqlite".
            log_path = project_root / "training_logs" / "surrogate.sqlite"
            source_config = project_teacher_config(
                training_fixture("teacher_c128_k16.toml"),
                "surrogate",
            )
            run_config = source_config.as_run_config()
            run_config["run"]["run_training"] = False
            run_config["run"]["resume_from_checkpoint"] = True
            run_config["run"]["steps"] = 17
            run_config["run"]["comment"] = "restored from sqlite"
            run_config["run"]["log_name"] = "surrogate.sqlite"
            upsert_run(
                log_path,
                run_id="surrogate_c128_k16:restored-run",
                checkpoint_path="checkpoints/surrogate.pt",
                config=run_config,
                metadata={"comment": "restored from sqlite"},
            )

            restored_path = restore_training_config_from_log(
                "surrogate",
                project_root=project_root,
                run_id="surrogate_c128_k16:restored-run",
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
            source_config = project_teacher_config(
                training_fixture("teacher_c128_k16.toml"),
                "decoder",
            )
            upsert_run(
                log_path,
                run_id="decoder_c128_k16:missing-checkpoint",
                checkpoint_path="checkpoints/missing_decoder.pt",
                config=source_config.as_run_config(),
            )

            with self.assertRaises(FileNotFoundError):
                render_decoder_run_gallery(
                    project_root=project_root,
                    log_path=log_path,
                    run_id="decoder_c128_k16:missing-checkpoint",
                    sample_count=1,
                )


if __name__ == "__main__":
    unittest.main()
