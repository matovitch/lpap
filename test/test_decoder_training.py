from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder_training import (
    LPAPDecoderModelConfig,
    LPAPDecoderRegularizationConfig,
    LPAPDecoderRunConfig,
    LPAPDecoderTrainingConfig,
    LPAPDecoderTeacherConfig,
    create_lpap_decoder_training_session,
    iter_lpap_decoder_training,
)
from lpap.energy_bank import EnergyBankConfig
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateValidationConfig,
)


class LPAPDecoderTrainingTest(unittest.TestCase):
    def test_session_trains_and_logs_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            surrogate = LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            save_training_checkpoint(
                root / "checkpoints" / "surrogate_synthetic.pt",
                model=surrogate,
                step=1,
                training_state={
                    "run_config": {
                        "data": LPAPSurrogateDataConfig(
                            batch_size=2,
                            bucket_count=4,
                            probe_count=4,
                            harmonics=SyntheticHarmonicConfig(harmonic_count=3),
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
            config = LPAPDecoderTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=4,
                    probe_count=4,
                    harmonics=SyntheticHarmonicConfig(harmonic_count=3),
                ),
                decoder=LPAPDecoderModelConfig(
                    frontend_initial_temperature=0.5,
                    hidden_dim=16,
                    layer_count=1,
                    head_count=4,
                ),
                validation=LPAPSurrogateValidationConfig(every=1, batch_size=3),
                teacher=LPAPDecoderTeacherConfig(require_checkpoint=True),
                regularization=LPAPDecoderRegularizationConfig(
                    source_ce_weight=0.25,
                    source_ce_l1_reference=0.1,
                    source_ce_power=2.0,
                ),
                run=LPAPDecoderRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-decoder",
                ),
            )
            session = create_lpap_decoder_training_session(
                project_root=root, config=config, device="cpu"
            )

            results = list(iter_lpap_decoder_training(session))

            self.assertEqual(len(results), 2)
            self.assertTrue(session.surrogate_checkpoint_loaded)
            self.assertEqual(session.surrogate_k_max, 2)
            self.assertEqual(session.surrogate_model_config["hidden_dim"], 16)
            self.assertEqual(session.prior_data.harmonics.harmonic_count, 3)
            self.assertEqual(session.prior_data.kind, "harmonics")
            self.assertEqual(len(session.surrogate.blocks), 1)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("reconstruction_l1", results[-1].metrics)
            self.assertIn("source_ce_regularizer", results[-1].metrics)
            self.assertIn("source_ce_weight", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertIn("validation_source_ce_regularizer", results[-1].metrics)
            self.assertTrue(any(result.improved for result in results))

    def test_session_trains_from_energy_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bank_path = root / "data" / "bank.pt"
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"energies": torch.randn(32, 16)}, bank_path)
            surrogate = LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            data = LPAPSurrogateDataConfig(
                batch_size=2,
                bucket_count=4,
                probe_count=4,
                kind="energy_bank",
                energy_bank=EnergyBankConfig(path="data/bank.pt"),
            )
            save_training_checkpoint(
                root / "checkpoints" / "surrogate_bank.pt",
                model=surrogate,
                step=1,
                training_state={
                    "run_config": {"data": data.as_dict()},
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
            config = LPAPDecoderTrainingConfig(
                data=data,
                decoder=LPAPDecoderModelConfig(
                    frontend_initial_temperature=0.5,
                    hidden_dim=16,
                    layer_count=1,
                    head_count=4,
                ),
                validation=LPAPSurrogateValidationConfig(every=1, batch_size=3),
                teacher=LPAPDecoderTeacherConfig(
                    checkpoint_name="surrogate_bank.pt",
                    require_checkpoint=True,
                ),
                regularization=LPAPDecoderRegularizationConfig(
                    source_ce_weight=0.25,
                    source_ce_l1_reference=0.1,
                    source_ce_power=2.0,
                ),
                run=LPAPDecoderRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-decoder-bank",
                    checkpoint_name="decoder_bank.pt",
                    log_name="decoder_bank.sqlite",
                ),
            )
            session = create_lpap_decoder_training_session(
                project_root=root, config=config, device="cpu"
            )
            self.assertEqual(session.prior_data.kind, "energy_bank")
            self.assertIsNotNone(session.energy_bank)
            results = list(iter_lpap_decoder_training(session))
            self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
