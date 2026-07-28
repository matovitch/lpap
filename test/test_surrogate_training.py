from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.data import SyntheticHarmonicConfig
from lpap.energy_bank import EnergyBankConfig
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateRunConfig,
    LPAPSurrogateTrainingConfig,
    LPAPSurrogateValidationConfig,
    create_lpap_surrogate_training_session,
    iter_lpap_surrogate_training,
)


class LPAPSurrogateTrainingTest(unittest.TestCase):
    def test_session_trains_and_logs_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LPAPSurrogateTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=4,
                    probe_count=4,
                    harmonics=SyntheticHarmonicConfig(harmonic_count=3),
                ),
                model=LPAPSurrogateModelConfig(
                    k_max=2,
                    hidden_dim=16,
                    layer_count=1,
                    head_count=4,
                ),
                run=LPAPSurrogateRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-surrogate",
                ),
                validation=LPAPSurrogateValidationConfig(every=1, batch_size=3),
            )
            session = create_lpap_surrogate_training_session(
                project_root=Path(temp_dir), config=config, device="cpu"
            )

            results = list(iter_lpap_surrogate_training(session))

            self.assertEqual(len(results), 2)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertTrue(any(result.improved for result in results))

    def test_session_trains_from_energy_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bank_path = root / "data" / "bank.pt"
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"energies": torch.randn(32, 16)}, bank_path)
            config = LPAPSurrogateTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=4,
                    probe_count=4,
                    kind="energy_bank",
                    energy_bank=EnergyBankConfig(path="data/bank.pt"),
                ),
                model=LPAPSurrogateModelConfig(
                    k_max=2,
                    hidden_dim=16,
                    layer_count=1,
                    head_count=4,
                ),
                run=LPAPSurrogateRunConfig(
                    steps=2,
                    display_every=1,
                    run_id="tiny-surrogate-bank",
                    checkpoint_name="surrogate_bank.pt",
                    log_name="surrogate_bank.sqlite",
                ),
                validation=LPAPSurrogateValidationConfig(every=1, batch_size=3),
            )
            session = create_lpap_surrogate_training_session(
                project_root=root, config=config, device="cpu"
            )
            self.assertIsNotNone(session.energy_bank)
            results = list(iter_lpap_surrogate_training(session))
            self.assertEqual(len(results), 2)
            self.assertEqual(session.config.data.kind, "energy_bank")


if __name__ == "__main__":
    unittest.main()
