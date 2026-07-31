from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

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


def _write_tiny_energy_bank(root: Path, *, rows: int = 32, dim: int = 16) -> None:
    path = root / "data" / "encoded_energies_ae_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"energies": torch.randn(rows, dim)}, path)


class LPAPSurrogateTrainingTest(unittest.TestCase):
    def test_session_trains_and_logs_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_energy_bank(root, dim=16)
            config = LPAPSurrogateTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=4,
                    probe_count=4,
                    energy_bank=EnergyBankConfig(path="data/encoded_energies_ae_best.pt"),
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
                project_root=root, config=config, device="cpu"
            )

            results = list(iter_lpap_surrogate_training(session))

            self.assertEqual(len(results), 2)
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertTrue(any(result.improved for result in results))


if __name__ == "__main__":
    unittest.main()
