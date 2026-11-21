from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from lpap.checkpoints import load_training_checkpoint
from lpap.energy_bank import EnergyBankConfig
from lpap.permutation import LPAP_FOLD_ORDER
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

            payload = load_training_checkpoint(session.checkpoint_path)
            training_state = payload["training_state"]
            self.assertEqual(training_state["fold_order"], LPAP_FOLD_ORDER)
            self.assertEqual(
                training_state["model_config"]["fold_order"], LPAP_FOLD_ORDER
            )
            self.assertTrue(
                torch.equal(training_state["permutation"], session.permutation.cpu())
            )

    def test_resume_reads_stored_permutation_ignoring_config_seed(self) -> None:
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
                    steps=1,
                    permutation_seed=111,
                    display_every=1,
                    run_id="tiny-surrogate",
                ),
                validation=LPAPSurrogateValidationConfig(every=1, batch_size=3),
            )
            first = create_lpap_surrogate_training_session(
                project_root=root, config=config, device="cpu"
            )
            list(iter_lpap_surrogate_training(first))
            saved_perm = first.permutation.cpu().clone()

            resumed_cfg = replace(
                config,
                run=replace(
                    config.run,
                    steps=2,
                    resume_from_checkpoint=True,
                    permutation_seed=999,
                ),
            )
            resumed = create_lpap_surrogate_training_session(
                project_root=root, config=resumed_cfg, device="cpu"
            )
            self.assertTrue(resumed.resume_info.resumed)
            torch.testing.assert_close(resumed.permutation.cpu(), saved_perm)


if __name__ == "__main__":
    unittest.main()
