from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
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
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateValidationConfig,
)


def _write_tiny_energy_bank(root: Path, *, rows: int = 32, dim: int = 16) -> None:
    path = root / "data" / "encoded_energies_ae_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"energies": torch.randn(rows, dim)}, path)


class LPAPDecoderTrainingTest(unittest.TestCase):
    def test_session_trains_and_logs_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_energy_bank(root, dim=16)
            surrogate = LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            permutation = make_grouped_permutation_indices(
                value_count=16, bucket_count=4, seed=123, device=torch.device("cpu")
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
                            energy_bank=EnergyBankConfig(
                                path="data/encoded_energies_ae_best.pt"
                            ),
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
                    "permutation": permutation,
                },
            )
            config = LPAPDecoderTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=4,
                    probe_count=4,
                    energy_bank=EnergyBankConfig(path="data/encoded_energies_ae_best.pt"),
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
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertEqual(session.energy_bank.shape[-1], 16)
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertTrue(any(result.improved for result in results))


if __name__ == "__main__":
    unittest.main()
