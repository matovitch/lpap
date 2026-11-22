from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import load_training_checkpoint, save_training_checkpoint
from lpap.decoder_training import (
    LPAPDecoderModelConfig,
    LPAPDecoderRegularizationConfig,
    LPAPDecoderRunConfig,
    LPAPDecoderTeacherConfig,
    LPAPDecoderTrainingConfig,
    collect_lpap_decoder_gallery,
    create_lpap_decoder_training_session,
    iter_lpap_decoder_training,
)
from lpap.energy_bank import EnergyBankConfig
from lpap.permutation import LPAP_FOLD_ORDER, make_permutation_indices
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
    def test_session_trains_logs_and_renders_gallery(self) -> None:
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
            permutation = make_permutation_indices(
                value_count=16, seed=123, device=torch.device("cpu")
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
                        "fold_order": LPAP_FOLD_ORDER,
                    },
                    "fold_order": LPAP_FOLD_ORDER,
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
            gallery = collect_lpap_decoder_gallery(session, sample_count=2)

            self.assertEqual(len(results), 2)
            self.assertEqual(len(gallery), 2)
            self.assertEqual(gallery[0].surrogate_hard.shape, (4, 4))
            self.assertTrue(session.checkpoint_path.exists())
            self.assertTrue(session.log_path.exists())
            self.assertEqual(session.energy_bank.shape[-1], 16)
            self.assertIn("loss", results[-1].metrics)
            self.assertIn("validation_loss", results[-1].metrics)
            self.assertTrue(any(result.improved for result in results))

            payload = load_training_checkpoint(session.checkpoint_path)
            self.assertEqual(payload["training_state"]["fold_order"], LPAP_FOLD_ORDER)

            # Fine-tune the trained decoder on exact LPAP without any surrogate.
            exact_config = LPAPDecoderTrainingConfig(
                data=config.data,
                decoder=config.decoder,
                validation=config.validation,
                teacher=LPAPDecoderTeacherConfig(
                    frontend_mode="exact",
                    exact_k_max=2,
                    init_decoder_checkpoint_name="decoder_synthetic.pt",
                    require_checkpoint=False,
                ),
                regularization=LPAPDecoderRegularizationConfig(source_ce_weight=0.0),
                run=LPAPDecoderRunConfig(
                    steps=2,
                    resume_from_checkpoint=False,
                    display_every=1,
                    run_id="tiny-decoder-exact",
                    checkpoint_name="decoder_exact.pt",
                    log_name="decoder_exact.sqlite",
                ),
            )
            exact_session = create_lpap_decoder_training_session(
                project_root=root, config=exact_config, device="cpu"
            )
            self.assertIsNone(exact_session.surrogate)
            torch.testing.assert_close(exact_session.permutation.cpu(), permutation)
            exact_results = list(iter_lpap_decoder_training(exact_session))
            self.assertEqual(len(exact_results), 2)
            self.assertEqual(exact_results[-1].metrics["mean_entropy"], 0.0)

            # Standalone exact-mode decoder (no init checkpoint, no surrogate) uses
            # permutation_seed matching Stage-1 surrogate's permutation.
            standalone_exact_config = LPAPDecoderTrainingConfig(
                data=config.data,
                decoder=config.decoder,
                validation=config.validation,
                teacher=LPAPDecoderTeacherConfig(
                    frontend_mode="exact",
                    exact_k_max=2,
                    require_checkpoint=False,
                ),
                regularization=LPAPDecoderRegularizationConfig(source_ce_weight=0.1),
                run=LPAPDecoderRunConfig(
                    steps=1,
                    permutation_seed=123,
                    resume_from_checkpoint=False,
                    display_every=1,
                    run_id="tiny-decoder-exact-fresh",
                    checkpoint_name="decoder_exact_fresh.pt",
                    log_name="decoder_exact_fresh.sqlite",
                ),
            )
            standalone_session = create_lpap_decoder_training_session(
                project_root=root, config=standalone_exact_config, device="cpu"
            )
            torch.testing.assert_close(
                standalone_session.permutation.cpu(), permutation
            )
            standalone_results = list(iter_lpap_decoder_training(standalone_session))
            self.assertEqual(len(standalone_results), 1)
            self.assertEqual(standalone_results[-1].metrics["mean_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
