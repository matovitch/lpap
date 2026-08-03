from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.decoder_training import (
    LPAPDecoderModelConfig,
    LPAPDecoderRunConfig,
    LPAPDecoderTeacherConfig,
    LPAPDecoderTrainingConfig,
    create_lpap_decoder_training_session,
)
from lpap.energy_bank import EnergyBankConfig
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import LPAPSurrogateDataConfig
from lpap.teacher_config import project_teacher_config


class TeacherConfigProjectionTest(unittest.TestCase):
    def test_project_from_repo_toml(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "configs/training/teacher_c128_k16.toml"
        surrogate = project_teacher_config(path, "surrogate")
        decoder = project_teacher_config(path, "decoder")
        self.assertEqual(surrogate.run.run_id, "surrogate_c128_k16")
        self.assertEqual(decoder.run.run_id, "decoder_c128_k16")
        self.assertEqual(surrogate.run.permutation_seed, 128)
        self.assertEqual(decoder.run.permutation_seed, 128)
        self.assertTrue(decoder.teacher.require_checkpoint)


class DecoderRequiresSurrogatePermutationTest(unittest.TestCase):
    def test_missing_permutation_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bank = root / "data" / "bank.pt"
            bank.parent.mkdir(parents=True)
            torch.save({"energies": torch.randn(8, 16)}, bank)
            surrogate = LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            save_training_checkpoint(
                root / "checkpoints" / "surrogate.pt",
                model=surrogate,
                step=1,
                training_state={
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
                    energy_bank=EnergyBankConfig(path="data/bank.pt"),
                ),
                decoder=LPAPDecoderModelConfig(
                    frontend_initial_temperature=0.5,
                    hidden_dim=16,
                    layer_count=1,
                    head_count=4,
                ),
                teacher=LPAPDecoderTeacherConfig(
                    checkpoint_name="surrogate.pt",
                    require_checkpoint=True,
                ),
                run=LPAPDecoderRunConfig(steps=1, run_id="tiny"),
            )
            with self.assertRaisesRegex(ValueError, "training_state.permutation"):
                create_lpap_decoder_training_session(
                    project_root=root, config=config, device="cpu"
                )

    def test_mismatched_permutation_seed_still_loads_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bank = root / "data" / "bank.pt"
            bank.parent.mkdir(parents=True)
            torch.save({"energies": torch.randn(8, 16)}, bank)
            permutation = make_grouped_permutation_indices(
                value_count=16, bucket_count=4, seed=999, device=torch.device("cpu")
            )
            surrogate = LPAPSurrogateTransformer(
                value_count=16,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            save_training_checkpoint(
                root / "checkpoints" / "surrogate.pt",
                model=surrogate,
                step=1,
                training_state={
                    "model_config": {
                        "value_count": 16,
                        "bucket_count": 4,
                        "probe_count": 4,
                        "k_max": 2,
                        "hidden_dim": 16,
                        "layer_count": 1,
                        "head_count": 4,
                        "permutation_seed": 999,
                    },
                    "permutation": permutation,
                },
            )
            config = LPAPDecoderTrainingConfig(
                data=LPAPSurrogateDataConfig(
                    batch_size=2,
                    bucket_count=4,
                    probe_count=4,
                    energy_bank=EnergyBankConfig(path="data/bank.pt"),
                ),
                decoder=LPAPDecoderModelConfig(
                    frontend_initial_temperature=0.5,
                    hidden_dim=16,
                    layer_count=1,
                    head_count=4,
                ),
                teacher=LPAPDecoderTeacherConfig(
                    checkpoint_name="surrogate.pt",
                    require_checkpoint=True,
                ),
                run=LPAPDecoderRunConfig(
                    steps=1,
                    run_id="tiny",
                    permutation_seed=123,
                ),
            )
            session = create_lpap_decoder_training_session(
                project_root=root, config=config, device="cpu"
            )
            self.assertTrue(torch.equal(session.permutation.cpu(), permutation))


if __name__ == "__main__":
    unittest.main()
