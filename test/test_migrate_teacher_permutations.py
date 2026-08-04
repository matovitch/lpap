from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lpap.checkpoints import (
    load_training_checkpoint,
    write_training_checkpoint_payload,
)
from lpap.migrate_teacher_permutations import (
    TRI_PAIR_TEACHER_CHECKPOINTS,
    assert_tri_pair_teacher_permutations_match,
    migrate_teacher_checkpoint_permutation,
    migrate_teacher_checkpoints_permutations,
)
from lpap.permutation import as_long_permutation, make_grouped_permutation_indices
from lpap.teacher_checkpoints import require_matching_pair_permutation


class MigrateTeacherPermutationsTest(unittest.TestCase):
    def test_migrate_writes_permutation_without_touching_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "surrogate.pt"
            model = nn.Linear(2, 1)
            original_weight = model.weight.detach().clone()
            wrong = torch.arange(16)
            write_training_checkpoint_payload(
                path,
                {
                    "step": 11,
                    "best_metric": 0.5,
                    "model_state": model.state_dict(),
                    "best_model_state": model.state_dict(),
                    "optimizer_state": None,
                    "training_state": {
                        "run_config": {"run": {"permutation_seed": 123}},
                        "model_config": {
                            "value_count": 16,
                            "bucket_count": 4,
                        },
                        "permutation": wrong,
                    },
                },
            )
            summary = migrate_teacher_checkpoint_permutation(
                path, permutation_seed=123
            )
            payload = load_training_checkpoint(path, map_location="cpu")
            expected = make_grouped_permutation_indices(
                value_count=16, bucket_count=4, seed=123, device="cpu"
            )
            torch.testing.assert_close(
                payload["training_state"]["permutation"], expected
            )
            self.assertNotIn("permutation_seed", payload["training_state"])
            self.assertNotIn(
                "permutation_seed", payload["training_state"]["model_config"]
            )
            torch.testing.assert_close(
                payload["model_state"]["weight"], original_weight
            )
            self.assertEqual(summary["permutation_seed"], 123)
            self.assertEqual(payload["step"], 11)

    def test_tri_pair_migrate_and_assert_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt_dir = root / "checkpoints"
            ckpt_dir.mkdir()
            specs = [
                ("surrogate_c128_k16.pt", 128, 123),
                ("decoder_c128_k16.pt", 128, 123),
                ("surrogate_c256_k24.pt", 256, 256),
                ("decoder_c256_k24.pt", 256, 256),
                ("surrogate_c512_k32.pt", 512, 512),
                ("decoder_c512_k32.pt", 512, 512),
            ]
            self.assertEqual(
                [name for name, *_ in specs], list(TRI_PAIR_TEACHER_CHECKPOINTS)
            )
            for name, buckets, seed in specs:
                model = nn.Linear(2, 1)
                write_training_checkpoint_payload(
                    ckpt_dir / name,
                    {
                        "step": 1,
                        "best_metric": 0.1,
                        "model_state": model.state_dict(),
                        "best_model_state": model.state_dict(),
                        "optimizer_state": None,
                        "training_state": {
                            "run_config": {"run": {"permutation_seed": seed}},
                            "model_config": {
                                "value_count": 1024,
                                "bucket_count": buckets,
                            },
                            "permutation": torch.arange(1024),
                        },
                    },
                )
            paths = [ckpt_dir / name for name in TRI_PAIR_TEACHER_CHECKPOINTS]
            seeds = [seed for _, _, seed in specs]
            migrate_teacher_checkpoints_permutations(paths, permutation_seeds=seeds)
            assert_tri_pair_teacher_permutations_match(project_root=root)
            for name, buckets, seed in specs:
                payload = load_training_checkpoint(
                    ckpt_dir / name, map_location="cpu"
                )
                expected = make_grouped_permutation_indices(
                    value_count=1024,
                    bucket_count=buckets,
                    seed=seed,
                    device="cpu",
                )
                torch.testing.assert_close(
                    payload["training_state"]["permutation"], expected
                )


class TeacherCheckpointPermutationTest(unittest.TestCase):
    def test_require_matching_pair_permutation(self) -> None:
        perm = torch.arange(8)
        matched = require_matching_pair_permutation(
            surrogate_permutation=perm,
            decoder_permutation=perm.clone(),
            value_count=8,
        )
        torch.testing.assert_close(matched, as_long_permutation(perm, value_count=8))
        with self.assertRaisesRegex(ValueError, "do not match"):
            require_matching_pair_permutation(
                surrogate_permutation=perm,
                decoder_permutation=torch.flip(perm, dims=(0,)),
                value_count=8,
                pair_name="c8",
            )


if __name__ == "__main__":
    unittest.main()
