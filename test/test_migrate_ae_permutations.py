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
from lpap.image_autoencoder_training import parse_lpap_pair_permutations
from lpap.migrate_ae_permutations import (
    cpu_seed_permutations_from_ae_model_config,
    migrate_ae_checkpoint_permutations_from_cpu_seeds,
)
from lpap.permutation import make_grouped_permutation_indices


class MigrateAePermutationsTest(unittest.TestCase):
    def test_cpu_seed_permutations_from_model_config(self) -> None:
        model_config = {
            "sequence_length": 16,
            "lpap_pair_names": ["c4", "c8"],
            "lpap_pair_surrogates": [
                {"bucket_count": 4, "permutation_seed": 123},
                {"bucket_count": 8, "permutation_seed": 456},
            ],
        }
        names, perms = cpu_seed_permutations_from_ae_model_config(model_config)
        self.assertEqual(names, ["c4", "c8"])
        self.assertEqual(len(perms), 2)
        torch.testing.assert_close(
            perms[0],
            make_grouped_permutation_indices(
                value_count=16, bucket_count=4, seed=123, device="cpu"
            ),
        )
        torch.testing.assert_close(
            perms[1],
            make_grouped_permutation_indices(
                value_count=16, bucket_count=8, seed=456, device="cpu"
            ),
        )

    def test_migrate_writes_lpap_pair_permutations_without_touching_weights(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "ae.pt"
            model = nn.Linear(2, 1)
            with torch.no_grad():
                model.weight.fill_(2.0)
                model.bias.fill_(-1.0)
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            payload = {
                "step": 32000,
                "epoch": 32000,
                "metrics": {},
                "metric_name": "validation_loss",
                "mode": "min",
                "best_metric": 0.0083,
                "model_state": state,
                "best_model_state": state,
                "optimizer_state": None,
                "training_state": {
                    "model_config": {
                        "sequence_length": 16,
                        "lpap_pair_names": ["c4"],
                        "lpap_pair_surrogates": [
                            {"bucket_count": 4, "permutation_seed": 123}
                        ],
                    },
                    "lpap_pair_names": ["c4"],
                },
            }
            write_training_checkpoint_payload(path, payload)

            out = root / "ae_migrated.pt"
            summary = migrate_ae_checkpoint_permutations_from_cpu_seeds(
                path, output_path=out
            )
            self.assertEqual(summary["step"], 32000)
            self.assertEqual(summary["lpap_pair_names"], ["c4"])

            migrated = load_training_checkpoint(out, map_location="cpu")
            perms = parse_lpap_pair_permutations(
                migrated["training_state"], pair_count=1, value_count=16
            )
            expected = make_grouped_permutation_indices(
                value_count=16, bucket_count=4, seed=123, device="cpu"
            )
            torch.testing.assert_close(perms[0], expected)
            torch.testing.assert_close(
                migrated["model_state"]["weight"], state["weight"]
            )
            self.assertEqual(migrated["best_metric"], 0.0083)


if __name__ == "__main__":
    unittest.main()
