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
from lpap.migrate_ae_permutations import migrate_ae_checkpoint_permutations
from lpap.permutation import make_permutation_indices
from lpap.teacher_checkpoints import parse_ae_lpap_pair_permutations


class MigrateAePermutationsTest(unittest.TestCase):
    def test_legacy_list_to_lpap_pairs_without_touching_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ae.pt"
            model = nn.Linear(2, 1)
            original = model.weight.detach().clone()
            perm_a = make_permutation_indices(
                value_count=16, seed=123, device="cpu",
)
            perm_b = make_permutation_indices(
                value_count=16, seed=456, device="cpu",
)
            write_training_checkpoint_payload(
                path,
                {
                    "step": 32000,
                    "best_metric": 0.008,
                    "model_state": model.state_dict(),
                    "best_model_state": model.state_dict(),
                    "optimizer_state": None,
                    "training_state": {
                        "model_config": {
                            "sequence_length": 16,
                            "lpap_pair_names": ["c4", "c8"],
                            "lpap_pair_surrogates": [
                                {
                                    "value_count": 16,
                                    "bucket_count": 4,
                                    "k_max": 2,
                                    "permutation_seed": 123,
                                },
                                {
                                    "value_count": 16,
                                    "bucket_count": 8,
                                    "k_max": 3,
                                    "permutation_seed": 456,
                                },
                            ],
                            "lpap_pair_decoders": [
                                {
                                    "value_count": 16,
                                    "bucket_count": 4,
                                    "probe_count": 2,
                                },
                                {
                                    "value_count": 16,
                                    "bucket_count": 8,
                                    "probe_count": 3,
                                },
                            ],
                            "surrogate": {"value_count": 16, "bucket_count": 4},
                            "decoder": {"value_count": 16, "bucket_count": 4},
                        },
                        "lpap_pair_names": ["c4", "c8"],
                        "lpap_pair_permutations": [perm_a, perm_b],
                        "surrogate_checkpoint_path": "checkpoints/old.pt",
                        "permutation_seed": 999,
                    },
                },
            )
            summary = migrate_ae_checkpoint_permutations(path)
            payload = load_training_checkpoint(path, map_location="cpu")
            ts = payload["training_state"]
            self.assertEqual(summary["source"], "legacy_lpap_pair_permutations")
            self.assertNotIn("lpap_pair_permutations", ts)
            self.assertNotIn("surrogate_checkpoint_path", ts)
            self.assertNotIn("permutation_seed", ts)
            model_config = ts["model_config"]
            self.assertNotIn("lpap_pair_surrogates", model_config)
            self.assertNotIn("lpap_pair_decoders", model_config)
            self.assertNotIn("lpap_pair_names", model_config)
            self.assertNotIn("surrogate", model_config)
            self.assertNotIn("decoder", model_config)
            self.assertEqual(
                [entry["name"] for entry in model_config["lpap_pairs"]],
                ["c4", "c8"],
            )
            self.assertNotIn(
                "permutation_seed", model_config["lpap_pairs"][0]["surrogate"]
            )
            perms = parse_ae_lpap_pair_permutations(
                ts, pair_count=2, value_count=16
            )
            torch.testing.assert_close(perms[0], perm_a)
            torch.testing.assert_close(perms[1], perm_b)
            self.assertEqual(
                [entry["name"] for entry in ts["lpap_pairs"]], ["c4", "c8"]
            )
            torch.testing.assert_close(payload["model_state"]["weight"], original)
            self.assertEqual(payload["step"], 32000)


if __name__ == "__main__":
    unittest.main()
