from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lpap.checkpoints import (
    load_training_checkpoint,
    metric_improved,
    save_training_checkpoint,
)


class CheckpointTest(unittest.TestCase):
    def test_metric_improved_handles_min_and_max(self) -> None:
        self.assertTrue(metric_improved(0.5, None, mode="min"))
        self.assertTrue(metric_improved(0.4, 0.5, mode="min"))
        self.assertFalse(metric_improved(0.6, 0.5, mode="min"))
        self.assertTrue(metric_improved(0.6, 0.5, mode="max"))
        self.assertFalse(metric_improved(0.4, 0.5, mode="max"))

    def test_checkpoint_stores_current_and_best_model_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            model = nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

            with torch.no_grad():
                model.weight.fill_(1.0)
                model.bias.fill_(0.25)
            first = save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=1,
                metrics={"loss": 0.5},
                metric_name="loss",
                mode="min",
                training_state={
                    "seen": 16,
                    "permutation_seed": 123,
                    "permutation": torch.tensor([2, 0, 1]),
                },
            )
            self.assertTrue(first.improved)

            first_payload = load_training_checkpoint(path)
            with torch.no_grad():
                model.weight.fill_(2.0)
                model.bias.fill_(0.5)
            second = save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=2,
                metrics={"loss": 0.7},
                metric_name="loss",
                best_metric=first_payload["best_metric"],
                best_model_state=first_payload["best_model_state"],
                mode="min",
                training_state=first_payload["training_state"],
            )
            self.assertFalse(second.improved)

            payload = load_training_checkpoint(path)
            self.assertEqual(payload["training_state"]["permutation_seed"], 123)
            torch.testing.assert_close(
                payload["training_state"]["permutation"], torch.tensor([2, 0, 1])
            )
            torch.testing.assert_close(
                payload["model_state"]["weight"], torch.full((1, 2), 2.0)
            )
            torch.testing.assert_close(
                payload["best_model_state"]["weight"], torch.full((1, 2), 1.0)
            )
            self.assertEqual(payload["best_metric"], 0.5)

            restored_best = nn.Linear(2, 1)
            load_training_checkpoint(path, model=restored_best, load_best=True)
            torch.testing.assert_close(restored_best.weight, torch.full((1, 2), 1.0))


if __name__ == "__main__":
    unittest.main()
