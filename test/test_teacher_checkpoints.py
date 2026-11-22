from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lpap.checkpoints import save_training_checkpoint
from lpap.decoder import LPAPDecoderTransformer
from lpap.permutation import LPAP_FOLD_ORDER, make_permutation_indices
from lpap.teacher_checkpoints import (
    load_decoder_source,
    validate_lpap_pair_matches_sequence_length,
)


def _surrogate_cfg(**overrides: object) -> dict[str, int | str]:
    base: dict[str, int | str] = {
        "value_count": 16,
        "bucket_count": 4,
        "probe_count": 4,
        "k_max": 2,
        "fold_order": LPAP_FOLD_ORDER,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _decoder_cfg(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "value_count": 16,
        "bucket_count": 4,
        "probe_count": 4,
        "k_max": 2,
        "fold_order": LPAP_FOLD_ORDER,
        "frontend_mode": "surrogate",
        "surrogate": _surrogate_cfg(),
    }
    base.update(overrides)
    return base


class TeacherCheckpointsTest(unittest.TestCase):
    def test_validate_pair_accepts_matching_k_max_and_frontend_schema(self) -> None:
        validate_lpap_pair_matches_sequence_length(
            sequence_length=16,
            surrogate_model_config=_surrogate_cfg(),
            decoder_model_config=_decoder_cfg(),
        )

    def test_validate_pair_rejects_k_max_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "k_max"):
            validate_lpap_pair_matches_sequence_length(
                sequence_length=16,
                surrogate_model_config=_surrogate_cfg(k_max=2),
                decoder_model_config=_decoder_cfg(k_max=3),
            )

    def test_validate_pair_rejects_missing_k_max(self) -> None:
        decoder_cfg = _decoder_cfg()
        del decoder_cfg["k_max"]
        with self.assertRaisesRegex(ValueError, "missing k_max"):
            validate_lpap_pair_matches_sequence_length(
                sequence_length=16,
                surrogate_model_config=_surrogate_cfg(),
                decoder_model_config=decoder_cfg,
            )

    def test_validate_pair_rejects_embedded_surrogate_schema_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, r"decoder\.surrogate\.k_max"):
            validate_lpap_pair_matches_sequence_length(
                sequence_length=16,
                surrogate_model_config=_surrogate_cfg(k_max=2),
                decoder_model_config=_decoder_cfg(
                    k_max=2,
                    surrogate=_surrogate_cfg(k_max=4),
                ),
            )

    def test_validate_pair_rejects_unsupported_frontend_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "frontend_mode"):
            validate_lpap_pair_matches_sequence_length(
                sequence_length=16,
                surrogate_model_config=_surrogate_cfg(),
                decoder_model_config=_decoder_cfg(frontend_mode="bogus"),
            )

    def test_validate_pair_rejects_fold_order_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "fold_order"):
            validate_lpap_pair_matches_sequence_length(
                sequence_length=16,
                surrogate_model_config=_surrogate_cfg(fold_order="wrong"),
                decoder_model_config=_decoder_cfg(),
            )

    def test_load_decoder_source_requires_k_max_in_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "decoder.pt"
            decoder = LPAPDecoderTransformer(
                bucket_count=4,
                probe_count=4,
                k_max=2,
                hidden_dim=16,
                layer_count=1,
                head_count=4,
            )
            save_training_checkpoint(
                path,
                model=decoder,
                step=1,
                training_state={
                    "model_config": {
                        "value_count": 16,
                        "bucket_count": 4,
                        "probe_count": 4,
                        "hidden_dim": 16,
                        "layer_count": 1,
                        "head_count": 4,
                        "fold_order": LPAP_FOLD_ORDER,
                    },
                    "permutation": make_permutation_indices(value_count=16, seed=1),
                },
            )
            with self.assertRaisesRegex(ValueError, "k_max"):
                load_decoder_source(
                    path=path, load_best=False, device=torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()
