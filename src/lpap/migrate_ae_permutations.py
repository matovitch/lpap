"""Migrate AE checkpoints to store concrete ``lpap_pair_permutations``.

Salvage path for runs that trained under CPU-seed ``make_grouped_permutation``
layouts but never persisted those tensors. Regenerates permutations from each
pair's ``permutation_seed`` / ``bucket_count`` in ``training_state.model_config``
using the device-stable CPU RNG, writes them into the checkpoint, and leaves
weights / optimizer / step untouched.

Example::

    PYTHONPATH=src python -m lpap.migrate_ae_permutations \\
      --checkpoint checkpoints/image_autoencoder_tri_lnorm.pt \\
      --output checkpoints/image_autoencoder_tri_lnorm.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from lpap.checkpoints import (
    load_training_checkpoint,
    write_training_checkpoint_payload,
)
from lpap.image_autoencoder_training import parse_lpap_pair_permutations
from lpap.permutation import make_grouped_permutation_indices


def _require_model_config(training_state: dict[str, Any]) -> dict[str, Any]:
    model_config = training_state.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("AE checkpoint training_state.model_config is required")
    return model_config


def cpu_seed_permutations_from_ae_model_config(
    model_config: dict[str, Any],
) -> tuple[list[str], list[torch.Tensor]]:
    """Build pair names + CPU-seed perms from AE ``model_config``."""
    pair_names = model_config.get("lpap_pair_names")
    pair_surrogates = model_config.get("lpap_pair_surrogates")
    if not isinstance(pair_names, list | tuple) or not pair_names:
        raise ValueError("model_config.lpap_pair_names must be a non-empty list")
    if not isinstance(pair_surrogates, list | tuple):
        raise ValueError("model_config.lpap_pair_surrogates must be a list")
    if len(pair_names) != len(pair_surrogates):
        raise ValueError(
            "lpap_pair_names and lpap_pair_surrogates lengths must match"
        )
    value_count = int(
        model_config.get("sequence_length")
        or model_config.get("value_count")
        or 0
    )
    if value_count <= 0:
        raise ValueError(
            "model_config.sequence_length (or value_count) must be positive"
        )

    permutations: list[torch.Tensor] = []
    names: list[str] = []
    for name, surrogate_cfg in zip(pair_names, pair_surrogates, strict=True):
        if not isinstance(surrogate_cfg, dict):
            raise ValueError("each lpap_pair_surrogates entry must be a dict")
        bucket_count = int(surrogate_cfg["bucket_count"])
        seed = int(surrogate_cfg["permutation_seed"])
        if value_count % bucket_count != 0:
            raise ValueError(
                f"value_count {value_count} not divisible by bucket_count "
                f"{bucket_count} for pair {name}"
            )
        permutations.append(
            make_grouped_permutation_indices(
                value_count=value_count,
                bucket_count=bucket_count,
                seed=seed,
                device="cpu",
            )
        )
        names.append(str(name))
    return names, permutations


def migrate_ae_checkpoint_permutations_from_cpu_seeds(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Embed CPU-seed ``lpap_pair_permutations`` into an AE checkpoint.

    Returns a small summary dict (paths, step, pair names).
    """
    source = Path(checkpoint_path)
    destination = Path(output_path) if output_path is not None else source
    payload = load_training_checkpoint(source, map_location="cpu")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("AE checkpoint training_state must be a dictionary")
    model_config = _require_model_config(training_state)
    pair_names, permutations = cpu_seed_permutations_from_ae_model_config(
        model_config
    )
    value_count = int(
        model_config.get("sequence_length") or model_config.get("value_count")
    )
    updated = dict(training_state)
    updated["lpap_pair_names"] = pair_names
    updated["lpap_pair_permutations"] = permutations
    # Validate shape contract used by resume.
    parse_lpap_pair_permutations(
        updated, pair_count=len(pair_names), value_count=value_count
    )
    payload = dict(payload)
    payload["training_state"] = updated
    write_training_checkpoint_payload(destination, payload)
    return {
        "checkpoint_path": str(source),
        "output_path": str(destination),
        "step": payload.get("step"),
        "best_metric": payload.get("best_metric"),
        "lpap_pair_names": pair_names,
        "value_count": value_count,
        "permutation_seeds": [
            int(cfg["permutation_seed"])
            for cfg in model_config["lpap_pair_surrogates"]
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="input AE checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path (default: overwrite --checkpoint)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = migrate_ae_checkpoint_permutations_from_cpu_seeds(
        args.checkpoint, output_path=args.output
    )
    print(
        "migrated AE permutations: "
        f"step={summary['step']} best={summary['best_metric']} "
        f"pairs={summary['lpap_pair_names']} "
        f"seeds={summary['permutation_seeds']} "
        f"-> {summary['output_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
