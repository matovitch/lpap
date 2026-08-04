"""Migrate AE checkpoints to self-contained ``lpap_pairs[{name, permutation}]``.

Salvage path for AE checkpoints that stored layouts as
``training_state.lpap_pair_permutations`` (or lacked them). Rewrites
``training_state.lpap_pairs`` to ``[{name, permutation}, ...]``, drops legacy
keys / seeds, and leaves weights / optimizer / step untouched.

Example::

    PYTHONPATH=src python -m lpap.migrate_ae_permutations \\
      --checkpoint checkpoints/image_autoencoder_tri_lnorm.pt
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
from lpap.permutation import as_long_permutation, make_grouped_permutation_indices
from lpap.teacher_checkpoints import (
    lpap_pair_training_state_record,
    parse_ae_lpap_pair_permutations,
)

# Fallback names / geometry when rebuilding from seeds (tri-pair AE).
_TRI_PAIR_FALLBACK: tuple[tuple[str, int, int], ...] = (
    ("c128_k16", 128, 123),
    ("c256_k24", 256, 256),
    ("c512_k32", 512, 512),
)


def _value_count_from_model_config(model_config: dict[str, Any]) -> int:
    value_count = int(
        model_config.get("sequence_length")
        or model_config.get("value_count")
        or 0
    )
    if value_count <= 0:
        raise ValueError(
            "AE model_config.sequence_length (or value_count) must be positive"
        )
    return value_count


def _pair_names_from_training_state(training_state: dict[str, Any]) -> list[str]:
    raw = training_state.get("lpap_pair_names")
    if isinstance(raw, list | tuple) and raw:
        return [str(name) for name in raw]
    model_config = training_state.get("model_config")
    if isinstance(model_config, dict):
        pairs = model_config.get("lpap_pairs")
        if isinstance(pairs, list | tuple) and pairs:
            names: list[str] = []
            for entry in pairs:
                if isinstance(entry, dict) and entry.get("name") is not None:
                    names.append(str(entry["name"]))
            if names:
                return names
        legacy = model_config.get("lpap_pair_names")
        if isinstance(legacy, list | tuple) and legacy:
            return [str(name) for name in legacy]
    return [name for name, _buckets, _seed in _TRI_PAIR_FALLBACK]


def _legacy_permutations(
    training_state: dict[str, Any],
    *,
    pair_count: int,
    value_count: int,
) -> list[torch.Tensor] | None:
    raw = training_state.get("lpap_pair_permutations")
    if raw is None:
        return None
    if not isinstance(raw, list | tuple):
        raise ValueError("lpap_pair_permutations must be a list")
    if len(raw) != pair_count:
        raise ValueError(
            f"lpap_pair_permutations length must be {pair_count}, got {len(raw)}"
        )
    return [
        as_long_permutation(torch.as_tensor(item), value_count=value_count)
        for item in raw
    ]


def _seed_permutations_from_model_config(
    model_config: dict[str, Any],
    *,
    pair_names: list[str],
    value_count: int,
) -> list[torch.Tensor]:
    pairs = model_config.get("lpap_pairs")
    if isinstance(pairs, list | tuple) and len(pairs) == len(pair_names):
        permutations: list[torch.Tensor] = []
        for name, entry in zip(pair_names, pairs, strict=True):
            if not isinstance(entry, dict):
                raise ValueError(f"model_config.lpap_pairs entry for {name} must be dict")
            surrogate = entry.get("surrogate")
            if not isinstance(surrogate, dict):
                raise ValueError(f"missing surrogate config for pair {name}")
            bucket_count = int(surrogate["bucket_count"])
            seed = int(
                surrogate.get("permutation_seed")
                or entry.get("permutation_seed")
                or 0
            )
            if seed <= 0:
                raise ValueError(f"missing permutation_seed for pair {name}")
            permutations.append(
                make_grouped_permutation_indices(
                    value_count=value_count,
                    bucket_count=bucket_count,
                    seed=seed,
                    device="cpu",
                )
            )
        return permutations

    surrogates = model_config.get("lpap_pair_surrogates")
    if isinstance(surrogates, list | tuple) and len(surrogates) == len(pair_names):
        permutations = []
        for name, surrogate in zip(pair_names, surrogates, strict=True):
            if not isinstance(surrogate, dict):
                raise ValueError(f"lpap_pair_surrogates entry for {name} must be dict")
            permutations.append(
                make_grouped_permutation_indices(
                    value_count=value_count,
                    bucket_count=int(surrogate["bucket_count"]),
                    seed=int(surrogate["permutation_seed"]),
                    device="cpu",
                )
            )
        return permutations

    # Hard-coded tri-pair salvage when model_config lacks seed-bearing pair blocks.
    if len(pair_names) == len(_TRI_PAIR_FALLBACK) and [
        name for name, *_ in _TRI_PAIR_FALLBACK
    ] == pair_names:
        return [
            make_grouped_permutation_indices(
                value_count=value_count,
                bucket_count=buckets,
                seed=seed,
                device="cpu",
            )
            for _name, buckets, seed in _TRI_PAIR_FALLBACK
        ]
    raise ValueError(
        "cannot rebuild AE permutations: no legacy tensors and no pair seed metadata"
    )


def migrate_ae_checkpoint_permutations(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rewrite AE ``training_state.lpap_pairs`` to ``[{name, permutation}, ...]``."""
    source = Path(checkpoint_path)
    destination = Path(output_path) if output_path is not None else source
    payload = load_training_checkpoint(source, map_location="cpu")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("AE checkpoint training_state must be a dictionary")
    model_config = training_state.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("AE checkpoint training_state.model_config is required")
    value_count = _value_count_from_model_config(model_config)
    pair_names = _pair_names_from_training_state(training_state)
    if not pair_names:
        raise ValueError("AE checkpoint has no pair names to migrate")

    source_kind = "legacy_lpap_pair_permutations"
    permutations = _legacy_permutations(
        training_state, pair_count=len(pair_names), value_count=value_count
    )
    if permutations is None:
        # Already new-shaped?
        raw_pairs = training_state.get("lpap_pairs")
        if (
            isinstance(raw_pairs, list | tuple)
            and raw_pairs
            and isinstance(raw_pairs[0], dict)
            and raw_pairs[0].get("permutation") is not None
        ):
            permutations = parse_ae_lpap_pair_permutations(
                training_state,
                pair_count=len(raw_pairs),
                value_count=value_count,
            )
            pair_names = [
                str(entry.get("name", f"pair{index}"))
                for index, entry in enumerate(raw_pairs)
            ]
            source_kind = "existing_lpap_pairs"
        else:
            permutations = _seed_permutations_from_model_config(
                model_config, pair_names=pair_names, value_count=value_count
            )
            source_kind = "model_config_seeds"

    updated = dict(training_state)
    updated["lpap_pairs"] = [
        lpap_pair_training_state_record(
            name=name, permutation=perm, value_count=value_count
        )
        for name, perm in zip(pair_names, permutations, strict=True)
    ]
    for key in (
        "lpap_pair_permutations",
        "lpap_pair_names",
        "surrogate_checkpoint_paths",
        "decoder_checkpoint_paths",
        "surrogate_checkpoint_path",
        "decoder_checkpoint_path",
    ):
        updated.pop(key, None)

    # Validate new contract.
    parse_ae_lpap_pair_permutations(
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
        "pair_names": pair_names,
        "value_count": value_count,
        "source": source_kind,
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
    summary = migrate_ae_checkpoint_permutations(
        args.checkpoint, output_path=args.output
    )
    print(
        "migrated AE permutations: "
        f"step={summary['step']} best={summary['best_metric']} "
        f"pairs={summary['pair_names']} source={summary['source']} "
        f"-> {summary['output_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
