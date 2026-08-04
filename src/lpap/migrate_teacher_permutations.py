"""Rewrite teacher ``training_state.permutation`` from ``model_config`` seeds.

One-shot salvage when a surrogate/decoder checkpoint has a stale or
device-dependent layout tensor. Rebuilds the concrete ``permutation`` from
``model_config`` (``permutation_seed`` + ``bucket_count`` + ``value_count``)
using the device-stable seed builder, then writes it back. Weights / optimizer
/ step are untouched.

Prefer stored ``training_state.permutation`` everywhere else — do not use this
as a normal load path.

Tri-pair AE teachers (all six)::

    PYTHONPATH=src python -m lpap.migrate_teacher_permutations \\
      --tri-pair --project-root .
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
from lpap.teacher_checkpoints import require_matching_pair_permutation

# Surrogate + decoder for each pair used by image_autoencoder_tri_lnorm.
TRI_PAIR_TEACHER_CHECKPOINTS: tuple[str, ...] = (
    "surrogate_c128_k16.pt",
    "decoder_c128_k16.pt",
    "surrogate_c256_k24.pt",
    "decoder_c256_k24.pt",
    "surrogate_c512_k32.pt",
    "decoder_c512_k32.pt",
)

TRI_PAIR_TEACHER_PAIRS: tuple[tuple[str, str], ...] = (
    ("surrogate_c128_k16.pt", "decoder_c128_k16.pt"),
    ("surrogate_c256_k24.pt", "decoder_c256_k24.pt"),
    ("surrogate_c512_k32.pt", "decoder_c512_k32.pt"),
)


def permutation_from_teacher_model_config(
    model_config: dict[str, Any],
    *,
    permutation_seed: int,
) -> torch.Tensor:
    """Build a layout tensor from teacher geometry + a salvage seed."""
    value_count = int(
        model_config.get("value_count") or model_config.get("sequence_length") or 0
    )
    bucket_count = int(model_config.get("bucket_count") or 0)
    if value_count <= 0:
        raise ValueError("model_config.value_count must be positive")
    if bucket_count <= 0:
        raise ValueError("model_config.bucket_count must be positive")
    if value_count % bucket_count != 0:
        raise ValueError(
            f"value_count {value_count} not divisible by bucket_count {bucket_count}"
        )
    return make_grouped_permutation_indices(
        value_count=value_count,
        bucket_count=bucket_count,
        seed=permutation_seed,
        device="cpu",
    )


def _permutation_seed_from_training_state(training_state: dict[str, Any]) -> int:
    """Salvage seed: prefer run_config (TOML mirror), else legacy model_config."""
    run_config = training_state.get("run_config")
    if isinstance(run_config, dict):
        run = run_config.get("run")
        if isinstance(run, dict) and "permutation_seed" in run:
            return int(run["permutation_seed"])
    model_config = training_state.get("model_config")
    if isinstance(model_config, dict) and "permutation_seed" in model_config:
        return int(model_config["permutation_seed"])
    top = training_state.get("permutation_seed")
    if top is not None:
        return int(top)
    raise ValueError(
        "cannot migrate permutation: no permutation_seed in run_config.run, "
        "model_config, or training_state"
    )


def migrate_teacher_checkpoint_permutation(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    permutation_seed: int | None = None,
) -> dict[str, Any]:
    """Write ``training_state.permutation`` rebuilt from geometry + salvage seed."""
    source = Path(checkpoint_path)
    destination = Path(output_path) if output_path is not None else source
    payload = load_training_checkpoint(source, map_location="cpu")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint training_state must be a dictionary")
    model_config = training_state.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("training_state.model_config is required")
    seed = (
        int(permutation_seed)
        if permutation_seed is not None
        else _permutation_seed_from_training_state(training_state)
    )
    permutation = permutation_from_teacher_model_config(
        model_config, permutation_seed=seed
    )
    value_count = int(model_config["value_count"])
    permutation = as_long_permutation(permutation, value_count=value_count)
    updated = dict(training_state)
    updated["permutation"] = permutation
    updated.pop("permutation_seed", None)
    if isinstance(updated.get("model_config"), dict):
        cleaned = dict(updated["model_config"])
        cleaned.pop("permutation_seed", None)
        if isinstance(cleaned.get("surrogate"), dict):
            surrogate = dict(cleaned["surrogate"])
            surrogate.pop("permutation_seed", None)
            cleaned["surrogate"] = surrogate
        updated["model_config"] = cleaned
    payload = dict(payload)
    payload["training_state"] = updated
    write_training_checkpoint_payload(destination, payload)
    return {
        "checkpoint_path": str(source),
        "output_path": str(destination),
        "step": payload.get("step"),
        "best_metric": payload.get("best_metric"),
        "value_count": value_count,
        "bucket_count": int(model_config["bucket_count"]),
        "permutation_seed": seed,
    }


def migrate_teacher_checkpoints_permutations(
    checkpoint_paths: list[Path] | tuple[Path, ...],
    *,
    permutation_seeds: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Migrate each teacher checkpoint in place."""
    if permutation_seeds is not None and len(permutation_seeds) != len(checkpoint_paths):
        raise ValueError("permutation_seeds length must match checkpoint_paths")
    summaries: list[dict[str, Any]] = []
    for index, path in enumerate(checkpoint_paths):
        seed = None if permutation_seeds is None else int(permutation_seeds[index])
        summaries.append(
            migrate_teacher_checkpoint_permutation(path, permutation_seed=seed)
        )
    return summaries



def assert_tri_pair_teacher_permutations_match(
    *,
    project_root: str | Path,
) -> None:
    """Require each surrogate/decoder pair under ``checkpoints/`` to match."""
    root = Path(project_root)
    for surrogate_name, decoder_name in TRI_PAIR_TEACHER_PAIRS:
        surrogate_path = root / "checkpoints" / surrogate_name
        decoder_path = root / "checkpoints" / decoder_name
        surrogate_payload = load_training_checkpoint(
            surrogate_path, map_location="cpu"
        )
        decoder_payload = load_training_checkpoint(decoder_path, map_location="cpu")
        surrogate_state = surrogate_payload.get("training_state") or {}
        decoder_state = decoder_payload.get("training_state") or {}
        if not isinstance(surrogate_state, dict) or not isinstance(decoder_state, dict):
            raise ValueError("teacher training_state must be a dictionary")
        value_count = int(
            (surrogate_state.get("model_config") or {}).get("value_count") or 0
        )
        require_matching_pair_permutation(
            surrogate_permutation=torch.as_tensor(surrogate_state["permutation"]),
            decoder_permutation=torch.as_tensor(decoder_state["permutation"]),
            value_count=value_count,
            surrogate_path=surrogate_path,
            decoder_path=decoder_path,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="input surrogate/decoder checkpoint (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path when migrating a single --checkpoint (default: overwrite)",
    )
    parser.add_argument(
        "--tri-pair",
        action="store_true",
        help=(
            "migrate all six tri-pair teacher checkpoints under "
            "<project-root>/checkpoints/"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="project root containing checkpoints/ (for --tri-pair)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override salvage permutation seed (else read from run_config / legacy fields)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.tri_pair:
        if args.checkpoint or args.output is not None:
            raise SystemExit(
                "migrate_teacher_permutations: use either --tri-pair or "
                "--checkpoint, not both"
            )
        paths = [
            args.project_root / "checkpoints" / name
            for name in TRI_PAIR_TEACHER_CHECKPOINTS
        ]
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"missing teacher checkpoint: {path}")
        # Default tri-pair seeds match current teacher TOMLs when ckpts lack seed.
        default_seeds = (123, 123, 256, 256, 512, 512)
        seeds = default_seeds if args.seed is None else (args.seed,) * len(paths)
        summaries = migrate_teacher_checkpoints_permutations(
            paths, permutation_seeds=seeds
        )
        for summary in summaries:
            print(
                "migrated teacher permutation: "
                f"step={summary['step']} best={summary['best_metric']} "
                f"C={summary['bucket_count']} seed={summary['permutation_seed']} "
                f"-> {summary['output_path']}"
            )
        assert_tri_pair_teacher_permutations_match(project_root=args.project_root)
        print("tri-pair surrogate/decoder permutations match")
        return 0

    checkpoints = args.checkpoint or []
    if not checkpoints:
        raise SystemExit(
            "migrate_teacher_permutations: pass --checkpoint (repeatable) "
            "or --tri-pair"
        )
    if args.output is not None and len(checkpoints) != 1:
        raise SystemExit("--output requires exactly one --checkpoint")
    for checkpoint in checkpoints:
        summary = migrate_teacher_checkpoint_permutation(
            checkpoint,
            output_path=args.output if len(checkpoints) == 1 else None,
            permutation_seed=args.seed,
        )
        print(
            "migrated teacher permutation: "
            f"step={summary['step']} best={summary['best_metric']} "
            f"C={summary['bucket_count']} seed={summary['permutation_seed']} "
            f"-> {summary['output_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
