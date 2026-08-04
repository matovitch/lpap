"""Load frozen surrogate/decoder checkpoints for AE (and similar) stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lpap.checkpoints import load_training_checkpoint
from lpap.decoder import LPAPDecoderTransformer
from lpap.decoder_training import _surrogate_model_config_from_checkpoint
from lpap.permutation import as_long_permutation
from lpap.surrogate import LPAPSurrogateTransformer


def resolve_checkpoint_path(
    root: Path,
    name: str,
    *,
    ensure: bool = False,
) -> Path:
    path = Path(name)
    resolved = path if path.is_absolute() else root / "checkpoints" / path
    if ensure and not resolved.is_file():
        from lpap.artifact_sync import ensure_project_artifact

        ensure_project_artifact(resolved, project_root=root)
    return resolved


def permutation_from_training_state(
    training_state: dict[str, Any] | None,
    *,
    value_count: int,
    path: Path,
    role: str,
) -> torch.Tensor:
    """Require ``training_state.permutation`` (surrogate or decoder schema)."""
    if not isinstance(training_state, dict):
        raise ValueError(
            f"{role} checkpoint training_state must be a dictionary: {path}"
        )
    raw = training_state.get("permutation")
    if raw is None:
        raise ValueError(
            f"{role} checkpoint is missing training_state.permutation: {path}"
        )
    return as_long_permutation(torch.as_tensor(raw), value_count=value_count)


def require_matching_pair_permutation(
    *,
    surrogate_permutation: torch.Tensor,
    decoder_permutation: torch.Tensor,
    value_count: int,
    surrogate_path: Path | str | None = None,
    decoder_path: Path | str | None = None,
    pair_name: str | None = None,
) -> torch.Tensor:
    """Assert surrogate/decoder ``permutation`` tensors match; return one clone."""
    left = as_long_permutation(surrogate_permutation, value_count=value_count)
    right = as_long_permutation(decoder_permutation, value_count=value_count)
    if not torch.equal(left, right):
        where = []
        if pair_name is not None:
            where.append(f"pair={pair_name}")
        if surrogate_path is not None:
            where.append(f"surrogate={surrogate_path}")
        if decoder_path is not None:
            where.append(f"decoder={decoder_path}")
        suffix = f" ({', '.join(where)})" if where else ""
        raise ValueError(
            "surrogate and decoder training_state.permutation do not match" + suffix
        )
    return left


def load_surrogate_source(
    *,
    path: Path,
    load_best: bool,
    require_checkpoint: bool,
    device: torch.device,
) -> tuple[LPAPSurrogateTransformer, dict[str, int], torch.Tensor | None]:
    model_config, payload = _surrogate_model_config_from_checkpoint(
        path=path, require_checkpoint=require_checkpoint
    )
    if model_config is None or payload is None:
        raise FileNotFoundError(f"surrogate checkpoint not found: {path}")
    surrogate = LPAPSurrogateTransformer(
        value_count=model_config["value_count"],
        probe_count=model_config["probe_count"],
        k_max=model_config["k_max"],
        hidden_dim=model_config["hidden_dim"],
        layer_count=model_config["layer_count"],
        head_count=model_config["head_count"],
    ).to(device)
    state_key = "best_model_state" if load_best else "model_state"
    surrogate.load_state_dict(payload[state_key])
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    training_state = payload.get("training_state")
    if (
        not isinstance(training_state, dict)
        or training_state.get("permutation") is None
    ):
        if require_checkpoint:
            raise ValueError(
                "surrogate checkpoint is missing training_state.permutation: "
                f"{path}"
            )
        return surrogate, model_config, None
    return (
        surrogate,
        model_config,
        as_long_permutation(
            torch.as_tensor(training_state["permutation"]),
            value_count=model_config["value_count"],
        ),
    )


def _decoder_model_config_from_checkpoint(
    path: Path,
) -> tuple[dict[str, object], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"decoder checkpoint not found: {path}")
    payload = load_training_checkpoint(path)
    training_state = payload.get("training_state", {})
    if not isinstance(training_state, dict):
        raise ValueError("decoder checkpoint training_state must be a dictionary")
    model_config = training_state.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("decoder checkpoint is missing training_state.model_config")
    required = {
        "value_count",
        "bucket_count",
        "probe_count",
        "frontend_initial_temperature",
        "hidden_dim",
        "layer_count",
        "head_count",
    }
    missing = sorted(required.difference(model_config))
    if missing:
        raise ValueError(
            "decoder checkpoint model_config is missing: " + ", ".join(missing)
        )
    return dict(model_config), payload


def load_decoder_source(
    *,
    path: Path,
    load_best: bool,
    device: torch.device,
    require_permutation: bool = True,
) -> tuple[LPAPDecoderTransformer, dict[str, object], torch.Tensor | None]:
    model_config, payload = _decoder_model_config_from_checkpoint(path)
    decoder = LPAPDecoderTransformer(
        value_count=int(model_config["value_count"]),
        frontend_initial_temperature=float(
            model_config["frontend_initial_temperature"]
        ),
        hidden_dim=int(model_config["hidden_dim"]),
        layer_count=int(model_config["layer_count"]),
        head_count=int(model_config["head_count"]),
    ).to(device)
    state_key = "best_model_state" if load_best else "model_state"
    decoder.load_state_dict(payload[state_key])
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    training_state = payload.get("training_state")
    if (
        not isinstance(training_state, dict)
        or training_state.get("permutation") is None
    ):
        if require_permutation:
            raise ValueError(
                "decoder checkpoint is missing training_state.permutation: "
                f"{path}"
            )
        return decoder, model_config, None
    return (
        decoder,
        model_config,
        as_long_permutation(
            torch.as_tensor(training_state["permutation"]),
            value_count=int(model_config["value_count"]),
        ),
    )


def validate_lpap_pair_matches_sequence_length(
    *,
    sequence_length: int,
    surrogate_model_config: dict[str, int],
    decoder_model_config: dict[str, object],
) -> None:
    expected = {
        "value_count": sequence_length,
        "bucket_count": int(decoder_model_config["bucket_count"]),
        "probe_count": int(decoder_model_config["probe_count"]),
    }
    mismatches = [
        f"{name} surrogate={surrogate_model_config[name]} expected={value}"
        for name, value in expected.items()
        if surrogate_model_config[name] != value
    ]
    decoder_value_count = int(decoder_model_config["value_count"])
    if decoder_value_count != sequence_length:
        mismatches.append(
            f"value_count decoder={decoder_value_count} expected={sequence_length}"
        )
    if mismatches:
        raise ValueError(
            "LPAP pair checkpoints do not match sequence length: "
            + "; ".join(mismatches)
        )


def lpap_pair_model_config_record(
    *,
    name: str,
    surrogate: dict[str, Any],
    decoder: dict[str, Any],
) -> dict[str, Any]:
    """Pair entry for AE ``model_config.lpap_pairs`` (no seed fields)."""
    return {
        "name": name,
        "surrogate": _without_permutation_seed(surrogate),
        "decoder": _without_permutation_seed(decoder),
    }


def lpap_pair_training_state_record(
    *,
    name: str,
    permutation: torch.Tensor,
    value_count: int,
) -> dict[str, Any]:
    """Self-contained pair layout for AE ``training_state.lpap_pairs``."""
    from lpap.permutation import as_long_permutation

    return {
        "name": name,
        "permutation": as_long_permutation(permutation, value_count=value_count),
    }


def parse_ae_lpap_pair_permutations(
    training_state: dict[str, Any],
    *,
    pair_count: int,
    value_count: int,
) -> list[torch.Tensor]:
    """Require AE ``training_state.lpap_pairs[*].permutation``."""
    from lpap.permutation import as_long_permutation

    raw_pairs = training_state.get("lpap_pairs")
    if not isinstance(raw_pairs, list | tuple):
        raise ValueError("AE checkpoint training_state.lpap_pairs must be a list")
    if len(raw_pairs) != pair_count:
        raise ValueError(
            f"AE lpap_pairs length must be {pair_count}, got {len(raw_pairs)}"
        )
    permutations: list[torch.Tensor] = []
    for index, entry in enumerate(raw_pairs):
        if not isinstance(entry, dict):
            raise ValueError(f"AE lpap_pairs[{index}] must be a dict")
        raw = entry.get("permutation")
        if raw is None:
            raise ValueError(
                f"AE lpap_pairs[{index}] is missing permutation "
                f"(pair={entry.get('name')!r})"
            )
        permutations.append(
            as_long_permutation(torch.as_tensor(raw), value_count=value_count)
        )
    return permutations


def apply_ae_lpap_pair_permutations(
    pairs: tuple[Any, ...] | list[Any],
    permutations: list[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[Any, ...]:
    """Replace in-memory pair layouts with AE-checkpoint permutations."""
    if len(pairs) != len(permutations):
        raise ValueError("pairs and permutations lengths must match")
    updated = []
    for runtime, permutation in zip(pairs, permutations, strict=True):
        updated.append(
            type(runtime)(
                name=runtime.name,
                surrogate_checkpoint_path=runtime.surrogate_checkpoint_path,
                decoder_checkpoint_path=runtime.decoder_checkpoint_path,
                permutation=permutation.to(device=device, dtype=torch.long),
                surrogate_model_config=runtime.surrogate_model_config,
                decoder_model_config=runtime.decoder_model_config,
            )
        )
    return tuple(updated)


def _without_permutation_seed(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_permutation_seed(item)
            for key, item in value.items()
            if key != "permutation_seed"
        }
    if isinstance(value, list):
        return [_without_permutation_seed(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_permutation_seed(item) for item in value)
    return value


__all__ = [
    "apply_ae_lpap_pair_permutations",
    "load_decoder_source",
    "load_surrogate_source",
    "lpap_pair_model_config_record",
    "lpap_pair_training_state_record",
    "parse_ae_lpap_pair_permutations",
    "permutation_from_training_state",
    "require_matching_pair_permutation",
    "resolve_checkpoint_path",
    "validate_lpap_pair_matches_sequence_length",
]
