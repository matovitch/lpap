"""Load frozen surrogate/decoder checkpoints for AE (and similar) stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lpap.checkpoints import load_training_checkpoint
from lpap.decoder import LPAPDecoderTransformer
from lpap.decoder_training import _surrogate_model_config_from_checkpoint
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
    saved_permutation: torch.Tensor | None = None
    raw_perm = (payload.get("training_state") or {}).get("permutation")
    if raw_perm is not None:
        saved_permutation = torch.as_tensor(raw_perm).long()
    return surrogate, model_config, saved_permutation


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
) -> tuple[LPAPDecoderTransformer, dict[str, object]]:
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
    return decoder, model_config


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


__all__ = [
    "load_decoder_source",
    "load_surrogate_source",
    "resolve_checkpoint_path",
    "validate_lpap_pair_matches_sequence_length",
]
