from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


CheckpointMode = Literal["min", "max"]


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int
    metric_name: str | None
    current_metric: float | None
    best_metric: float | None
    improved: bool


def _state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def metric_improved(
    current: float,
    best: float | None,
    *,
    mode: CheckpointMode,
) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError("mode must be 'min' or 'max'")


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    metric_name: str | None = None,
    best_metric: float | None = None,
    best_model_state: dict[str, torch.Tensor] | None = None,
    mode: CheckpointMode = "min",
    training_state: dict[str, Any] | None = None,
) -> CheckpointInfo:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_metric = (
        None if metric_name is None or metrics is None else metrics[metric_name]
    )
    improved = (
        False
        if current_metric is None
        else metric_improved(float(current_metric), best_metric, mode=mode)
    )
    current_model_state = _state_dict_to_cpu(model.state_dict())
    stored_best_model_state = (
        current_model_state
        if improved or best_model_state is None
        else _state_dict_to_cpu(best_model_state)
    )
    stored_best_metric = float(current_metric) if improved else best_metric

    payload: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "metrics": {} if metrics is None else dict(metrics),
        "metric_name": metric_name,
        "mode": mode,
        "best_metric": stored_best_metric,
        "model_state": current_model_state,
        "best_model_state": stored_best_model_state,
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "training_state": {} if training_state is None else dict(training_state),
    }
    torch.save(payload, checkpoint_path)
    return CheckpointInfo(
        path=checkpoint_path,
        step=step,
        metric_name=metric_name,
        current_metric=None if current_metric is None else float(current_metric),
        best_metric=stored_best_metric,
        improved=improved,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    load_best: bool = False,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    if model is not None:
        state_key = "best_model_state" if load_best else "model_state"
        model.load_state_dict(payload[state_key])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
