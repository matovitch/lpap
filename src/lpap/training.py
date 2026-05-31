from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lpap.checkpoints import (
    CheckpointMode,
    load_training_checkpoint,
    metric_improved,
    save_training_checkpoint,
)
from lpap.training_log import log_step_metrics, mark_run_status, upsert_run
from lpap.training_log import finish_run_attempt, start_run_attempt
from lpap.training_log import (
    make_run_display_name,
    make_run_instance_id,
    prune_run_history,
)


def _state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


@dataclass(frozen=True)
class TrainingRunConfig:
    run_id: str
    checkpoint_path: Path
    log_path: Path
    total_steps: int
    monitor: str = "loss"
    mode: CheckpointMode = "min"
    resume: bool = True
    checkpoint_every: int | None = 25
    checkpoint_on_improvement: bool = False
    checkpoint_at_end: bool = True
    log_every: int = 1
    display_every: int = 5
    keep_last_runs: int = 10
    note: str = ""
    tags: tuple[str, ...] = ()
    pinned: bool = False


@dataclass(frozen=True)
class TrainingResumeInfo:
    run_id: str
    base_run_id: str
    display_name: str
    attempt_id: int
    start_step: int
    resumed: bool
    message: str


@dataclass(frozen=True)
class TrainingStepResult:
    step: int
    metrics: dict[str, float]
    best_metric: float | None
    improved: bool
    checkpointed: bool
    logged: bool
    should_display: bool


class TrainingRun:
    def __init__(
        self,
        *,
        config: TrainingRunConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        run_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.run_config = {} if run_config is None else dict(run_config)
        self.model_config = {} if model_config is None else dict(model_config)
        self.metadata = {} if metadata is None else dict(metadata)
        self.base_run_id = config.run_id
        self.run_id = config.run_id
        self.display_name = config.run_id
        self.best_metric: float | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.start_step = 1
        self.attempt_id: int | None = None

    def resume_or_initialize(self) -> TrainingResumeInfo:
        resumed = False
        message = "starting fresh"
        checkpoint_step: int | None = None
        if self.config.resume and self.config.checkpoint_path.exists():
            payload = load_training_checkpoint(
                self.config.checkpoint_path, map_location="cpu"
            )
            training_state = payload.get("training_state", {})
            checkpoint_model_config = training_state.get("model_config")
            if (
                checkpoint_model_config is not None
                and checkpoint_model_config != self.model_config
            ):
                message = "checkpoint model config differs; starting fresh"
            else:
                self.model.load_state_dict(payload["model_state"])
                if (
                    self.optimizer is not None
                    and payload.get("optimizer_state") is not None
                ):
                    self.optimizer.load_state_dict(payload["optimizer_state"])
                self.best_metric = payload["best_metric"]
                self.best_model_state = payload["best_model_state"]
                checkpoint_step = int(payload["step"])
                self.start_step = checkpoint_step + 1
                self.run_id = training_state.get("run_id", self.run_id)
                checkpoint_metadata = training_state.get("metadata", {})
                if isinstance(checkpoint_metadata, dict):
                    self.display_name = str(
                        checkpoint_metadata.get(
                            "display_name", self.run_id.split(":", 1)[-1]
                        )
                    )
                resumed = True
                message = f"resumed from step `{checkpoint_step}`"
        if not resumed:
            self.display_name = make_run_display_name()
            self.run_id = make_run_instance_id(
                self.base_run_id, display_name=self.display_name
            )

        run_metadata = {
            **self.metadata,
            "base_run_id": self.base_run_id,
            "display_name": self.display_name,
            "note": self.config.note,
            "tags": list(self.config.tags),
            "pinned": self.config.pinned,
            "resumed": resumed,
            "start_step": self.start_step,
        }
        upsert_run(
            self.config.log_path,
            run_id=self.run_id,
            checkpoint_path=self.config.checkpoint_path,
            config=self.run_config,
            metadata=run_metadata,
        )
        self.attempt_id = start_run_attempt(
            self.config.log_path,
            run_id=self.run_id,
            resumed=resumed,
            start_step=self.start_step,
            checkpoint_step=checkpoint_step,
            message=message,
            metadata=run_metadata,
        )
        prune_run_history(
            self.config.log_path,
            base_run_id=self.base_run_id,
            keep_last=self.config.keep_last_runs,
        )
        return TrainingResumeInfo(
            run_id=self.run_id,
            base_run_id=self.base_run_id,
            display_name=self.display_name,
            attempt_id=self.attempt_id,
            start_step=self.start_step,
            resumed=resumed,
            message=message,
        )

    def record_step(
        self,
        *,
        step: int,
        metrics: dict[str, float],
        epoch: int | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> TrainingStepResult:
        current_metric = metrics.get(self.config.monitor)
        previous_best_metric = self.best_metric
        improved = current_metric is not None and metric_improved(
            current_metric,
            previous_best_metric,
            mode=self.config.mode,
        )
        if improved:
            self.best_metric = current_metric
            self.best_model_state = _state_dict_to_cpu(self.model.state_dict())

        checkpointed = (
            (improved and self.config.checkpoint_on_improvement)
            or (
                self.config.checkpoint_every is not None
                and step % self.config.checkpoint_every == 0
            )
            or (self.config.checkpoint_at_end and step == self.config.total_steps)
        )
        if checkpointed:
            checkpoint_info = save_training_checkpoint(
                self.config.checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                step=step,
                epoch=step if epoch is None else epoch,
                metrics=metrics,
                metric_name=self.config.monitor if current_metric is not None else None,
                best_metric=previous_best_metric if improved else self.best_metric,
                best_model_state=self.best_model_state,
                mode=self.config.mode,
                training_state={
                    "run_id": self.run_id,
                    "log_path": str(self.config.log_path),
                    "run_config": self.run_config,
                    "model_config": self.model_config,
                    "metadata": {
                        "base_run_id": self.base_run_id,
                        "display_name": self.display_name,
                        "note": self.config.note,
                        "tags": list(self.config.tags),
                        "pinned": self.config.pinned,
                    },
                    **({} if training_state is None else dict(training_state)),
                },
            )
            payload = load_training_checkpoint(self.config.checkpoint_path)
            self.best_metric = payload["best_metric"]
            self.best_model_state = payload["best_model_state"]
            improved = checkpoint_info.improved

        logged = step % self.config.log_every == 0 or step == self.config.total_steps
        if logged:
            log_step_metrics(
                self.config.log_path,
                run_id=self.run_id,
                attempt_id=self.attempt_id,
                step=step,
                epoch=step if epoch is None else epoch,
                metrics=metrics,
                best_metric_name=self.config.monitor,
                best_metric=self.best_metric,
                improved=improved,
            )

        return TrainingStepResult(
            step=step,
            metrics=dict(metrics),
            best_metric=self.best_metric,
            improved=improved,
            checkpointed=checkpointed,
            logged=logged,
            should_display=step % self.config.display_every == 0
            or step == self.config.total_steps,
        )

    def mark_finished(self) -> None:
        mark_run_status(self.config.log_path, run_id=self.run_id, status="finished")
        if self.attempt_id is not None:
            finish_run_attempt(
                self.config.log_path, attempt_id=self.attempt_id, status="finished"
            )
