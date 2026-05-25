from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lpap.data import SyntheticHarmonicConfig
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import (
    LPAPSurrogateMetrics,
    LPAPSurrogateTransformer,
    evaluate_lpap_surrogate_batch,
    train_lpap_surrogate_step,
)
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)


@dataclass(frozen=True)
class LPAPSurrogateDataConfig:
    batch_size: int = 32
    bucket_count: int = 64
    probe_count: int = 16
    harmonics: SyntheticHarmonicConfig = field(default_factory=SyntheticHarmonicConfig)

    @property
    def value_count(self) -> int:
        return self.bucket_count * self.probe_count

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if self.probe_count <= 0:
            raise ValueError("probe_count must be positive")
        self.harmonics.validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "bucket_count": self.bucket_count,
            "probe_count": self.probe_count,
            "value_count": self.value_count,
            "harmonics": self.harmonics.as_dict(),
        }


@dataclass(frozen=True)
class LPAPSurrogateModelConfig:
    k_max: int = 4
    hidden_dim: int = 128
    layer_count: int = 4
    head_count: int = 4

    def validate(self) -> None:
        if self.k_max <= 0:
            raise ValueError("k_max must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self.head_count <= 0:
            raise ValueError("head_count must be positive")
        if self.hidden_dim % self.head_count != 0:
            raise ValueError("hidden_dim must be divisible by head_count")

    def as_dict(self) -> dict[str, int]:
        return {
            "k_max": self.k_max,
            "hidden_dim": self.hidden_dim,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
        }


@dataclass(frozen=True)
class LPAPSurrogateOptimizerConfig:
    learning_rate: float = 1.0e-3

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def as_dict(self) -> dict[str, float]:
        return {"learning_rate": self.learning_rate}


@dataclass(frozen=True)
class LPAPSurrogateValidationConfig:
    enabled: bool = True
    every: int = 100
    batch_size: int = 256
    seed: int = 10_123
    validate_at_end: bool = True

    def validate(self) -> None:
        if self.every <= 0:
            raise ValueError("validation every must be positive")
        if self.batch_size <= 0:
            raise ValueError("validation batch_size must be positive")

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "every": self.every,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "validate_at_end": self.validate_at_end,
        }


@dataclass(frozen=True)
class LPAPSurrogateRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 123
    permutation_seed: int = 123
    display_every: int = 5
    log_every: int = 1
    run_id: str = "surrogate_synthetic"
    checkpoint_name: str = "surrogate_synthetic.pt"
    log_name: str = "surrogate.sqlite"

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.display_every <= 0 or self.log_every <= 0:
            raise ValueError("display/log cadence values must be positive")

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "run_training": self.run_training,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "steps": self.steps,
            "seed": self.seed,
            "permutation_seed": self.permutation_seed,
            "display_every": self.display_every,
            "log_every": self.log_every,
            "run_id": self.run_id,
            "checkpoint_name": self.checkpoint_name,
            "log_name": self.log_name,
        }


@dataclass(frozen=True)
class LPAPSurrogateTrainingConfig:
    data: LPAPSurrogateDataConfig = field(default_factory=LPAPSurrogateDataConfig)
    model: LPAPSurrogateModelConfig = field(default_factory=LPAPSurrogateModelConfig)
    optimizer: LPAPSurrogateOptimizerConfig = field(
        default_factory=LPAPSurrogateOptimizerConfig
    )
    validation: LPAPSurrogateValidationConfig = field(
        default_factory=LPAPSurrogateValidationConfig
    )
    run: LPAPSurrogateRunConfig = field(default_factory=LPAPSurrogateRunConfig)

    @property
    def value_count(self) -> int:
        return self.data.value_count

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.run.validate()

    def as_run_config(self) -> dict[str, object]:
        return {
            "data": self.data.as_dict(),
            "model": self.model.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(self) -> dict[str, int]:
        return {
            "value_count": self.value_count,
            "bucket_count": self.data.bucket_count,
            "probe_count": self.data.probe_count,
            "k_max": self.model.k_max,
            "hidden_dim": self.model.hidden_dim,
            "layer_count": self.model.layer_count,
            "head_count": self.model.head_count,
            "permutation_seed": self.run.permutation_seed,
        }


@dataclass(frozen=True)
class LPAPSurrogateTrainingSession:
    config: LPAPSurrogateTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    permutation: torch.Tensor
    model: LPAPSurrogateTransformer
    optimizer: torch.optim.Optimizer
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo


def create_lpap_surrogate_training_session(
    *,
    project_root: str | Path,
    config: LPAPSurrogateTrainingConfig,
    device: str | torch.device | None = None,
) -> LPAPSurrogateTrainingSession:
    config.validate()
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    torch.manual_seed(config.run.seed)
    root = Path(project_root)
    checkpoint_path = root / "checkpoints" / config.run.checkpoint_name
    log_path = root / "training_logs" / config.run.log_name
    permutation = make_grouped_permutation_indices(
        value_count=config.value_count,
        bucket_count=config.data.bucket_count,
        seed=config.run.permutation_seed,
        device=target_device,
    )
    model = LPAPSurrogateTransformer(
        probe_count=config.data.probe_count,
        k_max=config.model.k_max,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        head_count=config.model.head_count,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    training_run = TrainingRun(
        config=TrainingRunConfig(
            run_id=config.run.run_id,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            total_steps=config.run.steps,
            monitor="validation_loss",
            mode="min",
            resume=config.run.resume_from_checkpoint,
            checkpoint_every=None,
            checkpoint_on_improvement=True,
            checkpoint_at_end=False,
            log_every=config.run.log_every,
            display_every=config.run.display_every,
        ),
        model=model,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(),
        metadata={"device": str(target_device)},
    )
    resume_info = training_run.resume_or_initialize()
    generator = torch.Generator(device=target_device).manual_seed(
        config.run.seed + resume_info.start_step
    )
    validation_generator = torch.Generator(device=target_device).manual_seed(
        config.validation.seed + resume_info.start_step
    )
    return LPAPSurrogateTrainingSession(
        config=config,
        device=target_device,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        permutation=permutation,
        model=model,
        optimizer=optimizer,
        training_run=training_run,
        generator=generator,
        validation_generator=validation_generator,
        resume_info=resume_info,
    )


def validate_lpap_surrogate(
    session: LPAPSurrogateTrainingSession,
) -> LPAPSurrogateMetrics:
    config = session.config
    batch = config.data.harmonics.sample_batch(
        batch_size=config.validation.batch_size,
        n=config.value_count,
        generator=session.validation_generator,
        device=session.device,
    )
    return evaluate_lpap_surrogate_batch(
        model=session.model,
        values=batch,
        bucket_count=config.data.bucket_count,
        k_max=config.model.k_max,
        permutation=session.permutation,
    )


def should_validate_lpap_surrogate(
    *, step: int, config: LPAPSurrogateTrainingConfig
) -> bool:
    return config.validation.enabled and (
        step % config.validation.every == 0
        or (config.validation.validate_at_end and step == config.run.steps)
    )


def iter_lpap_surrogate_training(
    session: LPAPSurrogateTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    for step in range(session.resume_info.start_step, config.run.steps + 1):
        batch = config.data.harmonics.sample_batch(
            batch_size=config.data.batch_size,
            n=config.value_count,
            generator=session.generator,
            device=session.device,
        )
        metrics = train_lpap_surrogate_step(
            model=session.model,
            optimizer=session.optimizer,
            values=batch,
            bucket_count=config.data.bucket_count,
            k_max=config.model.k_max,
            permutation=session.permutation,
        )
        step_metrics = {
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "weighted_accuracy": metrics.weighted_accuracy,
            "mean_weight": metrics.mean_weight,
        }
        if should_validate_lpap_surrogate(step=step, config=config):
            validation_metrics = validate_lpap_surrogate(session)
            step_metrics.update(
                {
                    "validation_loss": validation_metrics.loss,
                    "validation_accuracy": validation_metrics.accuracy,
                    "validation_weighted_accuracy": validation_metrics.weighted_accuracy,
                    "validation_mean_weight": validation_metrics.mean_weight,
                }
            )
        yield session.training_run.record_step(
            step=step,
            epoch=step,
            metrics=step_metrics,
            training_state={
                "seed": config.run.seed,
                "permutation_seed": config.run.permutation_seed,
                "validation_seed": config.validation.seed,
                "permutation": session.permutation.detach().cpu(),
            },
        )

    session.training_run.mark_finished()
