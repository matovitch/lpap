from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lpap.energy_bank import (
    EnergyBankConfig,
    cycle_energy_bank_batches,
    energy_bank_config_from_dict,
    ensure_energy_bank,
    sample_energy_bank_values,
)
from lpap.permutation import make_permutation_indices
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
from lpap.training_log import load_run_record


@dataclass(frozen=True)
class LPAPSurrogateDataConfig:
    """LPAP teacher data: sample energies from an empirical i2e bank."""

    batch_size: int = 32
    bucket_count: int = 64
    probe_count: int = 16
    energy_bank: EnergyBankConfig = field(default_factory=EnergyBankConfig)

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
        self.energy_bank.validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "bucket_count": self.bucket_count,
            "probe_count": self.probe_count,
            "value_count": self.value_count,
            "energy_bank": self.energy_bank.as_dict(),
        }


def load_teacher_energy_bank(
    root: str | Path,
    data: LPAPSurrogateDataConfig,
) -> torch.Tensor:
    """Load the energy bank and check ``energy_dim == data.value_count``."""
    return ensure_energy_bank(
        Path(root),
        data.energy_bank,
        energy_dim=data.value_count,
    )


def sample_teacher_energy_batch(
    energy_bank: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return sample_energy_bank_values(
        energy_bank,
        batch_size=batch_size,
        generator=generator,
        device=device,
    )


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
    comment: str = ""
    pinned: bool = False

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
            "comment": self.comment,
            "pinned": self.pinned,
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
        }


def lpap_surrogate_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> LPAPSurrogateTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return LPAPSurrogateTrainingConfig(
        data=LPAPSurrogateDataConfig(
            batch_size=int(data["data"]["batch_size"]),
            bucket_count=int(data["data"]["bucket_count"]),
            probe_count=int(data["data"]["probe_count"]),
            energy_bank=energy_bank_config_from_dict(data["data"]["energy_bank"]),
        ),
        model=LPAPSurrogateModelConfig(
            k_max=int(data["model"]["k_max"]),
            hidden_dim=int(data["model"]["hidden_dim"]),
            layer_count=int(data["model"]["layer_count"]),
            head_count=int(data["model"]["head_count"]),
        ),
        optimizer=LPAPSurrogateOptimizerConfig(
            learning_rate=float(data["optimizer"]["learning_rate"])
        ),
        validation=LPAPSurrogateValidationConfig(
            enabled=bool(data["validation"]["enabled"]),
            every=int(data["validation"]["every"]),
            batch_size=int(data["validation"]["batch_size"]),
            seed=int(data["validation"]["seed"]),
            validate_at_end=bool(data["validation"]["validate_at_end"]),
        ),
        run=LPAPSurrogateRunConfig(
            run_training=bool(run_data["run_training"]),
            resume_from_checkpoint=bool(run_data["resume_from_checkpoint"]),
            steps=int(run_data["steps"]),
            seed=int(run_data["seed"]),
            permutation_seed=int(run_data["permutation_seed"]),
            display_every=int(run_data["display_every"]),
            log_every=int(run_data["log_every"]),
            run_id=str(run_data["run_id"]),
            checkpoint_name=str(run_data["checkpoint_name"]),
            log_name=str(run_data["log_name"]),
            comment=str(run_data.get("comment", "")),
            pinned=bool(run_data.get("pinned", False)),
        ),
    )


def rerun_lpap_surrogate_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> LPAPSurrogateTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return lpap_surrogate_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


@dataclass(frozen=True)
class LPAPSurrogateTrainingSession:
    config: LPAPSurrogateTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    energy_bank: torch.Tensor
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
    energy_bank = load_teacher_energy_bank(root, config.data)
    permutation = make_permutation_indices(
        value_count=config.value_count,
        seed=config.run.permutation_seed,
        device=target_device,
    )
    model = LPAPSurrogateTransformer(
        value_count=config.value_count,
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
            comment=config.run.comment,
            pinned=config.run.pinned,
        ),
        model=model,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(),
        metadata={
            "device": str(target_device),
            "energy_bank_path": str(
                root / config.data.energy_bank.path
                if not Path(config.data.energy_bank.path).is_absolute()
                else config.data.energy_bank.path
            ),
            "energy_bank_rows": int(energy_bank.shape[0]),
        },
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
        energy_bank=energy_bank,
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
    *,
    values: torch.Tensor | None = None,
) -> LPAPSurrogateMetrics:
    config = session.config
    batch = (
        values
        if values is not None
        else sample_teacher_energy_batch(
            session.energy_bank,
            batch_size=config.validation.batch_size,
            generator=session.validation_generator,
            device=session.device,
        )
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

    train_batches = cycle_energy_bank_batches(
        session.energy_bank,
        batch_size=config.data.batch_size,
        generator=session.generator,
        device=session.device,
    )
    validation_batches = cycle_energy_bank_batches(
        session.energy_bank,
        batch_size=config.validation.batch_size,
        generator=session.validation_generator,
        device=session.device,
    )
    for step in range(session.resume_info.start_step, config.run.steps + 1):
        batch = next(train_batches)
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
            validation_metrics = validate_lpap_surrogate(
                session, values=next(validation_batches)
            )
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
                "validation_seed": config.validation.seed,
                "permutation": session.permutation.detach().cpu(),
            },
        )

    session.training_run.mark_finished()
