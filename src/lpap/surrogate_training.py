from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from lpap.data import sample_synthetic_harmonic_batch
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer, train_lpap_surrogate_step
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)


@dataclass(frozen=True)
class LPAPSurrogateTrainingConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    batch_size: int = 32
    bucket_count: int = 64
    probe_count: int = 16
    k_max: int = 4
    harmonic_count: int = 16
    hidden_dim: int = 128
    layer_count: int = 4
    head_count: int = 4
    learning_rate: float = 1.0e-3
    seed: int = 123
    permutation_seed: int = 123
    checkpoint_every: int = 25
    checkpoint_on_improvement: bool = False
    display_every: int = 5
    log_every: int = 1
    run_id: str = "surrogate_synthetic"
    checkpoint_name: str = "surrogate_synthetic.pt"
    log_name: str = "surrogate.sqlite"

    @property
    def value_count(self) -> int:
        return self.bucket_count * self.probe_count

    def validate(self) -> None:
        if self.hidden_dim % self.head_count != 0:
            raise ValueError("hidden_dim must be divisible by head_count")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.checkpoint_every <= 0 or self.display_every <= 0 or self.log_every <= 0:
            raise ValueError("checkpoint/display/log cadence values must be positive")

    def as_run_config(self) -> dict[str, int | float | str | bool]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "bucket_count": self.bucket_count,
            "probe_count": self.probe_count,
            "value_count": self.value_count,
            "k_max": self.k_max,
            "harmonic_count": self.harmonic_count,
            "hidden_dim": self.hidden_dim,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "permutation_seed": self.permutation_seed,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_on_improvement": self.checkpoint_on_improvement,
            "display_every": self.display_every,
            "log_every": self.log_every,
            "run_id": self.run_id,
        }

    def model_config(self) -> dict[str, int]:
        return {
            "value_count": self.value_count,
            "bucket_count": self.bucket_count,
            "probe_count": self.probe_count,
            "k_max": self.k_max,
            "hidden_dim": self.hidden_dim,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "permutation_seed": self.permutation_seed,
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
    torch.manual_seed(config.seed)
    root = Path(project_root)
    checkpoint_path = root / "checkpoints" / config.checkpoint_name
    log_path = root / "training_logs" / config.log_name
    permutation = make_grouped_permutation_indices(
        value_count=config.value_count,
        bucket_count=config.bucket_count,
        seed=config.permutation_seed,
        device=target_device,
    )
    model = LPAPSurrogateTransformer(
        probe_count=config.probe_count,
        k_max=config.k_max,
        hidden_dim=config.hidden_dim,
        layer_count=config.layer_count,
        head_count=config.head_count,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    training_run = TrainingRun(
        config=TrainingRunConfig(
            run_id=config.run_id,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            total_steps=config.steps,
            monitor="loss",
            mode="min",
            resume=config.resume_from_checkpoint,
            checkpoint_every=config.checkpoint_every,
            checkpoint_on_improvement=config.checkpoint_on_improvement,
            log_every=config.log_every,
            display_every=config.display_every,
        ),
        model=model,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(),
        metadata={"device": str(target_device)},
    )
    resume_info = training_run.resume_or_initialize()
    generator = torch.Generator(device=target_device).manual_seed(
        config.seed + resume_info.start_step
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
        resume_info=resume_info,
    )


def iter_lpap_surrogate_training(
    session: LPAPSurrogateTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.steps:
        session.training_run.mark_finished()
        return

    for step in range(session.resume_info.start_step, config.steps + 1):
        batch = sample_synthetic_harmonic_batch(
            batch_size=config.batch_size,
            n=config.value_count,
            harmonic_count=config.harmonic_count,
            generator=session.generator,
            device=session.device,
        )
        metrics = train_lpap_surrogate_step(
            model=session.model,
            optimizer=session.optimizer,
            values=batch,
            bucket_count=config.bucket_count,
            k_max=config.k_max,
            permutation=session.permutation,
        )
        yield session.training_run.record_step(
            step=step,
            epoch=step,
            metrics={
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "weighted_accuracy": metrics.weighted_accuracy,
                "mean_weight": metrics.mean_weight,
            },
            training_state={
                "seed": config.seed,
                "permutation_seed": config.permutation_seed,
                "permutation": session.permutation.detach().cpu(),
            },
        )

    session.training_run.mark_finished()
