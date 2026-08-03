from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lpap.flow import DilatedConvFlow1d, FlowMatchingMetrics
from lpap.flow_training import (
    FlowImageConfig,
    FlowModelConfig,
    FlowOptimizerConfig,
    FlowTimeConfig,
    FlowValidationConfig,
    create_flow_session_core,
    cycle_image_batches,
    evaluate_bidirectional_flow_matching_batch,
    flow_metrics_dict,
    flow_model_config_from_dict,
    flow_model_metadata,
    flow_run_params_from_config,
    image_config_from_dict,
    integrate_flow_images_range,
    integration_diagnostics_range,
    optimizer_config_from_dict,
    prepare_image_sequence,
    should_validate_flow,
    time_config_from_dict,
    train_bidirectional_flow_matching_step,
    validate_image_flow_shape,
    validation_config_from_dict,
)
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingStepResult,
)
from lpap.training_log import load_run_record

# Global time: t=-1 image, t=0 energy, t=+1 image.
# Energy prior: signed log-normal (|e|~LogNormal(μ,σ²), coin-toss signs).
IMAGE_TO_ENERGY_T0 = -1.0
IMAGE_TO_ENERGY_T1 = 0.0
ENERGY_TO_IMAGE_T0 = 0.0
ENERGY_TO_IMAGE_T1 = 1.0


@dataclass(frozen=True)
class ImageEnergyFlowPriorConfig:
    """Signed log-normal energy marginal at ``t=0``.

    Magnitudes: ``|e| ~ LogNormal(μ, σ²)`` with median ``scale = exp(μ)``.
    Signs: i.i.d. fair coin ``±1`` (``P(+) = 0.5``).
    """

    sigma: float = 2.0
    scale: float = 1.0e-3

    @property
    def mu(self) -> float:
        return log(self.scale)

    def validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError("prior.sigma must be positive")
        if self.scale <= 0:
            raise ValueError("prior.scale must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "sigma": self.sigma,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class ImageEnergyFlowRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 789
    display_every: int = 5
    log_every: int = 1
    run_id: str = "image_energy_flow"
    checkpoint_name: str = "image_energy_flow.pt"
    log_name: str = "image_energy_flow.sqlite"
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
            "display_every": self.display_every,
            "log_every": self.log_every,
            "run_id": self.run_id,
            "checkpoint_name": self.checkpoint_name,
            "log_name": self.log_name,
            "comment": self.comment,
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class ImageEnergyFlowTrainingConfig:
    image: FlowImageConfig = field(default_factory=FlowImageConfig)
    prior: ImageEnergyFlowPriorConfig = field(
        default_factory=ImageEnergyFlowPriorConfig
    )
    flow: FlowModelConfig = field(default_factory=FlowModelConfig)
    time: FlowTimeConfig = field(default_factory=FlowTimeConfig)
    optimizer: FlowOptimizerConfig = field(default_factory=FlowOptimizerConfig)
    validation: FlowValidationConfig = field(default_factory=FlowValidationConfig)
    run: ImageEnergyFlowRunConfig = field(default_factory=ImageEnergyFlowRunConfig)

    @property
    def value_count(self) -> int:
        return self.flow.sequence_length

    def validate(self) -> None:
        self.image.validate()
        self.prior.validate()
        self.flow.validate()
        self.time.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.run.validate()
        validate_image_flow_shape(image=self.image, flow=self.flow)

    def as_run_config(self) -> dict[str, object]:
        return {
            "image": self.image.as_dict(),
            "prior": self.prior.as_dict(),
            "flow": self.flow.as_dict(),
            "time": self.time.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(self) -> dict[str, object]:
        return flow_model_metadata(
            image=self.image,
            flow=self.flow,
            extra={
                "prior": self.prior.as_dict(),
                "time_range": {
                    "image_to_energy": [IMAGE_TO_ENERGY_T0, IMAGE_TO_ENERGY_T1],
                    "energy_to_image": [ENERGY_TO_IMAGE_T0, ENERGY_TO_IMAGE_T1],
                },
            },
        )


def image_energy_flow_prior_config_from_dict(
    data: dict[str, Any],
) -> ImageEnergyFlowPriorConfig:
    return ImageEnergyFlowPriorConfig(
        sigma=float(data.get("sigma", 2.0)),
        scale=float(data.get("scale", 1.0e-3)),
    )


def image_energy_flow_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> ImageEnergyFlowTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return ImageEnergyFlowTrainingConfig(
        image=image_config_from_dict(data["image"]),
        prior=image_energy_flow_prior_config_from_dict(data["prior"]),
        flow=flow_model_config_from_dict(data["flow"]),
        time=time_config_from_dict(data["time"]),
        optimizer=optimizer_config_from_dict(data["optimizer"]),
        validation=validation_config_from_dict(data["validation"]),
        run=ImageEnergyFlowRunConfig(
            run_training=bool(run_data["run_training"]),
            resume_from_checkpoint=bool(run_data["resume_from_checkpoint"]),
            steps=int(run_data["steps"]),
            seed=int(run_data["seed"]),
            display_every=int(run_data["display_every"]),
            log_every=int(run_data["log_every"]),
            run_id=str(run_data["run_id"]),
            checkpoint_name=str(run_data["checkpoint_name"]),
            log_name=str(run_data["log_name"]),
            comment=str(run_data.get("comment", "")),
            pinned=bool(run_data.get("pinned", False)),
        ),
    )


def rerun_image_energy_flow_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> ImageEnergyFlowTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return image_energy_flow_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


@dataclass(frozen=True)
class ImageEnergyFlowTrainingSession:
    config: ImageEnergyFlowTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    image_dataset_path: Path
    image_loader: DataLoader
    validation_image_loader: DataLoader
    flow: DilatedConvFlow1d
    optimizer: torch.optim.Optimizer
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo


@dataclass(frozen=True)
class ImageEnergyFlowGalleryItem:
    image: torch.Tensor
    encoded: dict[int, torch.Tensor]
    reconstructed: dict[int, torch.Tensor]
    prior_energy: torch.Tensor
    from_prior: dict[int, torch.Tensor]


def create_image_energy_flow_training_session(
    *,
    project_root: str | Path,
    config: ImageEnergyFlowTrainingConfig,
    device: str | torch.device | None = None,
) -> ImageEnergyFlowTrainingSession:
    config.validate()
    core = create_flow_session_core(
        project_root=project_root,
        image=config.image,
        flow=config.flow,
        optimizer=config.optimizer,
        validation=config.validation,
        run=flow_run_params_from_config(config.run),
        seed=config.run.seed,
        run_config=config.as_run_config(),
        model_config=config.model_config(),
        metadata={"prior": config.prior.as_dict()},
        device=device,
    )
    return ImageEnergyFlowTrainingSession(
        config=config,
        device=core.device,
        checkpoint_path=core.checkpoint_path,
        log_path=core.log_path,
        image_dataset_path=core.image_dataset_path,
        image_loader=core.image_loader,
        validation_image_loader=core.validation_image_loader,
        flow=core.flow,
        optimizer=core.optimizer,
        training_run=core.training_run,
        generator=core.generator,
        validation_generator=core.validation_generator,
        resume_info=core.resume_info,
    )


def sample_image_energy_prior(
    prior: ImageEnergyFlowPriorConfig,
    *,
    batch_size: int,
    value_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``(batch, 1, value_count)`` signed log-normal energies."""
    prior.validate()
    # |e| = scale * exp(σ Z) = exp(μ + σ Z) with μ = log(scale).
    noise = torch.randn(
        batch_size,
        value_count,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    magnitudes = float(prior.scale) * torch.exp(float(prior.sigma) * noise)
    coins = torch.rand(
        batch_size,
        value_count,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    signs = torch.where(coins < 0.5, 1.0, -1.0)
    return (signs * magnitudes).unsqueeze(1)


def _sample_energy(
    *,
    config: ImageEnergyFlowTrainingConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return sample_image_energy_prior(
        config.prior,
        batch_size=batch_size,
        value_count=config.value_count,
        generator=generator,
        device=device,
    )


def train_image_energy_flow_step(
    *,
    model: DilatedConvFlow1d,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    config: ImageEnergyFlowTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
    energy: torch.Tensor | None = None,
) -> FlowMatchingMetrics:
    image = prepare_image_sequence(images, side=config.image.side, device=device)
    if energy is None:
        energy = _sample_energy(
            config=config,
            batch_size=image.shape[0],
            generator=generator,
            device=device,
        )
    else:
        energy = energy.to(device=device, dtype=torch.float32)
        if energy.ndim == 2:
            energy = energy.unsqueeze(1)
        if energy.shape[0] != image.shape[0]:
            raise ValueError(
                f"energy batch {energy.shape[0]} != image batch {image.shape[0]}"
            )
    return train_bidirectional_flow_matching_step(
        model=model,
        optimizer=optimizer,
        image=image,
        energy=energy,
        time_config=config.time,
        max_grad_norm=config.optimizer.max_grad_norm,
        generator=generator,
    )


def evaluate_image_energy_flow_batch(
    *,
    model: DilatedConvFlow1d,
    images: torch.Tensor,
    config: ImageEnergyFlowTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
    energy: torch.Tensor | None = None,
) -> tuple[FlowMatchingMetrics, dict[str, float]]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        image = prepare_image_sequence(images, side=config.image.side, device=device)
        if energy is None:
            energy = _sample_energy(
                config=config,
                batch_size=image.shape[0],
                generator=generator,
                device=device,
            )
        else:
            energy = energy.to(device=device, dtype=torch.float32)
            if energy.ndim == 2:
                energy = energy.unsqueeze(1)
            if energy.shape[0] != image.shape[0]:
                raise ValueError(
                    f"energy batch {energy.shape[0]} != image batch {image.shape[0]}"
                )
        metrics = evaluate_bidirectional_flow_matching_batch(
            model=model,
            image=image,
            energy=energy,
            time_config=config.time,
            generator=generator,
        )
        diagnostics = {}
        diagnostics.update(
            integration_diagnostics_range(
                model=model,
                start=image,
                steps=config.validation.euler_steps,
                prefix="encoded_energy",
                t0=IMAGE_TO_ENERGY_T0,
                t1=IMAGE_TO_ENERGY_T1,
            )
        )
        diagnostics.update(
            integration_diagnostics_range(
                model=model,
                start=energy,
                steps=config.validation.euler_steps,
                prefix="reconstructed_image",
                t0=ENERGY_TO_IMAGE_T0,
                t1=ENERGY_TO_IMAGE_T1,
            )
        )
    if was_training:
        model.train()
    return metrics, diagnostics


def collect_image_energy_flow_gallery(
    *,
    model: DilatedConvFlow1d,
    images: torch.Tensor,
    energies: torch.Tensor,
    side: int,
    steps: tuple[int, ...] = (64, 32, 16, 8, 4),
    device: torch.device,
) -> list[ImageEnergyFlowGalleryItem]:
    """Build gallery panels for both flow halves.

    ``encoded``: image → energy at each Euler budget.
    ``reconstructed``: round-trip decode of that encoded energy (same budget).
    ``from_prior``: energy prior samples → image (independent of the image batch).
    """
    from lpap.flow import integrate_euler_midpoint_time
    from lpap.hilbert import hilbert_unflatten_images

    if any(step_count <= 0 for step_count in steps):
        raise ValueError("integration steps must be positive")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        image_batch = images.to(device=device, dtype=torch.float32)
        energy_batch = energies.to(device=device, dtype=torch.float32)
        if energy_batch.ndim == 2:
            energy_batch = energy_batch.unsqueeze(1)
        start_image = prepare_image_sequence(image_batch, side=side, device=device)
        encoded: dict[int, torch.Tensor] = {}
        reconstructed: dict[int, torch.Tensor] = {}
        for step_count in steps:
            encoded_seq = integrate_euler_midpoint_time(
                model,
                start_image,
                step_count,
                t0=IMAGE_TO_ENERGY_T0,
                t1=IMAGE_TO_ENERGY_T1,
            )
            encoded[step_count] = hilbert_unflatten_images(encoded_seq, side=side)
            reconstructed_seq = integrate_euler_midpoint_time(
                model,
                encoded_seq,
                step_count,
                t0=ENERGY_TO_IMAGE_T0,
                t1=ENERGY_TO_IMAGE_T1,
            )
            reconstructed[step_count] = hilbert_unflatten_images(
                reconstructed_seq, side=side
            )
        from_prior = integrate_flow_images_range(
            model=model,
            start=energy_batch,
            steps=steps,
            side=side,
            t0=ENERGY_TO_IMAGE_T0,
            t1=ENERGY_TO_IMAGE_T1,
        )
        prior_energy = hilbert_unflatten_images(energy_batch, side=side)
    if was_training:
        model.train()
    return [
        ImageEnergyFlowGalleryItem(
            image=image_batch[index].detach().cpu(),
            encoded={
                step_count: values[index].detach().cpu()
                for step_count, values in encoded.items()
            },
            reconstructed={
                step_count: values[index].detach().cpu()
                for step_count, values in reconstructed.items()
            },
            prior_energy=prior_energy[index].detach().cpu(),
            from_prior={
                step_count: values[index].detach().cpu()
                for step_count, values in from_prior.items()
            },
        )
        for index in range(image_batch.shape[0])
    ]


def should_validate_image_energy_flow(
    *, step: int, config: ImageEnergyFlowTrainingConfig
) -> bool:
    return should_validate_flow(
        step=step, validation=config.validation, total_steps=config.run.steps
    )


def _metrics_dict(metrics: FlowMatchingMetrics) -> dict[str, float]:
    return flow_metrics_dict(metrics, source_prefix="image", target_prefix="energy")


def iter_image_energy_flow_training(
    session: ImageEnergyFlowTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    images_iter = cycle_image_batches(session.image_loader)
    validation_images_iter = cycle_image_batches(session.validation_image_loader)
    for step in range(session.resume_info.start_step, config.run.steps + 1):
        images = next(images_iter)
        metrics = train_image_energy_flow_step(
            model=session.flow,
            optimizer=session.optimizer,
            images=images,
            config=config,
            generator=session.generator,
            device=session.device,
        )
        step_metrics = _metrics_dict(metrics)
        if should_validate_image_energy_flow(step=step, config=config):
            validation_images = next(validation_images_iter)
            validation_metrics, diagnostics = evaluate_image_energy_flow_batch(
                model=session.flow,
                images=validation_images,
                config=config,
                generator=session.validation_generator,
                device=session.device,
            )
            step_metrics.update(
                {
                    f"validation_{name}": value
                    for name, value in _metrics_dict(validation_metrics).items()
                }
            )
            step_metrics.update(
                {f"validation_{name}": value for name, value in diagnostics.items()}
            )
        yield session.training_run.record_step(
            step=step,
            epoch=step,
            metrics=step_metrics,
            training_state={
                "seed": config.run.seed,
                "validation_seed": config.validation.seed,
                "image_dataset_path": str(session.image_dataset_path),
            },
        )

    session.training_run.mark_finished()


__all__ = [
    "ENERGY_TO_IMAGE_T0",
    "ENERGY_TO_IMAGE_T1",
    "IMAGE_TO_ENERGY_T0",
    "IMAGE_TO_ENERGY_T1",
    "ImageEnergyFlowGalleryItem",
    "ImageEnergyFlowPriorConfig",
    "ImageEnergyFlowRunConfig",
    "ImageEnergyFlowTrainingConfig",
    "ImageEnergyFlowTrainingSession",
    "collect_image_energy_flow_gallery",
    "create_image_energy_flow_training_session",
    "evaluate_image_energy_flow_batch",
    "image_energy_flow_prior_config_from_dict",
    "image_energy_flow_training_config_from_dict",
    "iter_image_energy_flow_training",
    "rerun_image_energy_flow_training_config_from_log",
    "sample_image_energy_prior",
    "should_validate_image_energy_flow",
    "train_image_energy_flow_step",
]
