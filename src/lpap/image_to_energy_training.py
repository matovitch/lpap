from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from torch.utils.data import DataLoader

from lpap.data import SyntheticHarmonicConfig
from lpap.energy_bank import (
    EnergyBankConfig,
    energy_bank_config_from_dict,
    load_energy_bank,
    resolve_energy_bank_path,
    sample_energy_bank_values,
)
from lpap.flow import (
    DilatedConvFlow1d,
    FlowMatchingMetrics,
)
from lpap.flow_training import (
    FlowImageConfig,
    FlowModelConfig,
    FlowOptimizerConfig,
    FlowTimeConfig,
    FlowValidationConfig,
    create_flow_session_core,
    cycle_image_batches,
    evaluate_flow_matching_batch,
    flow_metrics_dict,
    flow_model_config_from_dict,
    flow_model_metadata,
    flow_run_params_from_config,
    image_config_from_dict,
    integrate_flow_images,
    integration_diagnostics,
    optimizer_config_from_dict,
    prepare_image_sequence,
    sample_harmonic_values,
    sample_flow_time,
    should_validate_flow,
    time_config_from_dict,
    train_flow_matching_step,
    validate_image_flow_shape,
    validation_config_from_dict,
)
from lpap.surrogate_training import _synthetic_harmonic_config_from_dict
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingStepResult,
)
from lpap.training_log import load_run_record


ImageToEnergyImageConfig = FlowImageConfig
ImageToEnergyFlowConfig = FlowModelConfig
ImageToEnergyTimeConfig = FlowTimeConfig
ImageToEnergyOptimizerConfig = FlowOptimizerConfig
ImageToEnergyValidationConfig = FlowValidationConfig
sample_image_to_energy_time = sample_flow_time

ImageToEnergyTargetKind = Literal["harmonics", "energy_bank"]


@dataclass(frozen=True)
class ImageToEnergyTargetConfig:
    """Flow matching end distribution for image→energy.

    ``harmonics`` (default): sample synthetic harmonic energies.
    ``energy_bank``: sample rows from an empirical energy ``.pt`` bank,
    independently of the image batch (marginal prior, not paired map).
    """

    kind: ImageToEnergyTargetKind = "harmonics"
    harmonics: SyntheticHarmonicConfig = field(default_factory=SyntheticHarmonicConfig)
    energy_bank: EnergyBankConfig | None = None

    def validate(self) -> None:
        if self.kind == "energy_bank":
            if self.energy_bank is None:
                raise ValueError("target.energy_bank is required when kind=energy_bank")
            self.energy_bank.validate()
        elif self.kind == "harmonics":
            self.harmonics.validate()
        else:
            raise ValueError(f"unsupported target kind: {self.kind!r}")

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"kind": self.kind}
        if self.kind == "harmonics":
            data["harmonics"] = self.harmonics.as_dict()
        if self.energy_bank is not None:
            data["energy_bank"] = self.energy_bank.as_dict()
        return data


@dataclass(frozen=True)
class ImageToEnergyRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 789
    display_every: int = 5
    log_every: int = 1
    run_id: str = "image_to_energy"
    checkpoint_name: str = "image_to_energy.pt"
    log_name: str = "image_to_energy.sqlite"
    note: str = ""
    tags: tuple[str, ...] = ()
    pinned: bool = False

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.display_every <= 0 or self.log_every <= 0:
            raise ValueError("display/log cadence values must be positive")

    def as_dict(self) -> dict[str, int | str | bool | tuple[str, ...]]:
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
            "note": self.note,
            "tags": self.tags,
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class ImageToEnergyTrainingConfig:
    image: ImageToEnergyImageConfig = field(default_factory=ImageToEnergyImageConfig)
    target: ImageToEnergyTargetConfig = field(default_factory=ImageToEnergyTargetConfig)
    flow: ImageToEnergyFlowConfig = field(default_factory=ImageToEnergyFlowConfig)
    time: ImageToEnergyTimeConfig = field(default_factory=ImageToEnergyTimeConfig)
    optimizer: ImageToEnergyOptimizerConfig = field(
        default_factory=ImageToEnergyOptimizerConfig
    )
    validation: ImageToEnergyValidationConfig = field(
        default_factory=ImageToEnergyValidationConfig
    )
    run: ImageToEnergyRunConfig = field(default_factory=ImageToEnergyRunConfig)

    @property
    def value_count(self) -> int:
        return self.flow.sequence_length

    def validate(self) -> None:
        self.image.validate()
        self.target.validate()
        self.flow.validate()
        self.time.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.run.validate()
        validate_image_flow_shape(image=self.image, flow=self.flow)

    def as_run_config(self) -> dict[str, object]:
        return {
            "image": self.image.as_dict(),
            "target": self.target.as_dict(),
            "flow": self.flow.as_dict(),
            "time": self.time.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(self) -> dict[str, object]:
        return flow_model_metadata(
            image=self.image, flow=self.flow, extra={"target": self.target.as_dict()}
        )


def image_to_energy_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> ImageToEnergyTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return ImageToEnergyTrainingConfig(
        image=image_config_from_dict(data["image"]),
        target=_image_to_energy_target_config_from_dict(data["target"]),
        flow=flow_model_config_from_dict(data["flow"]),
        time=time_config_from_dict(data["time"]),
        optimizer=optimizer_config_from_dict(data["optimizer"]),
        validation=validation_config_from_dict(data["validation"]),
        run=ImageToEnergyRunConfig(
            run_training=bool(run_data["run_training"]),
            resume_from_checkpoint=bool(run_data["resume_from_checkpoint"]),
            steps=int(run_data["steps"]),
            seed=int(run_data["seed"]),
            display_every=int(run_data["display_every"]),
            log_every=int(run_data["log_every"]),
            run_id=str(run_data["run_id"]),
            checkpoint_name=str(run_data["checkpoint_name"]),
            log_name=str(run_data["log_name"]),
            note=str(run_data.get("note", "")),
            tags=tuple(str(tag) for tag in run_data.get("tags", ())),
            pinned=bool(run_data.get("pinned", False)),
        ),
    )


def _image_to_energy_target_config_from_dict(
    data: dict[str, Any],
) -> ImageToEnergyTargetConfig:
    kind = str(data.get("kind", "harmonics"))
    if kind not in ("harmonics", "energy_bank"):
        raise ValueError(f"unsupported target kind: {kind!r}")
    harmonics_data = data.get("harmonics")
    energy_bank_data = data.get("energy_bank")
    return ImageToEnergyTargetConfig(
        kind=kind,  # type: ignore[arg-type]
        harmonics=(
            SyntheticHarmonicConfig()
            if harmonics_data is None
            else _synthetic_harmonic_config_from_dict(harmonics_data)
        ),
        energy_bank=(
            None
            if energy_bank_data is None
            else energy_bank_config_from_dict(energy_bank_data)
        ),
    )


def rerun_image_to_energy_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> ImageToEnergyTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return image_to_energy_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


@dataclass(frozen=True)
class ImageToEnergyTrainingSession:
    config: ImageToEnergyTrainingConfig
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
    energy_bank: torch.Tensor | None = None


@dataclass(frozen=True)
class ImageToEnergyGalleryItem:
    image: torch.Tensor
    generated: dict[int, torch.Tensor]


def create_image_to_energy_training_session(
    *,
    project_root: str | Path,
    config: ImageToEnergyTrainingConfig,
    device: str | torch.device | None = None,
) -> ImageToEnergyTrainingSession:
    config.validate()
    root = Path(project_root)
    energy_bank: torch.Tensor | None = None
    if config.target.kind == "energy_bank":
        assert config.target.energy_bank is not None
        bank_path = resolve_energy_bank_path(root, config.target.energy_bank)
        energy_bank = load_energy_bank(
            bank_path, energies_key=config.target.energy_bank.energies_key
        )
        if int(energy_bank.shape[-1]) != config.value_count:
            raise ValueError(
                "energy bank energy_dim "
                f"{int(energy_bank.shape[-1])} does not match flow sequence_length "
                f"{config.value_count}"
            )
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
        device=device,
    )
    return ImageToEnergyTrainingSession(
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
        energy_bank=energy_bank,
    )


def _sample_targets(
    *,
    config: ImageToEnergyTrainingConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    energy_bank: torch.Tensor | None = None,
) -> torch.Tensor:
    if config.target.kind == "energy_bank":
        if energy_bank is None:
            raise ValueError("energy_bank tensor is required when target.kind=energy_bank")
        targets = sample_energy_bank_values(
            energy_bank,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
    else:
        targets = sample_harmonic_values(
            harmonics=config.target.harmonics,
            batch_size=batch_size,
            n=config.value_count,
            generator=generator,
            device=device,
        )
    return targets.unsqueeze(1)


def train_image_to_energy_step(
    *,
    model: DilatedConvFlow1d,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    config: ImageToEnergyTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
    energy_bank: torch.Tensor | None = None,
) -> FlowMatchingMetrics:
    start = prepare_image_sequence(images, side=config.image.side, device=device)
    end = _sample_targets(
        config=config,
        batch_size=start.shape[0],
        generator=generator,
        device=device,
        energy_bank=energy_bank,
    )
    return train_flow_matching_step(
        model=model,
        optimizer=optimizer,
        start=start,
        end=end,
        time_config=config.time,
        max_grad_norm=config.optimizer.max_grad_norm,
        generator=generator,
    )


def evaluate_image_to_energy_batch(
    *,
    model: DilatedConvFlow1d,
    images: torch.Tensor,
    config: ImageToEnergyTrainingConfig,
    generator: torch.Generator,
    device: torch.device,
    energy_bank: torch.Tensor | None = None,
) -> tuple[FlowMatchingMetrics, dict[str, float]]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        start = prepare_image_sequence(images, side=config.image.side, device=device)
        end = _sample_targets(
            config=config,
            batch_size=start.shape[0],
            generator=generator,
            device=device,
            energy_bank=energy_bank,
        )
        metrics = evaluate_flow_matching_batch(
            model=model,
            start=start,
            end=end,
            time_config=config.time,
            generator=generator,
        )
        diagnostics = integration_diagnostics(
            model=model,
            start=start,
            steps=config.validation.euler_steps,
            prefix="generated_energy",
        )
    if was_training:
        model.train()
    return metrics, diagnostics


def collect_image_to_energy_gallery(
    *,
    model: DilatedConvFlow1d,
    images: torch.Tensor,
    side: int,
    steps: tuple[int, ...] = (64, 32, 16, 8, 4),
    device: torch.device,
) -> list[ImageToEnergyGalleryItem]:
    if any(step_count <= 0 for step_count in steps):
        raise ValueError("integration steps must be positive")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        image_batch = images.to(device=device, dtype=torch.float32)
        start = prepare_image_sequence(image_batch, side=side, device=device)
        generated = integrate_flow_images(
            model=model, start=start, steps=steps, side=side
        )
    if was_training:
        model.train()
    return [
        ImageToEnergyGalleryItem(
            image=image_batch[index].detach().cpu(),
            generated={
                step_count: values[index].detach().cpu()
                for step_count, values in generated.items()
            },
        )
        for index in range(image_batch.shape[0])
    ]


def should_validate_image_to_energy(
    *, step: int, config: ImageToEnergyTrainingConfig
) -> bool:
    return should_validate_flow(
        step=step, validation=config.validation, total_steps=config.run.steps
    )


def _metrics_dict(metrics: FlowMatchingMetrics) -> dict[str, float]:
    return flow_metrics_dict(metrics, source_prefix="image", target_prefix="target")


def iter_image_to_energy_training(
    session: ImageToEnergyTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    images_iter = cycle_image_batches(session.image_loader)
    validation_images_iter = cycle_image_batches(session.validation_image_loader)
    for step in range(session.resume_info.start_step, config.run.steps + 1):
        images = next(images_iter)
        metrics = train_image_to_energy_step(
            model=session.flow,
            optimizer=session.optimizer,
            images=images,
            config=config,
            generator=session.generator,
            device=session.device,
            energy_bank=session.energy_bank,
        )
        step_metrics = _metrics_dict(metrics)
        if should_validate_image_to_energy(step=step, config=config):
            validation_images = next(validation_images_iter)
            validation_metrics, diagnostics = evaluate_image_to_energy_batch(
                model=session.flow,
                images=validation_images,
                config=config,
                generator=session.validation_generator,
                device=session.device,
                energy_bank=session.energy_bank,
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
