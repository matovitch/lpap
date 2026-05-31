from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader

from lpap.decoder import (
    LPAPDecoderTransformer,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_decoder_values,
)
from lpap.energy_to_image_training import (
    EnergyToImageSourceConfig,
    resolve_checkpoint_path,
    load_decoder_source,
    load_surrogate_source,
    validate_source_matches_config,
)
from lpap.flow import DilatedConvFlow1d, integrate_euler_midpoint_time
from lpap.flow_training import (
    FlowImageConfig,
    FlowModelConfig,
    FlowOptimizerConfig,
    FlowValidationConfig,
    cycle_image_batches,
    flow_model_config_from_dict,
    flow_model_metadata,
    image_config_from_dict,
    load_flow_checkpoint_state,
    load_flow_image_loader,
    optimizer_config_from_dict,
    prepare_image_sequence,
    sample_harmonic_values,
    validate_image_flow_shape,
    validation_config_from_dict,
)
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer, prepare_lpap_surrogate_batch
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)
from lpap.training_log import load_run_record


EnergyToImageReflowImageConfig = FlowImageConfig
EnergyToImageReflowFlowConfig = FlowModelConfig
EnergyToImageReflowOptimizerConfig = FlowOptimizerConfig
EnergyToImageReflowValidationConfig = FlowValidationConfig


@dataclass(frozen=True)
class EnergyToImageReflowTeacherConfig:
    checkpoint_name: str = "energy_to_image.pt"
    load_best: bool = True
    require_checkpoint: bool = True
    teacher_steps: int = 64
    warm_start_student: bool = True

    def validate(self) -> None:
        if self.teacher_steps <= 0:
            raise ValueError("teacher_steps must be positive")

    def as_dict(self) -> dict[str, str | bool | int]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "load_best": self.load_best,
            "require_checkpoint": self.require_checkpoint,
            "teacher_steps": self.teacher_steps,
            "warm_start_student": self.warm_start_student,
        }


@dataclass(frozen=True)
class EnergyToImageReflowConfig:
    student_steps: int = 8
    endpoint_l2_weight: float = 1.0
    image_anchor_l2_weight: float = 0.25

    def validate(self) -> None:
        if self.student_steps <= 0:
            raise ValueError("student_steps must be positive")
        if self.endpoint_l2_weight < 0:
            raise ValueError("endpoint_l2_weight must be non-negative")
        if self.image_anchor_l2_weight < 0:
            raise ValueError("image_anchor_l2_weight must be non-negative")
        if self.endpoint_l2_weight == 0 and self.image_anchor_l2_weight == 0:
            raise ValueError("at least one reflow loss weight must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "student_steps": self.student_steps,
            "endpoint_l2_weight": self.endpoint_l2_weight,
            "image_anchor_l2_weight": self.image_anchor_l2_weight,
        }


@dataclass(frozen=True)
class EnergyToImageReflowRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 1987
    display_every: int = 5
    log_every: int = 1
    run_id: str = "energy_to_image_reflow"
    checkpoint_name: str = "energy_to_image_reflow_8.pt"
    log_name: str = "energy_to_image_reflow.sqlite"
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
class EnergyToImageReflowTrainingConfig:
    image: EnergyToImageReflowImageConfig = field(
        default_factory=EnergyToImageReflowImageConfig
    )
    source: EnergyToImageSourceConfig = field(default_factory=EnergyToImageSourceConfig)
    flow: EnergyToImageReflowFlowConfig = field(
        default_factory=EnergyToImageReflowFlowConfig
    )
    teacher: EnergyToImageReflowTeacherConfig = field(
        default_factory=EnergyToImageReflowTeacherConfig
    )
    reflow: EnergyToImageReflowConfig = field(default_factory=EnergyToImageReflowConfig)
    optimizer: EnergyToImageReflowOptimizerConfig = field(
        default_factory=EnergyToImageReflowOptimizerConfig
    )
    validation: EnergyToImageReflowValidationConfig = field(
        default_factory=EnergyToImageReflowValidationConfig
    )
    run: EnergyToImageReflowRunConfig = field(
        default_factory=EnergyToImageReflowRunConfig
    )

    @property
    def value_count(self) -> int:
        return self.flow.sequence_length

    def validate(self) -> None:
        self.image.validate()
        self.flow.validate()
        self.teacher.validate()
        self.reflow.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.run.validate()
        validate_image_flow_shape(image=self.image, flow=self.flow)

    def as_run_config(self) -> dict[str, object]:
        return {
            "image": self.image.as_dict(),
            "source": self.source.as_dict(),
            "flow": self.flow.as_dict(),
            "teacher": self.teacher.as_dict(),
            "reflow": self.reflow.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(
        self,
        *,
        surrogate_model_config: dict[str, int],
        decoder_model_config: dict[str, object],
        harmonics: object,
    ) -> dict[str, object]:
        return flow_model_metadata(
            image=self.image,
            flow=self.flow,
            extra={
                "source": self.source.as_dict(),
                "teacher": self.teacher.as_dict(),
                "reflow": self.reflow.as_dict(),
                "surrogate": surrogate_model_config,
                "decoder": decoder_model_config,
                "harmonics": harmonics.as_dict(),
            },
        )


@dataclass(frozen=True)
class EnergyToImageReflowMetrics:
    loss: float
    teacher_endpoint_l2: float
    image_anchor_l2: float
    student_teacher_rel_l2_percent: float
    student_image_rms: float
    teacher_image_rms: float
    target_image_rms: float
    source_energy_rms: float


@dataclass(frozen=True)
class EnergyToImageReflowTrainingSession:
    config: EnergyToImageReflowTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    image_dataset_path: Path
    image_loader: DataLoader
    validation_image_loader: DataLoader
    surrogate_checkpoint_path: Path
    decoder_checkpoint_path: Path
    teacher_checkpoint_path: Path
    surrogate: LPAPSurrogateTransformer
    decoder: LPAPDecoderTransformer
    teacher_flow: DilatedConvFlow1d
    student_flow: DilatedConvFlow1d
    optimizer: torch.optim.Optimizer
    permutation: torch.Tensor
    harmonics: object
    surrogate_model_config: dict[str, int]
    decoder_model_config: dict[str, object]
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo


@dataclass(frozen=True)
class EnergyToImageReflowGalleryItem:
    source: torch.Tensor
    target: torch.Tensor
    teacher: torch.Tensor
    student: torch.Tensor
    error: torch.Tensor


def energy_to_image_reflow_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> EnergyToImageReflowTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return EnergyToImageReflowTrainingConfig(
        image=image_config_from_dict(data["image"]),
        source=EnergyToImageSourceConfig(
            surrogate_checkpoint_name=str(data["source"]["surrogate_checkpoint_name"]),
            decoder_checkpoint_name=str(data["source"]["decoder_checkpoint_name"]),
            load_best=bool(data["source"]["load_best"]),
            require_checkpoints=bool(data["source"]["require_checkpoints"]),
        ),
        flow=flow_model_config_from_dict(data["flow"]),
        teacher=EnergyToImageReflowTeacherConfig(
            checkpoint_name=str(data["teacher"]["checkpoint_name"]),
            load_best=bool(data["teacher"]["load_best"]),
            require_checkpoint=bool(data["teacher"]["require_checkpoint"]),
            teacher_steps=int(data["teacher"]["teacher_steps"]),
            warm_start_student=bool(data["teacher"]["warm_start_student"]),
        ),
        reflow=EnergyToImageReflowConfig(
            student_steps=int(data["reflow"]["student_steps"]),
            endpoint_l2_weight=float(data["reflow"]["endpoint_l2_weight"]),
            image_anchor_l2_weight=float(data["reflow"]["image_anchor_l2_weight"]),
        ),
        optimizer=optimizer_config_from_dict(data["optimizer"]),
        validation=validation_config_from_dict(data["validation"]),
        run=EnergyToImageReflowRunConfig(
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


def rerun_energy_to_image_reflow_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> EnergyToImageReflowTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return energy_to_image_reflow_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


def _sample_source_energy(
    *,
    session: EnergyToImageReflowTrainingSession,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    values = sample_harmonic_values(
        harmonics=session.harmonics,
        batch_size=batch_size,
        n=session.config.value_count,
        generator=generator,
        device=session.device,
    )
    with torch.no_grad():
        surrogate_tokens = prepare_lpap_surrogate_batch(
            values,
            bucket_count=int(session.decoder_model_config["bucket_count"]),
            permutation=session.permutation,
        )
        surrogate_logits = session.surrogate(surrogate_tokens)
        decoder_batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=int(session.decoder_model_config["bucket_count"]),
            k_max=int(session.surrogate_model_config["k_max"]),
            temperature=session.decoder.frontend_temperature(),
            permutation=session.permutation,
        )
        logits = session.decoder(decoder_batch.tokens)
        source = reconstruct_lpap_decoder_values(logits, decoder_batch)
    return source.unsqueeze(1)


def create_energy_to_image_reflow_training_session(
    *,
    project_root: str | Path,
    config: EnergyToImageReflowTrainingConfig,
    device: str | torch.device | None = None,
) -> EnergyToImageReflowTrainingSession:
    config.validate()
    root = Path(project_root)
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    torch.manual_seed(config.run.seed)
    checkpoint_path = root / "checkpoints" / config.run.checkpoint_name
    log_path = root / "training_logs" / config.run.log_name
    surrogate_checkpoint_path = resolve_checkpoint_path(
        root, config.source.surrogate_checkpoint_name
    )
    decoder_checkpoint_path = resolve_checkpoint_path(
        root, config.source.decoder_checkpoint_name
    )
    teacher_checkpoint_path = resolve_checkpoint_path(
        root, config.teacher.checkpoint_name
    )

    surrogate, surrogate_model_config, harmonics = load_surrogate_source(
        path=surrogate_checkpoint_path,
        load_best=config.source.load_best,
        require_checkpoint=config.source.require_checkpoints,
        device=target_device,
    )
    decoder, decoder_model_config = load_decoder_source(
        path=decoder_checkpoint_path,
        load_best=config.source.load_best,
        device=target_device,
    )
    validate_source_matches_config(
        config=config,
        surrogate_model_config=surrogate_model_config,
        decoder_model_config=decoder_model_config,
    )
    permutation = make_grouped_permutation_indices(
        value_count=config.value_count,
        bucket_count=int(decoder_model_config["bucket_count"]),
        seed=surrogate_model_config["permutation_seed"],
        device=target_device,
    )
    image_dataset_path, image_loader = load_flow_image_loader(
        root=root,
        config=config.image,
        batch_size=config.image.batch_size,
        shuffle=config.image.shuffle,
        seed=config.run.seed,
    )
    _validation_image_dataset_path, validation_image_loader = load_flow_image_loader(
        root=root,
        config=config.image,
        batch_size=config.validation.batch_size,
        shuffle=True,
        seed=config.validation.seed,
    )

    teacher_flow = DilatedConvFlow1d(**config.flow.as_dict()).to(target_device)
    teacher_state = load_flow_checkpoint_state(
        path=teacher_checkpoint_path,
        load_best=config.teacher.load_best,
        require_checkpoint=config.teacher.require_checkpoint,
        device=target_device,
    )
    if teacher_state is not None:
        teacher_flow.load_state_dict(teacher_state)
    teacher_flow.eval()
    for parameter in teacher_flow.parameters():
        parameter.requires_grad_(False)

    student_flow = DilatedConvFlow1d(**config.flow.as_dict()).to(target_device)
    if config.teacher.warm_start_student and teacher_state is not None:
        student_flow.load_state_dict(teacher_state)
    optimizer = torch.optim.AdamW(
        student_flow.parameters(), lr=config.optimizer.learning_rate
    )
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
            note=config.run.note,
            tags=config.run.tags,
            pinned=config.run.pinned,
        ),
        model=student_flow,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(
            surrogate_model_config=surrogate_model_config,
            decoder_model_config=decoder_model_config,
            harmonics=harmonics,
        ),
        metadata={
            "device": str(target_device),
            "image_dataset_path": str(image_dataset_path),
            "surrogate_checkpoint_path": str(surrogate_checkpoint_path),
            "decoder_checkpoint_path": str(decoder_checkpoint_path),
            "teacher_checkpoint_path": str(teacher_checkpoint_path),
        },
    )
    resume_info = training_run.resume_or_initialize()
    generator = torch.Generator(device=target_device).manual_seed(
        config.run.seed + resume_info.start_step
    )
    validation_generator = torch.Generator(device=target_device).manual_seed(
        config.validation.seed + resume_info.start_step
    )
    return EnergyToImageReflowTrainingSession(
        config=config,
        device=target_device,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        image_dataset_path=image_dataset_path,
        image_loader=image_loader,
        validation_image_loader=validation_image_loader,
        surrogate_checkpoint_path=surrogate_checkpoint_path,
        decoder_checkpoint_path=decoder_checkpoint_path,
        teacher_checkpoint_path=teacher_checkpoint_path,
        surrogate=surrogate,
        decoder=decoder,
        teacher_flow=teacher_flow,
        student_flow=student_flow,
        optimizer=optimizer,
        permutation=permutation,
        harmonics=harmonics,
        surrogate_model_config=surrogate_model_config,
        decoder_model_config=decoder_model_config,
        training_run=training_run,
        generator=generator,
        validation_generator=validation_generator,
        resume_info=resume_info,
    )


def _reflow_loss(
    *,
    session: EnergyToImageReflowTrainingSession,
    source: torch.Tensor,
    target_image: torch.Tensor,
) -> tuple[torch.Tensor, EnergyToImageReflowMetrics]:
    config = session.config
    with torch.no_grad():
        teacher_image = integrate_euler_midpoint_time(
            session.teacher_flow, source, config.teacher.teacher_steps
        )
    student_image = integrate_euler_midpoint_time(
        session.student_flow, source, config.reflow.student_steps
    )
    teacher_endpoint_l2 = torch_functional.mse_loss(student_image, teacher_image)
    image_anchor_l2 = torch_functional.mse_loss(student_image, target_image)
    loss = (
        config.reflow.endpoint_l2_weight * teacher_endpoint_l2
        + config.reflow.image_anchor_l2_weight * image_anchor_l2
    )
    flat_teacher = teacher_image.flatten(1)
    rel_l2 = (student_image.flatten(1) - flat_teacher).norm(dim=1) / flat_teacher.norm(
        dim=1
    ).clamp_min(torch.finfo(flat_teacher.dtype).eps)
    metrics = EnergyToImageReflowMetrics(
        loss=float(loss.detach().cpu()),
        teacher_endpoint_l2=float(teacher_endpoint_l2.detach().cpu()),
        image_anchor_l2=float(image_anchor_l2.detach().cpu()),
        student_teacher_rel_l2_percent=float((rel_l2.mean() * 100.0).detach().cpu()),
        student_image_rms=float(student_image.square().mean().sqrt().detach().cpu()),
        teacher_image_rms=float(teacher_image.square().mean().sqrt().detach().cpu()),
        target_image_rms=float(target_image.square().mean().sqrt().detach().cpu()),
        source_energy_rms=float(source.square().mean().sqrt().detach().cpu()),
    )
    return loss, metrics


def train_energy_to_image_reflow_step(
    *,
    session: EnergyToImageReflowTrainingSession,
    images: torch.Tensor,
    generator: torch.Generator,
) -> EnergyToImageReflowMetrics:
    session.student_flow.train()
    source = _sample_source_energy(
        session=session, batch_size=images.shape[0], generator=generator
    )
    target_image = prepare_image_sequence(
        images, side=session.config.image.side, device=session.device
    )
    session.optimizer.zero_grad(set_to_none=True)
    loss, metrics = _reflow_loss(
        session=session, source=source, target_image=target_image
    )
    loss.backward()
    if session.config.optimizer.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            session.student_flow.parameters(), session.config.optimizer.max_grad_norm
        )
    session.optimizer.step()
    return metrics


def evaluate_energy_to_image_reflow_batch(
    *,
    session: EnergyToImageReflowTrainingSession,
    images: torch.Tensor,
    generator: torch.Generator,
) -> tuple[EnergyToImageReflowMetrics, dict[str, float]]:
    was_training = session.student_flow.training
    session.student_flow.eval()
    with torch.no_grad():
        source = _sample_source_energy(
            session=session, batch_size=images.shape[0], generator=generator
        )
        target_image = prepare_image_sequence(
            images, side=session.config.image.side, device=session.device
        )
        _loss, metrics = _reflow_loss(
            session=session, source=source, target_image=target_image
        )
        diagnostics = {}
        teacher_image = integrate_euler_midpoint_time(
            session.teacher_flow, source, session.config.teacher.teacher_steps
        )
        for step_count in session.config.validation.euler_steps:
            student_image = integrate_euler_midpoint_time(
                session.student_flow, source, step_count
            )
            diagnostics[f"student_teacher_l2_steps_{step_count}"] = float(
                torch_functional.mse_loss(student_image, teacher_image).detach().cpu()
            )
            diagnostics[f"student_image_rms_steps_{step_count}"] = float(
                student_image.square().mean().sqrt().detach().cpu()
            )
    if was_training:
        session.student_flow.train()
    return metrics, diagnostics


def should_validate_energy_to_image_reflow(
    *, step: int, config: EnergyToImageReflowTrainingConfig
) -> bool:
    return config.validation.enabled and (
        step % config.validation.every == 0
        or (config.validation.validate_at_end and step == config.run.steps)
    )


def _metrics_dict(metrics: EnergyToImageReflowMetrics) -> dict[str, float]:
    return {
        "loss": metrics.loss,
        "teacher_endpoint_l2": metrics.teacher_endpoint_l2,
        "image_anchor_l2": metrics.image_anchor_l2,
        "student_teacher_rel_l2_percent": metrics.student_teacher_rel_l2_percent,
        "student_image_rms": metrics.student_image_rms,
        "teacher_image_rms": metrics.teacher_image_rms,
        "target_image_rms": metrics.target_image_rms,
        "source_energy_rms": metrics.source_energy_rms,
    }


def iter_energy_to_image_reflow_training(
    session: EnergyToImageReflowTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    images_iter = cycle_image_batches(session.image_loader)
    validation_images_iter = cycle_image_batches(session.validation_image_loader)
    for step in range(session.resume_info.start_step, config.run.steps + 1):
        images = next(images_iter)
        metrics = train_energy_to_image_reflow_step(
            session=session,
            images=images,
            generator=session.generator,
        )
        step_metrics = _metrics_dict(metrics)
        if should_validate_energy_to_image_reflow(step=step, config=config):
            validation_images = next(validation_images_iter)
            validation_metrics, diagnostics = evaluate_energy_to_image_reflow_batch(
                session=session,
                images=validation_images,
                generator=session.validation_generator,
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
                "surrogate_checkpoint_path": str(session.surrogate_checkpoint_path),
                "decoder_checkpoint_path": str(session.decoder_checkpoint_path),
                "teacher_checkpoint_path": str(session.teacher_checkpoint_path),
            },
        )

    session.training_run.mark_finished()


def collect_energy_to_image_reflow_gallery(
    session: EnergyToImageReflowTrainingSession,
    *,
    sample_count: int = 3,
) -> list[EnergyToImageReflowGalleryItem]:
    if sample_count <= 0:
        return []
    was_training = session.student_flow.training
    session.student_flow.eval()
    images_iter = cycle_image_batches(session.validation_image_loader)
    images = next(images_iter)[:sample_count]
    with torch.no_grad():
        source = _sample_source_energy(
            session=session,
            batch_size=images.shape[0],
            generator=session.validation_generator,
        )
        target = prepare_image_sequence(
            images, side=session.config.image.side, device=session.device
        )
        teacher = integrate_euler_midpoint_time(
            session.teacher_flow, source, session.config.teacher.teacher_steps
        )
        student = integrate_euler_midpoint_time(
            session.student_flow, source, session.config.reflow.student_steps
        )
        error = student - teacher
    if was_training:
        session.student_flow.train()
    return [
        EnergyToImageReflowGalleryItem(
            source=source[index, 0].detach().cpu(),
            target=target[index, 0].detach().cpu(),
            teacher=teacher[index, 0].detach().cpu(),
            student=student[index, 0].detach().cpu(),
            error=error[index, 0].detach().cpu(),
        )
        for index in range(images.shape[0])
    ]


__all__ = [
    "EnergyToImageReflowConfig",
    "EnergyToImageReflowFlowConfig",
    "EnergyToImageReflowGalleryItem",
    "EnergyToImageReflowImageConfig",
    "EnergyToImageReflowMetrics",
    "EnergyToImageReflowOptimizerConfig",
    "EnergyToImageReflowRunConfig",
    "EnergyToImageReflowTeacherConfig",
    "EnergyToImageReflowTrainingConfig",
    "EnergyToImageReflowTrainingSession",
    "EnergyToImageReflowValidationConfig",
    "collect_energy_to_image_reflow_gallery",
    "create_energy_to_image_reflow_training_session",
    "energy_to_image_reflow_training_config_from_dict",
    "evaluate_energy_to_image_reflow_batch",
    "iter_energy_to_image_reflow_training",
    "rerun_energy_to_image_reflow_training_config_from_log",
    "should_validate_energy_to_image_reflow",
    "train_energy_to_image_reflow_step",
]
