from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
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
    validate_image_flow_shape,
    validation_config_from_dict,
)
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import (
    LPAPSurrogateTargets,
    LPAPSurrogateTransformer,
    lpap_surrogate_loss,
    prepare_lpap_surrogate_batch,
)
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)
from lpap.training_log import load_run_record


ImageAutoencoderImageConfig = FlowImageConfig
ImageAutoencoderFlowConfig = FlowModelConfig
ImageAutoencoderOptimizerConfig = FlowOptimizerConfig
ImageAutoencoderValidationConfig = FlowValidationConfig


@dataclass(frozen=True)
class ImageAutoencoderSourceConfig:
    surrogate_checkpoint_name: str = "surrogate_synthetic.pt"
    decoder_checkpoint_name: str = "decoder_synthetic.pt"
    image_to_energy_checkpoint_name: str = "image_to_energy.pt"
    energy_to_image_checkpoint_name: str = "energy_to_image_reflow_8.pt"
    load_best: bool = True
    require_checkpoints: bool = True
    train_image_to_energy_flow: bool = True
    train_surrogate: bool = True
    train_decoder: bool = True
    train_energy_to_image_flow: bool = True

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "surrogate_checkpoint_name": self.surrogate_checkpoint_name,
            "decoder_checkpoint_name": self.decoder_checkpoint_name,
            "image_to_energy_checkpoint_name": self.image_to_energy_checkpoint_name,
            "energy_to_image_checkpoint_name": self.energy_to_image_checkpoint_name,
            "load_best": self.load_best,
            "require_checkpoints": self.require_checkpoints,
            "train_image_to_energy_flow": self.train_image_to_energy_flow,
            "train_surrogate": self.train_surrogate,
            "train_decoder": self.train_decoder,
            "train_energy_to_image_flow": self.train_energy_to_image_flow,
        }


@dataclass(frozen=True)
class ImageAutoencoderIntegrationConfig:
    image_to_energy_steps: int = 8
    energy_to_image_steps: int = 8

    def validate(self) -> None:
        if self.image_to_energy_steps <= 0:
            raise ValueError("image_to_energy_steps must be positive")
        if self.energy_to_image_steps <= 0:
            raise ValueError("energy_to_image_steps must be positive")

    def as_dict(self) -> dict[str, int]:
        return {
            "image_to_energy_steps": self.image_to_energy_steps,
            "energy_to_image_steps": self.energy_to_image_steps,
        }


@dataclass(frozen=True)
class ImageAutoencoderLossConfig:
    image_l2_weight: float = 1.0
    energy_l1_weight: float = 0.25
    surrogate_teacher_weight: float = 0.1
    detach_energy_target: bool = False

    def validate(self) -> None:
        if self.image_l2_weight < 0:
            raise ValueError("image_l2_weight must be non-negative")
        if self.energy_l1_weight < 0:
            raise ValueError("energy_l1_weight must be non-negative")
        if self.surrogate_teacher_weight < 0:
            raise ValueError("surrogate_teacher_weight must be non-negative")
        if (
            self.image_l2_weight == 0
            and self.energy_l1_weight == 0
            and self.surrogate_teacher_weight == 0
        ):
            raise ValueError("at least one loss weight must be positive")

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "image_l2_weight": self.image_l2_weight,
            "energy_l1_weight": self.energy_l1_weight,
            "surrogate_teacher_weight": self.surrogate_teacher_weight,
            "detach_energy_target": self.detach_energy_target,
        }


@dataclass(frozen=True)
class ImageAutoencoderRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 2987
    display_every: int = 5
    log_every: int = 1
    run_id: str = "image_autoencoder"
    checkpoint_name: str = "image_autoencoder.pt"
    log_name: str = "image_autoencoder.sqlite"
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
class ImageAutoencoderTrainingConfig:
    image: ImageAutoencoderImageConfig = field(
        default_factory=ImageAutoencoderImageConfig
    )
    source: ImageAutoencoderSourceConfig = field(
        default_factory=ImageAutoencoderSourceConfig
    )
    image_to_energy_flow: ImageAutoencoderFlowConfig = field(
        default_factory=ImageAutoencoderFlowConfig
    )
    energy_to_image_flow: ImageAutoencoderFlowConfig = field(
        default_factory=ImageAutoencoderFlowConfig
    )
    integration: ImageAutoencoderIntegrationConfig = field(
        default_factory=ImageAutoencoderIntegrationConfig
    )
    loss: ImageAutoencoderLossConfig = field(default_factory=ImageAutoencoderLossConfig)
    optimizer: ImageAutoencoderOptimizerConfig = field(
        default_factory=ImageAutoencoderOptimizerConfig
    )
    validation: ImageAutoencoderValidationConfig = field(
        default_factory=ImageAutoencoderValidationConfig
    )
    run: ImageAutoencoderRunConfig = field(default_factory=ImageAutoencoderRunConfig)

    @property
    def value_count(self) -> int:
        return self.image_to_energy_flow.sequence_length

    def validate(self) -> None:
        self.image.validate()
        self.image_to_energy_flow.validate()
        self.energy_to_image_flow.validate()
        self.integration.validate()
        self.loss.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.run.validate()
        validate_image_flow_shape(image=self.image, flow=self.image_to_energy_flow)
        validate_image_flow_shape(image=self.image, flow=self.energy_to_image_flow)
        if (
            self.image_to_energy_flow.sequence_length
            != self.energy_to_image_flow.sequence_length
        ):
            raise ValueError(
                "image_to_energy_flow and energy_to_image_flow sequence lengths must match"
            )

    def as_run_config(self) -> dict[str, object]:
        return {
            "image": self.image.as_dict(),
            "source": self.source.as_dict(),
            "image_to_energy_flow": self.image_to_energy_flow.as_dict(),
            "energy_to_image_flow": self.energy_to_image_flow.as_dict(),
            "integration": self.integration.as_dict(),
            "loss": self.loss.as_dict(),
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
            flow=self.image_to_energy_flow,
            extra={
                "source": self.source.as_dict(),
                "image_to_energy_flow": self.image_to_energy_flow.as_dict(),
                "energy_to_image_flow": self.energy_to_image_flow.as_dict(),
                "integration": self.integration.as_dict(),
                "loss": self.loss.as_dict(),
                "surrogate": surrogate_model_config,
                "decoder": decoder_model_config,
                "harmonics": harmonics.as_dict(),
            },
        )


@dataclass(frozen=True)
class ImageAutoencoderForward:
    image: torch.Tensor
    encoded_energy: torch.Tensor
    decoded_energy: torch.Tensor
    reconstructed_image: torch.Tensor
    surrogate_logits: torch.Tensor
    decoder_logits: torch.Tensor
    surrogate_targets: LPAPSurrogateTargets


@dataclass(frozen=True)
class ImageAutoencoderMetrics:
    loss: float
    image_reconstruction_l2: float
    energy_reconstruction_l1: float
    surrogate_teacher_ce: float
    surrogate_weighted_accuracy: float
    encoded_energy_rms: float
    decoded_energy_rms: float
    reconstructed_image_rms: float
    image_rms: float


class ImageAutoencoderModel(nn.Module):
    def __init__(
        self,
        *,
        image_to_energy_flow: DilatedConvFlow1d,
        surrogate: LPAPSurrogateTransformer,
        decoder: LPAPDecoderTransformer,
        energy_to_image_flow: DilatedConvFlow1d,
    ) -> None:
        super().__init__()
        self.image_to_energy_flow = image_to_energy_flow
        self.surrogate = surrogate
        self.decoder = decoder
        self.energy_to_image_flow = energy_to_image_flow

    def forward_chain(
        self,
        *,
        image: torch.Tensor,
        bucket_count: int,
        k_max: int,
        permutation: torch.Tensor,
        image_to_energy_steps: int,
        energy_to_image_steps: int,
    ) -> ImageAutoencoderForward:
        encoded_energy = integrate_euler_midpoint_time(
            self.image_to_energy_flow, image, image_to_energy_steps
        )
        values = encoded_energy[:, 0]
        surrogate_tokens = prepare_lpap_surrogate_batch(
            values, bucket_count=bucket_count, permutation=permutation
        )
        surrogate_logits = self.surrogate(surrogate_tokens)
        decoder_batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=bucket_count,
            k_max=k_max,
            temperature=self.decoder.frontend_temperature(),
            permutation=permutation,
        )
        decoder_logits = self.decoder(decoder_batch.tokens)
        decoded_energy = reconstruct_lpap_decoder_values(
            decoder_logits, decoder_batch
        ).unsqueeze(1)
        reconstructed_image = integrate_euler_midpoint_time(
            self.energy_to_image_flow, decoded_energy, energy_to_image_steps
        )
        return ImageAutoencoderForward(
            image=image,
            encoded_energy=encoded_energy,
            decoded_energy=decoded_energy,
            reconstructed_image=reconstructed_image,
            surrogate_logits=surrogate_logits,
            decoder_logits=decoder_logits,
            surrogate_targets=decoder_batch.surrogate_targets,
        )


@dataclass(frozen=True)
class ImageAutoencoderTrainingSession:
    config: ImageAutoencoderTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    image_dataset_path: Path
    image_loader: DataLoader
    validation_image_loader: DataLoader
    surrogate_checkpoint_path: Path
    decoder_checkpoint_path: Path
    image_to_energy_checkpoint_path: Path
    energy_to_image_checkpoint_path: Path
    model: ImageAutoencoderModel
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
class ImageAutoencoderGalleryItem:
    image: torch.Tensor
    reconstructed_image: torch.Tensor
    image_error: torch.Tensor
    encoded_energy: torch.Tensor
    decoded_energy: torch.Tensor
    energy_error: torch.Tensor


def image_autoencoder_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> ImageAutoencoderTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return ImageAutoencoderTrainingConfig(
        image=image_config_from_dict(data["image"]),
        source=ImageAutoencoderSourceConfig(
            surrogate_checkpoint_name=str(data["source"]["surrogate_checkpoint_name"]),
            decoder_checkpoint_name=str(data["source"]["decoder_checkpoint_name"]),
            image_to_energy_checkpoint_name=str(
                data["source"]["image_to_energy_checkpoint_name"]
            ),
            energy_to_image_checkpoint_name=str(
                data["source"]["energy_to_image_checkpoint_name"]
            ),
            load_best=bool(data["source"]["load_best"]),
            require_checkpoints=bool(data["source"]["require_checkpoints"]),
            train_image_to_energy_flow=bool(
                data["source"]["train_image_to_energy_flow"]
            ),
            train_surrogate=bool(data["source"]["train_surrogate"]),
            train_decoder=bool(data["source"]["train_decoder"]),
            train_energy_to_image_flow=bool(
                data["source"]["train_energy_to_image_flow"]
            ),
        ),
        image_to_energy_flow=flow_model_config_from_dict(data["image_to_energy_flow"]),
        energy_to_image_flow=flow_model_config_from_dict(data["energy_to_image_flow"]),
        integration=ImageAutoencoderIntegrationConfig(
            image_to_energy_steps=int(data["integration"]["image_to_energy_steps"]),
            energy_to_image_steps=int(data["integration"]["energy_to_image_steps"]),
        ),
        loss=ImageAutoencoderLossConfig(
            image_l2_weight=float(data["loss"]["image_l2_weight"]),
            energy_l1_weight=float(data["loss"]["energy_l1_weight"]),
            surrogate_teacher_weight=float(data["loss"]["surrogate_teacher_weight"]),
            detach_energy_target=bool(data["loss"]["detach_energy_target"]),
        ),
        optimizer=optimizer_config_from_dict(data["optimizer"]),
        validation=validation_config_from_dict(data["validation"]),
        run=ImageAutoencoderRunConfig(
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


def rerun_image_autoencoder_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> ImageAutoencoderTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return image_autoencoder_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


def _set_trainable(module: nn.Module, enabled: bool) -> None:
    module.train(enabled)
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _optimizer_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def create_image_autoencoder_training_session(
    *,
    project_root: str | Path,
    config: ImageAutoencoderTrainingConfig,
    device: str | torch.device | None = None,
) -> ImageAutoencoderTrainingSession:
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
    image_to_energy_checkpoint_path = resolve_checkpoint_path(
        root, config.source.image_to_energy_checkpoint_name
    )
    energy_to_image_checkpoint_path = resolve_checkpoint_path(
        root, config.source.energy_to_image_checkpoint_name
    )

    surrogate_source = EnergyToImageSourceConfig(
        surrogate_checkpoint_name=config.source.surrogate_checkpoint_name,
        decoder_checkpoint_name=config.source.decoder_checkpoint_name,
        load_best=config.source.load_best,
        require_checkpoints=config.source.require_checkpoints,
    )
    surrogate, surrogate_model_config, harmonics = load_surrogate_source(
        path=surrogate_checkpoint_path,
        load_best=surrogate_source.load_best,
        require_checkpoint=surrogate_source.require_checkpoints,
        device=target_device,
    )
    decoder, decoder_model_config = load_decoder_source(
        path=decoder_checkpoint_path,
        load_best=surrogate_source.load_best,
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
    image_to_energy_flow = DilatedConvFlow1d(
        **config.image_to_energy_flow.as_dict()
    ).to(target_device)
    image_to_energy_state = load_flow_checkpoint_state(
        path=image_to_energy_checkpoint_path,
        load_best=config.source.load_best,
        require_checkpoint=config.source.require_checkpoints,
        device=target_device,
    )
    if image_to_energy_state is not None:
        image_to_energy_flow.load_state_dict(image_to_energy_state)
    energy_to_image_flow = DilatedConvFlow1d(
        **config.energy_to_image_flow.as_dict()
    ).to(target_device)
    energy_to_image_state = load_flow_checkpoint_state(
        path=energy_to_image_checkpoint_path,
        load_best=config.source.load_best,
        require_checkpoint=config.source.require_checkpoints,
        device=target_device,
    )
    if energy_to_image_state is not None:
        energy_to_image_flow.load_state_dict(energy_to_image_state)
    model = ImageAutoencoderModel(
        image_to_energy_flow=image_to_energy_flow,
        surrogate=surrogate,
        decoder=decoder,
        energy_to_image_flow=energy_to_image_flow,
    ).to(target_device)
    _set_trainable(model.image_to_energy_flow, config.source.train_image_to_energy_flow)
    _set_trainable(model.surrogate, config.source.train_surrogate)
    _set_trainable(model.decoder, config.source.train_decoder)
    _set_trainable(model.energy_to_image_flow, config.source.train_energy_to_image_flow)
    parameters = _optimizer_parameters(model)
    if not parameters:
        raise ValueError("at least one image autoencoder component must be trainable")
    optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.learning_rate)
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
        model=model,
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
            "image_to_energy_checkpoint_path": str(image_to_energy_checkpoint_path),
            "energy_to_image_checkpoint_path": str(energy_to_image_checkpoint_path),
        },
    )
    resume_info = training_run.resume_or_initialize()
    generator = torch.Generator(device=target_device).manual_seed(
        config.run.seed + resume_info.start_step
    )
    validation_generator = torch.Generator(device=target_device).manual_seed(
        config.validation.seed + resume_info.start_step
    )
    return ImageAutoencoderTrainingSession(
        config=config,
        device=target_device,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        image_dataset_path=image_dataset_path,
        image_loader=image_loader,
        validation_image_loader=validation_image_loader,
        surrogate_checkpoint_path=surrogate_checkpoint_path,
        decoder_checkpoint_path=decoder_checkpoint_path,
        image_to_energy_checkpoint_path=image_to_energy_checkpoint_path,
        energy_to_image_checkpoint_path=energy_to_image_checkpoint_path,
        model=model,
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


def _forward_loss(
    *,
    session: ImageAutoencoderTrainingSession,
    image: torch.Tensor,
) -> tuple[torch.Tensor, ImageAutoencoderMetrics, ImageAutoencoderForward]:
    config = session.config
    output = session.model.forward_chain(
        image=image,
        bucket_count=int(session.decoder_model_config["bucket_count"]),
        k_max=int(session.surrogate_model_config["k_max"]),
        permutation=session.permutation,
        image_to_energy_steps=config.integration.image_to_energy_steps,
        energy_to_image_steps=config.integration.energy_to_image_steps,
    )
    surrogate_teacher_ce, surrogate_metrics = lpap_surrogate_loss(
        output.surrogate_logits, output.surrogate_targets
    )
    image_l2 = torch_functional.mse_loss(output.reconstructed_image, image)
    energy_target = (
        output.encoded_energy.detach()
        if config.loss.detach_energy_target
        else output.encoded_energy
    )
    energy_l1 = torch_functional.l1_loss(output.decoded_energy, energy_target)
    loss = (
        config.loss.image_l2_weight * image_l2
        + config.loss.energy_l1_weight * energy_l1
        + config.loss.surrogate_teacher_weight * surrogate_teacher_ce
    )
    metrics = ImageAutoencoderMetrics(
        loss=float(loss.detach().cpu()),
        image_reconstruction_l2=float(image_l2.detach().cpu()),
        energy_reconstruction_l1=float(energy_l1.detach().cpu()),
        surrogate_teacher_ce=float(surrogate_teacher_ce.detach().cpu()),
        surrogate_weighted_accuracy=surrogate_metrics.weighted_accuracy,
        encoded_energy_rms=float(
            output.encoded_energy.square().mean().sqrt().detach().cpu()
        ),
        decoded_energy_rms=float(
            output.decoded_energy.square().mean().sqrt().detach().cpu()
        ),
        reconstructed_image_rms=float(
            output.reconstructed_image.square().mean().sqrt().detach().cpu()
        ),
        image_rms=float(image.square().mean().sqrt().detach().cpu()),
    )
    return loss, metrics, output


def train_image_autoencoder_step(
    *,
    session: ImageAutoencoderTrainingSession,
    images: torch.Tensor,
) -> ImageAutoencoderMetrics:
    session.model.train()
    image = prepare_image_sequence(
        images, side=session.config.image.side, device=session.device
    )
    session.optimizer.zero_grad(set_to_none=True)
    loss, metrics, _output = _forward_loss(session=session, image=image)
    loss.backward()
    if session.config.optimizer.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            _optimizer_parameters(session.model), session.config.optimizer.max_grad_norm
        )
    session.optimizer.step()
    return metrics


def evaluate_image_autoencoder_batch(
    *,
    session: ImageAutoencoderTrainingSession,
    images: torch.Tensor,
) -> ImageAutoencoderMetrics:
    was_training = session.model.training
    session.model.eval()
    with torch.no_grad():
        image = prepare_image_sequence(
            images, side=session.config.image.side, device=session.device
        )
        _loss, metrics, _output = _forward_loss(session=session, image=image)
    if was_training:
        session.model.train()
    return metrics


def should_validate_image_autoencoder(
    *, step: int, config: ImageAutoencoderTrainingConfig
) -> bool:
    return config.validation.enabled and (
        step % config.validation.every == 0
        or (config.validation.validate_at_end and step == config.run.steps)
    )


def _metrics_dict(metrics: ImageAutoencoderMetrics) -> dict[str, float]:
    return {
        "loss": metrics.loss,
        "image_reconstruction_l2": metrics.image_reconstruction_l2,
        "energy_reconstruction_l1": metrics.energy_reconstruction_l1,
        "surrogate_teacher_ce": metrics.surrogate_teacher_ce,
        "weighted_accuracy": metrics.surrogate_weighted_accuracy,
        "encoded_energy_rms": metrics.encoded_energy_rms,
        "decoded_energy_rms": metrics.decoded_energy_rms,
        "reconstructed_image_rms": metrics.reconstructed_image_rms,
        "image_rms": metrics.image_rms,
    }


def iter_image_autoencoder_training(
    session: ImageAutoencoderTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    images_iter = cycle_image_batches(session.image_loader)
    validation_images_iter = cycle_image_batches(session.validation_image_loader)
    for step in range(session.resume_info.start_step, config.run.steps + 1):
        images = next(images_iter)
        metrics = train_image_autoencoder_step(session=session, images=images)
        step_metrics = _metrics_dict(metrics)
        if should_validate_image_autoencoder(step=step, config=config):
            validation_images = next(validation_images_iter)
            validation_metrics = evaluate_image_autoencoder_batch(
                session=session, images=validation_images
            )
            step_metrics.update(
                {
                    f"validation_{name}": value
                    for name, value in _metrics_dict(validation_metrics).items()
                }
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
                "image_to_energy_checkpoint_path": str(
                    session.image_to_energy_checkpoint_path
                ),
                "energy_to_image_checkpoint_path": str(
                    session.energy_to_image_checkpoint_path
                ),
            },
        )

    session.training_run.mark_finished()


def collect_image_autoencoder_gallery(
    session: ImageAutoencoderTrainingSession,
    *,
    sample_count: int = 3,
) -> list[ImageAutoencoderGalleryItem]:
    if sample_count <= 0:
        return []
    was_training = session.model.training
    session.model.eval()
    images_iter = cycle_image_batches(session.validation_image_loader)
    images = next(images_iter)[:sample_count]
    with torch.no_grad():
        image = prepare_image_sequence(
            images, side=session.config.image.side, device=session.device
        )
        _loss, _metrics, output = _forward_loss(session=session, image=image)
        image_error = output.reconstructed_image - image
        energy_error = output.decoded_energy - output.encoded_energy
    if was_training:
        session.model.train()
    return [
        ImageAutoencoderGalleryItem(
            image=image[index, 0].detach().cpu(),
            reconstructed_image=output.reconstructed_image[index, 0].detach().cpu(),
            image_error=image_error[index, 0].detach().cpu(),
            encoded_energy=output.encoded_energy[index, 0].detach().cpu(),
            decoded_energy=output.decoded_energy[index, 0].detach().cpu(),
            energy_error=energy_error[index, 0].detach().cpu(),
        )
        for index in range(images.shape[0])
    ]


__all__ = [
    "ImageAutoencoderFlowConfig",
    "ImageAutoencoderForward",
    "ImageAutoencoderGalleryItem",
    "ImageAutoencoderImageConfig",
    "ImageAutoencoderIntegrationConfig",
    "ImageAutoencoderLossConfig",
    "ImageAutoencoderMetrics",
    "ImageAutoencoderModel",
    "ImageAutoencoderOptimizerConfig",
    "ImageAutoencoderRunConfig",
    "ImageAutoencoderSourceConfig",
    "ImageAutoencoderTrainingConfig",
    "ImageAutoencoderTrainingSession",
    "ImageAutoencoderValidationConfig",
    "collect_image_autoencoder_gallery",
    "create_image_autoencoder_training_session",
    "evaluate_image_autoencoder_batch",
    "image_autoencoder_training_config_from_dict",
    "iter_image_autoencoder_training",
    "rerun_image_autoencoder_training_config_from_log",
    "should_validate_image_autoencoder",
    "train_image_autoencoder_step",
]
