from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lpap.checkpoints import load_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder import (
    LPAPDecoderTransformer,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_decoder_values,
)
from lpap.decoder_training import (
    _surrogate_harmonics_from_checkpoint,
    _surrogate_model_config_from_checkpoint,
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
    should_validate_flow,
    time_config_from_dict,
    train_flow_matching_step,
    validate_image_flow_shape,
    validation_config_from_dict,
)
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer, prepare_lpap_surrogate_batch
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingStepResult,
)
from lpap.training_log import load_run_record


EnergyToImageImageConfig = FlowImageConfig
EnergyToImageFlowConfig = FlowModelConfig
EnergyToImageTimeConfig = FlowTimeConfig
EnergyToImageOptimizerConfig = FlowOptimizerConfig
EnergyToImageValidationConfig = FlowValidationConfig


@dataclass(frozen=True)
class EnergyToImageSourceConfig:
    surrogate_checkpoint_name: str = "surrogate_synthetic.pt"
    decoder_checkpoint_name: str = "decoder_synthetic.pt"
    load_best: bool = True
    require_checkpoints: bool = True

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "surrogate_checkpoint_name": self.surrogate_checkpoint_name,
            "decoder_checkpoint_name": self.decoder_checkpoint_name,
            "load_best": self.load_best,
            "require_checkpoints": self.require_checkpoints,
        }


@dataclass(frozen=True)
class EnergyToImageRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 987
    display_every: int = 5
    log_every: int = 1
    run_id: str = "energy_to_image"
    checkpoint_name: str = "energy_to_image.pt"
    log_name: str = "energy_to_image.sqlite"
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
class EnergyToImageTrainingConfig:
    image: EnergyToImageImageConfig = field(default_factory=EnergyToImageImageConfig)
    source: EnergyToImageSourceConfig = field(default_factory=EnergyToImageSourceConfig)
    flow: EnergyToImageFlowConfig = field(default_factory=EnergyToImageFlowConfig)
    time: EnergyToImageTimeConfig = field(default_factory=EnergyToImageTimeConfig)
    optimizer: EnergyToImageOptimizerConfig = field(
        default_factory=EnergyToImageOptimizerConfig
    )
    validation: EnergyToImageValidationConfig = field(
        default_factory=EnergyToImageValidationConfig
    )
    run: EnergyToImageRunConfig = field(default_factory=EnergyToImageRunConfig)

    @property
    def value_count(self) -> int:
        return self.flow.sequence_length

    def validate(self) -> None:
        self.image.validate()
        self.flow.validate()
        self.time.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.run.validate()
        validate_image_flow_shape(image=self.image, flow=self.flow)

    def as_run_config(self) -> dict[str, object]:
        return {
            "image": self.image.as_dict(),
            "source": self.source.as_dict(),
            "flow": self.flow.as_dict(),
            "time": self.time.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(
        self,
        *,
        surrogate_model_config: dict[str, int],
        decoder_model_config: dict[str, object],
        harmonics: SyntheticHarmonicConfig,
    ) -> dict[str, object]:
        return flow_model_metadata(
            image=self.image,
            flow=self.flow,
            extra={
                "surrogate": surrogate_model_config,
                "decoder": decoder_model_config,
                "harmonics": harmonics.as_dict(),
            },
        )


def energy_to_image_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> EnergyToImageTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return EnergyToImageTrainingConfig(
        image=image_config_from_dict(data["image"]),
        source=EnergyToImageSourceConfig(
            surrogate_checkpoint_name=str(data["source"]["surrogate_checkpoint_name"]),
            decoder_checkpoint_name=str(data["source"]["decoder_checkpoint_name"]),
            load_best=bool(data["source"]["load_best"]),
            require_checkpoints=bool(data["source"]["require_checkpoints"]),
        ),
        flow=flow_model_config_from_dict(data["flow"]),
        time=time_config_from_dict(data["time"]),
        optimizer=optimizer_config_from_dict(data["optimizer"]),
        validation=validation_config_from_dict(data["validation"]),
        run=EnergyToImageRunConfig(
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


def rerun_energy_to_image_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> EnergyToImageTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return energy_to_image_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


@dataclass(frozen=True)
class EnergyToImageTrainingSession:
    config: EnergyToImageTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    image_dataset_path: Path
    image_loader: DataLoader
    validation_image_loader: DataLoader
    surrogate_checkpoint_path: Path
    decoder_checkpoint_path: Path
    surrogate: LPAPSurrogateTransformer
    decoder: LPAPDecoderTransformer
    flow: DilatedConvFlow1d
    optimizer: torch.optim.Optimizer
    permutation: torch.Tensor
    harmonics: SyntheticHarmonicConfig
    surrogate_model_config: dict[str, int]
    decoder_model_config: dict[str, object]
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo


@dataclass(frozen=True)
class EnergyToImageGalleryItem:
    source: torch.Tensor
    generated: dict[int, torch.Tensor]


def resolve_checkpoint_path(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / "checkpoints" / path


def load_surrogate_source(
    *,
    path: Path,
    load_best: bool,
    require_checkpoint: bool,
    device: torch.device,
) -> tuple[LPAPSurrogateTransformer, dict[str, int], SyntheticHarmonicConfig]:
    model_config, payload = _surrogate_model_config_from_checkpoint(
        path=path, require_checkpoint=require_checkpoint
    )
    if model_config is None or payload is None:
        raise FileNotFoundError(f"surrogate checkpoint not found: {path}")
    harmonics = _surrogate_harmonics_from_checkpoint(payload)
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
    return surrogate, model_config, harmonics


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


def validate_source_matches_config(
    *,
    config: EnergyToImageTrainingConfig,
    surrogate_model_config: dict[str, int],
    decoder_model_config: dict[str, object],
) -> None:
    expected = {
        "value_count": config.value_count,
        "bucket_count": int(decoder_model_config["bucket_count"]),
        "probe_count": int(decoder_model_config["probe_count"]),
    }
    mismatches = [
        f"{name} surrogate={surrogate_model_config[name]} expected={value}"
        for name, value in expected.items()
        if surrogate_model_config[name] != value
    ]
    decoder_value_count = int(decoder_model_config["value_count"])
    if decoder_value_count != config.value_count:
        mismatches.append(
            f"value_count decoder={decoder_value_count} expected={config.value_count}"
        )
    if mismatches:
        raise ValueError(
            "energy_to_image source checkpoints do not match config: "
            + "; ".join(mismatches)
        )


def create_energy_to_image_training_session(
    *,
    project_root: str | Path,
    config: EnergyToImageTrainingConfig,
    device: str | torch.device | None = None,
) -> EnergyToImageTrainingSession:
    config.validate()
    root = Path(project_root)
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    torch.manual_seed(config.run.seed)
    surrogate_checkpoint_path = resolve_checkpoint_path(
        root, config.source.surrogate_checkpoint_name
    )
    decoder_checkpoint_path = resolve_checkpoint_path(
        root, config.source.decoder_checkpoint_name
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
    core = create_flow_session_core(
        project_root=root,
        image=config.image,
        flow=config.flow,
        optimizer=config.optimizer,
        validation=config.validation,
        run=flow_run_params_from_config(config.run),
        seed=config.run.seed,
        run_config=config.as_run_config(),
        model_config=config.model_config(
            surrogate_model_config=surrogate_model_config,
            decoder_model_config=decoder_model_config,
            harmonics=harmonics,
        ),
        metadata={
            "surrogate_checkpoint_path": str(surrogate_checkpoint_path),
            "decoder_checkpoint_path": str(decoder_checkpoint_path),
        },
        device=target_device,
    )
    return EnergyToImageTrainingSession(
        config=config,
        device=core.device,
        checkpoint_path=core.checkpoint_path,
        log_path=core.log_path,
        image_dataset_path=core.image_dataset_path,
        image_loader=core.image_loader,
        validation_image_loader=core.validation_image_loader,
        surrogate_checkpoint_path=surrogate_checkpoint_path,
        decoder_checkpoint_path=decoder_checkpoint_path,
        surrogate=surrogate,
        decoder=decoder,
        flow=core.flow,
        optimizer=core.optimizer,
        permutation=permutation,
        harmonics=harmonics,
        surrogate_model_config=surrogate_model_config,
        decoder_model_config=decoder_model_config,
        training_run=core.training_run,
        generator=core.generator,
        validation_generator=core.validation_generator,
        resume_info=core.resume_info,
    )


def _prepare_image_batch(
    images: torch.Tensor, *, side: int, device: torch.device
) -> torch.Tensor:
    return prepare_image_sequence(images, side=side, device=device)


def _sample_source_energy(
    *,
    session: EnergyToImageTrainingSession,
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


def train_energy_to_image_step(
    *,
    session: EnergyToImageTrainingSession,
    images: torch.Tensor,
    generator: torch.Generator,
) -> FlowMatchingMetrics:
    config = session.config
    start = _sample_source_energy(
        session=session, batch_size=images.shape[0], generator=generator
    )
    end = _prepare_image_batch(images, side=config.image.side, device=session.device)
    return train_flow_matching_step(
        model=session.flow,
        optimizer=session.optimizer,
        start=start,
        end=end,
        time_config=config.time,
        max_grad_norm=config.optimizer.max_grad_norm,
        generator=generator,
    )


def evaluate_energy_to_image_batch(
    *,
    session: EnergyToImageTrainingSession,
    images: torch.Tensor,
    generator: torch.Generator,
) -> tuple[FlowMatchingMetrics, dict[str, float]]:
    was_training = session.flow.training
    session.flow.eval()
    with torch.no_grad():
        start = _sample_source_energy(
            session=session, batch_size=images.shape[0], generator=generator
        )
        end = _prepare_image_batch(
            images, side=session.config.image.side, device=session.device
        )
        metrics = evaluate_flow_matching_batch(
            model=session.flow,
            start=start,
            end=end,
            time_config=session.config.time,
            generator=generator,
        )
        diagnostics = integration_diagnostics(
            model=session.flow,
            start=start,
            steps=session.config.validation.euler_steps,
            prefix="generated_image",
        )
    if was_training:
        session.flow.train()
    return metrics, diagnostics


def collect_energy_to_image_gallery(
    session: EnergyToImageTrainingSession,
    *,
    sample_count: int = 3,
    steps: tuple[int, ...] = (64, 32, 16, 8, 4),
    generator: torch.Generator | None = None,
) -> list[EnergyToImageGalleryItem]:
    if sample_count <= 0:
        return []
    if any(step_count <= 0 for step_count in steps):
        raise ValueError("gallery integration steps must be positive")
    source_generator = generator
    if source_generator is None:
        source_generator = torch.Generator(device=session.device).manual_seed(
            session.config.validation.seed
        )

    was_training = session.flow.training
    session.flow.eval()
    with torch.no_grad():
        source = _sample_source_energy(
            session=session,
            batch_size=sample_count,
            generator=source_generator,
        )
        generated = integrate_flow_images(
            model=session.flow,
            start=source,
            steps=steps,
            side=session.config.image.side,
        )
    if was_training:
        session.flow.train()
    return [
        EnergyToImageGalleryItem(
            source=source[index, 0].detach().cpu(),
            generated={
                step_count: image[index, 0].detach().cpu()
                for step_count, image in generated.items()
            },
        )
        for index in range(sample_count)
    ]


def should_validate_energy_to_image(
    *, step: int, config: EnergyToImageTrainingConfig
) -> bool:
    return should_validate_flow(
        step=step, validation=config.validation, total_steps=config.run.steps
    )


def _metrics_dict(metrics: FlowMatchingMetrics) -> dict[str, float]:
    return flow_metrics_dict(
        metrics, source_prefix="source_energy", target_prefix="target_image"
    )


def iter_energy_to_image_training(
    session: EnergyToImageTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    images_iter = cycle_image_batches(session.image_loader)
    validation_images_iter = cycle_image_batches(session.validation_image_loader)
    for step in range(session.resume_info.start_step, config.run.steps + 1):
        images = next(images_iter)
        metrics = train_energy_to_image_step(
            session=session,
            images=images,
            generator=session.generator,
        )
        step_metrics = _metrics_dict(metrics)
        if should_validate_energy_to_image(step=step, config=config):
            validation_images = next(validation_images_iter)
            validation_metrics, diagnostics = evaluate_energy_to_image_batch(
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
            },
        )

    session.training_run.mark_finished()
