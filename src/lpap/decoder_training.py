from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lpap.checkpoints import load_training_checkpoint
from lpap.data import SyntheticHarmonicConfig
from lpap.decoder import (
    LPAPDecoderMetrics,
    LPAPDecoderTransformer,
    evaluate_lpap_decoder_batch,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_bucket_values,
    reconstruct_lpap_decoder_values,
    train_lpap_decoder_step,
)
from lpap.surrogate import prepare_lpap_surrogate_batch
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import LPAPSurrogateTransformer
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateOptimizerConfig,
    LPAPSurrogateValidationConfig,
    _synthetic_harmonic_config_from_dict,
)
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)
from lpap.training_log import load_run_record


@dataclass(frozen=True)
class LPAPDecoderModelConfig:
    frontend_initial_temperature: float = 0.25
    hidden_dim: int = 128
    layer_count: int = 4
    head_count: int = 4

    def validate(self) -> None:
        if self.frontend_initial_temperature <= 0:
            raise ValueError("frontend_initial_temperature must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self.head_count <= 0:
            raise ValueError("head_count must be positive")
        if self.hidden_dim % self.head_count != 0:
            raise ValueError("hidden_dim must be divisible by head_count")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "frontend_initial_temperature": self.frontend_initial_temperature,
            "hidden_dim": self.hidden_dim,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
        }


@dataclass(frozen=True)
class LPAPDecoderTeacherConfig:
    checkpoint_name: str = "surrogate_synthetic.pt"
    load_best: bool = True
    require_checkpoint: bool = False

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "load_best": self.load_best,
            "require_checkpoint": self.require_checkpoint,
        }


@dataclass(frozen=True)
class LPAPDecoderRegularizationConfig:
    source_ce_weight: float = 0.1
    source_ce_l1_reference: float = 0.05
    source_ce_power: float = 2.0

    def validate(self) -> None:
        if self.source_ce_weight < 0:
            raise ValueError("source_ce_weight must be non-negative")
        if self.source_ce_l1_reference <= 0:
            raise ValueError("source_ce_l1_reference must be positive")
        if self.source_ce_power <= 0:
            raise ValueError("source_ce_power must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "source_ce_weight": self.source_ce_weight,
            "source_ce_l1_reference": self.source_ce_l1_reference,
            "source_ce_power": self.source_ce_power,
        }


@dataclass(frozen=True)
class LPAPDecoderRunConfig:
    run_training: bool = True
    resume_from_checkpoint: bool = True
    steps: int = 1000
    seed: int = 456
    permutation_seed: int = 123
    display_every: int = 5
    log_every: int = 1
    run_id: str = "decoder_synthetic"
    checkpoint_name: str = "decoder_synthetic.pt"
    log_name: str = "decoder.sqlite"
    note: str = ""
    tags: tuple[str, ...] = ()
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
            "note": self.note,
            "tags": self.tags,
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class LPAPDecoderTrainingConfig:
    data: LPAPSurrogateDataConfig = field(default_factory=LPAPSurrogateDataConfig)
    decoder: LPAPDecoderModelConfig = field(default_factory=LPAPDecoderModelConfig)
    optimizer: LPAPSurrogateOptimizerConfig = field(
        default_factory=LPAPSurrogateOptimizerConfig
    )
    validation: LPAPSurrogateValidationConfig = field(
        default_factory=LPAPSurrogateValidationConfig
    )
    teacher: LPAPDecoderTeacherConfig = field(default_factory=LPAPDecoderTeacherConfig)
    regularization: LPAPDecoderRegularizationConfig = field(
        default_factory=LPAPDecoderRegularizationConfig
    )
    run: LPAPDecoderRunConfig = field(default_factory=LPAPDecoderRunConfig)

    @property
    def value_count(self) -> int:
        return self.data.value_count

    def validate(self) -> None:
        self.data.validate()
        self.decoder.validate()
        self.optimizer.validate()
        self.validation.validate()
        self.regularization.validate()
        self.run.validate()

    def as_run_config(self) -> dict[str, object]:
        data_config = self.data.as_dict()
        data_config.pop("harmonics")
        return {
            "data": data_config,
            "decoder": self.decoder.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "teacher": self.teacher.as_dict(),
            "regularization": self.regularization.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(self, *, surrogate_model_config: dict[str, int]) -> dict[str, Any]:
        return {
            "value_count": self.value_count,
            "bucket_count": self.data.bucket_count,
            "probe_count": self.data.probe_count,
            "surrogate": surrogate_model_config,
            "frontend_initial_temperature": self.decoder.frontend_initial_temperature,
            "hidden_dim": self.decoder.hidden_dim,
            "layer_count": self.decoder.layer_count,
            "head_count": self.decoder.head_count,
            "permutation_seed": self.run.permutation_seed,
            "regularization": self.regularization.as_dict(),
        }


def lpap_decoder_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> LPAPDecoderTrainingConfig:
    run_data = dict(data["run"])
    harmonics_data = data["data"].get("harmonics")
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return LPAPDecoderTrainingConfig(
        data=LPAPSurrogateDataConfig(
            batch_size=int(data["data"]["batch_size"]),
            bucket_count=int(data["data"]["bucket_count"]),
            probe_count=int(data["data"]["probe_count"]),
            harmonics=(
                SyntheticHarmonicConfig()
                if harmonics_data is None
                else _synthetic_harmonic_config_from_dict(harmonics_data)
            ),
        ),
        decoder=LPAPDecoderModelConfig(
            frontend_initial_temperature=float(
                data["decoder"]["frontend_initial_temperature"]
            ),
            hidden_dim=int(data["decoder"]["hidden_dim"]),
            layer_count=int(data["decoder"]["layer_count"]),
            head_count=int(data["decoder"]["head_count"]),
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
        teacher=LPAPDecoderTeacherConfig(
            checkpoint_name=str(data["teacher"]["checkpoint_name"]),
            load_best=bool(data["teacher"]["load_best"]),
            require_checkpoint=bool(data["teacher"]["require_checkpoint"]),
        ),
        regularization=LPAPDecoderRegularizationConfig(
            source_ce_weight=float(data["regularization"]["source_ce_weight"]),
            source_ce_l1_reference=float(
                data["regularization"]["source_ce_l1_reference"]
            ),
            source_ce_power=float(data["regularization"]["source_ce_power"]),
        ),
        run=LPAPDecoderRunConfig(
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
            note=str(run_data.get("note", "")),
            tags=tuple(str(tag) for tag in run_data.get("tags", ())),
            pinned=bool(run_data.get("pinned", False)),
        ),
    )


def rerun_lpap_decoder_training_config_from_log(
    path: str | Path,
    *,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> LPAPDecoderTrainingConfig:
    record = load_run_record(path, run_id=run_id)
    return lpap_decoder_training_config_from_dict(
        record["config"], resume_from_checkpoint=resume_from_checkpoint
    )


@dataclass(frozen=True)
class LPAPDecoderTrainingSession:
    config: LPAPDecoderTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    surrogate_checkpoint_path: Path
    surrogate_checkpoint_loaded: bool
    surrogate_model_config: dict[str, int]
    harmonics: SyntheticHarmonicConfig
    surrogate_k_max: int
    permutation: torch.Tensor
    surrogate: LPAPSurrogateTransformer
    decoder: LPAPDecoderTransformer
    optimizer: torch.optim.Optimizer
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo


@dataclass(frozen=True)
class LPAPDecoderGalleryItem:
    harmonics: torch.Tensor
    lpap: torch.Tensor
    decoder: torch.Tensor


def _fallback_surrogate_model_config(
    config: LPAPDecoderTrainingConfig,
) -> dict[str, int]:
    surrogate = LPAPSurrogateModelConfig()
    return {
        "value_count": config.value_count,
        "bucket_count": config.data.bucket_count,
        "probe_count": config.data.probe_count,
        "k_max": surrogate.k_max,
        "hidden_dim": surrogate.hidden_dim,
        "layer_count": surrogate.layer_count,
        "head_count": surrogate.head_count,
        "permutation_seed": config.run.permutation_seed,
    }


def _surrogate_model_config_from_checkpoint(
    *,
    path: Path,
    require_checkpoint: bool,
) -> tuple[dict[str, int] | None, dict[str, object] | None]:
    if not path.exists():
        if require_checkpoint:
            raise FileNotFoundError(f"surrogate checkpoint not found: {path}")
        return None, None
    payload = load_training_checkpoint(path)
    training_state = payload.get("training_state", {})
    model_config = training_state.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("surrogate checkpoint is missing training_state.model_config")
    required = {
        "value_count",
        "bucket_count",
        "probe_count",
        "k_max",
        "hidden_dim",
        "layer_count",
        "head_count",
        "permutation_seed",
    }
    missing = sorted(required.difference(model_config))
    if missing:
        raise ValueError(
            "surrogate checkpoint model_config is missing: " + ", ".join(missing)
        )
    return {name: int(model_config[name]) for name in required}, payload


def _surrogate_harmonics_from_checkpoint(payload: dict[str, object]) -> SyntheticHarmonicConfig:
    training_state = payload.get("training_state", {})
    if not isinstance(training_state, dict):
        raise ValueError("surrogate checkpoint training_state must be a dictionary")
    run_config = training_state.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(
            "surrogate checkpoint is missing training_state.run_config; "
            "regenerate the surrogate checkpoint so decoder training can inherit "
            "the teacher harmonic configuration"
        )
    data_config = run_config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("surrogate checkpoint is missing training_state.run_config.data")
    harmonics_config = data_config.get("harmonics")
    if not isinstance(harmonics_config, dict):
        raise ValueError(
            "surrogate checkpoint is missing training_state.run_config.data.harmonics"
        )
    return _synthetic_harmonic_config_from_dict(harmonics_config)


def _validate_teacher_matches_decoder(
    *,
    teacher_model_config: dict[str, int],
    config: LPAPDecoderTrainingConfig,
) -> None:
    expected = {
        "value_count": config.value_count,
        "bucket_count": config.data.bucket_count,
        "probe_count": config.data.probe_count,
        "permutation_seed": config.run.permutation_seed,
    }
    mismatches = [
        f"{name} checkpoint={teacher_model_config[name]} decoder={value}"
        for name, value in expected.items()
        if teacher_model_config[name] != value
    ]
    if mismatches:
        raise ValueError(
            "surrogate checkpoint does not match decoder configuration: "
            + "; ".join(mismatches)
        )


def _create_surrogate_teacher(
    *,
    path: Path,
    load_best: bool,
    require_checkpoint: bool,
    config: LPAPDecoderTrainingConfig,
    device: torch.device,
) -> tuple[LPAPSurrogateTransformer, bool, dict[str, int], SyntheticHarmonicConfig]:
    model_config, payload = _surrogate_model_config_from_checkpoint(
        path=path, require_checkpoint=require_checkpoint
    )
    loaded = payload is not None
    if model_config is None:
        model_config = _fallback_surrogate_model_config(config)
        harmonics = config.data.harmonics
    else:
        _validate_teacher_matches_decoder(
            teacher_model_config=model_config, config=config
        )
        harmonics = _surrogate_harmonics_from_checkpoint(payload)

    surrogate = LPAPSurrogateTransformer(
        value_count=model_config["value_count"],
        probe_count=model_config["probe_count"],
        k_max=model_config["k_max"],
        hidden_dim=model_config["hidden_dim"],
        layer_count=model_config["layer_count"],
        head_count=model_config["head_count"],
    ).to(device)
    if payload is not None:
        state_key = "best_model_state" if load_best else "model_state"
        surrogate.load_state_dict(payload[state_key])
    return surrogate, loaded, model_config, harmonics


def create_lpap_decoder_training_session(
    *,
    project_root: str | Path,
    config: LPAPDecoderTrainingConfig,
    device: str | torch.device | None = None,
) -> LPAPDecoderTrainingSession:
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
    surrogate_checkpoint_path = root / "checkpoints" / config.teacher.checkpoint_name
    permutation = make_grouped_permutation_indices(
        value_count=config.value_count,
        bucket_count=config.data.bucket_count,
        seed=config.run.permutation_seed,
        device=target_device,
    )
    surrogate, surrogate_loaded, surrogate_model_config, harmonics = _create_surrogate_teacher(
        path=surrogate_checkpoint_path,
        load_best=config.teacher.load_best,
        require_checkpoint=config.teacher.require_checkpoint,
        config=config,
        device=target_device,
    )
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)

    decoder = LPAPDecoderTransformer(
        value_count=config.value_count,
        frontend_initial_temperature=config.decoder.frontend_initial_temperature,
        hidden_dim=config.decoder.hidden_dim,
        layer_count=config.decoder.layer_count,
        head_count=config.decoder.head_count,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=config.optimizer.learning_rate
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
        model=decoder,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(surrogate_model_config=surrogate_model_config),
        metadata={
            "device": str(target_device),
            "surrogate_checkpoint_loaded": surrogate_loaded,
            "surrogate_model_config": surrogate_model_config,
            "harmonics": harmonics.as_dict(),
        },
    )
    resume_info = training_run.resume_or_initialize()
    generator = torch.Generator(device=target_device).manual_seed(
        config.run.seed + resume_info.start_step
    )
    validation_generator = torch.Generator(device=target_device).manual_seed(
        config.validation.seed + resume_info.start_step
    )
    return LPAPDecoderTrainingSession(
        config=config,
        device=target_device,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        surrogate_checkpoint_path=surrogate_checkpoint_path,
        surrogate_checkpoint_loaded=surrogate_loaded,
        surrogate_model_config=surrogate_model_config,
        harmonics=harmonics,
        surrogate_k_max=surrogate_model_config["k_max"],
        permutation=permutation,
        surrogate=surrogate,
        decoder=decoder,
        optimizer=optimizer,
        training_run=training_run,
        generator=generator,
        validation_generator=validation_generator,
        resume_info=resume_info,
    )


def validate_lpap_decoder(session: LPAPDecoderTrainingSession) -> LPAPDecoderMetrics:
    config = session.config
    batch = session.harmonics.sample_batch(
        batch_size=config.validation.batch_size,
        n=config.value_count,
        generator=session.validation_generator,
        device=session.device,
    )
    return evaluate_lpap_decoder_batch(
        decoder=session.decoder,
        surrogate=session.surrogate,
        values=batch,
        bucket_count=config.data.bucket_count,
        k_max=session.surrogate_k_max,
        permutation=session.permutation,
        source_ce_weight=config.regularization.source_ce_weight,
        source_ce_l1_reference=config.regularization.source_ce_l1_reference,
        source_ce_power=config.regularization.source_ce_power,
    )


def should_validate_lpap_decoder(
    *, step: int, config: LPAPDecoderTrainingConfig
) -> bool:
    return config.validation.enabled and (
        step % config.validation.every == 0
        or (config.validation.validate_at_end and step == config.run.steps)
    )


def iter_lpap_decoder_training(
    session: LPAPDecoderTrainingSession,
) -> Iterator[TrainingStepResult]:
    config = session.config
    if session.resume_info.start_step > config.run.steps:
        session.training_run.mark_finished()
        return

    for step in range(session.resume_info.start_step, config.run.steps + 1):
        batch = session.harmonics.sample_batch(
            batch_size=config.data.batch_size,
            n=config.value_count,
            generator=session.generator,
            device=session.device,
        )
        metrics = train_lpap_decoder_step(
            decoder=session.decoder,
            surrogate=session.surrogate,
            optimizer=session.optimizer,
            values=batch,
            bucket_count=config.data.bucket_count,
            k_max=session.surrogate_k_max,
            permutation=session.permutation,
            source_ce_weight=config.regularization.source_ce_weight,
            source_ce_l1_reference=config.regularization.source_ce_l1_reference,
            source_ce_power=config.regularization.source_ce_power,
        )
        step_metrics = {
            "loss": metrics.loss,
            "reconstruction_l1": metrics.reconstruction_l1,
            "source_ce": metrics.source_ce,
            "source_ce_regularizer": metrics.source_ce_regularizer,
            "source_ce_weight": metrics.source_ce_weight,
            "accuracy": metrics.accuracy,
            "weighted_accuracy": metrics.weighted_accuracy,
            "mean_weight": metrics.mean_weight,
            "mean_entropy": metrics.mean_entropy,
        }
        if should_validate_lpap_decoder(step=step, config=config):
            validation_metrics = validate_lpap_decoder(session)
            step_metrics.update(
                {
                    "validation_loss": validation_metrics.loss,
                    "validation_reconstruction_l1": validation_metrics.reconstruction_l1,
                    "validation_source_ce": validation_metrics.source_ce,
                    "validation_source_ce_regularizer": validation_metrics.source_ce_regularizer,
                    "validation_source_ce_weight": validation_metrics.source_ce_weight,
                    "validation_accuracy": validation_metrics.accuracy,
                    "validation_weighted_accuracy": validation_metrics.weighted_accuracy,
                    "validation_mean_weight": validation_metrics.mean_weight,
                    "validation_mean_entropy": validation_metrics.mean_entropy,
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
                "surrogate_checkpoint_path": str(session.surrogate_checkpoint_path),
                "surrogate_checkpoint_loaded": session.surrogate_checkpoint_loaded,
                "surrogate_model_config": session.surrogate_model_config,
                "permutation": session.permutation.detach().cpu(),
            },
        )

    session.training_run.mark_finished()


def collect_lpap_decoder_gallery(
    session: LPAPDecoderTrainingSession, *, sample_count: int = 3
) -> list[LPAPDecoderGalleryItem]:
    config = session.config
    decoder_was_training = session.decoder.training
    surrogate_was_training = session.surrogate.training
    session.decoder.eval()
    session.surrogate.eval()
    with torch.no_grad():
        values = session.harmonics.sample_batch(
            batch_size=sample_count,
            n=config.value_count,
            generator=session.validation_generator,
            device=session.device,
        )
        surrogate_tokens = prepare_lpap_surrogate_batch(
            values,
            bucket_count=config.data.bucket_count,
            permutation=session.permutation,
        )
        surrogate_logits = session.surrogate(surrogate_tokens)
        decoder_batch = prepare_lpap_decoder_batch(
            values=values,
            surrogate_logits=surrogate_logits,
            bucket_count=config.data.bucket_count,
            k_max=session.surrogate_k_max,
            temperature=session.decoder.frontend_temperature(),
            permutation=session.permutation,
        )
        logits = session.decoder(decoder_batch.tokens)
        lpap_values = reconstruct_lpap_bucket_values(decoder_batch)
        decoder_values = reconstruct_lpap_decoder_values(logits, decoder_batch)

    if decoder_was_training:
        session.decoder.train()
    if surrogate_was_training:
        session.surrogate.train()

    return [
        LPAPDecoderGalleryItem(
            harmonics=values[index].detach().cpu(),
            lpap=lpap_values[index].detach().cpu(),
            decoder=decoder_values[index].detach().cpu(),
        )
        for index in range(sample_count)
    ]
