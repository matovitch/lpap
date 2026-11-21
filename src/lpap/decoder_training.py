from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from lpap.checkpoints import load_training_checkpoint
from lpap.decoder import (
    LPAPDecoderMetrics,
    LPAPDecoderTransformer,
    evaluate_lpap_decoder_batch,
    evaluate_lpap_decoder_exact_batch,
    prepare_exact_lpap_decoder_batch,
    prepare_lpap_decoder_batch,
    reconstruct_lpap_bucket_values,
    reconstruct_lpap_decoder_values,
    train_lpap_decoder_exact_step,
    train_lpap_decoder_step,
)
from lpap.energy_bank import cycle_energy_bank_batches, energy_bank_config_from_dict
from lpap.hilbert import hilbert_unflatten_images
from lpap.permutation import (
    LPAP_FOLD_ORDER,
    as_long_permutation,
    make_permutation_indices,
)
from lpap.surrogate import (
    LPAPSurrogateTransformer,
    local_classes_to_source_indices,
    prepare_lpap_surrogate_batch,
)
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateOptimizerConfig,
    LPAPSurrogateValidationConfig,
    load_teacher_energy_bank,
    sample_teacher_energy_batch,
)
from lpap.training import (
    TrainingResumeInfo,
    TrainingRun,
    TrainingRunConfig,
    TrainingStepResult,
)
from lpap.training_log import load_run_record

DecoderFrontendMode = Literal["surrogate", "exact"]


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
    frontend_mode: DecoderFrontendMode = "surrogate"
    exact_k_max: int | None = None
    init_decoder_checkpoint_name: str = ""
    use_triton: bool = False

    def validate(self) -> None:
        if self.frontend_mode not in ("surrogate", "exact"):
            raise ValueError(f"unsupported frontend_mode: {self.frontend_mode!r}")
        if self.exact_k_max is not None and self.exact_k_max <= 0:
            raise ValueError("exact_k_max must be positive")

    def as_dict(self) -> dict[str, str | bool | int]:
        payload: dict[str, str | bool | int] = {
            "checkpoint_name": self.checkpoint_name,
            "load_best": self.load_best,
            "require_checkpoint": self.require_checkpoint,
            "frontend_mode": self.frontend_mode,
            "init_decoder_checkpoint_name": self.init_decoder_checkpoint_name,
            "use_triton": self.use_triton,
        }
        if self.exact_k_max is not None:
            payload["exact_k_max"] = self.exact_k_max
        return payload


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
        self.teacher.validate()
        if (
            self.teacher.exact_k_max is not None
            and self.teacher.exact_k_max > self.data.bucket_count
        ):
            raise ValueError("exact_k_max must be at most bucket_count")
        self.regularization.validate()
        self.run.validate()

    def as_run_config(self) -> dict[str, object]:
        return {
            "data": self.data.as_dict(),
            "decoder": self.decoder.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "validation": self.validation.as_dict(),
            "teacher": self.teacher.as_dict(),
            "regularization": self.regularization.as_dict(),
            "run": self.run.as_dict(),
        }

    def model_config(
        self, *, surrogate_model_config: dict[str, int | str]
    ) -> dict[str, Any]:
        return {
            "value_count": self.value_count,
            "bucket_count": self.data.bucket_count,
            "probe_count": self.data.probe_count,
            "k_max": int(surrogate_model_config["k_max"]),
            "fold_order": LPAP_FOLD_ORDER,
            "frontend_mode": self.teacher.frontend_mode,
            "surrogate": dict(surrogate_model_config),
            "frontend_initial_temperature": self.decoder.frontend_initial_temperature,
            "hidden_dim": self.decoder.hidden_dim,
            "layer_count": self.decoder.layer_count,
            "head_count": self.decoder.head_count,
            "regularization": self.regularization.as_dict(),
        }


def lpap_decoder_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> LPAPDecoderTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    teacher_data = data["teacher"]
    exact_k_max_raw = teacher_data.get("exact_k_max")
    return LPAPDecoderTrainingConfig(
        data=LPAPSurrogateDataConfig(
            batch_size=int(data["data"]["batch_size"]),
            bucket_count=int(data["data"]["bucket_count"]),
            probe_count=int(data["data"]["probe_count"]),
            energy_bank=energy_bank_config_from_dict(data["data"]["energy_bank"]),
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
            checkpoint_name=str(teacher_data["checkpoint_name"]),
            load_best=bool(teacher_data["load_best"]),
            require_checkpoint=bool(teacher_data["require_checkpoint"]),
            frontend_mode=teacher_data.get("frontend_mode", "surrogate"),
            exact_k_max=(
                None if exact_k_max_raw is None else int(exact_k_max_raw)
            ),
            init_decoder_checkpoint_name=str(
                teacher_data.get("init_decoder_checkpoint_name", "")
            ),
            use_triton=bool(teacher_data.get("use_triton", False)),
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
            comment=str(run_data.get("comment", "")),
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
    surrogate_model_config: dict[str, int | str]
    energy_bank: torch.Tensor
    surrogate_k_max: int
    permutation: torch.Tensor
    surrogate: LPAPSurrogateTransformer | None
    decoder: LPAPDecoderTransformer
    optimizer: torch.optim.Optimizer
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo


@dataclass(frozen=True)
class LPAPDecoderGalleryItem:
    energy: torch.Tensor
    lpap: torch.Tensor
    surrogate_hard: torch.Tensor
    decoder: torch.Tensor


def _fallback_surrogate_model_config(
    config: LPAPDecoderTrainingConfig,
) -> dict[str, int | str]:
    surrogate = LPAPSurrogateModelConfig()
    k_max = (
        config.teacher.exact_k_max
        if config.teacher.exact_k_max is not None
        else min(surrogate.k_max, config.data.bucket_count)
    )
    return {
        "value_count": config.value_count,
        "bucket_count": config.data.bucket_count,
        "probe_count": config.data.probe_count,
        "k_max": k_max,
        "hidden_dim": surrogate.hidden_dim,
        "layer_count": surrogate.layer_count,
        "head_count": surrogate.head_count,
        "fold_order": LPAP_FOLD_ORDER,
    }


def _surrogate_model_config_from_checkpoint(
    *,
    path: Path,
    require_checkpoint: bool,
) -> tuple[dict[str, int | str] | None, dict[str, object] | None]:
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
    }
    missing = sorted(required.difference(model_config))
    if missing:
        raise ValueError(
            "surrogate checkpoint model_config is missing: " + ", ".join(missing)
        )
    fold_order = model_config.get("fold_order") or training_state.get("fold_order")
    if fold_order != LPAP_FOLD_ORDER:
        raise ValueError(
            f"surrogate checkpoint fold_order={fold_order!r} does not match {LPAP_FOLD_ORDER!r}"
        )
    parsed: dict[str, int | str] = {name: int(model_config[name]) for name in required}
    parsed["fold_order"] = LPAP_FOLD_ORDER
    return parsed, payload


def _validate_teacher_matches_decoder(
    *,
    teacher_model_config: dict[str, int | str],
    config: LPAPDecoderTrainingConfig,
) -> None:
    expected = {
        "value_count": config.value_count,
        "bucket_count": config.data.bucket_count,
        "probe_count": config.data.probe_count,
    }
    mismatches = [
        f"{name} checkpoint={teacher_model_config[name]} decoder={value}"
        for name, value in expected.items()
        if int(teacher_model_config[name]) != value
    ]
    if mismatches:
        raise ValueError(
            "surrogate checkpoint does not match decoder configuration: "
            + "; ".join(mismatches)
        )


def _permutation_from_surrogate_payload(
    payload: dict[str, object] | None,
    *,
    value_count: int,
    require: bool,
    path: Path,
) -> torch.Tensor | None:
    if payload is None:
        if require:
            raise ValueError(f"surrogate checkpoint missing payload: {path}")
        return None
    raw_perm = (payload.get("training_state") or {}).get("permutation")
    if raw_perm is None:
        if require:
            raise ValueError(
                "surrogate checkpoint is missing training_state.permutation: "
                f"{path}"
            )
        return None
    return as_long_permutation(torch.as_tensor(raw_perm), value_count=value_count)


def _create_surrogate_teacher(
    *,
    path: Path,
    load_best: bool,
    require_checkpoint: bool,
    config: LPAPDecoderTrainingConfig,
    device: torch.device,
) -> tuple[
    LPAPSurrogateTransformer, bool, dict[str, int | str], torch.Tensor | None
]:
    model_config, payload = _surrogate_model_config_from_checkpoint(
        path=path, require_checkpoint=require_checkpoint
    )
    loaded = payload is not None
    if model_config is None:
        model_config = _fallback_surrogate_model_config(config)
    else:
        _validate_teacher_matches_decoder(
            teacher_model_config=model_config, config=config
        )

    surrogate = LPAPSurrogateTransformer(
        value_count=int(model_config["value_count"]),
        probe_count=int(model_config["probe_count"]),
        k_max=int(model_config["k_max"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layer_count=int(model_config["layer_count"]),
        head_count=int(model_config["head_count"]),
    ).to(device)
    saved_permutation = _permutation_from_surrogate_payload(
        payload,
        value_count=config.value_count,
        require=require_checkpoint,
        path=path,
    )
    if payload is not None:
        state_key = "best_model_state" if load_best else "model_state"
        surrogate.load_state_dict(payload[state_key])
    return surrogate, loaded, model_config, saved_permutation


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
    from lpap.teacher_checkpoints import (
        permutation_from_training_state,
        require_matching_pair_permutation,
        resolve_checkpoint_path,
    )

    energy_bank = load_teacher_energy_bank(root, config.data)
    surrogate_checkpoint_path = resolve_checkpoint_path(
        root,
        config.teacher.checkpoint_name,
        ensure=(
            config.teacher.frontend_mode == "surrogate"
            and config.teacher.require_checkpoint
        ),
    )

    surrogate: LPAPSurrogateTransformer | None = None
    surrogate_loaded = False
    teacher_permutation: torch.Tensor | None = None
    if config.teacher.frontend_mode == "surrogate":
        (
            surrogate,
            surrogate_loaded,
            surrogate_model_config,
            teacher_permutation,
        ) = _create_surrogate_teacher(
            path=surrogate_checkpoint_path,
            load_best=config.teacher.load_best,
            require_checkpoint=config.teacher.require_checkpoint,
            config=config,
            device=target_device,
        )
        surrogate.eval()
        for parameter in surrogate.parameters():
            parameter.requires_grad_(False)
    else:
        surrogate_model_config = _fallback_surrogate_model_config(config)

    decoder = LPAPDecoderTransformer(
        value_count=config.value_count,
        frontend_initial_temperature=config.decoder.frontend_initial_temperature,
        hidden_dim=config.decoder.hidden_dim,
        layer_count=config.decoder.layer_count,
        head_count=config.decoder.head_count,
    ).to(target_device)

    init_decoder_permutation: torch.Tensor | None = None
    if config.teacher.init_decoder_checkpoint_name:
        init_path = resolve_checkpoint_path(
            root,
            config.teacher.init_decoder_checkpoint_name,
            ensure=True,
        )
        if not init_path.exists():
            raise FileNotFoundError(
                f"init decoder checkpoint not found: {init_path}"
            )
        init_payload = load_training_checkpoint(init_path, map_location=target_device)
        init_training_state = init_payload.get("training_state")
        if not isinstance(init_training_state, dict):
            raise ValueError(
                f"init decoder checkpoint training_state must be a dict: {init_path}"
            )
        init_model_config = init_training_state.get("model_config") or {}
        if (
            isinstance(init_model_config, dict)
            and config.teacher.exact_k_max is None
            and "k_max" in init_model_config
        ):
            surrogate_model_config = dict(surrogate_model_config)
            surrogate_model_config["k_max"] = int(init_model_config["k_max"])
        init_decoder_permutation = permutation_from_training_state(
            init_training_state,
            value_count=config.value_count,
            path=init_path,
            role="decoder",
        )
        state_key = "best_model_state" if config.teacher.load_best else "model_state"
        decoder.load_state_dict(init_payload[state_key])

    if teacher_permutation is not None:
        permutation = teacher_permutation.to(device=target_device)
    elif init_decoder_permutation is not None:
        permutation = init_decoder_permutation.to(device=target_device)
    else:
        if (
            config.teacher.frontend_mode == "surrogate"
            and config.teacher.require_checkpoint
        ):
            raise ValueError(
                "surrogate checkpoint is missing training_state.permutation: "
                f"{surrogate_checkpoint_path}"
            )
        permutation = make_permutation_indices(
            value_count=config.value_count,
            seed=config.run.permutation_seed,
            device=target_device,
        )

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
            comment=config.run.comment,
            pinned=config.run.pinned,
        ),
        model=decoder,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(surrogate_model_config=surrogate_model_config),
        metadata={
            "device": str(target_device),
            "frontend_mode": config.teacher.frontend_mode,
            "surrogate_checkpoint_loaded": surrogate_loaded,
            "surrogate_model_config": surrogate_model_config,
            "energy_bank_path": str(config.data.energy_bank.path),
            "energy_bank_rows": int(energy_bank.shape[0]),
        },
    )
    resume_info = training_run.resume_or_initialize()
    if resume_info.resumed:
        decoder_payload = load_training_checkpoint(checkpoint_path, map_location="cpu")
        decoder_training_state = decoder_payload.get("training_state")
        if not isinstance(decoder_training_state, dict):
            raise ValueError("decoder checkpoint training_state must be a dictionary")
        fold_order = decoder_training_state.get("fold_order") or (
            decoder_training_state.get("model_config") or {}
        ).get("fold_order")
        if fold_order != LPAP_FOLD_ORDER:
            raise ValueError(
                f"decoder checkpoint fold_order={fold_order!r} does not match {LPAP_FOLD_ORDER!r}"
            )
        decoder_permutation = permutation_from_training_state(
            decoder_training_state,
            value_count=config.value_count,
            path=checkpoint_path,
            role="decoder",
        )
        if config.teacher.frontend_mode == "surrogate":
            permutation = require_matching_pair_permutation(
                surrogate_permutation=permutation.detach().cpu(),
                decoder_permutation=decoder_permutation,
                value_count=config.value_count,
                surrogate_path=surrogate_checkpoint_path,
                decoder_path=checkpoint_path,
            ).to(device=target_device)
        else:
            permutation = decoder_permutation.to(device=target_device)
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
        energy_bank=energy_bank,
        surrogate_k_max=int(surrogate_model_config["k_max"]),
        permutation=permutation,
        surrogate=surrogate,
        decoder=decoder,
        optimizer=optimizer,
        training_run=training_run,
        generator=generator,
        validation_generator=validation_generator,
        resume_info=resume_info,
    )


def validate_lpap_decoder(
    session: LPAPDecoderTrainingSession,
    *,
    values: torch.Tensor | None = None,
) -> LPAPDecoderMetrics:
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
    if config.teacher.frontend_mode == "exact":
        return evaluate_lpap_decoder_exact_batch(
            decoder=session.decoder,
            values=batch,
            bucket_count=config.data.bucket_count,
            k_max=session.surrogate_k_max,
            permutation=session.permutation,
            use_triton=config.teacher.use_triton,
            source_ce_weight=config.regularization.source_ce_weight,
            source_ce_l1_reference=config.regularization.source_ce_l1_reference,
            source_ce_power=config.regularization.source_ce_power,
        )
    if session.surrogate is None:
        raise RuntimeError("surrogate model is required when frontend_mode='surrogate'")
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
        if config.teacher.frontend_mode == "exact":
            metrics = train_lpap_decoder_exact_step(
                decoder=session.decoder,
                optimizer=session.optimizer,
                values=batch,
                bucket_count=config.data.bucket_count,
                k_max=session.surrogate_k_max,
                permutation=session.permutation,
                use_triton=config.teacher.use_triton,
                source_ce_weight=config.regularization.source_ce_weight,
                source_ce_l1_reference=config.regularization.source_ce_l1_reference,
                source_ce_power=config.regularization.source_ce_power,
            )
        else:
            if session.surrogate is None:
                raise RuntimeError(
                    "surrogate model is required when frontend_mode='surrogate'"
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
            validation_metrics = validate_lpap_decoder(
                session, values=next(validation_batches)
            )
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
                "validation_seed": config.validation.seed,
                "fold_order": LPAP_FOLD_ORDER,
                "permutation": session.permutation.detach().cpu(),
            },
        )

    session.training_run.mark_finished()


def collect_lpap_decoder_gallery(
    session: LPAPDecoderTrainingSession, *, sample_count: int = 3
) -> list[LPAPDecoderGalleryItem]:
    config = session.config
    decoder_was_training = session.decoder.training
    surrogate_was_training = (
        session.surrogate.training if session.surrogate is not None else False
    )
    session.decoder.eval()
    if session.surrogate is not None:
        session.surrogate.eval()
    with torch.no_grad():
        values = sample_teacher_energy_batch(
            session.energy_bank,
            batch_size=sample_count,
            generator=session.validation_generator,
            device=session.device,
        )
        if config.teacher.frontend_mode == "exact" or session.surrogate is None:
            decoder_batch = prepare_exact_lpap_decoder_batch(
                values=values,
                bucket_count=config.data.bucket_count,
                k_max=session.surrogate_k_max,
                permutation=session.permutation,
                use_triton=config.teacher.use_triton,
            )
            lpap_values = reconstruct_lpap_bucket_values(decoder_batch)
            surrogate_hard_values = lpap_values
        else:
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
                surrogate_tokens=surrogate_tokens,
            )
            lpap_values = reconstruct_lpap_bucket_values(decoder_batch)
            predicted_sources = local_classes_to_source_indices(
                surrogate_logits.argmax(dim=-1),
                bucket_count=config.data.bucket_count,
                probe_count=config.data.probe_count,
                permutation=session.permutation,
            )
            surrogate_hard_values = torch.zeros_like(values).scatter_add(
                1,
                predicted_sources,
                decoder_batch.surrogate_targets.buckets.to(dtype=values.dtype),
            )
        logits = session.decoder(decoder_batch.tokens)
        decoder_values = reconstruct_lpap_decoder_values(logits, decoder_batch)

    if decoder_was_training:
        session.decoder.train()
    if session.surrogate is not None and surrogate_was_training:
        session.surrogate.train()

    side = int(values.shape[-1] ** 0.5)
    if side * side != int(values.shape[-1]):
        raise ValueError(
            f"gallery energy length {int(values.shape[-1])} is not a square"
        )

    def _spatial(flat: torch.Tensor) -> torch.Tensor:
        return hilbert_unflatten_images(
            flat.detach().cpu().unsqueeze(0).unsqueeze(0), side=side
        )[0, 0]

    return [
        LPAPDecoderGalleryItem(
            energy=_spatial(values[index]),
            lpap=_spatial(lpap_values[index]),
            surrogate_hard=_spatial(surrogate_hard_values[index]),
            decoder=_spatial(decoder_values[index]),
        )
        for index in range(sample_count)
    ]
