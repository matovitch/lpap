from __future__ import annotations

import tomllib
import json
from pathlib import Path
from typing import Literal

import torch

from lpap.data import SyntheticHarmonicConfig
from lpap.decoder_training import (
    LPAPDecoderModelConfig,
    LPAPDecoderRegularizationConfig,
    LPAPDecoderRunConfig,
    LPAPDecoderTeacherConfig,
    LPAPDecoderTrainingConfig,
    LPAPDecoderTrainingSession,
    create_lpap_decoder_training_session,
    iter_lpap_decoder_training,
    lpap_decoder_training_config_from_dict,
    rerun_lpap_decoder_training_config_from_log,
)
from lpap.image_to_energy_training import (
    ImageToEnergyFlowConfig,
    ImageToEnergyImageConfig,
    ImageToEnergyOptimizerConfig,
    ImageToEnergyRunConfig,
    ImageToEnergyTargetConfig,
    ImageToEnergyTimeConfig,
    ImageToEnergyTrainingConfig,
    ImageToEnergyTrainingSession,
    ImageToEnergyValidationConfig,
    create_image_to_energy_training_session,
    image_to_energy_training_config_from_dict,
    iter_image_to_energy_training,
    rerun_image_to_energy_training_config_from_log,
)
from lpap.image_autoencoder_training import (
    ImageAutoencoderIntegrationConfig,
    ImageAutoencoderLossConfig,
    ImageAutoencoderRunConfig,
    ImageAutoencoderSourceConfig,
    ImageAutoencoderTrainingConfig,
    ImageAutoencoderTrainingSession,
    create_image_autoencoder_training_session,
    image_autoencoder_training_config_from_dict,
    iter_image_autoencoder_training,
    rerun_image_autoencoder_training_config_from_log,
)
from lpap.energy_to_image_training import (
    EnergyToImageRunConfig,
    EnergyToImageSourceConfig,
    EnergyToImageTrainingConfig,
    EnergyToImageTrainingSession,
    create_energy_to_image_training_session,
    energy_to_image_training_config_from_dict,
    iter_energy_to_image_training,
    rerun_energy_to_image_training_config_from_log,
)
from lpap.energy_to_image_reflow_training import (
    EnergyToImageReflowConfig,
    EnergyToImageReflowRunConfig,
    EnergyToImageReflowTeacherConfig,
    EnergyToImageReflowTrainingConfig,
    EnergyToImageReflowTrainingSession,
    create_energy_to_image_reflow_training_session,
    energy_to_image_reflow_training_config_from_dict,
    iter_energy_to_image_reflow_training,
    rerun_energy_to_image_reflow_training_config_from_log,
)
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateOptimizerConfig,
    LPAPSurrogateRunConfig,
    LPAPSurrogateTrainingConfig,
    LPAPSurrogateTrainingSession,
    LPAPSurrogateValidationConfig,
    create_lpap_surrogate_training_session,
    iter_lpap_surrogate_training,
    lpap_surrogate_training_config_from_dict,
    rerun_lpap_surrogate_training_config_from_log,
)
from lpap.training_log import list_training_runs

TrainingModelKind = Literal[
    "surrogate",
    "decoder",
    "image_to_energy",
    "energy_to_image",
    "energy_to_image_reflow",
    "image_autoencoder",
]
TrainingConfig = (
    LPAPSurrogateTrainingConfig
    | LPAPDecoderTrainingConfig
    | ImageToEnergyTrainingConfig
    | EnergyToImageTrainingConfig
    | EnergyToImageReflowTrainingConfig
    | ImageAutoencoderTrainingConfig
)
TrainingSession = (
    LPAPSurrogateTrainingSession
    | LPAPDecoderTrainingSession
    | ImageToEnergyTrainingSession
    | EnergyToImageTrainingSession
    | EnergyToImageReflowTrainingSession
    | ImageAutoencoderTrainingSession
)


def default_surrogate_training_config() -> LPAPSurrogateTrainingConfig:
    harmonics = SyntheticHarmonicConfig(
        harmonic_count=16,
        gain_variance=1.0,
        gain_half_life=4.0,
        spikiness_range=(4.0, 8.0),
        dtype=torch.float32,
    )
    return LPAPSurrogateTrainingConfig(
        data=LPAPSurrogateDataConfig(
            batch_size=32,
            bucket_count=128,
            probe_count=8,
            harmonics=harmonics,
        ),
        model=LPAPSurrogateModelConfig(
            k_max=4,
            hidden_dim=256,
            layer_count=8,
            head_count=8,
        ),
        optimizer=LPAPSurrogateOptimizerConfig(learning_rate=1.0e-3),
        validation=LPAPSurrogateValidationConfig(
            enabled=True,
            every=100,
            batch_size=256,
            seed=10_123,
            validate_at_end=True,
        ),
        run=LPAPSurrogateRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=10_000,
            seed=123,
            permutation_seed=123,
            display_every=5,
            log_every=1,
            run_id="surrogate_synthetic",
            checkpoint_name="surrogate_synthetic.pt",
            log_name="surrogate.sqlite",
            note="",
            tags=("baseline",),
            pinned=False,
        ),
    )


def default_decoder_training_config() -> LPAPDecoderTrainingConfig:
    return LPAPDecoderTrainingConfig(
        data=LPAPSurrogateDataConfig(
            batch_size=32,
            bucket_count=128,
            probe_count=8,
        ),
        decoder=LPAPDecoderModelConfig(
            frontend_initial_temperature=0.25,
            hidden_dim=256,
            layer_count=8,
            head_count=8,
        ),
        optimizer=LPAPSurrogateOptimizerConfig(learning_rate=1.0e-3),
        validation=LPAPSurrogateValidationConfig(
            enabled=True,
            every=100,
            batch_size=256,
            seed=20_123,
            validate_at_end=True,
        ),
        teacher=LPAPDecoderTeacherConfig(
            checkpoint_name="surrogate_synthetic.pt",
            load_best=True,
            require_checkpoint=True,
        ),
        regularization=LPAPDecoderRegularizationConfig(
            source_ce_weight=0.1,
            source_ce_l1_reference=0.05,
            source_ce_power=2.0,
        ),
        run=LPAPDecoderRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=10_000,
            seed=456,
            permutation_seed=123,
            display_every=5,
            log_every=1,
            run_id="decoder_synthetic",
            checkpoint_name="decoder_synthetic.pt",
            log_name="decoder.sqlite",
            note="",
            tags=("baseline",),
            pinned=False,
        ),
    )


def default_image_to_energy_training_config() -> ImageToEnergyTrainingConfig:
    harmonics = SyntheticHarmonicConfig(
        harmonic_count=16,
        gain_variance=1.0,
        gain_half_life=4.0,
        spikiness_range=(4.0, 8.0),
        dtype=torch.float32,
    )
    return ImageToEnergyTrainingConfig(
        image=ImageToEnergyImageConfig(
            dataset_path="data/images_32x32_gray.pt",
            batch_size=32,
            side=32,
            normalize=True,
            shuffle=True,
            num_workers=0,
        ),
        target=ImageToEnergyTargetConfig(harmonics=harmonics),
        flow=ImageToEnergyFlowConfig(
            sequence_length=1024,
            width=128,
            time_dim=128,
            dilation_cycles=2,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            kernel_size=3,
            zero_init_output=True,
        ),
        time=ImageToEnergyTimeConfig(
            distribution="beta",
            beta_alpha=0.1,
            beta_beta=0.1,
            eps=1.0e-4,
        ),
        optimizer=ImageToEnergyOptimizerConfig(
            learning_rate=1.0e-4,
            max_grad_norm=1.0,
        ),
        validation=ImageToEnergyValidationConfig(
            enabled=True,
            every=100,
            batch_size=128,
            seed=30_123,
            validate_at_end=True,
            euler_steps=(1, 4, 16),
        ),
        run=ImageToEnergyRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=10_000,
            seed=789,
            display_every=5,
            log_every=1,
            run_id="image_to_energy",
            checkpoint_name="image_to_energy.pt",
            log_name="image_to_energy.sqlite",
            note="",
            tags=("baseline",),
            pinned=False,
        ),
    )


def default_energy_to_image_training_config() -> EnergyToImageTrainingConfig:
    return EnergyToImageTrainingConfig(
        image=ImageToEnergyImageConfig(
            dataset_path="data/images_32x32_gray.pt",
            batch_size=32,
            side=32,
            normalize=True,
            shuffle=True,
            num_workers=0,
        ),
        source=EnergyToImageSourceConfig(
            surrogate_checkpoint_name="surrogate_synthetic.pt",
            decoder_checkpoint_name="decoder_synthetic.pt",
            load_best=True,
            require_checkpoints=True,
        ),
        flow=ImageToEnergyFlowConfig(
            sequence_length=1024,
            width=128,
            time_dim=128,
            dilation_cycles=2,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            kernel_size=3,
            zero_init_output=True,
        ),
        time=ImageToEnergyTimeConfig(
            distribution="beta",
            beta_alpha=0.1,
            beta_beta=0.1,
            eps=1.0e-4,
        ),
        optimizer=ImageToEnergyOptimizerConfig(
            learning_rate=1.0e-4,
            max_grad_norm=1.0,
        ),
        validation=ImageToEnergyValidationConfig(
            enabled=True,
            every=100,
            batch_size=128,
            seed=40_123,
            validate_at_end=True,
            euler_steps=(1, 4, 16),
        ),
        run=EnergyToImageRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=10_000,
            seed=987,
            display_every=5,
            log_every=1,
            run_id="energy_to_image",
            checkpoint_name="energy_to_image.pt",
            log_name="energy_to_image.sqlite",
            note="",
            tags=("baseline",),
            pinned=False,
        ),
    )


def default_energy_to_image_reflow_training_config() -> (
    EnergyToImageReflowTrainingConfig
):
    return EnergyToImageReflowTrainingConfig(
        image=ImageToEnergyImageConfig(
            dataset_path="data/images_32x32_gray.pt",
            batch_size=32,
            side=32,
            normalize=True,
            shuffle=True,
            num_workers=0,
        ),
        source=EnergyToImageSourceConfig(
            surrogate_checkpoint_name="surrogate_synthetic.pt",
            decoder_checkpoint_name="decoder_synthetic.pt",
            load_best=True,
            require_checkpoints=True,
        ),
        flow=ImageToEnergyFlowConfig(
            sequence_length=1024,
            width=128,
            time_dim=128,
            dilation_cycles=2,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            kernel_size=3,
            zero_init_output=True,
        ),
        teacher=EnergyToImageReflowTeacherConfig(
            checkpoint_name="energy_to_image.pt",
            load_best=True,
            require_checkpoint=True,
            teacher_steps=64,
            warm_start_student=True,
        ),
        reflow=EnergyToImageReflowConfig(
            student_steps=8,
            endpoint_l2_weight=1.0,
            image_anchor_l2_weight=0.25,
        ),
        optimizer=ImageToEnergyOptimizerConfig(
            learning_rate=5.0e-5,
            max_grad_norm=1.0,
        ),
        validation=ImageToEnergyValidationConfig(
            enabled=True,
            every=100,
            batch_size=128,
            seed=50_123,
            validate_at_end=True,
            euler_steps=(4, 8, 16),
        ),
        run=EnergyToImageReflowRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=10_000,
            seed=1_987,
            display_every=5,
            log_every=1,
            run_id="energy_to_image_reflow",
            checkpoint_name="energy_to_image_reflow_8.pt",
            log_name="energy_to_image_reflow.sqlite",
            note="",
            tags=("reflow", "8-step"),
            pinned=False,
        ),
    )


def default_image_autoencoder_training_config() -> ImageAutoencoderTrainingConfig:
    flow = ImageToEnergyFlowConfig(
        sequence_length=1024,
        width=128,
        time_dim=128,
        dilation_cycles=2,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size=3,
        zero_init_output=True,
    )
    return ImageAutoencoderTrainingConfig(
        image=ImageToEnergyImageConfig(
            dataset_path="data/images_32x32_gray.pt",
            batch_size=8,
            side=32,
            normalize=True,
            shuffle=True,
            num_workers=0,
        ),
        source=ImageAutoencoderSourceConfig(
            surrogate_checkpoint_name="surrogate_synthetic.pt",
            decoder_checkpoint_name="decoder_synthetic.pt",
            image_to_energy_checkpoint_name="image_to_energy.pt",
            energy_to_image_checkpoint_name="energy_to_image_reflow_8.pt",
            load_best=True,
            require_checkpoints=True,
            train_image_to_energy_flow=True,
            train_surrogate=True,
            train_decoder=True,
            train_energy_to_image_flow=True,
        ),
        image_to_energy_flow=flow,
        energy_to_image_flow=flow,
        integration=ImageAutoencoderIntegrationConfig(
            image_to_energy_steps=8,
            energy_to_image_steps=8,
        ),
        loss=ImageAutoencoderLossConfig(
            image_l2_weight=1.0,
            energy_l2_weight=0.25,
            energy_l1_weight=0.01,
            energy_l1_reference=0.05,
            surrogate_teacher_weight=0.1,
            detach_energy_target=False,
        ),
        optimizer=ImageToEnergyOptimizerConfig(
            learning_rate=2.0e-5,
            max_grad_norm=1.0,
        ),
        validation=ImageToEnergyValidationConfig(
            enabled=True,
            every=50,
            batch_size=8,
            seed=60_123,
            validate_at_end=True,
            euler_steps=(8,),
        ),
        run=ImageAutoencoderRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=10_000,
            seed=2_987,
            display_every=5,
            log_every=1,
            run_id="image_autoencoder",
            checkpoint_name="image_autoencoder.pt",
            log_name="image_autoencoder.sqlite",
            note="",
            tags=("e2e", "8-step"),
            pinned=False,
        ),
    )


def default_training_config(model_kind: TrainingModelKind) -> TrainingConfig:
    if model_kind == "surrogate":
        return default_surrogate_training_config()
    if model_kind == "decoder":
        return default_decoder_training_config()
    if model_kind == "image_to_energy":
        return default_image_to_energy_training_config()
    if model_kind == "energy_to_image":
        return default_energy_to_image_training_config()
    if model_kind == "energy_to_image_reflow":
        return default_energy_to_image_reflow_training_config()
    if model_kind == "image_autoencoder":
        return default_image_autoencoder_training_config()
    raise ValueError("unsupported model_kind")


def training_config_path(
    project_root: str | Path, model_kind: TrainingModelKind
) -> Path:
    return Path(project_root) / "configs" / "training" / f"{model_kind}.toml"


def training_config_from_file(
    path: str | Path, model_kind: TrainingModelKind
) -> TrainingConfig:
    with Path(path).open("rb") as file:
        data = tomllib.load(file)
    if model_kind == "surrogate":
        return lpap_surrogate_training_config_from_dict(data)
    if model_kind == "decoder":
        return lpap_decoder_training_config_from_dict(data)
    if model_kind == "image_to_energy":
        return image_to_energy_training_config_from_dict(data)
    if model_kind == "energy_to_image":
        return energy_to_image_training_config_from_dict(data)
    if model_kind == "energy_to_image_reflow":
        return energy_to_image_reflow_training_config_from_dict(data)
    if model_kind == "image_autoencoder":
        return image_autoencoder_training_config_from_dict(data)
    raise ValueError("unsupported model_kind")


def training_config_from_project_file(
    project_root: str | Path, model_kind: TrainingModelKind
) -> TrainingConfig:
    path = training_config_path(project_root, model_kind)
    if not path.exists():
        return default_training_config(model_kind)
    return training_config_from_file(path, model_kind)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def training_config_to_toml(config: TrainingConfig) -> str:
    data = config.as_run_config()
    lines: list[str] = []

    def write_table(name: str, table: dict[str, object]) -> None:
        lines.append(f"[{name}]")
        nested = []
        for key, value in table.items():
            if isinstance(value, dict):
                nested.append((key, value))
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
        for nested_key, nested_value in nested:
            write_table(f"{name}.{nested_key}", nested_value)

    for key, value in data.items():
        if not isinstance(value, dict):
            raise TypeError(f"expected config section for {key!r}")
        write_table(key, value)
    return "\n".join(lines).rstrip() + "\n"


def write_training_config_file(path: str | Path, config: TrainingConfig) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(training_config_to_toml(config), encoding="utf-8")
    return target


def restore_training_config_from_log(
    model_kind: TrainingModelKind,
    *,
    project_root: str | Path,
    run_id: str,
    resume_from_checkpoint: bool = False,
) -> Path:
    base_config = training_config_from_project_file(project_root, model_kind)
    config = training_config_from_log(
        model_kind,
        project_root=project_root,
        run_id=run_id,
        base_config=base_config,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return write_training_config_file(
        training_config_path(project_root, model_kind), config
    )


def training_log_path(project_root: str | Path, config: TrainingConfig) -> Path:
    return Path(project_root) / "training_logs" / config.run.log_name


def validation_regularizer_metric_names(config: TrainingConfig) -> tuple[str, ...]:
    if isinstance(config, LPAPDecoderTrainingConfig):
        if config.regularization.source_ce_weight > 0:
            return ("validation_source_ce_regularizer",)
    return ()


def training_config_from_log(
    model_kind: TrainingModelKind,
    *,
    project_root: str | Path,
    run_id: str,
    base_config: TrainingConfig | None = None,
    resume_from_checkpoint: bool = False,
) -> TrainingConfig:
    config = default_training_config(model_kind) if base_config is None else base_config
    log_path = training_log_path(project_root, config)
    if model_kind == "surrogate":
        return rerun_lpap_surrogate_training_config_from_log(
            log_path, run_id=run_id, resume_from_checkpoint=resume_from_checkpoint
        )
    if model_kind == "decoder":
        return rerun_lpap_decoder_training_config_from_log(
            log_path, run_id=run_id, resume_from_checkpoint=resume_from_checkpoint
        )
    if model_kind == "image_to_energy":
        return rerun_image_to_energy_training_config_from_log(
            log_path, run_id=run_id, resume_from_checkpoint=resume_from_checkpoint
        )
    if model_kind == "energy_to_image":
        return rerun_energy_to_image_training_config_from_log(
            log_path, run_id=run_id, resume_from_checkpoint=resume_from_checkpoint
        )
    if model_kind == "energy_to_image_reflow":
        return rerun_energy_to_image_reflow_training_config_from_log(
            log_path, run_id=run_id, resume_from_checkpoint=resume_from_checkpoint
        )
    return rerun_image_autoencoder_training_config_from_log(
        log_path, run_id=run_id, resume_from_checkpoint=resume_from_checkpoint
    )


def recent_training_runs(
    project_root: str | Path, config: TrainingConfig, *, limit: int = 10
) -> list[dict[str, object]]:
    return list_training_runs(
        training_log_path(project_root, config),
        base_run_id=config.run.run_id,
        limit=limit,
    )


def create_training_session(
    model_kind: TrainingModelKind,
    *,
    project_root: str | Path,
    config: TrainingConfig,
) -> TrainingSession:
    if model_kind == "surrogate":
        if not isinstance(config, LPAPSurrogateTrainingConfig):
            raise TypeError("surrogate training requires LPAPSurrogateTrainingConfig")
        return create_lpap_surrogate_training_session(
            project_root=project_root, config=config
        )
    if model_kind == "decoder":
        if not isinstance(config, LPAPDecoderTrainingConfig):
            raise TypeError("decoder training requires LPAPDecoderTrainingConfig")
        return create_lpap_decoder_training_session(
            project_root=project_root, config=config
        )
    if model_kind == "image_to_energy":
        if not isinstance(config, ImageToEnergyTrainingConfig):
            raise TypeError(
                "image_to_energy training requires ImageToEnergyTrainingConfig"
            )
        return create_image_to_energy_training_session(
            project_root=project_root, config=config
        )
    if model_kind == "energy_to_image":
        if not isinstance(config, EnergyToImageTrainingConfig):
            raise TypeError(
                "energy_to_image training requires EnergyToImageTrainingConfig"
            )
        return create_energy_to_image_training_session(
            project_root=project_root, config=config
        )
    if model_kind == "energy_to_image_reflow":
        if not isinstance(config, EnergyToImageReflowTrainingConfig):
            raise TypeError(
                "energy_to_image_reflow training requires EnergyToImageReflowTrainingConfig"
            )
        return create_energy_to_image_reflow_training_session(
            project_root=project_root, config=config
        )
    if not isinstance(config, ImageAutoencoderTrainingConfig):
        raise TypeError(
            "image_autoencoder training requires ImageAutoencoderTrainingConfig"
        )
    return create_image_autoencoder_training_session(
        project_root=project_root, config=config
    )


def iter_training(model_kind: TrainingModelKind, session: TrainingSession):
    if model_kind == "surrogate":
        if not isinstance(session, LPAPSurrogateTrainingSession):
            raise TypeError("surrogate training requires LPAPSurrogateTrainingSession")
        return iter_lpap_surrogate_training(session)
    if model_kind == "decoder":
        if not isinstance(session, LPAPDecoderTrainingSession):
            raise TypeError("decoder training requires LPAPDecoderTrainingSession")
        return iter_lpap_decoder_training(session)
    if model_kind == "image_to_energy":
        if not isinstance(session, ImageToEnergyTrainingSession):
            raise TypeError(
                "image_to_energy training requires ImageToEnergyTrainingSession"
            )
        return iter_image_to_energy_training(session)
    if model_kind == "energy_to_image":
        if not isinstance(session, EnergyToImageTrainingSession):
            raise TypeError(
                "energy_to_image training requires EnergyToImageTrainingSession"
            )
        return iter_energy_to_image_training(session)
    if model_kind == "energy_to_image_reflow":
        if not isinstance(session, EnergyToImageReflowTrainingSession):
            raise TypeError(
                "energy_to_image_reflow training requires EnergyToImageReflowTrainingSession"
            )
        return iter_energy_to_image_reflow_training(session)
    if not isinstance(session, ImageAutoencoderTrainingSession):
        raise TypeError(
            "image_autoencoder training requires ImageAutoencoderTrainingSession"
        )
    return iter_image_autoencoder_training(session)


def render_recent_runs_table(runs: list[dict[str, object]]) -> str:
    def metric_cell(value: object) -> str:
        return "" if value is None else f"{float(value):.4f}"

    rows = "".join(
        "<tr>"
        f"<td>{row['display_name']}</td>"
        f"<td>{row['run_id']}</td>"
        f"<td>{row['status']}</td>"
        f"<td>{row['last_step'] or ''}</td>"
        f"<td>{metric_cell(row['best_validation_loss'])}</td>"
        f"<td>{', '.join(row['tags'])}</td>"
        f"<td>{row['note']}</td>"
        "</tr>"
        for row in runs
    )
    return (
        "<table>"
        "<thead><tr><th>name</th><th>run instance</th><th>status</th>"
        "<th>last step</th><th>best validation loss</th><th>tags</th>"
        "<th>note</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )
