"""Image-only LPAP retraining pipeline (surrogates -> decoders -> AE -> exact fine-tune).

Keeps the bidirectional ``image_energy_flow`` checkpoint and its encoded energy
bank intact, then trains from scratch:
  1. ``surrogates`` (all configured teacher pairs)
  2. ``decoders`` (all configured teacher pairs, paired with fresh surrogates)
  3. ``autoencoder`` (tri-pair end-to-end AE initialized from flow + teachers)
  4. ``decoder_exact_finetune`` (optional decoder-only AE fine-tune on exact LPAP)
"""

from __future__ import annotations

import argparse
import json
import tomllib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch

from lpap.artifact_sync import ensure_project_artifact
from lpap.decoder_training import (
    create_lpap_decoder_training_session,
    iter_lpap_decoder_training,
)
from lpap.image_autoencoder_training import (
    ImageAutoencoderTrainingConfig,
    create_image_autoencoder_training_session,
    iter_image_autoencoder_training,
)
from lpap.notify import send_pushover
from lpap.surrogate_training import (
    create_lpap_surrogate_training_session,
    iter_lpap_surrogate_training,
)
from lpap.teacher_checkpoints import resolve_checkpoint_path
from lpap.teacher_config import project_teacher_config
from lpap.training_notebook import training_config_from_file

PipelineStage = Literal[
    "surrogates",
    "decoders",
    "autoencoder",
    "decoder_exact_finetune",
]
ALL_PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    "surrogates",
    "decoders",
    "autoencoder",
    "decoder_exact_finetune",
)


@dataclass(frozen=True)
class DecoderExactFinetuneConfig:
    steps: int = 3000
    learning_rate: float = 1.0e-4
    use_triton: bool = True
    image_l2_weight: float = 1.0
    energy_l1_weight: float = 0.05
    validation_every: int = 100
    validation_num_batches: int = 8
    display_every: int = 25
    log_every: int = 5
    run_id: str = "image_autoencoder_tri_exact_ft-v2"
    checkpoint_name: str = "image_autoencoder_tri_exact_ft-v2.pt"
    log_name: str = "image_autoencoder_tri_exact_ft-v2.sqlite"
    comment: str = "decoder-only exact LPAP fine-tune of tri-pair AE (v2)"

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("decoder_exact_finetune.steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("decoder_exact_finetune.learning_rate must be positive")
        if self.image_l2_weight < 0 or self.energy_l1_weight < 0:
            raise ValueError("decoder_exact_finetune loss weights must be non-negative")
        if self.image_l2_weight == 0 and self.energy_l1_weight == 0:
            raise ValueError(
                "decoder_exact_finetune requires a positive image_l2 or energy_l1 weight"
            )
        if self.validation_every <= 0 or self.validation_num_batches <= 0:
            raise ValueError("decoder_exact_finetune validation counts must be positive")
        if self.display_every <= 0 or self.log_every <= 0:
            raise ValueError("decoder_exact_finetune cadence values must be positive")


@dataclass(frozen=True)
class PipelinePerformanceConfig:
    """GPU throughput knobs (defaults keep the original fp32 behaviour).

    ``autocast_dtype``: ``"none"`` or ``"bf16"``; wraps every stage in CUDA
    autocast. AE flows always integrate in fp32 regardless (see
    ``ImageAutoencoderModel``). ``allow_tf32`` enables TF32 matmul/cuDNN.
    ``energy_bank_on_device`` moves the teacher bank to the training device.
    ``compile_flows`` wraps AE flow ``forward`` methods with ``torch.compile``
    (state-dict keys are unchanged).
    """

    autocast_dtype: str = "none"
    allow_tf32: bool = False
    energy_bank_on_device: bool = False
    compile_flows: bool = False

    def validate(self) -> None:
        if self.autocast_dtype not in ("none", "bf16"):
            raise ValueError("performance.autocast_dtype must be 'none' or 'bf16'")


@dataclass(frozen=True)
class RetrainPipelineConfig:
    flow_checkpoint: str
    energy_bank: str
    image_dataset: str
    teacher_configs: tuple[str, ...]
    autoencoder_config: str
    resume_from_checkpoint: bool = False
    ensure_artifacts: bool = True
    upload_artifacts_on_checkpoint: bool = True
    notify_on_finished: bool = True
    run_surrogates: bool = True
    run_decoders: bool = True
    run_autoencoder: bool = True
    run_decoder_exact_finetune: bool = True
    decoder_exact_finetune: DecoderExactFinetuneConfig = DecoderExactFinetuneConfig()
    performance: PipelinePerformanceConfig = PipelinePerformanceConfig()

    def validate(self) -> None:
        if not self.teacher_configs:
            raise ValueError("pipeline.teacher_configs must be non-empty")
        if not self.flow_checkpoint.strip():
            raise ValueError("pipeline.flow_checkpoint must be non-empty")
        if not self.energy_bank.strip():
            raise ValueError("pipeline.energy_bank must be non-empty")
        if not self.image_dataset.strip():
            raise ValueError("pipeline.image_dataset must be non-empty")
        if not self.autoencoder_config.strip():
            raise ValueError("pipeline.autoencoder_config must be non-empty")
        self.decoder_exact_finetune.validate()
        self.performance.validate()

    def enabled_stages(self) -> tuple[PipelineStage, ...]:
        flags: tuple[tuple[PipelineStage, bool], ...] = (
            ("surrogates", self.run_surrogates),
            ("decoders", self.run_decoders),
            ("autoencoder", self.run_autoencoder),
            ("decoder_exact_finetune", self.run_decoder_exact_finetune),
        )
        return tuple(stage for stage, enabled in flags if enabled)


@dataclass(frozen=True)
class RetrainStageResult:
    stage: PipelineStage
    name: str
    steps_completed: int
    checkpoint_path: str
    best_metric: float | None
    last_metrics: dict[str, float]


def load_retrain_pipeline_config(path: str | Path) -> RetrainPipelineConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        data = tomllib.load(file)
    pipeline = data.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError(f"pipeline TOML missing [pipeline]: {config_path}")
    stages = data.get("stages") or {}
    exact_raw = data.get("decoder_exact_finetune") or {}
    perf_raw = data.get("performance") or {}
    config = RetrainPipelineConfig(
        flow_checkpoint=str(pipeline["flow_checkpoint"]),
        energy_bank=str(pipeline["energy_bank"]),
        image_dataset=str(pipeline["image_dataset"]),
        teacher_configs=tuple(str(item) for item in pipeline["teacher_configs"]),
        autoencoder_config=str(pipeline["autoencoder_config"]),
        resume_from_checkpoint=bool(pipeline.get("resume_from_checkpoint", False)),
        ensure_artifacts=bool(pipeline.get("ensure_artifacts", True)),
        upload_artifacts_on_checkpoint=bool(
            pipeline.get("upload_artifacts_on_checkpoint", True)
        ),
        notify_on_finished=bool(pipeline.get("notify_on_finished", True)),
        run_surrogates=bool(stages.get("run_surrogates", True)),
        run_decoders=bool(stages.get("run_decoders", True)),
        run_autoencoder=bool(stages.get("run_autoencoder", True)),
        run_decoder_exact_finetune=bool(
            stages.get("run_decoder_exact_finetune", True)
        ),
        decoder_exact_finetune=DecoderExactFinetuneConfig(
            steps=int(exact_raw.get("steps", 3000)),
            learning_rate=float(exact_raw.get("learning_rate", 1.0e-4)),
            use_triton=bool(exact_raw.get("use_triton", True)),
            image_l2_weight=float(exact_raw.get("image_l2_weight", 1.0)),
            energy_l1_weight=float(exact_raw.get("energy_l1_weight", 0.05)),
            validation_every=int(exact_raw.get("validation_every", 100)),
            validation_num_batches=int(exact_raw.get("validation_num_batches", 8)),
            display_every=int(exact_raw.get("display_every", 25)),
            log_every=int(exact_raw.get("log_every", 5)),
            run_id=str(exact_raw.get("run_id", "image_autoencoder_tri_exact_ft-v2")),
            checkpoint_name=str(
                exact_raw.get("checkpoint_name", "image_autoencoder_tri_exact_ft-v2.pt")
            ),
            log_name=str(
                exact_raw.get("log_name", "image_autoencoder_tri_exact_ft-v2.sqlite")
            ),
            comment=str(
                exact_raw.get(
                    "comment", "decoder-only exact LPAP fine-tune of tri-pair AE (v2)"
                )
            ),
        ),
        performance=PipelinePerformanceConfig(
            autocast_dtype=str(perf_raw.get("autocast_dtype", "none")),
            allow_tf32=bool(perf_raw.get("allow_tf32", False)),
            energy_bank_on_device=bool(perf_raw.get("energy_bank_on_device", False)),
            compile_flows=bool(perf_raw.get("compile_flows", False)),
        ),
    )
    config.validate()
    return config


def build_exact_finetune_autoencoder_config(
    base: ImageAutoencoderTrainingConfig,
    exact: DecoderExactFinetuneConfig,
    *,
    resume_from_checkpoint: bool,
    upload_artifacts_on_checkpoint: bool,
    notify_on_finished: bool,
) -> ImageAutoencoderTrainingConfig:
    """Derive Stage-4 decoder-only exact-LPAP AE config from Stage-3 AE config."""
    config = replace(
        base,
        source=replace(
            base.source,
            load_best=True,
            require_checkpoints=True,
            train_image_to_energy_flow=False,
            train_surrogate=False,
            train_decoder=True,
            train_energy_to_image_flow=False,
            lpap_frontend_mode="exact",
            init_checkpoint_name=base.run.checkpoint_name,
            use_triton=exact.use_triton,
        ),
        loss=replace(
            base.loss,
            image_l2_weight=exact.image_l2_weight,
            energy_l1_weight=exact.energy_l1_weight,
            surrogate_teacher_weight=0.0,
            signed_mass_balance_weight=0.0,
            detach_energy_target=True,
        ),
        optimizer=replace(base.optimizer, learning_rate=exact.learning_rate),
        validation=replace(
            base.validation,
            every=exact.validation_every,
            num_batches=exact.validation_num_batches,
        ),
        run=replace(
            base.run,
            run_training=True,
            resume_from_checkpoint=resume_from_checkpoint,
            steps=exact.steps,
            display_every=exact.display_every,
            log_every=exact.log_every,
            run_id=exact.run_id,
            checkpoint_name=exact.checkpoint_name,
            log_name=exact.log_name,
            comment=exact.comment,
            upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
            notify_on_finished=notify_on_finished,
        ),
    )
    config.validate()
    return config


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ensure_prerequisites(
    root: Path,
    config: RetrainPipelineConfig,
    stages: Sequence[PipelineStage],
) -> None:
    need_teachers = "surrogates" in stages or "decoders" in stages
    need_ae = "autoencoder" in stages or "decoder_exact_finetune" in stages
    if need_teachers:
        bank_path = _resolve_path(root, config.energy_bank)
        if config.ensure_artifacts and not bank_path.is_file():
            ensure_project_artifact(bank_path, project_root=root)
        if not bank_path.is_file():
            raise FileNotFoundError(f"missing energy bank: {bank_path}")
    if need_ae:
        flow_path = resolve_checkpoint_path(
            root, config.flow_checkpoint, ensure=config.ensure_artifacts
        )
        if not flow_path.is_file():
            raise FileNotFoundError(f"missing flow checkpoint: {flow_path}")
        dataset_path = _resolve_path(root, config.image_dataset)
        if config.ensure_artifacts and not dataset_path.is_file():
            ensure_project_artifact(dataset_path, project_root=root)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"missing image dataset: {dataset_path}")


def _stage_autocast(perf: PipelinePerformanceConfig, device: Any) -> Any:
    import contextlib

    if perf.autocast_dtype != "bf16":
        return contextlib.nullcontext()
    if device is not None:
        target = torch.device(device)
    else:
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "performance.autocast_dtype='bf16' requires a CUDA device with "
                "bfloat16 support (torch.cuda.is_bf16_supported() is False); "
                "set performance.autocast_dtype='none'"
            )
    elif target.type != "cpu":
        raise RuntimeError(
            f"performance.autocast_dtype='bf16' is unsupported on device {target!s}; "
            "set performance.autocast_dtype='none'"
        )
    # cache_enabled=False: this context spans a whole stage (many optimizer
    # steps). The autocast weight-cast cache is only cleared on context exit,
    # so caching would keep feeding stale bf16 weight copies to the forward.
    return torch.autocast(
        device_type=target.type, dtype=torch.bfloat16, cache_enabled=False
    )


def _bank_to_device(session: Any, perf: PipelinePerformanceConfig) -> Any:
    if not perf.energy_bank_on_device:
        return session
    return replace(session, energy_bank=session.energy_bank.to(session.device))


def _maybe_compile_flows(session: Any, perf: PipelinePerformanceConfig) -> None:
    if not perf.compile_flows:
        return
    import torch

    for name in ("image_to_energy_flow", "energy_to_image_flow"):
        flow = getattr(session.model, name)
        flow.forward = torch.compile(flow.forward, dynamic=False)


def _consume_training_iter(
    iterator: Any,
    *,
    stage: PipelineStage,
    name: str,
    checkpoint_path: Path,
) -> RetrainStageResult:
    last_step = 0
    best_metric: float | None = None
    last_metrics: dict[str, float] = {}
    for result in iterator:
        last_step = int(result.step)
        best_metric = result.best_metric
        last_metrics = {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, int | float)
        }
        if result.should_display or result.improved or result.step % 50 == 0:
            loss = result.metrics.get("loss", float("nan"))
            vloss = result.metrics.get("validation_loss")
            vtxt = "n/a" if vloss is None else f"{vloss:.5f}"
            btxt = "n/a" if best_metric is None else f"{best_metric:.5f}"
            mark = " *" if result.improved else ""
            ck = " ckpt" if result.checkpointed else ""
            print(
                f"[{stage}:{name}] step={result.step} loss={loss:.5f} "
                f"val={vtxt} best={btxt}{mark}{ck}",
                flush=True,
            )
    return RetrainStageResult(
        stage=stage,
        name=name,
        steps_completed=last_step,
        checkpoint_path=str(checkpoint_path),
        best_metric=best_metric,
        last_metrics=last_metrics,
    )


def run_image_retrain_pipeline(
    *,
    project_root: str | Path,
    config: RetrainPipelineConfig,
    stages: Sequence[PipelineStage] | None = None,
    device: str | torch.device | None = None,
    resume: bool | None = None,
    upload_artifacts_on_checkpoint: bool | None = None,
    notify_on_finished: bool | None = None,
    step_overrides: dict[PipelineStage, int] | None = None,
) -> list[RetrainStageResult]:
    """Execute selected stages of the image-only retraining pipeline in order."""
    config.validate()
    root = Path(project_root)
    selected_stages = (
        tuple(stages) if stages is not None else config.enabled_stages()
    )
    for stage in selected_stages:
        if stage not in ALL_PIPELINE_STAGES:
            raise ValueError(f"unknown pipeline stage: {stage!r}")
    ordered_stages = tuple(
        stage for stage in ALL_PIPELINE_STAGES if stage in selected_stages
    )
    if not ordered_stages:
        return []

    resolved_resume = (
        config.resume_from_checkpoint if resume is None else bool(resume)
    )
    resolved_upload = (
        config.upload_artifacts_on_checkpoint
        if upload_artifacts_on_checkpoint is None
        else bool(upload_artifacts_on_checkpoint)
    )
    resolved_notify = (
        config.notify_on_finished
        if notify_on_finished is None
        else bool(notify_on_finished)
    )
    overrides = step_overrides or {}
    perf = config.performance
    # Fail loudly before loading artifacts or starting stages if bf16 is unsupported.
    _stage_autocast(perf, device)
    if perf.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    results: list[RetrainStageResult] = []
    current_stage = "prerequisites"

    try:
        _ensure_prerequisites(root, config, ordered_stages)

        if "surrogates" in ordered_stages:
            current_stage = "surrogates"
            for teacher_relpath in config.teacher_configs:
                teacher_path = _resolve_path(root, teacher_relpath)
                surrogate_cfg = project_teacher_config(
                    teacher_path,
                    "surrogate",
                    resume_from_checkpoint=resolved_resume,
                )
                if "surrogates" in overrides:
                    surrogate_cfg = replace(
                        surrogate_cfg,
                        run=replace(
                            surrogate_cfg.run, steps=int(overrides["surrogates"])
                        ),
                    )
                session = _bank_to_device(
                    create_lpap_surrogate_training_session(
                        project_root=root,
                        config=surrogate_cfg,
                        device=device,
                    ),
                    perf,
                )
                session.training_run.config = replace(
                    session.training_run.config,
                    upload_artifacts_on_checkpoint=resolved_upload,
                    notify_on_finished=False,
                )
                with _stage_autocast(perf, device):
                    results.append(
                        _consume_training_iter(
                            iter_lpap_surrogate_training(session),
                            stage="surrogates",
                            name=surrogate_cfg.run.run_id,
                            checkpoint_path=session.checkpoint_path,
                        )
                    )

        if "decoders" in ordered_stages:
            current_stage = "decoders"
            for teacher_relpath in config.teacher_configs:
                teacher_path = _resolve_path(root, teacher_relpath)
                decoder_cfg = project_teacher_config(
                    teacher_path,
                    "decoder",
                    resume_from_checkpoint=resolved_resume,
                )
                if "decoders" in overrides:
                    decoder_cfg = replace(
                        decoder_cfg,
                        run=replace(decoder_cfg.run, steps=int(overrides["decoders"])),
                    )
                session = _bank_to_device(
                    create_lpap_decoder_training_session(
                        project_root=root,
                        config=decoder_cfg,
                        device=device,
                    ),
                    perf,
                )
                session.training_run.config = replace(
                    session.training_run.config,
                    upload_artifacts_on_checkpoint=resolved_upload,
                    notify_on_finished=False,
                )
                with _stage_autocast(perf, device):
                    results.append(
                        _consume_training_iter(
                            iter_lpap_decoder_training(session),
                            stage="decoders",
                            name=decoder_cfg.run.run_id,
                            checkpoint_path=session.checkpoint_path,
                        )
                    )

        ae_base_cfg: ImageAutoencoderTrainingConfig | None = None
        if (
            "autoencoder" in ordered_stages
            or "decoder_exact_finetune" in ordered_stages
        ):
            ae_path = _resolve_path(root, config.autoencoder_config)
            loaded_ae = training_config_from_file(ae_path, "image_autoencoder")
            if not isinstance(loaded_ae, ImageAutoencoderTrainingConfig):
                raise TypeError(
                    "autoencoder_config did not load ImageAutoencoderTrainingConfig"
                )
            ae_base_cfg = loaded_ae

        if "autoencoder" in ordered_stages:
            current_stage = "autoencoder"
            assert ae_base_cfg is not None
            ae_cfg = replace(
                ae_base_cfg,
                run=replace(
                    ae_base_cfg.run,
                    run_training=True,
                    resume_from_checkpoint=resolved_resume,
                    steps=int(overrides.get("autoencoder", ae_base_cfg.run.steps)),
                    upload_artifacts_on_checkpoint=resolved_upload,
                    notify_on_finished=False,
                ),
            )
            session = create_image_autoencoder_training_session(
                project_root=root,
                config=ae_cfg,
                device=device,
            )
            _maybe_compile_flows(session, perf)
            with _stage_autocast(perf, device):
                results.append(
                    _consume_training_iter(
                        iter_image_autoencoder_training(session),
                        stage="autoencoder",
                        name=ae_cfg.run.run_id,
                        checkpoint_path=session.checkpoint_path,
                    )
                )

        if "decoder_exact_finetune" in ordered_stages:
            current_stage = "decoder_exact_finetune"
            assert ae_base_cfg is not None
            exact_cfg = config.decoder_exact_finetune
            if "decoder_exact_finetune" in overrides:
                exact_cfg = replace(
                    exact_cfg, steps=int(overrides["decoder_exact_finetune"])
                )
            ft_ae_cfg = build_exact_finetune_autoencoder_config(
                ae_base_cfg,
                exact_cfg,
                resume_from_checkpoint=resolved_resume,
                upload_artifacts_on_checkpoint=resolved_upload,
                notify_on_finished=False,
            )
            session = create_image_autoencoder_training_session(
                project_root=root,
                config=ft_ae_cfg,
                device=device,
            )
            _maybe_compile_flows(session, perf)
            with _stage_autocast(perf, device):
                results.append(
                    _consume_training_iter(
                        iter_image_autoencoder_training(session),
                        stage="decoder_exact_finetune",
                        name=ft_ae_cfg.run.run_id,
                        checkpoint_path=session.checkpoint_path,
                    )
                )
    except Exception as exc:
        if resolved_notify:
            failure_lines = [
                f"stage={current_stage}",
                f"error={type(exc).__name__}: {exc}",
            ]
            for item in results:
                best_str = (
                    "n/a" if item.best_metric is None else f"{item.best_metric:.5f}"
                )
                failure_lines.append(
                    f"done {item.stage}/{item.name}: "
                    f"steps={item.steps_completed} best={best_str}"
                )
            try:
                send_pushover(
                    "\n".join(failure_lines),
                    title="LPAP image retrain pipeline failed",
                )
            except Exception as notify_exc:
                warnings.warn(
                    f"Pushover failure notification failed: {notify_exc}",
                    stacklevel=2,
                )
        raise

    if resolved_notify:
        summary_lines = [
            f"{item.stage}/{item.name}: steps={item.steps_completed} "
            f"best={'n/a' if item.best_metric is None else f'{item.best_metric:.5f}'}"
            for item in results
        ]
        send_pushover(
            "\n".join(summary_lines) or "no stages executed",
            title="LPAP image retrain pipeline finished",
        )

    return results


def _parse_stages(raw: str | None) -> list[PipelineStage] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    parsed: list[PipelineStage] = []
    for item in items:
        if item not in ALL_PIPELINE_STAGES:
            raise ValueError(
                f"invalid stage {item!r}; choose from {', '.join(ALL_PIPELINE_STAGES)}"
            )
        parsed.append(item)  # type: ignore[arg-type]
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/retrain_image_pipeline.toml"),
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages: surrogates,decoders,autoencoder,decoder_exact_finetune",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--surrogate-steps", type=int, default=None)
    parser.add_argument("--decoder-steps", type=int, default=None)
    parser.add_argument("--ae-steps", type=int, default=None)
    parser.add_argument("--exact-ft-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = (
        args.config
        if args.config.is_absolute()
        else args.project_root / args.config
    )
    config = load_retrain_pipeline_config(config_path)
    stages = _parse_stages(args.stages)
    step_overrides: dict[PipelineStage, int] = {}
    if args.surrogate_steps is not None:
        step_overrides["surrogates"] = args.surrogate_steps
    if args.decoder_steps is not None:
        step_overrides["decoders"] = args.decoder_steps
    if args.ae_steps is not None:
        step_overrides["autoencoder"] = args.ae_steps
    if args.exact_ft_steps is not None:
        step_overrides["decoder_exact_finetune"] = args.exact_ft_steps
    results = run_image_retrain_pipeline(
        project_root=args.project_root,
        config=config,
        stages=stages,
        device=args.device,
        resume=True if args.resume else None,
        upload_artifacts_on_checkpoint=False if args.no_upload else None,
        notify_on_finished=False if args.no_notify else None,
        step_overrides=step_overrides or None,
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "stage": item.stage,
                        "name": item.name,
                        "steps_completed": item.steps_completed,
                        "checkpoint_path": item.checkpoint_path,
                        "best_metric": item.best_metric,
                        "last_metrics": item.last_metrics,
                    }
                    for item in results
                ],
                indent=2,
                sort_keys=True,
            )
        )
    print("RETRAIN_IMAGE_PIPELINE_DONE", flush=True)
    return 0


__all__ = [
    "ALL_PIPELINE_STAGES",
    "DecoderExactFinetuneConfig",
    "PipelineStage",
    "RetrainPipelineConfig",
    "RetrainStageResult",
    "build_exact_finetune_autoencoder_config",
    "load_retrain_pipeline_config",
    "main",
    "run_image_retrain_pipeline",
]


if __name__ == "__main__":
    raise SystemExit(main())
