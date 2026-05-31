from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from lpap.checkpoints import load_training_checkpoint
from lpap.data import SyntheticHarmonicConfig, load_image_tensor_dataset
from lpap.flow import (
    DilatedConvFlow1d,
    FlowMatchingMetrics,
    flow_matching_loss,
    integrate_euler_midpoint_time,
)
from lpap.hilbert import hilbert_flatten_images, hilbert_unflatten_images
from lpap.training import TrainingRun, TrainingRunConfig, TrainingResumeInfo


@dataclass(frozen=True)
class FlowImageConfig:
    dataset_path: str = "data/images_32x32_gray.pt"
    batch_size: int = 32
    side: int = 32
    normalize: bool = True
    shuffle: bool = True
    num_workers: int = 0

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("image batch_size must be positive")
        if self.side <= 0:
            raise ValueError("image side must be positive")
        if self.num_workers < 0:
            raise ValueError("image num_workers must be non-negative")

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "dataset_path": self.dataset_path,
            "batch_size": self.batch_size,
            "side": self.side,
            "normalize": self.normalize,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True)
class FlowModelConfig:
    sequence_length: int = 1024
    width: int = 128
    time_dim: int = 128
    dilation_cycles: int = 2
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    kernel_size: int = 3
    zero_init_output: bool = True

    def validate(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.time_dim <= 0:
            raise ValueError("time_dim must be positive")
        if self.dilation_cycles <= 0:
            raise ValueError("dilation_cycles must be positive")
        if not self.dilations or any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("dilations must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

    def as_dict(self) -> dict[str, int | bool | tuple[int, ...]]:
        return {
            "sequence_length": self.sequence_length,
            "width": self.width,
            "time_dim": self.time_dim,
            "dilation_cycles": self.dilation_cycles,
            "dilations": self.dilations,
            "kernel_size": self.kernel_size,
            "zero_init_output": self.zero_init_output,
        }


@dataclass(frozen=True)
class FlowTimeConfig:
    distribution: Literal["beta", "uniform"] = "beta"
    beta_alpha: float = 0.1
    beta_beta: float = 0.1
    eps: float = 1.0e-4

    def validate(self) -> None:
        if self.distribution not in ("beta", "uniform"):
            raise ValueError("time distribution must be 'beta' or 'uniform'")
        if self.beta_alpha <= 0 or self.beta_beta <= 0:
            raise ValueError("beta parameters must be positive")
        if not 0.0 <= self.eps < 0.5:
            raise ValueError("eps must be in [0, 0.5)")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "distribution": self.distribution,
            "beta_alpha": self.beta_alpha,
            "beta_beta": self.beta_beta,
            "eps": self.eps,
        }


@dataclass(frozen=True)
class FlowOptimizerConfig:
    learning_rate: float = 1.0e-4
    max_grad_norm: float | None = 1.0

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def as_dict(self) -> dict[str, float | None]:
        return {
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass(frozen=True)
class FlowValidationConfig:
    enabled: bool = True
    every: int = 100
    batch_size: int = 128
    seed: int = 30_123
    validate_at_end: bool = True
    euler_steps: tuple[int, ...] = (1, 4, 16)

    def validate(self) -> None:
        if self.every <= 0:
            raise ValueError("validation every must be positive")
        if self.batch_size <= 0:
            raise ValueError("validation batch_size must be positive")
        if any(steps <= 0 for steps in self.euler_steps):
            raise ValueError("validation euler_steps must be positive")

    def as_dict(self) -> dict[str, int | bool | tuple[int, ...]]:
        return {
            "enabled": self.enabled,
            "every": self.every,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "validate_at_end": self.validate_at_end,
            "euler_steps": self.euler_steps,
        }


@dataclass(frozen=True)
class FlowRunParams:
    run_id: str
    checkpoint_name: str
    log_name: str
    total_steps: int
    resume_from_checkpoint: bool
    display_every: int
    log_every: int
    note: str
    tags: tuple[str, ...]
    pinned: bool


def flow_run_params_from_config(run: object) -> FlowRunParams:
    return FlowRunParams(
        run_id=str(getattr(run, "run_id")),
        checkpoint_name=str(getattr(run, "checkpoint_name")),
        log_name=str(getattr(run, "log_name")),
        total_steps=int(getattr(run, "steps")),
        resume_from_checkpoint=bool(getattr(run, "resume_from_checkpoint")),
        display_every=int(getattr(run, "display_every")),
        log_every=int(getattr(run, "log_every")),
        note=str(getattr(run, "note")),
        tags=tuple(str(tag) for tag in getattr(run, "tags")),
        pinned=bool(getattr(run, "pinned")),
    )


@dataclass(frozen=True)
class FlowSessionCore:
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


def image_config_from_dict(data: dict[str, Any]) -> FlowImageConfig:
    return FlowImageConfig(
        dataset_path=str(data["dataset_path"]),
        batch_size=int(data["batch_size"]),
        side=int(data["side"]),
        normalize=bool(data["normalize"]),
        shuffle=bool(data["shuffle"]),
        num_workers=int(data["num_workers"]),
    )


def flow_model_config_from_dict(data: dict[str, Any]) -> FlowModelConfig:
    return FlowModelConfig(
        sequence_length=int(data["sequence_length"]),
        width=int(data["width"]),
        time_dim=int(data["time_dim"]),
        dilation_cycles=int(data["dilation_cycles"]),
        dilations=tuple(int(value) for value in data["dilations"]),
        kernel_size=int(data["kernel_size"]),
        zero_init_output=bool(data["zero_init_output"]),
    )


def time_config_from_dict(data: dict[str, Any]) -> FlowTimeConfig:
    return FlowTimeConfig(
        distribution=str(data["distribution"]),  # type: ignore[arg-type]
        beta_alpha=float(data["beta_alpha"]),
        beta_beta=float(data["beta_beta"]),
        eps=float(data["eps"]),
    )


def optimizer_config_from_dict(data: dict[str, Any]) -> FlowOptimizerConfig:
    return FlowOptimizerConfig(
        learning_rate=float(data["learning_rate"]),
        max_grad_norm=(
            None if data.get("max_grad_norm") is None else float(data["max_grad_norm"])
        ),
    )


def validation_config_from_dict(data: dict[str, Any]) -> FlowValidationConfig:
    return FlowValidationConfig(
        enabled=bool(data["enabled"]),
        every=int(data["every"]),
        batch_size=int(data["batch_size"]),
        seed=int(data["seed"]),
        validate_at_end=bool(data["validate_at_end"]),
        euler_steps=tuple(int(value) for value in data["euler_steps"]),
    )


def load_flow_checkpoint_state(
    *,
    path: str | Path,
    load_best: bool,
    require_checkpoint: bool,
    device: str | torch.device,
) -> dict[str, torch.Tensor] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        if require_checkpoint:
            raise FileNotFoundError(f"flow checkpoint not found: {checkpoint_path}")
        return None
    payload = load_training_checkpoint(
        checkpoint_path, map_location=torch.device(device)
    )
    state = payload.get("best_model_state") if load_best else payload.get("model_state")
    if state is None:
        state = payload["model_state"]
    return state


def validate_image_flow_shape(*, image: FlowImageConfig, flow: FlowModelConfig) -> None:
    if flow.sequence_length != image.side * image.side:
        raise ValueError("flow sequence_length must equal image side squared")


def flow_model_metadata(
    *, image: FlowImageConfig, flow: FlowModelConfig, extra: dict[str, object]
) -> dict[str, object]:
    return {
        "sequence_length": flow.sequence_length,
        "side": image.side,
        "width": flow.width,
        "time_dim": flow.time_dim,
        "dilation_cycles": flow.dilation_cycles,
        "dilations": flow.dilations,
        "kernel_size": flow.kernel_size,
        "zero_init_output": flow.zero_init_output,
        **extra,
    }


def sample_flow_time(
    *,
    batch_size: int,
    config: FlowTimeConfig,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    config.validate()
    target_device = torch.device("cpu") if device is None else torch.device(device)
    if config.distribution == "uniform":
        base = torch.rand(
            batch_size, generator=generator, device=target_device, dtype=dtype
        )
    else:
        alpha = torch.full(
            (batch_size,), config.beta_alpha, device=target_device, dtype=dtype
        )
        beta = torch.full(
            (batch_size,), config.beta_beta, device=target_device, dtype=dtype
        )
        # Beta(a, b) is sampled as the ratio of two Gamma draws so the result is
        # reproducible from `generator`. `torch._standard_gamma` is intentionally
        # used here because the public `torch.distributions` / `torch.Tensor.gamma_`
        # samplers do not accept an explicit Generator, which would break the
        # deterministic resume guarantees the training stack relies on.
        gamma_alpha = torch._standard_gamma(alpha, generator=generator)
        gamma_beta = torch._standard_gamma(beta, generator=generator)
        base = gamma_alpha / (gamma_alpha + gamma_beta).clamp_min(
            torch.finfo(dtype).tiny
        )
    return config.eps + (1.0 - 2.0 * config.eps) * base


def load_flow_image_loader(
    *,
    root: Path,
    config: FlowImageConfig,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[Path, DataLoader]:
    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    dataset = load_image_tensor_dataset(dataset_path, normalize=config.normalize)
    if dataset.images.shape[2:] != (config.side, config.side):
        raise ValueError("image dataset side does not match config")
    generator = torch.Generator().manual_seed(seed)
    return dataset_path, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
        generator=generator,
    )


def create_flow_session_core(
    *,
    project_root: str | Path,
    image: FlowImageConfig,
    flow: FlowModelConfig,
    optimizer: FlowOptimizerConfig,
    validation: FlowValidationConfig,
    run: FlowRunParams,
    seed: int,
    run_config: dict[str, object],
    model_config: dict[str, object],
    metadata: dict[str, object] | None = None,
    device: str | torch.device | None = None,
) -> FlowSessionCore:
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    torch.manual_seed(seed)
    root = Path(project_root)
    checkpoint_path = root / "checkpoints" / run.checkpoint_name
    log_path = root / "training_logs" / run.log_name
    image_dataset_path, image_loader = load_flow_image_loader(
        root=root,
        config=image,
        batch_size=image.batch_size,
        shuffle=image.shuffle,
        seed=seed,
    )
    _validation_image_dataset_path, validation_image_loader = load_flow_image_loader(
        root=root,
        config=image,
        batch_size=validation.batch_size,
        shuffle=True,
        seed=validation.seed,
    )
    flow_model = DilatedConvFlow1d(
        sequence_length=flow.sequence_length,
        width=flow.width,
        time_dim=flow.time_dim,
        dilation_cycles=flow.dilation_cycles,
        dilations=flow.dilations,
        kernel_size=flow.kernel_size,
        zero_init_output=flow.zero_init_output,
    ).to(target_device)
    optimizer_instance = torch.optim.AdamW(
        flow_model.parameters(), lr=optimizer.learning_rate
    )
    training_run = TrainingRun(
        config=TrainingRunConfig(
            run_id=run.run_id,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            total_steps=run.total_steps,
            monitor="validation_loss",
            mode="min",
            resume=run.resume_from_checkpoint,
            checkpoint_every=None,
            checkpoint_on_improvement=True,
            checkpoint_at_end=False,
            log_every=run.log_every,
            display_every=run.display_every,
            note=run.note,
            tags=run.tags,
            pinned=run.pinned,
        ),
        model=flow_model,
        optimizer=optimizer_instance,
        run_config=run_config,
        model_config=model_config,
        metadata={
            "device": str(target_device),
            "image_dataset_path": str(image_dataset_path),
            **({} if metadata is None else metadata),
        },
    )
    resume_info = training_run.resume_or_initialize()
    generator = torch.Generator(device=target_device).manual_seed(
        seed + resume_info.start_step
    )
    validation_generator = torch.Generator(device=target_device).manual_seed(
        validation.seed + resume_info.start_step
    )
    return FlowSessionCore(
        device=target_device,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        image_dataset_path=image_dataset_path,
        image_loader=image_loader,
        validation_image_loader=validation_image_loader,
        flow=flow_model,
        optimizer=optimizer_instance,
        training_run=training_run,
        generator=generator,
        validation_generator=validation_generator,
        resume_info=resume_info,
    )


def prepare_image_sequence(
    images: torch.Tensor, *, side: int, device: torch.device
) -> torch.Tensor:
    images = images.to(device=device, dtype=torch.float32)
    return hilbert_flatten_images(images, side=side)


def sample_harmonic_values(
    *,
    harmonics: SyntheticHarmonicConfig,
    batch_size: int,
    n: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    values = harmonics.sample_batch(
        batch_size=batch_size,
        n=n,
        generator=generator,
        device=device,
    )
    if not isinstance(values, torch.Tensor):
        raise TypeError("expected harmonic values tensor")
    return values


def train_flow_matching_step(
    *,
    model: DilatedConvFlow1d,
    optimizer: torch.optim.Optimizer,
    start: torch.Tensor,
    end: torch.Tensor,
    time_config: FlowTimeConfig,
    max_grad_norm: float | None,
    generator: torch.Generator,
) -> FlowMatchingMetrics:
    model.train()
    time = sample_flow_time(
        batch_size=start.shape[0],
        config=time_config,
        generator=generator,
        device=start.device,
        dtype=start.dtype,
    )
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = flow_matching_loss(model, start, end, time)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return metrics


def evaluate_flow_matching_batch(
    *,
    model: DilatedConvFlow1d,
    start: torch.Tensor,
    end: torch.Tensor,
    time_config: FlowTimeConfig,
    generator: torch.Generator,
) -> FlowMatchingMetrics:
    time = sample_flow_time(
        batch_size=start.shape[0],
        config=time_config,
        generator=generator,
        device=start.device,
        dtype=start.dtype,
    )
    _loss, metrics = flow_matching_loss(model, start, end, time)
    return metrics


def integration_diagnostics(
    *,
    model: DilatedConvFlow1d,
    start: torch.Tensor,
    steps: tuple[int, ...],
    prefix: str,
) -> dict[str, float]:
    diagnostics = {}
    for step_count in steps:
        generated = integrate_euler_midpoint_time(model, start, step_count)
        diagnostics[f"{prefix}_rms_steps_{step_count}"] = float(
            generated.square().mean().sqrt().detach().cpu()
        )
        diagnostics[f"{prefix}_mean_steps_{step_count}"] = float(
            generated.mean().detach().cpu()
        )
    return diagnostics


def integrate_flow_images(
    *,
    model: DilatedConvFlow1d,
    start: torch.Tensor,
    steps: tuple[int, ...],
    side: int,
) -> dict[int, torch.Tensor]:
    return {
        step_count: hilbert_unflatten_images(
            integrate_euler_midpoint_time(model, start, step_count), side=side
        )
        for step_count in steps
    }


def should_validate_flow(
    *, step: int, validation: FlowValidationConfig, total_steps: int
) -> bool:
    return validation.enabled and (
        step % validation.every == 0
        or (validation.validate_at_end and step == total_steps)
    )


def flow_metrics_dict(
    metrics: FlowMatchingMetrics,
    *,
    source_prefix: str,
    target_prefix: str,
) -> dict[str, float]:
    return {
        "loss": metrics.loss,
        "velocity_mse": metrics.velocity_mse,
        "velocity_cosine": metrics.velocity_cosine,
        "velocity_rel_l2_percent": metrics.velocity_rel_l2_percent,
        f"{source_prefix}_rms": metrics.image_rms,
        f"{target_prefix}_rms": metrics.target_rms,
        f"{source_prefix}_mean": metrics.image_mean,
        f"{target_prefix}_mean": metrics.target_mean,
    }


def cycle_image_batches(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yielded = False
        for images, _names in loader:
            yielded = True
            yield images
        if not yielded:
            raise ValueError("image dataset must contain at least one full batch")
