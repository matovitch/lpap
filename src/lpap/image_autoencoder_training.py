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
from lpap.hilbert import hilbert_unflatten_images
from lpap.image_autoencoder_loss import signed_mass_balance_loss
from lpap.image_energy_flow_training import (
    ENERGY_TO_IMAGE_T0,
    ENERGY_TO_IMAGE_T1,
    IMAGE_TO_ENERGY_T0,
    IMAGE_TO_ENERGY_T1,
)
from lpap.permutation import make_grouped_permutation_indices
from lpap.surrogate import (
    LPAPSurrogateTargets,
    LPAPSurrogateTransformer,
    lpap_surrogate_loss,
    prepare_lpap_surrogate_batch,
)
from lpap.teacher_checkpoints import (
    load_decoder_source,
    load_surrogate_source,
    resolve_checkpoint_path,
    validate_lpap_pair_matches_sequence_length,
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
class ImageAutoencoderLpapPairConfig:
    """One surrogate/decoder teacher pair (own C / permutation / k_max)."""

    surrogate_checkpoint_name: str = "surrogate_synthetic.pt"
    decoder_checkpoint_name: str = "decoder_synthetic.pt"
    name: str = ""

    def resolved_name(self, index: int) -> str:
        text = self.name.strip()
        return text if text else f"pair{index}"

    def as_dict(self) -> dict[str, str]:
        payload = {
            "surrogate_checkpoint_name": self.surrogate_checkpoint_name,
            "decoder_checkpoint_name": self.decoder_checkpoint_name,
        }
        if self.name.strip():
            payload["name"] = self.name.strip()
        return payload


@dataclass(frozen=True)
class ImageAutoencoderSourceConfig:
    lpap_pairs: tuple[ImageAutoencoderLpapPairConfig, ...] = (
        ImageAutoencoderLpapPairConfig(),
    )
    flow_checkpoint_name: str = "image_energy_flow.pt"
    load_best: bool = True
    require_checkpoints: bool = True
    train_image_to_energy_flow: bool = True
    train_surrogate: bool = True
    train_decoder: bool = True
    train_energy_to_image_flow: bool = True

    def validate(self) -> None:
        if not self.lpap_pairs:
            raise ValueError("source.lpap_pairs must be non-empty")

    @property
    def surrogate_checkpoint_name(self) -> str:
        return self.lpap_pairs[0].surrogate_checkpoint_name

    @property
    def decoder_checkpoint_name(self) -> str:
        return self.lpap_pairs[0].decoder_checkpoint_name

    def as_dict(self) -> dict[str, object]:
        return {
            "lpap_pairs": [pair.as_dict() for pair in self.lpap_pairs],
            "surrogate_checkpoint_name": self.surrogate_checkpoint_name,
            "decoder_checkpoint_name": self.decoder_checkpoint_name,
            "flow_checkpoint_name": self.flow_checkpoint_name,
            "load_best": self.load_best,
            "require_checkpoints": self.require_checkpoints,
            "train_image_to_energy_flow": self.train_image_to_energy_flow,
            "train_surrogate": self.train_surrogate,
            "train_decoder": self.train_decoder,
            "train_energy_to_image_flow": self.train_energy_to_image_flow,
        }


def lpap_pairs_from_source_dict(
    source_data: dict[str, Any],
) -> tuple[ImageAutoencoderLpapPairConfig, ...]:
    raw_pairs = source_data.get("lpap_pairs")
    if raw_pairs is not None:
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("source.lpap_pairs must be a non-empty list")
        return tuple(
            ImageAutoencoderLpapPairConfig(
                surrogate_checkpoint_name=str(pair["surrogate_checkpoint_name"]),
                decoder_checkpoint_name=str(pair["decoder_checkpoint_name"]),
                name=str(pair.get("name", "")),
            )
            for pair in raw_pairs
        )
    if (
        "surrogate_checkpoint_name" in source_data
        and "decoder_checkpoint_name" in source_data
    ):
        return (
            ImageAutoencoderLpapPairConfig(
                surrogate_checkpoint_name=str(source_data["surrogate_checkpoint_name"]),
                decoder_checkpoint_name=str(source_data["decoder_checkpoint_name"]),
            ),
        )
    raise ValueError(
        "source must define lpap_pairs or legacy surrogate/decoder_checkpoint_name"
    )


def image_autoencoder_source_config_from_dict(
    source_data: dict[str, Any],
) -> ImageAutoencoderSourceConfig:
    config = ImageAutoencoderSourceConfig(
        lpap_pairs=lpap_pairs_from_source_dict(source_data),
        flow_checkpoint_name=str(source_data["flow_checkpoint_name"]),
        load_best=bool(source_data["load_best"]),
        require_checkpoints=bool(source_data["require_checkpoints"]),
        train_image_to_energy_flow=bool(source_data["train_image_to_energy_flow"]),
        train_surrogate=bool(source_data["train_surrogate"]),
        train_decoder=bool(source_data["train_decoder"]),
        train_energy_to_image_flow=bool(source_data["train_energy_to_image_flow"]),
    )
    config.validate()
    return config


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
    energy_l1_weight: float = 0.5
    surrogate_teacher_weight: float = 0.05
    # Signed-mass on encoded energy e (see lpap.image_autoencoder_loss):
    #   m+/m- = mean(relu(+/- e)); scale by floor_tau (not by m++m-).
    #   L = ((m+-m-)/tau)^2 + floor_coef * sum_sides (relu(tau-m)/tau)^2
    signed_mass_balance_weight: float = 0.02
    signed_mass_floor_tau: float = 0.01
    signed_mass_floor_coef: float = 1.0
    detach_energy_target: bool = False

    def validate(self) -> None:
        if self.image_l2_weight < 0:
            raise ValueError("image_l2_weight must be non-negative")
        if self.energy_l1_weight < 0:
            raise ValueError("energy_l1_weight must be non-negative")
        if self.surrogate_teacher_weight < 0:
            raise ValueError("surrogate_teacher_weight must be non-negative")
        if self.signed_mass_balance_weight < 0:
            raise ValueError("signed_mass_balance_weight must be non-negative")
        if self.signed_mass_floor_tau <= 0:
            raise ValueError("signed_mass_floor_tau must be positive")
        if self.signed_mass_floor_coef < 0:
            raise ValueError("signed_mass_floor_coef must be non-negative")
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
            "signed_mass_balance_weight": self.signed_mass_balance_weight,
            "signed_mass_floor_tau": self.signed_mass_floor_tau,
            "signed_mass_floor_coef": self.signed_mass_floor_coef,
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
    comment: str = ""
    pinned: bool = False
    upload_artifacts_on_checkpoint: bool = False
    notify_on_finished: bool = False

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
            "comment": self.comment,
            "pinned": self.pinned,
            "upload_artifacts_on_checkpoint": self.upload_artifacts_on_checkpoint,
            "notify_on_finished": self.notify_on_finished,
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
        self.source.validate()
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
        pair_surrogate_configs: tuple[dict[str, int], ...],
        pair_decoder_configs: tuple[dict[str, object], ...],
        pair_names: tuple[str, ...],
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
                "lpap_pair_names": list(pair_names),
                "lpap_pair_surrogates": list(pair_surrogate_configs),
                "lpap_pair_decoders": list(pair_decoder_configs),
                # Legacy single-pair keys (pair 0) for older readers.
                "surrogate": pair_surrogate_configs[0],
                "decoder": pair_decoder_configs[0],
            },
        )


@dataclass(frozen=True)
class ImageAutoencoderPairForward:
    decoded_energy: torch.Tensor
    reconstructed_image: torch.Tensor
    surrogate_logits: torch.Tensor
    decoder_logits: torch.Tensor
    surrogate_targets: LPAPSurrogateTargets


@dataclass(frozen=True)
class ImageAutoencoderForward:
    image: torch.Tensor
    encoded_energy: torch.Tensor
    pairs: tuple[ImageAutoencoderPairForward, ...]

    @property
    def decoded_energy(self) -> torch.Tensor:
        return self.pairs[0].decoded_energy

    @property
    def reconstructed_image(self) -> torch.Tensor:
        return self.pairs[0].reconstructed_image

    @property
    def surrogate_logits(self) -> torch.Tensor:
        return self.pairs[0].surrogate_logits

    @property
    def decoder_logits(self) -> torch.Tensor:
        return self.pairs[0].decoder_logits

    @property
    def surrogate_targets(self) -> LPAPSurrogateTargets:
        return self.pairs[0].surrogate_targets


@dataclass(frozen=True)
class ImageAutoencoderMetrics:
    loss: float
    image_reconstruction_l2: float
    energy_reconstruction_l1: float
    surrogate_teacher_ce: float
    surrogate_weighted_accuracy: float
    signed_mass_balance: float
    signed_mass_imbalance: float
    signed_mass_gap: float
    signed_mass_floor: float
    encoded_positive_mass: float
    encoded_negative_mass: float
    encoded_energy_rms: float
    decoded_energy_rms: float
    reconstructed_image_rms: float
    image_rms: float
    pair_metrics: dict[str, float] = field(default_factory=dict)


class ImageAutoencoderModel(nn.Module):
    def __init__(
        self,
        *,
        image_to_energy_flow: DilatedConvFlow1d,
        surrogates: list[LPAPSurrogateTransformer] | nn.ModuleList,
        decoders: list[LPAPDecoderTransformer] | nn.ModuleList,
        energy_to_image_flow: DilatedConvFlow1d,
    ) -> None:
        super().__init__()
        if len(surrogates) == 0 or len(surrogates) != len(decoders):
            raise ValueError("surrogates and decoders must be non-empty and same length")
        self.image_to_energy_flow = image_to_energy_flow
        self.surrogates = nn.ModuleList(surrogates)
        self.decoders = nn.ModuleList(decoders)
        self.energy_to_image_flow = energy_to_image_flow

    @property
    def surrogate(self) -> LPAPSurrogateTransformer:
        return self.surrogates[0]  # type: ignore[return-value]

    @property
    def decoder(self) -> LPAPDecoderTransformer:
        return self.decoders[0]  # type: ignore[return-value]

    def forward_chain(
        self,
        *,
        image: torch.Tensor,
        bucket_counts: tuple[int, ...],
        k_maxs: tuple[int, ...],
        permutations: tuple[torch.Tensor, ...],
        image_to_energy_steps: int,
        energy_to_image_steps: int,
    ) -> ImageAutoencoderForward:
        if not (
            len(bucket_counts)
            == len(k_maxs)
            == len(permutations)
            == len(self.surrogates)
            == len(self.decoders)
        ):
            raise ValueError("pair runtime lengths must match ModuleList sizes")
        encoded_energy = integrate_euler_midpoint_time(
            self.image_to_energy_flow,
            image,
            image_to_energy_steps,
            t0=IMAGE_TO_ENERGY_T0,
            t1=IMAGE_TO_ENERGY_T1,
        )
        values = encoded_energy[:, 0]
        pair_outputs: list[ImageAutoencoderPairForward] = []
        for index, (surrogate, decoder) in enumerate(
            zip(self.surrogates, self.decoders, strict=True)
        ):
            bucket_count = bucket_counts[index]
            k_max = k_maxs[index]
            permutation = permutations[index]
            surrogate_tokens = prepare_lpap_surrogate_batch(
                values, bucket_count=bucket_count, permutation=permutation
            )
            surrogate_logits = surrogate(surrogate_tokens)
            decoder_batch = prepare_lpap_decoder_batch(
                values=values,
                surrogate_logits=surrogate_logits,
                bucket_count=bucket_count,
                k_max=k_max,
                temperature=decoder.frontend_temperature(),
                permutation=permutation,
            )
            decoder_logits = decoder(decoder_batch.tokens)
            decoded_energy = reconstruct_lpap_decoder_values(
                decoder_logits, decoder_batch
            ).unsqueeze(1)
            reconstructed_image = integrate_euler_midpoint_time(
                self.energy_to_image_flow,
                decoded_energy,
                energy_to_image_steps,
                t0=ENERGY_TO_IMAGE_T0,
                t1=ENERGY_TO_IMAGE_T1,
            )
            pair_outputs.append(
                ImageAutoencoderPairForward(
                    decoded_energy=decoded_energy,
                    reconstructed_image=reconstructed_image,
                    surrogate_logits=surrogate_logits,
                    decoder_logits=decoder_logits,
                    surrogate_targets=decoder_batch.surrogate_targets,
                )
            )
        return ImageAutoencoderForward(
            image=image,
            encoded_energy=encoded_energy,
            pairs=tuple(pair_outputs),
        )


@dataclass(frozen=True)
class ImageAutoencoderLpapPairRuntime:
    name: str
    surrogate_checkpoint_path: Path
    decoder_checkpoint_path: Path
    permutation: torch.Tensor
    surrogate_model_config: dict[str, int]
    decoder_model_config: dict[str, object]


@dataclass(frozen=True)
class ImageAutoencoderTrainingSession:
    config: ImageAutoencoderTrainingConfig
    device: torch.device
    checkpoint_path: Path
    log_path: Path
    image_dataset_path: Path
    image_loader: DataLoader
    validation_image_loader: DataLoader
    lpap_pairs: tuple[ImageAutoencoderLpapPairRuntime, ...]
    flow_checkpoint_path: Path
    model: ImageAutoencoderModel
    optimizer: torch.optim.Optimizer
    training_run: TrainingRun
    generator: torch.Generator
    validation_generator: torch.Generator
    resume_info: TrainingResumeInfo

    @property
    def surrogate_checkpoint_path(self) -> Path:
        return self.lpap_pairs[0].surrogate_checkpoint_path

    @property
    def decoder_checkpoint_path(self) -> Path:
        return self.lpap_pairs[0].decoder_checkpoint_path

    @property
    def permutation(self) -> torch.Tensor:
        return self.lpap_pairs[0].permutation

    @property
    def surrogate_model_config(self) -> dict[str, int]:
        return self.lpap_pairs[0].surrogate_model_config

    @property
    def decoder_model_config(self) -> dict[str, object]:
        return self.lpap_pairs[0].decoder_model_config


@dataclass(frozen=True)
class ImageAutoencoderGalleryPairItem:
    name: str
    decoded_energy: torch.Tensor
    reconstructed_image: torch.Tensor
    energy_error: torch.Tensor
    image_error: torch.Tensor


@dataclass(frozen=True)
class ImageAutoencoderGalleryItem:
    image: torch.Tensor
    encoded_energy: torch.Tensor
    pairs: tuple[ImageAutoencoderGalleryPairItem, ...]

    @property
    def reconstructed_image(self) -> torch.Tensor:
        return self.pairs[0].reconstructed_image

    @property
    def image_error(self) -> torch.Tensor:
        return self.pairs[0].image_error

    @property
    def decoded_energy(self) -> torch.Tensor:
        return self.pairs[0].decoded_energy

    @property
    def energy_error(self) -> torch.Tensor:
        return self.pairs[0].energy_error


def image_autoencoder_training_config_from_dict(
    data: dict[str, Any], *, resume_from_checkpoint: bool | None = None
) -> ImageAutoencoderTrainingConfig:
    run_data = dict(data["run"])
    if resume_from_checkpoint is not None:
        run_data["resume_from_checkpoint"] = resume_from_checkpoint
    return ImageAutoencoderTrainingConfig(
        image=image_config_from_dict(data["image"]),
        source=image_autoencoder_source_config_from_dict(data["source"]),
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
            signed_mass_balance_weight=float(
                data["loss"].get("signed_mass_balance_weight", 0.02)
            ),
            signed_mass_floor_tau=float(data["loss"].get("signed_mass_floor_tau", 0.01)),
            signed_mass_floor_coef=float(
                data["loss"].get("signed_mass_floor_coef", 1.0)
            ),
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
            comment=str(run_data.get("comment", "")),
            pinned=bool(run_data.get("pinned", False)),
            upload_artifacts_on_checkpoint=bool(
                run_data.get("upload_artifacts_on_checkpoint", False)
            ),
            notify_on_finished=bool(run_data.get("notify_on_finished", False)),
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
    flow_checkpoint_path = resolve_checkpoint_path(
        root, config.source.flow_checkpoint_name
    )

    pair_runtimes: list[ImageAutoencoderLpapPairRuntime] = []
    surrogates: list[LPAPSurrogateTransformer] = []
    decoders: list[LPAPDecoderTransformer] = []
    for index, pair_config in enumerate(config.source.lpap_pairs):
        pair_name = pair_config.resolved_name(index)
        surrogate_checkpoint_path = resolve_checkpoint_path(
            root, pair_config.surrogate_checkpoint_name
        )
        decoder_checkpoint_path = resolve_checkpoint_path(
            root, pair_config.decoder_checkpoint_name
        )
        surrogate, surrogate_model_config = load_surrogate_source(
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
        validate_lpap_pair_matches_sequence_length(
            sequence_length=config.value_count,
            surrogate_model_config=surrogate_model_config,
            decoder_model_config=decoder_model_config,
        )
        permutation = make_grouped_permutation_indices(
            value_count=config.value_count,
            bucket_count=int(decoder_model_config["bucket_count"]),
            seed=surrogate_model_config["permutation_seed"],
            device=target_device,
        )
        pair_runtimes.append(
            ImageAutoencoderLpapPairRuntime(
                name=pair_name,
                surrogate_checkpoint_path=surrogate_checkpoint_path,
                decoder_checkpoint_path=decoder_checkpoint_path,
                permutation=permutation,
                surrogate_model_config=surrogate_model_config,
                decoder_model_config=decoder_model_config,
            )
        )
        surrogates.append(surrogate)
        decoders.append(decoder)

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
    energy_to_image_flow = DilatedConvFlow1d(
        **config.energy_to_image_flow.as_dict()
    ).to(target_device)
    flow_state = load_flow_checkpoint_state(
        path=flow_checkpoint_path,
        load_best=config.source.load_best,
        require_checkpoint=config.source.require_checkpoints,
        device=target_device,
    )
    if flow_state is not None:
        image_to_energy_flow.load_state_dict(flow_state)
        energy_to_image_flow.load_state_dict(flow_state)
    model = ImageAutoencoderModel(
        image_to_energy_flow=image_to_energy_flow,
        surrogates=surrogates,
        decoders=decoders,
        energy_to_image_flow=energy_to_image_flow,
    ).to(target_device)
    _set_trainable(model.image_to_energy_flow, config.source.train_image_to_energy_flow)
    for surrogate_module in model.surrogates:
        _set_trainable(surrogate_module, config.source.train_surrogate)
    for decoder_module in model.decoders:
        _set_trainable(decoder_module, config.source.train_decoder)
    _set_trainable(model.energy_to_image_flow, config.source.train_energy_to_image_flow)
    parameters = _optimizer_parameters(model)
    if not parameters:
        raise ValueError("at least one image autoencoder component must be trainable")
    optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.learning_rate)
    pair_names = tuple(runtime.name for runtime in pair_runtimes)
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
            upload_artifacts_on_checkpoint=config.run.upload_artifacts_on_checkpoint,
            notify_on_finished=config.run.notify_on_finished,
            log_every=config.run.log_every,
            display_every=config.run.display_every,
            comment=config.run.comment,
            pinned=config.run.pinned,
        ),
        model=model,
        optimizer=optimizer,
        run_config=config.as_run_config(),
        model_config=config.model_config(
            pair_surrogate_configs=tuple(
                runtime.surrogate_model_config for runtime in pair_runtimes
            ),
            pair_decoder_configs=tuple(
                runtime.decoder_model_config for runtime in pair_runtimes
            ),
            pair_names=pair_names,
        ),
        metadata={
            "device": str(target_device),
            "image_dataset_path": str(image_dataset_path),
            "lpap_pair_names": list(pair_names),
            "surrogate_checkpoint_paths": [
                str(runtime.surrogate_checkpoint_path) for runtime in pair_runtimes
            ],
            "decoder_checkpoint_paths": [
                str(runtime.decoder_checkpoint_path) for runtime in pair_runtimes
            ],
            "surrogate_checkpoint_path": str(pair_runtimes[0].surrogate_checkpoint_path),
            "decoder_checkpoint_path": str(pair_runtimes[0].decoder_checkpoint_path),
            "flow_checkpoint_path": str(flow_checkpoint_path),
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
        lpap_pairs=tuple(pair_runtimes),
        flow_checkpoint_path=flow_checkpoint_path,
        model=model,
        optimizer=optimizer,
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
        bucket_counts=tuple(
            int(runtime.decoder_model_config["bucket_count"])
            for runtime in session.lpap_pairs
        ),
        k_maxs=tuple(
            int(runtime.surrogate_model_config["k_max"])
            for runtime in session.lpap_pairs
        ),
        permutations=tuple(runtime.permutation for runtime in session.lpap_pairs),
        image_to_energy_steps=config.integration.image_to_energy_steps,
        energy_to_image_steps=config.integration.energy_to_image_steps,
    )
    energy_target = (
        output.encoded_energy.detach()
        if config.loss.detach_energy_target
        else output.encoded_energy
    )
    image_l2_terms: list[torch.Tensor] = []
    energy_l1_terms: list[torch.Tensor] = []
    surrogate_ce_terms: list[torch.Tensor] = []
    pair_metric_values: dict[str, float] = {}
    weighted_accuracies: list[float] = []
    for runtime, pair_output in zip(session.lpap_pairs, output.pairs, strict=True):
        pair_ce, pair_surrogate_metrics = lpap_surrogate_loss(
            pair_output.surrogate_logits, pair_output.surrogate_targets
        )
        pair_image_l2 = torch_functional.mse_loss(
            pair_output.reconstructed_image, image
        )
        pair_energy_l1 = torch_functional.l1_loss(
            pair_output.decoded_energy, energy_target
        )
        image_l2_terms.append(pair_image_l2)
        energy_l1_terms.append(pair_energy_l1)
        surrogate_ce_terms.append(pair_ce)
        weighted_accuracies.append(pair_surrogate_metrics.weighted_accuracy)
        prefix = runtime.name
        pair_metric_values[f"image_reconstruction_l2/{prefix}"] = float(
            pair_image_l2.detach().cpu()
        )
        pair_metric_values[f"energy_reconstruction_l1/{prefix}"] = float(
            pair_energy_l1.detach().cpu()
        )
        pair_metric_values[f"surrogate_teacher_ce/{prefix}"] = float(
            pair_ce.detach().cpu()
        )
        pair_metric_values[f"weighted_accuracy/{prefix}"] = (
            pair_surrogate_metrics.weighted_accuracy
        )

    image_l2 = torch.stack(image_l2_terms).mean()
    energy_l1 = torch.stack(energy_l1_terms).mean()
    surrogate_teacher_ce = torch.stack(surrogate_ce_terms).mean()
    signed_mass, signed_mass_imbalance, signed_gap, signed_floor = (
        signed_mass_balance_loss(
            output.encoded_energy,
            floor_tau=config.loss.signed_mass_floor_tau,
            floor_coef=config.loss.signed_mass_floor_coef,
        )
    )
    loss = (
        config.loss.image_l2_weight * image_l2
        + config.loss.energy_l1_weight * energy_l1
        + config.loss.surrogate_teacher_weight * surrogate_teacher_ce
        + config.loss.signed_mass_balance_weight * signed_mass
    )
    positive_mass = torch_functional.relu(output.encoded_energy).mean()
    negative_mass = torch_functional.relu(-output.encoded_energy).mean()
    metrics = ImageAutoencoderMetrics(
        loss=float(loss.detach().cpu()),
        image_reconstruction_l2=float(image_l2.detach().cpu()),
        energy_reconstruction_l1=float(energy_l1.detach().cpu()),
        surrogate_teacher_ce=float(surrogate_teacher_ce.detach().cpu()),
        surrogate_weighted_accuracy=float(
            sum(weighted_accuracies) / len(weighted_accuracies)
        ),
        signed_mass_balance=float(signed_mass.detach().cpu()),
        signed_mass_imbalance=float(signed_mass_imbalance.detach().cpu()),
        signed_mass_gap=float(signed_gap.detach().cpu()),
        signed_mass_floor=float(signed_floor.detach().cpu()),
        encoded_positive_mass=float(positive_mass.detach().cpu()),
        encoded_negative_mass=float(negative_mass.detach().cpu()),
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
        pair_metrics=pair_metric_values,
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
    payload = {
        "loss": metrics.loss,
        "image_reconstruction_l2": metrics.image_reconstruction_l2,
        "energy_reconstruction_l1": metrics.energy_reconstruction_l1,
        "surrogate_teacher_ce": metrics.surrogate_teacher_ce,
        "weighted_accuracy": metrics.surrogate_weighted_accuracy,
        "signed_mass_balance": metrics.signed_mass_balance,
        "signed_mass_imbalance": metrics.signed_mass_imbalance,
        "signed_mass_gap": metrics.signed_mass_gap,
        "signed_mass_floor": metrics.signed_mass_floor,
        "encoded_positive_mass": metrics.encoded_positive_mass,
        "encoded_negative_mass": metrics.encoded_negative_mass,
        "encoded_energy_rms": metrics.encoded_energy_rms,
        "decoded_energy_rms": metrics.decoded_energy_rms,
        "reconstructed_image_rms": metrics.reconstructed_image_rms,
        "image_rms": metrics.image_rms,
    }
    payload.update(metrics.pair_metrics)
    return payload


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
                "lpap_pair_names": [runtime.name for runtime in session.lpap_pairs],
                "surrogate_checkpoint_paths": [
                    str(runtime.surrogate_checkpoint_path)
                    for runtime in session.lpap_pairs
                ],
                "decoder_checkpoint_paths": [
                    str(runtime.decoder_checkpoint_path)
                    for runtime in session.lpap_pairs
                ],
                "surrogate_checkpoint_path": str(session.surrogate_checkpoint_path),
                "decoder_checkpoint_path": str(session.decoder_checkpoint_path),
                "flow_checkpoint_path": str(session.flow_checkpoint_path),
            },
        )

    session.training_run.mark_finished()


def collect_image_autoencoder_gallery(
    session: ImageAutoencoderTrainingSession,
    *,
    sample_count: int = 1,
) -> list[ImageAutoencoderGalleryItem]:
    if sample_count <= 0:
        return []
    was_training = session.model.training
    session.model.eval()
    images_iter = cycle_image_batches(session.validation_image_loader)
    images = next(images_iter)[:sample_count]
    side = session.config.image.side
    with torch.no_grad():
        image = prepare_image_sequence(
            images, side=side, device=session.device
        )
        _loss, _metrics, output = _forward_loss(session=session, image=image)
        # Flows operate in Hilbert order; unflatten image *and* energy for spatial
        # gallery panels (row-major PNG). Leaving energy in Hilbert order makes
        # panels look scrambled even when the tensors are structurally fine.
        spatial_image = hilbert_unflatten_images(image, side=side)
        spatial_encoded = hilbert_unflatten_images(output.encoded_energy, side=side)
        items: list[ImageAutoencoderGalleryItem] = []
        for index in range(images.shape[0]):
            pair_items: list[ImageAutoencoderGalleryPairItem] = []
            encoded = spatial_encoded[index, 0]
            for runtime, pair_output in zip(
                session.lpap_pairs, output.pairs, strict=True
            ):
                spatial_reconstructed = hilbert_unflatten_images(
                    pair_output.reconstructed_image[index : index + 1], side=side
                )[0, 0]
                source_spatial = spatial_image[index, 0]
                decoded = hilbert_unflatten_images(
                    pair_output.decoded_energy[index : index + 1], side=side
                )[0, 0]
                pair_items.append(
                    ImageAutoencoderGalleryPairItem(
                        name=runtime.name,
                        decoded_energy=decoded.detach().cpu(),
                        reconstructed_image=spatial_reconstructed.detach().cpu(),
                        energy_error=(decoded - encoded).detach().cpu(),
                        image_error=(spatial_reconstructed - source_spatial)
                        .detach()
                        .cpu(),
                    )
                )
            items.append(
                ImageAutoencoderGalleryItem(
                    image=spatial_image[index, 0].detach().cpu(),
                    encoded_energy=encoded.detach().cpu(),
                    pairs=tuple(pair_items),
                )
            )
    if was_training:
        session.model.train()
    return items


__all__ = [
    "ImageAutoencoderFlowConfig",
    "ImageAutoencoderForward",
    "ImageAutoencoderGalleryItem",
    "ImageAutoencoderGalleryPairItem",
    "ImageAutoencoderImageConfig",
    "ImageAutoencoderIntegrationConfig",
    "ImageAutoencoderLossConfig",
    "ImageAutoencoderLpapPairConfig",
    "ImageAutoencoderLpapPairRuntime",
    "ImageAutoencoderMetrics",
    "ImageAutoencoderModel",
    "ImageAutoencoderOptimizerConfig",
    "ImageAutoencoderPairForward",
    "ImageAutoencoderRunConfig",
    "ImageAutoencoderSourceConfig",
    "ImageAutoencoderTrainingConfig",
    "ImageAutoencoderTrainingSession",
    "ImageAutoencoderValidationConfig",
    "collect_image_autoencoder_gallery",
    "create_image_autoencoder_training_session",
    "evaluate_image_autoencoder_batch",
    "image_autoencoder_source_config_from_dict",
    "image_autoencoder_training_config_from_dict",
    "iter_image_autoencoder_training",
    "lpap_pairs_from_source_dict",
    "rerun_image_autoencoder_training_config_from_log",
    "should_validate_image_autoencoder",
    "train_image_autoencoder_step",
]
