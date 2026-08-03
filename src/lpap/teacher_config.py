"""Shared LPAP teacher-pair TOML: one file for surrogate + decoder jobs."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lpap.decoder_training import (
    LPAPDecoderModelConfig,
    LPAPDecoderRegularizationConfig,
    LPAPDecoderRunConfig,
    LPAPDecoderTeacherConfig,
    LPAPDecoderTrainingConfig,
)
from lpap.energy_bank import EnergyBankConfig, energy_bank_config_from_dict
from lpap.surrogate_training import (
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateOptimizerConfig,
    LPAPSurrogateRunConfig,
    LPAPSurrogateTrainingConfig,
    LPAPSurrogateValidationConfig,
)

TeacherPhase = Literal["surrogate", "decoder"]


@dataclass(frozen=True)
class TeacherPairConfig:
    """Compatibility-critical fields shared by surrogate and decoder jobs."""

    name: str
    bucket_count: int
    probe_count: int
    k_max: int
    permutation_seed: int
    energy_bank: EnergyBankConfig
    batch_size: int
    raw: dict[str, Any]

    @property
    def value_count(self) -> int:
        return self.bucket_count * self.probe_count

    @property
    def surrogate_run_id(self) -> str:
        return f"surrogate_{self.name}"

    @property
    def decoder_run_id(self) -> str:
        return f"decoder_{self.name}"

    @property
    def surrogate_checkpoint_name(self) -> str:
        return f"{self.surrogate_run_id}.pt"

    @property
    def decoder_checkpoint_name(self) -> str:
        return f"{self.decoder_run_id}.pt"

    @property
    def surrogate_log_name(self) -> str:
        return f"{self.surrogate_run_id}.sqlite"

    @property
    def decoder_log_name(self) -> str:
        return f"{self.decoder_run_id}.sqlite"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("pair.name must be non-empty")
        if self.bucket_count <= 0 or self.probe_count <= 0:
            raise ValueError("pair bucket_count/probe_count must be positive")
        if self.k_max <= 0:
            raise ValueError("pair.k_max must be positive")
        if self.batch_size <= 0:
            raise ValueError("data.batch_size must be positive")
        self.energy_bank.validate()


def load_teacher_pair_toml(path: str | Path) -> TeacherPairConfig:
    path = Path(path)
    with path.open("rb") as file:
        data = tomllib.load(file)
    if "pair" not in data:
        raise ValueError(f"teacher TOML missing [pair]: {path}")
    pair = data["pair"]
    bank_raw = pair.get("energy_bank")
    if not isinstance(bank_raw, dict):
        raise ValueError(f"teacher TOML missing [pair.energy_bank]: {path}")
    data_section = data.get("data") or {}
    config = TeacherPairConfig(
        name=str(pair["name"]),
        bucket_count=int(pair["bucket_count"]),
        probe_count=int(pair["probe_count"]),
        k_max=int(pair["k_max"]),
        permutation_seed=int(pair["permutation_seed"]),
        energy_bank=energy_bank_config_from_dict(bank_raw),
        batch_size=int(data_section.get("batch_size", 32)),
        raw=data,
    )
    config.validate()
    return config


def _validation_from_dict(data: dict[str, Any]) -> LPAPSurrogateValidationConfig:
    return LPAPSurrogateValidationConfig(
        enabled=bool(data.get("enabled", True)),
        every=int(data.get("every", 100)),
        batch_size=int(data.get("batch_size", 256)),
        seed=int(data.get("seed", 10_123)),
        validate_at_end=bool(data.get("validate_at_end", True)),
    )


def _shared_data(pair: TeacherPairConfig) -> LPAPSurrogateDataConfig:
    return LPAPSurrogateDataConfig(
        batch_size=pair.batch_size,
        bucket_count=pair.bucket_count,
        probe_count=pair.probe_count,
        energy_bank=pair.energy_bank,
    )


def project_teacher_config(
    path: str | Path,
    phase: TeacherPhase,
    *,
    resume_from_checkpoint: bool | None = None,
) -> LPAPSurrogateTrainingConfig | LPAPDecoderTrainingConfig:
    """Load a ``teacher_*.toml`` and project to a surrogate or decoder config."""
    if phase not in ("surrogate", "decoder"):
        raise ValueError(f"unsupported teacher phase: {phase}")
    pair = load_teacher_pair_toml(path)
    if phase == "surrogate":
        return _project_surrogate(pair, resume_from_checkpoint=resume_from_checkpoint)
    return _project_decoder(pair, resume_from_checkpoint=resume_from_checkpoint)


def _project_surrogate(
    pair: TeacherPairConfig,
    *,
    resume_from_checkpoint: bool | None,
) -> LPAPSurrogateTrainingConfig:
    section = dict(pair.raw.get("surrogate") or {})
    run_raw = dict(section.pop("run", {}) or {})
    validation_raw = dict(section.pop("validation", {}) or {})
    if resume_from_checkpoint is not None:
        run_raw["resume_from_checkpoint"] = resume_from_checkpoint
    config = LPAPSurrogateTrainingConfig(
        data=_shared_data(pair),
        model=LPAPSurrogateModelConfig(
            k_max=pair.k_max,
            hidden_dim=int(section.get("hidden_dim", 256)),
            layer_count=int(section.get("layer_count", 12)),
            head_count=int(section.get("head_count", 8)),
        ),
        optimizer=LPAPSurrogateOptimizerConfig(
            learning_rate=float(section.get("learning_rate", 1.0e-3))
        ),
        validation=_validation_from_dict(validation_raw),
        run=LPAPSurrogateRunConfig(
            run_training=bool(run_raw.get("run_training", True)),
            resume_from_checkpoint=bool(run_raw.get("resume_from_checkpoint", True)),
            steps=int(run_raw.get("steps", 15_000)),
            seed=int(run_raw.get("seed", 123)),
            permutation_seed=pair.permutation_seed,
            display_every=int(run_raw.get("display_every", 5)),
            log_every=int(run_raw.get("log_every", 1)),
            run_id=pair.surrogate_run_id,
            checkpoint_name=pair.surrogate_checkpoint_name,
            log_name=pair.surrogate_log_name,
            comment=str(run_raw.get("comment", "")),
            pinned=bool(run_raw.get("pinned", False)),
        ),
    )
    config.validate()
    return config


def _project_decoder(
    pair: TeacherPairConfig,
    *,
    resume_from_checkpoint: bool | None,
) -> LPAPDecoderTrainingConfig:
    section = dict(pair.raw.get("decoder") or {})
    run_raw = dict(section.pop("run", {}) or {})
    validation_raw = dict(section.pop("validation", {}) or {})
    if resume_from_checkpoint is not None:
        run_raw["resume_from_checkpoint"] = resume_from_checkpoint
    config = LPAPDecoderTrainingConfig(
        data=_shared_data(pair),
        decoder=LPAPDecoderModelConfig(
            frontend_initial_temperature=float(
                section.get("frontend_initial_temperature", 0.25)
            ),
            hidden_dim=int(section.get("hidden_dim", 256)),
            layer_count=int(section.get("layer_count", 8)),
            head_count=int(section.get("head_count", 8)),
        ),
        optimizer=LPAPSurrogateOptimizerConfig(
            learning_rate=float(section.get("learning_rate", 1.0e-3))
        ),
        validation=_validation_from_dict(validation_raw),
        teacher=LPAPDecoderTeacherConfig(
            checkpoint_name=pair.surrogate_checkpoint_name,
            load_best=bool(section.get("load_best", True)),
            require_checkpoint=bool(section.get("require_checkpoint", True)),
        ),
        regularization=LPAPDecoderRegularizationConfig(
            source_ce_weight=float(section.get("source_ce_weight", 0.1)),
            source_ce_l1_reference=float(section.get("source_ce_l1_reference", 0.05)),
            source_ce_power=float(section.get("source_ce_power", 2.0)),
        ),
        run=LPAPDecoderRunConfig(
            run_training=bool(run_raw.get("run_training", True)),
            resume_from_checkpoint=bool(run_raw.get("resume_from_checkpoint", True)),
            steps=int(run_raw.get("steps", 15_000)),
            seed=int(run_raw.get("seed", 456)),
            permutation_seed=pair.permutation_seed,
            display_every=int(run_raw.get("display_every", 5)),
            log_every=int(run_raw.get("log_every", 1)),
            run_id=pair.decoder_run_id,
            checkpoint_name=pair.decoder_checkpoint_name,
            log_name=pair.decoder_log_name,
            comment=str(run_raw.get("comment", "")),
            pinned=bool(run_raw.get("pinned", False)),
        ),
    )
    config.validate()
    return config


__all__ = [
    "TeacherPairConfig",
    "TeacherPhase",
    "load_teacher_pair_toml",
    "project_teacher_config",
]
