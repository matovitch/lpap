"""Empirical energy banks for flow priors (alternative to synthetic harmonics).

Banks are float tensors of shape ``(n, energy_dim)``, typically produced by
running a trained image-to-energy model over an image dataset. Flow training
samples bank rows independently of image batches so the learned map targets
the energy *marginal*, not a paired ``(image, energy)`` joint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float


@dataclass(frozen=True)
class EnergyBankConfig:
    """Filesystem pointer to an energy bank ``.pt`` payload."""

    path: str = "data/encoded_energies_ae_best_131k.pt"
    energies_key: str = "energies"

    def validate(self) -> None:
        if not self.path:
            raise ValueError("energy bank path must be non-empty")
        if not self.energies_key:
            raise ValueError("energies_key must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "energies_key": self.energies_key}


def energy_bank_config_from_dict(data: dict[str, Any]) -> EnergyBankConfig:
    return EnergyBankConfig(
        path=str(data["path"]),
        energies_key=str(data.get("energies_key", "energies")),
    )


def resolve_energy_bank_path(root: Path, config: EnergyBankConfig) -> Path:
    path = Path(config.path)
    return path if path.is_absolute() else root / path


def load_energy_bank(
    path: str | Path,
    *,
    energies_key: str = "energies",
    map_location: str | torch.device | None = "cpu",
    mmap: bool = True,
    weights_only: bool = False,
) -> Float[torch.Tensor, "n energy"]:
    """Load an energy bank tensor from disk.

    Accepts either a raw 2-D float tensor or a dict payload with ``energies_key``
    (and optional ``metadata``). Uses ``mmap`` when possible for large banks.
    """
    payload = torch.load(
        Path(path),
        map_location=map_location,
        mmap=mmap,
        weights_only=weights_only,
    )
    if isinstance(payload, torch.Tensor):
        energies = payload
    elif isinstance(payload, dict):
        if energies_key not in payload:
            raise KeyError(
                f"energy bank at {path} missing key {energies_key!r}; "
                f"keys={sorted(payload)}"
            )
        energies = payload[energies_key]
    else:
        raise TypeError(
            f"energy bank at {path} must be a Tensor or dict, got {type(payload)}"
        )
    if not isinstance(energies, torch.Tensor):
        raise TypeError(
            f"energy bank values must be a Tensor, got {type(energies)}"
        )
    if energies.ndim != 2:
        raise ValueError(
            f"energy bank must have shape (n, energy_dim), got {tuple(energies.shape)}"
        )
    if energies.shape[0] == 0:
        raise ValueError("energy bank must contain at least one row")
    if energies.shape[1] == 0:
        raise ValueError("energy bank energy_dim must be positive")
    if energies.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            f"energy bank dtype must be floating, got {energies.dtype}"
        )
    return energies


def sample_energy_bank_values(
    energies: Float[torch.Tensor, "n energy"],
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Float[torch.Tensor, "batch energy"]:
    """Sample ``batch_size`` rows uniformly (with replacement) from the bank."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = int(energies.shape[0])
    indices = torch.randint(n, (batch_size,), generator=generator)
    sampled = energies[indices].to(device=device, dtype=torch.float32)
    return sampled


__all__ = [
    "EnergyBankConfig",
    "energy_bank_config_from_dict",
    "load_energy_bank",
    "resolve_energy_bank_path",
    "sample_energy_bank_values",
]
