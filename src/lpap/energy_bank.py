"""Empirical energy banks for flow priors (alternative to synthetic harmonics).

Banks are float tensors of shape ``(n, energy_dim)``, typically produced by
running a trained image-to-energy model over an image dataset. Training loops
iterate the bank in shuffled epochs (``cycle_energy_bank_batches``), still
independently of image batches so flows learn the energy *marginal*, not a
paired ``(image, energy)`` joint. ``sample_energy_bank_values`` remains for
one-off draws (galleries, probes).

Encode with ``lpap.energy_bank_encode.encode_image_dataset_to_energy_bank`` (or
``ImageTensorDataset.float_batch``) — never feed raw ``dataset.images`` (uint8)
into i2e when training used ``normalize=True``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from jaxtyping import Float

from lpap.data import SyntheticHarmonicConfig


EnergyPriorKind = Literal["harmonics", "energy_bank"]


@dataclass(frozen=True)
class EnergyBankConfig:
    """Filesystem pointer to an energy bank ``.pt`` payload."""

    path: str = "data/encoded_energies_ae_best.pt"
    energies_key: str = "energies"

    def validate(self) -> None:
        if not self.path:
            raise ValueError("energy bank path must be non-empty")
        if not self.energies_key:
            raise ValueError("energies_key must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "energies_key": self.energies_key}


@dataclass(frozen=True)
class EnergyBankScaleStats:
    """Summary stats used to catch unnormalized / image-like encode mistakes."""

    n: int
    energy_dim: int
    mean: float
    std: float
    min: float
    max: float
    mass_pos: float
    mass_neg: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "energy_dim": self.energy_dim,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "mass_pos": self.mass_pos,
            "mass_neg": self.mass_neg,
        }


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
        raise TypeError(f"energy bank values must be a Tensor, got {type(energies)}")
    if energies.ndim != 2:
        raise ValueError(
            f"energy bank must have shape (n, energy_dim), got {tuple(energies.shape)}"
        )
    if energies.shape[0] == 0:
        raise ValueError("energy bank must contain at least one row")
    if energies.shape[1] == 0:
        raise ValueError("energy bank energy_dim must be positive")
    if energies.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise TypeError(f"energy bank dtype must be floating, got {energies.dtype}")
    return energies


def load_energy_bank_for_flow(
    root: Path,
    config: EnergyBankConfig,
    *,
    sequence_length: int,
) -> Float[torch.Tensor, "n energy"]:
    """Load a bank and check it matches the flow sequence length.

    Lazily pulls ``config.path`` from the artifacts HF bucket when missing.
    """
    from lpap.artifact_sync import ensure_project_artifact

    path = ensure_project_artifact(
        resolve_energy_bank_path(root, config),
        project_root=root,
    )
    energies = load_energy_bank(path, energies_key=config.energies_key)
    if int(energies.shape[-1]) != sequence_length:
        raise ValueError(
            "energy bank energy_dim "
            f"{int(energies.shape[-1])} does not match flow sequence_length "
            f"{sequence_length}"
        )
    return energies


def energy_bank_scale_stats(
    energies: Float[torch.Tensor, "n energy"],
) -> EnergyBankScaleStats:
    """Compute mean/std/range/signed-mass stats for encode sanity checks."""
    if energies.ndim != 2:
        raise ValueError(
            f"energies must have shape (n, energy_dim), got {tuple(energies.shape)}"
        )
    values = energies.detach().float()
    return EnergyBankScaleStats(
        n=int(values.shape[0]),
        energy_dim=int(values.shape[1]),
        mean=float(values.mean()),
        std=float(values.std()),
        min=float(values.min()),
        max=float(values.max()),
        mass_pos=float(values.clamp_min(0).mean()),
        mass_neg=float((-values).clamp_min(0).mean()),
    )


def assert_energy_bank_scale(
    energies: Float[torch.Tensor, "n energy"],
    *,
    max_abs_mean: float = 0.05,
    max_std: float = 0.5,
    max_abs: float = 5.0,
    reference: Float[torch.Tensor, "n_ref energy"] | None = None,
    max_abs_mean_diff: float = 0.05,
    max_std_ratio: float = 3.0,
) -> EnergyBankScaleStats:
    """Raise if energies look like unnormalized images or otherwise off-scale.

    Defaults match AE i2e banks trained with ``normalize=True`` (mean≈0,
    std≈0.05–0.15). Optionally compare against a known-good reference bank.
    """
    stats = energy_bank_scale_stats(energies)
    if abs(stats.mean) > max_abs_mean:
        raise ValueError(
            "energy bank mean out of range for normalized AE encode: "
            f"mean={stats.mean:.5f} (max_abs_mean={max_abs_mean}). "
            "Did encode skip /255 (dataset.images is uint8 storage)?"
        )
    if stats.std > max_std:
        raise ValueError(
            "energy bank std out of range for normalized AE encode: "
            f"std={stats.std:.5f} (max_std={max_std}). "
            "Did encode skip /255?"
        )
    if max(abs(stats.min), abs(stats.max)) > max_abs:
        raise ValueError(
            "energy bank range out of bounds for normalized AE encode: "
            f"min={stats.min:.5f} max={stats.max:.5f} (max_abs={max_abs}). "
            "Did encode skip /255?"
        )
    if reference is not None:
        ref = energy_bank_scale_stats(reference)
        if abs(stats.mean - ref.mean) > max_abs_mean_diff:
            raise ValueError(
                "energy bank mean differs from reference: "
                f"mean={stats.mean:.5f} ref_mean={ref.mean:.5f} "
                f"(max_abs_mean_diff={max_abs_mean_diff})"
            )
        ref_std = max(ref.std, 1.0e-8)
        ratio = stats.std / ref_std
        if ratio > max_std_ratio or ratio < 1.0 / max_std_ratio:
            raise ValueError(
                "energy bank std differs from reference: "
                f"std={stats.std:.5f} ref_std={ref.std:.5f} ratio={ratio:.3f} "
                f"(max_std_ratio={max_std_ratio})"
            )
    return stats


def mean_row_correlation(
    left: Float[torch.Tensor, "n dim"],
    right: Float[torch.Tensor, "n dim"],
) -> float:
    """Mean per-row Pearson correlation between two ``(n, dim)`` tensors."""
    if left.shape != right.shape:
        raise ValueError(
            f"shape mismatch for mean_row_correlation: {tuple(left.shape)} vs "
            f"{tuple(right.shape)}"
        )
    if left.shape[0] == 0:
        raise ValueError("need at least one row")
    a = left.detach().float()
    b = right.detach().float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = a.norm(dim=1) * b.norm(dim=1)
    safe = denom > 1.0e-8
    corr = torch.zeros(a.shape[0], dtype=torch.float32)
    corr[safe] = (a[safe] * b[safe]).sum(dim=1) / denom[safe]
    return float(corr.mean())


def relative_rmse(
    left: Float[torch.Tensor, "..."],
    right: Float[torch.Tensor, "..."],
) -> float:
    """RMSE(left, right) / RMS(right); scale-sensitive (unlike Pearson)."""
    if left.shape != right.shape:
        raise ValueError(
            f"shape mismatch for relative_rmse: {tuple(left.shape)} vs "
            f"{tuple(right.shape)}"
        )
    a = left.detach().float()
    b = right.detach().float()
    rmse = float(torch.sqrt(torch.mean((a - b) ** 2)))
    denom = float(torch.sqrt(torch.mean(b**2)).clamp_min(1.0e-8))
    return rmse / denom


def assert_energies_not_raw_image_like(
    energies: Float[torch.Tensor, "n energy"],
    hilbert_raw_images: Float[torch.Tensor, "n energy"],
    *,
    min_relative_rmse: float = 0.5,
) -> float:
    """Raise if energies are absolutely close to raw ``0..255`` Hilbert images.

    Uses relative RMSE (scale-sensitive). Pearson correlation alone is *not*
    enough: ``/255`` images still correlate ~1.0 with uint8 Hilbert sequences.
    """
    rel = relative_rmse(energies, hilbert_raw_images)
    if rel < min_relative_rmse:
        raise ValueError(
            "encoded energies are too close to raw Hilbert images "
            f"(relative_rmse={rel:.3f} < {min_relative_rmse}). "
            "Likely skipped /255 when reading dataset.images."
        )
    return rel


def sample_energy_bank_values(
    energies: Float[torch.Tensor, "n energy"],
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Float[torch.Tensor, "batch energy"]:
    """Sample ``batch_size`` rows uniformly (with replacement) from the bank.

    Prefer ``cycle_energy_bank_batches`` for training loops (shuffled epochs).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = int(energies.shape[0])
    # Draw on the generator's device (CUDA generators reject CPU-only randint),
    # then index the (usually CPU/mmap) bank with CPU indices.
    indices = torch.randint(
        n, (batch_size,), device=generator.device, generator=generator
    )
    if indices.device.type != "cpu":
        indices = indices.cpu()
    return energies[indices].to(device=device, dtype=torch.float32)


def cycle_energy_bank_batches(
    energies: Float[torch.Tensor, "n energy"],
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    drop_last: bool = True,
) -> Iterator[Float[torch.Tensor, "batch energy"]]:
    """Yield shuffled bank batches forever (epoch-style, optional drop_last).

    Each epoch reshuffles with ``generator``. Bank rows stay unpaired from any
    image loader — only the iteration style matches a DataLoader epoch.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = int(energies.shape[0])
    if n < batch_size:
        raise ValueError(
            f"energy bank has {n} rows but batch_size={batch_size} "
            "(need at least one full batch)"
        )
    while True:
        order = torch.randperm(n, device=generator.device, generator=generator)
        if order.device.type != "cpu":
            order = order.cpu()
        if drop_last:
            usable = (n // batch_size) * batch_size
            order = order[:usable]
        elif n % batch_size != 0:
            raise ValueError(
                "drop_last=False is unsupported for energy-bank cycling "
                "(teachers/flows expect fixed batch sizes)"
            )
        for start in range(0, int(order.numel()), batch_size):
            indices = order[start : start + batch_size]
            yield energies[indices].to(device=device, dtype=torch.float32)


def sample_energy_prior_values(
    *,
    kind: EnergyPriorKind,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    sequence_length: int,
    harmonics: SyntheticHarmonicConfig | None = None,
    energy_bank: Float[torch.Tensor, "n energy"] | None = None,
) -> Float[torch.Tensor, "batch energy"]:
    """Sample a flow energy prior batch ``(batch, sequence_length)``.

    Image batches must be drawn independently so training targets the energy
    marginal rather than a paired ``(image, energy)`` map.
    """
    if kind == "energy_bank":
        if energy_bank is None:
            raise ValueError("energy_bank tensor is required when kind=energy_bank")
        return sample_energy_bank_values(
            energy_bank,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
    if kind == "harmonics":
        if harmonics is None:
            raise ValueError("harmonics config is required when kind=harmonics")
        values = harmonics.sample_batch(
            batch_size=batch_size,
            n=sequence_length,
            generator=generator,
            device=device,
        )
        if not isinstance(values, torch.Tensor):
            raise TypeError("expected harmonic values tensor")
        return values
    raise ValueError(f"unsupported energy prior kind: {kind!r}")


__all__ = [
    "EnergyBankConfig",
    "EnergyBankScaleStats",
    "EnergyPriorKind",
    "assert_energies_not_raw_image_like",
    "assert_energy_bank_scale",
    "energy_bank_config_from_dict",
    "energy_bank_scale_stats",
    "cycle_energy_bank_batches",
    "load_energy_bank",
    "load_energy_bank_for_flow",
    "mean_row_correlation",
    "relative_rmse",
    "resolve_energy_bank_path",
    "sample_energy_bank_values",
    "sample_energy_prior_values",
]
