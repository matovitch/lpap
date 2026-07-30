"""Encode an image dataset through an image-to-energy flow into a bank.

Always uses ``ImageTensorDataset.float_batch`` (``/255`` when ``normalize``) —
never raw ``dataset.images`` — and asserts scale before/after the full pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import nn

from lpap.data import ImageTensorDataset, float_image_batch
from lpap.energy_bank import (
    EnergyBankScaleStats,
    assert_energies_not_raw_image_like,
    assert_energy_bank_scale,
)
from lpap.flow import integrate_euler_midpoint_time
from lpap.flow_training import prepare_image_sequence
from lpap.image_energy_flow_training import IMAGE_TO_ENERGY_T0, IMAGE_TO_ENERGY_T1


@dataclass(frozen=True)
class EnergyBankEncodeResult:
    energies: Float[torch.Tensor, "n energy"]
    metadata: dict[str, Any]
    probe_stats: EnergyBankScaleStats
    final_stats: EnergyBankScaleStats
    probe_raw_image_rel_rmse: float


def encode_image_dataset_to_energy_bank(
    dataset: ImageTensorDataset,
    *,
    flow: nn.Module,
    side: int,
    image_to_energy_steps: int,
    device: torch.device,
    batch_size: int = 64,
    probe_batches: int = 2,
    progress_every: int = 50,
    progress: Callable[[str], None] | None = None,
    max_abs_mean: float = 0.05,
    max_std: float = 0.5,
    max_abs: float = 5.0,
    min_relative_rmse_vs_raw_hilbert: float = 0.5,
    reference_energies: Float[torch.Tensor, "n_ref energy"] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> EnergyBankEncodeResult:
    """Run i2e over ``dataset`` and return energies plus sanity metadata.

    Requires ``dataset.normalize=True`` so encode matches AE training I/O.
    Probes the first batches (scale + absolute distance vs raw Hilbert images),
    then encodes the full set and re-asserts scale.
    """
    if not dataset.normalize:
        raise ValueError(
            "encode_image_dataset_to_energy_bank requires dataset.normalize=True "
            "(uint8 storage must be /255 to match AE training)"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if probe_batches <= 0:
        raise ValueError("probe_batches must be positive")
    if image_to_energy_steps <= 0:
        raise ValueError("image_to_energy_steps must be positive")
    if side <= 0:
        raise ValueError("side must be positive")

    n = len(dataset)
    if dataset.images.shape[2:] != (side, side):
        raise ValueError(
            f"dataset side {tuple(dataset.images.shape[2:])} does not match "
            f"side={side}"
        )

    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    flow = flow.to(device)
    flow.eval()

    def encode_range(start: int, stop: int) -> Float[torch.Tensor, "batch energy"]:
        batch = dataset.float_batch(start, stop)
        seq = prepare_image_sequence(batch, side=side, device=device)
        with torch.no_grad():
            encoded = integrate_euler_midpoint_time(
                flow,
                seq,
                image_to_energy_steps,
                t0=IMAGE_TO_ENERGY_T0,
                t1=IMAGE_TO_ENERGY_T1,
            )
        return encoded[:, 0].detach().cpu()

    probe_stop = min(n, batch_size * probe_batches)
    log(f"probe encode 0:{probe_stop} / {n}")
    probe = encode_range(0, probe_stop)
    probe_stats = assert_energy_bank_scale(
        probe,
        max_abs_mean=max_abs_mean,
        max_std=max_std,
        max_abs=max_abs,
        reference=reference_energies,
    )
    raw_hilbert = prepare_image_sequence(
        float_image_batch(dataset.images[:probe_stop], normalize=False),
        side=side,
        device=torch.device("cpu"),
    )[:, 0]
    probe_raw_rmse = assert_energies_not_raw_image_like(
        probe,
        raw_hilbert,
        min_relative_rmse=min_relative_rmse_vs_raw_hilbert,
    )
    log(
        f"probe ok mean={probe_stats.mean:.5f} std={probe_stats.std:.5f} "
        f"raw_hilbert_rel_rmse={probe_raw_rmse:.3f}"
    )

    chunks: list[torch.Tensor] = []
    batch_index = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            chunks.append(encode_range(start, stop))
            if batch_index % progress_every == 0:
                log(f"encoded {stop}/{n}")
            batch_index += 1

    energies = torch.cat(chunks, dim=0).contiguous()
    final_stats = assert_energy_bank_scale(
        energies,
        max_abs_mean=max_abs_mean,
        max_std=max_std,
        max_abs=max_abs,
        reference=reference_energies,
    )
    log(
        f"full ok mean={final_stats.mean:.5f} std={final_stats.std:.5f} "
        f"min={final_stats.min:.5f} max={final_stats.max:.5f}"
    )

    metadata: dict[str, Any] = {
        "n_images": n,
        "energy_dim": int(energies.shape[-1]),
        "normalize": True,
        "normalize_applied": "ImageTensorDataset.float_batch",
        "image_to_energy_steps": int(image_to_energy_steps),
        "side": int(side),
        "batch_size": int(batch_size),
        "probe_batches": int(probe_batches),
        "probe_raw_hilbert_rel_rmse": probe_raw_rmse,
        "probe_stats": probe_stats.as_dict(),
        "final_stats": final_stats.as_dict(),
        "note": "empirical i2e energy marginal; encode via float_batch (/255)",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return EnergyBankEncodeResult(
        energies=energies,
        metadata=metadata,
        probe_stats=probe_stats,
        final_stats=final_stats,
        probe_raw_image_rel_rmse=probe_raw_rmse,
    )


__all__ = [
    "EnergyBankEncodeResult",
    "encode_image_dataset_to_energy_bank",
]
