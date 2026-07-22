"""Image-autoencoder loss terms and lambda-dial helpers.

Signed-mass design (plaintext)::

    m+ = mean(relu(e))
    m- = mean(relu(-e))
    tau = signed_mass_floor_tau   # target per-side mass scale

    # Gap is scaled by tau (NOT by m++m-), so e→0 is not a free "balanced" win.
    L_gap = ((m+ - m-) / (tau + eps))^2

    # Soft floor: each side should reach ~tau.
    L_floor = (relu(tau - m+) / (tau + eps))^2 + (relu(tau - m-) / (tau + eps))^2

    L_signed = L_gap + floor_coef * L_floor

Dial protocol
-------------
1. Build an AE session (teachers or a checkpoint).
2. Call ``probe_image_autoencoder_loss`` on a validation batch.
3. Inspect ``weighted`` contributions; image L2 should dominate, other terms
   should sit in a smaller but visible band (roughly 0.1x–0.5x of image).
4. Call ``suggest_image_autoencoder_loss_weights`` for a first guess, then
   short-train and re-probe before a long run.
5. Ask a human to validate gallery (recon + bipolar encoded energy) before
   committing the long e2e run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.nn import functional as torch_functional

from lpap.flow_training import cycle_image_batches, prepare_image_sequence


@dataclass(frozen=True)
class SignedMassStats:
    positive_mass: float
    negative_mass: float
    total_mass: float
    imbalance_ratio: float
    gap_loss: float
    floor_loss: float
    signed_loss: float


@dataclass(frozen=True)
class ImageAutoencoderLossProbe:
    """Unweighted metrics, weighted contributions, and signed-mass stats."""

    batch_size: int
    unweighted: dict[str, float]
    weighted: dict[str, float]
    weights: dict[str, float]
    signed_mass: SignedMassStats
    totals: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "unweighted": dict(self.unweighted),
            "weighted": dict(self.weighted),
            "weights": dict(self.weights),
            "signed_mass": {
                "positive_mass": self.signed_mass.positive_mass,
                "negative_mass": self.signed_mass.negative_mass,
                "total_mass": self.signed_mass.total_mass,
                "imbalance_ratio": self.signed_mass.imbalance_ratio,
                "gap_loss": self.signed_mass.gap_loss,
                "floor_loss": self.signed_mass.floor_loss,
                "signed_loss": self.signed_mass.signed_loss,
            },
            "totals": dict(self.totals),
        }


def signed_mass_balance_loss(
    energy: torch.Tensor,
    *,
    floor_tau: float = 0.01,
    floor_coef: float = 1.0,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(signed_loss, imbalance_ratio, gap_loss, floor_loss)``."""
    if floor_tau <= 0:
        raise ValueError("floor_tau must be positive")
    if floor_coef < 0:
        raise ValueError("floor_coef must be non-negative")
    positive_mass = torch_functional.relu(energy).mean()
    negative_mass = torch_functional.relu(-energy).mean()
    scale = floor_tau + eps
    gap_loss = ((positive_mass - negative_mass) / scale).square()
    floor_loss = (
        torch_functional.relu(floor_tau - positive_mass) / scale
    ).square() + (torch_functional.relu(floor_tau - negative_mass) / scale).square()
    signed_loss = gap_loss + floor_coef * floor_loss
    total_mass = positive_mass + negative_mass
    imbalance_ratio = (positive_mass - negative_mass) / (total_mass + eps)
    return signed_loss, imbalance_ratio, gap_loss, floor_loss


def format_loss_probe_report(probe: ImageAutoencoderLossProbe) -> str:
    lines = [
        "image-autoencoder loss probe",
        f"  batch_size: {probe.batch_size}",
        "  weights:",
    ]
    for key, value in probe.weights.items():
        lines.append(f"    {key}: {value:g}")
    lines.append("  unweighted → weighted:")
    for key in (
        "image_reconstruction_l2",
        "energy_reconstruction_l1",
        "surrogate_teacher_ce",
        "signed_mass_balance",
    ):
        lines.append(
            f"    {key}: {probe.unweighted[key]:.6f} -> {probe.weighted[key]:.6f}"
        )
    lines.append(
        "  signed-mass detail: "
        f"gap={probe.unweighted['signed_mass_gap']:.5f} "
        f"floor={probe.unweighted['signed_mass_floor']:.5f}"
    )
    lines.append(
        "  signed-mass: "
        f"m+={probe.signed_mass.positive_mass:.5f} "
        f"m-={probe.signed_mass.negative_mass:.5f} "
        f"ratio={probe.signed_mass.imbalance_ratio:.4f} "
        f"gap={probe.signed_mass.gap_loss:.5f} "
        f"floor={probe.signed_mass.floor_loss:.5f}"
    )
    lines.append(
        f"  total_weighted={probe.totals['weighted_loss']:.6f} "
        f"(image_share={probe.totals['image_share']:.3f})"
    )
    return "\n".join(lines)


def suggest_image_autoencoder_loss_weights(
    probe: ImageAutoencoderLossProbe,
    *,
    energy_l1_share: float = 0.15,
    surrogate_share: float = 0.35,
    signed_mass_share: float = 0.15,
) -> dict[str, float]:
    """Suggest weights so each term's contribution is a share of image L2.

    Shares are relative to the current unweighted ``image_reconstruction_l2``.
    """
    image = max(probe.unweighted["image_reconstruction_l2"], 1.0e-12)
    energy = max(probe.unweighted["energy_reconstruction_l1"], 1.0e-12)
    surrogate = max(probe.unweighted["surrogate_teacher_ce"], 1.0e-12)
    signed = max(probe.unweighted["signed_mass_balance"], 1.0e-12)
    return {
        "image_l2_weight": 1.0,
        "energy_l1_weight": (energy_l1_share * image) / energy,
        "surrogate_teacher_weight": (surrogate_share * image) / surrogate,
        "signed_mass_balance_weight": (signed_mass_share * image) / signed,
    }


def probe_image_autoencoder_loss(
    session: Any,
    *,
    batch_size: int | None = None,
    images: torch.Tensor | None = None,
) -> ImageAutoencoderLossProbe:
    """Evaluate loss-component magnitudes on one batch (no grad)."""
    from lpap.image_autoencoder_training import _forward_loss

    config = session.config
    if images is None:
        loader = session.validation_image_loader
        images = next(cycle_image_batches(loader))
    if batch_size is not None:
        images = images[:batch_size]
    was_training = session.model.training
    session.model.eval()
    with torch.no_grad():
        image = prepare_image_sequence(
            images, side=config.image.side, device=session.device
        )
        _loss, metrics, output = _forward_loss(session=session, image=image)
        signed_loss, imbalance, gap_loss, floor_loss = signed_mass_balance_loss(
            output.encoded_energy,
            floor_tau=config.loss.signed_mass_floor_tau,
            floor_coef=config.loss.signed_mass_floor_coef,
        )
        positive_mass = float(torch_functional.relu(output.encoded_energy).mean().cpu())
        negative_mass = float(torch_functional.relu(-output.encoded_energy).mean().cpu())
    if was_training:
        session.model.train()

    weights = {
        "image_l2_weight": config.loss.image_l2_weight,
        "energy_l1_weight": config.loss.energy_l1_weight,
        "surrogate_teacher_weight": config.loss.surrogate_teacher_weight,
        "signed_mass_balance_weight": config.loss.signed_mass_balance_weight,
        "signed_mass_floor_tau": config.loss.signed_mass_floor_tau,
        "signed_mass_floor_coef": config.loss.signed_mass_floor_coef,
    }
    unweighted = {
        "image_reconstruction_l2": metrics.image_reconstruction_l2,
        "energy_reconstruction_l1": metrics.energy_reconstruction_l1,
        "surrogate_teacher_ce": metrics.surrogate_teacher_ce,
        "signed_mass_balance": float(signed_loss.detach().cpu()),
        "signed_mass_gap": float(gap_loss.detach().cpu()),
        "signed_mass_floor": float(floor_loss.detach().cpu()),
    }
    weighted = {
        "image_reconstruction_l2": weights["image_l2_weight"]
        * unweighted["image_reconstruction_l2"],
        "energy_reconstruction_l1": weights["energy_l1_weight"]
        * unweighted["energy_reconstruction_l1"],
        "surrogate_teacher_ce": weights["surrogate_teacher_weight"]
        * unweighted["surrogate_teacher_ce"],
        "signed_mass_balance": weights["signed_mass_balance_weight"]
        * unweighted["signed_mass_balance"],
    }
    weighted_loss = float(sum(weighted.values()))
    image_share = weighted["image_reconstruction_l2"] / max(weighted_loss, 1.0e-12)
    return ImageAutoencoderLossProbe(
        batch_size=int(images.shape[0]),
        unweighted=unweighted,
        weighted=weighted,
        weights=weights,
        signed_mass=SignedMassStats(
            positive_mass=positive_mass,
            negative_mass=negative_mass,
            total_mass=positive_mass + negative_mass,
            imbalance_ratio=float(imbalance.detach().cpu()),
            gap_loss=float(gap_loss.detach().cpu()),
            floor_loss=float(floor_loss.detach().cpu()),
            signed_loss=float(signed_loss.detach().cpu()),
        ),
        totals={"weighted_loss": weighted_loss, "image_share": image_share},
    )


def apply_suggested_loss_weights(config: Any, suggested: dict[str, float]) -> Any:
    """Return a copy of an AE training config with updated loss weights."""
    return replace(
        config,
        loss=replace(
            config.loss,
            image_l2_weight=float(suggested["image_l2_weight"]),
            energy_l1_weight=float(suggested["energy_l1_weight"]),
            surrogate_teacher_weight=float(suggested["surrogate_teacher_weight"]),
            signed_mass_balance_weight=float(suggested["signed_mass_balance_weight"]),
        ),
    )


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from lpap.image_autoencoder_training import (
        create_image_autoencoder_training_session,
    )
    from lpap.training import load_training_checkpoint
    from lpap.training_notebook import default_image_autoencoder_training_config

    parser = argparse.ArgumentParser(
        description="Probe AE loss-component magnitudes for lambda dialing."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional AE checkpoint (default: config checkpoint under project root).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Print suggested weights targeting image-relative shares.",
    )
    args = parser.parse_args(argv)

    config = default_image_autoencoder_training_config()
    config = replace(config, run=replace(config.run, run_training=False))
    session = create_image_autoencoder_training_session(
        project_root=args.project_root, config=config
    )
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.project_root / "checkpoints" / config.run.checkpoint_name
    if checkpoint.exists():
        payload = load_training_checkpoint(checkpoint, map_location=session.device)
        state = payload.get("best_model_state") or payload["model_state"]
        session.model.load_state_dict(state)
        print(f"loaded checkpoint: {checkpoint} step={payload.get('step')}")
    else:
        print(f"no checkpoint at {checkpoint}; using teacher-initialized weights")

    probe = probe_image_autoencoder_loss(session, batch_size=args.batch_size)
    print(format_loss_probe_report(probe))
    if args.suggest:
        suggested = suggest_image_autoencoder_loss_weights(probe)
        print("suggested weights:")
        for key, value in suggested.items():
            print(f"  {key}: {value:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "ImageAutoencoderLossProbe",
    "SignedMassStats",
    "apply_suggested_loss_weights",
    "format_loss_probe_report",
    "probe_image_autoencoder_loss",
    "signed_mass_balance_loss",
    "suggest_image_autoencoder_loss_weights",
]
