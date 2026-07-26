"""Versioned molab training job helpers (detached bg workers).

Lives under repo ``molab/`` (synced to ``/marimo/molab/``); not part of ``lpap``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from molab.bg_worker import (
    BackgroundWorkerError,
    require_env_keys,
    spawn_detached_python,
)

AE_ENERGY_BANK_RUN_ID = "image_autoencoder_multi_harmonics"
AE_ENERGY_BANK_CHECKPOINT = "image_autoencoder_multi_harmonics.pt"
AE_ENERGY_BANK_LOG = "image_autoencoder_multi_harmonics.sqlite"
AE_ENERGY_BANK_BG_STEM = "image_autoencoder_multi_harmonics_bg"
AE_ENERGY_BANK_SCRIPT = "train_image_autoencoder_multi_harmonics_bg.py"


def ae_energy_bank_worker_source(
    *,
    target_steps: int,
    project_root: str | Path = "/marimo",
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
) -> str:
    """Return the Python source for the multi-pair AE detached worker.

    Initializes shared i2e/e2i from **harmonics-pretrained** flow checkpoints
    (not the weak energy-bank probes), with parallel c128+c256 LPAP teachers.
    """
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    root = Path(project_root)
    resolved_comment = comment or (
        f"multi-pair AE c128+c256 from harmonics flows; bg to {target_steps}; "
        "HF upload + notify"
    )
    upload = "True" if upload_artifacts_on_checkpoint else "False"
    notify = "True" if notify_on_finished else "False"
    # Keep worker self-contained: imports only public lpap APIs.
    return f'''from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lpap.image_autoencoder_training import (
    ImageAutoencoderRunConfig,
    ImageAutoencoderSourceConfig,
    ImageAutoencoderLpapPairConfig,
)
from lpap.training_notebook import (
    create_training_session,
    default_image_autoencoder_training_config,
    iter_training,
)

project_root = Path({str(root)!r})
TARGET = {int(target_steps)}
base = default_image_autoencoder_training_config()
config = replace(
    base,
    source=ImageAutoencoderSourceConfig(
        lpap_pairs=(
            ImageAutoencoderLpapPairConfig(
                surrogate_checkpoint_name="surrogate_synthetic.pt",
                decoder_checkpoint_name="decoder_synthetic.pt",
                name="c128",
            ),
            ImageAutoencoderLpapPairConfig(
                surrogate_checkpoint_name="surrogate_c256.pt",
                decoder_checkpoint_name="decoder_c256.pt",
                name="c256",
            ),
        ),
        image_to_energy_checkpoint_name="image_to_energy.pt",
        energy_to_image_checkpoint_name="energy_to_image.pt",
        load_best=True,
        require_checkpoints=True,
        train_image_to_energy_flow=True,
        train_surrogate=True,
        train_decoder=True,
        train_energy_to_image_flow=True,
    ),
    validation=replace(base.validation, every=50, batch_size=32),
    run=ImageAutoencoderRunConfig(
        run_training=True,
        resume_from_checkpoint=True,
        steps=TARGET,
        seed=base.run.seed,
        display_every=25,
        log_every=5,
        run_id="{AE_ENERGY_BANK_RUN_ID}",
        checkpoint_name="{AE_ENERGY_BANK_CHECKPOINT}",
        log_name="{AE_ENERGY_BANK_LOG}",
        comment={resolved_comment!r},
        pinned=True,
        upload_artifacts_on_checkpoint={upload},
        notify_on_finished={notify},
    ),
)

session = create_training_session(
    "image_autoencoder", project_root=project_root, config=config
)
print(
    f"device={{session.device}} start={{session.resume_info.start_step}} "
    f"target={{config.run.steps}} upload={{config.run.upload_artifacts_on_checkpoint}} "
    f"notify={{config.run.notify_on_finished}}",
    flush=True,
)
print(session.resume_info.message, flush=True)
print(f"best_so_far={{session.training_run.best_metric}}", flush=True)
print(
    "flows=image_to_energy.pt,energy_to_image.pt pairs=c128,c256",
    flush=True,
)
for result in iter_training("image_autoencoder", session):
    if result.should_display or result.improved or result.step % 50 == 0:
        loss = result.metrics.get("loss", float("nan"))
        vloss = result.metrics.get("validation_loss")
        img = result.metrics.get("image_reconstruction_l2", float("nan"))
        eng = result.metrics.get("energy_reconstruction_l1", float("nan"))
        img128 = result.metrics.get("image_reconstruction_l2/c128", float("nan"))
        img256 = result.metrics.get("image_reconstruction_l2/c256", float("nan"))
        vtxt = "n/a" if vloss is None else f"{{vloss:.5f}}"
        best = "n/a" if result.best_metric is None else f"{{result.best_metric:.5f}}"
        mark = " *" if result.improved else ""
        ck = " ckpt" if result.checkpointed else ""
        print(
            f"step={{result.step}} loss={{loss:.5f}} val={{vtxt}} "
            f"img_l2={{img:.5f}} energy_l1={{eng:.5f}} "
            f"img_c128={{img128:.5f}} img_c256={{img256:.5f}} "
            f"best={{best}}{{mark}}{{ck}}",
            flush=True,
        )
print("AE_MULTI_HARMONICS_DONE", flush=True)
'''


def launch_ae_energy_bank_bg(
    project_root: str | Path = "/marimo",
    *,
    target_steps: int,
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    require_secrets: bool = True,
) -> dict[str, Any]:
    """Write the versioned AE worker script and spawn it detached."""
    root = Path(project_root)
    logs = root / "training_logs"
    logs.mkdir(parents=True, exist_ok=True)
    script_path = logs / AE_ENERGY_BANK_SCRIPT
    pid_path = logs / f"{AE_ENERGY_BANK_BG_STEM}.pid"
    log_path = logs / f"{AE_ENERGY_BANK_BG_STEM}.log"
    script_path.write_text(
        ae_energy_bank_worker_source(
            target_steps=target_steps,
            project_root=root,
            upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
            notify_on_finished=notify_on_finished,
            comment=comment,
        ),
        encoding="utf-8",
    )
    env = None
    if require_secrets:
        needed: list[str] = []
        if upload_artifacts_on_checkpoint:
            needed.append("HF_TOKEN")
        if notify_on_finished:
            needed.extend(["PUSHOVER_TOKEN", "PUSHOVER_USER"])
        env = require_env_keys(needed) if needed else dict(os.environ)
    spawned = spawn_detached_python(
        script_path,
        cwd=root,
        pid_path=pid_path,
        log_path=log_path,
        env=env,
    )
    return {
        **spawned,
        "target_steps": int(target_steps),
        "run_id": AE_ENERGY_BANK_RUN_ID,
        "checkpoint_name": AE_ENERGY_BANK_CHECKPOINT,
        "log_name": AE_ENERGY_BANK_LOG,
        "bg_stem": AE_ENERGY_BANK_BG_STEM,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser(
        "launch-ae-energy-bank",
        help="write + spawn the AE energy-bank bg worker",
    )
    launch.add_argument("--project-root", type=Path, default=Path("/marimo"))
    launch.add_argument("--target-steps", type=int, required=True)
    launch.add_argument(
        "--no-upload",
        action="store_true",
        help="disable HF upload on checkpoint",
    )
    launch.add_argument(
        "--no-notify",
        action="store_true",
        help="disable notify_on_finished (env flag may still apply)",
    )
    launch.add_argument("--comment", default=None)
    launch.add_argument(
        "--allow-missing-secrets",
        action="store_true",
        help="do not require HF/Pushover env (local dry runs)",
    )
    launch.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "launch-ae-energy-bank":
        try:
            result = launch_ae_energy_bank_bg(
                args.project_root,
                target_steps=args.target_steps,
                upload_artifacts_on_checkpoint=not args.no_upload,
                notify_on_finished=not args.no_notify,
                comment=args.comment,
                require_secrets=not args.allow_missing_secrets,
            )
        except BackgroundWorkerError as exc:
            print(f"error: {exc}", flush=True)
            return 2
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"spawned pid={result['pid']} target={result['target_steps']} "
                f"log={result['log_path']}"
            )
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
