"""Versioned molab training job helpers (detached bg workers).

Lives under repo ``molab/`` (synced to ``/marimo/molab/``); not part of ``lpap``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from molab.bg_worker import (
    BackgroundWorkerError,
    require_env_keys,
    spawn_detached_python,
)

# Fresh multi-pair AE initialized from bank-pretrained flows (c128+c256 teachers).
AE_ENERGY_BANK_RUN_ID = "image_autoencoder_multi_energy_bank"
AE_ENERGY_BANK_CHECKPOINT = "image_autoencoder_multi_energy_bank.pt"
AE_ENERGY_BANK_LOG = "image_autoencoder_multi_energy_bank.sqlite"
AE_ENERGY_BANK_BG_STEM = "image_autoencoder_multi_energy_bank_bg"
AE_ENERGY_BANK_SCRIPT = "train_image_autoencoder_multi_energy_bank_bg.py"

FlowKind = Literal["image_energy_flow_energy_bank"]

_FLOW_SPECS: dict[FlowKind, dict[str, str]] = {
    "image_energy_flow_energy_bank": {
        "run_id": "image_energy_flow_energy_bank",
        "checkpoint": "image_energy_flow_energy_bank.pt",
        "log": "image_energy_flow_energy_bank.sqlite",
        "bg_stem": "image_energy_flow_energy_bank_bg",
        "script": "train_image_energy_flow_energy_bank_bg.py",
        "default_fn": "default_image_energy_flow_energy_bank_training_config",
        "done_marker": "IMAGE_ENERGY_FLOW_ENERGY_BANK_DONE",
    },
}


def ae_energy_bank_worker_source(
    *,
    target_steps: int,
    project_root: str | Path = "/marimo",
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    energy_bank_path: str = "data/encoded_energies_ae_best.pt",
    resume_from_checkpoint: bool = False,
) -> str:
    """Return Python source for the multi-pair AE worker (bank-pretrained flows)."""
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    root = Path(project_root)
    resolved_comment = comment or (
        f"multi-pair AE c128+c256 from bank flows; bg to {target_steps}; "
        "HF upload + notify"
    )
    upload = "True" if upload_artifacts_on_checkpoint else "False"
    notify = "True" if notify_on_finished else "False"
    resume = "True" if resume_from_checkpoint else "False"
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
                surrogate_checkpoint_name="surrogate_c128_k4.pt",
                decoder_checkpoint_name="decoder_c128_k4.pt",
                name="c128",
            ),
            ImageAutoencoderLpapPairConfig(
                surrogate_checkpoint_name="surrogate_c256_k4.pt",
                decoder_checkpoint_name="decoder_c256_k4.pt",
                name="c256",
            ),
        ),
        flow_checkpoint_name="image_energy_flow_energy_bank.pt",
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
        resume_from_checkpoint={resume},
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
    f"notify={{config.run.notify_on_finished}} resume={{config.run.resume_from_checkpoint}}",
    flush=True,
)
print(session.resume_info.message, flush=True)
print(f"best_so_far={{session.training_run.best_metric}}", flush=True)
print(
    "flow=image_energy_flow_energy_bank.pt "
    "pairs=c128,c256 bank={energy_bank_path!r}",
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
print("AE_MULTI_ENERGY_BANK_DONE", flush=True)
'''


def flow_energy_bank_worker_source(
    *,
    target_steps: int,
    project_root: str | Path = "/marimo",
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    energy_bank_path: str = "data/encoded_energies_ae_best.pt",
) -> str:
    """Return Python source for an unpaired energy-bank flow worker."""
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    kind: FlowKind = "image_energy_flow_energy_bank"
    spec = _FLOW_SPECS[kind]
    root = Path(project_root)
    resolved_comment = comment or (
        f"{kind} against {energy_bank_path}; bg to {target_steps}"
    )
    upload = "True" if upload_artifacts_on_checkpoint else "False"
    notify = "True" if notify_on_finished else "False"
    return f'''from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lpap.training_notebook import (
    create_training_session,
    {spec["default_fn"]},
    iter_training,
)

project_root = Path({str(root)!r})
TARGET = {int(target_steps)}
KIND = {kind!r}
base = {spec["default_fn"]}()
config = replace(
    base,
    prior=replace(
        base.prior,
        energy_bank=replace(base.prior.energy_bank, path={energy_bank_path!r}),
    ),
    run=replace(
        base.run,
        run_training=True,
        resume_from_checkpoint=False,
        steps=TARGET,
        display_every=25,
        log_every=5,
        run_id={spec["run_id"]!r},
        checkpoint_name={spec["checkpoint"]!r},
        log_name={spec["log"]!r},
        comment={resolved_comment!r},
        pinned=True,
    ),
)
session = create_training_session(KIND, project_root=project_root, config=config)
# Flow run configs do not yet carry these flags; set on TrainingRunConfig.
session.training_run.config = replace(
    session.training_run.config,
    upload_artifacts_on_checkpoint={upload},
    notify_on_finished={notify},
)
print(
    f"device={{session.device}} kind={{KIND}} start={{session.resume_info.start_step}} "
    f"target={{config.run.steps}} bank={energy_bank_path!r} "
    f"upload={{session.training_run.config.upload_artifacts_on_checkpoint}} "
    f"notify={{session.training_run.config.notify_on_finished}}",
    flush=True,
)
print(session.resume_info.message, flush=True)
for result in iter_training(KIND, session):
    if result.should_display or result.improved or result.step % 50 == 0:
        loss = result.metrics.get("loss", float("nan"))
        vloss = result.metrics.get("validation_loss")
        vtxt = "n/a" if vloss is None else f"{{vloss:.5f}}"
        best = "n/a" if result.best_metric is None else f"{{result.best_metric:.5f}}"
        mark = " *" if result.improved else ""
        ck = " ckpt" if result.checkpointed else ""
        print(
            f"step={{result.step}} loss={{loss:.5f}} val={{vtxt}} "
            f"best={{best}}{{mark}}{{ck}}",
            flush=True,
        )
print({spec["done_marker"]!r}, flush=True)
'''


def _spawn_job(
    *,
    project_root: Path,
    script_name: str,
    bg_stem: str,
    source: str,
    target_steps: int,
    run_id: str,
    checkpoint_name: str,
    log_name: str,
    upload_artifacts_on_checkpoint: bool,
    notify_on_finished: bool,
    require_secrets: bool,
) -> dict[str, Any]:
    logs = project_root / "training_logs"
    logs.mkdir(parents=True, exist_ok=True)
    script_path = logs / script_name
    pid_path = logs / f"{bg_stem}.pid"
    log_path = logs / f"{bg_stem}.log"
    script_path.write_text(source, encoding="utf-8")
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
        cwd=project_root,
        pid_path=pid_path,
        log_path=log_path,
        env=env,
    )
    return {
        **spawned,
        "target_steps": int(target_steps),
        "run_id": run_id,
        "checkpoint_name": checkpoint_name,
        "log_name": log_name,
        "bg_stem": bg_stem,
    }


def launch_ae_energy_bank_bg(
    project_root: str | Path = "/marimo",
    *,
    target_steps: int,
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    require_secrets: bool = True,
    energy_bank_path: str = "data/encoded_energies_ae_best.pt",
    resume_from_checkpoint: bool = False,
) -> dict[str, Any]:
    """Write + spawn the multi-pair AE worker (bank-pretrained flows)."""
    root = Path(project_root)
    return _spawn_job(
        project_root=root,
        script_name=AE_ENERGY_BANK_SCRIPT,
        bg_stem=AE_ENERGY_BANK_BG_STEM,
        source=ae_energy_bank_worker_source(
            target_steps=target_steps,
            project_root=root,
            upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
            notify_on_finished=notify_on_finished,
            comment=comment,
            energy_bank_path=energy_bank_path,
            resume_from_checkpoint=resume_from_checkpoint,
        ),
        target_steps=target_steps,
        run_id=AE_ENERGY_BANK_RUN_ID,
        checkpoint_name=AE_ENERGY_BANK_CHECKPOINT,
        log_name=AE_ENERGY_BANK_LOG,
        upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
        notify_on_finished=notify_on_finished,
        require_secrets=require_secrets,
    )


def launch_flow_energy_bank_bg(
    project_root: str | Path = "/marimo",
    *,
    target_steps: int,
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    require_secrets: bool = True,
    energy_bank_path: str = "data/encoded_energies_ae_best.pt",
) -> dict[str, Any]:
    """Write + spawn the bidirectional energy-bank flow worker."""
    root = Path(project_root)
    kind: FlowKind = "image_energy_flow_energy_bank"
    spec = _FLOW_SPECS[kind]
    return _spawn_job(
        project_root=root,
        script_name=spec["script"],
        bg_stem=spec["bg_stem"],
        source=flow_energy_bank_worker_source(
            target_steps=target_steps,
            project_root=root,
            upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
            notify_on_finished=notify_on_finished,
            comment=comment,
            energy_bank_path=energy_bank_path,
        ),
        target_steps=target_steps,
        run_id=spec["run_id"],
        checkpoint_name=spec["checkpoint"],
        log_name=spec["log"],
        upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
        notify_on_finished=notify_on_finished,
        require_secrets=require_secrets,
    )


def lpap_teacher_worker_source(
    *,
    backend_kind: Literal["surrogate", "decoder"],
    config_relpath: str,
    target_steps: int,
    project_root: str | Path = "/marimo",
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    resume_from_checkpoint: bool = False,
) -> str:
    """Return Python source for a surrogate/decoder worker from a TOML config."""
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if backend_kind not in ("surrogate", "decoder"):
        raise ValueError(f"unsupported backend_kind: {backend_kind}")
    root = Path(project_root)
    resolved_comment = comment or (
        f"{backend_kind} from {config_relpath}; bg to {target_steps}"
    )
    upload = "True" if upload_artifacts_on_checkpoint else "False"
    notify = "True" if notify_on_finished else "False"
    resume = "True" if resume_from_checkpoint else "False"
    done = {
        "surrogate": "SURROGATE_DONE",
        "decoder": "DECODER_DONE",
    }[backend_kind]
    return f'''from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lpap.training_notebook import (
    create_training_session,
    iter_training,
    training_config_from_file,
)

project_root = Path({str(root)!r})
TARGET = {int(target_steps)}
KIND = {backend_kind!r}
config_path = project_root / {config_relpath!r}
base = training_config_from_file(config_path, KIND)
config = replace(
    base,
    run=replace(
        base.run,
        run_training=True,
        resume_from_checkpoint={resume},
        steps=TARGET,
        display_every=25,
        log_every=5,
        comment={resolved_comment!r},
        pinned=True,
    ),
)
session = create_training_session(KIND, project_root=project_root, config=config)
session.training_run.config = replace(
    session.training_run.config,
    upload_artifacts_on_checkpoint={upload},
    notify_on_finished={notify},
)
print(
    f"device={{session.device}} kind={{KIND}} config={{config_path.name}} "
    f"start={{session.resume_info.start_step}} target={{config.run.steps}} "
    f"C={{config.data.bucket_count}} probes={{config.data.probe_count}} "
    f"N={{config.value_count}} upload={{session.training_run.config.upload_artifacts_on_checkpoint}} "
    f"notify={{session.training_run.config.notify_on_finished}}",
    flush=True,
)
print(session.resume_info.message, flush=True)
print(f"best_so_far={{session.training_run.best_metric}}", flush=True)
for result in iter_training(KIND, session):
    if result.should_display or result.improved or result.step % 50 == 0:
        loss = result.metrics.get("loss", float("nan"))
        vloss = result.metrics.get("validation_loss")
        acc = result.metrics.get("weighted_accuracy")
        vtxt = "n/a" if vloss is None else f"{{vloss:.5f}}"
        best = "n/a" if result.best_metric is None else f"{{result.best_metric:.5f}}"
        atxt = "n/a" if acc is None else f"{{acc:.5f}}"
        mark = " *" if result.improved else ""
        ck = " ckpt" if result.checkpointed else ""
        print(
            f"step={{result.step}} loss={{loss:.5f}} val={{vtxt}} "
            f"wacc={{atxt}} best={{best}}{{mark}}{{ck}}",
            flush=True,
        )
print({done!r}, flush=True)
'''


def launch_lpap_teacher_bg(
    *,
    backend_kind: Literal["surrogate", "decoder"],
    config_relpath: str,
    project_root: str | Path = "/marimo",
    target_steps: int,
    upload_artifacts_on_checkpoint: bool = True,
    notify_on_finished: bool = True,
    comment: str | None = None,
    require_secrets: bool = True,
    resume_from_checkpoint: bool = False,
) -> dict[str, Any]:
    """Write + spawn a surrogate or decoder worker from a training TOML."""
    root = Path(project_root)
    config_path = root / config_relpath
    if not config_path.is_file():
        raise BackgroundWorkerError(f"missing training config: {config_path}")
    # Derive artifact names from the TOML so pid/log stems stay stable.
    from lpap.training_notebook import training_config_from_file

    cfg = training_config_from_file(config_path, backend_kind)
    bg_stem = f"{cfg.run.run_id}_bg"
    return _spawn_job(
        project_root=root,
        script_name=f"train_{cfg.run.run_id}_bg.py",
        bg_stem=bg_stem,
        source=lpap_teacher_worker_source(
            backend_kind=backend_kind,
            config_relpath=config_relpath,
            target_steps=target_steps,
            project_root=root,
            upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
            notify_on_finished=notify_on_finished,
            comment=comment,
            resume_from_checkpoint=resume_from_checkpoint,
        ),
        target_steps=target_steps,
        run_id=cfg.run.run_id,
        checkpoint_name=cfg.run.checkpoint_name,
        log_name=cfg.run.log_name,
        upload_artifacts_on_checkpoint=upload_artifacts_on_checkpoint,
        notify_on_finished=notify_on_finished,
        require_secrets=require_secrets,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project-root", type=Path, default=Path("/marimo"))
        p.add_argument("--target-steps", type=int, required=True)
        p.add_argument("--no-upload", action="store_true")
        p.add_argument("--no-notify", action="store_true")
        p.add_argument("--comment", default=None)
        p.add_argument("--allow-missing-secrets", action="store_true")
        p.add_argument(
            "--energy-bank-path",
            default="data/encoded_energies_ae_best.pt",
        )
        p.add_argument("--json", action="store_true")

    launch = sub.add_parser(
        "launch-ae-energy-bank",
        help="spawn multi-pair AE from bank-pretrained flows",
    )
    add_common(launch)
    launch.add_argument(
        "--resume",
        action="store_true",
        help="resume from existing AE checkpoint instead of fresh init",
    )

    flow = sub.add_parser(
        "launch-flow-energy-bank",
        help="spawn the bidirectional image_energy_flow_energy_bank worker",
    )
    add_common(flow)

    teacher = sub.add_parser(
        "launch-lpap-teacher",
        help="spawn surrogate or decoder from a configs/training/*.toml",
    )
    teacher.add_argument(
        "--backend",
        required=True,
        choices=("surrogate", "decoder"),
    )
    teacher.add_argument(
        "--config",
        required=True,
        help="repo-relative TOML path, e.g. configs/training/surrogate_c512.toml",
    )
    teacher.add_argument("--resume", action="store_true")
    add_common(teacher)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "launch-ae-energy-bank":
            result = launch_ae_energy_bank_bg(
                args.project_root,
                target_steps=args.target_steps,
                upload_artifacts_on_checkpoint=not args.no_upload,
                notify_on_finished=not args.no_notify,
                comment=args.comment,
                require_secrets=not args.allow_missing_secrets,
                energy_bank_path=args.energy_bank_path,
                resume_from_checkpoint=bool(args.resume),
            )
        elif args.command == "launch-flow-energy-bank":
            result = launch_flow_energy_bank_bg(
                args.project_root,
                target_steps=args.target_steps,
                upload_artifacts_on_checkpoint=not args.no_upload,
                notify_on_finished=not args.no_notify,
                comment=args.comment,
                require_secrets=not args.allow_missing_secrets,
                energy_bank_path=args.energy_bank_path,
            )
        elif args.command == "launch-lpap-teacher":
            result = launch_lpap_teacher_bg(
                backend_kind=args.backend,
                config_relpath=args.config,
                project_root=args.project_root,
                target_steps=args.target_steps,
                upload_artifacts_on_checkpoint=not args.no_upload,
                notify_on_finished=not args.no_notify,
                comment=args.comment,
                require_secrets=not args.allow_missing_secrets,
                resume_from_checkpoint=bool(args.resume),
            )
        else:
            raise ValueError(f"unknown command: {args.command}")
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


if __name__ == "__main__":
    raise SystemExit(main())
