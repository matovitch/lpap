"""Export AE loss, teachers, and best-checkpoint gallery PNGs from a molab run.

Safe while training: logs come from the marimo file API (not molab-exec),
and the best AE checkpoint is pulled from the Hugging Face dual-slot pointer.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS_DEFAULT = ("c128_k16", "c256_k24", "c512_k32")
AE_STEM_DEFAULT = "image_autoencoder_tri_lnorm"
V2_VAL_LOSS = 0.01129
V2_PAIR_IMAGE_L2 = {"c512_k32": 0.00195, "c256_k24": 0.00462, "c128_k16": 0.01052}


def normalize_suffix(value: str | None) -> str:
    """Accept ``v3``, ``-v3``, or empty / ``-`` for no suffix."""
    text = (value or "").strip()
    if not text or text == "-":
        return ""
    return text if text.startswith("-") else f"-{text}"


def load_hf_token(*, secrets_path: Path | None = None) -> str | None:
    """Load an HF token into the environment without printing it.

    Resolution order: existing ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``,
    ``LPAP_SECRETS_TOML`` or ``secrets_path``, then the ``SECRETS_TOML`` env
    (which may be flattened onto one line).
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            os.environ["HF_TOKEN"] = value
            return value


    candidates: list[Path] = []
    if secrets_path is not None:
        candidates.append(secrets_path)
    env_path = (os.environ.get("LPAP_SECRETS_TOML") or "").strip()
    if env_path:
        candidates.append(Path(env_path))

    for path in candidates:
        if not path.is_file():
            continue
        token = _token_from_secrets_toml(path.read_text(encoding="utf-8"))
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            return token

    raw = os.environ.get("SECRETS_TOML")
    if raw:
        token = _token_from_secrets_toml(raw)
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            return token
    return None


def _token_from_secrets_toml(raw: str) -> str | None:
    import re
    import tomllib

    text = raw
    if "\n" not in text and "[" in text:
        # Flattened one-line dump: reintroduce newlines at tables and keys.
        text = re.sub(r"\s+\[", "\n[", text)
        text = re.sub(r"\]\s+", "]\n", text)
        text = re.sub(r"""(["'])\s+([A-Za-z_][A-Za-z0-9_]*\s*=)""", r"\1\n\2", text)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    huggingface = data.get("huggingface") if isinstance(data, dict) else None
    if not isinstance(huggingface, dict):
        return None
    token = str(huggingface.get("token") or "").strip()
    return token or None


def latest_run(con: sqlite3.Connection) -> str | None:
    for (rid,) in con.execute("select run_id from runs order by updated_at desc"):
        if con.execute(
            "select 1 from step_metrics where run_id=? limit 1", (rid,)
        ).fetchone():
            return str(rid)
    return None


def series(
    con: sqlite3.Connection, rid: str, name: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = con.execute(
        "select step, metric_value from step_metrics "
        "where run_id=? and metric_name=? order by step",
        (rid, name),
    ).fetchall()
    return (
        np.array([r[0] for r in rows], dtype=int),
        np.array([r[1] for r in rows], dtype=float),
    )


def resume_steps(con: sqlite3.Connection, rid: str) -> list[int]:
    try:
        rows = con.execute(
            "select step from run_events where run_id=? and kind=? order by step",
            (rid, "resume"),
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    return [int(r[0]) for r in rows]


def loss_weights(con: sqlite3.Connection, rid: str) -> dict[str, float]:
    defaults = {
        "img": 1.0,
        "e": 0.05,
        "ce": 0.05,
        "sm": 0.01,
    }
    try:
        row = con.execute(
            "select config_json from runs where run_id=?", (rid,)
        ).fetchone()
    except sqlite3.DatabaseError:
        return defaults
    if not row or not row[0]:
        return defaults
    try:
        loss = (json.loads(row[0]).get("loss") or {})
    except json.JSONDecodeError:
        return defaults
    return {
        "img": float(loss.get("image_l2_weight", defaults["img"])),
        "e": float(loss.get("energy_l1_weight", defaults["e"])),
        "ce": float(loss.get("surrogate_teacher_weight", defaults["ce"])),
        "sm": float(loss.get("signed_mass_balance_weight", defaults["sm"])),
    }


def _fetch_optional_log(remote: str, dest: Path) -> Path | None:
    from molab_file_api import MolabFileError, download_file

    try:
        path = download_file(remote, dest)
    except FileNotFoundError:
        return None
    except MolabFileError as exc:
        text = str(exc)
        if "No such file" in text or "Errno 2" in text:
            return None
        raise
    print(f"ok file-api {remote} -> {path} ({path.stat().st_size} bytes)", flush=True)
    return path


def fetch_ae_log(*, stem: str, logs_dir: Path) -> Path:
    dest = logs_dir / f"{stem}.sqlite"
    remote = f"/marimo/training_logs/{stem}.sqlite"
    path = _fetch_optional_log(remote, dest)
    if path is None:
        raise FileNotFoundError(f"AE log not on molab: {remote}")
    for suffix in ("-wal", "-shm"):
        _fetch_optional_log(f"{remote}{suffix}", logs_dir / f"{stem}.sqlite{suffix}")
    return path


def ensure_teacher_logs(
    *,
    suffix: str,
    pairs: tuple[str, ...],
    logs_dir: Path,
    project_root: Path,
) -> list[Path]:
    found: list[Path] = []
    for kind in ("surrogate", "decoder"):
        for pair in pairs:
            name = f"{kind}_{pair}{suffix}.sqlite"
            dest = logs_dir / name
            path = _fetch_optional_log(f"/marimo/training_logs/{name}", dest)
            if path is not None:
                found.append(path)
                continue
            if dest.is_file():
                found.append(dest)
                continue
            try:
                from lpap.artifact_sync import ensure_training_log

                ensure_training_log(project_root, name)
            except Exception as exc:
                print(
                    f"skip teacher log {name} ({type(exc).__name__})",
                    flush=True,
                )
                continue
            if dest.is_file():
                found.append(dest)
    return found


def ensure_best_checkpoint(
    *,
    stem: str,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    from lpap.artifact_sync import (
        download_checkpoint_from_bucket,
        read_checkpoint_pointer,
    )

    dest = project_root / "checkpoints" / f"{stem}.pt"
    pointer = read_checkpoint_pointer(stem, project_root=project_root)
    pointer_step = pointer.get("step")
    pointer_size = pointer.get("size")
    print(
        f"hf pointer {stem}.pt slot={pointer.get('slot')} "
        f"step={pointer_step} size={pointer_size}",
        flush=True,
    )
    reuse = False
    if dest.is_file() and pointer_size is not None and dest.stat().st_size == int(
        pointer_size
    ):
        local_step = _checkpoint_step(dest)
        if local_step is not None and pointer_step is not None and int(local_step) == int(
            pointer_step
        ):
            reuse = True
            print(
                f"reuse local checkpoint {dest} (step {local_step} matches pointer)",
                flush=True,
            )
    if not reuse:
        dest = download_checkpoint_from_bucket(
            dest, canonical_name=f"{stem}.pt", project_root=project_root
        )
        print(f"ok hf checkpoint {dest} ({dest.stat().st_size} bytes)", flush=True)
    return dest, pointer


def _checkpoint_step(path: Path) -> int | None:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or "step" not in payload:
        return None
    try:
        return int(payload["step"])
    except (TypeError, ValueError):
        return None


def render_teachers(
    *,
    logs_dir: Path,
    suffix: str,
    pairs: tuple[str, ...],
    output: Path,
) -> dict[str, Any]:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    summary: dict[str, Any] = {}
    any_plotted = False
    for i, pair in enumerate(pairs):
        color = f"C{i}"
        for kind, metric, axis in (
            ("surrogate", "validation_loss", ax[0]),
            ("decoder", "validation_reconstruction_l1", ax[1]),
        ):
            log = logs_dir / f"{kind}_{pair}{suffix}.sqlite"
            if not log.is_file():
                continue
            con = sqlite3.connect(f"file:{log}?mode=ro", uri=True)
            rid = latest_run(con)
            if rid is None:
                con.close()
                continue
            steps, values = series(con, rid, metric)
            if len(steps) == 0:
                con.close()
                continue
            axis.plot(steps, values, color=color, lw=1.3, label=pair)
            any_plotted = True
            _, val_loss = series(con, rid, "validation_loss")
            acc_s, acc = series(con, rid, "validation_accuracy")
            if kind == "decoder" and len(acc_s):
                ax[2].plot(acc_s, acc, color=color, lw=1.3, label=pair)
            summary[f"{kind}/{pair}"] = {
                "step": int(steps[-1]),
                "best_val": float(val_loss.min()) if len(val_loss) else None,
                "last": float(values[-1]),
                "acc": float(acc[-1]) if len(acc) else None,
            }
            con.close()
    for axis, title in zip(
        ax,
        (
            "surrogate validation loss",
            "decoder validation recon L1",
            "decoder validation accuracy",
        ),
    ):
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=7)
    ax[0].set_yscale("log")
    fig.suptitle(f"LPAP teachers{suffix}", fontsize=10)
    fig.tight_layout()
    if any_plotted:
        fig.savefig(output, dpi=130)
    plt.close(fig)
    if not any_plotted:
        print("skip teachers plot (no teacher logs)", flush=True)
        return summary
    print(f"wrote {output}", flush=True)
    return summary


def render_ae_loss(
    *,
    log_path: Path,
    suffix: str,
    pairs: tuple[str, ...],
    output: Path,
) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{log_path}?mode=ro", uri=True)
    rid = latest_run(con)
    if rid is None:
        con.close()
        raise RuntimeError(f"no metrics in {log_path}")
    weights = loss_weights(con, rid)
    st, loss = series(con, rid, "loss")
    _, img = series(con, rid, "image_reconstruction_l2")
    _, energy = series(con, rid, "energy_reconstruction_l1")
    _, ce = series(con, rid, "surrogate_teacher_ce")
    _, sm = series(con, rid, "signed_mass_balance")
    vs, vl = series(con, rid, "validation_loss")
    resumes = resume_steps(con, rid)
    n = min(map(len, (st, loss, img, energy, ce, sm)))
    st, loss, img, energy, ce, sm = (arr[:n] for arr in (st, loss, img, energy, ce, sm))
    fig, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True)
    axis = axes[0]
    axis.plot(st, loss, lw=1, color="#92400e", label="train loss")
    if len(vs):
        axis.plot(vs, vl, lw=1.5, color="#9467bd", marker="o", ms=3, label="val loss")
    axis.axhline(V2_VAL_LOSS, ls="--", color="gray", lw=1, label=f"v2 best val {V2_VAL_LOSS}")
    for i, step in enumerate(resumes):
        axis.axvline(
            step,
            color="#16a34a",
            ls=":",
            lw=1.1,
            alpha=0.85,
            label="resume" if i == 0 else None,
        )
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    axis.set_title(f"AE loss · {rid}")
    terms = {
        "image L2": weights["img"] * img,
        "surrogate CE": weights["ce"] * ce,
        "energy L1": weights["e"] * energy,
        "signed-mass": weights["sm"] * sm,
    }
    colors = {
        "image L2": "#1f77b4",
        "energy L1": "#ff7f0e",
        "surrogate CE": "#2ca02c",
        "signed-mass": "#d62728",
    }
    axis = axes[1]
    for name, values in terms.items():
        axis.plot(st, values, lw=1.2, color=colors[name], label=f"λ·{name}")
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    axis.set_title("weighted terms")
    axis = axes[2]
    axis.stackplot(
        st,
        *[values / (loss + 1e-12) for values in terms.values()],
        labels=list(terms),
        colors=[colors[name] for name in terms],
        alpha=0.85,
    )
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(True, alpha=0.3)
    axis.set_title("term share of total loss")
    axis = axes[3]
    ae_pairs: dict[str, tuple[int, float] | None] = {}
    for i, pair in enumerate(pairs):
        steps, values = series(con, rid, f"validation_image_reconstruction_l2/{pair}")
        axis.plot(steps, values, color=f"C{i}", marker="o", ms=3, lw=1.3, label=f"{pair} val image L2")
        if pair in V2_PAIR_IMAGE_L2:
            axis.axhline(V2_PAIR_IMAGE_L2[pair], color=f"C{i}", ls="--", lw=0.8)
        ae_pairs[pair] = (int(steps[-1]), float(values[-1])) if len(steps) else None
    axis.set_yscale("log")
    axis.set_xlabel("step")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    axis.set_title("per-pair validation image L2 (dashed = v2 final)")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)
    con.close()
    print(f"wrote {output}", flush=True)
    best_val = float(vl.min()) if len(vl) else None
    best_step = int(vs[int(vl.argmin())]) if len(vl) else None
    return {
        "run_id": rid,
        "train_step": int(st[-1]) if len(st) else None,
        "train_loss": float(loss[-1]) if len(loss) else None,
        "best_val": best_val,
        "best_val_step": best_step,
        "last_val": (int(vs[-1]), float(vl[-1])) if len(vs) else None,
        "val_points": list(zip(vs.tolist(), [round(v, 5) for v in vl.tolist()])),
        "per_pair_val_image_l2": ae_pairs,
        "resume_steps": resumes,
    }


def render_gallery(
    *,
    project_root: Path,
    checkpoint: Path,
    suffix: str,
    output: Path,
    sample_count: int = 6,
) -> dict[str, Any]:
    import torch
    from lpap.checkpoints import load_training_checkpoint
    from lpap.image_autoencoder_training import (
        collect_image_autoencoder_gallery,
        create_image_autoencoder_training_session,
    )
    from lpap.training_notebook import training_config_from_file

    torch.set_num_threads(8)
    toml_path = (
        project_root / "configs" / "training" / "generated" / f"image_autoencoder{suffix}.toml"
    )
    if not toml_path.is_file():
        toml_path = project_root / "configs" / "training" / "image_autoencoder.toml"
    cfg = training_config_from_file(toml_path, "image_autoencoder")
    cfg = replace(
        cfg,
        run=replace(
            cfg.run,
            run_training=False,
            resume_from_checkpoint=False,
            checkpoint_name="_export_scratch.pt",
            log_name="_export_scratch.sqlite",
            pinned=False,
        ),
    )
    sess = create_image_autoencoder_training_session(
        project_root=project_root, config=cfg, device="cpu"
    )
    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    sess.model.load_state_dict(payload.get("best_model_state") or payload["model_state"])
    items = collect_image_autoencoder_gallery(sess, sample_count=sample_count)
    side = int(cfg.image.side)

    def as_image(tensor: Any) -> np.ndarray:
        return tensor.detach().float().cpu().reshape(side, -1).numpy()

    ncol = 1 + 2 * len(items[0].pairs)
    fig, axes = plt.subplots(
        len(items), ncol, figsize=(1.6 * ncol, 1.6 * len(items)), squeeze=False
    )
    errs: dict[str, list[float]] = {}
    for row, item in enumerate(items):
        order = sorted(item.pairs, key=lambda p: -int(p.name.split("_")[0].lstrip("c")))
        panels = [("input", as_image(item.image), "gray")]
        panels += [(f"{p.name} recon", as_image(p.reconstructed_image), "gray") for p in order]
        panels += [(f"{p.name} |err|", np.abs(as_image(p.image_error)), "magma") for p in order]
        for pair in order:
            errs.setdefault(pair.name, []).append(float((pair.image_error.float() ** 2).mean()))
        input_arr = as_image(item.image)
        vmin, vmax = float(input_arr.min()), float(input_arr.max())
        for col, (title, arr, cmap) in enumerate(panels):
            axis = axes[row][col]
            if cmap == "gray":
                axis.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
            else:
                axis.imshow(arr, cmap=cmap)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=7)
    step = payload.get("step")
    best = payload.get("best_metric")
    fig.suptitle(
        f"{checkpoint.name} · step {step} · best val {best:.4g}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}", flush=True)
    return {
        "ckpt_step": step,
        "best_metric": best,
        "gallery_mean_sq_err": {k: round(float(np.mean(v)), 5) for k, v in errs.items()},
    }


def fetch_generated_config(*, suffix: str, project_root: Path) -> None:
    dest = project_root / "configs" / "training" / "generated" / f"image_autoencoder{suffix}.toml"
    _fetch_optional_log(
        f"/marimo/configs/training/generated/image_autoencoder{suffix}.toml",
        dest,
    )



def _glue_dashed_values(argv: list[str] | None) -> list[str] | None:
    """Allow ``--suffix -v3`` (argparse would otherwise treat -v3 as a flag)."""
    if argv is None:
        argv = sys.argv[1:]
    out: list[str] = []
    value_flags = {"--suffix", "--run", "--ae-stem"}
    i = 0
    while i < len(argv):
        current = argv[i]
        if (
            current in value_flags
            and i + 1 < len(argv)
            and argv[i + 1].startswith("-")
            and not argv[i + 1].startswith("--")
        ):
            out.append(f"{current}={argv[i + 1]}")
            i += 2
            continue
        out.append(current)
        i += 1
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export AE loss / gallery / teachers PNGs from a live molab run "
            "without touching the kernel."
        )
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="run suffix, with or without a leading dash (e.g. v3 or -v3)",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="alias for --suffix (e.g. --run v3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for ae_loss, ae_gallery, and teachers PNGs",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="local cache with configs/storage.toml, data/, and teacher checkpoints",
    )
    parser.add_argument(
        "--ae-stem",
        default=AE_STEM_DEFAULT,
        help=f"AE artifact stem without suffix (default {AE_STEM_DEFAULT})",
    )
    parser.add_argument(
        "--pairs",
        default=",".join(PAIRS_DEFAULT),
        help="comma-separated teacher pair names",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=6,
        help="gallery sample count (default 6)",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=None,
        help="optional configs/secrets.toml for the HF token",
    )
    args = parser.parse_args(_glue_dashed_values(argv))
    if args.suffix and args.run and normalize_suffix(args.suffix) != normalize_suffix(args.run):
        parser.error("--suffix and --run disagree")
    args.suffix = normalize_suffix(args.suffix if args.suffix is not None else args.run)
    args.pairs = tuple(p.strip() for p in str(args.pairs).split(",") if p.strip())
    args.output_dir = args.output_dir.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    repo_root = Path(__file__).resolve().parents[4]
    secrets = args.secrets
    if secrets is None:
        default_secrets = repo_root / "configs" / "secrets.toml"
        if default_secrets.is_file():
            secrets = default_secrets
    token = load_hf_token(secrets_path=secrets)
    if not token:
        print(
            "warning: no HF token found; checkpoint download will fail if the bucket is private",
            flush=True,
        )

    stem = f"{args.ae_stem}{args.suffix}"
    logs_dir = args.project_root / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fetch_generated_config(suffix=args.suffix, project_root=args.project_root)
    ae_log = fetch_ae_log(stem=stem, logs_dir=logs_dir)
    teacher_logs = ensure_teacher_logs(
        suffix=args.suffix,
        pairs=args.pairs,
        logs_dir=logs_dir,
        project_root=args.project_root,
    )
    print(f"teacher logs ready: {len(teacher_logs)}", flush=True)
    checkpoint, pointer = ensure_best_checkpoint(stem=stem, project_root=args.project_root)

    teachers = render_teachers(
        logs_dir=logs_dir,
        suffix=args.suffix,
        pairs=args.pairs,
        output=args.output_dir / f"teachers{args.suffix}.png",
    )
    loss_info = render_ae_loss(
        log_path=ae_log,
        suffix=args.suffix,
        pairs=args.pairs,
        output=args.output_dir / f"ae_loss{args.suffix}.png",
    )
    gallery = render_gallery(
        project_root=args.project_root,
        checkpoint=checkpoint,
        suffix=args.suffix,
        output=args.output_dir / f"ae_gallery{args.suffix}.png",
        sample_count=args.sample_count,
    )
    print("ae log", loss_info, flush=True)
    print("teachers", teachers, flush=True)
    print("gallery", gallery, flush=True)
    print("hf pointer step", pointer.get("step"), "size", pointer.get("size"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
