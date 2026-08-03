import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def ae_setup():
    # cell: ae_setup
    import sys
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo
    import torch

    if Path("/marimo").is_dir() and (
        Path("/marimo/notebook.py").exists() or Path("/marimo/checkpoints").is_dir()
    ):
        project_root = Path("/marimo")
    else:
        project_root = Path(__file__).resolve().parents[1]
        _src = project_root / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    (project_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (project_root / "training_logs").mkdir(parents=True, exist_ok=True)

    from lpap.image_autoencoder_training import (
        ImageAutoencoderLpapPairConfig,
        ImageAutoencoderRunConfig,
        ImageAutoencoderSourceConfig,
        collect_image_autoencoder_gallery,
        create_image_autoencoder_training_session,
    )
    from lpap.training import load_training_checkpoint
    from lpap.training_notebook import default_image_autoencoder_training_config

    _base = default_image_autoencoder_training_config()
    ae_base = replace(
        _base,
        source=ImageAutoencoderSourceConfig(
            lpap_pairs=(
                ImageAutoencoderLpapPairConfig(
                    surrogate_checkpoint_name="surrogate_c128_k16.pt",
                    decoder_checkpoint_name="decoder_c128_k16.pt",
                    name="c128_k16",
                ),
                ImageAutoencoderLpapPairConfig(
                    surrogate_checkpoint_name="surrogate_c256_k24.pt",
                    decoder_checkpoint_name="decoder_c256_k24.pt",
                    name="c256_k24",
                ),
            ),
            flow_checkpoint_name="image_energy_flow.pt",
            load_best=True,
            require_checkpoints=True,
            train_image_to_energy_flow=True,
            train_surrogate=True,
            train_decoder=True,
            train_energy_to_image_flow=True,
        ),
        run=ImageAutoencoderRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=20_000,
            seed=_base.run.seed,
            display_every=25,
            log_every=5,
            run_id="image_autoencoder_multi_flow",
            checkpoint_name="image_autoencoder_multi_flow.pt",
            log_name="image_autoencoder_multi_flow.sqlite",
            comment="multi-pair c128_k16+c256_k24 from image_energy_flow@10k; fresh 20k",
            pinned=False,
        ),
    )

    mo.md(
        f"""
    ## AE monitor · bidirectional flow

    - pairs: `{", ".join(p.name for p in ae_base.source.lpap_pairs)}`
    - flow: `{ae_base.source.flow_checkpoint_name}`
    - ckpt / log: `{ae_base.run.checkpoint_name}` / `{ae_base.run.log_name}`
    - Euler: i2e=`{ae_base.integration.image_to_energy_steps}` · e2i=`{ae_base.integration.energy_to_image_steps}`
    """
    )
    return (
        ae_base,
        collect_image_autoencoder_gallery,
        create_image_autoencoder_training_session,
        load_training_checkpoint,
        mo,
        project_root,
        replace,
        torch,
    )


@app.cell
def status(ae_base, mo, project_root, torch):
    # cell: status
    from molab.bg_worker import (
        bg_log_tail as _bg_log_tail,
        bg_worker_status as _bg_worker_status,
        last_bg_log_step as _last_bg_log_step,
    )
    import json as _json

    _needed = {
        "c128_surr": project_root / "checkpoints" / "surrogate_c128_k16.pt",
        "c128_dec": project_root / "checkpoints" / "decoder_c128_k16.pt",
        "c256_surr": project_root / "checkpoints" / "surrogate_c256_k24.pt",
        "c256_dec": project_root / "checkpoints" / "decoder_c256_k24.pt",
        "flow": project_root / "checkpoints" / "image_energy_flow.pt",
        "ae": project_root / "checkpoints" / ae_base.run.checkpoint_name,
    }
    _lines = []
    for _label, _path in _needed.items():
        if not _path.is_file():
            _lines.append(f"- `{_label}`: **missing**")
            continue
        _payload = torch.load(_path, map_location="cpu", weights_only=False)
        _ts = _payload.get("training_state_json")
        if isinstance(_ts, str):
            _ts = _json.loads(_ts)
        _mc = (_ts or {}).get("model_config") or {}
        _extra = ""
        if _mc.get("value_count") is not None:
            _extra = f" N={_mc.get('value_count')} C={_mc.get('bucket_count')}"
        _lines.append(
            f"- `{_label}`: step={_payload.get('step')} best={_payload.get('best_metric')}{_extra}"
        )

    _bg = _bg_worker_status(
        project_root / "training_logs" / "image_autoencoder_multi_flow_bg.pid"
    )
    _step = _last_bg_log_step(
        project_root / "training_logs" / "image_autoencoder_multi_flow_bg.log"
    )
    _tail = _bg_log_tail(
        project_root / "training_logs" / "image_autoencoder_multi_flow_bg.log",
        lines=8,
    )

    mo.md(
        f"""
    ### Status

    {chr(10).join(_lines)}

    - AE bg alive: **{_bg.get("alive")}** (pid={_bg.get("pid")}) · log step: `{_step}`
    ```
    {chr(10).join(_tail) if _tail else "(no AE bg log yet)"}
    ```
    """
    )
    return


@app.cell
def gallery_view(ae_gallery_items, ae_gallery_meta, display_gamma, mo):
    # cell: gallery_view
    from lpap.training_plots import render_image_autoencoder_gallery_html as _render_ae_gallery

    if not ae_gallery_meta.get("ok") or ae_gallery_items is None:
        ae_gallery = mo.md(ae_gallery_meta.get("message", "Gallery unavailable."))
    else:
        _html = _render_ae_gallery(
            ae_gallery_items,
            size=int(ae_gallery_meta["side"]),
            display_px=154,
            gamma=float(display_gamma.value),
        )
        ae_gallery = mo.vstack(
            [
                mo.md(
                    f"### AE gallery (6 samples) · step=`{ae_gallery_meta['step']}` "
                    f"best=`{ae_gallery_meta['best']}` · `{ae_gallery_meta['checkpoint_name']}` · "
                    f"γ=`{float(display_gamma.value):g}` · `{ae_gallery_meta.get('device', '?')}`"
                ),
                display_gamma,
                mo.Html(_html),
            ]
        )
    ae_gallery
    return


@app.cell
def gallery_cache(
    ae_base,
    collect_image_autoencoder_gallery,
    create_image_autoencoder_training_session,
    load_training_checkpoint,
    project_root,
    replace,
):
    # cell: gallery_cache
    import shutil as _shutil

    _live = project_root / "checkpoints" / ae_base.run.checkpoint_name
    _ckpt = project_root / "checkpoints" / "_gallery_snapshot.pt"
    if _live.is_file():
        try:
            _shutil.copy2(_live, _ckpt)
        except Exception:
            _ckpt = _live
    elif not _ckpt.is_file():
        _ckpt = _live

    ae_gallery_items = None
    ae_gallery_meta = {
        "ok": False,
        "message": f"Gallery waiting for checkpoint `{ae_base.run.checkpoint_name}`.",
    }

    if _ckpt.is_file():
        # CPU: bg AE worker holds the GPU.
        _gallery_cfg = replace(
            ae_base,
            run=replace(ae_base.run, resume_from_checkpoint=False, run_training=False),
        )
        _gallery_session = create_image_autoencoder_training_session(
            project_root=project_root, config=_gallery_cfg, device="cpu"
        )
        _payload = load_training_checkpoint(_ckpt, map_location=_gallery_session.device)
        _state = _payload.get("best_model_state") or _payload["model_state"]
        _gallery_session.model.load_state_dict(_state)
        ae_gallery_items = collect_image_autoencoder_gallery(
            _gallery_session, sample_count=6
        )
        ae_gallery_meta = {
            "ok": True,
            "step": _payload.get("step"),
            "best": _payload.get("best_metric"),
            "checkpoint_name": ae_base.run.checkpoint_name,
            "side": ae_base.image.side,
            "device": str(_gallery_session.device),
        }
    return ae_gallery_items, ae_gallery_meta


@app.cell
def gallery_gamma(mo):
    # cell: gallery_gamma
    display_gamma = mo.ui.slider(
        start=0.2,
        stop=2.0,
        value=1.0,
        step=0.05,
        label="display γ (<1 lifts small diffs / energy magnitudes)",
        show_value=True,
    )
    display_gamma
    return (display_gamma,)


if __name__ == "__main__":
    app.run()
