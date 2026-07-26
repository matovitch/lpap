import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def ae_setup():
    # cell: ae_setup
    import os
    import subprocess
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
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    (project_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (project_root / "training_logs").mkdir(parents=True, exist_ok=True)

    if not os.environ.get("HF_TOKEN", "").strip() and project_root == Path("/marimo"):
        mo.status.toast("HF_TOKEN missing — run molab-inject-secrets.sh", kind="danger")

    install_spec = "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
    _force = os.environ.get("LPAP_FORCE_REINSTALL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        import lpap  # noqa: F401
        if _force:
            raise ImportError("LPAP_FORCE_REINSTALL")
        install_note = "lpap already importable"
    except ImportError:
        if project_root == Path("/marimo"):
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-deps",
                    install_spec,
                ]
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "jaxtyping>=0.3.7",
                    "zstandard>=0.25.0,<0.26",
                    "huggingface_hub>=1.21.0,<2",
                ]
            )
            for _mod in list(sys.modules):
                if _mod == "lpap" or _mod.startswith("lpap."):
                    del sys.modules[_mod]
            install_note = f"force-reinstalled {install_spec}"
        else:
            install_note = f"using local sources at {project_root / 'src'}"

    from lpap.image_autoencoder_training import (
        ImageAutoencoderLpapPairConfig,
        ImageAutoencoderRunConfig,
        ImageAutoencoderSourceConfig,
        collect_image_autoencoder_gallery,
        create_image_autoencoder_training_session,
    )
    from lpap.training import load_training_checkpoint
    from lpap.training_notebook import (
        create_training_session,
        default_image_autoencoder_training_config,
        iter_training,
    )
    from lpap.training_plots import render_image_autoencoder_gallery_html

    _base = default_image_autoencoder_training_config()
    ae_base = replace(
        _base,
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
        run=ImageAutoencoderRunConfig(
            run_training=True,
            resume_from_checkpoint=True,
            steps=20_000,
            seed=_base.run.seed,
            display_every=25,
            log_every=5,
            run_id="image_autoencoder_multi_harmonics",
            checkpoint_name="image_autoencoder_multi_harmonics.pt",
            log_name="image_autoencoder_multi_harmonics.sqlite",
            comment="multi-pair c128+c256 from harmonics flows; fresh 20k",
            pinned=False,
        ),
    )
    _pair_names = [p.name for p in ae_base.source.lpap_pairs]
    assert len(ae_base.source.lpap_pairs) == 2

    mo.md(
        f"""
    ## image_autoencoder setup (harmonics flows · multi-C)

    - install: `{install_note}`
    - HF: `{"HF_TOKEN set" if os.environ.get("HF_TOKEN", "").strip() else "HF_TOKEN missing"}`
    - LPAP pairs: `{", ".join(_pair_names)}`
    - flows: i2e=`{ae_base.source.image_to_energy_checkpoint_name}` · e2i=`{ae_base.source.energy_to_image_checkpoint_name}`
    - Euler: i2e=`{ae_base.integration.image_to_energy_steps}` · e2i=`{ae_base.integration.energy_to_image_steps}`
    - ckpt / log: `{ae_base.run.checkpoint_name}` / `{ae_base.run.log_name}`
    - steps: `{ae_base.run.steps}`
    - comment: `{ae_base.run.comment}`
    """
    )

    return (
        ImageAutoencoderLpapPairConfig,
        ImageAutoencoderRunConfig,
        ImageAutoencoderSourceConfig,
        Path,
        ae_base,
        collect_image_autoencoder_gallery,
        create_image_autoencoder_training_session,
        create_training_session,
        default_image_autoencoder_training_config,
        install_note,
        install_spec,
        iter_training,
        load_training_checkpoint,
        mo,
        os,
        project_root,
        render_image_autoencoder_gallery_html,
        replace,
        subprocess,
        sys,
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
        "c128_surr": project_root / "checkpoints" / "surrogate_synthetic.pt",
        "c128_dec": project_root / "checkpoints" / "decoder_synthetic.pt",
        "c256_surr": project_root / "checkpoints" / "surrogate_c256.pt",
        "c256_dec": project_root / "checkpoints" / "decoder_c256.pt",
        "i2e": project_root / "checkpoints" / "image_to_energy.pt",
        "e2i": project_root / "checkpoints" / "energy_to_image.pt",
        "ae": project_root / "checkpoints" / ae_base.run.checkpoint_name,
    }
    _lines = []
    for label, path in _needed.items():
        if not path.is_file():
            _lines.append(f"- `{label}`: **missing**")
            continue
        _p = torch.load(path, map_location="cpu", weights_only=False)
        _ts = _p.get("training_state_json")
        if isinstance(_ts, str):
            _ts = _json.loads(_ts)
        _mc = (_ts or {}).get("model_config") or {}
        _n = _mc.get("value_count")
        _c = _mc.get("bucket_count")
        _extra = f" N={_n} C={_c}" if _n is not None else ""
        _lines.append(
            f"- `{label}`: step={_p.get('step')} best={_p.get('best_metric')}{_extra}"
        )

    _ae_pid = _bg_worker_status(
        project_root / "training_logs" / "image_autoencoder_multi_harmonics_bg.pid"
    )
    _ae_step = _last_bg_log_step(
        project_root / "training_logs" / "image_autoencoder_multi_harmonics_bg.log"
    )
    _tail = _bg_log_tail(
        project_root / "training_logs" / "image_autoencoder_multi_harmonics_bg.log",
        lines=8,
    )

    mo.md(
        f"""
    ### Artifact readiness (multi-pair AE · harmonics flows)

    {chr(10).join(_lines)}

    - AE bg alive: **{_ae_pid.get("alive")}** (pid={_ae_pid.get("pid")}) · log step: `{_ae_step}`
    ```
    {chr(10).join(_tail) if _tail else "(no AE bg log yet)"}
    ```
    """
    )

    return


@app.cell
def gallery_cache(ae_base, collect_image_autoencoder_gallery, create_image_autoencoder_training_session, load_training_checkpoint, project_root, replace):
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
    ae_gallery_meta = {"ok": False, "message": f"Gallery waiting for checkpoint `{ae_base.run.checkpoint_name}`."}

    if _ckpt.is_file():
        _gallery_cfg = replace(
            ae_base,
            run=replace(ae_base.run, resume_from_checkpoint=False, run_training=False),
        )
        _gallery_session = create_image_autoencoder_training_session(
            project_root=project_root, config=_gallery_cfg
        )
        _payload = load_training_checkpoint(_ckpt, map_location=_gallery_session.device)
        _state = _payload.get("best_model_state") or _payload["model_state"]
        _gallery_session.model.load_state_dict(_state)
        ae_gallery_items = collect_image_autoencoder_gallery(
            _gallery_session, sample_count=4
        )
        ae_gallery_meta = {
            "ok": True,
            "step": _payload.get("step"),
            "best": _payload.get("best_metric"),
            "checkpoint_name": ae_base.run.checkpoint_name,
            "side": ae_base.image.side,
        }

    return (
        ae_gallery_items,
        ae_gallery_meta,
    )


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
                    f"### AE gallery (4 samples) · ckpt step=`{ae_gallery_meta['step']}` "
                    f"best=`{ae_gallery_meta['best']}` · `{ae_gallery_meta['checkpoint_name']}` · "
                    f"γ=`{float(display_gamma.value):g}`"
                ),
                display_gamma,
                mo.Html(_html),
            ]
        )
    ae_gallery

    return (ae_gallery,)


@app.cell
def e0_peak_probe(ae_base, create_image_autoencoder_training_session, load_training_checkpoint, mo, project_root, replace, torch):
    # cell: e0_peak_probe
    from lpap.flow import integrate_euler_midpoint_time
    from lpap.flow_training import cycle_image_batches, prepare_image_sequence
    from lpap.hilbert import hilbert_unflatten_images
    from lpap.training_plots import _grayscale_png_img, _signed_png_img

    _ckpt = project_root / "checkpoints" / "_gallery_snapshot.pt"
    if not _ckpt.is_file():
        _ckpt = project_root / "checkpoints" / ae_base.run.checkpoint_name
    _peak_n_dist = 64
    _peak_n_show = 4
    _peak_display = 120

    if not _ckpt.is_file():
        peak_probe = mo.md(f"Peak probe waiting for `{_ckpt.name}`.")
    else:
        _peak_cfg = replace(
            ae_base,
            run=replace(ae_base.run, resume_from_checkpoint=False, run_training=False),
        )
        _peak_session = create_image_autoencoder_training_session(
            project_root=project_root, config=_peak_cfg
        )
        _payload = load_training_checkpoint(_ckpt, map_location=_peak_session.device)
        _state = _payload.get("best_model_state") or _payload["model_state"]
        _peak_session.model.load_state_dict(_state)
        _peak_session.model.eval()

        _side = _peak_session.config.image.side
        _i2e_steps = _peak_session.config.integration.image_to_energy_steps
        _e2i_steps = _peak_session.config.integration.energy_to_image_steps
        _images = next(cycle_image_batches(_peak_session.validation_image_loader))[
            :_peak_n_dist
        ]
        _seq = prepare_image_sequence(_images, side=_side, device=_peak_session.device)
        with torch.no_grad():
            _encoded = integrate_euler_midpoint_time(
                _peak_session.model.image_to_energy_flow, _seq, _i2e_steps
            )[
                :, 0
            ]  # (B, N)
            _e0 = _encoded[:, 0].detach().float().cpu()
            _qs = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            _qvals = torch.quantile(_e0, _qs)

            # Sweep: only-peak energies at distribution quantiles → e2i
            _sweep_vals = _qvals
            _sweep_energy = torch.zeros(
                len(_sweep_vals), 1, _encoded.shape[-1], device=_peak_session.device
            )
            for _i, _v in enumerate(_sweep_vals.tolist()):
                _sweep_energy[_i, 0, 0] = float(_v)
            _sweep_img = integrate_euler_midpoint_time(
                _peak_session.model.energy_to_image_flow, _sweep_energy, _e2i_steps
            )
            _sweep_spatial = hilbert_unflatten_images(_sweep_img, side=_side)[:, 0]

            # Per-sample ablations on first K images
            _show = _encoded[:_peak_n_show]
            _only = torch.zeros_like(_show)
            _only[:, 0] = _show[:, 0]
            _ablate = _show.clone()
            _ablate[:, 0] = 0.0
            _full_img = integrate_euler_midpoint_time(
                _peak_session.model.energy_to_image_flow,
                _show.unsqueeze(1),
                _e2i_steps,
            )
            _only_img = integrate_euler_midpoint_time(
                _peak_session.model.energy_to_image_flow,
                _only.unsqueeze(1),
                _e2i_steps,
            )
            _ablate_img = integrate_euler_midpoint_time(
                _peak_session.model.energy_to_image_flow,
                _ablate.unsqueeze(1),
                _e2i_steps,
            )
            _src_spatial = hilbert_unflatten_images(_seq[:_peak_n_show], side=_side)[:, 0]
            _full_spatial = hilbert_unflatten_images(_full_img, side=_side)[:, 0]
            _only_spatial = hilbert_unflatten_images(_only_img, side=_side)[:, 0]
            _ablate_spatial = hilbert_unflatten_images(_ablate_img, side=_side)[:, 0]

        _e0_max_abs = float(_e0.abs().max().clamp_min(1e-12))
        _dist_bits = " ".join(
            f"p{int(100 * float(q))}={float(v):.3f}"
            for q, v in zip(_qs.tolist(), _qvals.tolist(), strict=True)
        )
        _summary = mo.md(
            f"""
    ### e[0] peak probe (Hilbert / spatial corner)

    ckpt step=`{_payload.get("step")}` · n=`{_peak_n_dist}` encoded energies

    - mean=`{float(_e0.mean()):.4f}` std=`{float(_e0.std()):.4f}`
    - quantiles: `{_dist_bits}`
    - frac argmin@0=`{float((_encoded.argmin(dim=1) == 0).float().mean()):.2f}`

    Rows below: **only e[0]** (others zero) decoded by e2i across the e[0] distribution,
    then per-sample source / full energy / only-peak / ablate-peak.
    """
        )

        _sweep_panels = []
        for _i, _v in enumerate(_sweep_vals.tolist()):
            _energy_vis = torch.zeros(_side * _side)
            _energy_vis[0] = float(_v)
            _sweep_panels.append(
                "<div style='display:flex;gap:8px;align-items:flex-start;'>"
                + _signed_png_img(
                    _energy_vis,
                    size=_side,
                    max_abs=_e0_max_abs,
                    display_px=_peak_display,
                    label=f"only e0={float(_v):.3f}",
                )
                + _grayscale_png_img(
                    _sweep_spatial[_i].detach().cpu(),
                    size=_side,
                    display_px=_peak_display,
                    label="e2i image",
                )
                + "</div>"
            )

        _sample_rows = []
        for _i in range(_peak_n_show):
            _e0_i = float(_show[_i, 0])
            _energy_only = torch.zeros(_side * _side)
            _energy_only[0] = _e0_i
            _sample_rows.append(
                f"<div style='display:grid;gap:6px;'><div style='font-weight:700;'>"
                f"sample {_i} · e0={_e0_i:.3f}</div>"
                "<div style='display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start;'>"
                + _grayscale_png_img(
                    _src_spatial[_i].detach().cpu(),
                    size=_side,
                    display_px=_peak_display,
                    label="source",
                )
                + _grayscale_png_img(
                    _full_spatial[_i].detach().cpu(),
                    size=_side,
                    display_px=_peak_display,
                    label="e2i full e",
                )
                + _signed_png_img(
                    _energy_only,
                    size=_side,
                    max_abs=_e0_max_abs,
                    display_px=_peak_display,
                    label="only e0 energy",
                )
                + _grayscale_png_img(
                    _only_spatial[_i].detach().cpu(),
                    size=_side,
                    display_px=_peak_display,
                    label="e2i only e0",
                )
                + _grayscale_png_img(
                    _ablate_spatial[_i].detach().cpu(),
                    size=_side,
                    display_px=_peak_display,
                    label="e2i e0=0",
                )
                + "</div></div>"
            )

        peak_probe = mo.vstack(
            [
                _summary,
                mo.md("#### Sweep: only-peak energy → e2i (others zero)"),
                mo.Html(
                    "<div style='display:grid;gap:12px;color:#d7dae0;font:13px/1.4 system-ui,sans-serif;'>"
                    + "".join(_sweep_panels)
                    + "</div>"
                ),
                mo.md("#### Per-sample: source · full · only-peak · ablate-peak"),
                mo.Html(
                    "<div style='display:grid;gap:16px;color:#d7dae0;font:13px/1.4 system-ui,sans-serif;'>"
                    + "".join(_sample_rows)
                    + "</div>"
                ),
            ]
        )
    peak_probe

    return (peak_probe,)


if __name__ == "__main__":
    app.run()
