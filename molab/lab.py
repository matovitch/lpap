import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def ae_status():
    # cell: ae_status
    from pathlib import Path as _Path
    import json as _json
    import sqlite3 as _sqlite3

    import marimo as _mo
    import matplotlib.pyplot as plt
    import numpy as np

    from molab.bg_worker import (
        bg_log_tail as _bg_log_tail,
        bg_worker_status as _bg_worker_status,
        last_bg_log_step as _last_bg_log_step,
    )

    _root = _Path("/marimo")
    _run_id_prefix = "image_autoencoder_tri_lnorm"
    _ckpt_name = "image_autoencoder_tri_lnorm.pt"
    _log_name = "image_autoencoder_tri_lnorm.sqlite"
    _bg_stem = "image_autoencoder_tri_lnorm_bg"

    _needed = {
        "flow": _root / "checkpoints" / "image_energy_flow.pt",
        "c128_s": _root / "checkpoints" / "surrogate_c128_k16.pt",
        "c128_d": _root / "checkpoints" / "decoder_c128_k16.pt",
        "c256_s": _root / "checkpoints" / "surrogate_c256_k24.pt",
        "c256_d": _root / "checkpoints" / "decoder_c256_k24.pt",
        "c512_s": _root / "checkpoints" / "surrogate_c512_k32.pt",
        "c512_d": _root / "checkpoints" / "decoder_c512_k32.pt",
        "ae": _root / "checkpoints" / _ckpt_name,
    }
    _ckpt_lines = []
    for _label, _path in _needed.items():
        if not _path.is_file():
            _ckpt_lines.append(f"- `{_label}`: **missing**")
            continue
        import torch as _torch
        _payload = _torch.load(_path, map_location="cpu", weights_only=False)
        _ckpt_lines.append(
            f"- `{_label}`: step={_payload.get('step')} best={_payload.get('best_metric')}"
        )

    _bg = _bg_worker_status(_root / "training_logs" / f"{_bg_stem}.pid")
    _log_step = _last_bg_log_step(_root / "training_logs" / f"{_bg_stem}.log")
    _tail = _bg_log_tail(_root / "training_logs" / f"{_bg_stem}.log", lines=10)

    _sqlite = _root / "training_logs" / _log_name
    _fig = None
    _summary = "(no AE sqlite yet — re-run after first logged steps)"
    if _sqlite.is_file():
        _con = _sqlite3.connect(_sqlite)
        _runs = _con.execute(
            "select run_id, config_json, updated_at, status from runs order by updated_at desc"
        ).fetchall()
        _best = None
        for _rid, _cfg_json, _up, _status in _runs:
            _mx = _con.execute(
                "select max(step) from step_metrics where run_id=?", (_rid,)
            ).fetchone()[0]
            if _mx:
                _best = (_mx, _rid, _cfg_json, _status)
                break
        if _best is not None:
            _max_step, _run_id, _cfg_json, _status = _best
            _loss_cfg = (_json.loads(_cfg_json).get("loss") or {})
            _w_img = float(_loss_cfg.get("image_l2_weight", 1.0))
            _w_e = float(_loss_cfg.get("energy_l1_weight", 0.5))
            _w_ce = float(_loss_cfg.get("surrogate_teacher_weight", 0.05))
            _w_sm = float(_loss_cfg.get("signed_mass_balance_weight", 0.02))
            _tau = float(_loss_cfg.get("signed_mass_floor_tau", 0.01))

            def _series(_name: str) -> dict[int, float]:
                return dict(
                    _con.execute(
                        "select step, metric_value from step_metrics "
                        "where run_id=? and metric_name=? order by step",
                        (_run_id, _name),
                    )
                )

            _s_loss = _series("loss")
            _s_vloss = _series("validation_loss")
            _s_img = _series("image_reconstruction_l2")
            _s_e = _series("energy_reconstruction_l1")
            _s_ce = _series("surrogate_teacher_ce")
            _s_sm = _series("signed_mass_balance")
            _s_mp = _series("encoded_positive_mass")
            _s_mn = _series("encoded_negative_mass")
            _s_rms = _series("encoded_energy_rms")
            _common = sorted(
                set(_s_loss) & set(_s_img) & set(_s_e) & set(_s_ce) & set(_s_sm)
            )
            if _common:
                _steps = np.array(_common)
                def _at(_d: dict[int, float]) -> np.ndarray:
                    return np.array([_d[_s] for _s in _common], dtype=float)

                _loss = _at(_s_loss)
                _img_w = _w_img * _at(_s_img)
                _e_w = _w_e * _at(_s_e)
                _ce_w = _w_ce * _at(_s_ce)
                _sm_w = _w_sm * _at(_s_sm)
                _eps = 1e-12
                # Stack bottom→top: image L2, CE, energy L1, signed-mass
                _shares = {
                    "image L2": _img_w / (_loss + _eps),
                    "surrogate CE": _ce_w / (_loss + _eps),
                    "energy L1": _e_w / (_loss + _eps),
                    "signed-mass": _sm_w / (_loss + _eps),
                }
                _v_steps = sorted(_s_vloss)
                _late = set(_steps[-min(50, len(_steps)) :])
                _idx_late = [i for i, s in enumerate(_steps) if s in _late]
                _summary = (
                    f"run `{_run_id}` ({_status}) · sqlite_max=`{_max_step}` · "
                    f"λ img={_w_img:g} e={_w_e:g} ce={_w_ce:g} sm={_w_sm:g} τ={_tau:g}\n"
                    f"late shares: "
                    f"img={np.mean([_shares['image L2'][i] for i in _idx_late]):.3f} "
                    f"e={np.mean([_shares['energy L1'][i] for i in _idx_late]):.3f} "
                    f"ce={np.mean([_shares['surrogate CE'][i] for i in _idx_late]):.3f} "
                    f"sm={np.mean([_shares['signed-mass'][i] for i in _idx_late]):.3f}"
                )
                if _s_mp and _s_rms:
                    _m_late = [s for s in _late if s in _s_mp and s in _s_mn and s in _s_rms]
                    if _m_late:
                        _summary += (
                            f"\nm±/rms late: m+={np.mean([_s_mp[s] for s in _m_late]):.4f} "
                            f"m-={np.mean([_s_mn[s] for s in _m_late]):.4f} "
                            f"rms={np.mean([_s_rms[s] for s in _m_late]):.4f}"
                        )

                # Matplotlib tab10 defaults (original look), keyed for line/stack sync.
                _colors = {
                    "image L2": "#1f77b4",
                    "energy L1": "#ff7f0e",
                    "surrogate CE": "#2ca02c",
                    "signed-mass": "#d62728",
                }
                _fig, _axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
                _ax = _axes[0]
                _ax.plot(_steps, _loss, label="train loss", lw=1, color="#92400e")
                if _v_steps:
                    _ax.plot(
                        _v_steps,
                        [_s_vloss[s] for s in _v_steps],
                        label="val loss",
                        lw=1.5,
                        color="#9467bd",
                    )
                _ax.set_yscale("log")
                _ax.set_ylabel("loss")
                _ax.legend(fontsize=8)
                _ax.grid(True, alpha=0.3)
                _ax.set_title("AE loss")

                _ax = _axes[1]
                _ax.plot(
                    _steps,
                    _img_w,
                    label=f"λ·img ({_w_img:g})",
                    lw=1.2,
                    color=_colors["image L2"],
                )
                _ax.plot(
                    _steps,
                    _ce_w,
                    label=f"λ·CE ({_w_ce:g})",
                    lw=1.2,
                    color=_colors["surrogate CE"],
                )
                _ax.plot(
                    _steps,
                    _e_w,
                    label=f"λ·energy ({_w_e:g})",
                    lw=1.2,
                    color=_colors["energy L1"],
                )
                _ax.plot(
                    _steps,
                    _sm_w,
                    label=f"λ·signed ({_w_sm:g})",
                    lw=1.2,
                    color=_colors["signed-mass"],
                )
                _ax.set_yscale("log")
                _ax.set_ylabel("weighted term")
                _ax.legend(fontsize=8)
                _ax.grid(True, alpha=0.3)
                _ax.set_title("regularizer / recon contributions")

                _ax = _axes[2]
                _ax.stackplot(
                    _steps,
                    *[_shares[k] for k in _shares],
                    labels=list(_shares),
                    colors=[_colors[k] for k in _shares],
                    alpha=0.85,
                )
                _ax.set_ylim(0, 1)
                _ax.set_ylabel("share of loss")
                _ax.set_xlabel("step")
                _ax.legend(fontsize=8, loc="upper right")
                _ax.grid(True, alpha=0.3)
                _ax.set_title("term share of total loss")
                _fig.tight_layout()
        _con.close()

    _blocks = [
        _mo.md(
            f"""### AE status · `{_run_id_prefix}` → 70k (fresh sqlite/ckpt)

    {chr(10).join(_ckpt_lines)}

    - bg alive: **{_bg.get("alive")}** (pid=`{_bg.get("pid")}`) · log step: `{_log_step}`
    ```
    {chr(10).join(_tail) if _tail else "(no AE bg log yet)"}
    ```

    ```
    {_summary}
    ```
    """
        )
    ]
    if _fig is not None:
        _blocks.append(_fig)
    ae_status = _mo.vstack(_blocks)
    ae_status
    return


@app.cell
def ae_gallery():
    # cell: ae_gallery
    from pathlib import Path as _Path
    import shutil as _shutil
    from dataclasses import replace as _replace

    import marimo as _mo

    from lpap.artifact_sync import ensure_checkpoint as _ensure_checkpoint
    from lpap.checkpoints import load_training_checkpoint as _load_ckpt
    from lpap.image_autoencoder_training import (
        ImageAutoencoderLpapPairConfig as _Pair,
        ImageAutoencoderRunConfig as _Run,
        ImageAutoencoderSourceConfig as _Source,
        collect_image_autoencoder_gallery as _collect_gallery,
        create_image_autoencoder_training_session as _create_ae_session,
    )
    from lpap.training_notebook import default_image_autoencoder_training_config as _default_ae
    from lpap.training_plots import render_image_autoencoder_gallery_html as _render_ae

    _root = _Path("/marimo")
    _ckpt_name = "image_autoencoder_tri_lnorm.pt"
    _gamma = 1.0
    _ensure_checkpoint(_root, "image_energy_flow.pt")
    for _name in (
        "surrogate_c128_k16.pt",
        "decoder_c128_k16.pt",
        "surrogate_c256_k24.pt",
        "decoder_c256_k24.pt",
        "surrogate_c512_k32.pt",
        "decoder_c512_k32.pt",
    ):
        _ensure_checkpoint(_root, _name)

    _base = _default_ae()
    _ae_cfg = _replace(
        _base,
        source=_Source(
            lpap_pairs=(
                _Pair(
                    surrogate_checkpoint_name="surrogate_c128_k16.pt",
                    decoder_checkpoint_name="decoder_c128_k16.pt",
                    name="c128_k16",
                ),
                _Pair(
                    surrogate_checkpoint_name="surrogate_c256_k24.pt",
                    decoder_checkpoint_name="decoder_c256_k24.pt",
                    name="c256_k24",
                ),
                _Pair(
                    surrogate_checkpoint_name="surrogate_c512_k32.pt",
                    decoder_checkpoint_name="decoder_c512_k32.pt",
                    name="c512_k32",
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
        loss=_replace(_base.loss, signed_mass_floor_tau=0.1, signed_mass_balance_weight=1.0e-2, energy_l1_weight=0.05),
        run=_Run(
            run_training=False,
            resume_from_checkpoint=False,
            steps=70_000,
            seed=_base.run.seed,
            display_every=25,
            log_every=5,
            run_id="image_autoencoder_tri_lnorm",
            checkpoint_name=_ckpt_name,
            log_name="image_autoencoder_tri_lnorm.sqlite",
            comment="gallery",
            pinned=False,
        ),
    )

    _live = _root / "checkpoints" / _ckpt_name
    _snap = _root / "checkpoints" / "_gallery_snapshot_tri_lnorm.pt"
    if _live.is_file():
        try:
            _shutil.copy2(_live, _snap)
            _ckpt = _snap
        except Exception:
            _ckpt = _live
    else:
        _ckpt = _live

    if not _ckpt.is_file():
        ae_gallery = _mo.md(
            f"### AE gallery · waiting for `{_ckpt_name}` (re-run after first ckpt)"
        )
    else:
        _session = _create_ae_session(
            project_root=_root, config=_ae_cfg, device="cpu"
        )
        _payload = _load_ckpt(_ckpt, map_location=_session.device)
        _state = _payload.get("best_model_state") or _payload["model_state"]
        _session.model.load_state_dict(_state)
        _items = _collect_gallery(_session, sample_count=6)
        _html = _render_ae(
            _items,
            size=int(_ae_cfg.image.side),
            display_px=121,
            gamma=_gamma,
        )
        ae_gallery = _mo.vstack(
            [
                _mo.md(
                    f"### AE gallery (6 samples) · step=`{_payload.get('step')}` "
                    f"best=`{_payload.get('best_metric')}` · `{_ckpt_name}` · "
                    f"γ=`{_gamma:g}` · cpu"
                ),
                _mo.Html(_html),
            ]
        )
    ae_gallery
    return


@app.cell(hide_code=True)
def ae_energy_latent():
    # cell: ae_energy_latent
    """Energy/latent path gallery: source → oracle LPAP → surrogate hard → decoder soft.

    AE gallery "{pair} energy" is decoder-soft only. This cell decomposes the LPAP
    bottleneck so we can see whether failures are projection (oracle), assignment
    (surrogate), or soft reconstruction (decoder).
    """
    from pathlib import Path as _Path
    import shutil as _shutil
    from dataclasses import replace as _replace

    import marimo as _mo
    import torch as _torch

    from lpap.artifact_sync import ensure_checkpoint as _ensure_checkpoint
    from lpap.checkpoints import load_training_checkpoint as _load_ckpt
    from lpap.decoder import (
        prepare_lpap_decoder_batch as _prep_dec,
        reconstruct_lpap_bucket_values as _oracle_lpap,
        reconstruct_lpap_decoder_values as _decoder_soft,
    )
    from lpap.flow_training import (
        cycle_image_batches as _cycle_images,
        prepare_image_sequence as _prep_image,
    )
    from lpap.hilbert import hilbert_unflatten_images as _unflat
    from lpap.image_autoencoder_training import (
        ImageAutoencoderLpapPairConfig as _Pair,
        ImageAutoencoderRunConfig as _Run,
        ImageAutoencoderSourceConfig as _Source,
        create_image_autoencoder_training_session as _create_ae_session,
        _forward_loss as _fwd,
    )
    from lpap.surrogate import prepare_lpap_surrogate_batch as _prep_surr
    from lpap.training_notebook import default_image_autoencoder_training_config as _default_ae
    from lpap.training_plots import render_signed_triplet_gallery_html as _render_triplet

    _root = _Path("/marimo")
    _ckpt_name = "image_autoencoder_tri_lnorm.pt"
    _gamma = 1.0
    _sample_count = 6
    _focus = 0  # 0-based; sample 1 in the UI

    _ensure_checkpoint(_root, "image_energy_flow.pt")
    for _name in (
        "surrogate_c128_k16.pt",
        "decoder_c128_k16.pt",
        "surrogate_c256_k24.pt",
        "decoder_c256_k24.pt",
        "surrogate_c512_k32.pt",
        "decoder_c512_k32.pt",
    ):
        _ensure_checkpoint(_root, _name)

    _base = _default_ae()
    _ae_cfg = _replace(
        _base,
        source=_Source(
            lpap_pairs=(
                _Pair(
                    surrogate_checkpoint_name="surrogate_c128_k16.pt",
                    decoder_checkpoint_name="decoder_c128_k16.pt",
                    name="c128_k16",
                ),
                _Pair(
                    surrogate_checkpoint_name="surrogate_c256_k24.pt",
                    decoder_checkpoint_name="decoder_c256_k24.pt",
                    name="c256_k24",
                ),
                _Pair(
                    surrogate_checkpoint_name="surrogate_c512_k32.pt",
                    decoder_checkpoint_name="decoder_c512_k32.pt",
                    name="c512_k32",
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
        loss=_replace(
            _base.loss,
            signed_mass_floor_tau=0.1,
            signed_mass_balance_weight=1.0e-2,
            energy_l1_weight=0.05,
        ),
        run=_Run(
            run_training=False,
            resume_from_checkpoint=False,
            steps=70_000,
            seed=_base.run.seed,
            display_every=25,
            log_every=5,
            run_id="image_autoencoder_tri_lnorm",
            checkpoint_name=_ckpt_name,
            log_name="image_autoencoder_tri_lnorm.sqlite",
            comment="energy latent gallery",
            pinned=False,
        ),
    )

    _live = _root / "checkpoints" / _ckpt_name
    _snap = _root / "checkpoints" / "_gallery_snapshot_tri_lnorm.pt"
    if _live.is_file():
        try:
            _shutil.copy2(_live, _snap)
            _ckpt = _snap
        except Exception:
            _ckpt = _live
    else:
        _ckpt = _live

    if not _ckpt.is_file():
        ae_energy_latent = _mo.md(f"### AE energy/latent · waiting for `{_ckpt_name}`")
    else:
        _session = _create_ae_session(project_root=_root, config=_ae_cfg, device="cpu")
        _payload = _load_ckpt(_ckpt, map_location=_session.device)
        _session.model.load_state_dict(
            _payload.get("best_model_state") or _payload["model_state"]
        )
        _session.model.eval()
        _side = int(_ae_cfg.image.side)
        _images = next(_cycle_images(_session.validation_image_loader))[:_sample_count]
        with _torch.no_grad():
            _image = _prep_image(_images, side=_side, device=_session.device)
            _loss, _metrics, _out = _fwd(session=_session, image=_image)
            _values = _out.encoded_energy[:, 0]  # Hilbert-ordered

        def _spatial(flat: _torch.Tensor) -> _torch.Tensor:
            return _unflat(flat.detach().cpu().unsqueeze(0).unsqueeze(0), side=_side)[0, 0]

        def _stats(pred: _torch.Tensor, target: _torch.Tensor) -> str:
            err = pred - target
            corr = float(
                _torch.corrcoef(_torch.stack([target.reshape(-1), pred.reshape(-1)]))[0, 1]
            )
            return (
                f"L1={float(err.abs().mean()):.4f} "
                f"RMSE={float(err.pow(2).mean().sqrt()):.4f} "
                f"corr={corr:.4f}"
            )

        # Prefer higher-C first (matches AE gallery ordering).
        _pair_order = sorted(
            range(len(_session.lpap_pairs)),
            key=lambda i: -int(
                _session.lpap_pairs[i].surrogate_model_config["bucket_count"]
            ),
        )

        _blocks: list = [
            _mo.md(
                f"### AE energy / latent · step=`{_payload.get('step')}` · "
                f"`{_ckpt_name}` · γ=`{_gamma:g}`\n\n"
                "Pipeline for each pair (same encoded energy from image→energy flow):\n"
                "- **source energy** — flow encode (Hilbert → spatial for display)\n"
                "- **oracle LPAP** — true bucket masses at oracle source indices "
                "(LPAP projection ceiling; no learned assignment)\n"
                "- **surrogate hard** — true bucket masses at **surrogate argmax** "
                "locations (assignment error only)\n"
                "- **decoder soft** — softmax decoder reconstruction = AE "
                "`{pair} energy` (what trains / feeds energy→image)\n"
            )
        ]

        # Focus sample first, then remaining.
        _indices = [_focus] + [i for i in range(_sample_count) if i != _focus]
        for _si in _indices:
            _v = _values[_si : _si + 1]
            _src = _spatial(_v[0])
            _hdr = f"#### sample {_si + 1}" + (
                " · **focus**" if _si == _focus else ""
            )
            _blocks.append(_mo.md(_hdr))
            for _pi in _pair_order:
                _rt = _session.lpap_pairs[_pi]
                _C = int(_rt.surrogate_model_config["bucket_count"])
                _k = int(_rt.surrogate_model_config["k_max"])
                _surr = _session.model.surrogates[_pi]
                _dec = _session.model.decoders[_pi]
                _tok = _prep_surr(
                    _v, bucket_count=_C, permutation=_rt.permutation
                )
                _slog = _surr(_tok)
                _db = _prep_dec(
                    values=_v,
                    surrogate_logits=_slog,
                    bucket_count=_C,
                    k_max=_k,
                    temperature=_dec.frontend_temperature(),
                    permutation=_rt.permutation,
                )
                _dlog = _dec(_db.tokens)
                _oracle = _oracle_lpap(_db)[0]
                _hard = _torch.zeros_like(_v[0]).scatter_add(
                    0,
                    _slog.argmax(dim=-1)[0],
                    _db.surrogate_targets.buckets[0].to(dtype=_v.dtype),
                )
                _soft = _decoder_soft(_dlog, _db)[0]
                _surr_acc = float((_slog.argmax(-1) == _db.targets).float().mean())
                _dec_acc = float((_dlog.argmax(-1) == _db.targets).float().mean())
                _Item = type(
                    "LatentItem",
                    (),
                    {
                        "energy": _src,
                        "lpap": _spatial(_oracle),
                        "surrogate_hard": _spatial(_hard),
                        "decoder": _spatial(_soft),
                    },
                )
                _html = _render_triplet(
                    [_Item()],
                    size=_side,
                    display_px=121,
                    gamma=_gamma,
                )
                _blocks.append(
                    _mo.md(
                        f"**{_rt.name}** · surr_acc=`{_surr_acc:.3f}` · "
                        f"dec_acc=`{_dec_acc:.3f}` · T=`{float(_dec.frontend_temperature()):.3f}`  \n"
                        f"oracle `{_stats(_oracle, _v[0])}` · "
                        f"hard `{_stats(_hard, _v[0])}` · "
                        f"soft `{_stats(_soft, _v[0])}`"
                    )
                )
                _blocks.append(_mo.Html(_html))

        ae_energy_latent = _mo.vstack(_blocks)
    ae_energy_latent
    return


if __name__ == "__main__":
    app.run()
