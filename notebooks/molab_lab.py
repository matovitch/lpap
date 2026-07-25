import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import subprocess
    import sys
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo
    import torch

    install_spec = (
        "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
    )
    try:
        import lpap  # noqa: F401
        install_note = "lpap already importable"
    except ImportError:
        lpap_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            install_spec,
        ]
        deps_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "jaxtyping>=0.3.7",
        ]
        subprocess.check_call(lpap_cmd)
        subprocess.check_call(deps_cmd)
        install_note = (
            f"installed via: {' '.join(lpap_cmd)} ; {' '.join(deps_cmd)}"
        )

    # Molab sandbox cwd is typically /marimo; local checkout uses repo root.
    if Path("/marimo").is_dir() and Path("/marimo/notebook.py").exists():
        project_root = Path("/marimo")
    else:
        project_root = Path(__file__).resolve().parents[1]
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

    (project_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (project_root / "training_logs").mkdir(parents=True, exist_ok=True)

    from lpap.training_log import load_best_metric_row, load_metric_history
    from lpap.training_notebook import (
        create_training_session,
        iter_training,
        recent_training_runs,
        render_recent_runs_table,
        training_config_from_project_file,
        training_config_path,
        validation_regularizer_metric_names,
    )
    from lpap.training_plots import render_loss_history_svg

    return (
        create_training_session,
        install_note,
        install_spec,
        iter_training,
        load_best_metric_row,
        load_metric_history,
        mo,
        project_root,
        recent_training_runs,
        render_loss_history_svg,
        render_recent_runs_table,
        replace,
        torch,
        training_config_from_project_file,
        training_config_path,
        validation_regularizer_metric_names,
    )


@app.cell
def _(install_note, mo, project_root, torch):
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
    mo.md(
        f"""
# LPAP molab lab

Single durable notebook for remote GPU work (see `doc/molab-workflow.md`).

- install: `{install_note}`
- project root: `{project_root}`
- torch: `{torch.__version__}`
- CUDA: **{cuda_ok}** (`{device_name}`)

Prefer **one** marimo-pair session. Train in **chunks** so the agent can poll
SQLite between runs without interrupting a long cell.
"""
    )
    return cuda_ok, device_name


@app.cell
def _(mo):
    model_kind_picker = mo.ui.dropdown(
        options=[
            "surrogate",
            "decoder",
            "image_to_energy",
            "image_to_energy_energy_bank",
            "energy_to_image",
            "energy_to_image_energy_bank",
            "image_autoencoder",
        ],
        value="image_to_energy_energy_bank",
        label="Model kind",
    )
    target_steps = mo.ui.number(
        start=1, stop=1_000_000, step=100, value=2_000, label="Target steps"
    )
    chunk_steps = mo.ui.number(
        start=1, stop=1_000_000, step=100, value=500, label="Chunk steps"
    )
    display_every = mo.ui.number(
        start=1, stop=100_000, step=1, value=50, label="display_every"
    )
    log_every = mo.ui.number(
        start=1, stop=100_000, step=1, value=10, label="log_every"
    )
    run_training = mo.ui.checkbox(value=True, label="Run training chunk")
    mo.vstack(
        [
            model_kind_picker,
            mo.hstack([target_steps, chunk_steps]),
            mo.hstack([display_every, log_every]),
            run_training,
        ]
    )
    return (
        chunk_steps,
        display_every,
        log_every,
        model_kind_picker,
        run_training,
        target_steps,
    )


@app.cell
def _(
    display_every,
    log_every,
    model_kind_picker,
    mo,
    project_root,
    recent_training_runs,
    render_recent_runs_table,
    replace,
    target_steps,
    training_config_from_project_file,
    training_config_path,
):
    model_kind = model_kind_picker.value
    config_file = training_config_path(project_root, model_kind)
    base_config = training_config_from_project_file(project_root, model_kind)
    # Apply molab-friendly run knobs; steps are finalized per chunk below.
    config = replace(
        base_config,
        run=replace(
            base_config.run,
            steps=int(target_steps.value),
            display_every=int(display_every.value),
            log_every=int(log_every.value),
            resume_from_checkpoint=True,
            note="molab lab",
            tags=tuple(
                dict.fromkeys((*base_config.run.tags, "molab", "lab"))
            ),
        ),
    )
    recent_runs = recent_training_runs(project_root, config, limit=10)
    source = (
        f"TOML `{config_file}`"
        if config_file.exists()
        else f"defaults (`{type(base_config).__name__}`)"
    )
    mo.vstack(
        [
            mo.md(
                f"**model**: `{model_kind}`  \n"
                f"**config source**: {source}  \n"
                f"**target steps**: `{config.run.steps}`"
            ),
            mo.Html(render_recent_runs_table(recent_runs)),
        ]
    )
    return base_config, config, config_file, model_kind, recent_runs, source


@app.cell
def _(
    chunk_steps,
    config,
    create_training_session,
    cuda_ok,
    iter_training,
    load_best_metric_row,
    load_metric_history,
    mo,
    model_kind,
    project_root,
    render_loss_history_svg,
    replace,
    run_training,
    target_steps,
    torch,
    validation_regularizer_metric_names,
):
    def loss_history_plot(session):
        regularizer_metrics = validation_regularizer_metric_names(session.config)
        rows = load_metric_history(
            session.log_path,
            run_id=session.resume_info.run_id,
            metric_names=("loss", "validation_loss", *regularizer_metrics),
        )
        return mo.Html(
            render_loss_history_svg(
                rows, validation_regularizer_metrics=regularizer_metrics
            )
        )

    def best_checkpoint_weighted_accuracy(session):
        row = load_best_metric_row(
            session.log_path,
            run_id=session.resume_info.run_id,
            metric_name="validation_loss",
        )
        if row is None or row.get("validation_weighted_accuracy") is None:
            return "n/a"
        return f"{row['validation_weighted_accuracy']:.4f}"

    def render_training_output(*, rows, session, best_metric, message):
        best_metric_label = "n/a" if best_metric is None else f"{best_metric:.4f}"
        best_weighted_accuracy = best_checkpoint_weighted_accuracy(session)

        def metric_cell(row, name):
            value = row.get(name)
            return "" if value is None else f"{value:.4f}"

        try:
            ckpt_label = str(session.checkpoint_path.relative_to(project_root))
            log_label = str(session.log_path.relative_to(project_root))
        except ValueError:
            ckpt_label = str(session.checkpoint_path)
            log_label = str(session.log_path)

        panels = [
            mo.md(
                f"""
                **name**: `{session.resume_info.display_name}`  
                **experiment**: `{session.resume_info.base_run_id}`  
                **run instance**: `{session.resume_info.run_id}`  
                **checkpoint**: `{ckpt_label}`  
                **log**: `{log_label}`  
                **device**: `{session.device}` (cuda_ok={cuda_ok})  
                **best validation loss**: `{best_metric_label}`  
                **best checkpoint validation weighted accuracy**: `{best_weighted_accuracy}`
                {message}
                """
            ),
            loss_history_plot(session),
            mo.Html(
                """
                <table>
                  <thead><tr><th>step</th><th>train loss</th><th>validation loss</th><th>train weighted accuracy</th><th>validation weighted accuracy</th><th>best</th></tr></thead>
                  <tbody>
                """
                + "".join(
                    f"<tr><td>{row['step']}</td>"
                    f"<td>{metric_cell(row, 'loss')}</td>"
                    f"<td>{metric_cell(row, 'validation_loss')}</td>"
                    f"<td>{metric_cell(row, 'weighted_accuracy')}</td>"
                    f"<td>{metric_cell(row, 'validation_weighted_accuracy')}</td>"
                    f"<td>{'yes' if row['best'] else ''}</td></tr>"
                    for row in rows[-12:]
                )
                + "</tbody></table>"
            ),
        ]
        return mo.vstack(panels)

    if not run_training.value:
        output = mo.md("Enable **Run training chunk** to train.")
    else:
        try:
            target = int(target_steps.value)
            chunk = int(chunk_steps.value)
            ckpt_path = project_root / "checkpoints" / config.run.checkpoint_name
            if config.run.resume_from_checkpoint and ckpt_path.exists():
                _payload = torch.load(
                    ckpt_path, map_location="cpu", weights_only=True
                )
                start_step = int(_payload["step"]) + 1
            else:
                start_step = 1
            chunk_end = min(start_step + chunk - 1, target)
            chunk_config = replace(
                config, run=replace(config.run, steps=chunk_end)
            )
            session = create_training_session(
                model_kind, project_root=project_root, config=chunk_config
            )
        except (FileNotFoundError, ValueError, TypeError, KeyError, OSError) as error:
            output = mo.md(f"**Setup error:** {error}")
        else:
            history = []
            mo.output.replace(
                mo.md(
                    f"Chunk `{model_kind}` on `{session.device}`: "
                    f"steps `{session.resume_info.start_step}`…`{chunk_end}` "
                    f"(target `{target}`); {session.resume_info.message}."
                )
            )
            step_range = range(session.resume_info.start_step, chunk_end + 1)
            progress = mo.status.progress_bar(
                step_range,
                title=f"Training LPAP {model_kind} (chunk)",
                total=max(len(step_range), 1),
            )
            events = iter_training(model_kind, session)
            for _step_index, result in zip(progress, events, strict=False):
                history.append(
                    {"step": result.step, "best": result.improved, **result.metrics}
                )
                if result.should_display:
                    mo.output.replace(
                        render_training_output(
                            rows=history,
                            session=session,
                            best_metric=result.best_metric,
                            message=(
                                f"Step `{result.step}` / chunk end `{chunk_end}` "
                                f"(target `{target}`)."
                            ),
                        )
                    )

            if history:
                final = history[-1]
                remaining = max(0, target - final["step"])
                output = render_training_output(
                    rows=history,
                    session=session,
                    best_metric=session.training_run.best_metric,
                    message=(
                        f"Chunk finished at step `{final['step']}` "
                        f"(train loss `{final['loss']:.4f}`). "
                        f"Remaining to target: `{remaining}`. "
                        "Re-run this cell for the next chunk; agents should poll "
                        "SQLite between chunks, not during them."
                    ),
                )
            else:
                output = mo.md(
                    f"Already at/past chunk end. "
                    f"start=`{session.resume_info.start_step}`, "
                    f"chunk_end=`{chunk_end}`, target=`{target}`."
                )

    output
    return


if __name__ == "__main__":
    app.run()
