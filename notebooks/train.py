import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from lpap.training_log import load_best_metric_row, load_metric_history
    from lpap.training_notebook import (
        create_training_session,
        iter_training,
        recent_training_runs,
        render_recent_runs_table,
        restore_training_config_from_log,
        training_config_from_project_file,
        training_config_path,
        validation_regularizer_metric_names,
    )
    from lpap.training_plots import render_loss_history_svg

    return (
        create_training_session,
        iter_training,
        load_best_metric_row,
        load_metric_history,
        mo,
        project_root,
        recent_training_runs,
        render_loss_history_svg,
        render_recent_runs_table,
        restore_training_config_from_log,
        training_config_from_project_file,
        training_config_path,
        validation_regularizer_metric_names,
    )


@app.cell
def _(mo):
    mo.md("""
    # LPAP training
    """)
    return


@app.cell
def _(mo):
    model_kind_picker = mo.ui.dropdown(
        options=[
            "surrogate",
            "decoder",
            "image_to_energy",
            "energy_to_image",
            "energy_to_image_reflow",
            "image_autoencoder",
        ],
        value="surrogate",
        label="Model kind",
    )
    model_kind_picker
    return (model_kind_picker,)


@app.cell
def _(
    model_kind_picker,
    mo,
    project_root,
    recent_training_runs,
    training_config_from_project_file,
    training_config_path,
):
    model_kind = model_kind_picker.value
    config_file = training_config_path(project_root, model_kind)
    base_config = training_config_from_project_file(project_root, model_kind)
    recent_runs = recent_training_runs(project_root, base_config, limit=10)
    run_options = {
        f"{row['display_name']} | {row['status']} | step {row['last_step'] or 0}": row[
            "run_id"
        ]
        for row in recent_runs
    }
    if run_options:
        restore_run_picker = mo.ui.dropdown(
            options=run_options,
            value=next(iter(run_options)),
            label="Restore config from run",
        )
    else:
        restore_run_picker = mo.ui.dropdown(
            options={"No previous runs": ""},
            value="No previous runs",
            label="Restore config from run",
            disabled=True,
        )
    restore_button = mo.ui.run_button(
        label="Restore selected run to TOML", disabled=not run_options
    )
    mo.vstack(
        [
            mo.hstack([restore_run_picker, restore_button]),
            mo.md("Edit the TOML file directly, or restore one previous run into it."),
        ]
    )
    return config_file, model_kind, recent_runs, restore_button, restore_run_picker


@app.cell
def _(
    model_kind,
    project_root,
    restore_button,
    restore_run_picker,
    restore_training_config_from_log,
):
    restored_config_file = None
    if restore_button.value and restore_run_picker.value:
        restored_config_file = restore_training_config_from_log(
            model_kind,
            project_root=project_root,
            run_id=restore_run_picker.value,
        )

    restore_message = (
        f"Restored `{restore_run_picker.value}` to `{restored_config_file}`."
        if restored_config_file is not None
        else "Using the current TOML config."
    )
    return (restore_message,)


@app.cell
def _(
    model_kind,
    project_root,
    recent_runs,
    render_recent_runs_table,
    restore_message,
    training_config_from_project_file,
):
    config = training_config_from_project_file(project_root, model_kind)
    recent_runs_table = render_recent_runs_table(recent_runs)
    return config, recent_runs_table


@app.cell
def _(config_file, mo, model_kind, recent_runs_table, restore_message):
    mo.vstack(
        [
            mo.md(
                f"**model**: `{model_kind}`  \n**config file**: `{config_file}`  \n**source**: `config file`  \n{restore_message}"
            ),
            mo.Html(recent_runs_table),
        ]
    )
    return


@app.cell
def _(
    config,
    create_training_session,
    iter_training,
    load_best_metric_row,
    load_metric_history,
    mo,
    model_kind,
    project_root,
    render_loss_history_svg,
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

        panels = [
            mo.md(
                f"""
                **name**: `{session.resume_info.display_name}`  
                **experiment**: `{session.resume_info.base_run_id}`  
                **run instance**: `{session.resume_info.run_id}`  
                **checkpoint**: `{session.checkpoint_path.relative_to(project_root)}`  
                **log**: `{session.log_path.relative_to(project_root)}`  
                **run attempt**: `{session.resume_info.attempt_id}`  
                **best validation loss**: `{best_metric_label}`  
                **best checkpoint validation weighted accuracy**: `{best_weighted_accuracy}`
                {message}
                """
            ),
            loss_history_plot(session),
        ]
        panels.append(
            mo.Html(
                """
                <table>
                  <thead><tr><th>step</th><th>train loss</th><th>validation loss</th><th>train weighted accuracy</th><th>validation weighted accuracy</th><th>best</th></tr></thead>
                  <tbody>
                """
                + "".join(
                    f"<tr><td>{row['step']}</td><td>{metric_cell(row, 'loss')}</td><td>{metric_cell(row, 'validation_loss')}</td><td>{metric_cell(row, 'weighted_accuracy')}</td><td>{metric_cell(row, 'validation_weighted_accuracy')}</td><td>{'yes' if row['best'] else ''}</td></tr>"
                    for row in rows[-12:]
                )
                + "</tbody></table>"
            )
        )
        return mo.vstack(panels)

    if not config.run.run_training:
        output = mo.md("Set `run_training = True` in the configuration to train.")
    else:
        try:
            session = create_training_session(
                model_kind, project_root=project_root, config=config
            )
        except (FileNotFoundError, ValueError, TypeError) as error:
            output = mo.md(str(error))
        else:
            history = []
            mo.output.replace(
                mo.md(
                    f"Preparing `{model_kind}` training on `{session.device}` with `{config.value_count}` values per sample; {session.resume_info.message}."
                )
            )
            step_range = range(session.resume_info.start_step, config.run.steps + 1)
            progress = mo.status.progress_bar(
                step_range,
                title=f"Training LPAP {model_kind}",
                total=len(step_range),
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
                            message=f"Step `{result.step}` / `{config.run.steps}` complete.",
                        )
                    )

            if history:
                final = history[-1]
                output = render_training_output(
                    rows=history,
                    session=session,
                    best_metric=session.training_run.best_metric,
                    message=f"Final step `{final['step']}`: train loss `{final['loss']:.4f}`, best checkpoints are selected by validation loss.",
                )
            else:
                output = mo.md(
                    f"Checkpoint is already at step `{session.resume_info.start_step - 1}`, which is >= configured `steps={config.run.steps}`."
                )

    output
    return


if __name__ == "__main__":
    app.run()
