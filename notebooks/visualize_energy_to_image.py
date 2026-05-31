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

    from lpap.training_log import (
        list_training_runs,
        load_best_metric_row,
        load_metric_history,
    )
    from lpap.training_notebook import (
        default_energy_to_image_training_config,
        render_recent_runs_table,
        training_log_path,
    )
    from lpap.training_plots import render_loss_history_svg
    from lpap.visualization_notebook import render_energy_to_image_run_gallery

    return (
        default_energy_to_image_training_config,
        list_training_runs,
        load_best_metric_row,
        load_metric_history,
        mo,
        project_root,
        render_energy_to_image_run_gallery,
        render_loss_history_svg,
        render_recent_runs_table,
        training_log_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Energy-to-image visualization
    """)
    return


@app.cell
def _(
    default_energy_to_image_training_config,
    list_training_runs,
    mo,
    project_root,
    render_recent_runs_table,
    training_log_path,
):
    config = default_energy_to_image_training_config()
    log_path = training_log_path(project_root, config)
    recent_runs = list_training_runs(log_path, base_run_id=config.run.run_id, limit=10)
    run_options = {
        f"{row['display_name']} | {row['status']} | step {row['last_step'] or 0}": row[
            "run_id"
        ]
        for row in recent_runs
    }
    if run_options:
        run_picker = mo.ui.dropdown(
            options=run_options,
            value=next(iter(run_options)),
            label="Run",
        )
    else:
        run_picker = mo.ui.dropdown(
            options={"No energy-to-image runs": ""},
            value="No energy-to-image runs",
            label="Run",
            disabled=True,
        )
    recent_runs_table = render_recent_runs_table(recent_runs)
    mo.vstack([run_picker, mo.Html(recent_runs_table)])
    return log_path, run_picker


@app.cell
def _(
    load_best_metric_row,
    load_metric_history,
    log_path,
    mo,
    project_root,
    render_energy_to_image_run_gallery,
    render_loss_history_svg,
    run_picker,
):
    active_run_id = run_picker.value
    integration_steps = (64, 32, 16, 8)
    if not active_run_id:
        output = mo.md("No energy-to-image runs have been logged yet.")
    else:
        history = load_metric_history(
            log_path,
            run_id=active_run_id,
            metric_names=("loss", "validation_loss"),
        )
        best = load_best_metric_row(log_path, run_id=active_run_id)
        best_loss = "n/a" if best is None else f"{best['validation_loss']:.4f}"
        try:
            gallery = mo.Html(
                render_energy_to_image_run_gallery(
                    project_root=project_root,
                    log_path=log_path,
                    run_id=active_run_id,
                    integration_steps=integration_steps,
                )
            )
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
            gallery = mo.md(f"Energy-to-image gallery unavailable: `{error}`")
        output = mo.vstack(
            [
                mo.md(
                    f"**selected run**: `{active_run_id}`  \n**best validation loss**: `{best_loss}`"
                ),
                mo.Html(render_loss_history_svg(history)),
                gallery,
            ]
        )
    output
    return


if __name__ == "__main__":
    app.run()
