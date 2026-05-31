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
        default_surrogate_training_config,
        render_recent_runs_table,
        training_log_path,
    )
    from lpap.training_plots import render_loss_history_svg

    return (
        default_surrogate_training_config,
        list_training_runs,
        load_best_metric_row,
        load_metric_history,
        mo,
        project_root,
        render_loss_history_svg,
        render_recent_runs_table,
        training_log_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # LPAP surrogate visualization
    """)
    return


@app.cell
def _(
    default_surrogate_training_config,
    list_training_runs,
    project_root,
    render_recent_runs_table,
    training_log_path,
):
    config = default_surrogate_training_config()
    selected_run_id = ""
    log_path = training_log_path(project_root, config)
    recent_runs = list_training_runs(log_path, base_run_id=config.run.run_id, limit=10)
    active_run_id = selected_run_id or (recent_runs[0]["run_id"] if recent_runs else "")
    recent_runs_table = render_recent_runs_table(recent_runs)
    return active_run_id, log_path, recent_runs, recent_runs_table, selected_run_id


@app.cell
def _(
    active_run_id,
    load_best_metric_row,
    load_metric_history,
    log_path,
    mo,
    recent_runs_table,
    render_loss_history_svg,
    selected_run_id,
):
    if not active_run_id:
        output = mo.vstack(
            [
                mo.md("No surrogate runs have been logged yet."),
                mo.Html(recent_runs_table),
            ]
        )
    else:
        history = load_metric_history(
            log_path,
            run_id=active_run_id,
            metric_names=("loss", "validation_loss"),
        )
        best = load_best_metric_row(log_path, run_id=active_run_id)
        best_loss = "n/a" if best is None else f"{best['validation_loss']:.4f}"
        best_weighted_accuracy = (
            "n/a"
            if best is None or best.get("validation_weighted_accuracy") is None
            else f"{best['validation_weighted_accuracy']:.4f}"
        )
        output = mo.vstack(
            [
                mo.md(
                    f"**selected run**: `{active_run_id}`  \n**selection source**: `{selected_run_id or 'latest run'}`  \n**best validation loss**: `{best_loss}`  \n**best checkpoint validation weighted accuracy**: `{best_weighted_accuracy}`"
                ),
                mo.Html(render_loss_history_svg(history)),
                mo.Html(recent_runs_table),
            ]
        )
    output
    return


if __name__ == "__main__":
    app.run()
