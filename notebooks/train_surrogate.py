import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import torch

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from lpap.data import SyntheticHarmonicConfig
    from lpap.surrogate_training import (
        LPAPSurrogateDataConfig,
        LPAPSurrogateModelConfig,
        LPAPSurrogateOptimizerConfig,
        LPAPSurrogateRunConfig,
        LPAPSurrogateTrainingConfig,
        LPAPSurrogateValidationConfig,
        create_lpap_surrogate_training_session,
        iter_lpap_surrogate_training,
    )

    return (
        LPAPSurrogateDataConfig,
        LPAPSurrogateModelConfig,
        LPAPSurrogateOptimizerConfig,
        LPAPSurrogateRunConfig,
        LPAPSurrogateTrainingConfig,
        LPAPSurrogateValidationConfig,
        SyntheticHarmonicConfig,
        create_lpap_surrogate_training_session,
        iter_lpap_surrogate_training,
        mo,
        project_root,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # LPAP surrogate training
    """)
    return


@app.cell
def _(
    LPAPSurrogateDataConfig,
    LPAPSurrogateModelConfig,
    LPAPSurrogateOptimizerConfig,
    LPAPSurrogateRunConfig,
    LPAPSurrogateTrainingConfig,
    LPAPSurrogateValidationConfig,
    SyntheticHarmonicConfig,
    torch,
):
    harmonics = SyntheticHarmonicConfig(
        harmonic_count=16,
        gain_variance=1.0,
        gain_half_life=4.0,
        spikiness_range=(4.0, 8.0),
        dtype=torch.float32,
    )
    data = LPAPSurrogateDataConfig(
        batch_size=32,
        bucket_count=128,
        probe_count=8,
        harmonics=harmonics,
    )
    model = LPAPSurrogateModelConfig(
        k_max=4,
        hidden_dim=256,
        layer_count=8,
        head_count=8,
    )
    optimizer = LPAPSurrogateOptimizerConfig(learning_rate=1.0e-3)
    validation = LPAPSurrogateValidationConfig(
        enabled=True,
        every=100,
        batch_size=256,
        seed=10_123,
        validate_at_end=True,
    )
    run = LPAPSurrogateRunConfig(
        run_training=True,
        resume_from_checkpoint=True,
        steps=4000,
        seed=123,
        permutation_seed=123,
        display_every=5,
        log_every=1,
        run_id="surrogate_synthetic",
        checkpoint_name="surrogate_synthetic.pt",
        log_name="surrogate.sqlite",
    )
    config = LPAPSurrogateTrainingConfig(
        data=data,
        model=model,
        optimizer=optimizer,
        validation=validation,
        run=run,
    )
    return (config,)


@app.cell
def _(
    config,
    create_lpap_surrogate_training_session,
    iter_lpap_surrogate_training,
    mo,
    project_root,
):
    def render_training_output(*, rows, session, best_metric, message):
        best_metric_label = "n/a" if best_metric is None else f"{best_metric:.4f}"

        def metric_cell(row, name):
            value = row.get(name)
            return "" if value is None else f"{value:.4f}"

        return mo.vstack(
            [
                mo.md(
                    f"""
                    **device**: `{session.device}`  
                    **checkpoint**: `{session.checkpoint_path.relative_to(project_root)}`  
                    **log**: `{session.log_path.relative_to(project_root)}`  
                                        **best validation loss**: `{best_metric_label}`
                    {message}
                    """
                ),
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
                ),
            ]
        )

    if not config.run.run_training:
        history = []
        output = mo.md(
            "Set `run_training = True` in the configuration cell to train a fresh surrogate on synthetic harmonic batches."
        )
    else:
        try:
            session = create_lpap_surrogate_training_session(
                project_root=project_root, config=config
            )
        except ValueError as error:
            history = []
            output = mo.md(str(error))
        else:
            history = []
            mo.output.replace(
                mo.md(
                    f"Preparing training on `{session.device}` with `{config.value_count}` values per sample; {session.resume_info.message}."
                )
            )

            step_range = range(session.resume_info.start_step, config.run.steps + 1)
            progress = mo.status.progress_bar(
                step_range,
                title="Training LPAP surrogate",
                total=len(step_range),
            )
            events = iter_lpap_surrogate_training(session)
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
