import marimo

__generated_with = "0.23.5"
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

    from lpap.surrogate_training import (
        LPAPSurrogateTrainingConfig,
        create_lpap_surrogate_training_session,
        iter_lpap_surrogate_training,
    )

    return (
        LPAPSurrogateTrainingConfig,
        create_lpap_surrogate_training_session,
        iter_lpap_surrogate_training,
        mo,
        project_root,
    )


@app.cell
def _(mo):
    mo.md("""
    # LPAP surrogate training
    """)
    return


@app.cell
def _():
    run_training = True
    resume_from_checkpoint = True
    steps = 4000
    batch_size = 32
    bucket_count = 128
    probe_count = 8
    k_max = 4
    harmonic_count = 16
    hidden_dim = 256
    layer_count = 8
    head_count = 8
    learning_rate = 1.0e-3
    seed = 123
    permutation_seed = 123
    checkpoint_every = 25
    checkpoint_on_improvement = False
    display_every = 5
    log_every = 1
    run_id = "surrogate_synthetic"
    return (
        batch_size,
        bucket_count,
        checkpoint_every,
        checkpoint_on_improvement,
        display_every,
        harmonic_count,
        head_count,
        hidden_dim,
        k_max,
        layer_count,
        learning_rate,
        log_every,
        permutation_seed,
        probe_count,
        resume_from_checkpoint,
        run_id,
        run_training,
        seed,
        steps,
    )


@app.cell
def _(
    LPAPSurrogateTrainingConfig,
    batch_size,
    bucket_count,
    checkpoint_every,
    checkpoint_on_improvement,
    create_lpap_surrogate_training_session,
    display_every,
    harmonic_count,
    head_count,
    hidden_dim,
    iter_lpap_surrogate_training,
    k_max,
    layer_count,
    learning_rate,
    log_every,
    mo,
    permutation_seed,
    probe_count,
    project_root,
    resume_from_checkpoint,
    run_id,
    run_training,
    seed,
    steps,
):
    def render_training_output(*, rows, session, best_metric, message):
        return mo.vstack(
            [
                mo.md(
                    f"""
                    **device**: `{session.device}`  
                    **checkpoint**: `{session.checkpoint_path.relative_to(project_root)}`  
                    **log**: `{session.log_path.relative_to(project_root)}`  
                    **best loss**: `{best_metric:.4f}`  
                    {message}
                    """
                ),
                mo.Html(
                    """
                    <table>
                      <thead><tr><th>step</th><th>loss</th><th>accuracy</th><th>weighted accuracy</th><th>mean weight</th><th>best</th></tr></thead>
                      <tbody>
                    """
                    + "".join(
                        f"<tr><td>{row['step']}</td><td>{row['loss']:.4f}</td><td>{row['accuracy']:.4f}</td><td>{row['weighted_accuracy']:.4f}</td><td>{row['mean_weight']:.4f}</td><td>{'yes' if row['best'] else ''}</td></tr>"
                        for row in rows[-12:]
                    )
                    + "</tbody></table>"
                ),
            ]
        )

    if not run_training:
        history = []
        output = mo.md(
            "Set `run_training = True` in the configuration cell to train a fresh surrogate on synthetic harmonic batches."
        )
    else:
        config = LPAPSurrogateTrainingConfig(
            run_training=run_training,
            resume_from_checkpoint=resume_from_checkpoint,
            steps=steps,
            batch_size=batch_size,
            bucket_count=bucket_count,
            probe_count=probe_count,
            k_max=k_max,
            harmonic_count=harmonic_count,
            hidden_dim=hidden_dim,
            layer_count=layer_count,
            head_count=head_count,
            learning_rate=learning_rate,
            seed=seed,
            permutation_seed=permutation_seed,
            checkpoint_every=checkpoint_every,
            checkpoint_on_improvement=checkpoint_on_improvement,
            display_every=display_every,
            log_every=log_every,
            run_id=run_id,
        )
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

            step_range = range(session.resume_info.start_step, steps + 1)
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
                            message=f"Step `{result.step}` / `{steps}` complete.",
                        )
                    )

            if history:
                final = history[-1]
                output = render_training_output(
                    rows=history,
                    session=session,
                    best_metric=session.training_run.best_metric,
                    message=f"Final step `{final['step']}`: loss `{final['loss']:.4f}`, weighted accuracy `{final['weighted_accuracy']:.4f}`.",
                )
            else:
                output = mo.md(
                    f"Checkpoint is already at step `{session.resume_info.start_step - 1}`, which is >= configured `steps={steps}`."
                )

    output
    return


if __name__ == "__main__":
    app.run()
