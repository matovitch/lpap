# Agent Notes

This project uses Pixi. Prefer Pixi for Python package installs, environment commands, and reusable project commands.

- Use `pixi add` or `pixi add --pypi` instead of calling `pip` directly.
- Run project commands through `pixi run` when practical.
- Add reusable commands as Pixi tasks in `pixi.toml`; use task arguments for paths or modes that should be configurable.
- Keep generated datasets and large archives out of Git unless the project explicitly chooses a large-data mechanism.
- Use `jaxtyping` annotations for tensor-facing APIs. Prefer explicit semantic dimension names such as `batch`, `n`, `buckets`, `channel`, `height`, and `width`, and include tensor dtype families such as `Float`, `Int`, or `UInt8` when practical.
- Use marimo notebooks as pair-programming scratchpads for visual and interactive exploration. Prefer source-controlled `.py` notebooks under `notebooks/`, launch them through Pixi tasks, and set `PYTHONPATH=src` so notebook cells import the local `lpap` package.
- For agent-assisted marimo sessions, start notebooks with `--no-token` when practical so `marimo-team/marimo-pair` can discover the server. Discover an existing server before starting a new one, and mutate live notebooks through marimo code mode rather than editing the notebook file while a marimo server is running.
- Keep notebook cells reactive and small: one setup/import cell, one control cell, one computation cell, and one visualization/output cell is a good default for LPAP experiments.
- Keep reusable training/checkpoint/logging logic in `src/lpap/` helpers. Marimo notebooks should mostly declare config, call helpers, and render results rather than containing long training loops or persistence code.
- Save local training checkpoints under `checkpoints/` and keep them out of Git. Prefer checkpoint payloads with separate `model_state` and `best_model_state`, plus optimizer state and lightweight training metadata when available.
- Save local training logs under `training_logs/` as SQLite databases. Keep run configuration in one table and per-step or per-epoch KPIs in another so notebooks can resume and inspect training without parsing checkpoint payloads.
- In marimo training loops, avoid checkpointing, SQLite writes, or `mo.output.replace` on every step unless the run is tiny. Use configurable cadences such as `checkpoint_every`, `log_every`, and `display_every`.
- See `.github/skills/pixi-workflow/SKILL.md` for the local Pixi workflow skill.
