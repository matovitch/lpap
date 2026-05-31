---
name: pixi-workflow
description: "Use when: working in this LPAP repo with Python dependencies, Pixi tasks, dataset conversion commands, or environment setup. Prefer Pixi commands and reusable task definitions."
---

# Pixi Workflow

When working in this repository:

- Prefer `pixi run <task>` or `pixi run <command>` for commands that need the project environment.
- Install conda packages with `pixi add <package>`.
- Install PyPI packages with `pixi add --pypi <package>` only when no suitable conda package exists.
- Add reusable workflows to `pixi.toml` under `[tasks]` with a short `description`.
- Use Pixi task arguments for configurable paths instead of duplicating tasks.
- Avoid writing ad hoc shell scripts when a Pixi task can name and document the workflow.
- Keep large generated data out of Git unless Git LFS, DVC, or an external dataset host is intentionally configured.
- For tensor-facing Python APIs, use `jaxtyping` with meaningful dimension names and dtype aliases. Good names in this repo include `batch`, `n`, `buckets`, `channel`, `height`, and `width`; prefer annotations like `Float[torch.Tensor, "batch n"]` and `UInt8[torch.Tensor, "batch channel height width"]` where they match the API.
- Use marimo notebooks under `notebooks/` for interactive training and visualization, launched through Pixi tasks with `PYTHONPATH=src`.
- Start agent-assisted marimo sessions with `--no-token` when practical so marimo-pair tooling can discover them. If a marimo server is running, mutate the live notebook through marimo code mode instead of editing the file on disk.
- Keep marimo notebooks thin: define editable config variables, call reusable helpers in `src/lpap/`, and render outputs. Put training loops, checkpointing, and SQLite logging in source modules with tests.
- Keep checkpoints under `checkpoints/` and SQLite training logs under `training_logs/`; both should remain local artifacts unless a deliberate model/data versioning system is added.

## Verifying Changes

- Before committing, run `pixi run lint` (Ruff) and `pixi run test` (the unittest suite); both must pass.
- Tests live under `test/` with one module per `src/lpap/` source module. When you change a module, add or update its matching test instead of leaving it uncovered.
- Keep the suite CPU-runnable. CUDA and Triton-GPU parity tests are gated with `unittest.skipUnless(torch.cuda.is_available(), ...)`. CI uses CPU-only runners and mocks the `__cuda` virtual package via `CONDA_OVERRIDE_CUDA` (set in `.github/workflows/test.yml`), so anything not gated must run without a GPU. `lpap_triton` already falls back to `lpap_torch` on non-CUDA tensors.
- Ruff ignores the jaxtyping shape-string false positives (`F722`/`F821`) globally through `pyproject.toml`; do not sprinkle inline `# noqa` for them.

## Useful Facts

- This is a research repo: local checkpoint and SQLite schemas are not kept backward-compatible. Regenerate stale artifacts rather than migrating, unless migration is explicitly requested.
- Train models in dependency order (surrogate → decoder, the two flows, the reflow student) before the end-to-end `image_autoencoder`; model-dependent config (harmonics, bucket layout) is read from upstream checkpoints, not duplicated in TOML.
