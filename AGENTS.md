# Agent Notes

## Tool layers

Use the matching skill; don’t mix host/project logic into marimo-pair.

| Layer | Skill | Tools | Pixi? |
| --- | --- | --- | --- |
| **marimo** | `~/.cursor/skills/marimo-pair` | `execute-code`, `execute-watch`, `notebook-map`, `notebook-ready` | No |
| **molab** | `.github/skills/molab-workflow` | `molab-exec.sh`, `molab-sync.sh`, …; helpers in repo `molab/` (synced to `/marimo/molab/`) | No |
| **project** | `.github/skills/pixi-workflow` | `pixi run …`, `src/lpap/` (e.g. `train-status`, `artifacts-*`) | Yes locally |

On molab, call the same `lpap` modules inside `molab-exec` after git install. Details: [doc/molab-workflow.md](doc/molab-workflow.md).

## Project invariants

- **Pixi** for env/installs/tasks (`pixi add` / `pixi run`); add reusable work as tasks in `pixi.toml`.
- **jaxtyping** on tensor APIs (`batch`, `n`, `buckets`, … + `Float`/`Int`/`UInt8`). Ruff ignores `F722`/`F821` globally — no inline `# noqa`.
- **Notebooks** under `notebooks/` via Pixi (`PYTHONPATH=src`). Thin cells (config → helpers → render); training/checkpoint/log logic lives in `src/lpap/`. Live session → `marimo._code_mode`, not disk edits. Prefer `--no-token` for pairing discovery.
- **Artifacts**: `checkpoints/` (`model_state` + `best_model_state`) and `training_logs/` SQLite — keep out of Git. No backward compat; regenerate when schemas change. Cadence writes (`log_every`, `display_every`, …), not every step.
- **Verify** with `pixi run lint` and `pixi run test` before commit. One test module per `src/lpap/` module; keep tests CPU-runnable; gate CUDA/Triton with `torch.cuda.is_available()`.

## Molab (summer)

Worktree `lpap-molab` / `molab-summer` (leave `../lpap` on `main`). Human opens
GPU + pair and pastes **URL/token** (`MOLAB_URL` / `MOLAB_TOKEN`); leave
`MOLAB_SESSION` unset unless multi-session (`molab-exec` errors on stale ids).
Needs local `configs/secrets.toml` for sync. **Before launch:**
`molab-train-status` (reuse a live run). Long AE: `molab-sync` →
`molab-launch-ae-energy-bank` → poll status (Pushover/HF). Short/shared →
visible code-mode cells; scratchpad = probes. Full rules:
`.github/skills/molab-workflow/SKILL.md`.
