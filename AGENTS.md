# Agent Notes

Keywords **MUST** / **SHOULD** / **MAY** follow [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119)
(common AGENTS.md pattern: compact priority, not a full IETF rewrite).

## Collaboration

- Prefer debate over silent agreement when stakes are high (wrong data, wasted
  GPU, irreversible ops). Opposing views are welcome unless the user opts out.
- After a clear agent mistake, **MUST** offer a short post-mortem (what / why /
  guardrail) and wait for the user before deep-diving unless they already asked.
- If the user drifts from these norms, briefly remind them of this section.

## Tool layers

Use the matching skill; don’t mix host/project logic into marimo-pair.

| Layer | Skill | Tools | Pixi? |
| --- | --- | --- | --- |
| **marimo** | `~/.cursor/skills/marimo-pair` | `execute-code`, `execute-watch`, `notebook-map`, `notebook-ready` | No |
| **molab** | `.github/skills/molab-workflow` | `molab-exec`, `molab-sync`, `molab-notify`, `molab-train-status`, …; helpers in repo `molab/` | No |
| **project** | `.github/skills/pixi-workflow` | `pixi run …`, `src/lpap/` (e.g. `train-status`, `artifacts-*`) | Yes locally |

On molab, call the same `lpap` modules inside `molab-exec` after git install. Details: [doc/molab-workflow.md](doc/molab-workflow.md).

## Project invariants

- **Pixi** for env/installs/tasks (`pixi add` / `pixi run`); add reusable work as tasks in `pixi.toml`.
- **jaxtyping** on tensor APIs (`batch`, `n`, `buckets`, … + `Float`/`Int`/`UInt8`). Ruff ignores `F722`/`F821` globally — no inline `# noqa`.
- **Notebooks** under `notebooks/` via Pixi (`PYTHONPATH=src`). Thin cells (config → helpers → render); training/checkpoint/log logic lives in `src/lpap/`. Live session → `marimo._code_mode`, not disk edits. Prefer `--no-token` for pairing discovery.
- **Artifacts**: `checkpoints/` (`model_state` + `best_model_state`) and `training_logs/` SQLite — keep out of Git. No backward compat; regenerate when schemas change. Cadence writes (`log_every`, `display_every`, …), not every step.
- **Verify** with `pixi run lint` and `pixi run test` before commit. One test module per `src/lpap/` module; keep tests CPU-runnable; gate CUDA/Triton with `torch.cuda.is_available()`.

## Molab

Worktree `lpap-molab` on `main` (default). Sibling `../lpap` parks the pre-molab
tip on `main-pre-molab`. Human opens
GPU + pair and pastes **URL/token** (`MOLAB_URL` / `MOLAB_TOKEN`); leave
`MOLAB_SESSION` unset unless multi-session (`molab-exec` errors on stale ids).
Needs local `configs/secrets.toml` for sync. Only notebook + `storage/` /
`public/` / `layouts/` persist; treat `data/`, `checkpoints/`, `training_logs/`
as session caches and **lazily** `ensure_*` from HF (no preload in
`molab-sync`). **Before launch:** `molab-train-status` (reuse a live run).
Long AE: `molab-sync` → `molab-launch-ae-energy-bank` → poll status
(Pushover/HF). Short/shared → visible code-mode cells; scratchpad = probes.
Full rules: `.github/skills/molab-workflow/SKILL.md`.

**Long waits in the same turn:** bg training already notifies on finish when
configured. Agent-side waits (encode, HF upload, poll loops) **MUST** end with
`molab-notify.sh` (or `lpap.notify.send_pushover`) so the human gets a phone
ping while approving turns — do not rely on chat alone.
