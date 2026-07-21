---
name: molab-workflow
description: >-
  Use when working on LPAP via molab remote GPUs, marimo-pair to a molab URL,
  the lpap-molab worktree, or molab-summer branch. Covers pairing, one-notebook
  policy, git install of lpap, chunked training, and progress polling.
---

# Molab workflow skill

Read [doc/molab-workflow.md](../../doc/molab-workflow.md) for the full notes.
This skill is the short agent checklist.

## Setup

- Work in the sibling worktree `lpap-molab` on branch `molab-summer`.
- Leave `../lpap` on `main` as the local Pixi home.
- Pair with the user-provided molab URL + token via
  `~/.cursor/skills/marimo-pair/scripts/execute-code.sh`.

## Hard rules

1. **One paired notebook.** Do not ask for a second molab notebook unless the
   current session is dead. Prefer a single generic lab notebook.
2. **Prefer code-mode cells for shared work.** Long training / status updates
   should live in visible notebook cells (`cm.create_cell` / `run_cell` with
   `hide_code=False` and `mo.output.replace` / progress UI) so the human is
   not blind. Use the scratchpad only for short probes, installs, or HF
   upload/download — not as the only place a multi-minute train runs.
3. **No parallel `execute-code` during training.** A second call can
   `MarimoInterrupt` the train cell. Chunk train → poll → chunk.
4. **Live kernel is source of truth.** Use `marimo._code_mode` (`cm`) to change
   cells while paired; do not edit the notebook `.py` on disk during the
   session.
5. **Install from git**, not PyPI or conda/prefix.dev:

   ```text
   pip install --no-deps "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
   pip install "jaxtyping>=0.3.7"
   ```

6. **Artifacts** under `/marimo/checkpoints` and `/marimo/training_logs`.
   Sync via `lpap.artifact_sync` to HF bucket `matovitch/lpap-molab-artifacts`
   (`pixi run artifacts-upload` / `artifacts-download`). Remind the user to
   sync before idle kill if they care about the run.

## Progress polling

Between chunks only:

```bash
bash ~/.cursor/skills/marimo-pair/scripts/execute-code.sh \
  --url "$MOLAB_URL" --token "$MOLAB_TOKEN" <<'PY'
import sqlite3
conn = sqlite3.connect("/marimo/training_logs/surrogate.sqlite")
print("max_step", conn.execute("SELECT MAX(step) FROM step_metrics").fetchone()[0])
PY
```

Adjust the sqlite path/name for the active `model_kind`.

## Training order

`surrogate` → `decoder` → image flows → reflow → `image_autoencoder`.
Surrogate needs no image dataset (synthetic harmonics).

## Notebook inventory

- Primary remote surface: `notebooks/molab_lab.py` (generic chunked trainer;
  same model_kind coverage as local `notebooks/train.py`).
- Optional smoke: `notebooks/molab_poc.py`.
- Local Pixi: `notebooks/train.py` unchanged on `main`.
- Local preview: `pixi run notebook-molab-lab` from the molab worktree.
