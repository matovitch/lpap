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
2. **No parallel `execute-code` during training.** A second call can
   `MarimoInterrupt` the train cell. Chunk train → poll → chunk.
3. **Live kernel is source of truth.** Use `marimo._code_mode` (`cm`) to change
   cells while paired; do not edit the notebook `.py` on disk during the
   session.
4. **Install from git**, not PyPI or conda/prefix.dev:

   ```text
   pip install --no-deps "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
   pip install "jaxtyping>=0.3.7"
   ```

5. **Artifacts** under `/marimo/checkpoints` and `/marimo/training_logs`.
   Remind the user to download before idle kill.

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
