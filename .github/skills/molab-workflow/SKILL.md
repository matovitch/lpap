---
name: molab-workflow
description: >-
  Use when working on LPAP via molab remote GPUs, marimo-pair to a molab URL,
  the lpap-molab worktree, or molab-summer branch. Covers pairing, one-notebook
  policy, git install of lpap, chunked training, and artifact sync.
---

# Molab workflow skill

Read [doc/molab-workflow.md](../../doc/molab-workflow.md) for full notes.

## Setup

- Worktree: `lpap-molab` / branch `molab-summer` (leave `../lpap` on `main`).
- Export targeting env once per session:

  ```bash
  export MOLAB_URL='https://….sb.molab.run/'
  export MOLAB_TOKEN='…'          # or MARIMO_TOKEN
  export MOLAB_SESSION='s_…'      # if the server has multiple sessions
  ```

- Prefer the wrapper (forwards to marimo-pair `execute-code.sh`):

  ```bash
  bash .github/skills/molab-workflow/scripts/molab-exec.sh -c 'print(1)'
  bash .github/skills/molab-workflow/scripts/molab-exec.sh <<'PY'
  print(1)
  PY
  ```

  Override script location with `MARIMO_PAIR_SCRIPTS` if needed (default
  `~/.cursor/skills/marimo-pair/scripts`).

- Before cell surgery on a messy remote notebook:

  ```bash
  bash ~/.cursor/skills/marimo-pair/scripts/notebook-map.sh --url "$MOLAB_URL" --token "$MOLAB_TOKEN"
  bash ~/.cursor/skills/marimo-pair/scripts/notebook-ready.sh --url "$MOLAB_URL" --token "$MOLAB_TOKEN" mo torch
  ```

  Prefer `edit_cell` / delete detached POC cells over stacking new ones.

## Hard rules

1. **One paired notebook** (`notebooks/molab_lab.py` as the durable surface).
2. **Code-mode cells for shared training** (`hide_code=False`, progress UI).
   Scratchpad only for probes / installs / HF sync.
3. **No parallel `execute-code` during a train cell** — chunk → poll → chunk.
   Agent-side waits: `execute-watch.sh` (short probes), not kernel `sleep`.
4. **Live kernel is source of truth** — use `cm`, do not edit `.py` on disk
   while paired.
5. **Install from git** (force-reinstall after new commits if needed):

   ```text
   pip install --no-deps --force-reinstall \
     "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
   pip install "jaxtyping>=0.3.7"
   ```

6. **Sync artifacts** with `lpap.artifact_sync` ↔ bucket
   `matovitch/lpap-molab-artifacts` before idle kill. For long AE runs, set
   `upload_artifacts_on_checkpoint=True` on the run config so each improved
   checkpoint (+ SQLite) uploads automatically (and again on `mark_finished`).
7. **Images** from public bucket `matovitch/lpap-images` via
   `ensure_image_tensor_archive` / `pixi run data-download` (cached `.pt`).

## Background worker status (notebook host)

Detached jobs on molab should write a pidfile + log under `/marimo/…`. Probe
with a **short** `molab-exec` (do not sleep in the scratchpad):

```bash
bash .github/skills/molab-workflow/scripts/molab-exec.sh <<'PY'
from pathlib import Path
import os
pid_path = Path("/marimo/training_logs/example.pid")
log_path = Path("/marimo/training_logs/example.log")
alive = False
if pid_path.exists():
    try:
        os.kill(int(pid_path.read_text().strip()), 0)
        alive = True
    except OSError:
        alive = False
print("alive", alive)
if log_path.exists():
    print("\n".join(log_path.read_text().splitlines()[-12:]))
PY
```

For LPAP run KPIs from checkpoint/SQLite, prefer
`python -m lpap.training_status` inside `molab-exec` (same module as
`pixi run train-status` locally).

## Training order

`surrogate` → `decoder` → image flows → `image_autoencoder`.
