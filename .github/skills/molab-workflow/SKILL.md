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
- Pair with user URL + token via
  `~/.cursor/skills/marimo-pair/scripts/execute-code.sh`.
- Before cell surgery on a messy remote notebook, run
  `~/.cursor/skills/marimo-pair/scripts/notebook-map.sh --url …`
  (prefer `edit_cell` / delete detached POC cells over stacking new ones).

## Hard rules

1. **One paired notebook** (`notebooks/molab_lab.py` as the durable surface).
2. **Code-mode cells for shared training** (`hide_code=False`, progress UI).
   Scratchpad only for probes / installs / HF sync.
3. **No parallel `execute-code` during a train cell** — chunk → poll → chunk.
4. **Live kernel is source of truth** — use `cm`, do not edit `.py` on disk
   while paired.
5. **Install from git** (force-reinstall after new commits if needed):

   ```text
   pip install --no-deps --force-reinstall \
     "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
   pip install "jaxtyping>=0.3.7"
   ```

6. **Sync artifacts** with `lpap.artifact_sync` ↔ bucket
   `matovitch/lpap-molab-artifacts` before idle kill.
7. **Images** from public bucket `matovitch/lpap-images` via
   `ensure_image_tensor_archive` / `pixi run data-download` (cached `.pt`).

## Training order

`surrogate` → `decoder` → image flows → reflow → `image_autoencoder`.
