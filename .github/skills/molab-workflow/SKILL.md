---
name: molab-workflow
description: >-
  Use when working on LPAP via molab remote GPUs, marimo-pair to a molab URL,
  the lpap-molab worktree, or molab-summer branch. Covers pairing, molab-sync,
  detached long AE runs, short shared code-mode cells, and artifact sync.
---

# Molab workflow skill

Read [doc/molab-workflow.md](../../doc/molab-workflow.md) for full notes.

## Setup

- Worktree: `lpap-molab` / branch `molab-summer` (leave `../lpap` on `main`).
- Export targeting env once per session:

  ```bash
  export MOLAB_URL='https://….sb.molab.run/'
  export MOLAB_TOKEN='…'          # or MARIMO_TOKEN
  # Do NOT set MOLAB_SESSION unless the server has multiple sessions.
  ```

  **Default: leave `MOLAB_SESSION` unset** so execute-code auto-picks the only
  active session. If you set it (or pass `--session`), `molab-exec` checks
  `/api/sessions` and **exits with an error** when the id is stale — it will
  not silently hit a dead session.

- Prefer the wrapper (forwards to marimo-pair `execute-code.sh`):

  ```bash
  bash .github/skills/molab-workflow/scripts/molab-exec.sh -c 'print(1)'
  bash .github/skills/molab-workflow/scripts/molab-exec.sh <<'PY'
  print(1)
  PY
  ```

  Override script location with `MARIMO_PAIR_SCRIPTS` if needed (default
  `~/.cursor/skills/marimo-pair/scripts`).

## Script index

All under `.github/skills/molab-workflow/scripts/` (require `MOLAB_URL` +
`MOLAB_TOKEN` unless noted):

| Script | Purpose |
|--------|---------|
| `molab-exec.sh` | Scratchpad / `cm` against the paired kernel |
| `molab-sync.sh` | Reinstall `lpap` from git + storage/helpers/secrets |
| `molab-inject-secrets.sh` | Secrets → kernel env only |
| `molab-train-status.sh` | AE bg / ckpt / SQLite summary |
| `molab-notify.sh` | Pushover ping (end of agent-side long waits) |
| `molab-launch-ae-energy-bank.sh` | Detached multi-pair AE (bank flows) |
| `molab-launch-flow-energy-bank.sh` | Detached bidirectional energy-bank flow |
| `molab-push-package-file.sh` | Hot-patch one local `src/lpap/*.py` into site-packages + reload |
| `molab-export-notebook.sh` | Backport live cells → `notebooks/molab_lab.py` |

**Hot-patch vs sync:** use `molab-push-package-file.sh` for short try-before-commit
probes (e.g. gallery γ). It is **ephemeral** — `molab-sync` / pip reinstall
overwrites it. Notebook `from lpap… import …` bindings stay stale until the
owning cell is re-run. Durable path: commit + push + `molab-sync.sh`.

**Backport:** while paired, live kernel is source of truth. When the lab cells
stabilize, run `molab-export-notebook.sh` (or `--dry-run`) before committing
`notebooks/molab_lab.py`. Prefer named cells first (see below).

- **After push / kernel restart — one-shot sync:**

  ```bash
  bash .github/skills/molab-workflow/scripts/molab-sync.sh
  # optional: --ref <sha|branch>  --skip-install|--skip-secrets|--skip-storage|--skip-helpers
  ```

  Force-reinstalls `lpap` from git, copies `configs/storage.toml` →
  `/marimo/configs/storage.toml`, copies repo `molab/` helpers →
  `/marimo/molab/`, injects `configs/secrets.toml` → kernel env
  (`HF_TOKEN`, `PUSHOVER_*`, `LPAP_NOTIFY_ON_FINISHED`). Removes legacy
  `/marimo/.hf_token` / `.pushover_*` files.

- **Secrets only** (if sync is too heavy):

  ```bash
  bash .github/skills/molab-workflow/scripts/molab-inject-secrets.sh
  ```

  Pass the same env into detached bg workers at spawn (the versioned launcher
  does this from the kernel env).

- Before cell surgery on a messy remote notebook:

  ```bash
  bash ~/.cursor/skills/marimo-pair/scripts/notebook-map.sh --url "$MOLAB_URL" --token "$MOLAB_TOKEN"
  bash ~/.cursor/skills/marimo-pair/scripts/notebook-ready.sh --url "$MOLAB_URL" --token "$MOLAB_TOKEN" mo torch
  ```

  Prefer `edit_cell` / delete detached POC cells over stacking new ones.

## Cell names (shared notebook)

Humans usually **cannot see marimo cell ids** (`QWLa`, …). Name every durable
lab cell so agent and human share the same labels.

- Set a marimo **cell name** via `ctx.edit_cell(..., name="…")` /
  `ctx.create_cell(..., name="…")`. Prefer snake_case role names
  (`ae_setup`, `status`, `gallery_cache`, `gallery_gamma`, `gallery_view`,
  `e0_peak_probe`).
- Put `# cell: <name>` as the **first line** of the cell body (matches the
  marimo name). Update it if you rename.
- In chat, refer to cells **by name** (“re-run `gallery_cache`”), not by id.
- Agents may still use ids internally; `notebook-map.sh` prints both.
- Avoid naming markdown-only cells (clutters the UI). Scratchpad probes stay
  unnamed / ephemeral — promote + name only if they become durable.
- **Do not force-reinstall `lpap` in durable `ae_setup` on every run** — that
  wipes hot-patches and surprises re-run-all. Install only when missing (or
  behind `LPAP_FORCE_REINSTALL=1`); day-to-day package updates go through
  `molab-sync.sh`.

Current AE lab roles (rename/extend as the notebook evolves):

| Name | Role |
|------|------|
| `ae_setup` | imports + AE config |
| `status` | bg worker / train status |
| `gallery_cache` | ckpt load + forward (slow refresh) |
| `gallery_gamma` | display-γ slider |
| `gallery_view` | PNG gallery render (cheap) |
| `e0_peak_probe` | e0 peak diagnostics |

## Hard rules

1. **One paired notebook** (`notebooks/molab_lab.py` as the durable surface).
2. **Default long-train path = detached bg worker** (`molab-launch-ae-energy-bank.sh`
   + pid/log + Pushover/HF). Use **visible code-mode cells** only for short /
   shared chunks the human should watch. Scratchpad = probes / installs / sync.
3. **No parallel `execute-code` during a train cell** — and do not spawn a
   second bg worker while one is alive (launcher refuses). Agent-side waits:
   harness sleep/poll + `molab-train-status.sh`, not kernel `sleep`.
4. **Live kernel is source of truth** — use `cm`, do not edit `.py` on disk
   while paired.
5. **Install via `molab-sync.sh`** after new commits (or the git pip one-liner
   in the doc). `lpap` is not on PyPI.
6. **Sync artifacts** with `lpap.artifact_sync` ↔ `artifacts.bucket` from
   `/marimo/configs/storage.toml`. Long AE runs: launcher sets
   `upload_artifacts_on_checkpoint=True` (+ notify on finished).
7. **Images** via `ensure_image_tensor_archive` / `pixi run data-download`
   using `images.*` from that same storage.toml (cached `.pt`).
8. **Name durable cells** (see above); talk about them by name with the human.
9. **No unconditional force-reinstall in lab setup cells** — use `molab-sync`
   (or `LPAP_FORCE_REINSTALL`) instead of wiping the env on every re-run-all.

## Long AE energy-bank runs (default)

```bash
bash .github/skills/molab-workflow/scripts/molab-sync.sh
bash .github/skills/molab-workflow/scripts/molab-launch-ae-energy-bank.sh --target-steps 58200
bash .github/skills/molab-workflow/scripts/molab-train-status.sh
```

Implementation: repo [`molab/jobs.py`](../../../molab/jobs.py)
(`launch_ae_energy_bank_bg`, synced to `/marimo/molab/`) writes
`training_logs/train_image_autoencoder_multi_energy_bank_bg.py` and spawns with
pid/log under `image_autoencoder_multi_energy_bank_bg.*`.

Bank flow retrain (after encoding `data/encoded_energies_ae_best.pt`):

```bash
bash .github/skills/molab-workflow/scripts/molab-launch-flow-energy-bank.sh \
  --target-steps 10000
```

**Encode bank checklist (do not skip):**
1. Use `encode_image_dataset_to_energy_bank` / `ImageTensorDataset.float_batch`
   — never raw `dataset.images` (uint8). `normalize=True` only applies in
   `__getitem__` / `float_batch`.
2. Probe asserts must pass (mean≈0, std≲0.5, not close to raw Hilbert).
3. Glance gallery energy after first bank-flow / AE best before long runs.

## Background worker status

```bash
bash .github/skills/molab-workflow/scripts/molab-train-status.sh
# or locally / in molab-exec:
python -m lpap.training_status --project-root /marimo \
  --checkpoint image_autoencoder_multi_energy_bank.pt \
  --log image_autoencoder_multi_energy_bank.sqlite \
  --run-id image_autoencoder_multi_energy_bank \
  --bg-stem image_autoencoder_multi_energy_bank_bg
```

Reports ckpt step, SQLite max step, and bg alive / last log step.

## Training order

`surrogate` → `decoder` → image flows (or energy-bank flows) → `image_autoencoder`.
