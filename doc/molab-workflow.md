# Molab remote GPU workflow (agents)

Drive free [molab](https://molab.marimo.io) GPUs from a local agent via
[marimo-pair](https://github.com/marimo-team/marimo-pair). The default branch
`main` lives in the `lpap-molab` worktree; the sibling checkout parks the
pre-molab tip on `main-pre-molab`.

## Topology

```text
Laptop agent (Cursor/Zed)     molab sandbox
  worktree lpap-molab    →      one paired notebook
  marimo-pair skill      →      HTTP + token (no SSH)
  git push main          →      uv/pip install lpap from git
```

| Role | Path / branch |
| --- | --- |
| Molab / agent home | `../lpap-molab` on `main` |
| Parked pre-molab tip | `../lpap` on `main-pre-molab` |
| Pair skill | `~/.cursor/skills/marimo-pair` |
| Molab wrapper | `.github/skills/molab-workflow/scripts/molab-exec.sh` |

Push package/notebook changes on `main` and install that ref on molab.

## Pairing commands

```bash
export MOLAB_URL='https://….sb.molab.run/'
export MOLAB_TOKEN='…'       # or MARIMO_TOKEN
# Leave MOLAB_SESSION unset unless the server has multiple sessions.

bash .github/skills/molab-workflow/scripts/molab-exec.sh -c 'print(1)'
```

`molab-exec` forwards to marimo-pair `execute-code.sh` (override scripts dir
with `MARIMO_PAIR_SCRIPTS`). Prefer it over pasting long `--url` / `--token`
flags on every call. When `MOLAB_SESSION` / `--session` is set, it validates
the id against `/api/sessions` and fails loudly if stale.

### Secrets injection

Keep real credentials in gitignored [`configs/secrets.toml`](../configs/secrets.toml)
(start from [`configs/secrets.toml.example`](../configs/secrets.toml.example)).
That file is the only secret store — inject sets kernel env only (no
`/marimo/.hf_token` / `.pushover_*` files):

```bash
bash .github/skills/molab-workflow/scripts/molab-inject-secrets.sh
```

Prefer the one-shot post-push sync (reinstall `lpap`, copy `storage.toml`,
copy repo `molab/` helpers → `/marimo/molab/`, inject secrets):

```bash
bash .github/skills/molab-workflow/scripts/molab-sync.sh
# or pin a SHA:  bash …/molab-sync.sh --ref abd69b3
```

Do not put secrets in notebook cells. Re-run sync/inject after every kernel
restart; detached workers inherit env at spawn time.

With Pushover creds present, inject also sets `LPAP_NOTIFY_ON_FINISHED=1` so
`TrainingRun.mark_finished` pings you (or set `notify_on_finished=True` on the
run config). Helpers: `lpap.notify.send_pushover` / `notify_training_finished`.

Prefer **omitting** `MOLAB_SESSION` unless the server has multiple sessions.
When set, `molab-exec` checks `/api/sessions` and exits if the id is missing
(stale ids used to fail silently).

Companion marimo-pair tools (pass URL/token explicitly, or reuse env via
wrappers you compose):

- `notebook-map.sh` — DAG / detached-cell audit before cell surgery
- `notebook-ready.sh` — gate that public names exist in `ctx.globals`
- `execute-watch.sh` — agent-side short probes until a log pattern matches

Extra molab-workflow helpers (same `MOLAB_*` env):

- `molab-push-package-file.sh` — hot-patch one `src/lpap/*.py` into the kernel
  (ephemeral; prefer commit + `molab-sync` for durable updates)
- `molab-export-notebook.sh` — backport live cells → `notebooks/molab_lab.py`
- `molab-train-status.sh` / `molab-launch-ae-energy-bank.sh` / `molab-launch-flow-energy-bank.sh` — long AE / bank-flow runs
- `molab-notify.sh` — Pushover on job finish / handoff (prefer over agent polling)

## Capabilities

**Agent can (once paired):** scratchpad Python; create/edit/run cells via
`marimo._code_mode`; install pip/uv packages; read checkpoints/SQLite; toast.

**Agent cannot:** open molab sessions, attach GPUs, SSH, or use conda/prefix.dev
channels (`uv`/pip only). One pair URL/token at a time.

**Human:** open the durable notebook, attach GPU, **Pair with an agent**, paste
URL + token. Prefer reusing the same molab notebook URL across sessions.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/molab_lab.py` | Primary remote lab: install, CUDA smoke, short shared trains by `model_kind` |
| Local `notebooks/train.py` | Pixi training UI on `main` |
| Local `notebooks/visualize_*.py` | Curves/galleries after `pixi run artifacts-download` |

Controls on the lab notebook: model kind, target steps, chunk steps,
`display_every`, `log_every`. For multi-hour AE prefer the detached launcher
instead of long lab cells. Train logic stays in `src/lpap/`.

While paired, mutate the **live** notebook with `cm` (not the `.py` on disk).
Backport stable cell structure to `molab_lab.py` on `main` when useful.

**Cell naming:** durable lab cells get a marimo name + `# cell: <name>` first
line (`ae_setup`, `status`, `gallery_cache`, …). Talk about cells by name, not
by opaque ids. See `.github/skills/molab-workflow/SKILL.md`. Backport with
`molab-export-notebook.sh` when the live lab should land in git.

## Package install

```bash
python -m pip install --no-deps \
  "lpap @ git+https://github.com/matovitch/lpap.git@main"
python -m pip install "jaxtyping>=0.3.7"
```

Install from `@main` (default branch). `lpap` is not on PyPI — do not
`uv add lpap==0.1.0`. After pulling new commits, `--force-reinstall` the git
ref if imports are missing.

Sandbox layout (molab persistence, from Aug 2026): only notebook source and
`storage/`, `public/`, `layouts/` survive between sessions. Treat
`/marimo/data`, `/marimo/checkpoints`, and `/marimo/training_logs` as
**session-local caches**. Large files (>1 GB: images, energy banks) must live
on Hugging Face and are **lazily pulled** when training/code needs them
(`ensure_image_tensor_archive`, `ensure_project_artifact` /
`load_energy_bank_for_flow`, checkpoint `ensure=` / resume). Prefer
`upload_artifacts_on_checkpoint=True` so HF stays the source of truth — do not
preload everything at sync time.

Sandbox artifacts (ephemeral): `/marimo/checkpoints/*.pt`,
`/marimo/training_logs/*.sqlite`, `/marimo/data/*` (idle ~90 min / session
~12 h).

## Shared training

**Default for multi-hour AE:** versioned detached worker (pidfile + log), not a
blocking code-mode cell:

```bash
bash .github/skills/molab-workflow/scripts/molab-sync.sh
bash .github/skills/molab-workflow/scripts/molab-launch-ae-energy-bank.sh --target-steps 58200
bash .github/skills/molab-workflow/scripts/molab-train-status.sh
```

Launcher refuses if the previous bg pid is still alive. Status reports
checkpoint step, SQLite max step, and bg liveness / last log step.

**Prefer Pushover over agent polling for long jobs.** Detached workers should
run with `notify_on_finished=True` (and upload-on-checkpoint as usual). After
a successful spawn, confirm notify/secrets briefly, then stop — the human
wakes the agent when Pushover fires. Do not park in harness `AwaitShell` /
`molab-train-status` loops waiting for multi-minute or multi-hour runs.
`molab-train-status.sh` is for on-demand checks, not a completion waiter.
Never `sleep` in the kernel to wait on bg work.

**Short / shared chunks only:** visible code-mode cells (`hide_code=False`,
progress bar / `mo.output.replace`) when the human should watch the run.
Scratchpad is for probes, installs, and artifact sync — not long training.

- Do not start a second `execute-code` while a train cell runs (`MarimoInterrupt`).
- For cell chunks: status checks **between** chunks; resume with
  `resume_from_checkpoint=True`.
- DAG: do not redefine `mo` / `torch` / `lpap`; use `_` locals; import training
  helpers once in setup.
- Cadences: e.g. `display_every=50`, `log_every=10`; checkpoint on improvement.

On-demand KPIs (local or inside `molab-exec`):

```bash
bash .github/skills/molab-workflow/scripts/molab-train-status.sh
# or:
bash .github/skills/molab-workflow/scripts/molab-exec.sh <<'PY'
from lpap.training_status import summarize_training_status
print(summarize_training_status(
    project_root="/marimo",
    checkpoint_name="image_autoencoder_multi_energy_bank.pt",
    log_name="image_autoencoder_multi_energy_bank.sqlite",
    run_id="image_autoencoder_multi_energy_bank",
    bg_stem="image_autoencoder_multi_energy_bank_bg",
))
PY
```

## Artifact sync (HF Storage Bucket)

Bucket settings come from [`configs/storage.toml`](../configs/storage.toml)
(`/artifacts.bucket`). On molab, `molab-sync.sh` copies that file to
`/marimo/configs/storage.toml` (upload/fetch raise if it is missing). Write
auth is `HF_TOKEN` only — from `configs/secrets.toml` via inject, or
`export HF_TOKEN=…` locally. Local download of public buckets needs no login.

**Training checkpoints** use a dual-slot HF layout so a mid-upload kill cannot
clobber the last good weights:

- `checkpoints/<stem>.slot0.pt` / `.slot1.pt` — alternating cold/hot objects
- `checkpoints/<stem>.current.json` — pointer (`slot`, `sha256`, `size`, optional `step`)

Local code and configs still use `checkpoints/<stem>.pt`. `ensure_checkpoint` /
resume **require** the pointer (missing pointer raises; no bare-`.pt` fallback).
Downloads use `download_bucket_files` / `get_bucket_paths_info` —
`HfFileSystem.exists`/`open` can stay stale for freshly written dual-slot keys.
SQLite under `training_logs/` remains a single best-effort key after the
checkpoint promote.

One-time migration of legacy bare HF objects:

```bash
# local or molab kernel
PYTHONPATH=src python -m lpap.artifact_sync migrate-checkpoints --project-root .
# optional: only some names
PYTHONPATH=src python -m lpap.artifact_sync migrate-checkpoints --project-root . \
  --checkpoint image_autoencoder_tri_bank_flow.pt
```

```python
# molab
from lpap.artifact_sync import upload_training_artifacts
upload_training_artifacts(
    "/marimo",
    checkpoint_names=("surrogate_synthetic.pt", "decoder_synthetic.pt"),
    log_names=("surrogate.sqlite", "decoder.sqlite"),
)
```

```bash
# local
pixi run artifacts-download
# or with extra names:
PYTHONPATH=src python -m lpap.artifact_sync download --project-root . \
  --checkpoint decoder_synthetic.pt --log decoder.sqlite
```

Viz notebooks resolve molab absolute `/marimo/checkpoints/...` log paths to
local `checkpoints/<name>`.

## Image dataset (separate bucket)

Training images: `images.bucket` / `images.remote_zst` from storage.toml
(public by default). Local or molab:

```bash
pixi run data-download
# or:
from lpap.dataset_fetch import ensure_image_tensor_archive
ensure_image_tensor_archive("/marimo")  # or project root
```

Caches `data/images_32x32_gray.pt` (skips if present). Do not confuse with
`lpap-molab-artifacts`.

## Training order

Bank curriculum (keep the loop): prior energies (harmonics or AE-encoded bank)
→ unpaired `image_energy_flow_energy_bank` → **re-encode** images via that
flow’s i2e → bank teachers (`surrogate`/`decoder`) → `image_autoencoder`.
Teachers train on the post-flow bank, not on the prior energies.
Teachers sample the energy bank (no image dataset); the AE needs images + flow
+ teacher checkpoints.

## Session checklist

**Human:** open lab notebook → attach GPU → pair → paste URL/token (leave
`MOLAB_SESSION` unset unless multi-session).  
**Agent (long AE):** `molab-sync` → `molab-launch-ae-energy-bank` (notify +
HF) → stop; human wakes agent on Pushover.
**Agent (short/shared):** visible code-mode cells only.  
**Local:** `artifacts-download` → viz notebooks.
