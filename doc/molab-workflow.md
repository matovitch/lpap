# Molab remote GPU workflow (agents)

Drive free [molab](https://molab.marimo.io) GPUs from a local agent via
[marimo-pair](https://github.com/marimo-team/marimo-pair). Keep the original
Pixi checkout on `main` for local work.

## Topology

```text
Laptop agent (Cursor/Zed)     molab sandbox
  worktree lpap-molab    →      one paired notebook
  marimo-pair skill      →      HTTP + token (no SSH)
  git push molab-summer  →      uv/pip install lpap from git
```

| Role | Path / branch |
| --- | --- |
| Local Pixi home | `../lpap` on `main` |
| Molab / agent home | `../lpap-molab` on `molab-summer` |
| Pair skill | `~/.cursor/skills/marimo-pair` |
| Molab wrapper | `.github/skills/molab-workflow/scripts/molab-exec.sh` |

Push package/notebook changes on `molab-summer` and install that ref on molab.
Do not mix routine Pixi work into the molab worktree unless backporting.

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
Backport stable cell structure to `molab_lab.py` on `molab-summer` when useful.

**Cell naming:** durable lab cells get a marimo name + `# cell: <name>` first
line (`ae_setup`, `status`, `gallery_cache`, …). Talk about cells by name, not
by opaque ids. See `.github/skills/molab-workflow/SKILL.md`.

## Package install

```bash
python -m pip install --no-deps \
  "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
python -m pip install "jaxtyping>=0.3.7"
```

`molab-summer` relaxes `requires-python` for molab; `main` may stay stricter.
`lpap` is not on PyPI — do not `uv add lpap==0.1.0`. After pulling new commits,
`--force-reinstall` the git ref if imports are missing.

Sandbox artifacts: `/marimo/checkpoints/*.pt`, `/marimo/training_logs/*.sqlite`
(idle ~90 min / session ~12 h).

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
Agent-side waits use the harness sleep/poll loop (`AwaitShell` + short
`molab-train-status`); do not `sleep` in the kernel.

**Short / shared chunks only:** visible code-mode cells (`hide_code=False`,
progress bar / `mo.output.replace`) when the human should watch the run.
Scratchpad is for probes, installs, and artifact sync — not long training.

- Do not start a second `execute-code` while a train cell runs (`MarimoInterrupt`).
- For cell chunks: status polls **between** chunks; resume with
  `resume_from_checkpoint=True`.
- DAG: do not redefine `mo` / `torch` / `lpap`; use `_` locals; import training
  helpers once in setup.
- Cadences: e.g. `display_every=50`, `log_every=10`; checkpoint on improvement.

Poll KPIs (local or inside `molab-exec`):

```bash
bash .github/skills/molab-workflow/scripts/molab-train-status.sh
# or:
bash .github/skills/molab-workflow/scripts/molab-exec.sh <<'PY'
from lpap.training_status import summarize_training_status
print(summarize_training_status(
    project_root="/marimo",
    checkpoint_name="image_autoencoder_energy_bank.pt",
    log_name="image_autoencoder_energy_bank.sqlite",
    run_id="image_autoencoder_energy_bank",
    bg_stem="image_autoencoder_energy_bank_bg",
))
PY
```

## Artifact sync (HF Storage Bucket)

Bucket settings come from [`configs/storage.toml`](../configs/storage.toml)
(`/artifacts.bucket`). On molab, `molab-sync.sh` copies that file to
`/marimo/configs/storage.toml` (upload/fetch raise if it is missing). Write
auth is `HF_TOKEN` only — from `configs/secrets.toml` via inject, or
`export HF_TOKEN=…` locally. Local download of public buckets needs no login.

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

`surrogate` → `decoder` → image flows → `image_autoencoder`.
Surrogate/decoder need no image dataset (synthetic harmonics).

## Session checklist

**Human:** open lab notebook → attach GPU → pair → paste URL/token (leave
`MOLAB_SESSION` unset unless multi-session).  
**Agent (long AE):** `molab-sync` → `molab-launch-ae-energy-bank` → poll
`molab-train-status` (Pushover + HF upload-on-checkpoint).  
**Agent (short/shared):** visible code-mode cells only.  
**Local:** `artifacts-download` → viz notebooks.
