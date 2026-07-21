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

Push package/notebook changes on `molab-summer` and install that ref on molab.
Do not mix routine Pixi work into the molab worktree unless backporting.

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
| `notebooks/molab_lab.py` | Primary remote lab: install, CUDA smoke, chunked train by `model_kind` |
| Local `notebooks/train.py` | Pixi training UI on `main` |
| Local `notebooks/visualize_*.py` | Curves/galleries after `pixi run artifacts-download` |

Controls on the lab notebook: model kind, target steps, chunk steps,
`display_every`, `log_every`. Train logic stays in `src/lpap/`.

While paired, mutate the **live** notebook with `cm` (not the `.py` on disk).
Backport stable cell structure to `molab_lab.py` on `molab-summer` when useful.

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

## Shared training (code mode)

Put multi-minute trains in **visible cells** (`hide_code=False`, progress bar /
`mo.output.replace`) so the human sees the run. Scratchpad is for probes,
installs, and artifact sync only.

- Do not start a second `execute-code` while a train cell runs (`MarimoInterrupt`).
- Prefer chunked training + SQLite polls **between** chunks; resume with
  `resume_from_checkpoint=True`.
- DAG: do not redefine `mo` / `torch` / `lpap`; use `_` locals; import training
  helpers once in setup.
- Cadences: e.g. `display_every=50`, `log_every=10`; checkpoint on improvement.

Poll between chunks:

```python
import sqlite3
conn = sqlite3.connect("/marimo/training_logs/<model>.sqlite")
print(conn.execute("SELECT MAX(step) FROM step_metrics").fetchone()[0])
```

## Artifact sync (HF Storage Bucket)

Bucket: `matovitch/lpap-molab-artifacts` (public). Write token on molab via
`/marimo/.hf_token` or `HF_TOKEN` (gitignored). Local download needs no login.

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

## Training order

`surrogate` → `decoder` → image flows → reflow → `image_autoencoder`.
Surrogate/decoder need no image dataset (synthetic harmonics).

## Session checklist

**Human:** open lab notebook → attach RTX Pro 6000 → pair → paste URL/token.  
**Agent:** connect smoke → code-mode train in chunks → HF upload before idle.  
**Local:** `artifacts-download` → `pixi run notebook-surrogate` / `notebook-decoder`.
