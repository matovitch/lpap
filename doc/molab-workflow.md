# Molab remote GPU workflow (agents)

Summer workflow: drive free [molab](https://molab.marimo.io) RTX Pro 6000
compute from a local agent via [marimo-pair](https://github.com/marimo-team/marimo-pair),
while keeping the original Pixi checkout on `main` for local work.

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

Do not mix day-to-day Pixi edits into the molab worktree unless you intend to
backport. Prefer pushing package/notebook changes on `molab-summer` and
installing that ref on molab.

## What the agent can and cannot do on molab

**Can (once paired):**

- Execute scratchpad Python in the live kernel (`execute-code.sh --url …`)
- Create / edit / run / delete **cells** via `marimo._code_mode` (`cm`)
- Install packages the way the notebook already does (pip/uv in-kernel)
- Read checkpoints, SQLite logs, and globals for progress
- Toast the UI (`mo.status.toast`)

**Cannot:**

- Open a new molab notebook session or attach a GPU (browser UI only)
- SSH into the sandbox
- Pair reliably to several notebooks at once (one active pair URL/token)
- Install conda packages from a prefix.dev channel with `uv` (molab is
  uv/pip; publish wheels or `git+https` instead)

**Human bottleneck:** create or open the molab notebook, attach GPU, click
**Pair with an agent**, paste URL + token into the agent. After that the agent
should stay inside that one notebook.

## Notebook policy (keep the set small)

Prefer **one long-lived generic lab notebook** on molab, not a fleet of
one-off notebooks.

| Notebook | Purpose |
| --- | --- |
| `notebooks/molab_lab.py` | **Primary.** Smoke + install + generic chunked training by `model_kind` (same idea as local `notebooks/train.py`) |
| `notebooks/molab_poc.py` | Optional minimal CUDA/`lpap_torch` smoke; fold into lab when convenient |
| Local `notebooks/train.py` | Unchanged Pixi workflow on `main` |

`molab_lab.py` controls: model kind, target steps, **chunk steps**, `display_every`,
`log_every`. Each run of the train cell advances at most one chunk and resumes
from the checkpoint, so agents can poll SQLite between chunks.


Rules:

1. One molab session ↔ one pair token. Reuse the same notebook across models.
2. Mutate the **live** notebook with `cm`, not the `.py` on disk, while paired.
3. After a good session, backport durable cell structure into
   `notebooks/molab_lab.py` on `molab-summer` and push so the next molab open
   is not empty archaeology.
4. Keep training logic in `src/lpap/`; notebook cells should select
   `model_kind`, tweak run knobs, call `create_training_session` /
   `iter_training`, and render progress.

## Package install on molab

Molab preinstalls torch. Install the project with git and skip resolving torch:

```bash
python -m pip install --no-deps \
  "lpap @ git+https://github.com/matovitch/lpap.git@molab-summer"
python -m pip install "jaxtyping>=0.3.7"
```

Notes:

- `requires-python` on `molab-summer` is relaxed to `>=3.11` so molab’s Python
  can install the package (`main` may stay stricter for local Pixi).
- Auto-`uv add lpap==0.1.0` from PyPI will fail (package is not on PyPI). Prefer
  the git URL above, or import from an already-installed environment.
- prefix.dev / Pixi conda channels are for **local** Pixi, not molab uv.

Artifacts live under the sandbox cwd (typically `/marimo`):

- `checkpoints/*.pt`
- `training_logs/*.sqlite`

Download them before idle shutdown (~90 minutes idle, ~12 hour max session).

## Pairing and code-mode gotchas (lessons from the surrogate run)

### Do not parallel-pair during a long cell

`execute-code.sh` shares the kernel. A second pair request while a training
cell is running can raise `MarimoInterrupt` and kill the loop. Progress may
still be on disk (SQLite / checkpoint) if `TrainingRun` flushed earlier.

**Safe patterns:**

1. **Single long call** — one `execute-code` runs setup + train and blocks
   until the cell finishes; await that process only.
2. **Chunked training (preferred for agents)** — train N steps (e.g. 500–2000),
   return, then scratchpad-query SQLite (`MAX(step)`, latest loss), then start
   the next chunk. Resume uses `resume_from_checkpoint=True`.

### How to poll progress

From scratchpad (between chunks, or after a run):

```python
import sqlite3
conn = sqlite3.connect("/marimo/training_logs/surrogate.sqlite")
print(conn.execute("SELECT MAX(step) FROM step_metrics").fetchone()[0])
print(conn.execute(
    "SELECT step, metric_name, metric_value FROM step_metrics "
    "ORDER BY step DESC, metric_name LIMIT 12"
).fetchall())
```

Also inspect `/marimo/checkpoints/<name>.pt` (`step`, `best_metric`).

Cell status via code mode (`ctx.cells[cid].status`) shows idle/stale/running
but does not stream per-step metrics by itself.

### Marimo DAG rules when adding cells

- Do not re-import public names already defined (`mo`, `torch`, `lpap`).
- Use leading-underscore names for cell-local temporaries.
- Pass `hide_code=False` on `create_cell` so humans can see agent-added cells.
- Training helpers are lazy exports: prefer
  `from lpap.training_notebook import …` inside a setup cell once.

### Cadences

Follow the same policy as local training: do not
`mo.output.replace` / log / checkpoint every step. On molab defaults that
worked well: `display_every=50`, `log_every=10`, checkpoint on validation
improvement.

## Surrogate trial (reference outcome)

- Model order starts at **surrogate** (synthetic harmonics only; no image data).
- Run on molab: 10k steps, CUDA, checkpoint
  `/marimo/checkpoints/surrogate_synthetic.pt`.
- Interruptions mid-cell were recoverable via resume from checkpoint + SQLite
  `max(step)`.
- Approximate final: train loss ~0.34, best validation loss ~0.34, validation
  weighted accuracy ~0.90.

Next model in dependency order: **decoder** (teacher = surrogate checkpoint).

## Human checklist (each molab session)

1. Open the single lab notebook on molab (GitHub sync from `molab-summer` or
   reopen the same molab URL).
2. Attach **RTX Pro 6000**.
3. Pair with agent → paste URL + token into the agent on the `lpap-molab`
   worktree.
4. Agent: verify CUDA + `import lpap`, then train/eval inside that notebook.
5. Before leaving: download checkpoints (and logs if useful); avoid 90+ minutes
   idle if a run should continue.

## Agent checklist

1. Confirm pair URL/token; smoke `print("connected")`.
2. Avoid starting a second `execute-code` while a train cell is active.
3. Prefer chunked training + SQLite polls.
4. Persist durable notebook structure back to git on `molab-summer` when the
   lab cells stabilize.
5. Do not change the `main` worktree Pixi workflow unless backporting
   deliberately.
