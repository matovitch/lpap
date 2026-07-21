# Molab POC (summer worktree)

Sibling worktree for free molab GPU runs. Local Pixi workflow stays on
`main` in the original checkout.

## Layout

- Worktree: `../lpap-molab` (branch `molab-summer`)
- Notebook: `notebooks/molab_poc.py`
- Package install on molab uses git, not Pixi / prefix.dev

## Push

```bash
cd ../lpap-molab
git push -u origin molab-summer
```

## Molab steps

1. Open [molab](https://molab.marimo.io) and create or GitHub-sync
   `notebooks/molab_poc.py` from branch `molab-summer`.
2. Attach an **RTX Pro 6000** via notebook specs.
3. Run all cells (first cell installs
   `lpap @ git+https://github.com/matovitch/lpap.git@molab-summer`
   with `--no-deps` if needed).
4. Confirm Status **PASS**.

## Cursor pairing

Skill installed locally at `~/.cursor/skills/marimo-pair`
(clone of [marimo-team/marimo-pair](https://github.com/marimo-team/marimo-pair);
`npx` was not available on this machine).

On molab: Actions → **Pair with an agent** → paste the prompt into
Cursor/Zed Agent (open the `lpap-molab` worktree). Ask the agent to re-run
the smoke cells.
