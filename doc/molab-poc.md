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

```bash
npx skills add marimo-team/marimo-pair
```

On molab: Actions → **Pair with an agent** → paste the prompt into
Cursor Agent (open the worktree). Ask the agent to re-run the smoke cells.
