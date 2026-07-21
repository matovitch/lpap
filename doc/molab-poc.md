# Molab POC (summer worktree)

Sibling worktree for free molab GPU runs. Local Pixi workflow stays on
`main` in the original checkout.

**Full agent workflow (pairing, polling, notebook policy, lessons):**
see [molab-workflow.md](molab-workflow.md).

## Layout

- Worktree: `../lpap-molab` (branch `molab-summer`)
- Primary remote notebook (target): `notebooks/molab_lab.py`
- Smoke notebook: `notebooks/molab_poc.py`
- Package install on molab uses git, not Pixi / prefix.dev

## Push

```bash
cd ../lpap-molab
git push -u origin molab-summer
```

## Molab steps

1. Open [molab](https://molab.marimo.io) and create or GitHub-sync the lab
   notebook from branch `molab-summer` (prefer one durable notebook URL).
2. Attach an **RTX Pro 6000** via notebook specs.
3. Run install/smoke cells (install
   `lpap @ git+https://github.com/matovitch/lpap.git@molab-summer`
   with `--no-deps`, then `jaxtyping`, if needed).
4. Actions → **Pair with an agent** → paste URL + token into the agent on the
   `lpap-molab` worktree.

## Cursor pairing

Skill installed locally at `~/.cursor/skills/marimo-pair`
(clone of [marimo-team/marimo-pair](https://github.com/marimo-team/marimo-pair);
`npx` was not available on this machine). Project notes:
`.github/skills/molab-workflow/SKILL.md`.
