# Teacher permutation source-of-truth transition

## Context

Resume loss spiked because the AE session layout disagreed with teacher
checkpoints for **c256**: that surrogate/decoder pair still stores a legacy
(non–CPU-seed) `training_state.permutation`, while the running AE was trained
under the device-stable CPU-seed layout for seed `256`.

An interim salvage stored `lpap_pair_permutations` on the AE checkpoint and
resumed from that. Gallery cells create a *fresh* AE session
(`resume_from_checkpoint=False`), so they still load teacher perms — c256
gallery panels look wrong while GPU training (using the AE snapshot) looks fine.

## Rules (no backward compatibility)

- **Tensors only.** Seeds in TOML / `model_config` document how a fresh teacher
  was built; they are not used to reconstruct layouts when a checkpoint exists.
- **Surrogate and decoder each store** `training_state.permutation` (duplicated
  on purpose). For a teacher pair the two tensors **must match**.
- **AE does not store** a third copy. Drop `lpap_pair_permutations` from AE
  checkpoints.
- **AE create / resume / gallery:** for each pair, load both stored
  permutations, assert equality, use that layout. Symmetric — no preference
  for surrogate vs decoder.
- **AE metadata shape:** `model_config.lpap_pairs` / `training_state.lpap_pairs`
  use teacher-shaped records (`name` + `surrogate`/`decoder` configs, or
  checkpoint paths). Layout tensors stay on the teacher files.
- **Decoder training:** when a surrogate checkpoint is required, copy its
  stored permutation (no seed fallback) and write the same tensor into the
  decoder checkpoint.

## Phases

0. **Plan doc** — this file (replaces the AE-third-copy salvage plan).
1. **Revert overlay WIP** — done.
2. **Shared matching load** — helper + decoder load returns perm; remove
   decoder seed fallback when surrogate ckpt is required.
3. **AE** — always use matching teacher perms; remove AE
   `lpap_pair_permutations` save/resume/tests.
4. **Migrate CLI** — `migrate_teacher_permutations` rewrites teacher
   `training_state.permutation` without touching weights.
5. **HF + molab** — migrate/upload **all six** tri-pair teachers
   (`surrogate`/`decoder` × `c128_k16`, `c256_k24`, `c512_k32`) to CPU-seed
   layouts so the set is globally consistent; `molab-sync`; re-run gallery.
   Do not interrupt the healthy bg AE run.
6. **Gallery** — only after HF teachers match (no overlay shims).

## Historical salvage (completed, then superseded)

- AE `lpap_pair_permutations` + `migrate_ae_permutations` (`d3636dc`,
  `b15fcdf`): kept the 32k→70k resume healthy.
- SQLite cleanup + molab resume: done; train loss ≈0.008 after resume.

Those AE-side pieces are removed in phases 3–4 once teachers are the SoT.

## Teacher migration usage (phase 4+)

All six current-run teachers (preferred)::

```bash
pixi run migrate-teacher-permutations
# or:
PYTHONPATH=src python -m lpap.migrate_teacher_permutations --tri-pair --project-root .
```

Then upload each via artifact sync and force-download on molab before gallery.
