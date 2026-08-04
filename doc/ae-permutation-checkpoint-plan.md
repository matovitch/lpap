# Teacher / AE permutation checkpoint policy

## Rules (no backward compatibility)

- **Tensors only in checkpoints.** Store concrete `training_state.permutation`
  (LongTensor). Do **not** store `permutation_seed` in checkpoint payloads —
  seeds live in TOML / SQLite run config if needed for archaeology.
- **Surrogate and decoder** each store `training_state.permutation`. For a
  teacher pair the two tensors **must match** at AE create time.
- **AE is self-contained:** `training_state.lpap_pairs[i]` holds
  `{name, permutation}` — no pretrained path pointers, no seed.
- **AE create:** load teacher weights + matching teacher perms from source
  config paths; session uses that layout.
- **AE resume:** layouts come from AE `lpap_pairs[*].permutation` (ignore
  teacher files for layout once the AE checkpoint exists).
- **Decoder training:** when a surrogate checkpoint is required, copy its
  stored permutation (no seed fallback) and write the same tensor into the
  decoder checkpoint.

## Phases

1. Package SoT + helpers — done (`69210f7` and follow-ups).
2. Self-contained AE pair permutations; strip seeds/paths from ckpt payloads.
3. **HF** — migrate all six tri-pair teachers; then molab-sync / gallery when
   safe vs the live run.
4. Gallery validation after teachers + new `lpap` are consistent.

## Teacher migration (salvage)

Rebuilds `training_state.permutation` from geometry + seed found in
`run_config.run.permutation_seed` (or legacy model_config). Strips seed fields
from the written payload.

```bash
pixi run migrate-teacher-permutations
```
