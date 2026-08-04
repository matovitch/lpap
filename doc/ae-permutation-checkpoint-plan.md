# AE permutation checkpoint fix

## Context

Resume loss spiked because the AE did not store the LPAP permutations it trained
under. On create it reloaded teacher `training_state.permutation` (c256 teacher =
legacy CUDA-PRNG layout) while the weights expected CPU-seed layout.

## Rules (no backward compatibility)

- **Tensors only** for AE pair layouts: never use `permutation_seed` /
  `make_grouped_permutation_indices` as an AE session source of truth.
- **Fresh AE run:** each pair’s permutation **must** come from the teacher
  surrogate checkpoint’s stored `training_state.permutation` (raise if missing).
- **Resume AE run:** permutations **must** come from the AE checkpoint’s
  `training_state.lpap_pair_permutations` (raise if missing / wrong count / wrong
  length). Do not fall back to teacher or seed.
- **Save:** every AE checkpoint write includes `lpap_pair_permutations` (CPU
  `LongTensor`s, same order as `lpap_pair_names`).

Seeds may remain in teacher/model configs as documentation of how a teacher was
originally built; they are not used to reconstruct AE session permutations.

## Phases

1. **Checkpoint save/load (this PR)** — implement rules above + unit tests.
2. **Migration script** — rewrite `image_autoencoder_tri_lnorm.pt` training_state
   with the concrete perms the weights need (CPU-seed regen for this salvage),
   dump/upload.
3. **SQLite cleanup** — drop failed-resume / probe noise after step 32000.
4. **Molab resume** — sync new `lpap` + migrated artifacts; resume to 70k.

## Step 1 touch list

- `src/lpap/image_autoencoder_training.py`
- `test/test_image_autoencoder_training.py`
