# Roadmap: AE iteration 2 (bank-prior loop)

Status note (2026-08-02): tri-pair harmonics-prior AE
(`image_autoencoder_tri_flow`, pairs `c128_k16` + `c256_k24` + `c512_k32`)
was running well around step ~25k with good reconstruction. Finish / resume
that run first (target 70k; HF is SoT across molab 12h cuts).

## Goal

Run a **second curriculum loop** that is intentionally more detached from the
original **harmonics** flow prior: use the trained AE to define a new energy
marginal, pretrain flow + teachers on that bank, then train a fresh AE.

Unpaired bank-flow pretraining is the point — it will **not** recover the AE’s
weights; it only supplies a new interesting init for the next AE.

## After iteration-1 AE finishes

1. **Encode energy bank from AE best**
   - Encode the image dataset with the best AE i2e path (same discipline as
     before: `encode_image_dataset_to_energy_bank` / float normalized images,
     never raw uint8).
   - Save a new bank artifact (do not silently overwrite the old harmonics-era
     bank without renaming), e.g. under `data/` + HF artifacts bucket.
   - Smoke-check bank stats (mean/std) and a quick energy gallery.

2. **Re-pretrain bidirectional flow on the new bank (unpaired)**
   - Config pattern: `configs/training/image_energy_flow_energy_bank.toml`
     (bank energy marginal, independent of image batches).
   - New checkpoint stem (e.g. `image_energy_flow_energy_bank` or a dated
     iteration-2 name) so harmonics `image_energy_flow.pt` stays available.
   - Upload-on-improvement + notify, as usual on molab.

3. **Re-pretrain all three teachers from scratch on the new bank**
   - Fresh surrogate → decoder chains for `c128_k16`, `c256_k24`, `c512_k32`.
   - **Do not** reuse the AE’s finetuned surrogate/decoder weights (or the
     previous bank teachers) as the teacher init for this loop — avoid carrying
     minor AE-coupled biases into the next AE.
   - Same TOML geometry / order as iteration 1; new checkpoint names or a clear
     iteration suffix if keeping parallel artifacts.

4. **Train a full multi-pair AE (iteration 2)**
   - Init flows from the **new bank-pretrained** flow (not harmonics).
   - Teachers = the **new** bank teachers from step 3.
   - Fresh AE run stem (do not resume `image_autoencoder_tri_flow`).
   - Keep multi-batch val (`every` / `num_batches`) and HF upload/notify.

## Explicit non-goals for this loop

- Distilling or warm-starting teachers from the iteration-1 AE LPAP modules.
- Expecting bank-flow pretrain to match AE energy maps (unpaired by design).

## Pointers

- Curriculum sketch: [molab-workflow.md](molab-workflow.md) (Training order)
- Stack overview: [training-stack.md](training-stack.md)
- Iteration-1 bidir AE launcher: `molab-launch-ae-bidirectional-flow.sh`
  (currently tri-pair → `image_autoencoder_tri_flow.*`)
- Bank flow launcher: `molab-launch-flow-energy-bank.sh`
- Teacher launcher: `molab-launch-lpap-teacher.sh`
