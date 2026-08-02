# Roadmap: AE iteration 2 (bank-prior loop)

Status note (2026-08-02 night): iteration-1 tri-pair AE stopped early; durable
HF ckpt **step 47000**, `best_metric≈0.00838`
(`image_autoencoder_tri_flow.{pt,sqlite}`). **Step 1 done:** encoded
`data/encoded_energies_tri_flow_best.pt` from AE-best i2e (verified
`source_step=47000`; local alias `encoded_energies_ae_best.pt`). **Next:** step 2
bank-flow pretrain.

## Goal

Close the same loop as the harmonics curriculum, with the AE-encoded bank
playing the role harmonics used to play as a *prior* energy marginal:

```text
prior energies  →  unpaired bank-flow pretrain  →  re-encode images via new i2e
                →  fresh teachers on that bank  →  new AE
```

Unpaired bank-flow pretraining will **not** recover the previous AE’s weights;
it only supplies a new interesting flow init. Teachers train on the
**post-flow** bank (re-export), not on the AE bank directly — same pattern as
“don’t train teachers on raw harmonics; train on energies after flow
pretraining.”

## After iteration-1 AE finishes

1. **Encode energy bank from AE best** (prior bank for this loop)
   - Encode the image dataset with the best AE i2e path
     (`encode_image_dataset_to_energy_bank` / float normalized images, never
     raw uint8).
   - Save a distinct artifact (do not silently overwrite an older bank), e.g.
     `data/encoded_energies_tri_flow_best.pt` + HF.
   - Smoke-check bank stats (mean/std) and a quick energy gallery.

2. **Re-pretrain bidirectional flow on that prior bank (unpaired)**
   - Config pattern: `configs/training/image_energy_flow_energy_bank.toml`
     (bank energy marginal, independent of image batches).
   - New checkpoint stem (e.g. `image_energy_flow_energy_bank` or a dated
     iteration-2 name) so harmonics `image_energy_flow.pt` stays available.
   - Upload-on-improvement + notify, as usual on molab.

3. **Re-encode energy bank through the new flow’s i2e**
   - Same encode discipline as step 1, but weights = **best** i2e from step 2.
   - Size for teachers: with epoch-style bank iteration, ~`steps × batch_size`
     rows is enough (e.g. 15k×32 = 480k); full corpus is optional.
   - New bank name (e.g. `data/encoded_energies_bank_flow_best.pt`); keep the
     AE prior bank around for forensics / ablations.
   - Smoke-check stats again before teachers.

4. **Re-pretrain all three teachers from scratch on the post-flow bank**
   - Fresh surrogate → decoder chains for `c128_k16`, `c256_k24`, `c512_k32`.
   - **Do not** reuse the AE’s finetuned surrogate/decoder weights (or the
     previous bank teachers) as the teacher init for this loop.
   - Point `[data.energy_bank]` at the step-3 bank, not the AE prior bank.
   - New checkpoint names or a clear iteration suffix if keeping parallel
     artifacts.

5. **Train a full multi-pair AE (iteration 2)**
   - Init flows from the **step-2 bank-pretrained** flow (not harmonics).
   - Teachers = the **step-4** teachers.
   - Fresh AE run stem (do not resume `image_autoencoder_tri_flow`).
   - Keep multi-batch val (`every` / `num_batches`) and HF upload/notify.

## Explicit non-goals for this loop

- Distilling or warm-starting teachers from the iteration-1 AE LPAP modules.
- Expecting bank-flow pretrain to match AE energy maps (unpaired by design).
- Training teachers directly on the AE-encoded prior bank (step 1) — that bank
  is only the flow’s energy marginal, analogous to harmonics.

## Pointers

- Curriculum sketch: [molab-workflow.md](molab-workflow.md) (Training order)
- Stack overview: [training-stack.md](training-stack.md)
- Iteration-1 bidir AE launcher: `molab-launch-ae-bidirectional-flow.sh`
  (currently tri-pair → `image_autoencoder_tri_flow.*`)
- Bank flow launcher: `molab-launch-flow-energy-bank.sh`
- Teacher launcher: `molab-launch-lpap-teacher.sh`
