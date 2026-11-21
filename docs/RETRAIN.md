# Image-Only LPAP Retraining Pipeline

This document specifies the end-to-end retraining order, configs, entry points,
runtime estimates, and comparison metrics for the 32x32 grayscale image
autoencoder stack using the unified column-major LPAP operator and local
`(dib, probe)` surrogate logits.

## Reusable Upstream Artifacts

Both upstream flow artifacts are independent of LPAP, permutation layout, fold
order, surrogates, and decoders, so they are kept as-is:

| Artifact | Keep? | Reason |
| --- | --- | --- |
| `checkpoints/image_energy_flow.pt` | **Keep** | Pure bidirectional image <-> energy flow (`DilatedConvFlow1d`); does not touch LPAP. |
| `data/encoded_energies_bank_flow_best_4step.pt` | **Keep** | Offline 4-step encodings produced directly from `image_energy_flow.pt` and `data/images_32x32_gray.pt`. |
| `data/images_32x32_gray.pt` | **Keep** | Source 32x32 grayscale image dataset. |

All three teacher surrogates, all three teacher decoders, and the tri-pair image
autoencoder train from scratch (`resume_from_checkpoint = false`).

## Stage Order And Configs

The master pipeline config is
[`configs/training/retrain_image_pipeline.toml`](../configs/training/retrain_image_pipeline.toml).
It executes four ordered stages:

| Order | Stage | Config(s) | Outputs (`checkpoints/` & `training_logs/`) |
| --- | --- | --- | --- |
| 1 | `surrogates` (x3) | `configs/training/teacher_c128_k16.toml`, `teacher_c256_k24.toml`, `teacher_c512_k32.toml` | `surrogate_c128_k16-v2.{pt,sqlite}`, `surrogate_c256_k24-v2.{pt,sqlite}`, `surrogate_c512_k32-v2.{pt,sqlite}` |
| 2 | `decoders` (x3) | Same 3 `teacher_*.toml` files (each loads its fresh `-v2` surrogate checkpoint + stored `permutation` and `fold_order`) | `decoder_c128_k16-v2.{pt,sqlite}`, `decoder_c256_k24-v2.{pt,sqlite}`, `decoder_c512_k32-v2.{pt,sqlite}` |
| 3 | `autoencoder` | `configs/training/image_autoencoder.toml` (tri-pair `c128_k16` + `c256_k24` + `c512_k32` using `-v2` teachers + existing `image_energy_flow.pt`) | `image_autoencoder_tri_lnorm-v2.{pt,sqlite}` |
| 4 | `decoder_exact_finetune` (optional) | `[decoder_exact_finetune]` in `configs/training/retrain_image_pipeline.toml` (loads Stage-3 `-v2` AE, freezes `i2e`/`surrogates`/`e2i`, trains only `decoders` on exact LPAP `(a, normalized_dib, 0)`) | `image_autoencoder_tri_exact_ft-v2.{pt,sqlite}` |

All newly trained artifacts carry the `-v2` suffix so HF checkpoint uploads
never overwrite prior checkpoints. Each saved checkpoint records
`fold_order = "column_major"` and its concrete `permutation` tensor(s) in
`training_state`.

## Entry Points

### Local / Direct CLI

```bash
# Full pipeline (all 4 stages)
pixi run retrain-image-pipeline

# Equivalent direct Python command (with optional stage subset or overrides)
PYTHONPATH=src python -m lpap.retrain_image_pipeline \
  --project-root . \
  --config configs/training/retrain_image_pipeline.toml \
  --stages surrogates,decoders,autoencoder,decoder_exact_finetune
```

Useful flags:

- `--stages`: comma-separated subset of `surrogates,decoders,autoencoder,decoder_exact_finetune`.
- `--resume`: resume checkpoints instead of starting from scratch.
- `--no-upload` / `--no-notify`: disable HF checkpoint upload or Pushover notification (for local runs).
- `--surrogate-steps`, `--decoder-steps`, `--ae-steps`, `--exact-ft-steps`: override step counts per stage without editing TOML.

### Molab Detached GPU Worker

After syncing the repo and `configs/training/` into `/marimo` (`molab-sync`):

```bash
python -m molab.jobs launch-retrain-image-pipeline \
  --project-root /marimo \
  --config configs/training/retrain_image_pipeline.toml
```

This spawns a detached Python worker (`training_logs/run_retrain_image_pipeline-v2_bg.py`),
tracks its PID/log under `training_logs/retrain_image_pipeline-v2_bg.{pid,log}`,
uploads checkpoints to HF storage on validation improvement, and sends a
Pushover summary when the pipeline completes.

To run stages individually on molab (e.g. if splitting across sessions or
running the 3 teacher pairs in parallel on separate GPUs), pass `--stages`:

```bash
python -m molab.jobs launch-retrain-image-pipeline --project-root /marimo --stages surrogates,decoders
python -m molab.jobs launch-retrain-image-pipeline --project-root /marimo --stages autoencoder
python -m molab.jobs launch-retrain-image-pipeline --project-root /marimo --stages decoder_exact_finetune
```

## Expected GPU Runtime Estimates

Rough single-GPU wall-clock estimates on a molab L4 / A10G-class GPU (FP32,
batch size 32, `N = 1024`; expect ~1.5-2x longer on T4, ~0.4-0.6x on A100):

| Stage | Steps | Dominant cost | Rough GPU estimate |
| --- | --- | --- | --- |
| `surrogates` (`c128_k16`) | 15,000 | 12-layer RoPE transformer over `C=128`, local `128`-class head + sync-free `lpap_tokens_torch` | ~8-15 min |
| `surrogates` (`c256_k24`) | 15,000 | 12-layer RoPE transformer over `C=256`, local `96`-class head | ~15-25 min |
| `surrogates` (`c512_k32`) | 15,000 | 12-layer RoPE transformer over `C=512`, local `64`-class head | ~30-50 min |
| `decoders` (`c128`, `c256`, `c512`) | 3 x 15,000 | Frozen surrogate forward + 8-layer decoder transformer (`1024`-class output) | ~45-80 min total |
| `autoencoder` (tri-pair) | 70,000 | 4-step `i2e` + 3 surrogate/decoder pairs + 3x 4-step `e2i` per step | ~2-4 hr |
| `decoder_exact_finetune` | 3,000 | Frozen 4-step `i2e` + exact `lpap_tokens_triton` + trainable decoders + frozen `e2i` | ~5-10 min |

Total sequential wall time on one L4/A10G: roughly 4-6 hours (dominated by the
70k-step tri-pair AE stage).

## Metrics To Compare Against The Prior Run

Inspect SQLite logs (`training_logs/*.sqlite`) and `molab/lab.py` galleries to
evaluate the `-v2` checkpoints:

1. **Surrogates (`surrogate_c{128_k16,256_k24,512_k32}-v2.sqlite`)**
   - `validation_loss` (cross-entropy over `k_max * P` local classes: 128, 96, 64).
   - `validation_weighted_accuracy` and `validation_accuracy`.
2. **Decoders (`decoder_c{128_k16,256_k24,512_k32}-v2.sqlite`)**
   - `validation_reconstruction_l1` (primary decoder KPI).
   - `validation_weighted_accuracy`, `validation_source_ce`, and `validation_mean_entropy`.
   - Learned frontend temperature `decoder.frontend_temperature()`.
3. **Tri-pair AE (`image_autoencoder_tri_lnorm-v2.sqlite`)**
   - `validation_image_reconstruction_l2` overall and per pair (`c128_k16`, `c256_k24`, `c512_k32`).
   - `validation_energy_reconstruction_l1` overall and per pair.
   - `validation_surrogate_teacher_ce` and `validation_weighted_accuracy` per pair.
   - Signed-mass health: `validation_encoded_positive_mass`, `validation_encoded_negative_mass`, `validation_signed_mass_imbalance`, and `validation_encoded_energy_rms` (confirm encoded energies stay bipolar near `floor_tau = 0.1` rather than collapsing toward 0).
   - Visual check in `molab/lab.py` (`ae_gallery` and `ae_energy_latent`: oracle LPAP vs surrogate-hard vs decoder-soft energy and reconstructed images).
4. **Exact-LPAP decoder fine-tune (`image_autoencoder_tri_exact_ft-v2.sqlite`)**
   - Compare Stage-4 `validation_image_reconstruction_l2` and `validation_energy_reconstruction_l1` (using exact LPAP with `entropy = 0`) directly against Stage-3 surrogate-frontend validation metrics and Stage-3 evaluated at step 0 under exact LPAP (measures closure of the surrogate-to-exact train-test gap).
