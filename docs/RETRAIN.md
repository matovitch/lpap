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
`training_state`. Pair loads require matching `value_count`, `bucket_count`,
`probe_count`, and `k_max` (and `fold_order` / `frontend_mode` when those
fields are present).

Surrogate blocks use periodic circular RoPE (frequencies `2π m / N` on `N = C`
tokens) and SDPA. The cosine/sine tables are non-persistent buffers, so a
checkpoint trained on another frequency schedule loads and then runs under
the live tables. This pipeline trains surrogates and the autoencoder from
scratch. Decoder source-index CE weights are `w_b = |a_b|.detach()` with
`a_b` the soft frontend amplitude (`source_ce_weight = 0.1`,
`source_ce_l1_reference = 0.05`, `source_ce_power = 2` in the teacher TOMLs).

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

`molab-sync` copies this checkout's `configs/training/` into `/marimo`.
The lab setup cell then calls `lpap.training_configs.sync_training_configs`
and replaces those TOMLs with the archive of the installed lpap commit.

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

## Scheduled budget

Teacher and AE stages use batch size 128 (`N = 1024`). Teacher learning rates
are `1.0e-3`. The AE learning rate is `4.0e-5`. Exact-LPAP fine-tune uses
`1.0e-4`. Cadence is `display_every = 25`, `log_every = 5`. Validation batch
size is 256 for teachers and 32 for the AE.

| Stage | Steps | What runs |
| --- | --- | --- |
| `surrogates` (`c128_k16`) | 7,500 | 12-layer circular-RoPE transformer, `C=128`, local 128-class head, SDPA |
| `surrogates` (`c256_k24`) | 7,500 | `C=256`, local 96-class head |
| `surrogates` (`c512_k32`) | 7,500 | `C=512`, local 64-class head |
| `decoders` (`c128`, `c256`, `c512`) | 3 × 7,500 | Frozen surrogate + 8-layer decoder (`1024`-class output) |
| `autoencoder` (tri-pair) | 17,500 | 4-step `i2e` + 3 surrogate/decoder pairs + 3× 4-step `e2i` |
| `decoder_exact_finetune` | 750 | Frozen 4-step `i2e` + exact `lpap_tokens_triton` + trainable decoders + frozen `e2i` |

7,500 teacher steps at batch 128 is 960k samples. 17,500 AE steps is 2.24M
images. Exact fine-tune is 750 steps at the AE batch.

`[performance]` in `retrain_image_pipeline.toml` (`PipelinePerformanceConfig`):

- `autocast_dtype = "bf16"` wraps every stage in
  `torch.autocast(dtype=bfloat16, cache_enabled=False)`. The cache stays off
  because one context covers the whole stage. CUDA bf16 requires
  `torch.cuda.is_bf16_supported()`; any other non-CPU device raises before
  stages start.
- `allow_tf32` enables TF32 matmul and cuDNN.
- `energy_bank_on_device` keeps the teacher bank on the training device.
- `compile_flows` wraps each AE flow `forward` with `torch.compile`.

Both flow directions integrate in fp32 inside that autocast.

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
   - RMS: `validation_decoded_energy_rms` and `validation_reconstructed_image_rms` are the mean across pairs. Each pair also logs `validation_decoded_energy_rms/<name>` and `validation_reconstructed_image_rms/<name>`. `validation_image_rms` is the shared input RMS.
   - Visual check in `molab/lab.py` (`ae_gallery` and `ae_energy_latent`: oracle LPAP vs surrogate-hard vs decoder-soft energy and reconstructed images).
4. **Exact-LPAP decoder fine-tune (`image_autoencoder_tri_exact_ft-v2.sqlite`)**
   - Compare Stage-4 `validation_image_reconstruction_l2` and `validation_energy_reconstruction_l1` (using exact LPAP with `entropy = 0`) directly against Stage-3 surrogate-frontend validation metrics and Stage-3 evaluated at step 0 under exact LPAP (measures closure of the surrogate-to-exact train-test gap).
