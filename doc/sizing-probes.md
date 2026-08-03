# Sizing probes (capacity / integration)

See the [documentation index](index.md) and [training stack](training-stack.md).

Defaults in `configs/training/*.toml` were shaped around a consumer GPU
(e.g. RTX 5060-class). On larger GPUs (molab RTX Pro 6000), capacity and
Euler/integration budgets are likely under-sized. This note is the **manual
procedure** for exploring those knobs **before** building automated tuner
code.

## Goals

- Pick stage-local capacity (width, depth, heads, dilations, …) and, where
  relevant, integration step counts.
- Keep probes short, comparable, and disposable.
- Promote only a clear winner to `baseline`; retrain dependents when the
  upstream interface (weights / harmonic layout) changes.

## Principles

1. **Stage-local.** Size `surrogate` before `decoder`; size each flow before
   autoencoder. Do not retune the whole stack in one sweep.
2. **Freeze upstream.** When probing stage \(N\), pin upstream checkpoints
   (or a chosen reference pair).
3. **Short probes, long confirm.** Typical: **1–2k steps** to rank; **~10k**
   only for the winner (or a tight top-2).
4. **One primary axis (or a tiny grid).** Prefer a written \(2 \times 2\) /
   \(2 \times 3\) over open-ended search.
5. **Score quality and cost.** Primary val metric **and** step time / peak
   VRAM / parameter count.
6. **Human gallery veto** for flows (loss can look fine while maps are junk).
7. **Artifacts are disposable.** Tag probes `probe` + `size-grid`; never
   overwrite a pinned baseline until you intentionally promote.

## Per-stage defaults (starting grids)

| Stage | First grid | Hold fixed | Rank by |
| --- | --- | --- | --- |
| `surrogate` | `hidden_dim` ∈ {256, 512}, `layer_count` ∈ {8, 12} | energy bank, buckets, `k_max`, heads (must divide width) | val loss, weighted accuracy |
| `decoder` | match surrogate width/depth family if applicable | **surrogate checkpoint** | recon / decoder metrics |
| `image_to_energy` / `energy_to_image` | `width` ∈ {128, 192, 256}; optional `dilation_cycles` ∈ {2, 3} | sequence length, data | val FM loss + gallery |
| Integration | Euler / `teacher_steps` / `student_steps` | chosen width | quality vs step-count curve |
| `image_autoencoder` | batch, unroll steps | frozen or light FT pieces | image recon + look |

Adjust the numeric sets after the first pass; do not expand the grid until the
small one has a clear knee or a clear winner.

## Manual protocol

1. **Choose one stage** and write down the grid (cells ≤ ~6).
2. **Isolate artifacts:** distinct `run_id` / `checkpoint_name` per cell
   (shared probe SQLite is fine). Example:
   `surrogate_probe_h512_l8`, `surrogate_probe_h512_l8.pt`,
   log `surrogate_size_probes.sqlite`.
3. **Fix seed, step budget, val cadence** across cells.
   Use `resume_from_checkpoint=False` for fresh probes.
4. **Run** on molab (or local). Record ms/step and peak memory in the run
   `comment` when useful (free prose only).
5. **Table the results** (best val metric, last step, params, time, mem).
6. **Pick a default.** Optional 10k confirm on the winner only.
7. **Promote:** update TOML defaults; retrain **downstream**
   stages that depend on the new checkpoint.
8. **Keep v0 baselines** on the HF bucket until you deliberately replace them
   (rename or overwrite with intent).

## What tuner code should wait for

When we automate later, start from this contract:

- Declared grid: `model_kind`, axes, budget, primary metric, cost metrics.
- One run record per cell; a summary table (markdown/CSV).
- No auto-promote to `baseline` without a human (or an explicit gate + gallery
  check for flows).

Until then, prefer a one-off script or notebook cell that loops the grid and
prints the summary table.

## Surrogate pilot (first use of this procedure)

**Grid**

| `hidden_dim` | `layer_count` | `head_count` |
| --- | --- | --- |
| 256 | 8 | 8 |
| 256 | 12 | 8 |
| 512 | 8 | 8 |
| 512 | 12 | 8 |

**Budget:** 2000 steps each; validation every 100; same data/`k_max`
as the current default surrogate config.

**Artifacts:** checkpoints
`surrogate_probe_h{dim}_l{layers}.pt`, shared log
`training_logs/surrogate_size_probes.sqlite`, comment
`("probe", "size-grid", "surrogate")`.

**Do not overwrite** `surrogate_synthetic.pt` during the pilot.

After the four runs, compare best validation loss (and weighted accuracy if
logged). Promote only if a larger model clearly wins on quality without a
absurd cost jump; otherwise keep 256×8 as v0 and revisit after e2i exists.

### Pilot results (molab RTX Pro 6000, 2000 steps each)

| Cell | Params | Best val ↓ | Val WA ↑ | ms/step | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| **h256_l12** | 9.7M | **2.057** | **0.526** | 22.0 | 1.22 |
| h256_l8 (current default shape) | 6.6M | 2.265 | 0.491 | 16.8 | 0.85 |
| h512_l8 | 25.8M | 3.229 | 0.349 | 27.0 | 1.71 |
| h512_l12 | 38.4M | 3.387 | 0.344 | 37.4 | 2.52 |

Shared log: `training_logs/surrogate_size_probes.sqlite`. Baseline
`surrogate_synthetic.pt` was **not** overwritten.

**Reading:** under a fixed 2k budget, **depth helps more than width**. 512-d
cells look under-trained (worse loss, lower WA), not proven worse at full
horizon. VRAM is trivial on Pro 6000 for all four.

**Tentative next (still in this procedure):**

1. Optional **10k confirm** on `256×12` vs `256×8` (top-2), same seeds.
2. Only if 256×12 still wins, promote TOML to `layer_count=12`, retrain a full
   baseline, then **retrain decoder** (and later e2i).
3. Defer a longer 512-d sweep until after e2i pressure tests, or run a
   single 10k `512×8` as a stretch check.

### 10k confirm (256×8 vs 256×12)

Same seeds / data / cadence as the pilot; fresh runs (did not resume 2k
probes). Log: `surrogate_size_confirm.sqlite`; checkpoints
`surrogate_confirm_h256_l{8,12}.pt`.

| Cell | Best val ↓ | Best ckpt WA ↑ | ms/step | Peak GB | Wall (10k) |
| --- | ---: | ---: | ---: | ---: | ---: |
| h256_l8 | 0.348 | 0.903 | 15.6 | 0.85 | 156 s |
| **h256_l12** | **0.293** | **0.921** | 20.5 | 1.22 | 205 s |

**Decision:** promote **`hidden_dim=256`, `layer_count=12`** as the new
surrogate default. Cost is modest (~1.3× step time, still ≪ Pro 6000
budget).

**Promoted (2026-07-22):** `default_surrogate_training_config()` and later
teacher pair TOMLs use `layer_count=12`. On molab, baseline
`surrogate_synthetic.pt` was replaced by `surrogate_confirm_h256_l12.pt`
(best val ≈ 0.293, WA ≈ 0.921). Old 8-layer baseline kept only if still present
under a non-baseline name; decoder must be retrained against the new teacher.

## Decoder pilot

Freeze the promoted surrogate teacher. Sweep decoder capacity only:

| `hidden_dim` | `layer_count` | `head_count` |
| --- | --- | --- |
| 256 | 8 | 8 |
| 256 | 12 | 8 |
| 512 | 8 | 8 |
| 512 | 12 | 8 |

**Budget:** 2000 steps; validation every 100; comment
`("probe", "size-grid", "decoder")`; shared log
`decoder_size_probes.sqlite`; checkpoints
`decoder_probe_h{dim}_l{layers}.pt`. Do not overwrite
`decoder_synthetic.pt` during the pilot.

### Decoder pilot results (teacher = promoted surrogate l12, 2000 steps)

| Cell | Params | Best val ↓ | Last source CE | ms/step | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| **h256_l8** | 6.6M | **0.00661** | 0.670 | 21.4 | 0.98 |
| h256_l12 | 9.7M | 0.00715 | **0.633** | 26.0 | 1.36 |
| h512_l8 | 25.7M | 0.0745 | 2.35 | 29.7 | 1.83 |
| h512_l12 | 38.4M | 0.0740 | 2.34 | 39.4 | 2.65 |

**Reading:** 512-d is undertrained at 2k (same pattern as surrogate). Among
256-d cells, **8 layers edges val loss**; 12 layers edges the CE regularizer
term. Gap is small → **10k confirm** on `256×8` vs `256×12` before promoting.

### Decoder 10k confirm (teacher = promoted surrogate l12)

| Cell | Best val ↓ | Ckpt source CE | ms/step | Peak GB | Wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| **h256_l8** | **0.00518** | 0.505 | 20.4 | 0.98 | 204 s |
| h256_l12 | 0.00532 | **0.473** | 25.2 | 1.36 | 252 s |

**Decision:** keep decoder **`hidden_dim=256`, `layer_count=8`**. Primary
monitor (`validation_loss`) prefers 8 layers; CE is only a regularizer. Cost
also favors 8. Promote `decoder_confirm_h256_l8.pt` → `decoder_synthetic.pt`
so the baseline matches the new surrogate teacher (old decoder is stale).
