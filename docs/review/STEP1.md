# LPAP Code Review — STEP 1 of 3

Scope: (a) repository architecture map and (b) correctness audit of the LPAP operator and its direct interfaces (`ops.py`, `triton_ops.py`, `permutation.py`, `surrogate.py`, `decoder.py`), with an eye toward extending the codebase to LPAP bottlenecks on frozen LLM hidden states / embeddings (`(B, T, D)` tensors, small residual/flow AE, multiple `C` branches, next-token KL objective).

No existing source files are modified in this step. New artifacts on this branch: this report (`docs/review/STEP1.md`) and `test/test_lpap_review_step1.py`.

---

## (a) Architecture Map

### 1. Repository Layout & Modules

| Area | Files | Role |
| --- | --- | --- |
| **Core LPAP operator** | `src/lpap/ops.py`, `src/lpap/triton_ops.py`, `src/lpap/permutation.py` | Reference PyTorch LPAP (`lpap_torch`), CUDA Triton kernel (`lpap_triton`), CPU-seeded permutation + fold/unfold helpers. |
| **Learned LPAP bottleneck** | `src/lpap/transformer.py`, `src/lpap/surrogate.py`, `src/lpap/decoder.py` | RoPE `TransformerBlock`, `LPAPSurrogateTransformer` + duplicated Python LPAP target builder (`lpap_surrogate_targets`), decoder frontend (`prepare_lpap_decoder_batch`) + `LPAPDecoderTransformer`. |
| **Image & flow domain** | `src/lpap/hilbert.py`, `src/lpap/data.py`, `src/lpap/flow.py` | 2D Hilbert curve flatten/unflatten (`32x32 -> 1024`), uint8 `ImageTensorDataset` + `float_image_batch`, 1D dilated Conv1d flow (`DilatedConvFlow1d`) + bidirectional flow matching + Euler midpoint integrator. |
| **Empirical energy bank** | `src/lpap/energy_bank.py`, `src/lpap/energy_bank_encode.py` | Offline `(n, energy_dim)` energy bank storage/validation/epoch cycling and i2e flow encoder with scale guards. |
| **Training stacks** | `src/lpap/flow_training.py`, `src/lpap/image_energy_flow_training.py`, `src/lpap/surrogate_training.py`, `src/lpap/decoder_training.py`, `src/lpap/image_autoencoder_training.py`, `src/lpap/image_autoencoder_loss.py` | Config dataclasses, session builders, train/val loops, gallery collectors, signed-mass balance loss and lambda-dial CLI (`ae-loss-probe`). |
| **Config & checkpoint orchestration** | `src/lpap/training.py`, `src/lpap/checkpoints.py`, `src/lpap/training_log.py`, `src/lpap/teacher_config.py`, `src/lpap/teacher_checkpoints.py`, `src/lpap/training_notebook.py` | `TrainingRun` (resume, best tracking, cadence checkpoint/SQLite logging, optional HF upload & Pushover notify), atomic `.pt` checkpoints with JSON-encoded `training_state` (tensors serialized as JSON lists), SQLite run/attempt/metric store, shared teacher pair TOML projection, model-kind dispatcher. |
| **Storage, sync, notify, status, viz** | `src/lpap/storage.py`, `src/lpap/artifact_sync.py`, `src/lpap/dataset_fetch.py`, `src/lpap/notify.py`, `src/lpap/training_status.py`, `src/lpap/training_plots.py`, `src/lpap/visualization_notebook.py` | `configs/storage.toml` loader, dual-slot HF bucket checkpoint sync (`slot0`/`slot1` + `current.json`), zstd image archive fetch, Pushover client, status CLI, pure-Python PNG/SVG gallery renderers. |
| **One-off migration CLIs** | `src/lpap/migrate_teacher_permutations.py`, `src/lpap/migrate_ae_permutations.py` | Salvage CLIs to rewrite stored permutation tensors in teacher and AE checkpoints. |
| **Molab remote GPU layer** | `molab/bg_worker.py`, `molab/jobs.py`, `molab/lab.py`, `.github/skills/molab-workflow/scripts/*` | Detached background training launchers (generate standalone Python scripts under `training_logs/`), marimo status/gallery notebook, bash wrappers over `marimo-pair`. |
| **Bench, docs, animations, tests** | `bench/bench_lpap.py`, `animations/lpap_operator.py`, `doc/*`, `test/*` | PyTorch vs Triton benchmark, Manim walkthrough, TeX algorithm/loss frames, 171 existing CPU-runnable unit tests + 3 CUDA-gated tests. |

### 2. Data Flow & Training Pipeline

```mermaid
flowchart TB
  subgraph Stage1["1 · Bidirectional Flow (image_energy_flow_training.py)"]
    img_ds["data/images_32x32_gray.pt<br/>uint8 (M, 1, 32, 32)"] -->|"/255 + hilbert_flatten"| x_seq["image seq (B, 1, 1024)"]
    prior["signed log-normal prior<br/>t = 0"] --> fm["DilatedConvFlow1d<br/>t in [-1, 1]"]
    x_seq --> fm
    fm --> flow_ckpt["checkpoints/image_energy_flow.pt"]
  end

  subgraph Stage2["2 · Energy Bank Encode (energy_bank_encode.py)"]
    flow_ckpt -->|"Euler i2e [-1, 0]"| bank["data/encoded_energies_*.pt<br/>(M, 1024)"]
  end

  subgraph Stage3["3 · Teachers per C_i (teacher_c{C}_k{K}.toml)"]
    bank --> perm["apply_permutation + fold_permutation_tokens<br/>(B, C, N/C)"]
    perm --> sur["LPAPSurrogateTransformer<br/>local logits (B, C, k_max*P)"]
    perm --> oracle["lpap_surrogate_targets<br/>(exact Python LPAP)"]
    oracle -->|"weighted CE"| sur
    sur --> sur_ckpt["checkpoints/surrogate_{name}.pt"]
    sur_ckpt -->|"frozen logits"| front["prepare_lpap_decoder_batch<br/>(amp, dib, entropy) (B, C, 3)"]
    bank --> front
    front --> dec["LPAPDecoderTransformer<br/>logits (B, C, N) -> recon (B, N)"]
    dec --> dec_ckpt["checkpoints/decoder_{name}.pt"]
  end

  subgraph Stage4["4 · Multi-C Autoencoder (image_autoencoder_training.py)"]
    x_ae["image seq (B, 1, 1024)"] --> i2e["i2e flow (cloned)"] --> e_hat["encoded energy (B, 1024)"]
    e_hat -->|"parallel pairs C_1..C_m"| pair["surrogate_i -> frontend -> decoder_i"]
    pair --> d_hat["decoded energy_i (B, 1, 1024)"] --> e2i["e2i flow (cloned)"] --> x_hat["recon image_i (B, 1, 1024)"]
    e_hat -. "signed-mass + L1 + teacher CE" .-> loss((Total AE Loss))
    x_hat -. "image L2" .-> loss
  end

  flow_ckpt --> i2e
  flow_ckpt --> e2i
  sur_ckpt --> pair
  dec_ckpt --> pair
```

### 3. Entry Points & Configs

- **Pixi tasks (`pixi.toml`)**: `test` (`unittest discover`), `lint` / `format` (`ruff`), `bench-lpap`, `ae-loss-probe`, `train-status`, `data-download`, `artifacts-download` / `artifacts-upload`, `migrate-teacher-permutations`, `migrate-ae-permutations`, `notebook-lab`, `manim-*`, `tex-*`.
- **Module CLIs**: `python -m lpap.artifact_sync`, `lpap.dataset_fetch`, `lpap.training_status`, `lpap.image_autoencoder_loss`, `lpap.migrate_teacher_permutations`, `lpap.migrate_ae_permutations`, `molab.jobs` (`launch-image-energy-flow`, `launch-lpap-teacher`, `launch-ae-bidirectional-flow`).
- **Configs**:
  - `configs/storage.toml`: HF bucket IDs (`artifacts.bucket`, `images.bucket`) and archive paths (required under project root).
  - `configs/training/image_energy_flow.toml`, `image_autoencoder.toml`, `teacher_c{128_k16,256_k24,512_k32}.toml`: live operator configs.
  - `test/resources/training/*.toml`: frozen contract fixtures for unit tests.
  - `configs/secrets.toml.example`: gitignored `HF_TOKEN` and Pushover template.
- **AE config**: `configs/training/image_autoencoder.toml` is the tri-pair AE (`c128_k16`, `c256_k24`, `c512_k32`, `-v2` checkpoints, batch 128, 17,500 steps). `molab/jobs.py` loads that TOML. The image retrain pipeline is `configs/training/retrain_image_pipeline.toml` (teacher stages 7,500 steps at batch 128); see [RETRAIN.md](../RETRAIN.md).

### 4. Environment & Dependencies

- **Pixi (`pixi.toml`)**: `linux-64`, Python `3.14.x`, `[system-requirements] cuda = "12.0"`, conda-forge `pytorch-gpu >=2.10`, `triton >=3.6`, `jaxtyping`, `ruff`, `cuda-*`, `marimo`, `zstandard`, `manim`, `tectonic`, `poppler`; PyPI `huggingface-hub >=1.21`.
- **CI (`.github/workflows/test.yml`)**: Sets `CONDA_OVERRIDE_CUDA="12.0"` on CPU runners so `pixi install` solves the GPU lockfile; runs `pixi run lint` and `pixi run test`.
- **`pyproject.toml`**: Lists only `jaxtyping>=0.3.7` and `torch>=2.10` (omits `triton`, `zstandard`, `huggingface-hub`), even though `src/lpap/__init__.py:115` eagerly imports `lpap.triton_ops` (which unconditionally executes `import triton`). On any CPU-only or non-Linux environment without `triton`, `import lpap` fails immediately.
- **Public package import surface (`src/lpap/__init__.py`)**: Core modules are imported eagerly; heavy training/notebook/plot modules use PEP 562 lazy `__getattr__`.

### 5. Structural Gaps Relevant to the LLM Hidden-State Goal

- **2-D batch assumption**: `lpap_torch` and `permutation.py` accept arbitrary leading dims `(*batch, N)`, while `prepare_lpap_surrogate_batch`, `lpap_surrogate_targets`, `LPAPSurrogateTransformer`, `prepare_lpap_decoder_batch`, and `LPAPDecoderTransformer` enforce 2-D `(B, N)` / 3-D `(B, C, *)`. LLM token sequences `(B, T, D)` must either flatten `(B*T, D)` at the boundary or generalize leading batch dims across surrogate/decoder APIs.
- **Image/Hilbert coupling in AE**: `image_autoencoder_training.py` couples the multi-`C` parallel LPAP bottleneck (`ImageAutoencoderModel`) tightly to `DilatedConvFlow1d`, `hilbert_unflatten_images`, and Euler rollout steps. Extracting a domain-agnostic multi-`C` LPAP bottleneck module is straightforward once the operator/frontend bugs below are resolved.
- **Performance bottleneck on GPU**: Both `lpap_torch` and `lpap_surrogate_targets` perform boolean-mask indexing inside the `for roll_count in range(k_max)` loop, triggering CPU-GPU synchronizations on every roll step, and `lpap_triton` is unused and unsuitable for large `(C, K)` as written (see Findings 3, 6, 8).

---

## (b) LPAP Operator Correctness Audit

### Spec vs Implementation Summary

| Spec requirement (`README.md`, `doc/lpap.md`, `doc/lpap-pseudocode.md`) | Implementation status |
| --- | --- |
| Fixed seeded random permutation as hash | Implemented in `permutation.py` (CPU-seeded `randperm(N)`, device-stable). Not integrated into `ops.py`; `as_long_permutation` / `invert_permutation_indices` do not validate bijectivity. |
| Fold `N` into `C x (N/C)` lanes | **Divergent**: `ops.py` & `triton_ops.py` use row-major `(C, P)` matching `doc/lpap.md`; `permutation.py` & `surrogate.py` use column-major `(P, C).T`. |
| Top-`C` by `abs(value)` with linear probing & `K` roll budget | Matches on distinct finite non-zero inputs (`K = C` recovers exact top-`C`; `K < C` stops cleanly leaving unplaced peaks in `remaining`). Fails invariants on zeros, ties, `NaN`, and `N = 0`. |
| DIB recording (`k + dibs_diff`, swap-back `old_dib - k`) | Exact for real non-zero values when `0 <= k < C`. **Broken in `decoder_dibs_from_source_logits`** (ignores `permutation`); polluted by phantom initial zeros and `>=` tie-churn (`dib >= C` when `K > C`, `dib > 0` when `C = 1` or `C = N`). |
| In-place modification of input tensor | **Spec mismatch**: `doc/lpap.md:155,224` states input is modified in place; `ops.py:41` and `triton_ops.py:116` clone/copy and `test/test_lpap.py:20` asserts input is unchanged. |

### Edge-Case Matrix

| Edge case | Observed behavior | Verdict |
| --- | --- | --- |
| **`K` exhausted (`K < C`)** | Stops at `k_max`; unplaced peaks remain in residual; `0 <= dib < k_max` on distinct finite values. | Pass |
| **`K >= C` (distinct finite values)** | `K = C` recovers exact top-`C` with `0 <= dib < C`; extra rolls `k >= C` are no-ops. | Pass |
| **Ties (`|a| == |b|`)** | `>=` gate swaps equal-amplitude values, increasing DIB unnecessarily; for `K > C` or `C = 1`, pushes `dib >= C`. | **Bug** (Finding 2) |
| **`C == N` (`P == 1`)** | Non-zero values keep `dib = 0`; any `0.0` entry swaps with phantom zeros for `K > 1`, yielding `dib > 0`. | **Bug** (Finding 2) |
| **`C > N` (`N > 0`)** | Rejected with `ValueError` (`N % C != 0`). | Pass |
| **`N == 0` (empty last dim)** | Passes `0 % C == 0`; `lpap_torch` crashes with `IndexError` in `argmax`; `lpap_triton` does OOB GPU read/write. | **Bug** (Finding 5) |
| **`C == 1`** | Distinct values keep `dib = 0`; tied max amplitudes yield `dib = 1` for `K > 1`. | **Bug** (Finding 2) |
| **Zero / sparse inputs** | Phantom initial zeros swapped into `work` at `k = 0` get picked at `k >= 1`, producing duplicate/wrong `source_indices` (`values[source_indices] != buckets`) and zeroing autograd gradients on all-zero inputs. | **Bug** (Finding 2) |
| **Negative / signed values** | Selects by `abs()`, preserves original signs. | Pass |
| **`NaN` / `Inf`** | `Inf` selects `±inf` (ties swap on `>=`). `NaN` in `lpap_torch` wins `argmax` then fails `>=`, masking all finite values in that lane; in `lpap_triton`, `NaN` causes OOB GPU memory access. | **Bug** (Findings 4, 7) |
| **Batch dims (`1D`, `3D`, `4D`, `B = 0`)** | `lpap_torch` supports arbitrary `(*batch, N)` and `B = 0`. `lpap_triton` launches an empty `(0,)` grid for `B = 0`. Surrogate/decoder APIs restrict inputs to 2-D `(B, N)`. | Partial (Findings 6, 11) |
| **Dtype / contiguity / device** | `lpap_torch` handles `fp16`/`bf16`/`fp32`/`fp64`, non-contiguous tensors, rejects integer dtypes. `lpap_triton` stores untyped Python `0`/`0.0` scalars to `*i64`/non-`fp32` pointers and detaches autograd on CUDA. | Partial (Finding 6) |
| **Gradient flow / surrogate** | `lpap_torch` routes exact gradients through `buckets` & `remaining` across multi-hop swap-backs (fails on all-zero `K > 1` due to phantom zeros). `lpap_triton` silently detaches on CUDA. Decoder frontend flows grads through `amplitudes` & `entropy`; `dibs` uses hard `argmax`. | Partial (Findings 2, 6, 12) |
| **Determinism** | Bit-exact under `torch.use_deterministic_algorithms(True)` for `lpap_torch`, `make_permutation_indices`, and `lpap_surrogate_targets`. | Pass |

---

### Ranked Findings

#### Critical

1. **`decoder_dibs_from_source_logits` ignores `permutation`, corrupting decoder DIB tokens**
   - **Location**: `src/lpap/decoder.py:52-66`, called at `src/lpap/decoder.py:113`
   - **What**: `surrogate_logits` predicts unpermuted source indices `idx = P[pos]` (`surrogate.py:115-118`, `decoder.py:112`). `decoder_dibs_from_source_logits` computes `source_buckets = source_indices % bucket_count` directly on `idx` without applying `P^{-1}`.
   - **Why**: For any non-identity permutation (every real training config), `P[pos] % C != pos % C`. Even with a 100% oracle surrogate, `prepare_lpap_decoder_batch(...).dibs` does not match `targets.dibs` (verified: oracle returns `[3, 0, 3, 2]` vs true `[0, 0, 0, 0]`). The decoder receives `(i - (idx % C)) % C` in `[0, C-1]` (an unpermuted stride-`C` comb) instead of true LPAP DIBs in `[0, k_max-1]`, breaking the exact-LPAP equivalence contract (`doc/lpap.md:46-52`) and making exact LPAP `(a, d)` unusable as a drop-in decoder input at inference time.
   - **Suggested fix**: Pass `permutation` into `decoder_dibs_from_source_logits`; when provided, map `permuted_positions = invert_permutation_indices(permutation.to(logits.device))[source_indices]` before `% bucket_count`. Revisit the DIB normalizer (`decoder.py:114`: `max(bucket_count - 1, 1)` compresses true DIBs into `[0, (k_max - 1)/(C - 1)]`).

2. **Phantom initial zeros + `>=` gate corrupt `source_indices`, DIBs, and zero-input gradients**
   - **Location**: `src/lpap/surrogate.py:50-106`, `src/lpap/ops.py:47-88`, `src/lpap/triton_ops.py:48-79`, `src/lpap/decoder.py:127, 232-240`
   - **What**: `buckets` is initialized to `0.0` (with `dibs = 0`, `bucket_indices = 0`) and swaps whenever `abs(candidate) >= abs(bucket)`. At `k = 0`, every bucket swaps its synthetic initial `0.0` (carrying `bucket_indices = 0`) into `work`.
   - **Why**:
     - **Duplicate/false `source_indices`**: On sparse inputs (e.g., `[10, 0, ..., 0]`, or ReLU/zero-padded LLM activations), at `k >= 1` `argmax` picks the phantom zero carrying `work_indices = 0`, assigning another bucket the exact `source_index` of `10.0`. `values.gather(1, targets.source_indices)` returns `[10, 0, 10, 0]` instead of `[10, 0, 0, 0]`. In `lpap_decoder_loss`, because `batch.weights = amplitudes.abs().detach()` (soft surrogate expectation, typically `> 0`), `source_ce` actively trains the decoder toward that duplicated index.
     - **Invalid DIBs on edge cases**: Produces `dib > 0` when `C == N` (`P == 1`, zero collisions), `dib == 1` when `C == 1` with ties (`[5, -5]`), and `dib >= C` when `K > C` with ties (exceeding the `[0, C-1]` decoder range). Equal-amplitude ties at `k >= 1` also evict lower-DIB peaks for higher-DIB ties with zero amplitude gain.
     - **Zeroed autograd gradients**: On all-zero inputs with `k_max >= 2`, `buckets` holds phantom constants at `k = 1`, so `buckets.sum().backward()` produces `values.grad == 0`.
   - **Suggested fix**: Track an explicit occupied/valid mask (or initialize empty buckets with sentinel `-inf` amplitude and mark vacated `work` slots invalid with `-inf` score), never swapping uninitialized sentinels back into `work`, and use strict `>` (`abs(candidate) > abs(bucket)`) for already-occupied buckets. Note: this changes tie/zero semantics relative to current `doc/lpap.md:155`; update the spec accordingly.

#### High

3. **Row-major vs column-major fold mismatch between `ops.py`/`triton_ops.py` and `permutation.py`/`surrogate.py`, plus duplicated LPAP loop**
   - **Location**: `src/lpap/ops.py:41`, `src/lpap/triton_ops.py:44` vs `src/lpap/permutation.py:78-89`, `src/lpap/surrogate.py:109`
   - **What**: `lpap_torch` and `lpap_triton` fold flat `(..., N)` as row-major `(C, P)` (`lane = pos // P`), matching `doc/lpap.md:130-137`. `fold_permutation_tokens` and `lpap_surrogate_targets` fold as column-major `(P, C).T` (`lane = pos % C`), matching `animations/lpap_operator.py:45-47`. Meanwhile `surrogate.py:38-127` duplicates the entire LPAP loop instead of calling `ops.py`/`triton_ops.py`, leaving `lpap_torch` and `lpap_triton` unused across the training stack.
   - **Why**: Calling `lpap_torch(values, C, K)` and `lpap_surrogate_targets(prepare_lpap_surrogate_batch(values, C), K)` on the same flat `(B, N)` tensor produces completely different buckets and DIBs. Any future consolidation that swaps `lpap_surrogate_targets` onto `lpap_torch`/`lpap_triton` will silently break trained checkpoints unless both use one canonical fold convention.
   - **Suggested fix**: Pick one canonical fold layout (column-major `pos = p * C + b` matches existing checkpoints, `fold_permutation_tokens`, and `source_buckets = pos % C`; or row-major `pos = b * P + p` matching `doc/lpap.md`), unify `ops.py` / `triton_ops.py` / `surrogate.py` onto a single core operator that optionally returns `source_positions`, and align `doc/lpap.md`.

4. **Out-of-bounds GPU memory access on `NaN` in `_lpap_kernel` (static inspection)**
   - **Location**: `src/lpap/triton_ops.py:58-64`
   - **What**: `_lpap_kernel` computes `index_candidates = tl.where((scores == max_score) & probe_mask, probe_offsets, block_probe_count)` and `candidate_index = tl.min(index_candidates, axis=0)`, then unconditionally loads/stores `work_ptr + source_base + candidate_index`.
   - **Why**: If `max_score` is `NaN` (any all-`NaN` lane, or any single `NaN` if `tl.max` propagates `NaN`), `scores == max_score` is `False` everywhere, so `candidate_index == block_probe_count >= probe_count`, causing out-of-bounds GPU reads and writes (corrupting adjacent lanes/batch items or crashing with illegal memory access).
   - **Suggested fix**: Guard `candidate_index` (e.g., clamp or mask loads/stores with `candidate_index < probe_count` and suppress `should_write` when invalid) and define consistent `NaN` handling across PyTorch and Triton.

5. **Empty value dimension (`N == 0`) passes validation, crashes `lpap_torch`, and causes OOB GPU access in `lpap_triton`**
   - **Location**: `src/lpap/ops.py:16-19`, `src/lpap/triton_ops.py:100-112`, `src/lpap/permutation.py:85`
   - **What**: `values.shape[-1] % bucket_count != 0` evaluates `0 % C == 0` as valid.
   - **Why**: `lpap_torch` crashes inside the loop with `IndexError: argmax(): Expected reduction dim 2 to have non-zero size`. `lpap_triton` computes `_next_power_of_2(0) == 2`, evaluates `probe_mask` as all `False`, gets `candidate_index == 2`, and performs OOB GPU memory access.
   - **Suggested fix**: Require `values.shape[-1] > 0` (i.e., `probe_count > 0`) in `_validate_lpap_args`, `lpap_triton`, `fold_permutation_tokens`, and `lpap_surrogate_targets`.

6. **Triton kernel scalability, autograd parity, `int32` offset overflow, empty batch, and scalar store dtypes (static inspection)**
   - **Location**: `src/lpap/triton_ops.py:37-79, 82-143`
   - **What**:
     1. Nested `tl.static_range(0, k_max)` × `tl.static_range(0, bucket_count)` (`triton_ops.py:43, 51-52`) statically unrolls `C + K * C` bodies (16,896 iterations for `C=512, K=32`), exploding compile time and IR size.
     2. `lpap_triton` has no `torch.autograd.Function` wrapper: on CPU it delegates to `lpap_torch` (`requires_grad=True`), whereas on CUDA it silently returns detached tensors (`requires_grad=False`).
     3. `batch_index = tl.program_id(0)` is `int32`; `batch_index * value_count` overflows at `2^31` elements (reachable with LLM `(B, T, D)` batches).
     4. `batch_count == 0` launches `_lpap_kernel[(0,)]` instead of short-circuiting.
     5. `tl.store` writes Python `0` (`i32`) to `*i64` (`diff_ptr`/`dibs_ptr`) and `0.0` (`f32`) to `*f16`/`*bf16`/`*f64` (`buckets_ptr`) without explicit dtype casts.
   - **Suggested fix**: Use dynamic `range` (or parallelize over buckets in shared memory / registers), cast `tl.program_id(0).to(tl.int64)`, short-circuit `batch_count == 0`, cast store literals explicitly, and either reject `requires_grad=True` or wrap with a custom backward routing gradient.

#### Medium

7. **`NaN` in a lane silently hides all finite elements in `lpap_torch`**
   - **Location**: `src/lpap/ops.py:68-76`, `src/lpap/surrogate.py:77-88`
   - **What**: `argmax` selects `NaN`, then `abs(NaN) >= abs(bucket)` is `False`, so the `NaN` stays in `work` and blocks every finite element in that lane on every roll.
   - **Why**: Unlike standard `NaN` propagation (which surfaces numerical divergence immediately), a single `NaN` silently drops valid activations while leaving `buckets` finite.
   - **Suggested fix**: Either propagate `NaN` explicitly (raise or let `NaN` win) or mask `NaN` scores to `-inf` so finite elements remain selectable; keep PyTorch and Triton identical.

8. **Boolean mask indexing in `lpap_torch` and `lpap_surrogate_targets` forces per-roll CPU-GPU syncs**
   - **Location**: `src/lpap/ops.py:81-88`, `src/lpap/surrogate.py:95-106`
   - **What**: `work[batch_grid[update], source_lane_grid[update], candidate_positions[update]] = ...` indexes with boolean tensor `update` inside `for roll_count in range(k_max)`.
   - **Why**: On CUDA, boolean indexing calls `aten::nonzero`, stalling the GPU with host-device synchronization on every roll step (`k_max = 16..32` × 3 pairs in tri-pair AE) and blocking `torch.compile` full-graph capture.
   - **Suggested fix**: Because `source_lanes` is a bijection per roll, use fixed-shape indexing with `torch.where(update, ...)` (zero host-device syncs, `torch.compile`-friendly).

9. **Non-bijective permutations accepted silently; `invert_permutation_indices` exposes uninitialized memory**
   - **Location**: `src/lpap/permutation.py:31-58`
   - **What**: `as_long_permutation` checks only `ndim == 1` and `numel() == value_count`. `invert_permutation_indices` allocates `torch.empty_like(permutation)` and scatters into `inverse[permutation]`.
   - **Why**: A corrupted checkpoint or bad caller tensor (e.g., `[0, 0, 0, 0]`) passes validation and leaves uninitialized garbage values in `inverse`.
   - **Suggested fix**: Validate `permutation.min() >= 0`, `permutation.max() < n`, and uniqueness (e.g., `torch.equal(permutation.sort().values, torch.arange(n, device=permutation.device))` or boolean seen mask).

10. **Eager `import triton` at package import time breaks CPU-only / non-Linux imports**
    - **Location**: `src/lpap/__init__.py:115`, `src/lpap/triton_ops.py:13-14`, `pyproject.toml:9-12`
    - **What**: `import lpap` eagerly imports `lpap.triton_ops`, which unconditionally imports `triton`, even though `triton` is omitted from `pyproject.toml` dependencies and `lpap_triton` falls back to `lpap_torch` on CPU.
    - **Why**: Prevents importing `lpap` on macOS, Windows, or lightweight CPU environments (and forces `CONDA_OVERRIDE_CUDA="12.0"` in CI).
    - **Suggested fix**: Move `lpap_triton` to `_LAZY_MODULE_EXPORTS` in `src/lpap/__init__.py` and guard `import triton` inside `triton_ops.py` (falling back to `lpap_torch` when CUDA or Triton is unavailable).

11. **Missing input validations in `decoder.py`**
    - **Location**: `src/lpap/decoder.py:52-66` and `src/lpap/decoder.py:81-93`
    - **What**:
      - `prepare_lpap_decoder_batch` validates `temperature > 0` for `int | float`, but skips validation when `temperature` is a `torch.Tensor` (allowing negative or zero temperature -> inverted softmax or `NaN`).
      - `decoder_dibs_from_source_logits` does not check `logits.shape[1] == bucket_count`, raising a raw broadcasting `RuntimeError` instead of `ValueError`.
    - **Suggested fix**: Validate `(temperature_tensor > 0).all()` (or clamp/assert positive scalar tensor) and check `logits.shape[1] != bucket_count -> ValueError`.

#### Low

12. **Redundant exact-LPAP recomputation & surrogate-to-encoder gradient coupling in AE forward**
    - **Location**: `src/lpap/image_autoencoder_training.py:492-503`, `src/lpap/decoder.py:95-106`
    - **What**: `ImageAutoencoderModel.forward_chain` calls `prepare_lpap_surrogate_batch`, runs `surrogate`, and then calls `prepare_lpap_decoder_batch`, which calls `prepare_lpap_surrogate_batch` and `lpap_surrogate_targets` again (even at inference when targets are not needed). Also, `surrogate_tokens` is not detached in `forward_chain`, so `surrogate_teacher_ce` backpropagates into `encoded_energy` (`image_to_energy_flow`).
    - **Why**: Unnecessary compute during training/inference; unclear whether encoder gradients from `surrogate_teacher_ce` are intentional (they reward making `encoded_energy` easier for the surrogate to classify).
    - **Suggested fix**: Allow passing precomputed `tokens` / optional `compute_targets: bool = True` to `prepare_lpap_decoder_batch`, and make gradient flow from `surrogate_teacher_ce` to `encoded_energy` an explicit config choice.

13. **Surrogate attention**
    - **Location**: `src/lpap/surrogate.py`, `src/lpap/transformer.py`
    - **What**: `LPAPSurrogateTransformer` uses a circular-backward mask `(q - k) % C < k_max`, periodic circular RoPE (frequencies `2π m / N`, `circular=True`), and SDPA. The head predicts local `(dib, probe)` logits of size `k_max * P`. Cosine/sine tables are non-persistent buffers, rebuilt on the forward.

14. **LPAP input mutation**
    - **Location**: `doc/lpap.md`, `doc/lpap-pseudocode.md`, `src/lpap/ops.py`
    - **What**: The operator clones the input into an internal working copy. `lpap.md` and the pseudocode describe that copy; the caller’s `values` stay unchanged.

---

## Tests Added & Results

Added `test/test_lpap_review_step1.py` (21 tests: 13 `@unittest.expectedFailure` bug proofs + 8 passing edge-case verifications).

Executed on CPU (`Python 3.14.10`, `torch 2.15.0+cpu`, `triton 3.9.0`, `ruff 0.17.6`):

- `ruff check src/lpap molab test bench animations`: **All checks passed**
- `PYTHONPATH=src:. python -m unittest -v test/test_lpap_review_step1.py`: **21 tests — 8 ok, 13 expected failures (all 13 suspected bugs reproduced)**
- `PYTHONPATH=src:. python -m unittest discover -s test`: **192 tests — OK (176 ok, 3 CUDA-gated skipped, 13 expected failures)**

Bug proofs in `LpapSuspectedBugsTest` (every test fails on current code):

| Test | Finding | Observed failure on current code |
| --- | --- | --- |
| `test_oracle_decoder_dibs_match_lpap_targets_with_permutation` | #1 (Critical) | Oracle decoder DIBs `[[3, 0, 3, 2], ...]` != LPAP DIBs `[[0, 0, 0, 0], ...]` |
| `test_lpap_torch_matches_surrogate_targets_on_same_flat_input` | #3 (High) | `lpap_torch` buckets `[15, 8, 12, 16]` != `lpap_surrogate_targets` `[13, 14, 15, 16]` |
| `test_sparse_input_source_indices_map_back_to_bucket_values` | #2 (Critical) | `values[source_indices] = [10, 0, 10, 0]` != `buckets = [10, 0, 0, 0]` (duplicate index `0`) |
| `test_c_equals_n_zero_element_keeps_zero_dib` | #2 (Critical) | `dibs = [[0, 0, 2, 0]]` != `[[0, 0, 0, 0]]` |
| `test_c_equals_1_with_tie_keeps_zero_dib` | #2 (Critical) | `dibs = [[1]]` != `[[0]]` |
| `test_k_max_gt_bucket_count_with_ties_keeps_dib_below_c` | #2 (Critical) | `dibs = [[2, 1]]` contains `dib >= C=2` |
| `test_all_zero_input_preserves_nonzero_autograd_grad_for_k_gt_1` | #2 (Critical) | `values.grad.sum() == 0.0` != `3.0` |
| `test_empty_value_dimension_raises_value_error` | #5 (High) | Raises `IndexError` in `argmax` instead of `ValueError` |
| `test_nan_in_lane_does_not_mask_finite_values` | #7 (Medium) | `buckets = [[0.0, 3.0]]` (`10.0` masked by `NaN`) |
| `test_as_long_permutation_rejects_non_bijection` | #9 (Medium) | Accepts `[0, 0, 0, 0]` without raising `ValueError` |
| `test_invert_permutation_rejects_non_bijection` | #9 (Medium) | Returns uninitialized garbage instead of raising `ValueError` |
| `test_prepare_decoder_batch_rejects_non_positive_tensor_temperature` | #11 (Medium) | Accepts `torch.tensor(-1.0)` without raising `ValueError` |
| `test_decoder_dibs_validates_bucket_count_matches_logits_dim1` | #11 (Medium) | Raises broadcasting `RuntimeError` instead of `ValueError` |

---

## Open Questions for the Author

1. **Canonical fold convention (Finding #3)**: Should the unified LPAP operator standardize on **column-major** `pos = p * C + b` (matching `fold_permutation_tokens`, `surrogate.py`, `animations/lpap_operator.py`, and existing teacher/AE checkpoints) or **row-major** `pos = b * P + p` (matching `ops.py`, `triton_ops.py`, and `doc/lpap.md`)?
2. **Occupied mask + strict `>` vs current `>=` zero-table rule (Finding #2)**: Are you open to replacing the initial-zero table + `>=` gate with an explicit occupied/empty mask + strict `>` for occupied buckets (fixing phantom zero aliasing, `dib >= C`, `C = 1`/`C = N` DIB drift, and all-zero autograd), or do you want to keep `>=` in `ops.py` and only isolate phantom zeros in `lpap_surrogate_targets`?
3. **Fixing `decoder_dibs_from_source_logits` (Finding #1) & DIB encoding**: Fixing `permutation` in `decoder_dibs_from_source_logits` invalidates existing decoder/AE checkpoints (which were trained on `(i - idx % C) % C`). Since the next target is LLM hidden states (fresh models anyway), should we also switch DIB normalization from `dib / (C - 1)` to `dib / max(min(k_max, C) - 1, 1)` or a learned/sinusoidal DIB embedding, and should the surrogate predict local `(roll, probe)` logits (`k_max * P` classes, permutation-invariant) instead of full `N` classes?
4. **Direct STE/routing LPAP vs learned surrogate for LLM hidden states**: `lpap_torch` already propagates exact straight-through routing gradients through `buckets` (verified even across multi-hop swap-backs). For the LLM bottleneck, do you still want the learned `LPAPSurrogateTransformer` (for soft differentiable index selection + entropy), or do you want to support an exact-LPAP bypass mode (`values -> exact LPAP (a, d) -> decoder`) as an ablation / fast baseline?
5. **Triton kernel priority**: Given that `lpap_torch` rewritten with `torch.where` (no boolean-mask syncs) + `torch.compile` runs fully fused on GPU with autograd support and `source_indices` output, is keeping and rewriting `lpap_triton` still desired, or should `torch.compile`-friendly pure PyTorch be the primary implementation?

---

## Follow-up

### 1. Fold Convention & Checkpoint Metadata

**Where permutation seeds, permutation tensors, and fold order are recorded:**

- **Checkpoints (`.pt` -> `training_state_json`)**:
  - `permutation_seed` is **explicitly stripped** from every checkpoint by `_without_permutation_seed` (`src/lpap/checkpoints.py:99-111, 145-147, 223`). No checkpoint stores `permutation_seed`.
  - Instead, checkpoints store the concrete **permutation tensor** (serialized as a JSON int list in `training_state_json`):
    - Surrogate & decoder checkpoints: `training_state["permutation"]` (`src/lpap/surrogate_training.py:486`, `src/lpap/decoder_training.py:677`).
    - AE checkpoints: `training_state["lpap_pairs"][i]["permutation"]` (`src/lpap/image_autoencoder_training.py:1192-1199`, `src/lpap/teacher_checkpoints.py:239-251`).
  - Note: on resume, decoder (`src/lpap/decoder_training.py:520-542`) and AE (`src/lpap/image_autoencoder_training.py:854-871`) reload and verify `training_state["permutation"]`, whereas surrogate resume (`src/lpap/surrogate_training.py:329-375`) regenerates `permutation` from `config.run.permutation_seed` (CPU-seeded `randperm`) rather than loading the tensor from the checkpoint.
- **SQLite training logs (`training_logs/*.sqlite` -> `runs.config_json`)**: `permutation_seed` **is** recorded here because `TrainingRun` writes `config.as_run_config()` without stripping (`src/lpap/training.py:162-168`, `src/lpap/surrogate_training.py:183`, `src/lpap/decoder_training.py:133`).
- **Fold convention**: **Not recorded anywhere** (not in checkpoints, `model_config`, `training_state`, TOML configs, or SQLite). It is hardcoded implicitly in `fold_permutation_tokens` (`src/lpap/permutation.py:89`) and `lpap_surrogate_targets` (`src/lpap/surrogate.py:109`).

**Column-major vs row-major tradeoff:**

- **Why column-major (`pos = p * C + b`, `bucket = pos % C`) is reasonable**:
  - Matches `fold_permutation_tokens`, `surrogate.py`, `animations/lpap_operator.py:45-47`, and all existing surrogate checkpoints (keeping column-major preserves existing surrogate checkpoints — see Section 5 below).
  - Aligns naturally with hash-modulo arithmetic (`source_bucket = pos % C`).
  - Once `apply_permutation` runs, a random gather already allocates a new tensor; pre-folding `perm.reshape(P, C).T.contiguous()` produces a contiguous `(B, C, P)` tensor in one gather with **zero** extra transpose overhead.
- **Concrete pushback for row-major (`pos = b * P + p`, `reshape(..., C, P)`)**:
  - Without permutation (direct `lpap(values)` calls, benchmarks, tests), row-major `reshape(..., C, P)` is a **zero-copy contiguous view** (coalesced probe reads in Triton), whereas column-major `reshape(..., P, C).movedim(-1, -2)` produces a non-contiguous view with stride `C` along probes — forcing either uncoalesced stride-`C` GPU loads or an extra `.contiguous()` transpose copy.
  - Row-major already matches `ops.py:41`, `triton_ops.py:44`, `doc/lpap.md:130-138`, `doc/lpap-pseudocode.md:43`, and `test/test_lpap.py`.
- **Conclusion**: Keeping **column-major** is fine and preserves existing surrogate checkpoints, provided we (a) unify `ops.py`, `triton_ops.py`, `doc/lpap.md`, and `doc/lpap-pseudocode.md` onto column-major, (b) make the internal core operator take contiguous `(..., C, P)` folded tokens (or fuse `perm_folded` gather) so Triton never does stride-`C` inner-loop reads, and (c) record `fold_order = "column_major"` in `model_config` if ambiguity can recur.

---

### 2. Worked Trace for Finding #2, `>=` Semantics, Fix, and Severity Downgrade

**Concrete trace of current `lpap_surrogate_targets` on `x = [10.0, 0.0, 0.0, 0.0]` (`C=2, P=2, K=2`, identity permutation):**

Folded column-major: lane 0 holds `[10.0 (idx 0), 0.0 (idx 2)]`; lane 1 holds `[0.0 (idx 1), 0.0 (idx 3)]`.

- **Init**:
  - `work = [[10.0, 0.0], [0.0, 0.0]]`, `work_indices = [[0, 1], [0, 1]]`, `dibs_diff = [[0, 0], [0, 0]]`
  - `buckets = [0.0, 0.0]` (phantom zeros), `bucket_indices = [0, 0]`, `dibs = [0, 0]`
- **Roll `k = 0`** (`source_lanes = [0, 1]`):
  - `cand_pos = [0, 0]`, `cand = [10.0, 0.0]`, `sel_idx = [0, 0]`, `cand_dib = [0, 0]`, `update = [True, True]`
  - Both buckets swap their phantom initial `0.0` (with `bucket_indices = 0`) back into `work[:, :, 0]`:
  - `work = [[0.0, 0.0], [0.0, 0.0]]`, `work_indices = [[0, 1], [0, 1]]`, `dibs_diff = [[0, 0], [0, 0]]`
  - `buckets = [10.0, 0.0]`, `bucket_indices = [0, 0]`, `dibs = [0, 0]`
- **Roll `k = 1`** (`source_lanes = [1, 0]`):
  - Bucket 0 reads lane 1 (`[0.0, 0.0]`): `|0| >= |10|` is `False` -> keeps `10.0`.
  - Bucket 1 (holding real `0.0` from idx 1!) reads lane 0 (`[phantom_0 (work_indices=0), real_0 (work_indices=1)]`):
    - `argmax` picks slot `0` — the **phantom zero** that Bucket 0 swapped into lane 0 at `k = 0`, carrying `work_indices = 0` (the probe slot of `10.0`).
    - Because of `>=`, `|0.0| >= |0.0|` is `True`, so the **phantom zero re-enters Bucket 1**:
  - `cand_pos = [0, 0]`, `cand = [0.0, 0.0]`, `sel_idx = [0, 0]`, `cand_dib = [1, 1]`, `update = [False, True]`
  - `buckets = [10.0, 0.0]`, `bucket_indices = [0, 0]`, `dibs = [0, 1]`
- **Index recovery (`surrogate.py:108-109`)**:
  - `source_buckets = ([0, 1] - [0, 1]) % 2 = [0, 0]`
  - `source_indices = bucket_indices * 2 + source_buckets = [0, 0]` -> **`x[source_indices] = [10.0, 10.0] != buckets = [10.0, 0.0]`** (duplicate index `0`).
  - On all-zero `[0, 0, 0, 0]` with `K = 2`, both buckets pick phantom constants at `k = 1`, zeroing `values.grad`.

**Plain assessment & severity downgrade (Critical -> Medium):**

- The author's intuition holds on **dense non-zero inputs** (`|x_i| > 0`): at `k = 0` every bucket receives a strictly positive amplitude, so at `k >= 1` swapped-back phantom zeros (`|0| < |bucket|`) **never re-enter**.
- The phantom-zero bug triggers **only when a bucket holds an exact `0.0`** (e.g., an all-zero lane at `k = 0`, sparse/ReLU/padded activations, all-zero tensors, or `C = N` with a zero entry).
- Furthermore, in `lpap_surrogate_loss`, zero buckets receive `weights = buckets.abs() = 0`, so duplicate `source_indices` on zero buckets contribute **zero** surrogate CE loss (they do still leak into decoder `source_ce`, where `weights = amplitudes.abs()` uses the non-zero soft surrogate expectation, and corrupt `dibs` / zero-input gradients).
- **Finding #2 is therefore downgraded from Critical to Medium**: dormant on the current dense float32 image energy bank, but a real correctness hazard for sparse/ReLU/padded LLM activations and edge-case invariants.

**Nuance on `>=` vs Robin Hood hashing:**

- On the Manim grid `_ROWS` (`C=4, P=3, K=3`), `>=` reduces total DIB from `4` (under strict `>`) to `3` (`dibs = [2, 1, 0, 0]` vs `[2, 2, 0, 0]`) because at `k=1`, `-6` (`dib=1`) evicts `6` (`dib=0`), which then re-enters bucket 1 (`5 < 6`) at `k=2` with `dib=1`.
- However, because LPAP drops displaced items when all buckets are strong (unlike fixed-key-set Robin Hood hashing where total DIB is conserved), unconditional `>=` **increases** DIB whenever an evicted tied element finds no weaker bucket downstream (on 200 random integer `4x3` grids with `K=4`, mean `sum(dib)` is `3.09` with `>=` vs `1.08` with `>`, and mean `max(dib)` is `1.72` vs `0.96`). And if `K > C` (or `C = 1, K >= 2`), `>=` on ties wraps around and produces `dib >= C`.

**Proposed fix keeping `>=` on real values:**

Track `work_valid` (bool, init `True`) and `bucket_valid` (bool, init `False`), and validate/cap `k_max <= C` (or stop rolls at `min(k_max, C)` so `0 <= dib < C` always holds by induction):

- Candidate scoring: `scores = torch.where(lane_valid, lane_values.abs(), -1.0)` (since `abs >= 0`, vacated slots can never win `argmax`).
- Update rule: `update = cand_valid & (~bucket_valid | (cand.abs() >= buckets.abs()))`.
- Swap-back: write `(old_bucket, old_dib - k, old_probe)` into `work` with `work_valid = old_bucket_valid` — so filling an initially empty bucket marks its vacated `work` slot `False` and **never** injects a phantom zero back into `work`.
- **Verification**:
  - Manim `_ROWS` (`C=4, P=3, K=3`): `buckets = [7, 6, -8, 9]`, `dibs = [2, 1, 0, 0]`, `source_indices = [6, 4, 2, 3]` — **bit-exact match with `animations/lpap_operator.py`**.
  - `[10, 0, 0, 0]` (`C=2, P=2, K=2`): `buckets = [10, 0]`, `dibs = [0, 1]`, `source_indices = [0, 2]`, `x[source_indices] = [10, 0]` (no duplicate).
  - `C = N` `[1, 2, 0, 4]` (`K=3`): `dibs = [0, 0, 0, 0]` (all `work` slots become invalid after `k = 0`).
  - All-zero `[0, 0, 0, 0]` (`K=2`): buckets hold real input elements, preserving autograd gradients.

---

### 3. Local `(dib, probe)` Logits vs Full-`N` Source-Index Logits

**What the surrogate predicts:**

- Under LPAP with `k_eff = min(k_max, C)` rolls, every item that lands in bucket `i` has `dib = d in {0, ..., k_eff - 1}` and original probe slot `p in {0, ..., P - 1}` in lane `(i - d) % C`. Only `M = k_eff * P` out of `N = C * P` source positions are reachable by bucket `i`.
- Replace `LPAPSurrogateTransformer.output = nn.Linear(hidden_dim, N)` with `nn.Linear(hidden_dim, k_eff * P)`, producing `local_logits` of shape `(B, C, k_eff * P)` (viewable as `(B, C, k_eff, P)`), where class `c = d * P + p` selects probe `p` from lane `(i - d) % C`.
- Calling this `(dib, probe)` is more exact than `(roll, probe)`: on swap-back re-entry at roll `k`, `d = dibs_diff + k <= k < k_eff` is the item's true DIB, not the roll step `k`.

**Class count comparison (`M = k_eff * P` vs `N`):**

| Config | `N` | `C` | `P` | `k_eff` | Full-`N` classes | Local `(d, p)` classes | Reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `c128_k16` | 1024 | 128 | 8 | 16 | 1024 | **128** | 8x |
| `c256_k24` | 1024 | 256 | 4 | 24 | 1024 | **96** | 10.7x |
| `c512_k32` | 1024 | 512 | 2 | 32 | 1024 | **64** | 16x |
| LLM `D=4096, C=256, K=16` | 4096 | 256 | 16 | 16 | 4096 | **256** | 16x |
| LLM `D=4096, C=512, K=16` | 4096 | 512 | 8 | 16 | 4096 | **128** | 32x |

**Recovery of targets, DIB, amplitude, entropy, and global `source_index`:**

- **Surrogate training target**: `target_class = targets.dibs * P + targets.bucket_probes` (no `permutation` lookup required).
- **DIB**:
  - Hard argmax: `best = local_logits.argmax(dim=-1)` -> `dib = best // P` (guaranteed in `[0, k_eff - 1]`; makes Bug #1 structurally impossible).
  - Optional soft DIB: `E[d] = sum_d P(d) * d` where `P(d) = probs.reshape(B, C, k_eff, P).sum(dim=-1)`.
- **Amplitude (decoder frontend)**: gather each bucket's `k_eff` visible lanes from already-folded `tokens (B, C, P)` into `visible[b, i, d, p] = tokens[b, (i - d) % C, p]` of shape `(B, C, k_eff, P)`, then `amplitudes = (probs * visible).sum(dim=(-2, -1))` (`O(B * C * k_eff * P)` instead of `O(B * C * N)`).
- **Entropy**: `-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)` over the `k_eff * P` reachable candidates (bounded by `log(k_eff * P)` instead of `log(N)`).
- **Global unpermuted `source_index`** (for decoder `source_ce`, scatter, or galleries):
  - `d = best // P`, `p = best % P`, `src_bucket = (bucket_index - d) % C`
  - Column-major: `permuted_pos = p * C + src_bucket` -> `source_index = permutation[permuted_pos]`.

**Permutation invariance:**

- With full-`N` logits, the surrogate takes permuted `tokens` and predicts unpermuted `0..N-1`, so its `output` linear layer must memorize the specific `permutation` table; changing `permutation_seed` breaks the surrogate, and buckets cannot share output structure.
- With local `(dib, probe)` logits, both inputs and outputs live purely in folded-lane coordinates. The **surrogate** becomes **permutation-invariant** and **bucket-shift-equivariant** (one trained surrogate per `(C, P, K)` works with any permutation).
- The **decoder** remains **permutation-dependent** because it maps `(a, d)` back to unpermuted indices `0..N-1` using the specific collision sets induced by `permutation`.

**Pros / Cons & Retraining Impact:**

- **Pros**: 8x–32x smaller head/logits/softmax (important at LLM `D = 2048..4096`); zero wasted probability mass on unreachable indices; structurally eliminates Bug #1; permutation-invariant surrogate; cleaner entropy; easy soft DIB option.
- **Cons**: Changes `LPAPSurrogateTransformer.output` shape `(hidden_dim, N) -> (hidden_dim, k_eff * P)`, invalidating existing surrogate checkpoints; requires capping `k_eff = min(k_max, C)`.
- **Retraining impact**: Adopts Scenario B below (surrogates must retrain too). Strongly recommended for the fresh LLM stack; optional for the image rerun depending on whether reusing the 3 existing surrogate checkpoints matters more than retraining them once.

---

### 4. Exact Triton Op at Inference (+ Optional Decoder-Only Fine-Tuning)

Requirements to run exact `lpap_triton` at inference (`values -> exact_lpap -> (a, d, entropy=0) -> decoder`) and optionally fine-tune only the decoder on it:

1. **Bug #1 fixed + matching frontend contract**: Decoder must be trained on true permuted DIBs (not `(i - P[pos] % C) % C`). Exact-LPAP frontend must apply the same DIB normalizer (`dib / max(bucket_count - 1, 1)` or `dib / max(min(k_max, bucket_count) - 1, 1)`) and pass `entropy = torch.zeros_like(amplitudes)` (unless the entropy channel is removed).
2. **Fold & permutation unification**: Unify `lpap_triton` with the canonical fold (column-major) and apply the checkpoint's stored `permutation` before folding (ideally passing contiguous `(B, C, P)` tokens produced by a single fused `values[..., perm_folded]` gather).
3. **Mandatory Triton kernel fixes (`src/lpap/triton_ops.py`)**:
   - **Replace `tl.static_range` (`triton_ops.py:43, 51-52`) with dynamic `range`** (blocker: `C=512, K=32` currently statically unrolls `17,408` loop bodies). For actual inference speed at small batch sizes (`B = 1..32`), also consider keeping `buckets`/`dibs` in SRAM/registers or parallelizing over buckets rather than doing serial `K * C` global-memory round-trips inside 1 program per batch item, and benchmark against `torch.compile`-d `torch.where` PyTorch.
   - **Guard `NaN` OOB indexing** (`triton_ops.py:58-64`: clamp/mask `candidate_index < probe_count` and suppress writes when invalid).
   - **Validate `value_count > 0` and short-circuit `batch_count == 0`**.
   - **Use `int64` offsets**: `batch_index = tl.program_id(0).to(tl.int64)` to prevent `int32` overflow on large LLM `(B * T, D)` batches.
   - **Explicit store dtypes**: cast `0` to `tl.int64` and `0.0` to `values.dtype` (`fp16`/`bf16` safe).
   - **Add `work_valid`/`bucket_valid` mask** (Section 2) so Triton matches `lpap_torch` on zero/sparse inputs.
4. **Gradients & decoder-only fine-tuning**:
   - **Inference (`torch.no_grad()`)**: No custom backward needed.
   - **Decoder-only fine-tuning on exact LPAP**: Encoder / `values` are frozen/detached, so gradients stop at `decoder_tokens` and **no custom backward in `lpap_triton` is needed**. Still, `lpap_triton` should raise `RuntimeError` when `values.requires_grad` instead of silently detaching on CUDA. (If end-to-end encoder gradients through exact LPAP are ever wanted later, returning `source_indices` makes a straight-through amplitude backward a ~10-line `scatter_add_`.)
   - **`source_indices` output**: Not needed for inference or decoder fine-tuning with `source_ce_weight = 0.0`. Only needed if decoder fine-tuning keeps `source_ce_weight > 0` (adds one `int32`/`int64` `work_probe`/`bucket_probe` tracker to the kernel).
   - **Train-test gap**: Switching from soft surrogate `(E[a], argmax_dib, H > 0)` to exact LPAP `(exact_a, exact_dib, H = 0)` leaves `decoder.log_frontend_temperature` unused and shifts input sharpness/entropy to 0; short decoder-only fine-tuning on exact LPAP directly closes that gap.

---

### 5. Checkpoint Invalidation & Minimal Retraining Order (Image Experiments)

Artifacts in the current image pipeline: `image_energy_flow.pt` -> `data/encoded_energies_bank_flow_best_4step.pt` -> 3 surrogates (`c128_k16`, `c256_k24`, `c512_k32`) -> 3 decoders (`c128_k16`, `c256_k24`, `c512_k32`) -> `image_autoencoder_tri_lnorm.pt`.

| Artifact | Scenario A: Minimal fix (keep col-major + full-`N` surrogate; fix Bug #1 + Bug #2 mask) | Scenario B: Also switch surrogate to local `(dib, probe)` logits (or row-major fold) | Why |
| --- | --- | --- | --- |
| `image_energy_flow.pt` | **Keep** | **Keep** | Pure image <-> energy flow; no LPAP. |
| `data/encoded_energies_bank_flow_best_4step.pt` | **Keep** | **Keep** | Encoded directly from `image_energy_flow.pt`. |
| `surrogate_c{128_k16,256_k24,512_k32}.pt` | **Keep** | **Retrain** | Under Scenario A, Bug #1 is never called in surrogate training, `k_max < C`, dense float32 energies have ~0 exact zeros, and zero buckets have `weight = 0` in `lpap_surrogate_loss`. Verify by re-evaluating saved surrogates after the Bug #2 fix (val metrics should match). Under Scenario B, output head / fold changes. |
| `decoder_c{128_k16,256_k24,512_k32}.pt` | **Retrain** | **Retrain** | Every training step fed corrupted DIBs from Bug #1 (`(i - P[pos] % C) % C`). |
| `image_autoencoder_tri_lnorm.pt` | **Retrain** | **Retrain** | Initialized from invalidated decoders and co-trained (`train_* = True` in `molab/jobs.py:242-245`) through the buggy `prepare_lpap_decoder_batch` DIB path. |

**Minimal retraining order:**

- **Scenario A (minimal rerun — keep existing surrogates)**:
  1. Keep `image_energy_flow.pt`, `encoded_energies_bank_flow_best_4step.pt`, and `surrogate_c{128_k16,256_k24,512_k32}.pt` (run one validation check on each surrogate after the Bug #2 mask fix to confirm bit-identical/matching val loss).
  2. Retrain `decoder_c128_k16.pt`, `decoder_c256_k24.pt`, and `decoder_c512_k32.pt` from scratch (`resume_from_checkpoint=False`). All 3 are independent (can run in parallel).
  3. Retrain `image_autoencoder_tri_lnorm.pt` from scratch (`resume_from_checkpoint=False`) once all 3 decoders finish.
  4. *(Optional)* Short decoder-only fine-tuning on exact `lpap_triton` `(a, d, entropy=0)`.
- **Scenario B (if upgrading surrogate to local `(dib, probe)` logits or changing fold)**:
  1. Keep `image_energy_flow.pt` and `encoded_energies_bank_flow_best_4step.pt`.
  2. Retrain the 3 surrogates (`c128_k16`, `c256_k24`, `c512_k32`) from scratch; each paired decoder can start as soon as its own surrogate finishes.
  3. Retrain `image_autoencoder_tri_lnorm.pt` from scratch once all 3 pairs finish.

