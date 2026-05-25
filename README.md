# LPAP

LPAP stands for Linear Probing Amplitude Pooling.

This project explores a pooling operator that reduces a flat tensor of `N` elements into `C` buckets, where `N` is a multiple of `C`. Values are selected by largest absolute amplitude, placed into a compact bucket table, and tracked with integer DIB values that record distance from each value's initial bucket.

The operator is intended for batched use, with a `k_max` argument that limits the maximum number of probing rolls per batch item.

The current repository is an early research scaffold for a PyTorch/Triton implementation.

Implemented entry points:

- `lpap.lpap_torch`: PyTorch reference implementation.
- `lpap.lpap_triton`: Triton implementation with CPU fallback for non-CUDA tensors.
- `lpap.make_grouped_permutation_indices`: fixed seeded LPAP front-end permutation.
- `lpap.LPAPSurrogateTransformer`: RoPE transformer surrogate for LPAP bucket-index prediction.
- `lpap.TrainingRun`: generic checkpoint/resume/log-cadence helper for pretraining loops.
- `lpap.training_log`: SQLite run configuration and arbitrary KPI logging helpers.

## Documentation

- [LPAP operator notes](doc/lpap.md)
- [Dataset storage notes](doc/data-storage.md)

## Environment

The project uses Pixi. From the repository root:

```sh
pixi install
```

The declared environment includes Python, PyTorch GPU, Triton, jaxtyping, and Ruff.

Run the test suite with:

```sh
pixi run test
```

Run a small LPAP implementation benchmark with:

```sh
pixi run bench-lpap
```

Open the synthetic harmonic visualization notebook with:

```sh
pixi run notebook-synthetic
```

Open the LPAP surrogate training notebook with:

```sh
pixi run notebook-surrogate
```

The surrogate notebook resumes from `checkpoints/surrogate_synthetic.pt` when available and logs run configuration plus per-step KPIs to `training_logs/surrogate.sqlite`.

## Data

Large local dataset artifacts under `data/` are intentionally ignored by Git. The local training artifact is `data/images_32x32_gray.pt`. Load it with `lpap.data.load_image_tensor_dataset` or construct a dataloader with `lpap.data.image_dataloader`.

The project also includes a batched synthetic harmonic generator. Use `lpap.data.sample_synthetic_harmonic_batch` for direct tensor generation, or `lpap.data.synthetic_harmonic_dataloader` for a prebatched iterable dataloader.
