# Glossary

## Data And Layout

- **Energy**: A flat length-`N` tensor of scalar values. Synthetic harmonics and image-to-energy outputs both live in this representation.
- **Image sequence**: A grayscale image flattened to length `N` through the Hilbert ordering helpers.
- **Hilbert flattening**: The image-to-1D mapping used before flow models. It preserves more spatial locality than raster order.
- **Bucket**: One LPAP output lane. The value count `N` is split into `bucket_count` buckets, each with `probe_count = N / bucket_count` source slots.
- **Probe**: A source slot inside a bucket before LPAP selection.
- **DIB**: Distance in buckets from a value's original bucket to the bucket where it is selected.
- **Grouped permutation**: The fixed seeded permutation applied before LPAP tokenization so each bucket receives balanced source positions.

## LPAP Models

- **LPAP operator**: The exact pooling rule that selects high-amplitude values into buckets with DIB metadata.
- **Surrogate**: A transformer trained to predict full-`N` source-index logits for each LPAP bucket.
- **Decoder**: A transformer that reconstructs source energy values from surrogate logits and LPAP-derived decoder tokens.
- **Teacher**: A frozen model or exact operator used to supervise another model. The surrogate is supervised by exact LPAP targets; reflow uses a high-step energy-to-image flow teacher.

## Flow Models

- **Image-to-energy flow (`image_to_energy`)**: A 1D flow that maps Hilbert-flattened grayscale images to energy.
- **Energy-to-image flow (`energy_to_image`)**: A 1D flow that maps decoder-projected energy to Hilbert-flattened grayscale images.
- **Reflow (`energy_to_image_reflow`)**: Distillation that trains a low-step student flow to match a high-step energy-to-image teacher.
- **Student steps**: The number of unrolled Euler midpoint steps used by a low-step flow during reflow or image-autoencoder training.
- **Teacher steps**: The larger number of integration steps used to produce a higher-quality reflow target.

## Autoencoders

- **Inner energy path**: The energy-domain path `energy -> surrogate -> decoder -> energy`.
- **Image autoencoder (`image_autoencoder`)**: The total end-to-end path `image -> image_to_energy -> surrogate -> decoder -> energy_to_image -> image`.
- **Energy reconstruction L1**: The loss comparing decoded energy against encoded energy inside the image autoencoder.
- **LPAP teacher cross-entropy**: The auxiliary loss that keeps surrogate predictions anchored to exact LPAP source-index targets.

## Artifacts

- **Checkpoint**: A `.pt` payload under `checkpoints/` containing model state, best model state when available, optimizer state, metrics, and training metadata.
- **SQLite log**: A local database under `training_logs/` containing run configuration, run attempts, scalar KPIs, and checkpoint paths.
- **Run config**: The serializable TOML-compatible configuration that describes a training run.
- **Model config**: Checkpoint metadata describing model-dependent dimensions and source settings. Checkpoints are authoritative for this information.
