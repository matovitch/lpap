# Training Stack

See the [documentation index](index.md) for the full documentation map and the [glossary](glossary.md) for project terminology.

LPAP currently has six trainable model kinds wired into one shared marimo training notebook:

- `surrogate`: learns full-`N` source-index logits for LPAP bucket selections on synthetic harmonic energy.
- `decoder`: reconstructs source energy values from frozen surrogate logits.
- `image_to_energy`: trains a flow from Hilbert-flattened grayscale images to synthetic harmonic energy.
- `energy_to_image`: trains a flow from decoder-projected harmonic energy to Hilbert-flattened grayscale images.
- `energy_to_image_reflow`: distills a high-step frozen `energy_to_image` teacher into a low-step student flow, with 8 steps as the default target for later unrolled image-autoencoder work.
- `image_autoencoder`: trains the total end-to-end grayscale image autoencoder over 1D Hilbert-flattened image sequences, using image-to-energy flow, the LPAP surrogate/decoder inner energy path, and energy-to-image flow.

## Training Overview

```mermaid
flowchart TD
    config[Training TOML] --> train_notebook[notebooks/train.py]
    train_notebook --> dispatch[lpap.training_notebook]
    dispatch --> kinds{{Model kind}}
    kinds --> surrogate[surrogate]
    kinds --> decoder[decoder]
    kinds --> image_to_energy[image_to_energy]
    kinds --> energy_to_image[energy_to_image]
    kinds --> energy_to_image_reflow[energy_to_image_reflow]
    kinds --> image_autoencoder[image_autoencoder]
    surrogate & decoder & image_to_energy & energy_to_image & energy_to_image_reflow & image_autoencoder --> session[Training session]
    session --> checkpoint[(Checkpoint files)]
    session --> sqlite[(SQLite logs)]
```

The shared notebook handles configuration loading, previous-run discovery, rerun restoration, progress display, and loss plotting. Model-specific galleries live in the visualization notebooks. The model-specific training modules keep the parts that differ by model kind.

## Checkpoints And Logs

`TrainingRun` owns checkpoint and SQLite log updates for all model kinds.

```mermaid
flowchart LR
    step[Training step]
    metrics[Metric dict]
    record[TrainingRun.record_step]
    ckpt[Checkpoint]
    log[SQLite metrics]
    best[Best model state]

    step --> metrics --> record
    record --> log
    record --> best
    best --> ckpt
    record --> ckpt
```

Checkpoint payloads include:

- `model_state` and, when available, `best_model_state`
- optimizer state
- current and best metrics
- `training_state.run_config`
- `training_state.model_config`
- lightweight metadata such as run id and display name

SQLite logs include run configuration, metadata, attempts, scalar KPIs, and checkpoint paths. SQLite is informational and ergonomic; checkpoints are authoritative for model-dependent configuration.

This is a research repository. Local checkpoint and SQLite schemas are allowed to change, and stale artifacts should be regenerated instead of migrated unless migration is explicitly useful.

## Model Dependencies

```mermaid
flowchart TD
    harmonics_config[Surrogate TOML harmonics]
    surrogate_ckpt[Surrogate checkpoint]
    decoder_ckpt[Decoder checkpoint]
    image_flow_config[Image to energy TOML]
    energy_flow_config[Energy to image TOML]
    reflow_config[Energy to image reflow TOML]
    image_autoencoder_config[Image autoencoder TOML]

    harmonics_config --> surrogate_ckpt
    surrogate_ckpt --> decoder_ckpt
    surrogate_ckpt --> energy_flow_config
    decoder_ckpt --> energy_flow_config
    energy_flow_config --> e2i_teacher[High step E2I teacher]
    e2i_teacher --> reflow_config
    image_flow_config --> image_to_energy[Image to energy training]
    energy_flow_config --> energy_to_image[Energy to image training]
    reflow_config --> energy_to_image_reflow[Energy to image reflow training]
    image_flow_config --> image_autoencoder_config
    reflow_config --> image_autoencoder_config
    surrogate_ckpt --> image_autoencoder_config
    decoder_ckpt --> image_autoencoder_config
    image_autoencoder_config --> image_autoencoder[Image autoencoder training]

    surrogate_ckpt -. harmonic config .-> decoder_ckpt
    surrogate_ckpt -. harmonic config .-> energy_to_image
    decoder_ckpt -. decoder projection .-> energy_to_image
```

The decoder does not duplicate harmonic source settings in its TOML. It reads them from the surrogate checkpoint. `energy_to_image` follows the same rule: it samples harmonics from the surrogate checkpoint's stored run config, passes them through the frozen surrogate and decoder, and uses the decoder reconstruction as its source distribution.

`energy_to_image_reflow` keeps that same source distribution, freezes a trained `energy_to_image` flow as a high-step teacher, and trains a student flow by integrating the student for a smaller number of Euler midpoint steps. The default configuration uses a 64-step teacher target and an 8-step student rollout. Its checkpoint is still a plain `DilatedConvFlow1d` state dict, so later experiments can consume it wherever an energy-to-image flow is expected.

`image_autoencoder` is the total autoencoder. It Hilbert-flattens a grayscale image, rolls an image-to-energy flow forward for a small fixed number of differentiable steps, passes the encoded energy through the LPAP surrogate and decoder, then rolls an energy-to-image flow forward to reconstruct the image.

```mermaid
flowchart LR
    img[Grayscale image] --> i2e[Image-to-energy flow<br/>Euler rollout]
    i2e --> enc[Encoded energy]
    enc --> sur[LPAP surrogate]
    sur --> dec[LPAP decoder]
    dec --> den[Decoded energy]
    den --> e2i[Energy-to-image flow<br/>Euler rollout]
    e2i --> rec[Reconstructed image]

    sur -. "vs exact LPAP teacher" .-> ce["λ_ce · weighted teacher CE"]
    den -. "vs encoded energy" .-> el1["λ_energy · inner energy L1"]
    rec -. "vs input image" .-> il2["λ_image · image L2"]

    ce --> total((Total loss))
    el1 --> total
    il2 --> total
```

The training loss (`_forward_loss`) is a fixed-weight sum of three terms; there is
no weight schedule, so each weight λ is a constant prorating coefficient:

- **Image reconstruction L2** (`image_l2_weight`, default `1.0`): MSE between the reconstructed and input image. The primary objective.
- **Inner energy reconstruction L1** (`energy_l1_weight`, default `0.25`): mean absolute error between the decoder-reconstructed energy and the encoded energy. Keeps the LPAP surrogate/decoder path a faithful autoencoder of the encoded energy. The encoded-energy target can optionally be detached (`detach_energy_target`).
- **Surrogate teacher cross-entropy** (`surrogate_teacher_weight`, default `0.1`): the amplitude-weighted cross-entropy of `lpap_surrogate_loss` against the exact LPAP source indices, keeping the differentiable surrogate aligned with the exact operator.

The metric dict also logs the raw (unweighted) `image_reconstruction_l2`, `energy_reconstruction_l1`, `surrogate_teacher_ce`, and surrogate `weighted_accuracy`, plus RMS gauges for the encoded/decoded energy and input/reconstructed image.

## Flow Training Factorization

The two image/energy flow modules share one implementation spine in `lpap.flow_training`.

```mermaid
flowchart TD
    shared[lpap.flow_training]
    shared --> cfg[Shared config dataclasses]
    shared --> data[Image loading and Hilbert flattening]
    shared --> time[Beta or uniform time sampling]
    shared --> core[Flow session core]
    shared --> loss[Flow matching train and eval]
    shared --> diag[Integration diagnostics]

    cfg & data & time & core & loss & diag --> image_module[lpap.image_to_energy_training]
    cfg & data & time & core & loss & diag --> energy_module[lpap.energy_to_image_training]
```

The direction-specific modules still own the parts that are genuinely different:

- `image_to_energy_training.py` owns image source preparation and direct synthetic harmonic targets.
- `energy_to_image_training.py` owns surrogate/decoder checkpoint loading and decoder-projected harmonic sources.

## Notebooks

Use Pixi tasks from the repository root:

```sh
pixi run notebook-train
pixi run notebook-synthetic
pixi run notebook-surrogate
pixi run notebook-decoder
pixi run notebook-image-to-energy
pixi run notebook-energy-to-image
pixi run notebook-energy-to-image-reflow
pixi run notebook-image-autoencoder
```

The visualization notebooks select logged runs from SQLite, load the corresponding checkpoint, and render model-specific galleries. The flow visualizers show integration results at multiple Euler midpoint step counts. The reflow visualizer compares source energy, the high-step teacher image, the low-step student image, a sampled image anchor, and the student-teacher error. The image autoencoder visualizer compares grayscale input/reconstruction/error and encoded/decoded energy/error.

## Testing

The current test suite covers the LPAP operator, surrogate and decoder behavior, shared logging/checkpointing, Hilbert image ordering, flow matching utilities, training notebook dispatch, gallery rendering, small CPU training loops for both flow directions, energy-to-image reflow, and the total image autoencoder.

Run all tests with:

```sh
pixi run test
```
