# Training Stack

See the [documentation index](index.md) for the full documentation map and the [glossary](glossary.md) for project terminology.

LPAP currently wires these trainable model kinds into the shared marimo training
notebook (energy-bank variants share the corresponding flow backend):

- `surrogate`: learns full-`N` source-index logits for LPAP bucket selections on synthetic harmonic energy.
- `decoder`: reconstructs source energy values from frozen surrogate logits.
- `image_to_energy` / `image_to_energy_energy_bank`: flow from Hilbert-flattened grayscale images to an energy prior (synthetic harmonics or an empirical energy bank).
- `energy_to_image` / `energy_to_image_energy_bank`: flow from an energy prior (decoder-projected harmonics or an empirical energy bank) to Hilbert-flattened grayscale images.
- `image_autoencoder`: end-to-end grayscale AE (image-to-energy → LPAP surrogate/decoder → energy-to-image).

## Training Overview

```mermaid
flowchart TD
    config[Training TOML] --> train_notebook[notebooks/train.py]
    train_notebook --> dispatch[lpap.training_notebook]
    dispatch --> kinds{{Model kind}}
    kinds --> surrogate[surrogate]
    kinds --> decoder[decoder]
    kinds --> image_to_energy[image_to_energy (+ energy_bank)]
    kinds --> energy_to_image[energy_to_image (+ energy_bank)]
    kinds --> image_autoencoder[image_autoencoder]
    surrogate & decoder & image_to_energy & energy_to_image & image_autoencoder --> session[Training session]
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
    energy_bank[Empirical energy bank .pt]
    image_autoencoder_config[Image autoencoder TOML]

    harmonics_config --> surrogate_ckpt
    surrogate_ckpt --> decoder_ckpt
    surrogate_ckpt --> energy_flow_config
    decoder_ckpt --> energy_flow_config
    energy_bank --> image_flow_config
    energy_bank --> energy_flow_config
    image_flow_config --> image_to_energy[Image to energy training]
    energy_flow_config --> energy_to_image[Energy to image training]
    image_flow_config --> image_autoencoder_config
    energy_flow_config --> image_autoencoder_config
    surrogate_ckpt --> image_autoencoder_config
    decoder_ckpt --> image_autoencoder_config
    image_autoencoder_config --> image_autoencoder[Image autoencoder training]

    surrogate_ckpt -. harmonic config .-> decoder_ckpt
    surrogate_ckpt -. harmonic config .-> energy_to_image
    decoder_ckpt -. decoder projection .-> energy_to_image
```

The decoder does not duplicate harmonic source settings in its TOML. It reads them from the surrogate checkpoint. In harmonics mode, `energy_to_image` follows the same rule: it samples harmonics from the surrogate checkpoint's stored run config, passes them through the frozen surrogate and decoder (`source.teacher`), and uses the decoder reconstruction as its source distribution. In `energy_bank` mode it samples empirical energies directly and skips surrogate/decoder loading.

Both image flows can also target/source an empirical energy bank (see `configs/training/*_energy_bank.toml` and `lpap.energy_bank`). Bank rows are sampled independently of image batches so training learns the energy *marginal*, not a paired joint map.

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
    enc -. "signed-mass gap/floor" .-> sm["λ_signed · signed-mass"]

    ce --> total((Total loss))
    el1 --> total
    il2 --> total
    sm --> total
```

The training loss (`_forward_loss`) is a fixed-weight sum; there is no weight
schedule, so each λ is a constant prorating coefficient (defaults below match
`configs/training/image_autoencoder.toml`):

- **Image reconstruction L2** (`image_l2_weight`, default `1.0`): MSE between the reconstructed and input image. The primary objective.
- **Inner energy reconstruction L1** (`energy_l1_weight`, default `0.5`): mean absolute error between the decoder-reconstructed energy and the encoded energy. Keeps the LPAP path a faithful autoencoder of the encoded energy. The encoded-energy target can optionally be detached (`detach_energy_target`).
- **Surrogate teacher cross-entropy** (`surrogate_teacher_weight`, default `0.05`): amplitude-weighted CE of `lpap_surrogate_loss` against exact LPAP source indices.
- **Signed-mass gap/floor** (`signed_mass_balance_weight`, default `0.02`; see `lpap.image_autoencoder_loss`): on encoded energy `e`, with `m± = mean(relu(±e))` and scale `tau` (`signed_mass_floor_tau`, default `0.01`):

```text
L_gap   = ((m+ - m-) / tau)^2
L_floor = (relu(tau - m+)/tau)^2 + (relu(tau - m-)/tau)^2
L       = L_gap + floor_coef * L_floor
```

  Gap is scaled by `tau` (not by `m+ + m-`), so collapsing `e → 0` is not a free
  “balanced” win; the floor pushes each side toward ~`tau`.

Current AE defaults also use **16-step** Euler on both flows with the
`energy_to_image.pt` teacher, and a longer default budget of **20k** steps.

### Dialing AE lambdas

Use the loss probe before long e2e runs:

```sh
pixi run ae-loss-probe
```

That prints unweighted vs weighted contributions and can `--suggest` weights so
secondary terms sit at chosen shares of image L2. Re-probe after short trains;
validate with the molab L→R gallery (spatial image panels + encoded/decoded
energy signs) before committing a long run.

The metric dict also logs raw (unweighted) reconstruction terms, surrogate
`weighted_accuracy`, signed-mass gap/floor/imbalance and per-side masses, plus
RMS gauges for encoded/decoded energy and input/reconstructed image.

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

- `image_to_energy_training.py` owns image source preparation and energy prior targets (harmonics or bank).
- `energy_to_image_training.py` owns energy prior sources (nested harmonics teacher or bank).

## Notebooks

Use Pixi tasks from the repository root:

```sh
pixi run notebook-train
pixi run notebook-synthetic
pixi run notebook-surrogate
pixi run notebook-decoder
pixi run notebook-image-to-energy
pixi run notebook-energy-to-image
pixi run notebook-image-autoencoder
```

The visualization notebooks select logged runs from SQLite, load the corresponding checkpoint, and render model-specific galleries. The flow visualizers show integration results at multiple Euler midpoint step counts. The image autoencoder visualizer compares grayscale input/reconstruction/error and encoded/decoded energy/error.

For multi-hour AE runs on remote GPUs, prefer the detached molab launcher rather
than long local notebook cells — see [molab workflow](molab-workflow.md).
`TrainingRun` supports `upload_artifacts_on_checkpoint` and `notify_on_finished`
(Pushover via env / secrets inject).

## Testing

The suite covers the LPAP operator, surrogate/decoder behavior, logging/checkpoints, Hilbert ordering, flow matching, energy-bank priors, notebook dispatch, galleries, small CPU flow/AE loops, storage config, artifact sync helpers, and notify. Repo-top `molab/` helpers have their own unit tests (`PYTHONPATH=src:.`).

```sh
pixi run test
```
