# Image-Energy Flow Implementation

`image_energy_flow` replaces the former one-way image-to-energy and
energy-to-image training stacks with one bidirectional flow.

- Normalized Hilbert-flattened images occupy `t=-1`.
- The energy marginal (synthetic harmonics or an empirical energy bank) occupies
  `t=0`.
- The return image endpoint occupies `t=+1`.

Training samples both branches of this path. Image encoding integrates from
`IMAGE_TO_ENERGY_T0=-1` to `IMAGE_TO_ENERGY_T1=0`; decoding integrates from
`ENERGY_TO_IMAGE_T0=0` to `ENERGY_TO_IMAGE_T1=1`.

The relevant durable surfaces are:

- TOML: `configs/training/image_energy_flow.toml` and
  `configs/training/image_energy_flow_energy_bank.toml`
- Training module: `src/lpap/image_energy_flow_training.py`
- Test module: `test/test_image_energy_flow_training.py`
- Gallery renderer: `render_image_energy_flow_gallery_html`

The image autoencoder loads one `flow_checkpoint_name` and clones its state into
its image→energy and energy→image branches at initialization.
