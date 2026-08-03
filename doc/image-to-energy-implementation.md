# Image-Energy Flow Implementation

`image_energy_flow` is one bidirectional flow on `t∈[-1, 1]`:

- Normalized Hilbert-flattened images occupy `t=-1`.
- A signed log-normal energy prior occupies `t=0`:
  `|e| = scale · exp(σ Z)` with `Z~N(0,1)` (so `scale` is the median of `|e|`),
  and independent fair-coin signs (``P(+) = 0.5``). Configurable via TOML
  ``[prior]`` ``sigma`` and ``scale``.
- The return image endpoint occupies `t=+1`.

Training samples both branches of this path. Image encoding integrates from
`IMAGE_TO_ENERGY_T0=-1` to `IMAGE_TO_ENERGY_T1=0`; decoding integrates from
`ENERGY_TO_IMAGE_T0=0` to `ENERGY_TO_IMAGE_T1=1`.

The relevant durable surfaces are:

- TOML: `configs/training/image_energy_flow.toml`
- Training module: `src/lpap/image_energy_flow_training.py`
- Test module: `test/test_image_energy_flow_training.py`
- Gallery renderer: `render_image_energy_flow_gallery_html`

After training, encode the image dataset once through the flow’s i2e branch to
build an empirical energy bank for surrogate/decoder teachers. The image
autoencoder loads one `flow_checkpoint_name` and clones its state into its
image→energy and energy→image branches at initialization.
