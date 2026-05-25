import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import sys
    from pathlib import Path

    import marimo as mo
    import torch

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from lpap.data import SyntheticHarmonicConfig

    return SyntheticHarmonicConfig, html, mo, torch


@app.cell
def _(mo):
    mo.md("""
    # Synthetic harmonic sample
    """)
    return


@app.cell
def _(SyntheticHarmonicConfig, torch):
    sample_count = 1024
    seed = 1234
    harmonics = SyntheticHarmonicConfig(
        harmonic_count=12,
        gain_variance=1.0,
        gain_half_life=4.0,
        spikiness_range=(4.0, 8.0),
        dtype=torch.float32,
    )
    return harmonics, sample_count, seed


@app.cell
def _(harmonics, sample_count, seed, torch):
    generator = torch.Generator().manual_seed(seed)

    sample = harmonics.sample_batch(
        batch_size=1,
        n=sample_count,
        generator=generator,
        return_parameters=True,
    )
    amplitudes = sample["values"][0].detach().cpu()
    return (amplitudes,)


@app.cell
def _(amplitudes, html):
    def signed_pixel_image(values):
        grid = values.reshape(32, 32)
        max_abs = float(grid.abs().max().clamp_min(1.0e-12))

        pixels = []
        for amplitude in grid.flatten().tolist():
            scaled = max(-1.0, min(1.0, amplitude / max_abs))
            red = round(255 * max(scaled, 0.0))
            blue = round(255 * max(-scaled, 0.0))
            title = html.escape(f"{amplitude:.4f}")
            pixels.append(
                f'<div title="{title}" style="background: rgb({red}, 0, {blue});"></div>'
            )

        return "".join(pixels)

    def amplitude_curve(values):
        width = 640
        height = 220
        padding = 24
        max_abs = float(values.abs().max().clamp_min(1.0e-12))
        points = []
        for index, amplitude in enumerate(values.tolist()):
            x = padding + index * (width - 2 * padding) / (len(values) - 1)
            scaled = max(-1.0, min(1.0, amplitude / max_abs))
            y = height / 2 - scaled * (height / 2 - padding)
            points.append(f"{x:.2f},{y:.2f}")
        escaped_points = html.escape(" ".join(points), quote=True)
        zero_y = height / 2

        return f"""
        <svg viewBox="0 0 {width} {height}" width="100%" height="220" role="img" aria-label="Harmonic amplitude curve from x equals 0 to x equals 1">
          <rect x="0" y="0" width="{width}" height="{height}" fill="#090a0d" />
          <line x1="{padding}" y1="{zero_y}" x2="{width - padding}" y2="{zero_y}" stroke="#4b5563" stroke-width="1" />
          <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#4b5563" stroke-width="1" />
          <line x1="{width - padding}" y1="{padding}" x2="{width - padding}" y2="{height - padding}" stroke="#30333a" stroke-width="1" />
          <text x="{padding}" y="{height - 6}" fill="#aab0bd" font-size="11" font-family="system-ui, sans-serif">0</text>
          <text x="{width - padding - 8}" y="{height - 6}" fill="#aab0bd" font-size="11" font-family="system-ui, sans-serif">1</text>
          <text x="{padding + 4}" y="{padding - 8}" fill="#aab0bd" font-size="11" font-family="system-ui, sans-serif">amplitude</text>
          <polyline points="{escaped_points}" fill="none" stroke="#52d1ff" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" />
        </svg>
        """

    image_pixels = signed_pixel_image(amplitudes)
    curve_svg = amplitude_curve(amplitudes)
    return curve_svg, image_pixels


@app.cell
def _(amplitudes, curve_svg, harmonics, image_pixels, mo, sample_count, seed):
    max_abs_amplitude = float(amplitudes.abs().max())
    mean_amplitude = float(amplitudes.mean())
    spikiness_min, spikiness_max = harmonics.spikiness_range

    mo.vstack(
        [
            mo.Html(
                f"""
                                <div style="display: grid; gap: 18px; font: 14px/1.45 system-ui, sans-serif; color: #d7dae0;">
                                    <div style="width: min(100%, 720px); border: 1px solid #30333a; background: #090a0d;">
                                        {curve_svg}
                                    </div>
                                    <div style="display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap;">
                                        <div
                                            style="
                                                display: grid;
                                                grid-template-columns: repeat(32, 10px);
                                                grid-template-rows: repeat(32, 10px);
                                                width: 320px;
                                                height: 320px;
                                                border: 1px solid #30333a;
                                                background: #000;
                                            "
                                        >{image_pixels}</div>
                                        <div style="min-width: 180px;">
                                            <div><strong>N</strong>: {sample_count}</div>
                                            <div><strong>seed</strong>: {seed}</div>
                                            <div><strong>harmonics</strong>: {harmonics.harmonic_count}</div>
                                            <div><strong>gain variance</strong>: {harmonics.gain_variance:.4f}</div>
                                            <div><strong>gain half-life</strong>: {harmonics.gain_half_life:.4f}</div>
                                            <div><strong>spikiness</strong>: {spikiness_min:.4f} to {spikiness_max:.4f}</div>
                                            <div><strong>x range</strong>: 0 to 1</div>
                                            <div><strong>image</strong>: 32 x 32</div>
                                            <div><strong>max |amplitude|</strong>: {max_abs_amplitude:.4f}</div>
                                            <div><strong>mean amplitude</strong>: {mean_amplitude:.4f}</div>
                                            <div style="margin-top: 12px; display: flex; align-items: center; gap: 8px;">
                                                <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
                                                <span>negative / zero / positive</span>
                                            </div>
                    </div>
                  </div>
                </div>
                """
            )
        ]
    )
    return


if __name__ == "__main__":
    app.run()
