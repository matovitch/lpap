from __future__ import annotations

import base64
import re
import struct
import zlib
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import torch


def _metric_points(
    rows: Sequence[Mapping[str, Any]], metric_name: str
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        value = row.get(metric_name)
        if value is not None:
            points.append((float(row["step"]), float(value)))
    return points


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _legend_item(
    *,
    x: int,
    y: int,
    color: str,
    label: str,
    dash: str = "",
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x}" y1="{y}" x2="{x + 16}" y2="{y}" '
        f'stroke="{color}" stroke-width="3"{dash_attribute} />'
        f'<text x="{x + 22}" y="{y + 5}" fill="#222" font-size="12">'
        f"{escape(label)}</text>"
    )


def render_loss_history_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_metric: str = "loss",
    validation_metric: str = "validation_loss",
    validation_regularizer_metrics: Sequence[str] = (),
    width: int = 720,
    height: int = 280,
) -> str:
    train = _metric_points(rows, train_metric)
    validation = _metric_points(rows, validation_metric)
    regularizers = [
        (metric_name, _metric_points(rows, metric_name))
        for metric_name in validation_regularizer_metrics
    ]
    regularizers = [(name, points) for name, points in regularizers if points]
    if not train and not validation and not regularizers:
        return "<p>No loss history has been logged yet.</p>"

    all_points = [
        *train,
        *validation,
        *(point for _name, points in regularizers for point in points),
    ]
    min_step = min(step for step, _value in all_points)
    max_step = max(step for step, _value in all_points)
    min_loss = min(value for _step, value in all_points)
    max_loss = max(value for _step, value in all_points)
    if min_step == max_step:
        min_step -= 1.0
        max_step += 1.0
    if min_loss == max_loss:
        min_loss -= 0.5
        max_loss += 0.5

    left = 56
    right = 18
    top = 20
    bottom = 38
    plot_width = width - left - right
    plot_height = height - top - bottom

    def project(point: tuple[float, float]) -> tuple[float, float]:
        step, value = point
        x = left + (step - min_step) / (max_step - min_step) * plot_width
        y = top + (max_loss - value) / (max_loss - min_loss) * plot_height
        return x, y

    train_svg_points = [project(point) for point in train]
    validation_svg_points = [project(point) for point in validation]
    regularizer_svg_series = [
        (name, [project(point) for point in points]) for name, points in regularizers
    ]
    attempts: list[tuple[float, Any]] = []
    previous_attempt = None
    for row in rows:
        attempt_id = row.get("attempt_id")
        if attempt_id is not None and previous_attempt is not None:
            if attempt_id != previous_attempt:
                attempts.append((float(row["step"]), attempt_id))
        if attempt_id is not None:
            previous_attempt = attempt_id

    resume_lines = []
    for step, attempt_id in attempts:
        x, _y = project((step, min_loss))
        resume_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + plot_height}" stroke="#777" stroke-dasharray="4 4" />'
            f'<text x="{x + 4:.2f}" y="{top + 12}" fill="#555" font-size="11">'
            f"attempt {escape(str(attempt_id))}</text>"
        )

    validation_circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2563eb" />'
        for x, y in validation_svg_points
    )
    train_line = (
        f'<polyline fill="none" stroke="#92400e" stroke-width="1.5" '
        f'points="{_polyline(train_svg_points)}" />'
        if len(train_svg_points) >= 2
        else ""
    )
    validation_line = (
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" '
        f'points="{_polyline(validation_svg_points)}" />'
        if len(validation_svg_points) >= 2
        else ""
    )
    regularizer_colors = ("#7c3aed", "#059669", "#dc2626", "#0891b2")
    regularizer_lines = []
    for index, (_name, points) in enumerate(regularizer_svg_series):
        color = regularizer_colors[index % len(regularizer_colors)]
        if len(points) >= 2:
            regularizer_lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.8" '
                f'stroke-dasharray="5 4" points="{_polyline(points)}" />'
            )
        elif len(points) == 1:
            x, y = points[0]
            regularizer_lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" />'
            )
    if len(train_svg_points) == 1:
        x, y = train_svg_points[0]
        train_line += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#92400e" />'

    legend_items = [
        ("#92400e", "train loss", ""),
        ("#2563eb", "validation loss", ""),
        *(
            (
                regularizer_colors[index % len(regularizer_colors)],
                name.removeprefix("validation_").replace("_", " "),
                "5 4",
            )
            for index, (name, _points) in enumerate(regularizer_svg_series)
        ),
    ]
    legend = "".join(
        _legend_item(
            x=left + 12 + index * 156,
            y=top + 13,
            color=color,
            label=label,
            dash=dash,
        )
        for index, (color, label, dash) in enumerate(legend_items)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Train and validation loss history">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
      <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" />
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" />
      {"".join(resume_lines)}
      {train_line}
      {validation_line}
    {"".join(regularizer_lines)}
      {validation_circles}
      <text x="{left}" y="{height - 10}" fill="#222" font-size="12">step {min_step:.0f}</text>
      <text x="{left + plot_width - 62}" y="{height - 10}" fill="#222" font-size="12">step {max_step:.0f}</text>
      <text x="8" y="{top + 4}" fill="#222" font-size="12">{max_loss:.3f}</text>
      <text x="8" y="{top + plot_height}" fill="#222" font-size="12">{min_loss:.3f}</text>
            {legend}
    </svg>
    """


def _signed_pixels(values: torch.Tensor, *, size: int, max_abs: float) -> str:
    pixels = []
    for amplitude in values.reshape(size * size).tolist():
        scaled = max(-1.0, min(1.0, float(amplitude) / max_abs))
        red = round(255 * max(scaled, 0.0))
        blue = round(255 * max(-scaled, 0.0))
        pixels.append(
            f'<div title="{escape(f"{float(amplitude):.4f}")}" '
            f'style="background: rgb({red}, 0, {blue});"></div>'
        )
    return "".join(pixels)


def _grayscale_pixels(values: torch.Tensor, *, size: int) -> str:
    pixels = []
    for amplitude in values.reshape(size * size).tolist():
        level = round(255 * max(0.0, min(1.0, float(amplitude))))
        pixels.append(
            f'<div title="{escape(f"{float(amplitude):.4f}")}" '
            f'style="background: rgb({level}, {level}, {level});"></div>'
        )
    return "".join(pixels)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _rgb_png_data_uri(rgb: bytes, *, width: int, height: int) -> str:
    if len(rgb) != width * height * 3:
        raise ValueError("rgb buffer length must equal width*height*3")
    rows = b"".join(
        b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _apply_display_gamma(scaled: float, *, gamma: float) -> float:
    """Apply gamma in normalized space; ``gamma < 1`` lifts small magnitudes."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if gamma == 1.0:
        return scaled
    sign = 1.0 if scaled >= 0.0 else -1.0
    return sign * (abs(scaled) ** gamma)


def _grayscale_png_img(
    values: torch.Tensor,
    *,
    size: int,
    display_px: int,
    label: str,
    gamma: float = 1.0,
) -> str:
    flat = values.detach().cpu().reshape(size * size)
    rgb = bytearray()
    for amplitude in flat.tolist():
        unit = max(0.0, min(1.0, float(amplitude)))
        level = round(255 * _apply_display_gamma(unit, gamma=gamma))
        rgb.extend((level, level, level))
    uri = _rgb_png_data_uri(bytes(rgb), width=size, height=size)
    return f"""
        <div style="display: grid; gap: 4px;">
            <div style="font-weight: 600;">{escape(label)}</div>
            <img src="{uri}" width="{display_px}" height="{display_px}"
                 style="image-rendering: pixelated; border: 1px solid #30333a; background: #000;"
                 alt="{escape(label)}" />
        </div>
        """


def _signed_png_img(
    values: torch.Tensor,
    *,
    size: int,
    max_abs: float,
    display_px: int,
    label: str,
    gamma: float = 1.0,
) -> str:
    flat = values.detach().cpu().reshape(size * size)
    scale = max(float(max_abs), 1.0e-12)
    rgb = bytearray()
    for amplitude in flat.tolist():
        scaled = max(-1.0, min(1.0, float(amplitude) / scale))
        shaped = _apply_display_gamma(scaled, gamma=gamma)
        red = round(255 * max(shaped, 0.0))
        blue = round(255 * max(-shaped, 0.0))
        rgb.extend((red, 0, blue))
    uri = _rgb_png_data_uri(bytes(rgb), width=size, height=size)
    return f"""
        <div style="display: grid; gap: 4px;">
            <div style="font-weight: 600;">{escape(label)}</div>
            <img src="{uri}" width="{display_px}" height="{display_px}"
                 style="image-rendering: pixelated; border: 1px solid #30333a; background: #000;"
                 alt="{escape(label)}" />
        </div>
        """


_AE_GALLERY_PAIR_ORDER = ("c512_k32", "c256_k24", "c128_k16", "c256", "c128")


def _ae_gallery_pair_bucket_count(name: str) -> int | None:
    """Parse ``C`` from names like ``c512_k32`` / ``c128``; ``None`` if unknown."""
    match = re.fullmatch(r"c(\d+)(?:_.*)?", str(name).strip())
    if match is None:
        return None
    return int(match.group(1))


def _ordered_ae_gallery_pairs(pairs: Sequence[Any]) -> list[Any]:
    """Order pairs C-descending so gallery reads source → fine → coarse."""
    indexed = list(enumerate(pairs))

    def sort_key(item: tuple[int, Any]) -> tuple[int, int, int]:
        index, pair = item
        name = str(getattr(pair, "name", ""))
        bucket_count = _ae_gallery_pair_bucket_count(name)
        if bucket_count is not None:
            # Higher C first; stable by original index.
            return (0, -bucket_count, index)
        try:
            preferred = _AE_GALLERY_PAIR_ORDER.index(name)
        except ValueError:
            preferred = len(_AE_GALLERY_PAIR_ORDER)
        return (1, preferred, index)

    return [pair for _, pair in sorted(indexed, key=sort_key)]


def render_signed_triplet_gallery_html(
    items: Sequence[Any],
    *,
    size: int = 32,
    display_px: int = 128,
    gamma: float = 1.0,
) -> str:
    if not items:
        return "<p>No gallery samples are available.</p>"
    if display_px <= 0:
        raise ValueError("display_px must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    panels = []
    labels = ("source energy", "oracle LPAP", "surrogate hard", "decoder soft")
    keys = ("energy", "lpap", "surrogate_hard", "decoder")
    for item_index, item in enumerate(items, start=1):
        tensors = [getattr(item, key).detach().cpu().reshape(-1) for key in keys]
        expected_count = size * size
        if any(tensor.numel() != expected_count for tensor in tensors):
            raise ValueError(f"gallery tensors must contain {expected_count} values")
        max_abs = max(
            float(tensor.abs().max().clamp_min(1.0e-12)) for tensor in tensors
        )
        tiles = "".join(
            _signed_png_img(
                tensor,
                size=size,
                max_abs=max_abs,
                display_px=display_px,
                label=label,
                gamma=gamma,
            )
            for label, tensor in zip(labels, tensors, strict=True)
        )
        panels.append(
            f"""
            <div style="display: grid; gap: 10px;">
              <div style="font-weight: 700;">sample {item_index}</div>
              <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start;">{tiles}</div>
            </div>
            """
        )

    return f"""
    <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
      {"".join(panels)}
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
        <span>negative / zero / positive, scaled per sample triplet (γ={gamma:g})</span>
      </div>
    </div>
    """


def render_image_to_energy_gallery_html(
    items: Sequence[Any],
    *,
    steps: Sequence[int] = (64, 32, 16, 8, 4),
    size: int = 32,
    display_px: int = 128,
    gamma: float = 1.0,
) -> str:
    if not items:
        return "<p>No image-to-energy gallery samples are available.</p>"
    if display_px <= 0:
        raise ValueError("display_px must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    expected_count = size * size
    panels = []
    for item_index, item in enumerate(items, start=1):
        image = item.image.detach().cpu().reshape(-1)
        if image.numel() != expected_count:
            raise ValueError(f"gallery images must contain {expected_count} values")
        generated = {
            int(step_count): item.generated[int(step_count)].detach().cpu().reshape(-1)
            for step_count in steps
        }
        if any(tensor.numel() != expected_count for tensor in generated.values()):
            raise ValueError(
                f"generated energy tensors must contain {expected_count} values"
            )
        max_abs = max(
            float(tensor.abs().max().clamp_min(1.0e-12))
            for tensor in generated.values()
        )
        tiles = [
            _grayscale_png_img(
                image,
                size=size,
                display_px=display_px,
                label="image",
                gamma=gamma,
            )
        ]
        tiles.extend(
            _signed_png_img(
                generated[step_count],
                size=size,
                max_abs=max_abs,
                display_px=display_px,
                label=f"{step_count} steps",
                gamma=gamma,
            )
            for step_count in steps
        )
        panels.append(
            f"""
            <div style="display: grid; gap: 10px;">
              <div style="font-weight: 700;">sample {item_index}</div>
              <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start;">{"".join(tiles)}</div>
            </div>
            """
        )

    return f"""
        <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
            {"".join(panels)}
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
                <span>energy: negative / zero / positive, scaled per sample row</span>
            </div>
        </div>
        """


def render_energy_to_image_gallery_html(
    items: Sequence[Any],
    *,
    steps: Sequence[int] = (64, 32, 16, 8, 4),
    size: int = 32,
    display_px: int = 128,
    gamma: float = 1.0,
) -> str:
    if not items:
        return "<p>No energy-to-image gallery samples are available.</p>"
    if display_px <= 0:
        raise ValueError("display_px must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    expected_count = size * size
    panels = []
    for item_index, item in enumerate(items, start=1):
        source = item.source.detach().cpu().reshape(-1)
        if source.numel() != expected_count:
            raise ValueError(f"gallery sources must contain {expected_count} values")
        generated = {
            int(step_count): item.generated[int(step_count)].detach().cpu().reshape(-1)
            for step_count in steps
        }
        if any(tensor.numel() != expected_count for tensor in generated.values()):
            raise ValueError(f"generated images must contain {expected_count} values")
        max_abs = float(source.abs().max().clamp_min(1.0e-12))
        tiles = [
            _signed_png_img(
                source,
                size=size,
                max_abs=max_abs,
                display_px=display_px,
                label="source",
                gamma=gamma,
            )
        ]
        tiles.extend(
            _grayscale_png_img(
                generated[step_count],
                size=size,
                display_px=display_px,
                label=f"{step_count} steps",
                gamma=gamma,
            )
            for step_count in steps
        )
        panels.append(
            f"""
            <div style="display: grid; gap: 10px;">
              <div style="font-weight: 700;">sample {item_index}</div>
              <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start;">{"".join(tiles)}</div>
            </div>
            """
        )

    return f"""
        <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
            {"".join(panels)}
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
            <span>source: negative / zero / positive, scaled per sample row</span>
            </div>
        </div>
        """


def render_image_energy_flow_gallery_html(
    items: Sequence[Any],
    *,
    steps: Sequence[int] = (64, 32, 16, 8, 4),
    size: int = 32,
    display_px: int = 128,
    gamma: float = 1.0,
) -> str:
    """Render both legs of a bidirectional image/energy flow gallery."""
    if not items:
        return "<p>No image-energy flow gallery samples are available.</p>"

    image_to_energy_items = [
        type(
            "ImageEnergyFlowEncodedGalleryItem",
            (),
            {"image": item.image, "generated": item.encoded},
        )()
        for item in items
    ]
    round_trip_items = [
        type(
            "ImageEnergyFlowRoundTripGalleryItem",
            (),
            {
                "source": item.encoded[max(item.encoded)],
                "generated": item.reconstructed,
            },
        )()
        for item in items
    ]
    prior_items = [
        type(
            "ImageEnergyFlowPriorGalleryItem",
            (),
            {
                "source": item.prior_energy,
                "generated": item.from_prior,
            },
        )()
        for item in items
    ]
    return (
        "<h3>image → energy (t: −1 → 0)</h3>"
        + render_image_to_energy_gallery_html(
            image_to_energy_items,
            steps=steps,
            size=size,
            display_px=display_px,
            gamma=gamma,
        )
        + "<h3>round-trip energy → image (t: 0 → +1 from encoded)</h3>"
        + render_energy_to_image_gallery_html(
            round_trip_items,
            steps=steps,
            size=size,
            display_px=display_px,
            gamma=gamma,
        )
        + "<h3>prior energy → image (t: 0 → +1)</h3>"
        + render_energy_to_image_gallery_html(
            prior_items,
            steps=steps,
            size=size,
            display_px=display_px,
            gamma=gamma,
        )
    )


def render_image_autoencoder_gallery_html(
    items: Sequence[Any],
    *,
    size: int = 32,
    display_px: int = 154,
    gamma: float = 1.0,
) -> str:
    if not items:
        return "<p>No image autoencoder gallery samples are available.</p>"
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    expected_count = size * size
    panels = []
    for item_index, item in enumerate(items, start=1):
        pairs = _ordered_ae_gallery_pairs(getattr(item, "pairs", ()))
        if not pairs:
            # Legacy single-pair gallery items (flat fields).
            pairs = [
                type(
                    "LegacyPair",
                    (),
                    {
                        "name": "pair0",
                        "decoded_energy": item.decoded_energy,
                        "reconstructed_image": item.reconstructed_image,
                        "energy_error": item.energy_error,
                        "image_error": item.image_error,
                    },
                )()
            ]
        image = item.image.detach().cpu().reshape(-1)
        encoded_energy = item.encoded_energy.detach().cpu().reshape(-1)
        if image.numel() != expected_count or encoded_energy.numel() != expected_count:
            raise ValueError(f"gallery tensors must contain {expected_count} values")
        for pair in pairs:
            for tensor in (
                pair.decoded_energy,
                pair.reconstructed_image,
                pair.energy_error,
                pair.image_error,
            ):
                if tensor.detach().cpu().reshape(-1).numel() != expected_count:
                    raise ValueError(
                        f"gallery tensors must contain {expected_count} values"
                    )

        energy_max_abs = max(
            float(encoded_energy.abs().max().clamp_min(1.0e-12)),
            *(
                float(pair.decoded_energy.detach().cpu().abs().max().clamp_min(1.0e-12))
                for pair in pairs
            ),
        )
        energy_error_max_abs = max(
            (
                float(pair.energy_error.detach().cpu().abs().max().clamp_min(1.0e-12))
                for pair in pairs
            ),
            default=1.0e-12,
        )
        image_error_max_abs = max(
            (
                float(pair.image_error.detach().cpu().abs().max().clamp_min(1.0e-12))
                for pair in pairs
            ),
            default=1.0e-12,
        )

        energy_row = [
            _signed_png_img(
                encoded_energy,
                size=size,
                max_abs=energy_max_abs,
                display_px=display_px,
                label="source energy",
                gamma=gamma,
            )
        ]
        energy_row.extend(
            _signed_png_img(
                pair.decoded_energy,
                size=size,
                max_abs=energy_max_abs,
                display_px=display_px,
                label=f"{pair.name} energy",
                gamma=gamma,
            )
            for pair in pairs
        )
        energy_row.extend(
            _signed_png_img(
                pair.energy_error,
                size=size,
                max_abs=energy_error_max_abs,
                display_px=display_px,
                label=f"Δ energy {pair.name}",
                gamma=gamma,
            )
            for pair in pairs
        )

        image_row = [
            _grayscale_png_img(
                image,
                size=size,
                display_px=display_px,
                label="source image",
                gamma=gamma,
            )
        ]
        image_row.extend(
            _grayscale_png_img(
                pair.reconstructed_image,
                size=size,
                display_px=display_px,
                label=f"{pair.name} image",
                gamma=gamma,
            )
            for pair in pairs
        )
        image_row.extend(
            _signed_png_img(
                pair.image_error,
                size=size,
                max_abs=image_error_max_abs,
                display_px=display_px,
                label=f"Δ image {pair.name}",
                gamma=gamma,
            )
            for pair in pairs
        )

        panels.append(
            f"""
            <div style="display: grid; gap: 10px;">
                <div style="font-weight: 700;">sample {item_index}</div>
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">energy</div>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start;">
                        {"".join(energy_row)}
                    </div>
                </div>
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">image</div>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start;">
                        {"".join(image_row)}
                    </div>
                </div>
            </div>
            """
        )

    return f"""
        <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
            {"".join(panels)}
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
                <span>energy then image: source → higher-C → lower-C → diffs (signed: − / 0 / +); display γ={gamma:g} (&lt;1 lifts small values)</span>
            </div>
        </div>
        """
