from __future__ import annotations

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


def render_signed_triplet_gallery_html(items: Sequence[Any], *, size: int = 32) -> str:
    if not items:
        return "<p>No gallery samples are available.</p>"

    panels = []
    labels = ("harmonics", "LPAP selected", "decoder")
    keys = ("harmonics", "lpap", "decoder")
    for item_index, item in enumerate(items, start=1):
        tensors = [getattr(item, key).detach().cpu().reshape(-1) for key in keys]
        expected_count = size * size
        if any(tensor.numel() != expected_count for tensor in tensors):
            raise ValueError(f"gallery tensors must contain {expected_count} values")
        max_abs = max(
            float(tensor.abs().max().clamp_min(1.0e-12)) for tensor in tensors
        )
        grids = "".join(
            f"""
            <div style="display: grid; gap: 6px;">
              <div style="font-weight: 600;">{escape(label)}</div>
              <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                {_signed_pixels(tensor, size=size, max_abs=max_abs)}
              </div>
            </div>
            """
            for label, tensor in zip(labels, tensors, strict=True)
        )
        panels.append(
            f"""
            <div style="display: grid; gap: 10px;">
              <div style="font-weight: 700;">sample {item_index}</div>
              <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start;">{grids}</div>
            </div>
            """
        )

    return f"""
    <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
      {"".join(panels)}
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
        <span>negative / zero / positive, scaled per sample triplet</span>
      </div>
    </div>
    """


def render_image_to_energy_gallery_html(
        items: Sequence[Any],
        *,
        steps: Sequence[int] = (64, 32, 16, 8, 4),
        size: int = 32,
) -> str:
        if not items:
                return "<p>No image-to-energy gallery samples are available.</p>"

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
                        raise ValueError(f"generated energy tensors must contain {expected_count} values")
                max_abs = max(
                        float(tensor.abs().max().clamp_min(1.0e-12))
                        for tensor in generated.values()
                )
                image_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">image</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_grayscale_pixels(image, size=size)}
                    </div>
                </div>
                """
                energy_panels = "".join(
                        f"""
                        <div style="display: grid; gap: 6px;">
                            <div style="font-weight: 600;">{step_count} steps</div>
                            <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                                {_signed_pixels(generated[step_count], size=size, max_abs=max_abs)}
                            </div>
                        </div>
                        """
                        for step_count in steps
                )
                panels.append(
                        f"""
                        <div style="display: grid; gap: 10px;">
                            <div style="font-weight: 700;">sample {item_index}</div>
                            <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start;">{image_panel}{energy_panels}</div>
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
    ) -> str:
        if not items:
            return "<p>No energy-to-image gallery samples are available.</p>"

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
            source_panel = f"""
            <div style="display: grid; gap: 6px;">
                <div style="font-weight: 600;">source</div>
                <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                {_signed_pixels(source, size=size, max_abs=max_abs)}
                </div>
            </div>
            """
            image_panels = "".join(
                f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">{step_count} steps</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                    {_grayscale_pixels(generated[step_count], size=size)}
                    </div>
                </div>
                """
                for step_count in steps
            )
            panels.append(
                f"""
                <div style="display: grid; gap: 10px;">
                    <div style="font-weight: 700;">sample {item_index}</div>
                    <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start;">{source_panel}{image_panels}</div>
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


def render_energy_to_image_reflow_gallery_html(
        items: Sequence[Any],
        *,
        size: int = 32,
) -> str:
        if not items:
                return "<p>No energy-to-image reflow gallery samples are available.</p>"

        expected_count = size * size
        panels = []
        for item_index, item in enumerate(items, start=1):
                source = item.source.detach().cpu().reshape(-1)
                target = item.target.detach().cpu().reshape(-1)
                teacher = item.teacher.detach().cpu().reshape(-1)
                student = item.student.detach().cpu().reshape(-1)
                error = item.error.detach().cpu().reshape(-1)
                tensors = (source, target, teacher, student, error)
                if any(tensor.numel() != expected_count for tensor in tensors):
                        raise ValueError(f"gallery tensors must contain {expected_count} values")
                source_max_abs = float(source.abs().max().clamp_min(1.0e-12))
                error_max_abs = float(error.abs().max().clamp_min(1.0e-12))
                source_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">source energy</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_signed_pixels(source, size=size, max_abs=source_max_abs)}
                    </div>
                </div>
                """
                target_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">image sample</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_grayscale_pixels(target, size=size)}
                    </div>
                </div>
                """
                teacher_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">teacher</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_grayscale_pixels(teacher, size=size)}
                    </div>
                </div>
                """
                student_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">student</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_grayscale_pixels(student, size=size)}
                    </div>
                </div>
                """
                error_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">student - teacher</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_signed_pixels(error, size=size, max_abs=error_max_abs)}
                    </div>
                </div>
                """
                panels.append(
                        f"""
                        <div style="display: grid; gap: 10px;">
                            <div style="font-weight: 700;">sample {item_index}</div>
                            <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start;">{source_panel}{target_panel}{teacher_panel}{student_panel}{error_panel}</div>
                        </div>
                        """
                )

        return f"""
        <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
            {"".join(panels)}
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
                <span>source and error: negative / zero / positive, scaled per panel</span>
            </div>
        </div>
        """


def render_image_autoencoder_gallery_html(
        items: Sequence[Any],
        *,
        size: int = 32,
) -> str:
        if not items:
                return "<p>No image autoencoder gallery samples are available.</p>"

        expected_count = size * size
        panels = []
        for item_index, item in enumerate(items, start=1):
                image = item.image.detach().cpu().reshape(-1)
                reconstructed = item.reconstructed_image.detach().cpu().reshape(-1)
                image_error = item.image_error.detach().cpu().reshape(-1)
                encoded_energy = item.encoded_energy.detach().cpu().reshape(-1)
                decoded_energy = item.decoded_energy.detach().cpu().reshape(-1)
                energy_error = item.energy_error.detach().cpu().reshape(-1)
                tensors = (
                        image,
                        reconstructed,
                        image_error,
                        encoded_energy,
                        decoded_energy,
                        energy_error,
                )
                if any(tensor.numel() != expected_count for tensor in tensors):
                        raise ValueError(f"gallery tensors must contain {expected_count} values")
                image_error_max_abs = float(image_error.abs().max().clamp_min(1.0e-12))
                energy_max_abs = max(
                        float(encoded_energy.abs().max().clamp_min(1.0e-12)),
                        float(decoded_energy.abs().max().clamp_min(1.0e-12)),
                )
                energy_error_max_abs = float(energy_error.abs().max().clamp_min(1.0e-12))
                image_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">image</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_grayscale_pixels(image, size=size)}
                    </div>
                </div>
                """
                reconstructed_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">reconstruction</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_grayscale_pixels(reconstructed, size=size)}
                    </div>
                </div>
                """
                image_error_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">image error</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_signed_pixels(image_error, size=size, max_abs=image_error_max_abs)}
                    </div>
                </div>
                """
                encoded_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">encoded energy</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_signed_pixels(encoded_energy, size=size, max_abs=energy_max_abs)}
                    </div>
                </div>
                """
                decoded_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">decoded energy</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_signed_pixels(decoded_energy, size=size, max_abs=energy_max_abs)}
                    </div>
                </div>
                """
                energy_error_panel = f"""
                <div style="display: grid; gap: 6px;">
                    <div style="font-weight: 600;">energy error</div>
                    <div style="display: grid; grid-template-columns: repeat({size}, 6px); grid-template-rows: repeat({size}, 6px); width: {size * 6}px; height: {size * 6}px; border: 1px solid #30333a; background: #000;">
                        {_signed_pixels(energy_error, size=size, max_abs=energy_error_max_abs)}
                    </div>
                </div>
                """
                panels.append(
                        f"""
                        <div style="display: grid; gap: 10px;">
                            <div style="font-weight: 700;">sample {item_index}</div>
                            <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start;">{image_panel}{reconstructed_panel}{image_error_panel}{encoded_panel}{decoded_panel}{energy_error_panel}</div>
                        </div>
                        """
                )

        return f"""
        <div style="display: grid; gap: 18px; font: 13px/1.4 system-ui, sans-serif; color: #d7dae0;">
            {"".join(panels)}
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span style="width: 44px; height: 12px; background: linear-gradient(90deg, #004cff, #000, #ff2600); border: 1px solid #30333a;"></span>
                <span>energy and error: negative / zero / positive, scaled per sample</span>
            </div>
        </div>
        """
