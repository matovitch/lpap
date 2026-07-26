"""Local helpers for molab-workflow scripts (no molab connection required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def repo_root_from_script(script_file: str | Path) -> Path:
    """``…/lpap-molab/.github/skills/molab-workflow/scripts/<this>`` → repo root."""
    return Path(script_file).resolve().parents[4]


def resolve_lpap_package_file(
    path_arg: str | Path,
    *,
    repo_root: Path,
) -> tuple[Path, str, Path]:
    """Map a local path to ``(absolute_file, module_name, path_relative_to_lpap)``.

    Accepts ``src/lpap/….py``, ``lpap/….py``, a bare ``training_plots.py`` under
    ``src/lpap/``, or an absolute path inside ``src/lpap/``.
    """
    raw = Path(path_arg)
    src_lpap = (repo_root / "src" / "lpap").resolve()
    if not src_lpap.is_dir():
        raise FileNotFoundError(f"missing package tree: {src_lpap}")

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                (repo_root / raw).resolve(),
                (repo_root / "src" / raw).resolve(),
                (src_lpap / raw).resolve(),
            ]
        )
        if raw.suffix == ".py" and "/" not in str(raw).replace("\\", "/"):
            matches = sorted(src_lpap.rglob(raw.name))
            candidates.extend(matches)

    seen: set[Path] = set()
    resolved: Path | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and candidate.suffix == ".py":
            try:
                candidate.relative_to(src_lpap)
            except ValueError:
                continue
            resolved = candidate
            break

    if resolved is None:
        raise FileNotFoundError(
            f"could not resolve {path_arg!r} under {src_lpap} "
            "(pass src/lpap/….py)"
        )

    rel = resolved.relative_to(src_lpap)
    if rel.name == "__init__.py":
        parts = list(rel.parts[:-1])
    else:
        parts = list(rel.with_suffix("").parts)
    if not parts:
        module_name = "lpap"
    else:
        module_name = "lpap." + ".".join(parts)
    return resolved, module_name, rel


def generate_molab_lab_source(
    cells: Sequence[Mapping[str, Any]],
    *,
    width: str = "medium",
) -> str:
    """Build a marimo ``.py`` notebook from live cell dicts.

    Each cell mapping needs ``code`` and ``name``. Optional ``hide_code`` /
    ``disabled`` / ``column`` mirror marimo ``CellConfig``.
    """
    from marimo._ast.app import _AppConfig
    from marimo._ast.cell import CellConfig
    from marimo._ast.codegen import generate_filecontents

    if not cells:
        raise ValueError("no cells to export")

    codes: list[str] = []
    names: list[str] = []
    configs: list[CellConfig] = []
    unnamed = 0
    for cell in cells:
        code = str(cell.get("code") or "")
        name = str(cell.get("name") or "_")
        if name in {"", "_"}:
            unnamed += 1
            name = "_"
        codes.append(code)
        names.append(name)
        configs.append(
            CellConfig(
                hide_code=bool(cell.get("hide_code", False)),
                disabled=bool(cell.get("disabled", False)),
                column=cell.get("column"),
            )
        )

    source = generate_filecontents(
        codes,
        names,
        configs,
        config=_AppConfig(width=width),
    )
    if unnamed:
        # Keep going: marimo allows multiple ``def _``. Callers may warn.
        pass
    return source


def count_unnamed_cells(cells: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for cell in cells if str(cell.get("name") or "_") in {"", "_"})
