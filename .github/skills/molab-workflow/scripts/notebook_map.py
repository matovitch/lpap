"""Compact live-notebook DAG map for agent scratchpads (token-efficient).

Library helpers for marimo ``cm`` scratchpads. Prefer the wrapper:

    bash .github/skills/molab-workflow/scripts/notebook-map.sh [map|owners|issues|cell|json]
    bash .github/skills/molab-workflow/scripts/notebook-map.sh cell CELL_ID_OR_NAME

Or in a scratchpad after loading this file's helpers:

    async with cm.get_context() as ctx:
        print(format_map(ctx))
"""

from __future__ import annotations

import json
import os
from typing import Any


def _err_summaries(cell: Any) -> list[str]:
    out: list[str] = []
    for err in cell.errors or []:
        kind = getattr(err, "kind", None) or type(err).__name__
        msg = getattr(err, "msg", None) or str(err)
        out.append(f"{kind}:{msg}"[:140])
    return out


def _head(code: str, n: int = 1, width: int = 72) -> str:
    lines = [ln.strip() for ln in (code or "").strip().splitlines() if ln.strip()]
    if not lines:
        return "-"
    return " ; ".join(lines[:n])[:width]


def _as_id_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (set, frozenset, list, tuple)):
        return {str(item) for item in value}
    return {str(value)}


def _adjacency(graph: Any) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Normalize parents/children maps to id -> set[id]."""
    parents: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {}
    for cell_id, vals in dict(getattr(graph, "parents", {}) or {}).items():
        parents[str(cell_id)] = _as_id_set(vals)
    for cell_id, vals in dict(getattr(graph, "children", {}) or {}).items():
        children[str(cell_id)] = _as_id_set(vals)
    return parents, children


def _collect_ghosts(
    *,
    doc_ids: set[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
    owners: dict[str, list[str]],
    conflicts: list[Any],
) -> tuple[list[str], list[dict[str, str]], dict[str, list[str]]]:
    """Find cell ids referenced by the graph but absent from the document.

    Ghosts appear after incomplete deletes: gone from ``ctx.cells`` /
    ``graph.cells``, but still listed in adjacency or definition registries.
    ``delete_cell(ghost_id)`` fails; deletes that walk ``descendants`` can
    raise ``KeyError``. Do not treat this as ordinary multiply-defined debt.
    """
    referenced: set[str] = set()
    referenced.update(parents)
    referenced.update(children)
    for src, dsts in parents.items():
        referenced.add(src)
        referenced.update(dsts)
    for src, dsts in children.items():
        referenced.add(src)
        referenced.update(dsts)
    for cell_ids in owners.values():
        referenced.update(cell_ids)
    for item in conflicts:
        if isinstance(item, dict) and "cells" in item:
            referenced.update(str(cid) for cid in item["cells"])

    ghosts = sorted(referenced - doc_ids)

    dangling: list[dict[str, str]] = []
    for child_id, parent_ids in parents.items():
        for parent_id in sorted(parent_ids):
            if parent_id not in doc_ids or child_id not in doc_ids:
                dangling.append(
                    {"via": "parent", "from": parent_id, "to": child_id}
                )
    for parent_id, child_ids in children.items():
        for child_id in sorted(child_ids):
            # Avoid duplicating the same edge already reported via parents.
            if parent_id in doc_ids and child_id in doc_ids:
                continue
            edge = {"via": "child", "from": parent_id, "to": child_id}
            if edge not in dangling and {
                "via": "parent",
                "from": parent_id,
                "to": child_id,
            } not in dangling:
                dangling.append(edge)

    ghost_set = set(ghosts)
    ghost_owners = {
        name: sorted(cid for cid in cell_ids if cid in ghost_set)
        for name, cell_ids in owners.items()
        if any(cid in ghost_set for cid in cell_ids)
    }
    return ghosts, dangling, ghost_owners


def collect_map(ctx: Any) -> dict[str, Any]:
    """Return a JSON-serializable notebook map from a code-mode context."""
    graph = ctx.graph
    parents, children = _adjacency(graph)
    graph_cells = getattr(graph, "cells", {}) or {}
    doc_ids = {str(cell_id) for cell_id in ctx.cells.keys()}

    cells: list[dict[str, Any]] = []
    for index, cell_id in enumerate(ctx.cells.keys()):
        cell_id_s = str(cell_id)
        cell = ctx.cells[cell_id]
        impl = graph_cells.get(cell_id)
        if impl is None:
            impl = graph_cells.get(cell_id_s)
        defs = sorted(impl.defs) if impl is not None else []
        refs = sorted(impl.refs) if impl is not None else []
        code = cell.code or ""
        cells.append(
            {
                "i": index,
                "id": cell_id_s,
                "name": cell.name,
                "status": str(cell.status),
                "lines": code.count("\n") + (1 if code else 0),
                "in_graph": cell_id in graph_cells or cell_id_s in graph_cells,
                "defs": defs,
                "refs": refs,
                "parents": sorted(parents.get(cell_id_s, ())),
                "children": sorted(children.get(cell_id_s, ())),
                "errs": _err_summaries(cell),
                "head": _head(code),
            }
        )

    owners: dict[str, list[str]] = {}
    definitions = getattr(graph, "definitions", {}) or {}
    for name, cell_ids in definitions.items():
        owners[str(name)] = sorted(_as_id_set(cell_ids))

    multiply: list[Any] = []
    if hasattr(graph, "get_multiply_defined"):
        raw_multiply = graph.get_multiply_defined() or []
        # API may return names or (name, owners) tuples.
        for item in raw_multiply:
            if isinstance(item, (tuple, list)) and item:
                multiply.append(str(item[0]))
            else:
                multiply.append(item)
        multiply = sorted(set(multiply), key=str)

    conflicts: list[Any] = []
    if hasattr(graph, "get_multiply_defined_conflicts"):
        raw = graph.get_multiply_defined_conflicts() or {}
        if isinstance(raw, dict):
            conflicts = [
                {"name": str(name), "cells": sorted(_as_id_set(cell_ids))}
                for name, cell_ids in sorted(raw.items(), key=lambda kv: str(kv[0]))
            ]
        else:
            conflicts = [{"raw": str(raw)}]

    cycles = []
    if getattr(graph, "cycles", None):
        cycles = [sorted(cycle) for cycle in graph.cycles]

    stale = sorted(graph.get_stale()) if hasattr(graph, "get_stale") else []

    orphans = [
        row["id"]
        for row in cells
        if row["in_graph"]
        and not row["parents"]
        and not row["children"]
        and not row["defs"]
    ]
    detached = [row["id"] for row in cells if not row["in_graph"]]
    errored = [row["id"] for row in cells if row["errs"]]

    ghosts, dangling_edges, ghost_owners = _collect_ghosts(
        doc_ids=doc_ids,
        parents=parents,
        children=children,
        owners=owners,
        conflicts=conflicts,
    )

    # Annotate conflicts that include at least one ghost owner.
    ghost_set = set(ghosts)
    for item in conflicts:
        if isinstance(item, dict) and "cells" in item:
            item["ghost_cells"] = sorted(
                cid for cid in item["cells"] if cid in ghost_set
            )

    return {
        "n": len(cells),
        "cells": cells,
        "owners": owners,
        "multiply": multiply,
        "conflicts": conflicts,
        "cycles": cycles,
        "stale": stale,
        "orphans": orphans,
        "detached": detached,
        "errored": errored,
        "ghosts": ghosts,
        "dangling_edges": dangling_edges,
        "ghost_owners": ghost_owners,
        "hint": (
            "ghost ids are absent from the document; delete_cell(id) fails and "
            "descendants() may KeyError — scrub adjacency/definition_conflicts "
            "or restart the session; do not recreate cells to 'win' multiply"
            if ghosts
            else None
        ),
    }


_ISSUE_KEYS = (
    "ghosts",
    "ghost_owners",
    "dangling_edges",
    "multiply",
    "conflicts",
    "cycles",
    "stale",
    "orphans",
    "detached",
    "errored",
)


def _format_owner_cell(cell_id: str, ghosts: set[str]) -> str:
    return f"{cell_id}!ghost" if cell_id in ghosts else cell_id


def format_map(
    ctx: Any, *, mode: str | None = None, cell: str | None = None
) -> str:
    """Render a compact, LLM-friendly notebook map."""
    mode = str(mode or os.environ.get("NOTEBOOK_MAP_MODE") or "map").strip().lower()
    cell = cell or os.environ.get("NOTEBOOK_MAP_CELL")
    if cell is not None:
        cell = str(cell)
    data = collect_map(ctx)
    ghost_set = set(data["ghosts"])

    if mode == "json":
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    if mode == "owners":
        lines = [f"# OWNERS n={len(data['owners'])} ghosts={len(data['ghosts'])}"]
        for name, cell_ids in sorted(data["owners"].items()):
            labeled = [_format_owner_cell(cid, ghost_set) for cid in cell_ids]
            lines.append(f"{name}\t{','.join(labeled)}")
        if data["ghosts"]:
            lines.append(f"# GHOSTS\t{json.dumps(data['ghosts'], separators=(',', ':'))}")
            if data.get("hint"):
                lines.append(f"# HINT\t{data['hint']}")
        return "\n".join(lines)

    if mode == "issues":
        lines = ["# ISSUES"]
        for key in _ISSUE_KEYS:
            value = data[key]
            if value:
                lines.append(
                    f"{key}\t{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
                )
        if data.get("hint") and data["ghosts"]:
            lines.append(f"hint\t{data['hint']}")
        if len(lines) == 1:
            lines.append("ok\tnone")
        return "\n".join(lines)

    if mode == "cell":
        if not cell:
            return "# ERROR NOTEBOOK_MAP_CELL required"
        target = None
        for row in data["cells"]:
            if cell in {row["id"], row["name"], str(row["i"])}:
                target = row
                break
        if target is None:
            if cell in ghost_set:
                owners = [
                    name
                    for name, cids in data["ghost_owners"].items()
                    if cell in cids
                ]
                edges = [
                    edge
                    for edge in data["dangling_edges"]
                    if cell in {edge["from"], edge["to"]}
                ]
                return "\n".join(
                    [
                        f"# GHOST {cell} (not in document)",
                        f"owns\t{','.join(owners) or '-'}",
                        f"dangling\t{json.dumps(edges, separators=(',', ':'))}",
                        f"hint\t{data.get('hint') or '-'}",
                    ]
                )
            return f"# ERROR cell not found: {cell}"
        full = ctx.cells[target["id"]]
        return "\n".join(
            [
                f"# CELL {target['id']} name={target['name']} "
                f"status={target['status']}",
                f"defs\t{','.join(target['defs']) or '-'}",
                f"refs\t{','.join(target['refs']) or '-'}",
                f"parents\t{','.join(target['parents']) or '-'}",
                f"children\t{','.join(target['children']) or '-'}",
                f"errs\t{';'.join(target['errs']) or '-'}",
                "---",
                full.code or "",
            ]
        )

    lines = [
        f"# NBMAP n={data['n']} owners={len(data['owners'])} "
        f"stale={len(data['stale'])} orphans={len(data['orphans'])} "
        f"detached={len(data['detached'])} errored={len(data['errored'])} "
        f"multiply={len(data['multiply'])} ghosts={len(data['ghosts'])} "
        f"cycles={len(data['cycles'])}",
        "# i id name status lines in_graph defs refs parents children errs | head",
    ]
    for row in data["cells"]:
        lines.append(
            "{i}\t{id}\t{name}\t{status}\t{lines}\t{in_graph}\t{defs}\t{refs}\t"
            "{parents}\t{children}\t{errs}\t| {head}".format(
                i=row["i"],
                id=row["id"],
                name=row["name"] or "_",
                status=row["status"],
                lines=row["lines"],
                in_graph=int(row["in_graph"]),
                defs=",".join(row["defs"]) or "-",
                refs=",".join(row["refs"]) or "-",
                parents=",".join(row["parents"]) or "-",
                children=",".join(row["children"]) or "-",
                errs=";".join(row["errs"]) or "-",
                head=row["head"],
            )
        )
    issue_present = any(data[key] for key in _ISSUE_KEYS)
    if issue_present:
        lines.append("# ISSUES")
        for key in _ISSUE_KEYS:
            if data[key]:
                lines.append(
                    f"{key}\t{json.dumps(data[key], ensure_ascii=False, separators=(',', ':'))}"
                )
        if data.get("hint") and data["ghosts"]:
            lines.append(f"hint\t{data['hint']}")
    return "\n".join(lines)
