#!/usr/bin/env bash
# Export the live paired molab notebook into molab/lab.py (backport).
#
# Pulls named cells via marimo._code_mode and rewrites the durable lab file with
# marimo's codegen (correct signatures / returns). Prefer named cells
# (see molab-workflow skill).
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION
#
# Usage:
#   bash …/molab-export-notebook.sh
#   bash …/molab-export-notebook.sh --output /tmp/lab.py
#   bash …/molab-export-notebook.sh --dry-run
#   bash …/molab-export-notebook.sh --stdout
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
lib_py="$script_dir/molab_lib.py"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
output_path="$repo_root/molab/lab.py"
dry_run=0
to_stdout=0

# Codegen needs marimo (Pixi env). Override with MOLAB_PYTHON if needed.
if [[ -n "${MOLAB_PYTHON:-}" ]]; then
  python_cmd=("$MOLAB_PYTHON")
elif [[ -x "$(command -v pixi)" && -f "$repo_root/pixi.toml" ]]; then
  python_cmd=(pixi run --manifest-path "$repo_root/pixi.toml" python)
else
  python_cmd=(python3)
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output_path="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --stdout)
      to_stdout=1
      shift
      ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-export-notebook: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$molab_exec" ]]; then
  echo "molab-export-notebook: missing $molab_exec" >&2
  exit 1
fi

echo "molab-export-notebook: fetching live cells …" >&2
cells_json="$("$molab_exec" <<'PY'
import json
import marimo._code_mode as cm

async with cm.get_context(skip_validation=True, skip_staleness_check=True) as ctx:
    rows = []
    for cell_id in ctx.cells.keys():
        cell = ctx.cells[cell_id]
        cfg = cell.config
        rows.append(
            {
                "id": cell_id,
                "name": cell.name,
                "code": cell.code or "",
                "hide_code": bool(getattr(cfg, "hide_code", False)),
                "disabled": bool(getattr(cfg, "disabled", False)),
                "column": getattr(cfg, "column", None),
            }
        )
    print(json.dumps(rows))
PY
)"

# Drop marimo-pair "Warning: connecting…" lines if present.
cells_json="$(
  "${python_cmd[@]}" -c '
import json, sys
raw = sys.stdin.read()
start = raw.find("[")
end = raw.rfind("]")
if start < 0 or end < start:
    raise SystemExit("molab-export-notebook: no JSON array in molab-exec output")
json.loads(raw[start : end + 1])
print(raw[start : end + 1])
' <<<"$cells_json"
)"

export MOLAB_EXPORT_CELLS_JSON="$cells_json"
export MOLAB_EXPORT_OUTPUT="$output_path"
export MOLAB_EXPORT_DRY_RUN="$dry_run"
export MOLAB_EXPORT_STDOUT="$to_stdout"

PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "${python_cmd[@]}" - <<'PY'
import json
import os
import sys
from pathlib import Path

from molab_lib import count_unnamed_cells, generate_molab_lab_source

cells = json.loads(os.environ["MOLAB_EXPORT_CELLS_JSON"])
unnamed = count_unnamed_cells(cells)
if unnamed:
    print(
        f"molab-export-notebook: warning: {unnamed} unnamed cell(s) "
        f"(name='_'); prefer naming durable lab cells",
        file=sys.stderr,
    )
for cell in cells:
    name = cell.get("name") or "_"
    lines = (cell.get("code") or "").count("\n") + 1
    print(f"  - {name}: {lines} lines", file=sys.stderr)

source = generate_molab_lab_source(cells, width="medium")
dry = os.environ.get("MOLAB_EXPORT_DRY_RUN") == "1"
to_stdout = os.environ.get("MOLAB_EXPORT_STDOUT") == "1"
output = Path(os.environ["MOLAB_EXPORT_OUTPUT"])

if to_stdout or dry:
    sys.stdout.write(source)
    if not source.endswith("\n"):
        sys.stdout.write("\n")
    if dry and not to_stdout:
        print(
            f"molab-export-notebook: dry-run ({len(source)} bytes); "
            f"would write {output}",
            file=sys.stderr,
        )
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(f"molab-export-notebook: wrote {output} ({len(source)} bytes)", file=sys.stderr)
PY
