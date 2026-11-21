#!/usr/bin/env bash
# Press a notebook UI element as if the human clicked it (fire-and-forget).
#
# Queues an UpdateUIElementCommand on the kernel and returns at once; the cells
# that depend on the element then run as a normal notebook run, independent of
# this client. Use this for run_buttons that start long work (Prepare session,
# Start training). ctx.set_ui_value instead runs the dependents INSIDE the
# scratch request, and they are interrupted when that request ends.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
#
# Usage:
#   molab-press.sh <element_name> [json_value]   # value defaults to true
#   molab-press.sh train_button
#   molab-press.sh resume_box false
#   molab-press.sh stage_select '["autoencoder"]'
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,17p' "$0" | sed 's/^# \?//'
  exit $(( $# < 1 ))
fi
name="$1"
value="${2:-true}"
if [[ ! "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "molab-press: invalid element name: $name" >&2
  exit 1
fi
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$value" 2>/dev/null || {
  echo "molab-press: value must be JSON (true, false, 3, \"x\", [...]): $value" >&2
  exit 1
}

payload="$(mktemp)"
trap 'rm -f "$payload"' EXIT
cat >"$payload" <<PY
import json

import marimo._code_mode as cm
from marimo._code_mode._context import UIElementId, UpdateUIElementCommand

async with cm.get_context() as ctx:
    element = ctx.globals.get("$name")
    if element is None or not hasattr(element, "_id"):
        raise LookupError("no UI element named '$name' in the notebook globals")
    ctx.enqueue_command(
        UpdateUIElementCommand(
            object_ids=[UIElementId(element._id)], values=[json.loads('''$value''')]
        )
    )
print("queued $name =", '''$value''')
PY
"$script_dir/molab-exec.sh" "$payload"
