#!/usr/bin/env bash
# Gate on kernel globals: fail if requested public names are missing from ctx.globals.
# Use after edit_cell (document can be ahead of the kernel).
# This is execute-code: do not run it while a cell or scratchpad is in flight.
#
# Required: MOLAB_URL (or --url) plus MOLAB_TOKEN / MARIMO_TOKEN (or --token).
# Optional: MOLAB_SESSION / --session; MARIMO_PAIR_SCRIPTS.
#
# Usage:
#   notebook-ready.sh [--url URL] [--token TOKEN] [--session ID] NAME [NAME ...]
#
# Exit 0 if all names are present; exit 1 if any are missing or the probe fails.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=molab_pair_lib.sh
source "$script_dir/molab_pair_lib.sh"
MOLAB_PAIR_CALLER="notebook-ready"
molab_require_execute_code || exit 1

url="${MOLAB_URL:-}"
token="${MOLAB_TOKEN:-${MARIMO_TOKEN:-}}"
session="${MOLAB_SESSION:-}"
names=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --token) token="$2"; shift 2 ;;
    --session) session="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    --)
      shift
      names+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      names+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$url" || ${#names[@]} -eq 0 ]]; then
  echo "Usage: notebook-ready.sh [--url URL] [--token TOKEN] [--session ID] NAME [NAME ...]" >&2
  echo "Set MOLAB_URL (and MOLAB_TOKEN) or pass --url / --token." >&2
  exit 1
fi
if [[ -z "$token" ]]; then
  echo "notebook-ready: set MOLAB_TOKEN / MARIMO_TOKEN or pass --token" >&2
  exit 1
fi

args=(--url "$url" --token "$token")
if [[ -n "$session" ]]; then
  args+=(--session "$session")
fi

# JSON array of requested names (safe to embed as a Python literal).
names_json=$(printf '%s\n' "${names[@]}" | python3 -c 'import json,sys; print(json.dumps([line.rstrip("\n") for line in sys.stdin if line.strip()]))')

output=$(
  {
    printf '_names = %s\n' "$names_json"
    cat <<'PY'
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    missing = []
    for name in _names:
        present = name in ctx.globals
        status = "ok" if present else "missing"
        print(f"{name}\t{status}")
        if not present:
            missing.append(name)
    print("READY_OK" if not missing else "READY_FAIL")
PY
  } | "$MOLAB_EXECUTE_CODE" "${args[@]}"
)

printf '%s\n' "$output"

if printf '%s\n' "$output" | grep -qx 'READY_OK'; then
  exit 0
fi
exit 1
