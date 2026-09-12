#!/usr/bin/env bash
# Compact DAG / cell map for a live marimo session (token-efficient for agents).
# This is execute-code: do not run it while a cell or scratchpad is in flight.
#
# Required: MOLAB_URL (or --url) plus MOLAB_TOKEN / MARIMO_TOKEN (or --token).
# Optional: MOLAB_SESSION / --session; MARIMO_PAIR_SCRIPTS.
#
# Usage:
#   notebook-map.sh [--url URL] [--token TOKEN] [--session ID] [map|owners|issues|json]
#   notebook-map.sh [--url URL] [--token TOKEN] [--session ID] cell CELL_ID_OR_NAME_OR_INDEX
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=molab_pair_lib.sh
source "$script_dir/molab_pair_lib.sh"
MOLAB_PAIR_CALLER="notebook-map"
molab_require_execute_code || exit 1
map_lib="$script_dir/notebook_map.py"

url="${MOLAB_URL:-}"
token="${MOLAB_TOKEN:-${MARIMO_TOKEN:-}}"
session="${MOLAB_SESSION:-}"
mode="map"
cell=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --token) token="$2"; shift 2 ;;
    --session) session="$2"; shift 2 ;;
    map|owners|issues|json) mode="$1"; shift ;;
    cell)
      mode="cell"
      cell="${2:-}"
      if [[ -z "$cell" ]]; then
        echo "cell mode requires an id/name/index" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      sed -n '2,11p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$url" ]]; then
  echo "Usage: notebook-map.sh [--url URL] [--token TOKEN] [--session ID] [map|owners|issues|json|cell ID]" >&2
  echo "Set MOLAB_URL (and MOLAB_TOKEN) or pass --url / --token." >&2
  exit 1
fi
if [[ -z "$token" ]]; then
  echo "notebook-map: set MOLAB_TOKEN / MARIMO_TOKEN or pass --token" >&2
  exit 1
fi

args=(--url "$url" --token "$token")
if [[ -n "$session" ]]; then
  args+=(--session "$session")
fi

# Ship library + runner into the scratchpad.
{
  cat "$map_lib"
  printf '\nimport marimo._code_mode as cm\n'
  printf 'async with cm.get_context() as ctx:\n'
  # Quote as Python string literals (avoid bare `map` shadowing builtins).
  py_mode=$(printf '%s' "$mode" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
  if [[ "$mode" == "cell" ]]; then
    py_cell=$(printf '%s' "$cell" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    printf '    print(format_map(ctx, mode=%s, cell=%s))\n' "$py_mode" "$py_cell"
  else
    printf '    print(format_map(ctx, mode=%s))\n' "$py_mode"
  fi
} | "$MOLAB_EXECUTE_CODE" "${args[@]}"
