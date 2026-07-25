#!/usr/bin/env bash
# Run marimo-pair execute-code against a molab session using env targeting.
#
# Required env:
#   MOLAB_URL                 notebook / server URL
#   MOLAB_TOKEN or MARIMO_TOKEN
# Optional:
#   MOLAB_SESSION             session id — only when the server has several.
#                             Prefer unset: auto-select the single active session.
#                             If set, molab-exec validates it against /api/sessions
#                             and fails loudly on a stale id.
#   MARIMO_PAIR_SCRIPTS       override path to marimo-pair scripts dir
#                             (default: ~/.cursor/skills/marimo-pair/scripts)
#
# Usage (same payload forms as execute-code.sh):
#   molab-exec.sh -c 'print(1)'
#   molab-exec.sh <<'PY'
#   print(1)
#   PY
#   molab-exec.sh probe.py
#
# Extra passthrough: --session ID overrides MOLAB_SESSION for one call.
set -euo pipefail

scripts_dir="${MARIMO_PAIR_SCRIPTS:-$HOME/.cursor/skills/marimo-pair/scripts}"
execute_code="$scripts_dir/execute-code.sh"

if [[ ! -x "$execute_code" ]]; then
  echo "molab-exec: execute-code.sh not found or not executable: $execute_code" >&2
  echo "Set MARIMO_PAIR_SCRIPTS to the marimo-pair scripts directory." >&2
  exit 1
fi

url="${MOLAB_URL:-}"
token="${MOLAB_TOKEN:-${MARIMO_TOKEN:-}}"
session="${MOLAB_SESSION:-}"

passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      session="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,26p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      passthrough+=("$@")
      break
      ;;
  esac
done

if [[ -z "$url" ]]; then
  echo "molab-exec: set MOLAB_URL to the paired notebook URL" >&2
  exit 1
fi
if [[ -z "$token" ]]; then
  echo "molab-exec: set MOLAB_TOKEN or MARIMO_TOKEN" >&2
  exit 1
fi

# When an explicit session id is set, verify it is still live. Stale ids used to
# produce empty/wrong results instead of a clear error.
if [[ -n "$session" ]]; then
  if ! command -v jq >/dev/null 2>&1; then
    echo "molab-exec: jq is required to validate MOLAB_SESSION" >&2
    exit 1
  fi
  base="${url%/}"
  sessions_resp="$(
    curl -sf -H "Authorization: Bearer ${token}" "${base}/api/sessions"
  )" || {
    echo "molab-exec: failed to list sessions at ${base}/api/sessions" >&2
    exit 1
  }
  if ! echo "$sessions_resp" | jq -e --arg sid "$session" 'has($sid)' >/dev/null; then
    echo "molab-exec: session '${session}' is not active (stale MOLAB_SESSION / --session?)." >&2
    echo "Active sessions:" >&2
    if [[ -z "$(echo "$sessions_resp" | jq -r 'keys[]' 2>/dev/null)" ]]; then
      echo "  (none — open the notebook in the browser)" >&2
    else
      echo "$sessions_resp" | jq -r 'to_entries[] | "  \(.key)  \(.value.filename // "")"' >&2
    fi
    echo "Unset MOLAB_SESSION to auto-select when there is exactly one session." >&2
    exit 1
  fi
fi

args=(--url "$url" --token "$token")
if [[ -n "$session" ]]; then
  args+=(--session "$session")
fi

if [[ ${#passthrough[@]} -eq 0 ]]; then
  if [[ -t 0 ]]; then
    echo "Usage: molab-exec.sh -c 'code' | molab-exec.sh script.py | molab-exec.sh <<'PY' ..." >&2
    exit 1
  fi
  exec "$execute_code" "${args[@]}"
fi

exec "$execute_code" "${args[@]}" "${passthrough[@]}"
