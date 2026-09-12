#!/usr/bin/env bash
# Poll a live marimo session with short execute-code probes (agent-side loop).
# Do not sleep inside the kernel — that holds the scratchpad and can be interrupted.
#
# Required: MOLAB_URL (or --url) plus MOLAB_TOKEN / MARIMO_TOKEN (or --token).
# Optional: MOLAB_SESSION / --session; MARIMO_PAIR_SCRIPTS.
#
# Usage:
#   execute-watch.sh [--url URL] [--token TOKEN] [--session ID] \
#     [--pattern REGEX] [--interval SECONDS] [--timeout SECONDS] \
#     (-c CODE | SCRIPT.py)
#
# Exit 0 when stdout matches --pattern (default: any successful probe).
# Exit 1 on timeout or repeated probe failure.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=molab_pair_lib.sh
source "$script_dir/molab_pair_lib.sh"
MOLAB_PAIR_CALLER="execute-watch"
molab_require_execute_code || exit 1

url="${MOLAB_URL:-}"
token="${MOLAB_TOKEN:-${MARIMO_TOKEN:-}}"
session="${MOLAB_SESSION:-}"
pattern=""
interval=15
timeout=600
code=""
script=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --token) token="$2"; shift 2 ;;
    --session) session="$2"; shift 2 ;;
    --pattern) pattern="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --timeout) timeout="$2"; shift 2 ;;
    -c) code="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      script="$1"
      shift
      break
      ;;
  esac
done

if [[ -z "$code" && -z "$script" ]]; then
  echo "Usage: execute-watch.sh [--url URL] [opts] (-c CODE | SCRIPT.py)" >&2
  exit 1
fi
if [[ -z "$url" ]]; then
  echo "execute-watch: set MOLAB_URL or pass --url" >&2
  exit 1
fi
if [[ -z "$token" ]]; then
  echo "execute-watch: set MOLAB_TOKEN / MARIMO_TOKEN or pass --token" >&2
  exit 1
fi

args=(--url "$url" --token "$token")
if [[ -n "$session" ]]; then
  args+=(--session "$session")
fi

probe_args=("${args[@]}")
if [[ -n "$code" ]]; then
  probe_args+=(-c "$code")
else
  probe_args+=("$script")
fi

start_ts=$(date +%s)
attempt=0
while true; do
  now=$(date +%s)
  elapsed=$((now - start_ts))
  if (( elapsed >= timeout )); then
    echo "execute-watch: timeout after ${timeout}s (${attempt} probes)" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  set +e
  output=$("$MOLAB_EXECUTE_CODE" "${probe_args[@]}" 2>&1)
  status=$?
  set -e
  printf '[probe %d t=%ss exit=%s]\n%s\n' "$attempt" "$elapsed" "$status" "$output"
  if [[ $status -eq 0 ]]; then
    if [[ -z "$pattern" ]] || printf '%s\n' "$output" | grep -E -q -- "$pattern"; then
      exit 0
    fi
  fi
  # Sleep on the agent host between short probes.
  remaining=$((timeout - elapsed))
  if (( remaining <= 0 )); then
    echo "execute-watch: timeout after ${timeout}s (${attempt} probes)" >&2
    exit 1
  fi
  sleep_for=$interval
  if (( sleep_for > remaining )); then
    sleep_for=$remaining
  fi
  sleep "$sleep_for"
done
