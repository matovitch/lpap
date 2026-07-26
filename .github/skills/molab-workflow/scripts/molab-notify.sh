#!/usr/bin/env bash
# Send a Pushover ping via the paired molab kernel (uses injected secrets).
#
# Use after long agent-side waits (encode, train polls, uploads) so the human
# gets a phone ping even when they must approve/follow the agent turn-by-turn.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION
#
# Usage:
#   bash …/molab-notify.sh --title "lpap: bank encode" --message "BANK_OK …"
#   bash …/molab-notify.sh --title "done" --message "ok" --priority 0
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
title="lpap molab"
message=""
priority=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)
      title="$2"
      shift 2
      ;;
    --message)
      message="$2"
      shift 2
      ;;
    --priority)
      priority="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-notify: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$message" ]]; then
  echo "molab-notify: --message is required" >&2
  exit 1
fi

title_b64="$(printf '%s' "$title" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"
message_b64="$(printf '%s' "$message" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"

"$molab_exec" <<PY
import base64
from lpap.notify import send_pushover

title = base64.b64decode("${title_b64}").decode()
message = base64.b64decode("${message_b64}").decode()
result = send_pushover(message, title=title, priority=int("${priority}"))
print("pushover_ok", result.get("request"), flush=True)
PY
