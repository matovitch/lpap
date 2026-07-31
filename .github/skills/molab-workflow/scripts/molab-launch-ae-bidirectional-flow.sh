#!/usr/bin/env bash
# Launch multi-pair AE from one bidirectional harmonics flow on molab.
#
# Uses image_energy_flow.pt + c128_k16 / c256_k24 teachers.
# Requires a prior molab-sync (so lpap + secrets are present). Refuses if the
# previous bg pid is still alive.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION (prefer unset unless multi-session)
#
# Usage:
#   bash .github/skills/molab-workflow/scripts/molab-launch-ae-bidirectional-flow.sh --target-steps 20000
#   bash .github/skills/molab-workflow/scripts/molab-launch-ae-bidirectional-flow.sh --target-steps 20000 --resume
#   bash .github/skills/molab-workflow/scripts/molab-launch-ae-bidirectional-flow.sh --target-steps 20000 --json
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
target_steps=""
as_json=0
comment=""
upload=1
notify=1
resume=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-steps)
      target_steps="$2"
      shift 2
      ;;
    --comment)
      comment="$2"
      shift 2
      ;;
    --no-upload)
      upload=0
      shift
      ;;
    --no-notify)
      notify=0
      shift
      ;;
    --resume)
      resume=1
      shift
      ;;
    --json)
      as_json=1
      shift
      ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-launch-ae-bidirectional-flow: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$target_steps" ]]; then
  echo "molab-launch-ae-bidirectional-flow: --target-steps is required" >&2
  exit 1
fi
if [[ ! "$target_steps" =~ ^[0-9]+$ ]]; then
  echo "molab-launch-ae-bidirectional-flow: --target-steps must be an integer" >&2
  exit 1
fi

comment_b64=""
if [[ -n "$comment" ]]; then
  comment_b64="$(printf '%s' "$comment" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"
fi

"$molab_exec" <<PY
import base64
import json
import sys

sys.path.insert(0, "/marimo")
from molab.jobs import launch_ae_bidirectional_flow_bg

comment = None
comment_b64 = "${comment_b64}"
if comment_b64:
    comment = base64.b64decode(comment_b64).decode("utf-8")
result = launch_ae_bidirectional_flow_bg(
    project_root="/marimo",
    target_steps=${target_steps},
    upload_artifacts_on_checkpoint=${upload} == 1,
    notify_on_finished=${notify} == 1,
    comment=comment,
    resume_from_checkpoint=${resume} == 1,
)
if ${as_json} == 1:
    print(json.dumps(result, indent=2, sort_keys=True))
else:
    print(
        f"spawned pid={result['pid']} target={result['target_steps']} "
        f"log={result['log_path']}"
    )
PY
