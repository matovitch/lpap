#!/usr/bin/env bash
# Launch a surrogate or decoder bg worker from a shared teacher_*.toml on molab.
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION
#
# Usage:
#   bash …/molab-launch-lpap-teacher.sh --backend surrogate --config configs/training/teacher_c512_k32.toml --target-steps 10000
#   bash …/molab-launch-lpap-teacher.sh --backend decoder --config configs/training/teacher_c512_k32.toml --target-steps 10000
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
backend=""
config=""
target_steps=""
comment=""
upload=1
notify=1
resume=0
as_json=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --target-steps)
      target_steps="$2"
      shift 2
      ;;
    --comment)
      comment="$2"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    --no-upload)
      upload=0
      shift
      ;;
    --no-notify)
      notify=0
      shift
      ;;
    --json)
      as_json=1
      shift
      ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-launch-lpap-teacher: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$backend" || -z "$config" || -z "$target_steps" ]]; then
  echo "molab-launch-lpap-teacher: --backend, --config, and --target-steps are required" >&2
  exit 1
fi
if [[ ! "$target_steps" =~ ^[0-9]+$ ]]; then
  echo "molab-launch-lpap-teacher: --target-steps must be an integer" >&2
  exit 1
fi

comment_b64=""
if [[ -n "$comment" ]]; then
  comment_b64="$(printf '%s' "$comment" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"
fi
config_b64="$(printf '%s' "$config" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"

"$molab_exec" <<PY
import base64
import json
import sys

sys.path.insert(0, "/marimo")
from molab.jobs import launch_lpap_teacher_bg

backend = $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$backend")
config = base64.b64decode("${config_b64}").decode()
comment = None
comment_b64 = "${comment_b64}"
if comment_b64:
    comment = base64.b64decode(comment_b64).decode()
result = launch_lpap_teacher_bg(
    backend_kind=backend,
    config_relpath=config,
    project_root="/marimo",
    target_steps=int("${target_steps}"),
    upload_artifacts_on_checkpoint=bool(${upload}),
    notify_on_finished=bool(${notify}),
    comment=comment,
    resume_from_checkpoint=bool(${resume}),
)
if ${as_json} == 1:
    print(json.dumps(result, indent=2, sort_keys=True))
else:
    print(
        f"spawned pid={result['pid']} backend={backend} target={result['target_steps']} "
        f"log={result['log_path']}"
    )
PY
