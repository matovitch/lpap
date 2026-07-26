#!/usr/bin/env bash
# Launch an unpaired energy-bank flow worker on molab (i2e or e2i).
#
# Required env: MOLAB_URL, MOLAB_TOKEN or MARIMO_TOKEN
# Optional: MOLAB_SESSION
#
# Usage:
#   bash …/molab-launch-flow-energy-bank.sh --kind image_to_energy_energy_bank --target-steps 10000
#   bash …/molab-launch-flow-energy-bank.sh --kind energy_to_image_energy_bank --target-steps 10000
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
molab_exec="$script_dir/molab-exec.sh"
kind=""
target_steps=""
comment=""
upload=1
notify=1
energy_bank_path="data/encoded_energies_ae_best.pt"
as_json=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)
      kind="$2"
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
    --energy-bank-path)
      energy_bank_path="$2"
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
    --json)
      as_json=1
      shift
      ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "molab-launch-flow-energy-bank: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$kind" || -z "$target_steps" ]]; then
  echo "molab-launch-flow-energy-bank: --kind and --target-steps are required" >&2
  exit 1
fi
if [[ ! "$target_steps" =~ ^[0-9]+$ ]]; then
  echo "molab-launch-flow-energy-bank: --target-steps must be an integer" >&2
  exit 1
fi

comment_b64=""
if [[ -n "$comment" ]]; then
  comment_b64="$(printf '%s' "$comment" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"
fi
bank_b64="$(printf '%s' "$energy_bank_path" | python3 -c 'import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode())')"

"$molab_exec" <<PY
import base64
import json
import sys

sys.path.insert(0, "/marimo")
from molab.jobs import launch_flow_energy_bank_bg

kind = $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$kind")
comment = None
comment_b64 = "${comment_b64}"
if comment_b64:
    comment = base64.b64decode(comment_b64).decode()
energy_bank_path = base64.b64decode("${bank_b64}").decode()
result = launch_flow_energy_bank_bg(
    kind,
    "/marimo",
    target_steps=int("${target_steps}"),
    upload_artifacts_on_checkpoint=bool(${upload}),
    notify_on_finished=bool(${notify}),
    comment=comment,
    energy_bank_path=energy_bank_path,
)
if ${as_json} == 1:
    print(json.dumps(result, indent=2, sort_keys=True))
else:
    print(
        f"spawned pid={result['pid']} kind={kind} target={result['target_steps']} "
        f"log={result['log_path']}"
    )
PY
